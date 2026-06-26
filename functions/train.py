"""End-to-end joint training of the DPD + FFE through the differentiable channel.

Single stage, direct-learning architecture: DPD and FFE are trained TOGETHER with a
SYMBOL cross-entropy loss (M-way), under realistic noise. The gradient backpropagates
through the whole chain (TX -> channel -> RX). Long sliding windows of fresh random bits
are used each step, and edge_guard symbols are dropped at both ends so filter transients
never enter the loss (no tiny blocks).

The constellation/levels are LEARNED, not imposed: there is NO equispacing/level-target
term (that biased the result toward the textbook optimum). Symbol cross-entropy is the
key: per-bit BCE could 'abandon' one bit (a degenerate 2-level collapse, a comfortable
local minimum), but confusing two symbols costs full CE, so the 4 levels stay distinct
and self-organize. Per-bit posteriors are recovered by marginalizing the softmax.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from channel import OpticalChannel
from transmitter import Transmitter, bits_to_symbols
from receiver import Receiver, bit_posteriors
from utils import add_awgn, measure_ber, theoretical_ber_unipolar, theoretical_ber_unipolar_ase

def random_bits(num_bits, num_symbols, device):
    return torch.randint(0, 2, (num_bits, num_symbols), device=device)


def level_gap_spread(photocurrent, symbols, samples_per_symbol, guard, modulation_order):
    """Diagnostic ONLY (not in the loss): how equispaced the LEARNED levels turned out.
    Returns std/mean of the sorted level-gaps (0 = perfectly equispaced). Label-invariant:
    it just measures the self-organized constellation, it imposes nothing."""
    centers = photocurrent[::samples_per_symbol][:symbols.shape[0]]
    measured = centers[guard:symbols.shape[0] - guard]
    syms = symbols[guard:symbols.shape[0] - guard]
    means = torch.sort(torch.stack([measured[syms == q].mean() for q in range(modulation_order)])).values
    gaps = means[1:] - means[:-1]
    return (gaps.std() / (gaps.mean().abs() + 1e-9)).item()


def link_photocurrent(transmitter, channel, bits, ebn0_db, config, receiver=None):
    """Photocurrent with noise injected per the configured regime: ASE on the optical field
    (inside the channel) or thermal/electrical AWGN on the photocurrent (at the ADC)."""
    drive = transmitter(bits)
    if config.noise_regime == "ase":
        if getattr(config, "rx_filter", None) == "time-rect" and receiver is not None:
            # Optical matched filter on the field, then the ENVELOPE |z| = sqrt(zr^2 + zi^2) as the decision
            # statistic (Forestieri's envelope detector, Fig. A.5). Deciding on the amplitude (not the
            # intensity z^2) is what lets the E2E reach A.47: the ASE noise is ~Gaussian on the field, so the
            # optimal symbol logit is quadratic in the amplitude z; on z^2 it would need sqrt(z^2)=|z|, a
            # nonlinearity the small FFE represents poorly -> a gap that blows up at high SNR.
            xr, xi = channel.complex_envelope(drive, ebn0_db)
            mr = receiver.matched_and_decimate(xr).squeeze(0).squeeze(0)
            mi = receiver.matched_and_decimate(xi).squeeze(0).squeeze(0)
            envelope = torch.sqrt((mr ** 2 + mi ** 2).clamp(min=0))
            return envelope.repeat_interleave(config.samples_per_symbol_sim)
        return channel(drive, ase_ebn0_db=ebn0_db)
    photocurrent = channel(drive)
    return add_awgn(photocurrent, ebn0_db, config.bits_per_symbol, config.samples_per_symbol_sim)


def forward_link(transmitter, channel, receiver, bits, ebn0_db, config):
    return receiver(link_photocurrent(transmitter, channel, bits, ebn0_db, config, receiver))


def gray_code(n):
    """Standard reflected binary Gray code of integer n."""
    return n ^ (n >> 1)


def amplitude_gray_bits(transmitter, channel, receiver, config, device, num_symbols=40000):
    """Map each symbol to its Gray-by-AMPLITUDE bits: (M, num_bits) long tensor.
    Sort the learned level means low->high and label them with the Gray code (00,01,11,10,...), so
    adjacent levels differ by one bit. This is the standard PAM amplitude-Gray labelling that A.19
    assumes and that the threshold detector already uses (gray=[0,1,3,2]); it removes the 4/3 (~1.33x)
    penalty the binary map costs the E2E (whose symbol<->level ordering is arbitrary). It only LABELS
    the learned levels (no constellation prior). Calibrated from the noiseless decision statistic."""
    M = config.modulation_order
    k = config.bits_per_symbol
    guard = config.edge_guard_symbols
    with torch.no_grad():
        bits = random_bits(k, num_symbols, device)
        symbols = bits_to_symbols(bits, k)
        stat = receiver.matched_and_decimate(
            link_photocurrent(transmitter, channel, bits, 40.0, config, receiver)).reshape(-1)  # ~noiseless
        n = min(stat.shape[0], symbols.shape[0])
        stat, syms = stat[:n], symbols[:n]
        best_means = None                                          # align stat<->symbol (max level separation)
        for shift in range(-2, 3):
            rolled = torch.roll(syms, shift)
            means = torch.stack([stat[guard:n - guard][rolled[guard:n - guard] == m].mean() for m in range(M)])
            if not torch.all(torch.isfinite(means)):
                continue
            if best_means is None or (means.max() - means.min()) > (best_means.max() - best_means.min()):
                best_means = means
        order = torch.argsort(best_means)                          # symbols, low -> high amplitude
    mapping = torch.zeros(M, k, dtype=torch.long, device=device)
    for rank in range(M):
        code = gray_code(rank)
        mapping[int(order[rank])] = torch.tensor([(code >> (k - 1 - i)) & 1 for i in range(k)], device=device)
    return mapping


def train(config, device, num_steps, ebn0_range=(7.0, 19.0)):
    """Single-stage joint DPD+FFE training with SYMBOL cross-entropy. The constellation/levels are
    LEARNED, not imposed (no equispacing/level target). Symbol CE keeps the 4 levels distinct (per-bit
    BCE alone could abandon a bit -> a degenerate collapse). The per-bit LLRs are recovered at eval by
    marginalizing the softmax over the amplitude-Gray map (amplitude_gray_bits)."""
    transmitter = Transmitter(config).to(device)
    channel = OpticalChannel(config).to(device)
    receiver = Receiver(config).to(device)
    window = config.minibatch_symbols + 2 * config.edge_guard_symbols
    guard = config.edge_guard_symbols
    sps = config.samples_per_symbol_sim
    equalizer = getattr(config, "equalizer", "end-to-end")
    if equalizer == "threshold":                              # threshold detector: nothing to train
        print("=== equalizer='threshold' -> threshold detector (A.19 reference), no training ===")
        return transmitter, channel, receiver
    params = list(receiver.parameters())                      # "ffe": train the FFE only (TX levels fixed)
    if equalizer == "end-to-end":
        params = list(transmitter.parameters()) + params      # "end-to-end": also train the DPD
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    print(f"=== training (equalizer={equalizer}) — symbol cross-entropy ===")
    for step in range(num_steps):
        ebn0_db = float(np.random.uniform(*ebn0_range))
        bits = random_bits(config.bits_per_symbol, window, device)
        symbols = bits_to_symbols(bits, config.bits_per_symbol)
        noisy = link_photocurrent(transmitter, channel, bits, ebn0_db, config, receiver)  # ASE -> envelope; thermal -> photocurrent
        logits = receiver(noisy)                                  # (num_symbols, M) symbol logits
        n = logits.shape[0]
        loss = F.cross_entropy(logits[guard:n - guard], symbols[guard:n - guard])   # symbol CE: no collapse, no imposed levels
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, num_steps // 8) == 0 or step == num_steps - 1:
            ber, _ = measure_ber(bit_posteriors(logits, config.bits_per_symbol), bits, guard)
            with torch.no_grad():
                spread = level_gap_spread(channel(transmitter(bits)), symbols, sps, guard, config.modulation_order)
            print(f"  step {step:4d}  ebn0 {ebn0_db:4.1f}  ce {loss.item():.8f}  gap-spread {spread:.3f}  BER {ber:.10f}")
    return transmitter, channel, receiver


def evaluate(transmitter, channel, receiver, config, ebn0_db_list, num_symbols, device):
    transmitter.eval()
    receiver.eval()
    k = config.bits_per_symbol
    # amplitude-Gray labelling of the LEARNED levels -> per-bit LLRs reach A.19 (no 4/3 binary penalty)
    sym_bits = amplitude_gray_bits(transmitter, channel, receiver, config, device)
    results = []
    with torch.no_grad():
        bits = random_bits(k, num_symbols, device)
        ref_bits = sym_bits[bits_to_symbols(bits, k)].T                    # (k, N) amplitude-Gray reference
        for ebn0_db in ebn0_db_list:
            logits = forward_link(transmitter, channel, receiver, bits, ebn0_db, config)
            post = bit_posteriors(logits, k, symbol_bits=sym_bits)         # P(b_i = 1) under amplitude-Gray
            ber, shift = measure_ber(post, ref_bits, config.edge_guard_symbols)
            results.append(ber)
            print(f"Eb/N0 {ebn0_db:5.1f} dB   BER {ber:.3e}   (shift {shift})")
    return np.array(results)


def evaluate_threshold(transmitter, channel, receiver, config, ebn0_db_list, num_symbols, device):
    """Optimum threshold detector = the A.19 / A.47 reference (config.equalizer == "threshold"). No DPD/FFE.
      thermal (A.19): decide on the integrated photocurrent (intensity); genie optimum thresholds (midway
                      between the measured level means); Gray.
      ase (A.47)    : COHERENT matched filter on the field + envelope |z| = sqrt(zr^2 + zi^2); Forestieri's
                      thresholds (midway between the NOISELESS signal amplitudes); Gray.
    The TX emits the regime's optimum fixed levels (equispaced intensity for A.19, amplitude for A.47)."""
    transmitter.eval()
    receiver.eval()
    M = config.modulation_order
    k = config.bits_per_symbol
    guard = config.edge_guard_symbols
    gray = torch.tensor([0, 1, 3, 2], device=device)[:M]          # symbol <-> rank (involution)
    results = []
    with torch.no_grad():
        bits = random_bits(k, num_symbols, device)
        symbols = bits_to_symbols(bits, k)
        rank = gray[symbols]

        def decode(stream, thresholds, shift):
            est = gray[torch.bucketize(stream, thresholds).clamp(max=M - 1)]
            est_bits = torch.stack([(est >> (k - 1 - i)) & 1 for i in range(k)])
            ref = torch.roll(bits, shift, dims=1)
            return (est_bits[:, guard:-guard] != ref[:, guard:-guard]).float().mean().item()

        if config.noise_regime == "ase":
            # Direct detection (Forestieri Fig. A.4): the optical matched filter ho(t)=p*(T-t) shapes the
            # field x(t), the square-law photodiode forms the photocurrent z^2 = |x*ho|^2; decide on z^2
            # with thresholds = (midpoint between the amplitude levels)^2 (= the Fig. A.5 thresholds, squared).
            def photocurrent(ebn0_db):
                xr, xi = channel.complex_envelope(transmitter(bits), ebn0_db)      # noisy complex envelope x+n
                mr = receiver.matched_and_decimate(xr).squeeze()                   # optical matched filter, per quadrature
                mi = receiver.matched_and_decimate(xi).squeeze()
                return mr ** 2 + mi ** 2                                           # square-law photodiode -> z^2
            z2 = photocurrent(None)                                                # noiseless signal photocurrent
            sig, best_shift = None, 0
            for shift in range(-2, 3):                                             # align on the level separation
                means = torch.stack([z2[torch.roll(rank, shift) == a].mean() for a in range(M)])
                if torch.all(means[1:] > means[:-1]) and (sig is None or
                                                          means.max() - means.min() > sig.max() - sig.min()):
                    sig, best_shift = means, shift
            amp = torch.sqrt(sig.clamp(min=0))                                     # field amplitudes A_m = sqrt(z^2)
            thresholds = ((amp[:-1] + amp[1:]) / 2) ** 2                           # (amplitude midpoint)^2  (Fig. A.4)
            for ebn0_db in ebn0_db_list:
                ber = decode(photocurrent(ebn0_db), thresholds, best_shift)
                results.append(ber)
                print(f"Eb/N0 {ebn0_db:5.1f} dB   BER {ber:.3e}   (direct detection, A.47)")
        else:
            for ebn0_db in ebn0_db_list:
                stream = receiver.matched_and_decimate(
                    link_photocurrent(transmitter, channel, bits, ebn0_db, config)).squeeze()
                best = 1.0                                        # small group-delay shift search
                for shift in range(-2, 3):
                    means = torch.stack([stream[torch.roll(rank, shift) == a].mean() for a in range(M)])
                    if not torch.all(means[1:] > means[:-1]):
                        continue
                    best = min(best, decode(stream, (means[:-1] + means[1:]) / 2, shift))
                results.append(best)
                print(f"Eb/N0 {ebn0_db:5.1f} dB   BER {best:.3e}   (threshold detector, A.19)")
    return np.array(results)


if __name__ == "__main__":
    config = Config()
    config.minibatch_symbols = 4096
    config.edge_guard_symbols = 48
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(config.summary())
    print(f"\n--- training (device={device}) ---")
    tx, ch, rx = train(config, device, num_steps=2000)

    print("\n--- evaluation vs theory ---")
    ebn0_list = np.arange(8, 24, 2.0)
    measured = evaluate(tx, ch, rx, config, ebn0_list, num_symbols=200_000, device=device)
    if config.noise_regime == "ase":
        theory = theoretical_ber_unipolar_ase(ebn0_list, config.modulation_order)
    else:
        theory = theoretical_ber_unipolar(ebn0_list, config.modulation_order)
    for e, m, t in zip(ebn0_list, measured, theory):
        print(f"Eb/N0 {e:4.0f}  measured {m:.2e}  theory({config.noise_regime}) {t:.2e}")

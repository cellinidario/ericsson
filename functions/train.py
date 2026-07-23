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
from surrogate import build_channel
from transmitter import Transmitter, bits_to_symbols, coded_bpam_target, differential_decode
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


def _measure_ber_diff(bit_probability, raw_bits, edge_guard, max_shift=3):
    """BER for the differential-precoder case: the RX posteriors estimate the sign STATE; decide it,
    differential-decode (raw_phase[k]=state[k] XOR state[k-1]), then compare to the RAW bits. The
    shift search is done on the DECODED stream (group-delay robustness), mirroring measure_ber."""
    import torch
    state = (bit_probability > 0.5).to(torch.int64)              # decided sign state (row1) + amplitude (row0)
    decoded = differential_decode(state).to(torch.int64)
    raw = raw_bits.to(torch.int64)
    n = decoded.shape[1]
    best_ber, best_shift = 1.0, 0
    for shift in range(-max_shift, max_shift + 1):
        a0, a1 = edge_guard + max(0, shift), n - edge_guard + min(0, shift)
        b0, b1 = edge_guard - min(0, shift), n - edge_guard - max(0, shift)
        err = (decoded[:, a0:a1] != raw[:, b0:b1]).float().mean().item()
        if err < best_ber:
            best_ber, best_shift = err, shift
    return best_ber, best_shift


def link_photocurrent(transmitter, channel, bits, ebn0_db, config, receiver=None):
    """Photocurrent with noise injected per the configured regime: ASE on the optical field
    (inside the channel) or thermal/electrical AWGN on the photocurrent (at the ADC)."""
    drive = transmitter(bits)
    if config.noise_regime == "ase":
        if (getattr(config, "rx_filter", None) == "time-rect" and receiver is not None
                and getattr(config, "modulation_format", "upam-4") != "bpam-4"):
            # (unipolar only: the per-symbol envelope statistic below is 1/symbol; bpam-4 needs the
            # 2-sps photocurrent from the plain channel path, where the optical filter on the field
            # creates the T/2 interference the receiver reads.)
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
    channel = build_channel(config).to(device)     # "physics" (default) or a frozen data-trained surrogate
    receiver = Receiver(config).to(device)
    window = config.minibatch_symbols + 2 * config.edge_guard_symbols
    guard = config.edge_guard_symbols
    sps = config.samples_per_symbol_sim
    equalizer = getattr(config, "equalizer", "end-to-end")
    if equalizer == "threshold":                              # threshold detector: nothing to train
        print("=== equalizer='threshold' -> threshold detector (A.19 reference), no training ===")
        return transmitter, channel, receiver
    bpam = getattr(config, "modulation_format", "upam-4") == "bpam-4"

    # WARM-START (bpam-4 E2E): break the TX<->RX co-adaptation deadlock that keeps from-scratch joint
    # training in a quasi-unipolar local min. This is BASIN SELECTION, not answer-imposition: after the two
    # priming phases the network is fine-tuned FREELY and the constellation stays learned (it rediscovers the
    # optimal |A2/A1| and can beat the classical detector). Needs the differential precoder in E2E.
    if bpam and equalizer == "end-to-end" and getattr(config, "bpam_warm_start", False):
        if not getattr(transmitter, "precode_e2e", False):
            raise RuntimeError("bpam_warm_start needs bpam_precode_e2e=True (classical BPAM alphabet)")
        # Phase 0 -- distil the DPD onto the classical BPAM constellation (supervised MSE to fixed_levels).
        tx_opt = torch.optim.Adam(transmitter.parameters(), lr=config.learning_rate)
        for _ in range(getattr(config, "warm_distill_steps", 8000)):
            bits = random_bits(config.bits_per_symbol, window, device)
            coded = transmitter._precode_bpam(bits)
            target = transmitter.fixed_levels[(coded[0] * 2 + coded[1]).long()]     # classical drive/symbol
            drive = transmitter.symbol_drive_levels(bits)[0]
            tx_opt.zero_grad(); F.mse_loss(drive, target).backward(); tx_opt.step()
        # Phase 1 -- RX-only: let the FFE learn to decode the (now classical-BPAM) TX before joint fine-tune.
        rx_opt = torch.optim.Adam(receiver.parameters(), lr=config.learning_rate)
        for _ in range(getattr(config, "warm_rxonly_steps", 15000)):
            ebn0_db = float(np.random.uniform(*ebn0_range))
            bits = random_bits(config.bits_per_symbol, window, device)
            noisy = link_photocurrent(transmitter, channel, bits, ebn0_db, config, receiver)
            logits = receiver(noisy); n = logits.shape[0]
            post = bit_posteriors(logits[guard:n - guard], config.bits_per_symbol)
            rx_opt.zero_grad()
            F.binary_cross_entropy(post.clamp(1e-6, 1 - 1e-6), bits[:, guard:n - guard].float()).backward()
            rx_opt.step()
        print("=== warm-start done (TX distilled to classical BPAM, RX primed) -> free joint fine-tune ===")

    # STAGED CURRICULUM (bpam-4 E2E): break the deadlock WITHOUT imposing any constellation. (1) joint train
    # at low SNR, where the ~2 dB BPAM advantage makes the loss prefer bipolar signalling, so the TX
    # self-organizes bipolar; (2) freeze the TX and train the RX to convergence, so it learns to decode that
    # bipolar TX (the RX was the piece that floored when co-adapting); (3) free joint fine-tune (main loop).
    if bpam and equalizer == "end-to-end" and getattr(config, "bpam_curriculum", False):
        if not getattr(transmitter, "precode_e2e", False):
            raise RuntimeError("bpam_curriculum needs bpam_precode_e2e=True (sign otherwise undecodable in DD)")
        def _bce(bits_b, logits_b):
            m = logits_b.shape[0]
            post = bit_posteriors(logits_b[guard:m - guard], config.bits_per_symbol)
            return F.binary_cross_entropy(post.clamp(1e-6, 1 - 1e-6), bits_b[:, guard:m - guard].float())
        # Stage 1 -- joint at low SNR (TX self-organizes bipolar; no imposed levels)
        lo = getattr(config, "curr_lowsnr_ebn0", 10.0)
        opt1 = torch.optim.Adam(list(transmitter.parameters()) + list(receiver.parameters()), lr=config.learning_rate)
        for _ in range(getattr(config, "curr_lowsnr_steps", 20000)):
            bits = random_bits(config.bits_per_symbol, window, device)
            noisy = link_photocurrent(transmitter, channel, bits, lo, config, receiver)
            opt1.zero_grad(); _bce(bits, receiver(noisy)).backward(); opt1.step()
        # Stage 2 -- freeze TX, train RX to convergence over the full SNR range
        opt2 = torch.optim.Adam(receiver.parameters(), lr=config.learning_rate)
        for _ in range(getattr(config, "curr_rxonly_steps", 20000)):
            ebn0_db = float(np.random.uniform(*ebn0_range))
            bits = random_bits(config.bits_per_symbol, window, device)
            noisy = link_photocurrent(transmitter, channel, bits, ebn0_db, config, receiver)
            opt2.zero_grad(); _bce(bits, receiver(noisy)).backward(); opt2.step()
        print("=== curriculum priming done (TX self-organized bipolar @low SNR, RX converged) -> joint fine-tune ===")

    params = list(receiver.parameters())                      # "ffe": train the FFE only (TX levels fixed)
    if equalizer == "end-to-end":
        params = list(transmitter.parameters()) + params      # "end-to-end": also train the DPD
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    # Optional LR decay. Default None = constant LR (every result before 22/7 was produced this way,
    # so those runs stay reproducible). A constant 1e-3 over 100k steps leaves the decision boundaries
    # jittering in gradient noise; that floor is invisible at low SNR (noise dominates) but sets the
    # high-SNR BER -- the same "damage grows with OSNR" signature as the gap to Li's receiver.
    scheduler = None
    if getattr(config, "lr_schedule", None) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=config.learning_rate * 0.01)
    # Differential-precoder target: when the TX precodes the sign (ffe always; E2E if precode_e2e),
    # the RX observes the sign STATE, not the raw phase bit. bpam_diff_target="coded" trains/evaluates
    # against that state and differential-decodes before the BER (Marco's well-posed scheme).
    precode_on = bpam and (equalizer != "end-to-end" or getattr(config, "bpam_precode_e2e", False))
    coded_target = precode_on and getattr(config, "bpam_diff_target", "raw") == "coded"
    loss_name = "symbol CE + per-bit BCE" if bpam else "symbol cross-entropy"
    if coded_target:
        loss_name += " (coded sign target)"
    print(f"=== training (equalizer={equalizer}) — {loss_name} ===")
    for step in range(num_steps):
        ebn0_db = float(np.random.uniform(*ebn0_range))
        bits = random_bits(config.bits_per_symbol, window, device)
        # target the RX is trained against: the coded sign state (well posed with the precoder) or the
        # raw bits (legacy / E2E without precoder). Row 0 (amplitude) is identical in both.
        target_bits = coded_bpam_target(bits) if coded_target else bits
        symbols = bits_to_symbols(target_bits, config.bits_per_symbol)
        noisy = link_photocurrent(transmitter, channel, bits, ebn0_db, config, receiver)  # ASE -> envelope; thermal -> photocurrent
        if bpam and receiver.bit_head is not None:
            bit_logits, logits = receiver.bit_and_symbol_logits(noisy)   # sigmoid bit head + aux symbol head
        else:
            bit_logits, logits = None, receiver(noisy)            # (num_symbols, M) symbol logits
        n = logits.shape[0]
        if bpam:
            # bpam-4 double-rate route: COMBINED loss = symbol cross-entropy + per-bit BCE. With sub-symbol
            # waveform freedom the TX rides each bit on an independent T/2 slot (double-rate OOK/hybrid), the
            # 4 symbols are absolutely DD-observable, and this goes below the symbol-rate PAM theory. BCE
            # alone abandons one bit (BER 0.25 collapse); the CE term keeps the 4 symbols distinct for the
            # whole training, BCE keeps the per-bit posteriors calibrated (soft-FEC LLRs).
            # With rx_bit_head the BCE acts on the SIGMOID head output z_k (Marco's formalism) and the
            # symbol softmax is a training-only auxiliary head.
            if bit_logits is not None:
                posteriors = torch.sigmoid(bit_logits[guard:n - guard]).T                  # (k, n) P(b=1)
            else:
                posteriors = bit_posteriors(logits[guard:n - guard], config.bits_per_symbol)  # (k, n) P(b=1)
            bce = F.binary_cross_entropy(posteriors.clamp(1e-6, 1 - 1e-6), target_bits[:, guard:n - guard].float())
            if getattr(config, "bpam_loss", "ce+bce") == "bce":
                loss = bce                                       # true BPAM: allow sign-degenerate intensities
            else:
                ce = F.cross_entropy(logits[guard:n - guard], symbols[guard:n - guard])
                loss = ce + bce                                  # double-rate/hybrid: CE keeps 4 levels distinct
        else:
            # upam-4: symbol cross-entropy (no collapse, no imposed levels). Measured: factorized bit
            # losses (BCE, soft-BER) fail the A.19 gate AND collapse on upam.
            loss = F.cross_entropy(logits[guard:n - guard], symbols[guard:n - guard])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % max(1, num_steps // 8) == 0 or step == num_steps - 1:
            log_post = (torch.sigmoid(bit_logits).T if bit_logits is not None
                        else bit_posteriors(logits, config.bits_per_symbol))
            if coded_target:                              # decide the sign state, then differential-decode
                decided = differential_decode((log_post > 0.5).to(log_post.dtype))
                ber = (decided[:, guard:n - guard] != bits[:, guard:n - guard]).float().mean().item()
            else:
                ber, _ = measure_ber(log_post, bits, guard)
            with torch.no_grad():
                spread = level_gap_spread(channel(transmitter(bits)), symbols, sps, guard, config.modulation_order)
            print(f"  step {step:4d}  ebn0 {ebn0_db:4.1f}  ce {loss.item():.8f}  gap-spread {spread:.3f}  BER {ber:.10f}")
    return transmitter, channel, receiver


def evaluate(transmitter, channel, receiver, config, ebn0_db_list, num_symbols, device):
    transmitter.eval()
    receiver.eval()
    k = config.bits_per_symbol
    equalizer = getattr(config, "equalizer", "end-to-end")
    precode_on = (getattr(config, "modulation_format", "upam-4") == "bpam-4"
                  and (equalizer != "end-to-end" or getattr(config, "bpam_precode_e2e", False)))
    coded_target = precode_on and getattr(config, "bpam_diff_target", "raw") == "coded"
    if getattr(config, "modulation_format", "upam-4") == "bpam-4":
        # bpam-4: the BCE training target pins the bit labelling to the plain BINARY map -> same map here
        sym_bits = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long, device=device)
    else:
        # amplitude-Gray labelling of the LEARNED levels -> per-bit LLRs reach A.19 (no 4/3 binary penalty)
        sym_bits = amplitude_gray_bits(transmitter, channel, receiver, config, device)
    results = []
    with torch.no_grad():
        bits = random_bits(k, num_symbols, device)
        ref_bits = sym_bits[bits_to_symbols(bits, k)].T                    # (k, N) amplitude-Gray reference
        use_bit_head = (getattr(config, "modulation_format", "upam-4") == "bpam-4"
                        and receiver.bit_head is not None)
        for ebn0_db in ebn0_db_list:
            if use_bit_head:                                               # z_k from the sigmoid head directly
                noisy = link_photocurrent(transmitter, channel, bits, ebn0_db, config, receiver)
                post = receiver.bit_posteriors_direct(noisy)
                if coded_target:                                           # decide sign state -> differential-decode -> BER vs raw bits
                    ber, shift = _measure_ber_diff(post, bits, config.edge_guard_symbols)
                else:
                    ber, shift = measure_ber(post, bits, config.edge_guard_symbols)
            else:
                logits = forward_link(transmitter, channel, receiver, bits, ebn0_db, config)
                post = bit_posteriors(logits, k, symbol_bits=sym_bits)     # P(b_i = 1) under amplitude-Gray
                ber, shift = measure_ber(post, ref_bits, config.edge_guard_symbols)
            results.append(ber)
            print(f"Eb/N0 {ebn0_db:5.1f} dB   BER {ber:.3e}   (shift {shift})")
    return np.array(results)


def _optimized_threshold(stat, labels, num_candidates=200):
    """Numerically optimized binary decision threshold (Secondini2020 optimizes all thresholds
    numerically, Sec. III: with square-law detection they are not midway between the levels)."""
    lo = stat[labels == 0]
    hi = stat[labels == 1]
    candidates = torch.linspace(float(lo.median()), float(hi.median()), num_candidates, device=stat.device)
    errors = torch.stack([(lo > c).float().mean() + (hi <= c).float().mean() for c in candidates])
    return candidates[int(errors.argmin())]


def _evaluate_threshold_bpam(transmitter, channel, receiver, config, ebn0_db_list, num_symbols, device):
    """Classical BPAM-4/DD detector of Secondini2020 as the no-DSP reference.
    Even (symbol-centre) samples of the photocurrent give the AMPLITUDE bit through an optimized
    threshold; the odd (T/2) samples, which sense the interference between adjacent pulses, form the
    auxiliary variable z_k = y'_k - c*(y_k + y_{k+1}) (their eq. 9; c = |h+-1/h0|^2 = 0.25, the ideal
    value, held fixed as in their realistic scenarios) and give the DIFFERENTIAL phase bit through a
    second optimized threshold. Thresholds are optimized on a training half and the BER is measured on
    the held-out half. ASE (optical-noise) regime only -- the regime the BPAM format targets."""
    if config.noise_regime != "ase":
        raise NotImplementedError("bpam-4 threshold detector: ASE regime only (the regime BPAM targets)")
    if config.samples_per_symbol_rx != 2:
        raise RuntimeError("bpam-4 needs samples_per_symbol_rx = 2 (config.set_modulation_format sets it)")
    guard = max(config.edge_guard_symbols, 8)
    results = []
    with torch.no_grad():
        bits = random_bits(2, num_symbols, device)        # row 0: amplitude bit, row 1: phase-difference bit
        drive = transmitter(bits)                         # differential sign precoding happens inside the TX

        def z2_stream(ebn0_db):
            xr, xi = channel.complex_envelope(drive, ebn0_db)
            mr = receiver.matched_and_decimate(xr).reshape(-1)
            mi = receiver.matched_and_decimate(xi).reshape(-1)
            return mr ** 2 + mi ** 2                      # photocurrent at 2 samples/symbol, interleaved

        # alignment on the noiseless statistic: sampling parity (which interleave is the symbol centre)
        # and integer symbol shift, chosen to maximize the amplitude-group separation on the even samples
        s0 = z2_stream(None)
        n = s0.shape[0] // 2
        best = None
        for parity in (0, 1):
            even0 = s0[parity::2][:n]
            for shift in range(-2, 3):
                a = torch.roll(bits[0, :n], shift)
                separation = even0[a == 1].mean() - even0[a == 0].mean()
                if best is None or separation > best[0]:
                    best = (separation, parity, shift)
        _, parity, shift = best
        amp_ref = torch.roll(bits[0, :n], shift)
        ph_ref = torch.roll(bits[1, :n], shift)
        half = n // 2                                     # thresholds trained on [0:half], BER on [half:n]
        for ebn0_db in ebn0_db_list:
            s = z2_stream(ebn0_db)
            even = s[parity::2][:n]
            odd = s[1 - parity::2][:n]
            amp_thr = _optimized_threshold(even[guard:half], amp_ref[guard:half])
            amp_err = ((even[half:n - guard] > amp_thr).long() != amp_ref[half:n - guard]).float().mean()
            # The T/2 sample odd_k straddles either the pulse pair (k, k+1) or (k-1, k), depending on the
            # decimation phase. The SUBTRACTED even samples in z (eq. 9) must be the SAME pair, and the
            # phase-bit label aligns accordingly: pair (k, k+1) -> z_k senses dphi_{k+1} (shift by 1);
            # pair (k-1, k) -> z_k senses dphi_k. Try both on the training half, keep the better.
            best_ph = None
            for direction in (-1, +1):                    # -1: pair (k, k+1);  +1: pair (k-1, k)
                z = odd - 0.25 * even - 0.25 * torch.roll(even, direction)
                zr = torch.roll(z, 1) if direction == -1 else z    # align z to dphi_k
                ph_thr = _optimized_threshold(-zr[guard:half], ph_ref[guard:half])
                train_err = ((-zr[guard:half] > ph_thr).long() != ph_ref[guard:half]).float().mean()
                test_err = ((-zr[half:n - guard] > ph_thr).long() != ph_ref[half:n - guard]).float().mean()
                if best_ph is None or train_err < best_ph[0]:
                    best_ph = (train_err, test_err)
            ber = 0.5 * float(amp_err) + 0.5 * float(best_ph[1])
            results.append(ber)
            print(f"Eb/N0 {ebn0_db:5.1f} dB   BER {ber:.3e}   (BPAM threshold detector)")
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
    if getattr(config, "modulation_format", "upam-4") == "bpam-4":
        return _evaluate_threshold_bpam(transmitter, channel, receiver, config, ebn0_db_list,
                                        num_symbols, device)
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
            if sig is None:
                raise RuntimeError(
                    "evaluate_threshold (ase): no shift gives 4 strictly increasing intensity means -- "
                    "the per-symbol statistic does not separate the levels (heavy ISI, e.g. freq-rect, "
                    "or a misaligned sampling phase). This simple threshold reference only applies to "
                    "low-ISI pulses (time-rect); use 'ffe'/'end-to-end' for band-limited pulses.")
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

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


def link_photocurrent(transmitter, channel, bits, ebn0_db, config):
    """Photocurrent with noise injected per the configured regime: ASE on the optical field
    (inside the channel) or thermal/electrical AWGN on the photocurrent (at the ADC)."""
    drive = transmitter(bits)
    if config.noise_regime == "ase":
        return channel(drive, ase_ebn0_db=ebn0_db)
    photocurrent = channel(drive)
    return add_awgn(photocurrent, ebn0_db, config.bits_per_symbol, config.samples_per_symbol_sim)


def forward_link(transmitter, channel, receiver, bits, ebn0_db, config):
    return receiver(link_photocurrent(transmitter, channel, bits, ebn0_db, config))


def train(config, device, num_steps, ebn0_range=(7.0, 19.0)):
    """Single-stage joint DPD+FFE training, BCE only: the constellation is LEARNED, not imposed."""
    transmitter = Transmitter(config).to(device)
    channel = OpticalChannel(config).to(device)
    receiver = Receiver(config).to(device)
    window = config.minibatch_symbols + 2 * config.edge_guard_symbols
    guard = config.edge_guard_symbols
    sps = config.samples_per_symbol_sim
    equalizer = getattr(config, "equalizer", "joint")
    if equalizer is None:                                     # threshold detector: nothing to train
        print("=== equalizer=None -> threshold detector (A.19 reference), no training ===")
        return transmitter, channel, receiver
    params = list(receiver.parameters())                      # "ffe": train the FFE only (TX levels fixed)
    if equalizer == "joint":
        params = list(transmitter.parameters()) + params      # "joint": also train the DPD
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    print(f"=== training (equalizer={equalizer}) — symbol cross-entropy ===")
    for step in range(num_steps):
        ebn0_db = float(np.random.uniform(*ebn0_range))
        bits = random_bits(config.bits_per_symbol, window, device)
        symbols = bits_to_symbols(bits, config.bits_per_symbol)
        drive = transmitter(bits)
        if config.noise_regime == "ase":
            noisy = channel(drive, ase_ebn0_db=ebn0_db)           # ASE beat noise on the field
        else:
            noisy = add_awgn(channel(drive), ebn0_db, config.bits_per_symbol, sps)
        logits = receiver(noisy)                                  # (num_symbols, M) symbol logits
        n = logits.shape[0]
        loss = F.cross_entropy(logits[guard:n - guard], symbols[guard:n - guard])   # symbol CE: no collapse, no imposed levels
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, num_steps // 8) == 0 or step == num_steps - 1:
            ber, _ = measure_ber(bit_posteriors(logits, config.bits_per_symbol), bits, guard)
            with torch.no_grad():
                spread = level_gap_spread(channel(drive), symbols, sps, guard, config.modulation_order)
            print(f"  step {step:4d}  ebn0 {ebn0_db:4.1f}  ce {loss.item():.4f}  gap-spread {spread:.3f}  BER {ber:.4f}")
    return transmitter, channel, receiver


def evaluate(transmitter, channel, receiver, config, ebn0_db_list, num_symbols, device):
    transmitter.eval()
    receiver.eval()
    results = []
    with torch.no_grad():
        bits = random_bits(config.bits_per_symbol, num_symbols, device)
        for ebn0_db in ebn0_db_list:
            logits = forward_link(transmitter, channel, receiver, bits, ebn0_db, config)
            ber, shift = measure_ber(bit_posteriors(logits, config.bits_per_symbol), bits, config.edge_guard_symbols)
            results.append(ber)
            print(f"Eb/N0 {ebn0_db:5.1f} dB   BER {ber:.3e}   (shift {shift})")
    return np.array(results)


def evaluate_threshold(transmitter, channel, receiver, config, ebn0_db_list, num_symbols, device):
    """A.19-reference detector (config.equalizer is None): matched filter / integrate-and-dump, then
    GENIE optimum thresholds (midway between the measured level means) + Gray decode. No DPD, no FFE.
    The TX emits fixed equispaced-intensity levels; this is exactly Forestieri's optimum receiver."""
    transmitter.eval()
    receiver.eval()
    M = config.modulation_order
    k = config.bits_per_symbol
    guard = config.edge_guard_symbols
    gray = torch.tensor([0, 1, 3, 2], device=device)[:M]          # symbol <-> intensity rank (involution)
    results = []
    with torch.no_grad():
        bits = random_bits(k, num_symbols, device)
        symbols = bits_to_symbols(bits, k)
        rank = gray[symbols]
        for ebn0_db in ebn0_db_list:
            photo = link_photocurrent(transmitter, channel, bits, ebn0_db, config)
            stream = receiver.matched_and_decimate(photo).squeeze()        # (num_symbols,) at the symbol rate
            best = 1.0                                                      # small group-delay shift search
            for shift in range(-2, 3):
                r = torch.roll(rank, shift)
                means = torch.stack([stream[r == a].mean() for a in range(M)])
                if not torch.all(means[1:] > means[:-1]):
                    continue
                thresholds = (means[:-1] + means[1:]) / 2                   # optimum: midway between levels
                est_rank = torch.bucketize(stream, thresholds).clamp(max=M - 1)
                est_sym = gray[est_rank]
                est_bits = torch.stack([(est_sym >> (k - 1 - i)) & 1 for i in range(k)])
                ref = torch.roll(bits, shift, dims=1)
                ber = (est_bits[:, guard:-guard] != ref[:, guard:-guard]).float().mean().item()
                best = min(best, ber)
            results.append(best)
            print(f"Eb/N0 {ebn0_db:5.1f} dB   BER {best:.3e}   (threshold detector)")
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

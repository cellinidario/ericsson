"""End-to-end joint training of the DPD + FFE through the differentiable channel.

Single stage, direct-learning architecture: DPD and FFE are trained TOGETHER, with
realistic noise and an equispacing loss. The gradient of the loss backpropagates
through the whole chain (TX -> channel -> RX). Long sliding windows of fresh random
bits are used each step, and edge_guard symbols are dropped at both ends so filter
transients never enter the loss (no tiny blocks).

The equispacing loss pins the DPD to 4 clean Gray-ordered levels, so it cannot
collapse the LSB under noise; meanwhile the noise makes the FFE MMSE (not zero-
forcing). This is why a single joint stage is enough (no separate FFE-only stage).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from channel import OpticalChannel
from transmitter import Transmitter, bits_to_symbols
from receiver import Receiver
from utils import add_awgn, measure_ber, theoretical_ber_unipolar

# PAM-4 Gray order: level rank assigned to each symbol so adjacent levels differ by 1 bit
GRAY_LEVEL = [0.0, 1.0, 3.0, 2.0]


def random_bits(num_bits, num_symbols, device):
    return torch.randint(0, 2, (num_bits, num_symbols), device=device)


def equispacing_loss(photocurrent, symbols, samples_per_symbol, guard):
    """MSE that forces the decision-instant intensity to be a Gray-ordered, equispaced
    function of the symbol -> 4 clean equispaced levels (a beautiful eye)."""
    gray_level = torch.tensor(GRAY_LEVEL, device=photocurrent.device)
    centers = photocurrent[::samples_per_symbol][:symbols.shape[0]]
    measured = centers[guard:symbols.shape[0] - guard]
    target = gray_level[symbols[guard:symbols.shape[0] - guard]]
    measured = (measured - measured.mean()) / (measured.std() + 1e-6)
    target = (target - target.mean()) / (target.std() + 1e-6)
    return ((measured - target) ** 2).mean()


def forward_link(transmitter, channel, receiver, bits, ebn0_db, config, add_noise=True):
    photocurrent = channel(transmitter(bits), add_noise=False)
    if add_noise:
        photocurrent = add_awgn(photocurrent, ebn0_db, config.bits_per_symbol, config.samples_per_symbol_sim)
    return receiver(photocurrent)


def train(config, device, num_steps, ebn0_range=(7.0, 19.0)):
    """Single-stage joint DPD+FFE training (BCE for decodability + equispacing for a clean eye)."""
    transmitter = Transmitter(config).to(device)
    channel = OpticalChannel(config).to(device)
    receiver = Receiver(config).to(device)
    window = config.minibatch_symbols + 2 * config.edge_guard_symbols
    guard = config.edge_guard_symbols
    sps = config.samples_per_symbol_sim
    optimizer = torch.optim.Adam(list(transmitter.parameters()) + list(receiver.parameters()),
                                 lr=config.learning_rate)
    print("=== joint training (DPD + FFE together, noise + equispacing) ===")
    for step in range(num_steps):
        ebn0_db = float(np.random.uniform(*ebn0_range))
        bits = random_bits(config.bits_per_symbol, window, device)
        symbols = bits_to_symbols(bits, config.bits_per_symbol)
        photocurrent = channel(transmitter(bits), add_noise=False)
        bit_probability = receiver(add_awgn(photocurrent, ebn0_db, config.bits_per_symbol, sps))
        target = bits.float()[:, guard:bit_probability.shape[1] - guard]
        prediction = bit_probability[:, guard:bit_probability.shape[1] - guard]
        bce = F.binary_cross_entropy(prediction, target)
        level = equispacing_loss(photocurrent, symbols, sps, guard)
        loss = bce + config.equispacing_weight * level
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(1, num_steps // 8) == 0 or step == num_steps - 1:
            ber, _ = measure_ber(bit_probability, bits, guard)
            print(f"  step {step:4d}  ebn0 {ebn0_db:4.1f}  bce {bce.item():.4f}  level {level.item():.4f}  BER {ber:.4f}")
    return transmitter, channel, receiver


def evaluate(transmitter, channel, receiver, config, ebn0_db_list, num_symbols, device):
    transmitter.eval()
    receiver.eval()
    results = []
    with torch.no_grad():
        bits = random_bits(config.bits_per_symbol, num_symbols, device)
        for ebn0_db in ebn0_db_list:
            bit_probability = forward_link(transmitter, channel, receiver, bits, ebn0_db, config)
            ber, shift = measure_ber(bit_probability, bits, config.edge_guard_symbols)
            results.append(ber)
            print(f"Eb/N0 {ebn0_db:5.1f} dB   BER {ber:.3e}   (shift {shift})")
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
    theory = theoretical_ber_unipolar(ebn0_list, config.modulation_order)
    for e, m, t in zip(ebn0_list, measured, theory):
        print(f"Eb/N0 {e:4.0f}  measured {m:.2e}  theory(unipolar) {t:.2e}")

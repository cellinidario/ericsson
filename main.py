"""Single entry point: train the unipolar PAM-4 end-to-end, then produce the two
deliverables — a clean eye diagram and a BER vs Eb/N0 waterfall against theory.

The Eb/N0 axis is calibrated with an ideal-PAM-4 reference through the identical
pipeline, so the comparison with the textbook curve is honest.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

from config import Config
from train import train, evaluate, random_bits
from transmitter import bits_to_symbols
from utils import (ideal_reference_ber, theoretical_ber_unipolar, decision_phase_by_separation,
                   calibrate_ebn0_offset)


def draw_eye(ax, photocurrent, samples_per_symbol, phase, title, fine_sps=40, skip=100, num_symbols=4000):
    fine = resample_poly(photocurrent[:(skip + num_symbols + 8) * samples_per_symbol], fine_sps, samples_per_symbol)
    offset = int(round(phase * fine_sps / samples_per_symbol))
    centers = (skip + np.arange(num_symbols)) * fine_sps + offset
    keep = (centers - fine_sps >= 0) & (centers + fine_sps < len(fine))
    centers = centers[keep]
    window = np.arange(-fine_sps, fine_sps)[:, None] + centers[None, :]
    ax.plot(np.arange(-fine_sps, fine_sps) / fine_sps, fine[window], color=(0.85, 0.33, 0.10, 0.03), linewidth=0.4)
    ax.axvline(0, color="k", linestyle=":", linewidth=1)
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Symbol time")
    ax.set_ylabel("Photodetected power")
    ax.set_title(title)


def main():
    config = Config()
    config.minibatch_symbols = 4096
    config.edge_guard_symbols = 48
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)

    transmitter, channel, receiver = train(config, device, num_steps=7000)
    sps = config.samples_per_symbol_sim

    # ----- eye + levels -----
    with torch.no_grad():
        bits = random_bits(config.bits_per_symbol, 6000, device)
        symbols = bits_to_symbols(bits, config.bits_per_symbol).cpu().numpy()
        photocurrent = channel(transmitter(bits), add_noise=False).cpu().numpy()
    phase, levels = decision_phase_by_separation(photocurrent, sps, symbols)
    gaps = np.diff(np.sort(levels))
    print(f"\nlevel means (sym 0..3): {np.round(levels, 4)}")
    print(f"sorted gaps: {np.round(gaps, 4)}  equispacing std/mean = {gaps.std() / gaps.mean():.3f}")

    # ----- BER waterfall + honest (reference-calibrated) Eb/N0 axis -----
    ebn0_db = np.arange(8, 26, 2.0)
    print("\n--- E2E ---")
    measured = evaluate(transmitter, channel, receiver, config, ebn0_db, 400000, device)
    print("--- ideal reference ---")
    reference = ideal_reference_ber(ebn0_db, config, 400000, device)
    offset = calibrate_ebn0_offset(reference, ebn0_db, config.modulation_order)
    print(f"Eb/N0 calibration offset (definition) = {offset:.2f} dB")
    theory = theoretical_ber_unipolar(ebn0_db - offset, config.modulation_order)

    # ----- figure: eye (left) + waterfall (right) -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_eye(axes[0], photocurrent, sps, phase, "Unipolar PAM-4 eye (photodetected)")
    floor = lambda a: np.maximum(a, 1e-7)
    axes[1].semilogy(ebn0_db - offset, floor(measured), "-o", linewidth=2, label="E2E autoencoder (DPD + FFE)")
    axes[1].semilogy(ebn0_db - offset, floor(theory), "k--", linewidth=1.5, label="Theoretical unipolar PAM-4")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("Eb/N0 [dB] (calibrated)")
    axes[1].set_ylabel("BER")
    axes[1].set_ylim(1e-7, 1)
    axes[1].set_title("BER vs Eb/N0")
    axes[1].legend(loc="lower left")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "pam4_eye_and_ber.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out_path}")

    # save the trained configuration (weights + params + results) so we can restore it
    checkpoint = {
        "transmitter": transmitter.state_dict(),
        "receiver": receiver.state_dict(),
        "config": {k: v for k, v in vars(config).items()},
        "ebn0_db": ebn0_db,
        "measured": measured,
        "reference": reference,
        "calibration_offset_db": offset,
    }
    torch.save(checkpoint, os.path.join(results_dir, "pam4_baseline.pt"))
    print("saved checkpoint pam4_baseline.pt")


if __name__ == "__main__":
    main()

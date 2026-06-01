"""Single entry point: train the unipolar PAM-4 end-to-end, then produce the two
deliverables — a clean eye diagram and a BER vs Eb/N0 waterfall against theory.

The x-axis is Eb/N0 (energy per bit / one-sided N0) — the convention-free figure of
merit, with no bandwidth choice to argue about. The theoretical reference is chosen by
config.noise_regime: Forestieri eq.(A.19) for thermal/electrical noise, eq.(A.47) for the
ASE-dominant regime (signal-spontaneous beat noise). Both are BER(Eb/N0) directly.
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
from utils import theoretical_ber_unipolar, theoretical_ber_unipolar_ase, decision_phase_by_separation


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

    transmitter, channel, receiver = train(config, device, num_steps=10000)
    sps = config.samples_per_symbol_sim

    # ----- eye + levels -----
    with torch.no_grad():
        bits = random_bits(config.bits_per_symbol, 6000, device)
        symbols = bits_to_symbols(bits, config.bits_per_symbol).cpu().numpy()
        photocurrent = channel(transmitter(bits)).cpu().numpy()
    phase, levels = decision_phase_by_separation(photocurrent, sps, symbols)
    gaps = np.diff(np.sort(levels))
    print(f"\nlevel means (sym 0..3): {np.round(levels, 4)}")
    print(f"sorted gaps: {np.round(gaps, 4)}  equispacing std/mean = {gaps.std() / gaps.mean():.3f}")

    # ----- BER waterfall vs Eb/N0 (energy per bit / one-sided N0: convention-free) -----
    ebn0_db = np.arange(6, 22, 2.0)
    print("\n--- E2E ---")
    measured = evaluate(transmitter, channel, receiver, config, ebn0_db, 400000, device)
    # theory reference depends on the noise regime (both are BER(Eb/N0) directly)
    if config.noise_regime == "ase":
        theory = theoretical_ber_unipolar_ase(ebn0_db, config.modulation_order)
        theory_label = "Theory unipolar PAM-4 (ASE, A.47)"
    else:
        theory = theoretical_ber_unipolar(ebn0_db, config.modulation_order)
        theory_label = "Theory unipolar PAM-4 (thermal, A.19)"

    # ----- figure: eye (left) + waterfall (right) -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_eye(axes[0], photocurrent, sps, phase, "Unipolar PAM-4 eye (photodetected)")
    # floor measured at the smallest resolvable BER (1/num_eval), theory is left unfloored
    measured_floor = np.maximum(measured, 1.0 / 400000)
    axes[1].semilogy(ebn0_db, measured_floor, "-o", linewidth=2, label="E2E autoencoder (DPD + FFE)")
    axes[1].semilogy(ebn0_db, theory, "k--", linewidth=1.5, label=theory_label)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel(r"$E_b/N_0$ [dB]")
    axes[1].set_ylabel("BER")
    axes[1].set_ylim(1e-9, 1)
    axes[1].set_title(f"BER vs Eb/N0  ({config.noise_regime} noise)")
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
    }
    checkpoint_name = f"pam4_{config.noise_regime}.pt"          # one checkpoint per noise regime
    torch.save(checkpoint, os.path.join(results_dir, checkpoint_name))
    print(f"saved checkpoint {checkpoint_name}")


if __name__ == "__main__":
    main()

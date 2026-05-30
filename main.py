"""Single entry point: train the unipolar PAM-4 end-to-end, then produce the two
deliverables — a clean eye diagram and a BER vs SNR waterfall against theory.

The x-axis is the per-sample SNR (signal_power / noise_variance) as set by add_awgn.
The theoretical PAM-4 curve from Forestieri eq.(A.19) is given as BER(Eb/N0) with
N0 one-sided; it is converted to BER(SNR) via Eb/N0 = SNR + 10*log10(B/R) with
B = f_s/2 (one-sided noise bandwidth) and R = R_b = k * R_s.
For K_sim=4, k=2 the conversion is identically zero (Eb/N0 = SNR).
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
from utils import theoretical_ber_unipolar, decision_phase_by_separation


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

    # ----- BER waterfall vs SNR (per-sample) -----
    # sweep directly on integer SNR; add_awgn takes a label such that
    #   snr_per_sample_db = label + 10*log10(k/K_sim)
    # so label = SNR - 10*log10(k/K_sim) (= SNR + 3 dB for K=4, k=2)
    snr_db = np.arange(6, 22, 2.0)
    label_db = snr_db - 10 * np.log10(config.bits_per_symbol / sps)
    print("\n--- E2E ---")
    measured = evaluate(transmitter, channel, receiver, config, label_db, 400000, device)
    # Forestieri (A.19) in terms of Eb/N0 (one-sided N0); convert with Eb/N0 = SNR + 10*log10(B/R),
    # B = f_s/2, R = R_b -> B/R = K_sim/(2k). For K_sim=4, k=2 this is exactly 0 dB.
    ebn0_for_theory = snr_db + 10 * np.log10(sps / (2 * config.bits_per_symbol))
    theory = theoretical_ber_unipolar(ebn0_for_theory, config.modulation_order)

    # ----- figure: eye (left) + waterfall (right) -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_eye(axes[0], photocurrent, sps, phase, "Unipolar PAM-4 eye (photodetected)")
    # floor measured at the smallest resolvable BER (1/num_eval), theory is left unfloored
    measured_floor = np.maximum(measured, 1.0 / 400000)
    axes[1].semilogy(snr_db, measured_floor, "-o", linewidth=2, label="E2E autoencoder (DPD + FFE)")
    axes[1].semilogy(snr_db, theory, "k--", linewidth=1.5, label="Theoretical unipolar PAM-4")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_xlabel("SNR [dB]")
    axes[1].set_ylabel("BER")
    axes[1].set_ylim(1e-9, 1)
    axes[1].set_title("BER vs SNR")
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
        "label_db": label_db,
        "snr_db": snr_db,
        "measured": measured,
    }
    torch.save(checkpoint, os.path.join(results_dir, "pam4_baseline.pt"))
    print("saved checkpoint pam4_baseline.pt")


if __name__ == "__main__":
    main()

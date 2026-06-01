"""Side-by-side BER waterfalls for the two noise regimes, each against its own theory:
thermal/electrical (Forestieri A.19) and optical ASE beat noise (A.47). Same link, same
training recipe, only the noise differs. Trains a per-regime checkpoint if missing.

Goal: make visible how close the E2E autoencoder gets to ITS bound in each regime. The two
theory curves nearly coincide (A.47 is ~0.5 dB from A.19), so the ASE optimum is essentially
the thermal one -- any extra distance on the ASE panel is the intensity-domain receiver,
not the noise.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from transmitter import Transmitter
from receiver import Receiver
from train import train, evaluate
from utils import theoretical_ber_unipolar, theoretical_ber_unipolar_ase

NUM_STEPS = 10000
NUM_EVAL = 400000
EBN0_DB = np.arange(6, 22, 2.0)


def load_or_train(regime, device):
    """Load results/pam4_<regime>.pt or train it from scratch (and cache it)."""
    config = Config()
    config.noise_regime = regime
    config.minibatch_symbols = 4096
    config.edge_guard_symbols = 48
    path = os.path.join("results", f"pam4_{regime}.pt")
    if os.path.exists(path):
        print(f"[{regime}] loading {path}")
        ckpt = torch.load(path, weights_only=False, map_location=device)
        return config, np.asarray(ckpt["ebn0_db"]), np.asarray(ckpt["measured"])
    print(f"[{regime}] no checkpoint -> training from scratch")
    torch.manual_seed(0)
    np.random.seed(0)
    transmitter, channel, receiver = train(config, device, num_steps=NUM_STEPS)
    measured = evaluate(transmitter, channel, receiver, config, EBN0_DB, NUM_EVAL, device)
    os.makedirs("results", exist_ok=True)
    torch.save({"transmitter": transmitter.state_dict(), "receiver": receiver.state_dict(),
                "config": {k: v for k, v in vars(config).items()},
                "ebn0_db": EBN0_DB, "measured": measured}, path)
    print(f"[{regime}] saved {path}")
    return config, EBN0_DB, measured


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panels = [
        ("thermal", theoretical_ber_unipolar, "Thermal (electrical) noise", "Theory A.19"),
        ("ase", theoretical_ber_unipolar_ase, "Optical ASE beat noise", "Theory A.47"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    print("\n  Eb/N0       thermal E2E       ASE E2E")
    rows = {}
    for ax, (regime, theory_fn, title, theory_label) in zip(axes, panels):
        config, ebn0_db, measured = load_or_train(regime, device)
        theory = theory_fn(ebn0_db, config.modulation_order)
        floor = np.maximum(measured, 1.0 / NUM_EVAL)
        ax.semilogy(ebn0_db, floor, "-o", linewidth=2, label="E2E autoencoder (DPD + FFE)")
        ax.semilogy(ebn0_db, theory, "k--", linewidth=1.5, label=theory_label)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel(r"$E_b/N_0$ [dB]")
        ax.set_ylim(1e-9, 1)
        ax.set_title(title)
        ax.legend(loc="lower left")
        rows[regime] = measured
    axes[0].set_ylabel("BER")
    fig.suptitle("E2E unipolar PAM-4 — same link, two noise regimes, each vs its own bound")
    fig.tight_layout()
    out_path = os.path.join("results", "pam4_regime_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out_path}")

    for e, m_th, m_ase in zip(EBN0_DB, rows["thermal"], rows["ase"]):
        print(f"  {e:5.1f}       {m_th:.3e}        {m_ase:.3e}")


if __name__ == "__main__":
    main()

"""Point-by-point spectral analysis of the ASE link, to reason about the filter bandwidths.
Drives the channel with the trained DPD (results/pam4_ase.pt), captures the signal/ASE PSDs at
each stage and overlays the filter responses, in the optical domain (complex field, before the
photodiode) and the electrical domain (real photocurrent, after it). Prints the key numbers and
saves results/pam4_spectral.png. Run from the MLforCPO2 folder.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from pulse_shaping import root_raised_cosine
from config import Config
from transmitter import Transmitter
from channel import OpticalChannel
from train import random_bits

EBN0_DB = 16.0
CHECKPOINT = os.path.join("results", "pam4_ase.pt")


def psd(signal, sample_rate, onesided):
    freq, power = welch(signal, sample_rate, nperseg=8192, return_onesided=onesided, detrend="constant")
    if not onesided:
        freq, power = np.fft.fftshift(freq), np.fft.fftshift(power)
    return freq / 1e9, power


def filter_mag_db(taps, sample_rate):
    n = 8192
    transfer = np.fft.fftshift(np.fft.fft(taps, n))
    freq = np.fft.fftshift(np.fft.fftfreq(n, 1 / sample_rate)) / 1e9
    return freq, 20 * np.log10(np.abs(transfer) + 1e-12)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CHECKPOINT, weights_only=False, map_location=device)
    config = Config()
    config.noise_regime = "ase"
    config.rx_sqrt_companding = ckpt.get("rx_sqrt_companding")     # restore the build-time choice
    transmitter = Transmitter(config).to(device)
    transmitter.load_state_dict(ckpt["transmitter"])
    transmitter.eval()
    channel = OpticalChannel(config).to(device)
    sample_rate = config.sim_sample_rate

    # ----- run the trained signal through the channel, capturing each stage -----
    torch.manual_seed(1)
    bits = random_bits(config.bits_per_symbol, 60000, device)
    with torch.no_grad():
        drive = transmitter(bits)
        combined = channel.segment_filter(drive.unsqueeze(0)).sum(dim=1, keepdim=True) / channel.num_segments
        phase = (np.pi / (2 * channel.vpi)) * (combined - channel.bias_volt)
        field_real = 0.5 * (1 + channel.gamma) * torch.cos(phase) * channel.field_loss
        field_imag = 0.5 * (1 - channel.gamma) * torch.sin(phase) * channel.field_loss
        field_real, field_imag = channel._apply_dispersion(field_real, field_imag)
        mean_power = (field_real.pow(2) + field_imag.pow(2)).mean()
        std = torch.sqrt(mean_power * (config.samples_per_symbol_sim / config.bits_per_symbol)
                         / (2 * 10 ** (EBN0_DB / 10)))
        noise_real, noise_imag = std * torch.randn_like(field_real), std * torch.randn_like(field_imag)
        photocurrent = (channel.optical_filter(field_real + noise_real).pow(2)
                        + channel.optical_filter(field_imag + noise_imag).pow(2))
        photocurrent_pd = channel.pd_filter(photocurrent)

    signal_field = (field_real + 1j * field_imag).squeeze().cpu().numpy()
    ase_field = (noise_real + 1j * noise_imag).squeeze().cpu().numpy()
    photocurrent_np = photocurrent.squeeze().cpu().numpy()
    photocurrent_pd_np = photocurrent_pd.squeeze().cpu().numpy()

    f_sig, p_sig = psd(signal_field, sample_rate, False)
    f_ase, p_ase = psd(ase_field, sample_rate, False)
    f_pc, p_pc = psd(photocurrent_np, sample_rate, True)
    f_pcpd, p_pcpd = psd(photocurrent_pd_np, sample_rate, True)

    # ----- key numbers -----
    cutoff = config.optical_filter_bandwidth / 2e9
    sig_peak = np.median(p_sig[np.abs(f_sig) < 8])
    sig_at_edge = np.interp(cutoff, f_sig[f_sig >= 0], p_sig[f_sig >= 0])
    ase_level = np.median(p_ase)
    ase_kept = p_ase[np.abs(f_ase) <= cutoff].sum() / p_ase.sum()
    sig_db = 10 * np.log10(p_sig[f_sig >= 0] / sig_peak)
    bw10 = f_sig[f_sig >= 0][np.where(sig_db > -10)[0][-1]]
    print(f"symbol rate                       : {config.symbol_rate/1e9:.0f} GBaud "
          f"(RRC one-sided band ~ {config.symbol_rate*(1+config.rrc_rolloff)/2e9:.1f} GHz)")
    print(f"optical filter cutoff (B_o/2)     : {cutoff:.1f} GHz")
    print(f"signal -10 dB one-sided bandwidth : {bw10:.1f} GHz")
    print(f"signal at the optical cutoff      : {10*np.log10(sig_at_edge/sig_peak):+.1f} dB re peak")
    print(f"signal-to-ASE PSD ratio (in band) : {10*np.log10(sig_peak/ase_level):+.1f} dB")
    print(f"ASE power kept by optical filter  : {100*ase_kept:.0f}% (cut {100*(1-ase_kept):.0f}%)")
    print(f"photodiode cutoff                 : {config.photodiode_bandwidth/1e9:.0f} GHz")

    # ----- figure -----
    opt_taps = channel.optical_filter.weight.detach().cpu().numpy().flatten()[::-1]
    pd_taps = channel.pd_filter.weight.detach().cpu().numpy().flatten()[::-1]
    rrc_taps = root_raised_cosine(config.rrc_rolloff, config.rrc_span_symbols, config.samples_per_symbol_sim)
    f_opt, h_opt = filter_mag_db(opt_taps, sample_rate)
    f_pdf, h_pd = filter_mag_db(pd_taps, sample_rate)
    f_rrc, h_rrc = filter_mag_db(rrc_taps, sample_rate)

    ref = 10 * np.log10(sig_peak)
    fig, (ax_opt, ax_elec) = plt.subplots(1, 2, figsize=(15, 5.6))
    ax_opt.plot(f_sig, 10 * np.log10(p_sig) - ref, color="C0", lw=1.6, label="optical signal field", zorder=3)
    ax_opt.plot(f_ase, 10 * np.log10(p_ase) - ref, color="C3", lw=1.4, label="ASE (input, white)", zorder=2)
    ax_opt.plot(f_opt, h_opt, color="k", lw=1.4, ls="--",
                label=f"optical filter |H| ({channel.optical_filter_type})", zorder=4)
    ax_opt.axvspan(-cutoff, cutoff, color="C2", alpha=0.07)
    ax_opt.set_xlim(-40, 40)
    ax_opt.set_ylim(-50, 5)
    ax_opt.set_xlabel("frequency [GHz]  (0 = optical carrier)")
    ax_opt.set_ylabel("PSD / |H| [dB] (re signal peak)")
    ax_opt.set_title("Optical domain — complex field, before the PD")
    ax_opt.grid(alpha=0.3)
    ax_opt.legend(loc="lower center", fontsize=8)

    ref_pc = 10 * np.log10(np.median(p_pc[f_pc < 3]))
    ax_elec.plot(f_pc, 10 * np.log10(p_pc) - ref_pc, color="C0", lw=1.6, label="photocurrent (signal+beat)", zorder=3)
    ax_elec.plot(f_pcpd, 10 * np.log10(p_pcpd) - ref_pc, color="C2", lw=1.4, label="after PD low-pass", zorder=2)
    ax_elec.plot(f_pdf, h_pd, color="C1", lw=1.4, ls="--", label=f"PD |H| ({config.photodiode_bandwidth/1e9:.0f} GHz)", zorder=4)
    ax_elec.plot(f_rrc, h_rrc - h_rrc.max(), color="k", lw=1.4, ls=":", label="RRC matched |H|", zorder=4)
    ax_elec.set_xlim(0, 40)
    ax_elec.set_ylim(-50, 5)
    ax_elec.set_xlabel("frequency [GHz]")
    ax_elec.set_ylabel("PSD / |H| [dB]")
    ax_elec.set_title("Electrical domain — photocurrent, after the PD")
    ax_elec.grid(alpha=0.3)
    ax_elec.legend(loc="lower center", fontsize=8)

    fig.suptitle(f"ASE link spectral analysis @ Eb/N0 = {EBN0_DB:.0f} dB")
    fig.tight_layout()
    out_path = os.path.join("results", "pam4_spectral.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

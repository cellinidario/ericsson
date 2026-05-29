"""Physics-inspired differentiable optical channel (IM/DD).

Mirrors the device physics, as in the digital-surrogate approach:
  segmented MZM  : one FIR filter per segment (electro-optic + driver low-pass),
                   then the two arms via cos/sin, recombined into the optical field
  optical fiber  : scalar loss (1310 nm -> chromatic dispersion ~ 0)
  photodiode     : square-modulus, additive Gaussian thermal noise, low-pass filter

Everything is differentiable, so the E2E gradient backpropagates through it. The FIR
coefficients are set analytically here (the "true" channel); the same module can be
re-fit on white-noise data to play the role of a learned surrogate.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # conda MKL + torch OpenMP coexistence (Windows)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def lowpass_fir(cutoff_hz, sample_rate, num_taps):
    """Unit-DC-gain windowed-sinc low-pass FIR (odd length)."""
    if num_taps % 2 == 0:
        num_taps = num_taps + 1
    middle = (num_taps - 1) / 2
    normalized_cutoff = cutoff_hz / (sample_rate / 2)      # 1.0 == Nyquist
    n = np.arange(num_taps) - middle
    ideal = normalized_cutoff * np.sinc(normalized_cutoff * n)
    window = np.hamming(num_taps)
    taps = ideal * window
    taps = taps / np.sum(taps)
    return taps.astype(np.float32)


class OpticalChannel(nn.Module):
    def __init__(self, config, segment_filter_taps=65, pd_filter_taps=65):
        super().__init__()
        self.num_segments = config.num_mzm_segments
        self.vpi = config.mzm_vpi_volt
        self.bias_volt = config.mzm_vpi_volt                      # default operating point
        extinction_linear = config.extinction_ratio_linear
        self.gamma = (np.sqrt(extinction_linear) - 1) / (np.sqrt(extinction_linear) + 1)
        self.field_loss = float(np.sqrt(config.fiber_loss_linear))
        self.thermal_noise_std = 0.0                              # set later for an operating point

        # one low-pass FIR per MZM segment (electro-optic + driver bandwidth)
        segment_taps = lowpass_fir(config.mzm_bandwidth, config.sim_sample_rate, segment_filter_taps)
        segment_weight = torch.tensor(segment_taps).flip(0).repeat(self.num_segments, 1, 1)
        self.segment_filter = nn.Conv1d(self.num_segments, self.num_segments,
                                        kernel_size=len(segment_taps), groups=self.num_segments,
                                        padding=len(segment_taps) // 2, bias=False)
        self.segment_filter.weight = nn.Parameter(segment_weight)

        # photodiode low-pass FIR (single channel: the photocurrent)
        pd_taps = lowpass_fir(config.photodiode_bandwidth, config.sim_sample_rate, pd_filter_taps)
        pd_weight = torch.tensor(pd_taps).flip(0).view(1, 1, -1)
        self.pd_filter = nn.Conv1d(1, 1, kernel_size=len(pd_taps), padding=len(pd_taps) // 2, bias=False)
        self.pd_filter.weight = nn.Parameter(pd_weight)

    def set_noise_from_signal(self, photocurrent, ebn0_db, bits_per_symbol, samples_per_symbol_sim):
        """Set thermal-noise std so the link sits at a given Eb/N0 (measured signal power)."""
        signal_power = photocurrent.detach().pow(2).mean().item()
        # per-sample noise power for target Eb/N0: N0*f_s spread over the sim band
        ebn0_linear = 10 ** (ebn0_db / 10)
        energy_per_bit = signal_power * samples_per_symbol_sim / bits_per_symbol
        noise_power = energy_per_bit / ebn0_linear
        self.thermal_noise_std = float(np.sqrt(noise_power))

    def forward(self, drive, add_noise=True):
        """drive: (num_segments, num_samples) at the sim rate -> photocurrent: (num_samples,)."""
        x = drive.unsqueeze(0)                                    # (1, segments, samples)
        filtered = self.segment_filter(x)                         # per-segment EO/driver low-pass
        drive_sum = filtered.sum(dim=1, keepdim=True)             # combine segments -> (1,1,samples)

        phase = (np.pi / (2 * self.vpi)) * (drive_sum - self.bias_volt)
        field_real = 0.5 * (1 + self.gamma) * torch.cos(phase)
        field_imag = 0.5 * (1 - self.gamma) * torch.sin(phase)
        field_real = field_real * self.field_loss
        field_imag = field_imag * self.field_loss

        photocurrent = field_real.pow(2) + field_imag.pow(2)      # square-law detection
        if add_noise and self.thermal_noise_std > 0:
            photocurrent = photocurrent + self.thermal_noise_std * torch.randn_like(photocurrent)
        photocurrent = self.pd_filter(photocurrent)               # photodiode bandwidth
        return photocurrent.squeeze(0).squeeze(0)


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channel = OpticalChannel(cfg).to(device)

    # static transfer check: sweep a DC drive level, look at output intensity (1 segment)
    levels = torch.linspace(0, 2 * cfg.mzm_vpi_volt, 9, device=device)
    intensity = []
    for level in levels:
        drive = level * torch.ones(cfg.num_mzm_segments, 4096, device=device)
        out = channel(drive, add_noise=False)
        intensity.append(out[2048].item())
    print("DC drive  :", np.round(levels.cpu().numpy(), 2))
    print("intensity :", np.round(np.array(intensity), 4))

    # gradient check: loss must backpropagate to the drive
    drive = torch.zeros(cfg.num_mzm_segments, 8192, device=device, requires_grad=True)
    out = channel(drive, add_noise=False)
    out.pow(2).mean().backward()
    print("grad to drive ok, mean|grad| =", float(drive.grad.abs().mean()))
    print("device:", device)

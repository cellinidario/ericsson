"""Physics-inspired differentiable optical channel (IM/DD).

Mirrors the device physics, as in the digital-surrogate approach:
  segmented MZM  : one FIR low-pass per segment (each with its own bandwidth,
                   modelling driver mismatch), the segment outputs are AVERAGED
                   to drive the MZM phase -> the operating region is invariant
                   in the number of segments (V_pi-normalized), then cos/sin into
                   the optical field
  optical fiber  : scalar loss + chromatic dispersion exp(j*beta2*L*omega^2/2)
                   applied to the COMPLEX optical envelope (1310 nm: small beta2)
  ASE noise      : complex Gaussian added to the optical FIELD before detection
                   (optically-preamplified receiver, single polarization). Since
                   it precedes the square law it becomes signal-dependent beat
                   noise (signal-spontaneous) after detection: upper levels get
                   noisier. Reference: Forestieri eq.(A.47).
  photodiode     : square-modulus, low-pass FIR

Everything is differentiable, so the E2E gradient backpropagates through it.
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
        self.sim_sample_rate = config.sim_sample_rate
        self.samples_per_symbol_sim = config.samples_per_symbol_sim   # K_sim, for the Eb/N0 bookkeeping
        self.bits_per_symbol = config.bits_per_symbol                 # k

        # one low-pass FIR per MZM segment with its own bandwidth (driver mismatch)
        scales = self._segment_scales(config)
        segment_taps = np.stack([lowpass_fir(scale * config.mzm_bandwidth,
                                             config.sim_sample_rate, segment_filter_taps)
                                 for scale in scales], axis=0)
        # Conv1d cross-correlates -> flip taps to implement actual convolution; weight shape (Cout, Cin/groups, K)
        segment_weight = torch.tensor(segment_taps[:, ::-1].copy()).unsqueeze(1)
        self.segment_filter = nn.Conv1d(self.num_segments, self.num_segments,
                                        kernel_size=segment_filter_taps, groups=self.num_segments,
                                        padding=segment_filter_taps // 2, bias=False)
        self.segment_filter.weight = nn.Parameter(segment_weight)

        # fiber chromatic dispersion: H(omega) = exp(j * beta2 * L / 2 * omega^2)
        # beta2 [ps^2/km] -> [s^2/m]:  1 ps^2/km = 1e-24 s^2 / 1e3 m = 1e-27 s^2/m
        beta2_s2_per_m = config.fiber_beta2_ps2_per_km * 1e-27
        fiber_length_m = config.fiber_length_km * 1e3
        self.dispersion_coeff = 0.5 * beta2_s2_per_m * fiber_length_m   # [s^2], signed

        # photodiode low-pass FIR (single channel: the photocurrent)
        pd_taps = lowpass_fir(config.photodiode_bandwidth, config.sim_sample_rate, pd_filter_taps)
        pd_weight = torch.tensor(pd_taps).flip(0).view(1, 1, -1)
        self.pd_filter = nn.Conv1d(1, 1, kernel_size=len(pd_taps), padding=len(pd_taps) // 2, bias=False)
        self.pd_filter.weight = nn.Parameter(pd_weight)

    def _segment_scales(self, config):
        """Per-segment bandwidth multipliers (defaults to all-equal if not enough are listed)."""
        scales = getattr(config, "segment_bandwidth_scales", None)
        if scales is None or len(scales) < self.num_segments:
            return np.ones(self.num_segments, dtype=np.float32)
        return np.asarray(scales[:self.num_segments], dtype=np.float32)

    def _apply_dispersion(self, field_real, field_imag):
        """Circular CD via FFT on the complex envelope. Negligible wrap for small beta2*L."""
        if self.dispersion_coeff == 0.0:
            return field_real, field_imag
        field = torch.complex(field_real.squeeze(0).squeeze(0),
                              field_imag.squeeze(0).squeeze(0))
        num_samples = field.shape[-1]
        omega = 2 * np.pi * torch.fft.fftfreq(num_samples, d=1.0 / self.sim_sample_rate,
                                              device=field.device, dtype=torch.float32)
        transfer = torch.exp(1j * self.dispersion_coeff * omega ** 2)
        out = torch.fft.ifft(torch.fft.fft(field) * transfer)
        return out.real.view(1, 1, -1), out.imag.view(1, 1, -1)

    def _add_ase(self, field_real, field_imag, ebn0_db):
        """Complex Gaussian ASE on the optical field, scaled to a target Eb/N0 (one-sided N0,
        single polarization, as in Forestieri A.47).

        Eb/N0 = Ps * (K_sim / k) / (2 * sigma^2)  ->  sigma^2 per quadrature, where
        Ps = mean optical field power and Eb/N0 is the photons-per-bit at the preamp input.
        With this scaling an ideal matched-filter envelope receiver reproduces (A.47) exactly.
        """
        mean_optical_power = (field_real.pow(2) + field_imag.pow(2)).mean().detach()
        snr_linear = 10 ** (ebn0_db / 10)
        quad_var = mean_optical_power * (self.samples_per_symbol_sim / self.bits_per_symbol) / (2 * snr_linear)
        std = torch.sqrt(quad_var)
        field_real = field_real + std * torch.randn_like(field_real)
        field_imag = field_imag + std * torch.randn_like(field_imag)
        return field_real, field_imag

    def forward(self, drive, ase_ebn0_db=None):
        """drive: (num_segments, num_samples) at the sim rate -> photocurrent: (num_samples,).
        ase_ebn0_db=None -> noiseless; otherwise inject ASE on the field at that Eb/N0."""
        x = drive.unsqueeze(0)                                    # (1, segments, samples)
        filtered = self.segment_filter(x)                         # per-segment EO/driver low-pass
        # MEAN over segments -> phase ~ V_pi/2 mapping is N-segment invariant
        drive_combined = filtered.sum(dim=1, keepdim=True) / self.num_segments

        phase = (np.pi / (2 * self.vpi)) * (drive_combined - self.bias_volt)
        field_real = 0.5 * (1 + self.gamma) * torch.cos(phase)
        field_imag = 0.5 * (1 - self.gamma) * torch.sin(phase)
        field_real = field_real * self.field_loss
        field_imag = field_imag * self.field_loss
        # chromatic dispersion in the optical band, on the complex envelope
        field_real, field_imag = self._apply_dispersion(field_real, field_imag)
        # ASE beat noise: added to the FIELD (before the square law) -> signal-dependent
        if ase_ebn0_db is not None:
            field_real, field_imag = self._add_ase(field_real, field_imag, ase_ebn0_db)

        photocurrent = field_real.pow(2) + field_imag.pow(2)      # square-law detection
        photocurrent = self.pd_filter(photocurrent)               # photodiode bandwidth
        return photocurrent.squeeze(0).squeeze(0)


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channel = OpticalChannel(cfg).to(device)

    # static transfer check: sweep a DC drive level, look at output intensity
    levels = torch.linspace(0, 2 * cfg.mzm_vpi_volt, 9, device=device)
    intensity = []
    for level in levels:
        drive = level * torch.ones(cfg.num_mzm_segments, 4096, device=device)
        out = channel(drive)
        intensity.append(out[2048].item())
    print("segments  :", cfg.num_mzm_segments,
          "bandwidth scales:", cfg.segment_bandwidth_scales[:cfg.num_mzm_segments])
    print("beta2*L   :", cfg.fiber_beta2_ps2_per_km * cfg.fiber_length_km, "ps^2")
    print("DC drive  :", np.round(levels.cpu().numpy(), 2))
    print("intensity :", np.round(np.array(intensity), 4))

    # gradient check: loss must backpropagate to the drive (with ASE on)
    drive = torch.zeros(cfg.num_mzm_segments, 8192, device=device, requires_grad=True)
    out = channel(drive, ase_ebn0_db=14.0)
    out.pow(2).mean().backward()
    print("grad to drive ok, mean|grad| =", float(drive.grad.abs().mean()))
    print("device:", device)

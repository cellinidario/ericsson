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

from scipy.special import erf

from pulse_shaping import root_raised_cosine


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


def supergaussian_fir(bw_3db_hz, order, sample_rate, num_taps):
    """Super-Gaussian low-pass FIR: H(f) = exp(-0.5*ln2*(f/bw)^(2*order)) (the realistic optical
    filter shape, as in trans_func.m). order=1 -> Gaussian; high order -> flat-top toward brick-wall.
    bw_3db_hz is the 3-dB (power) bandwidth. Built by frequency sampling + IFFT, windowed."""
    if num_taps % 2 == 0:
        num_taps = num_taps + 1
    n_fft = 8192
    freq = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)
    transfer = np.exp(-0.5 * np.log(2) * (np.abs(freq) / bw_3db_hz) ** (2 * order))
    impulse = np.fft.fftshift(np.fft.ifft(transfer).real)
    center = n_fft // 2
    half = num_taps // 2
    taps = impulse[center - half:center + half + 1] * np.hamming(num_taps)
    taps = taps / np.sum(taps)
    return taps.astype(np.float32)


def wss_fir(bandpass_hz, otf_bw_hz, sample_rate, num_taps):
    """Finisar WaveShaper (WSS) optical bandpass, as in trans_func.m 'WSS' (Pulikkaseril, Opt.Exp.
    2011): H(f) = 0.5*[erf((B/2 - f)/s) - erf((-B/2 - f)/s)], s = BWotf/(2*sqrt(ln2)). A flat-top
    passband of full width bandpass_hz with Gaussian-smoothed edges set by the OTF 3-dB otf_bw_hz.
    Built by frequency sampling + IFFT, windowed."""
    if num_taps % 2 == 0:
        num_taps = num_taps + 1
    n_fft = 8192
    freq = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)
    sigma = otf_bw_hz / (2 * np.sqrt(np.log(2)))
    half_bp = 0.5 * bandpass_hz
    transfer = 0.5 * (erf((half_bp - freq) / sigma) - erf((-half_bp - freq) / sigma))
    impulse = np.fft.fftshift(np.fft.ifft(transfer).real)
    center = n_fft // 2
    half = num_taps // 2
    taps = impulse[center - half:center + half + 1] * np.hamming(num_taps)
    taps = taps / np.sum(taps)
    return taps.astype(np.float32)


class OpticalChannel(nn.Module):
    def __init__(self, config, segment_filter_taps=65, pd_filter_taps=65, optical_filter_taps=65):
        super().__init__()
        self.num_segments = config.num_mzm_segments
        self.modulator = getattr(config, "modulator", "mzm")      # "mzm" | "linear" (ideal baseline)
        self.modulation_format = getattr(config, "modulation_format", "upam-4")   # "upam-4" | "bpam-4"
        self.vpi = config.mzm_vpi_volt
        self.bias_volt = config.mzm_vpi_volt                      # bias at Vpi: with drive in [0, Vpi] (upam-4)
                                                                  # arg spans [-pi/2, 0] -> field cos(arg) in [0, 1];
                                                                  # with drive in [-Vpi, Vpi] (bpam-4) arg spans
                                                                  # [-pi, 0] -> field in [-1, +1], null at drive 0
                                                                  # (the null-bias BPAM operation of Secondini2020)
        self.drive_min = config.drive_min_volt                   # for the linear-modulator power mapping
        self.drive_max = config.drive_max_volt
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

        # optical bandpass before the PD (only meaningful for a preamplified/ASE receiver). In
        # baseband (carrier at DC) it is a COMPLEX low-pass on the field -> the same real FIR on
        # each quadrature, before the square law, to suppress out-of-band ASE. Types: "wss" (Finisar
        # WaveShaper, realistic), "supergaussian", "matched" (RRC, A.47 optimum), "brickwall", None.
        optical_bw = getattr(config, "optical_filter_bandwidth", None)
        optical_type = getattr(config, "optical_filter_type", "auto")
        if optical_type == "auto":               # ase: realistic WSS (preamp RX); thermal: none
            optical_type = "wss" if config.noise_regime == "ase" else None
        if optical_type is None or optical_bw is None or optical_bw >= config.sim_sample_rate:
            self.optical_filter = None
            self.optical_filter_type = None
        else:
            if optical_type == "wss":
                bandpass = getattr(config, "wss_bandpass_factor", 1.6) * config.symbol_rate
                otf = getattr(config, "wss_otf_bandwidth", 18e9)
                optical_taps = wss_fir(bandpass, otf, config.sim_sample_rate, optical_filter_taps)
            elif optical_type == "matched":
                rrc = root_raised_cosine(config.rrc_rolloff, config.rrc_span_symbols, config.samples_per_symbol_sim)
                optical_taps = (rrc / rrc.sum()).astype(np.float32)
            elif optical_type == "supergaussian":
                order = getattr(config, "optical_filter_order", 1)
                optical_taps = supergaussian_fir(optical_bw / 2, order, config.sim_sample_rate, optical_filter_taps)
            else:
                optical_taps = lowpass_fir(optical_bw / 2, config.sim_sample_rate, optical_filter_taps)
            optical_weight = torch.tensor(optical_taps[::-1].copy()).view(1, 1, -1)
            self.optical_filter = nn.Conv1d(1, 1, kernel_size=len(optical_taps),
                                            padding=len(optical_taps) // 2, bias=False)
            self.optical_filter.weight = nn.Parameter(optical_weight)
            self.optical_filter_type = optical_type

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
        # sum over segments -> total arm phase (each segment adds phase; Marco's sum(v,2))
        drive_combined = filtered.sum(dim=1, keepdim=True)

        # modulator: drive -> optical field.
        #   "mzm"    : E = 1/2 ( exp(j*phi) + gamma*exp(-j*phi) ),  phi = pi/(2*Vpi)*(V - Vbias),
        #              gamma = (sqrt(ER)-1)/(sqrt(ER)+1) sets the finite extinction ratio.
        #   "linear" : ideal linear-INTENSITY modulator (optical power proportional to drive, chirp-free
        #              field = sqrt(power)) -- removes ONLY the MZM nonlinearity.
        # Everything after -- dispersion, ASE, optical filter, square-law photodiode -- is KEPT for both.
        if self.modulator == "linear":
            if self.modulation_format == "bpam-4":
                # ideal linear-FIELD modulator (Secondini2020, Fig. 2): bipolar field proportional to drive
                field_real = (drive_combined / self.drive_max) * self.field_loss
            else:
                # ideal linear-INTENSITY modulator: optical power proportional to drive, chirp-free field
                power = (drive_combined - self.drive_min) / (self.drive_max - self.drive_min)
                field_real = torch.sqrt(F.relu(power) + 1e-6) * self.field_loss   # eps tames the sqrt slope at P~0
            field_imag = torch.zeros_like(field_real)
        else:
            arg = (np.pi / (2 * self.vpi)) * (drive_combined - self.bias_volt)
            field = 0.5 * (torch.exp(1j * arg) + self.gamma * torch.exp(-1j * arg))
            field_real = field.real * self.field_loss
            field_imag = field.imag * self.field_loss
        # chromatic dispersion in the optical band, on the complex envelope
        field_real, field_imag = self._apply_dispersion(field_real, field_imag)
        # ASE beat noise: added to the FIELD (before the square law) -> signal-dependent
        if ase_ebn0_db is not None:
            field_real, field_imag = self._add_ase(field_real, field_imag, ase_ebn0_db)
        # optical bandpass (baseband complex low-pass on the field), before the square law
        if self.optical_filter is not None:
            field_real = self.optical_filter(field_real)
            field_imag = self.optical_filter(field_imag)

        photocurrent = field_real.pow(2) + field_imag.pow(2)      # square-law detection
        photocurrent = self.pd_filter(photocurrent)               # photodiode bandwidth
        return photocurrent.squeeze(0).squeeze(0)

    def complex_envelope(self, drive, ase_ebn0_db=None):
        """Noisy complex envelope x(t) + n(t) of the optical signal (modulator + dispersion + ASE),
        BEFORE the optical filter and the photodiode -- Forestieri's x(t). The A.47 receiver applies
        the optical matched filter ho(t) = p*(T-t) to it; the square-law photodiode then forms the
        photocurrent |.|^2 (direct/incoherent detection, Forestieri Fig. A.4). The same BER follows
        from the equivalent envelope-detector view (Fig. A.5: decide on z = sqrt(|.|^2), thresholds in
        amplitude). Returns (real, imag) of the field, each (num_samples,)."""
        x = drive.unsqueeze(0)
        drive_combined = self.segment_filter(x).sum(dim=1, keepdim=True)
        if self.modulator == "linear":
            if self.modulation_format == "bpam-4":
                field_real = (drive_combined / self.drive_max) * self.field_loss   # linear-FIELD (bipolar)
            else:
                power = (drive_combined - self.drive_min) / (self.drive_max - self.drive_min)
                field_real = torch.sqrt(F.relu(power) + 1e-6) * self.field_loss
            field_imag = torch.zeros_like(field_real)
        else:
            arg = (np.pi / (2 * self.vpi)) * (drive_combined - self.bias_volt)
            field = 0.5 * (torch.exp(1j * arg) + self.gamma * torch.exp(-1j * arg))
            field_real = field.real * self.field_loss
            field_imag = field.imag * self.field_loss
        field_real, field_imag = self._apply_dispersion(field_real, field_imag)
        if ase_ebn0_db is not None:
            field_real, field_imag = self._add_ase(field_real, field_imag, ase_ebn0_db)
        return field_real.squeeze(0).squeeze(0), field_imag.squeeze(0).squeeze(0)


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

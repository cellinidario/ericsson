"""System and training parameters for the end-to-end IM/DD PAM-4 autoencoder.

All physical quantities are SI (Hz, volt, metre) unless the name says otherwise.
One source of truth: every other module builds a Config and reads from it.
"""

import numpy as np


class Config:
    # standard single-mode-fibre parameters per band (textbook SMF; refine with Marco's numbers)
    WAVELENGTH_BANDS = {
        "back-to-back": {"wavelength_nm": 1310.0, "fiber_beta2_ps2_per_km": 0.0,   "fiber_loss_db_per_km": 0.0},
        "oband":        {"wavelength_nm": 1310.0, "fiber_beta2_ps2_per_km": -1.0,  "fiber_loss_db_per_km": 0.32},
        "cband":        {"wavelength_nm": 1550.0, "fiber_beta2_ps2_per_km": -21.7, "fiber_loss_db_per_km": 0.20},
    }
    # "back-to-back" = transparent fibre (no dispersion, no loss; output = input)

    def __init__(self):
        # ===== scenario knobs (flip these per experiment; the notebook sets the same names) =====
        self.wavelength_band = "oband"          # "oband" 1310 nm (near zero-dispersion) | "cband" 1550 nm (~20x)
        self.noise_regime = "ase"               # "ase": optical ASE beat noise on the field (theory A.47)
                                                # "thermal": electrical AWGN at the ADC (theory A.19); other off
        self.modulator = "mzm"                  # "mzm": nonlinear cos^2 intensity transfer
                                                # "linear": ideal linear-INTENSITY modulator (optical power
                                                # proportional to drive). Removes ONLY the MZM nonlinearity;
                                                # dispersion, ASE, optical filter and the square-law
                                                # photodiode are KEPT -> still IM/DD, a clean baseline.
        self.tx_filter = "rrc"                  # pulse/matched filter, same at TX and RX (H_TX = H_RX):
        self.rx_filter = "rrc"                  #   "rrc"       : root-raised-cosine, roll-off 0.85 (band-limited)
                                                #   "freq-rect" : ideal-rect 20 GHz brickwall (Marco's JLT model;
                                                #                 a rectangle in FREQUENCY -> sinc tails in time)
                                                #   "time-rect" : rectangle in TIME (one symbol; sample-and-hold at
                                                #                 TX, integrate-and-dump at RX). Confined to the
                                                #                 symbol -> no overlap -> reaches A.19 with a wide
                                                #                 front-end. ("rect" is kept as an alias of freq-rect.)
        self.equalizer = "end-to-end"           # detection/equalization stage:
                                                #   "end-to-end" : DPD (TX) + FFE (RX) trained together (the E2E)
                                                #   "ffe"        : FFE only (RX); TX uses fixed equispaced-intensity levels
                                                #   "threshold"  : threshold detector only (no DPD, no FFE) -> the A.19
                                                #                  reference (fixed optimum levels + optimum thresholds)

        # ===== link / modulation =====
        self.symbol_rate = 20e9                 # baud (OFC paper: 20 GBaud)
        self.bits_per_symbol = 2                # PAM-4 -> 2 bit/symbol
        self.modulation_order = 4               # M = 2**bits_per_symbol
        self.pam_format = "unipolar"            # "unipolar" | "bipolar" (BPAM is a future item)

        # ===== oversampling =====
        self.samples_per_symbol_sim = 4         # analog channel rate; 4 is enough if MZM BW <= 30 GHz
                                                # (tested: square-law aliasing < 0.1% of swing at K=4)
        self.samples_per_symbol_rx = 1          # ADC / FFE rate. Unipolar PAM is a sufficient statistic at
                                                # symbol rate (matched filter -> no ISI at 1 sps); 2 sps is
                                                # only needed for BPAM. (Was 2 in the OFC config.)

        # ===== pulse shaping (fixed at TX, matched at RX -> Fork A, no time-mux) =====
        self.rrc_rolloff = 0.85                 # RRC roll-off (tx_filter / rx_filter == "rrc")
        self.rrc_span_symbols = 16              # filter truncation length in symbols (both rrc and rect)
        self.rect_filter_bandwidth = 20e9       # rect cutoff (tx_filter / rx_filter == "rect"): Marco's 20 GHz,
                                                # the DAC/ADC Nyquist (H_TX = H_RX = rect, no RRC)

        # ===== MZM (modulator == "mzm"): E = 1/2 (e^{j*phi} + gamma*e^{-j*phi}) =====
        self.num_mzm_segments = 1               # single segment (Marco's JLT model); >1 -> segmented MZM (phases sum)
        self.mzm_vpi_volt = 5.0                 # half-wave voltage V_pi
        self.mzm_extinction_ratio_db = 33.0     # finite extinction ratio -> gamma = (sqrt(ER)-1)/(sqrt(ER)+1)
        self.mzm_bandwidth = 30e9               # per-segment EO bandwidth (30 GHz >> Nyquist -> low ISI)
        self.segment_bandwidth_scales = (1.0,)  # per-segment BW = scale * mzm_bandwidth, one entry per segment;
                                                # a longer tuple (e.g. 0.85,0.90,0.95,1.00) models driver mismatch

        # ===== optical fibre (wavelength / dispersion / loss come from the band) =====
        self.fiber_length_km = 10.238           # spool length
        self._apply_wavelength_band()           # wavelength_band -> wavelength_nm, fiber_beta2_ps2_per_km,
                                                # fiber_loss_db_per_km (per WAVELENGTH_BANDS)

        # ===== optical bandpass before the photodiode (preamplified / ASE receiver) =====
        # Full optical width B_o around the carrier; in baseband a complex low-pass of cutoff B_o/2 on the
        # field, before the square law, to suppress out-of-band ASE. 37 GHz ~= R_s*(1+rolloff).
        self.optical_filter_bandwidth = 37e9
        self.optical_filter_type = "auto"       # AUTO: "wss" for ase (the realistic Finisar WaveShaper the
                                                # group uses), "none" for thermal (no preamp -> no filter).
                                                # Force: "wss" / "supergaussian" / "matched" / "brickwall" / "none".
        self.wss_bandpass_factor = 1.53         # WSS passband full width B = factor * symbol_rate (Marco's value)
        self.wss_otf_bandwidth = 18e9           # WSS OTF 3-dB bandwidth (edge sharpness)
        self.optical_filter_order = 4           # super-Gaussian order ("supergaussian" type only)

        # ===== photodiode =====
        self.photodiode_bandwidth = 25e9        # post-detection low-pass bandwidth

        # ===== DPD (transmitter DSP), Fork A: symbol-rate predistortion with memory =====
        self.dpd_memory_symbols = 5             # context window (LUT-feasible: 2^(2*5) = 1024 rows)
        self.dpd_hidden_width = 8               # leaky-ReLU hidden width
        self.drive_min_volt = 0.0               # DPD output range, lower bound
        self.drive_max_volt = 5.0               # upper bound = Vpi -> monotonic half of the MZM transfer
                                                # (init at quadrature: max slope, good gradient). This bound
                                                # is the only constraint: the 4 levels are LEARNED by the BCE,
                                                # not imposed (no equispacing/level-target term).

        # ===== FFE (receiver DSP) =====
        self.ffe_memory_symbols = 11            # context window at the decision (odd -> centered); the channel
                                                # memory is ~1 symbol, so 11 taps are already generous
        self.ffe_hidden_width = 8               # nonlinear capacity. Small effect in O-band/ASE, but a STRONG
                                                # lever in C-band (the nonlinear CD distortion: ~3.7x at 20 dB)

        # ===== training =====
        self.edge_guard_symbols = 64            # symbols dropped at both ends (filter transients)
        self.minibatch_symbols = 8192           # sliding-window minibatch length, in symbols
        self.learning_rate = 1e-3

        self._compute_derived()

    def _apply_wavelength_band(self):
        """Set wavelength, dispersion and loss from the selected band (WAVELENGTH_BANDS)."""
        params = self.WAVELENGTH_BANDS[self.wavelength_band]
        self.wavelength_nm = params["wavelength_nm"]
        self.fiber_beta2_ps2_per_km = params["fiber_beta2_ps2_per_km"]
        self.fiber_loss_db_per_km = params["fiber_loss_db_per_km"]

    def set_wavelength_band(self, band):
        """Select 'oband' (1310 nm, near zero-dispersion) or 'cband' (1550 nm, high dispersion)."""
        if band not in self.WAVELENGTH_BANDS:
            raise ValueError(f"wavelength_band must be one of {list(self.WAVELENGTH_BANDS)}, got {band!r}")
        self.wavelength_band = band
        self._apply_wavelength_band()
        self._compute_derived()

    def _compute_derived(self):
        self.bit_rate = self.symbol_rate * self.bits_per_symbol
        self.sim_sample_rate = self.symbol_rate * self.samples_per_symbol_sim
        self.rx_sample_rate = self.symbol_rate * self.samples_per_symbol_rx
        self.sim_nyquist = self.sim_sample_rate / 2
        self.extinction_ratio_linear = 10 ** (self.mzm_extinction_ratio_db / 10)
        self.fiber_loss_linear = 10 ** (-self.fiber_loss_db_per_km * self.fiber_length_km / 10)

    def summary(self):
        lines = []
        lines.append(f"symbol rate          : {self.symbol_rate/1e9:.0f} GBaud")
        lines.append(f"bit rate             : {self.bit_rate/1e9:.0f} Gb/s  (PAM-{self.modulation_order})")
        lines.append(f"sim sample rate      : {self.sim_sample_rate/1e9:.0f} GHz  ({self.samples_per_symbol_sim} sps, Nyquist {self.sim_nyquist/1e9:.0f} GHz)")
        lines.append(f"rx sample rate       : {self.rx_sample_rate/1e9:.0f} GHz  ({self.samples_per_symbol_rx} sps)")
        lines.append(f"photodiode bandwidth : {self.photodiode_bandwidth/1e9:.0f} GHz  (representable: {self.photodiode_bandwidth < self.sim_nyquist})")
        lines.append(f"modulator            : {self.modulator}" +
                     (f"  (MZM {self.mzm_bandwidth/1e9:.0f} GHz, {self.num_mzm_segments} segments)" if self.modulator == "mzm" else ""))
        pulse_names = {"rrc": f"RRC roll-off {self.rrc_rolloff}",
                       "freq-rect": f"freq-rect {self.rect_filter_bandwidth/1e9:.0f} GHz (brickwall)",
                       "rect": f"freq-rect {self.rect_filter_bandwidth/1e9:.0f} GHz (brickwall)",
                       "time-rect": "time-rect (one-symbol rectangle, integrate-and-dump)"}
        if self.tx_filter == self.rx_filter:
            lines.append(f"pulse shape          : {pulse_names.get(self.tx_filter, self.tx_filter)}")
        else:
            lines.append(f"pulse shape          : TX {pulse_names.get(self.tx_filter, self.tx_filter)} / "
                         f"RX {pulse_names.get(self.rx_filter, self.rx_filter)}")
        eq_names = {"end-to-end": "DPD (TX) + FFE (RX), trained jointly", "ffe": "FFE only (fixed TX levels)",
                    "threshold": "threshold detector (no DPD/FFE -> A.19 reference)"}
        lines.append(f"equalizer            : {eq_names.get(self.equalizer, self.equalizer)}")
        lines.append(f"wavelength band      : {self.wavelength_band}  ({self.wavelength_nm:.0f} nm, "
                     f"beta2 {self.fiber_beta2_ps2_per_km:.1f} ps^2/km, L {self.fiber_length_km:.3f} km, "
                     f"beta2*L {self.fiber_beta2_ps2_per_km * self.fiber_length_km:.1f} ps^2)")
        lines.append(f"noise regime         : {self.noise_regime}")
        optical_type = self.optical_filter_type
        if optical_type == "auto":
            optical_type = "wss" if self.noise_regime == "ase" else "none"
        if optical_type == "none" or not self.optical_filter_bandwidth:
            lines.append("optical filter       : none")
        else:
            lines.append(f"optical filter       : {optical_type}, Bo {self.optical_filter_bandwidth/1e9:.0f} GHz")
        return "\n".join(lines)


if __name__ == "__main__":
    config = Config()
    print(config.summary())

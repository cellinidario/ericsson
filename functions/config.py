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
        self.modulation_format = "upam-4"       # "upam-4": unipolar PAM-4 (conventional IM/DD)
                                                # "bpam-4": bipolar PAM-4 with direct detection (Secondini2020): 
                                                # field levels {+-A, +-2A}, 1 bit on the
                                                # 2 amplitudes + 1 bit DIFFERENTIALLY encoded on the sign
                                                # (delta-phi in {0, pi}); the receiver needs 2 samples/symbol
                                                # (odd samples sense the sign through pulse interference).
                                                # Set via set_modulation_format() -> derives drive range + rx sps.

        # ===== oversampling =====
        self.samples_per_symbol_sim = 4         # analog channel rate; 4 is enough if MZM BW <= 30 GHz
                                                # (tested: square-law aliasing < 0.1% of swing at K=4)
        self.samples_per_symbol_rx = 1          # ADC / FFE rate. Unipolar PAM is a sufficient statistic at
                                                # symbol rate (matched filter -> no ISI at 1 sps); 2 sps is
                                                # only needed for BPAM. (Was 2 in the OFC config.)
        self.bpam_curriculum = False            # bpam-4 E2E: staged curriculum (NO imposed levels). (1) joint
                                                # train at low SNR -> the TX self-organizes bipolar; (2) freeze
                                                # TX, train RX to convergence -> breaks the co-adaptation
                                                # deadlock; (3) free joint fine-tune. Needs bpam_precode_e2e.
        self.curr_lowsnr_ebn0 = 10.0            # stage-1 fixed training Eb/N0 [dB]
        self.curr_lowsnr_steps = 20000          # stage-1 joint-at-low-SNR steps
        self.curr_rxonly_steps = 20000          # stage-2 frozen-TX RX-only steps
        self.bpam_warm_start = False            # bpam-4 E2E: warm-start the TX on the classical BPAM
                                                # constellation (distill) + prime the RX, then free joint
                                                # fine-tune. Breaks the TX<->RX co-adaptation deadlock so the
                                                # E2E reaches the below-theory BPAM basin. Needs
                                                # bpam_precode_e2e=True. Basin selection, not level imposition.
        self.warm_distill_steps = 8000          # phase-0 TX distillation steps
        self.warm_rxonly_steps = 15000          # phase-1 RX-only priming steps
        self.tx_init_gain = 1.0                 # DPD weight-init multiplier. >1 spreads the initial per-symbol
                                                # drives across the full (bipolar) range instead of starting at
                                                # the null -> lets the E2E explore bipolar signalling from step
                                                # 0 (escapes the quasi-unipolar local min). Imposes no levels.
        self.bpam_loss = "ce+bce"               # bpam-4 E2E training loss: "ce+bce" (symbol-CE keeps 4
                                                # distinct intensities -> double-rate/hybrid route) or "bce"
                                                # (per-bit only; ALLOWS sign-degenerate intensities |+-A|^2
                                                # -> true BPAM, sign recovered by T/2 interference).
        self.bpam_precode_e2e = False           # bpam-4 E2E: apply the differential sign precoder in the
                                                # end-to-end path too (True). Needed for the classical-BPAM
                                                # route in a band-limited channel (freq-rect), where the sign
                                                # is only DIFFERENTIALLY observable and double-rate OOK does
                                                # not fit the 20 GHz TX filter. False (default) = raw bits =
                                                # the double-rate route (works when the band allows 2 sps).
        self.tx_waveform_freedom = True         # bpam-4 E2E + time-rect: DPD emits sps_rx values per symbol
                                                # (sub-symbol waveform freedom) so each bit can ride an
                                                # independent T/2 slot -> double-rate OOK/hybrid that goes
                                                # BELOW the symbol-rate PAM theory using the extra RX
                                                # bandwidth. False -> classical 1 level/symbol BPAM (needs
                                                # the differential precoder; sits ~on the theory).

        # ===== pulse shaping (fixed at TX, matched at RX -> Fork A, no time-mux) =====
        self.rrc_rolloff = 0.85                 # RRC roll-off (tx_filter / rx_filter == "rrc")
        self.rrc_span_symbols = 16              # filter truncation length in symbols (both rrc and rect)
        self.rect_filter_bandwidth = 20e9       # rect cutoff (tx_filter / rx_filter == "rect"): Marco's 20 GHz,
                                                # the DAC/ADC Nyquist (H_TX = H_RX = rect, no RRC)

        # ===== MZM (modulator == "mzm"): E = 1/2 (e^{j*phi} + gamma*e^{-j*phi}) =====
        self.num_mzm_segments = 1               # single segment (Marco's JLT model); >1 -> segmented MZM (phases sum)
        self.mzm_vpi_volt = 5.0                 # half-wave voltage V_pi
        self.mzm_extinction_ratio_db = 25.0     # finite extinction ratio -> gamma = (sqrt(ER)-1)/(sqrt(ER)+1)
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
                                                # group uses), None for thermal (no preamp -> no filter).
                                                # Force: "wss" / "supergaussian" / "matched" / "brickwall" / None.
        self.wss_bandpass_factor = 1.53         # WSS passband full width B = factor * symbol_rate (Marco's value)
        self.wss_otf_bandwidth = 18e9           # WSS OTF 3-dB bandwidth (edge sharpness)
        self.optical_filter_order = 4           # super-Gaussian order ("supergaussian" type only)

        # ===== photodiode =====
        self.photodiode_bandwidth = 25e9        # post-detection low-pass bandwidth

        # ===== JLT reference chain: RX digital filter + converter quantization =====
        self.rx_gaussian_bw = None              # digital Gaussian LPF on the photocurrent before decimation
                                                # (JLT agreed setup: 10e9). None -> off.
        self.dac_bits = None                    # DAC resolution N_DAC (JLT: 6); quantizes the TX drive
                                                # waveform over [drive_min, drive_max]. None -> ideal DAC.
                                                # Trained through with a straight-through estimator (STE).
        self.adc_bits = None                    # ADC resolution N_ADC; quantizes the decimated RX samples.
                                                # None -> ideal ADC. STE in training.

        # ===== DPD (transmitter DSP), Fork A: symbol-rate predistortion with memory =====
        self.dpd_memory_symbols = 5             # context window (LUT-feasible: 2^(2*5) = 1024 rows)
        self.dpd_hidden_width = 8               # leaky-ReLU hidden width
        self.dpd_hidden_layers = 1              # hidden DEPTH of the DPD (memory stays dpd_memory_symbols;
                                                # depth adds pre-distortion capacity, e.g. C-band CD pre-comp)
        self.tx_values_per_symbol = None        # E2E waveform granularity: None -> samples_per_symbol_rx;
                                                # 4 -> sim-rate values (finer spectral control for CD pre-comp)
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
        self.ffe_hidden_layers = 1              # hidden DEPTH of the FFE (1 = classic two-FC receiver). Depth
                                                # adds nonlinear inversion capacity (C-band CD x square-law).
        self.ffe_nonlinear = True               # True: leaky-ReLU FFE (nonlinear). False: a purely LINEAR
                                                # FFE (no activation) -> the classical linear equalizer baseline

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

    def set_modulation_format(self, fmt):
        """Select 'upam-4' (unipolar, conventional IM/DD) or 'bpam-4' (bipolar with DD).
        Derives the format-dependent settings:
          upam-4: drive in [0, Vpi]   -> the monotonic UNIPOLAR half of the MZM field transfer; 1 rx sps suffices.
          bpam-4: drive in [-Vpi, Vpi] -> with the null bias the field cos(arg) sweeps [-1, +1] monotonically
                  (arg in [-pi, 0], zero field at drive 0); the receiver MUST sample at 2 sps (the odd,
                  half-symbol samples carry the sign information via inter-pulse interference)."""
        if fmt not in ("upam-4", "bpam-4"):
            raise ValueError(f"modulation_format must be 'upam-4' or 'bpam-4', got {fmt!r}")
        self.modulation_format = fmt
        if fmt == "bpam-4":
            self.drive_min_volt = -self.mzm_vpi_volt
            self.drive_max_volt = +self.mzm_vpi_volt
            self.samples_per_symbol_rx = 2
        else:
            self.drive_min_volt = 0.0
            self.drive_max_volt = self.mzm_vpi_volt
            self.samples_per_symbol_rx = 1
        self._compute_derived()

    def _compute_derived(self):
        # normalize + validate the equalizer knob ("autoencoder" is an alias for "end-to-end");
        # an unknown string must FAIL LOUDLY, not silently degrade to fixed-levels/RX-only training
        self.equalizer = {"autoencoder": "end-to-end"}.get(self.equalizer, self.equalizer)
        if self.equalizer not in ("threshold", "ffe", "end-to-end"):
            raise ValueError(f"equalizer must be 'threshold', 'ffe', 'end-to-end' (alias 'autoencoder'), "
                             f"got {self.equalizer!r}")
        self.bit_rate = self.symbol_rate * self.bits_per_symbol
        self.sim_sample_rate = self.symbol_rate * self.samples_per_symbol_sim
        self.rx_sample_rate = self.symbol_rate * self.samples_per_symbol_rx
        self.sim_nyquist = self.sim_sample_rate / 2
        self.extinction_ratio_linear = 10 ** (self.mzm_extinction_ratio_db / 10)
        self.fiber_loss_linear = 10 ** (-self.fiber_loss_db_per_km * self.fiber_length_km / 10)

    def summary(self):
        lines = []
        lines.append(f"symbol rate          : {self.symbol_rate/1e9:.0f} GBaud")
        fmt_names = {"upam-4": "unipolar PAM-4", "bpam-4": "bipolar PAM-4/DD (differential sign, 2 rx sps)"}
        lines.append(f"bit rate             : {self.bit_rate/1e9:.0f} Gb/s  "
                     f"({fmt_names.get(self.modulation_format, self.modulation_format)})")
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
            optical_type = "wss" if self.noise_regime == "ase" else None
        if optical_type is None or not self.optical_filter_bandwidth:
            lines.append("optical filter       : none")
        else:
            lines.append(f"optical filter       : {optical_type}, Bo {self.optical_filter_bandwidth/1e9:.0f} GHz")
        return "\n".join(lines)


if __name__ == "__main__":
    config = Config()
    print(config.summary())

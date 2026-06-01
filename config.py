"""System and training parameters for the end-to-end IM/DD PAM-4 autoencoder.

All physical quantities are SI (Hz, volt, metre) unless the name says otherwise.
One source of truth: every other module builds a Config and reads from it.
"""

import numpy as np


class Config:
    def __init__(self):
        # ----- link / modulation -----
        self.symbol_rate = 20e9                 # baud (OFC paper: 20 GBaud)
        self.bits_per_symbol = 2                # PAM-4 -> 2 bit/symbol
        self.modulation_order = 4               # M = 2**bits_per_symbol
        self.pam_format = "unipolar"            # "unipolar" or "bipolar" (start unipolar)

        # ----- oversampling -----
        self.samples_per_symbol_sim = 4         # analog channel rate; 4 is enough if MZM BW <= 30 GHz
                                                # (tested: square-law aliasing < 0.1% of swing at K=4)
        self.samples_per_symbol_rx = 2          # ADC / FFE rate after decimation (OFC: 2 sps)

        # ----- pulse shaping (fixed RRC at TX, matched at RX -> Fork A, no time-mux) -----
        self.rrc_rolloff = 0.85                 # OFC paper roll-off factor
        self.rrc_span_symbols = 16              # RRC truncation length, in symbols

        # ----- segmented MZM -----
        self.num_mzm_segments = 4               # 4 driver segments, each with its own EO bandwidth
        self.mzm_vpi_volt = 5.0                 # half-wave voltage
        self.mzm_extinction_ratio_db = 33.0     # finite extinction ratio
        self.mzm_bandwidth = 30e9               # NOMINAL per-segment bandwidth (driver-limited).
                                                # actual segments are spread around this (see channel.py),
                                                # 30 GHz >> signal Nyquist -> minimal ISI, and lets K_sim=4.
        self.segment_bandwidth_scales = (0.85, 0.90, 0.95, 1.00)  # per-segment BW = scale * mzm_bandwidth

        # ----- optical channel (1310 nm: small residual dispersion) -----
        self.wavelength_nm = 1310.0
        self.fiber_length_km = 10.0
        self.fiber_loss_db_per_km = 0.32        # typical SMF at O-band
        self.fiber_beta2_ps2_per_km = -1.0      # residual chromatic dispersion at 1310 nm

        # ----- optical bandpass before the photodiode (preamplified receiver) -----
        # Full optical width B_o around the carrier; in baseband it is a complex low-pass of
        # cutoff B_o/2 on the field (suppresses out-of-band ASE before the square law).
        # 37 GHz ~= R_s*(1+rolloff), matched to the signal optical band. None disables it.
        self.optical_filter_bandwidth = 37e9

        # ----- photodiode -----
        self.photodiode_bandwidth = 25e9        # post-detection low-pass bandwidth
        self.adc_resolution_bits = 8            # ideal 8-bit quantizer (RX)

        # ----- DPD (transmitter DSP), Fork A: symbol-rate predistortion with memory -----
        self.dpd_memory_symbols = 5             # context window (temporal memory of predistortion)
        self.dpd_hidden_width = 8               # leaky-ReLU hidden layer width
        self.drive_min_volt = 0.0               # DPD output range: lower bound
        self.drive_max_volt = 5.0               # upper bound = Vpi -> monotonic half of MZM transfer,
                                                # init (tanh=0) lands at quadrature (max slope, good gradient)
        self.equispacing_weight = 2.0           # weight of the level-MSE term: pins 4 clean, Gray-ordered
                                                # levels. The TARGET spacing depends on the noise regime
                                                # (see equispacing_loss): equispaced INTENSITY for thermal,
                                                # equispaced AMPLITUDE for ASE (Forestieri A.2, Fig. A.6).

        # ----- FFE (receiver DSP) -----
        self.ffe_memory_symbols = 11            # context window at the decision (odd -> centered)
        self.ffe_hidden_width = 8               # 16 was tested in the ASE regime: only ~0.5 dB at
                                                # 20 dB, so the floor is NOT FFE capacity (it is the
                                                # intensity-domain receiver vs the A.47 envelope bound)
        self.rx_sqrt_companding = None          # DSP sqrt after the ADC (photocurrent -> envelope):
                                                # decide on amplitude. None = AUTO: ON for "ase" (beat
                                                # noise is additive in amplitude) and OFF for "thermal"
                                                # (noise additive in intensity -> sqrt hurts). True/False forces.

        # ----- training -----
        self.num_symbols_train = 200_000        # long sequence (no tiny blocks; edges discarded)
        self.num_symbols_eval = 500_000         # longer sequence for low-BER statistics
        self.edge_guard_symbols = 64            # symbols dropped at both ends (filter transients)
        self.surrogate_epochs = 200             # digital-surrogate regression epochs
        self.e2e_epochs = 400                   # joint DPD+FFE epochs
        self.minibatch_symbols = 8192           # sliding-window minibatch length, in symbols
        self.learning_rate = 1e-3

        # ----- noise regime -----
        # "ase"     : amplified-spontaneous-emission (optical) beat noise, added to the COMPLEX
        #             field before the photodiode -> signal-dependent after the square law.
        #             Reference: Forestieri eq.(A.47). Thermal/electrical noise OFF.
        # "thermal" : white Gaussian (thermal/electronics) noise on the photocurrent at the ADC,
        #             i.e. electrical noise (optical noise OFF). Reference: Forestieri eq.(A.19).
        self.noise_regime = "ase"

        # ----- noise sweep for the BER vs Eb/N0 curve -----
        self.ebn0_db_sweep = np.arange(6, 26, 2.0)

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
        lines.append(f"MZM bandwidth        : {self.mzm_bandwidth/1e9:.0f} GHz  (representable: {self.mzm_bandwidth < self.sim_nyquist})")
        lines.append(f"photodiode bandwidth : {self.photodiode_bandwidth/1e9:.0f} GHz  (representable: {self.photodiode_bandwidth < self.sim_nyquist})")
        lines.append(f"MZM segments         : {self.num_mzm_segments}")
        lines.append(f"pulse shape          : RRC roll-off {self.rrc_rolloff}")
        lines.append(f"noise regime         : {self.noise_regime}")
        lines.append(f"optical filter Bo    : {self.optical_filter_bandwidth/1e9:.0f} GHz" if self.optical_filter_bandwidth else "optical filter Bo    : none")
        companding = self.rx_sqrt_companding if self.rx_sqrt_companding is not None else (self.noise_regime == "ase")
        lines.append(f"rx sqrt companding   : {companding}")
        return "\n".join(lines)


if __name__ == "__main__":
    config = Config()
    print(config.summary())

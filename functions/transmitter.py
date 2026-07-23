"""Transmitter: PAM-4 mapping + digital predistortion (DPD).

Fork A: the DPD works at SYMBOL RATE with a temporal context window, outputs one
predistorted drive level per MZM segment, and a FIXED RRC does the oversampling.
This keeps the time-multiplexing degree of freedom out of the network: one level
per symbol, shaped by a Nyquist pulse.

The DPD is implemented as two fully-connected layers applied to a sliding window
of `dpd_memory_symbols` consecutive symbols (literal "FC" in the paper sense).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pulse_shaping import root_raised_cosine, brickwall
from utils import quantize_ste


def bits_to_symbols(bits, bits_per_symbol):
    """Map bit rows (bits_per_symbol, num_symbols) to integer symbols 0..M-1 (Gray-free, MSB first)."""
    weights = 2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=bits.device).view(-1, 1)
    return (bits * weights).sum(dim=0)


def coded_bpam_target(bits):
    """The bits the RECEIVER actually observes when the differential precoder is on: row 0 is the
    amplitude bit (unchanged), row 1 is the transmitted sign STATE (cumsum%2). Training against this
    is well posed (each symbol carries its own sign state); the raw phase bit is recovered afterwards
    by differential_decode. (For row 0, coded == raw.)"""
    sign_state = torch.cumsum(bits[1], dim=0) % 2
    return torch.stack([bits[0], sign_state])


def differential_decode(sign_state_bits):
    """Invert the precoder on the SIGN row: raw_phase[k] = sign_state[k] XOR sign_state[k-1].
    Input/return shape (num_bits, num_symbols); row 0 (amplitude) passes through unchanged."""
    out = sign_state_bits.clone()
    row = sign_state_bits[1].to(torch.int64)
    out[1] = (row ^ torch.roll(row, 1, dims=0)).to(sign_state_bits.dtype)
    out[1, 0] = row[0]                                  # first symbol: no predecessor (state = raw)
    return out


class Transmitter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_bits = config.bits_per_symbol
        self.modulation_order = config.modulation_order
        self.num_segments = config.num_mzm_segments
        self.samples_per_symbol = config.samples_per_symbol_sim
        self.drive_min = config.drive_min_volt
        self.drive_max = config.drive_max_volt
        self.equalizer = getattr(config, "equalizer", "end-to-end")   # "end-to-end" -> DPD; else fixed levels
        self.pulse_kind = getattr(config, "tx_filter", "rrc")
        self.noise_regime = getattr(config, "noise_regime", "thermal")
        self.modulation_format = getattr(config, "modulation_format", "upam-4")   # "upam-4" | "bpam-4"
        self.modulator_kind = getattr(config, "modulator", "mzm")                 # needed for bpam fixed levels
        self.precode_e2e = getattr(config, "bpam_precode_e2e", False)              # differential precoder in E2E too
        self.dac_bits = getattr(config, "dac_bits", None)                          # DAC resolution (None = ideal)
        self.output_activation = getattr(config, "dpd_output_activation", "tanh")  # "tanh" | "hardtanh"
        tx_gauss = getattr(config, "tx_gaussian_bw", None)                         # Gaussian cascade after pulse filter
        if tx_gauss:
            from channel import supergaussian_fir
            g = supergaussian_fir(tx_gauss, 1, config.sim_sample_rate, 65)
            self.register_buffer("tx_gauss", torch.tensor(g[::-1].copy(), dtype=torch.float32).view(1, 1, -1))
        else:
            self.tx_gauss = None

        # Sub-symbol waveform freedom: in end-to-end the DPD emits `samples_per_symbol_rx` drive values
        # PER SYMBOL (T/2 spacing), instead of a single symbol-rate level. This is the degree of freedom
        # the RX 2-sps sampling needs: two independent half-symbol slots let the autoencoder express
        # time-multiplexed signalling (e.g. double-rate OOK, one bit/slot) rather than being confined to
        # symbol-rate PAM. time-rect: each value is held over its T/2 slot. Band-limited pulses
        # (freq-rect/rrc): the values are zero-stuffed at T/2 spacing and shaped by the same FIR -- for
        # the 20 GHz brickwall at 20 GBd the T/2 slots are exactly at critical Nyquist (40 Gslot/s in
        # 20 GHz), so double-rate signalling FITS the band. upam-4 (rx=1 sps) stays at 1 value/symbol.
        waveform_freedom = getattr(config, "tx_waveform_freedom", True)   # True -> sps_rx values/symbol (double-OOK)
        values = getattr(config, "tx_values_per_symbol", None)            # None -> sps_rx; e.g. 4 = sim rate,
        self.tx_subsymbols = ((values or config.samples_per_symbol_rx)    # finer TX granularity for spectral
                              if waveform_freedom and self.equalizer == "end-to-end" else 1)  # pre-compensation (CD)

        self.memory = config.dpd_memory_symbols
        # hidden widths: a per-layer list (dpd_hidden_widths, e.g. [32, 64, 16] = the "complex" TX)
        # or the legacy single width replicated dpd_hidden_layers times.
        widths = getattr(config, "dpd_hidden_widths", None)
        if not widths:
            widths = [config.dpd_hidden_width] * max(1, getattr(config, "dpd_hidden_layers", 1))
        # DPD: fully-connected layers over a sliding window of `memory` symbols (used only when equalizer == "end-to-end")
        self.context_layer = nn.Linear(self.memory * self.num_bits, widths[0])
        # optional extra hidden layers (depth adds pre-distortion capacity at the SAME memory of 5 symbols)
        self.extra_layers = nn.ModuleList(nn.Linear(widths[i], widths[i + 1])
                                          for i in range(len(widths) - 1))
        self.segment_layer = nn.Linear(widths[-1], self.num_segments * self.tx_subsymbols)
        # Exploratory init: the default small-weight init leaves tanh~0 -> every symbol starts at the drive
        # midpoint (for bpam-4 that is the MZM null, field~0) -> the joint optimiser gets stuck in a
        # quasi-unipolar local min (the RX cannot reward a sign it does not yet see). Scaling the DPD weights
        # up makes the symbols start SPREAD across the full drive range [drive_min, drive_max] (bipolar for
        # bpam-4), so the TX explores bipolar signalling from step 0. This imposes NO specific levels -- only a
        # wider starting spread; gradient descent still learns the constellation.
        gain = getattr(config, "tx_init_gain", 1.0)
        if gain != 1.0:
            with torch.no_grad():
                self.context_layer.weight *= gain
                self.segment_layer.weight *= gain

        # fixed optimum levels for the non-DPD modes ("ffe", "threshold"):
        if self.modulation_format == "bpam-4":
            # BPAM alphabet (Secondini2020, eq. 11): field levels {+-A, +-2A} normalized to {+-1/2, +-1}.
            # Drive that produces field f: MZM (null bias)  V = Vpi*(1 - (2/pi)*arccos(f))  -> {+-Vpi/3, +-Vpi},
            # the 4 voltages evenly spaced over the whole MZM characteristic (Secondini2020, eq. 21);
            # linear-FIELD modulator  V = f*Vmax. Indexed by (amp_bit*2 + sign_state).
            field = torch.tensor([+0.5, -0.5, +1.0, -1.0], dtype=torch.float32)   # (amp,state)=(0,0),(0,1),(1,0),(1,1)
            swing = getattr(config, "bpam_classic_drive_swing", None)
            if self.modulator_kind == "mzm" and swing:
                # LINEAR drive at a fraction `swing` of Vpi (quasi-linear MZM region, Secondini/prof).
                # The arccos mapping below sends +-1.0 to +-Vpi (the MZM rails), where the optical PHASE
                # saturates and the adjacent-pulse cross term that carries the differential SIGN in DD is
                # compressed -- measured to raise the phase BER ~4x. The linear drive keeps the phase
                # relation and recovers the sign (verified: matches the reference receiver).
                vpi = config.mzm_vpi_volt
                fixed = vpi * swing * field
            elif self.modulator_kind == "mzm":
                vpi = config.mzm_vpi_volt
                fixed = vpi * (1.0 - (2.0 / np.pi) * torch.arccos(field))
            else:
                fixed = field * self.drive_max
        else:
            # unipolar (Gray):
            #   thermal (A.19): equispaced INTENSITY  (intensity ~ rank   -> linear drive ~ rank)
            #   ase     (A.47): equispaced AMPLITUDE  (intensity ~ rank^2 -> linear drive ~ rank^2)
            gray_rank = torch.tensor([0, 1, 3, 2], dtype=torch.float32)[: self.modulation_order]
            norm_rank = gray_rank / (self.modulation_order - 1)
            frac = norm_rank ** 2 if self.noise_regime == "ase" else norm_rank
            fixed = self.drive_min + (self.drive_max - self.drive_min) * frac
        self.register_buffer("fixed_levels", fixed, persistent=False)   # derived from config -> not in state_dict

        # TX pulse: "time-rect" = one-symbol rectangle (sample-and-hold, no convolution); the band-limited
        # pulses ("rrc", "freq-rect"/"rect") are FIR-convolved (unit-peak: symbol-instant drive == level).
        if self.pulse_kind == "time-rect":
            self.register_buffer("rrc", torch.ones(1, 1, self.samples_per_symbol))   # not convolved; kept for API
        else:
            if self.pulse_kind in ("freq-rect", "rect"):
                tx_taps = brickwall(config.rect_filter_bandwidth, config.sim_sample_rate,
                                    config.rrc_span_symbols, self.samples_per_symbol)
            elif self.pulse_kind == "nrz":
                from pulse_shaping import nrz_pulse                 # prof's NRZ: h(T/2) ~ 50% (keeps the sign)
                tx_taps = nrz_pulse(config.rrc_rolloff, config.rrc_span_symbols, self.samples_per_symbol,
                                    getattr(config, "nrz_rect_cutoff_ratio", None))
            else:
                tx_taps = root_raised_cosine(config.rrc_rolloff, config.rrc_span_symbols, self.samples_per_symbol)
            tx_taps = tx_taps / tx_taps.max()
            self.register_buffer("rrc", torch.tensor(tx_taps, dtype=torch.float32).view(1, 1, -1))

    def _precode_bpam(self, bits):
        """Differential sign precoder for bpam-4 (OUTSIDE the network, like the Gray map: a discrete
        structure a finite-window net cannot learn, because the absolute sign is unobservable in DD --
        only sign CHANGES between adjacent pulses are). Row 0 = amplitude bit; row 1 = phase-difference
        bit (dphi in {0, pi}). Returns (amp_bit, sign_state), sign_state_k = XOR-cumulated phase bits.

        Marco's recursion c2[k] = c2[k-1] XOR b2[k] equals cumsum%2 (each output = XOR of all
        preceding input phase bits); it is NOT the wrong adjacent XOR b2[k-1] XOR b2[k]."""
        sign_state = torch.cumsum(bits[1], dim=0) % 2
        return torch.stack([bits[0], sign_state])

    def symbol_drive_levels(self, bits):
        """Per-symbol drive levels (num_segments, num_symbols), BEFORE pulse shaping. bits are the raw
        information bits; for bpam-4 the differential sign precoder is applied internally."""
        num_symbols = bits.shape[1]
        # Double-rate route: NO precoder in end-to-end -- the network sees RAW bits and, with sub-symbol
        # waveform freedom, is free to place each bit on an independent T/2 slot (double-rate OOK / hybrid)
        # that goes below the symbol-rate PAM theory by using the extra RX bandwidth. The differential
        # precoder is kept only for the fixed-level modes (classical BPAM, where the sign is differential).
        precode = self.modulation_format == "bpam-4" and (
            self.equalizer != "end-to-end" or getattr(self, "precode_e2e", False))
        coded = self._precode_bpam(bits) if precode else bits
        if self.equalizer == "end-to-end":
            x = coded.unsqueeze(0).float()                     # (1, num_bits, num_symbols)
            pad = self.memory // 2                             # sliding window of `memory` symbols
            x_pad = F.pad(x, (pad, pad))
            windows = x_pad.unfold(2, self.memory, 1)          # (1, num_bits, num_symbols, memory)
            flat = windows.permute(0, 2, 1, 3).reshape(1, num_symbols, -1)
            hidden = F.leaky_relu(self.context_layer(flat))
            for layer in self.extra_layers:                    # optional DPD depth (same 5-symbol memory)
                hidden = F.leaky_relu(layer(hidden))
            raw = self.segment_layer(hidden)                   # (1, num_symbols, num_segments*tx_subsymbols)
            # output bounding: "tanh" (smooth, saturating) or "hardtanh" (linear, hard clip at the rails)
            bounded = torch.tanh(raw) if self.output_activation == "tanh" else F.hardtanh(raw)
            # (1, num_symbols, num_segments, tx_subsymbols) -> (num_segments, num_symbols * tx_subsymbols):
            # the tx_subsymbols values of a symbol are placed consecutively in time (one per T/2 slot).
            bounded = bounded.view(1, num_symbols, self.num_segments, self.tx_subsymbols)
            bounded = bounded.permute(0, 2, 1, 3).reshape(1, self.num_segments, num_symbols * self.tx_subsymbols)
            drive_levels = self.drive_min + (self.drive_max - self.drive_min) * 0.5 * (bounded + 1.0)
            return drive_levels.squeeze(0)                     # (num_segments, num_symbols * tx_subsymbols)
        if self.modulation_format == "bpam-4":                 # no DPD: fixed BPAM levels via (amp, sign)
            index = coded[0] * 2 + coded[1]                    # (num_symbols,) in 0..3
            return self.fixed_levels[index].unsqueeze(0).expand(self.num_segments, -1)
        symbols = bits_to_symbols(bits, self.num_bits)         # no DPD: fixed equispaced-intensity levels
        return self.fixed_levels[symbols].unsqueeze(0).expand(self.num_segments, -1)

    def forward(self, bits):
        """bits: (num_bits, num_symbols) -> drive waveform: (num_segments, num_symbols * sps)."""
        num_symbols = bits.shape[1]
        drive_levels = self.symbol_drive_levels(bits)

        if self.pulse_kind == "time-rect":                     # sample-and-hold; tx_subsymbols slots per symbol
            hold = self.samples_per_symbol // self.tx_subsymbols   # sim-samples per T/2 slot (== sps if 1 value/symbol)
            drive_wave = drive_levels.repeat_interleave(hold, dim=1)
        else:                                                  # band-limited pulse: zero-stuff + FIR
            upsampled = torch.zeros(self.num_segments, num_symbols * self.samples_per_symbol,
                                    device=drive_levels.device)
            stride = self.samples_per_symbol // self.tx_subsymbols   # T/2 spacing when tx_subsymbols = 2
            upsampled[:, ::stride] = drive_levels
            rrc_bank = self.rrc.repeat(self.num_segments, 1, 1)
            drive_wave = F.conv1d(upsampled.unsqueeze(0), rrc_bank,
                                  padding=self.rrc.shape[-1] // 2, groups=self.num_segments).squeeze(0)
        if self.tx_gauss is not None:                          # Gaussian cascade after the TX pulse filter
            drive_wave = F.conv1d(drive_wave.unsqueeze(1), self.tx_gauss,
                                  padding=self.tx_gauss.shape[-1] // 2).squeeze(1)
        if self.dac_bits is not None:                          # DAC: quantize the digital waveform (STE)
            drive_wave = quantize_ste(drive_wave, self.dac_bits, self.drive_min, self.drive_max)
        return drive_wave


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tx = Transmitter(cfg).to(device)

    num_symbols = 1000
    bits = torch.randint(0, 2, (cfg.bits_per_symbol, num_symbols), device=device)
    drive = tx(bits)
    print("bits shape :", tuple(bits.shape))
    print("drive shape:", tuple(drive.shape), "(expected", (cfg.num_mzm_segments, num_symbols * cfg.samples_per_symbol_sim), ")")
    print("drive range:", round(float(drive.min()), 3), "to", round(float(drive.max()), 3), "V")
    drive.pow(2).mean().backward()
    print("grad to DPD ok, mean|grad| =", float(tx.context_layer.weight.grad.abs().mean()))

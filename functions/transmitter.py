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


def bits_to_symbols(bits, bits_per_symbol):
    """Map bit rows (bits_per_symbol, num_symbols) to integer symbols 0..M-1 (Gray-free, MSB first)."""
    weights = 2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=bits.device).view(-1, 1)
    return (bits * weights).sum(dim=0)


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

        self.memory = config.dpd_memory_symbols
        hidden = config.dpd_hidden_width
        # DPD: fully-connected layers over a sliding window of `memory` symbols (used only when equalizer == "end-to-end")
        self.context_layer = nn.Linear(self.memory * self.num_bits, hidden)
        self.segment_layer = nn.Linear(hidden, self.num_segments)

        # fixed optimum levels (Gray) for the non-DPD modes ("ffe", "threshold"):
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
            else:
                tx_taps = root_raised_cosine(config.rrc_rolloff, config.rrc_span_symbols, self.samples_per_symbol)
            tx_taps = tx_taps / tx_taps.max()
            self.register_buffer("rrc", torch.tensor(tx_taps, dtype=torch.float32).view(1, 1, -1))

    def forward(self, bits):
        """bits: (num_bits, num_symbols) -> drive waveform: (num_segments, num_symbols * sps)."""
        num_symbols = bits.shape[1]
        if self.equalizer == "end-to-end":
            x = bits.unsqueeze(0).float()                      # (1, num_bits, num_symbols)
            pad = self.memory // 2                             # sliding window of `memory` symbols
            x_pad = F.pad(x, (pad, pad))
            windows = x_pad.unfold(2, self.memory, 1)          # (1, num_bits, num_symbols, memory)
            flat = windows.permute(0, 2, 1, 3).reshape(1, num_symbols, -1)
            hidden = F.leaky_relu(self.context_layer(flat))
            bounded = torch.tanh(self.segment_layer(hidden))   # (1, num_symbols, num_segments) in [-1, 1]
            bounded = bounded.permute(0, 2, 1)                 # (1, num_segments, num_symbols)
            drive_levels = self.drive_min + (self.drive_max - self.drive_min) * 0.5 * (bounded + 1.0)
            drive_levels = drive_levels.squeeze(0)             # (num_segments, num_symbols)
        else:                                                  # no DPD: fixed equispaced-intensity levels
            symbols = bits_to_symbols(bits, self.num_bits)     # (num_symbols,)
            drive_levels = self.fixed_levels[symbols].unsqueeze(0).expand(self.num_segments, -1)

        if self.pulse_kind == "time-rect":                     # one-symbol rectangle = sample-and-hold
            drive_wave = drive_levels.repeat_interleave(self.samples_per_symbol, dim=1)
        else:                                                  # band-limited pulse: zero-stuff + FIR
            upsampled = torch.zeros(self.num_segments, num_symbols * self.samples_per_symbol,
                                    device=drive_levels.device)
            upsampled[:, ::self.samples_per_symbol] = drive_levels
            rrc_bank = self.rrc.repeat(self.num_segments, 1, 1)
            drive_wave = F.conv1d(upsampled.unsqueeze(0), rrc_bank,
                                  padding=self.rrc.shape[-1] // 2, groups=self.num_segments).squeeze(0)
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

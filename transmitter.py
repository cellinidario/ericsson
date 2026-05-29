"""Transmitter: PAM-4 mapping + digital predistortion (DPD).

Fork A: the DPD works at SYMBOL RATE with a temporal context window, outputs one
predistorted drive level per MZM segment, and a FIXED RRC does the oversampling.
This keeps the time-multiplexing degree of freedom out of the network: one level
per symbol, shaped by a Nyquist pulse.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pulse_shaping import root_raised_cosine


def bits_to_symbols(bits, bits_per_symbol):
    """Map bit rows (bits_per_symbol, num_symbols) to integer symbols 0..M-1 (Gray-free, MSB first)."""
    weights = 2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=bits.device).view(-1, 1)
    return (bits * weights).sum(dim=0)


class Transmitter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_bits = config.bits_per_symbol
        self.num_segments = config.num_mzm_segments
        self.samples_per_symbol = config.samples_per_symbol_sim
        self.drive_min = config.drive_min_volt
        self.drive_max = config.drive_max_volt

        memory = config.dpd_memory_symbols
        hidden = config.dpd_hidden_width
        self.context_layer = nn.Conv1d(self.num_bits, hidden, kernel_size=memory, padding=memory // 2)
        self.segment_layer = nn.Conv1d(hidden, self.num_segments, kernel_size=1)

        rrc_taps = root_raised_cosine(config.rrc_rolloff, config.rrc_span_symbols, self.samples_per_symbol)
        rrc_taps = rrc_taps / rrc_taps.max()           # unit-peak: symbol-instant drive == DPD level (volts)
        self.register_buffer("rrc", torch.tensor(rrc_taps, dtype=torch.float32).view(1, 1, -1))

    def forward(self, bits):
        """bits: (num_bits, num_symbols) -> drive waveform: (num_segments, num_symbols * sps)."""
        x = bits.unsqueeze(0).float()                          # (1, num_bits, num_symbols)
        hidden = F.leaky_relu(self.context_layer(x))
        bounded = torch.tanh(self.segment_layer(hidden))       # (1, num_segments, num_symbols) in [-1, 1]
        drive_levels = self.drive_min + (self.drive_max - self.drive_min) * 0.5 * (bounded + 1.0)
        drive_levels = drive_levels.squeeze(0)                 # (num_segments, num_symbols)

        num_symbols = drive_levels.shape[1]
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

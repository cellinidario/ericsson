"""Receiver: matched RRC filter + decimation to the ADC rate + feed-forward equalizer.

The matched filter limits the photocurrent to the signal band before decimation,
so sampling down to samples_per_symbol_rx is alias-free. The FFE is two fully-
connected layers applied to a sliding window of `ffe_memory_symbols * samples_per_symbol_rx`
samples, producing one bit-probability vector per symbol (trained by BCE).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pulse_shaping import root_raised_cosine


class Receiver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_bits = config.bits_per_symbol
        self.samples_per_symbol_sim = config.samples_per_symbol_sim
        self.samples_per_symbol_rx = config.samples_per_symbol_rx
        self.decimation = self.samples_per_symbol_sim // self.samples_per_symbol_rx

        matched_taps = root_raised_cosine(config.rrc_rolloff, config.rrc_span_symbols, self.samples_per_symbol_sim)
        self.register_buffer("matched", torch.tensor(matched_taps, dtype=torch.float32).view(1, 1, -1))

        hidden = config.ffe_hidden_width
        # window of `memory` symbols, each contributing `samples_per_symbol_rx` samples
        self.window_size = config.ffe_memory_symbols * self.samples_per_symbol_rx
        self.context_layer = nn.Linear(self.window_size, hidden)
        self.bit_layer = nn.Linear(hidden, self.num_bits)

    def forward(self, photocurrent):
        """photocurrent: (num_samples,) at the sim rate -> bit probabilities: (num_bits, num_symbols)."""
        x = photocurrent.view(1, 1, -1)
        x = F.conv1d(x, self.matched, padding=self.matched.shape[-1] // 2)   # matched filter
        x = x[:, :, ::self.decimation]                                       # decimate to ADC rate
        # sliding window of `window_size` samples, stride = samples_per_symbol_rx -> one window per symbol
        pad = (self.window_size - self.samples_per_symbol_rx) // 2
        x_pad = F.pad(x, (pad, pad))
        windows = x_pad.unfold(2, self.window_size, self.samples_per_symbol_rx)   # (1, 1, num_symbols, window_size)
        num_symbols = windows.shape[2]
        flat = windows.permute(0, 2, 1, 3).reshape(1, num_symbols, -1)            # (1, num_symbols, window_size)
        hidden = F.leaky_relu(self.context_layer(flat))                           # (1, num_symbols, hidden)
        bit_probability = torch.sigmoid(self.bit_layer(hidden)).squeeze(0).t()    # (num_bits, num_symbols)
        return bit_probability


if __name__ == "__main__":
    from config import Config
    from transmitter import Transmitter
    from channel import OpticalChannel
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tx = Transmitter(cfg).to(device)
    channel = OpticalChannel(cfg).to(device)
    rx = Receiver(cfg).to(device)

    num_symbols = 2000
    bits = torch.randint(0, 2, (cfg.bits_per_symbol, num_symbols), device=device)
    drive = tx(bits)
    photocurrent = channel(drive, add_noise=False)
    bit_probability = rx(photocurrent)

    print("bits        :", tuple(bits.shape))
    print("photocurrent:", tuple(photocurrent.shape))
    print("bit prob    :", tuple(bit_probability.shape), "(expected", (cfg.bits_per_symbol, num_symbols), ")")

    # end-to-end gradient: BCE loss must reach both TX and RX parameters
    target = bits.float()[:, :bit_probability.shape[1]]
    loss = F.binary_cross_entropy(bit_probability[:, :target.shape[1]], target)
    loss.backward()
    tx_grad = tx.context_layer.weight.grad.abs().mean()
    rx_grad = rx.context_layer.weight.grad.abs().mean()
    print("E2E loss    :", round(float(loss), 4))
    print("grad TX/RX  :", float(tx_grad), "/", float(rx_grad), "(both > 0 => chain wired)")

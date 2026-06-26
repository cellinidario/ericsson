"""Receiver: matched filter + decimation to the ADC rate + feed-forward classifier.

The matched filter limits the photocurrent to the signal band before decimation, so
sampling down to samples_per_symbol_rx is alias-free. The FFE is two fully-connected
layers over a sliding window of `ffe_memory_symbols * samples_per_symbol_rx` samples,
producing SYMBOL logits (M-way) per symbol, trained by symbol cross-entropy. Symbol CE
keeps the constellation from collapsing (per-bit BCE alone could abandon a bit -> a
degenerate local minimum); the levels are LEARNED, not imposed. The per-bit LLRs for
soft-FEC (Marco's P(b_i|context), the "bitwise" output) are recovered by marginalizing
the symbol softmax over the bit<->symbol map (bit_posteriors). Pass the amplitude-Gray
map (train.amplitude_gray_bits) so adjacent levels differ by one bit -> reaches the A.19
bound; the default binary map costs the 4/3 (~1.33x) Gray penalty.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pulse_shaping import root_raised_cosine, brickwall


class Receiver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_bits = config.bits_per_symbol
        self.modulation_order = config.modulation_order
        self.samples_per_symbol_sim = config.samples_per_symbol_sim
        self.samples_per_symbol_rx = config.samples_per_symbol_rx
        self.decimation = self.samples_per_symbol_sim // self.samples_per_symbol_rx

        self.pulse_kind = getattr(config, "rx_filter", "rrc")
        # RX matched filter (keep = tx_filter to stay matched). "time-rect" -> integrate-and-dump (applied as a
        # reshape-sum in matched_and_decimate); the boxcar buffer below is only used by the eye-diagram cell.
        if self.pulse_kind == "time-rect":
            taps = np.ones(self.samples_per_symbol_sim, dtype=np.float32)
            matched_taps = taps / taps.sum()                     # unit gain: averages the symbol (held A -> A)
        elif self.pulse_kind in ("freq-rect", "rect"):
            matched_taps = brickwall(config.rect_filter_bandwidth, config.sim_sample_rate,
                                     config.rrc_span_symbols, self.samples_per_symbol_sim)
        else:
            matched_taps = root_raised_cosine(config.rrc_rolloff, config.rrc_span_symbols, self.samples_per_symbol_sim)
        self.register_buffer("matched", torch.tensor(matched_taps, dtype=torch.float32).view(1, 1, -1))

        hidden = config.ffe_hidden_width
        # window of `memory` symbols, each contributing `samples_per_symbol_rx` samples
        self.window_size = config.ffe_memory_symbols * self.samples_per_symbol_rx
        self.context_layer = nn.Linear(self.window_size, hidden)
        self.symbol_head = nn.Linear(hidden, self.modulation_order)   # M-way symbol classifier
        self.ffe_nonlinear = getattr(config, "ffe_nonlinear", True)   # False -> purely linear FFE (no activation)

    def matched_and_decimate(self, photocurrent):
        """Matched filter + downsample to the symbol rate -> (1, 1, num_symbols * sps_rx).
        "time-rect": integrate-and-dump (sum the sps_sim samples of each symbol -> one statistic/symbol);
        band-limited pulses: FIR matched filter + decimation at the symbol centre (offset 0)."""
        if self.pulse_kind == "time-rect":
            sps = self.samples_per_symbol_sim
            n = photocurrent.shape[-1] // sps
            return photocurrent[:n * sps].view(n, sps).mean(dim=1).view(1, 1, -1)   # integrate-and-dump, unit gain
        x = F.conv1d(photocurrent.view(1, 1, -1), self.matched, padding=self.matched.shape[-1] // 2)
        return x[:, :, ::self.decimation]

    def features(self, photocurrent):
        """Shared FFE features per symbol: (1, num_symbols, hidden)."""
        x = self.matched_and_decimate(photocurrent)
        # sliding window of `window_size` samples, stride = samples_per_symbol_rx -> one window per symbol
        pad = (self.window_size - self.samples_per_symbol_rx) // 2
        x_pad = F.pad(x, (pad, pad))
        windows = x_pad.unfold(2, self.window_size, self.samples_per_symbol_rx)   # (1, 1, num_symbols, window_size)
        num_symbols = windows.shape[2]
        flat = windows.permute(0, 2, 1, 3).reshape(1, num_symbols, -1)            # (1, num_symbols, window_size)
        hidden = self.context_layer(flat)                                         # (1, num_symbols, hidden)
        if self.ffe_nonlinear:
            hidden = F.leaky_relu(hidden)                                         # nonlinear FFE (linear if False)
        return hidden

    def forward(self, photocurrent):
        """photocurrent: (num_samples,) at the sim rate -> symbol logits: (num_symbols, M)."""
        return self.symbol_head(self.features(photocurrent)).squeeze(0)           # (num_symbols, M)


def bit_posteriors(symbol_logits, num_bits, symbol_bits=None):
    """Per-bit posteriors P(b_i = 1) from the symbol softmax: (num_bits, num_symbols).
    symbol_bits: optional (M, num_bits) map giving each symbol's bit labels (e.g. the amplitude-Gray
    map from train.amplitude_gray_bits). Default None -> the BINARY map matching bits_to_symbols
    (symbol = sum_i b_i * 2**(num_bits-1-i)), which carries the 4/3 Gray penalty for the E2E."""
    probs = torch.softmax(symbol_logits, dim=-1)                  # (num_symbols, M)
    if symbol_bits is None:
        sym = torch.arange(probs.shape[-1], device=probs.device)
        bits = [probs[:, ((sym >> (num_bits - 1 - i)) & 1) == 1].sum(dim=-1) for i in range(num_bits)]
    else:
        mask = symbol_bits.to(probs.device, probs.dtype)          # (M, num_bits)
        bits = [(probs * mask[:, i]).sum(dim=-1) for i in range(num_bits)]
    return torch.stack(bits)                                      # (num_bits, num_symbols)


if __name__ == "__main__":
    from config import Config
    from transmitter import Transmitter, bits_to_symbols
    from channel import OpticalChannel
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tx = Transmitter(cfg).to(device)
    channel = OpticalChannel(cfg).to(device)
    rx = Receiver(cfg).to(device)

    num_symbols = 2000
    bits = torch.randint(0, 2, (cfg.bits_per_symbol, num_symbols), device=device)
    symbols = bits_to_symbols(bits, cfg.bits_per_symbol)
    logits = rx(channel(tx(bits)))
    print("symbol logits :", tuple(logits.shape), "(expected", (num_symbols, cfg.modulation_order), ")")
    print("bit posteriors:", tuple(bit_posteriors(logits, cfg.bits_per_symbol).shape),
          "(expected", (cfg.bits_per_symbol, num_symbols), ")")

    # end-to-end gradient: symbol cross-entropy must reach both TX and RX parameters
    loss = F.cross_entropy(logits, symbols[:logits.shape[0]])
    loss.backward()
    print("E2E CE loss   :", round(float(loss), 4),
          "| grad TX/RX:", float(tx.context_layer.weight.grad.abs().mean()),
          "/", float(rx.context_layer.weight.grad.abs().mean()), "(both > 0 => chain wired)")

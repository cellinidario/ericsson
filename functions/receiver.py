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
from channel import supergaussian_fir
from utils import quantize_ste


class Receiver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_bits = config.bits_per_symbol
        self.modulation_order = config.modulation_order
        self.samples_per_symbol_sim = config.samples_per_symbol_sim
        self.samples_per_symbol_rx = config.samples_per_symbol_rx
        self.decimation = self.samples_per_symbol_sim // self.samples_per_symbol_rx

        self.pulse_kind = getattr(config, "rx_filter", "rrc")
        self.modulation_format = getattr(config, "modulation_format", "upam-4")   # "upam-4" | "bpam-4"
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

        # optional digital Gaussian LPF on the photocurrent before decimation (JLT setup: 10 GHz)
        gauss_bw = getattr(config, "rx_gaussian_bw", None)
        if gauss_bw:
            taps = supergaussian_fir(gauss_bw, 1, config.sim_sample_rate, 65)
            self.register_buffer("rx_gauss", torch.tensor(taps[::-1].copy(), dtype=torch.float32).view(1, 1, -1))
        else:
            self.rx_gauss = None
        self.adc_bits = getattr(config, "adc_bits", None)      # ADC resolution (None = ideal)

        # hidden widths: either a per-layer list (ffe_hidden_widths, e.g. [32, 64, 16] = Asfand's
        # "complex" receiver) or the legacy single width replicated ffe_hidden_layers times.
        widths = getattr(config, "ffe_hidden_widths", None)
        if not widths:
            widths = [config.ffe_hidden_width] * max(1, getattr(config, "ffe_hidden_layers", 1))
        # window of `memory` symbols, each contributing `samples_per_symbol_rx` samples.
        # int(): ffe_memory_symbols may be a half-integer to express an ODD window in samples --
        # Asfand's receivers use 11 samples = 5.5 symbols at 2 sps (Marco, 22/7).
        self.window_size = int(round(config.ffe_memory_symbols * self.samples_per_symbol_rx))
        self.context_layer = nn.Linear(self.window_size, widths[0])
        # optional extra hidden layers (depth = nonlinear inversion capacity, e.g. the short CD x square-law
        # distortion in C-band; 1 = the original two-FC receiver, no behaviour change)
        self.extra_layers = nn.ModuleList(nn.Linear(widths[i], widths[i + 1])
                                          for i in range(len(widths) - 1))
        self.symbol_head = nn.Linear(widths[-1], self.modulation_order)   # M-way symbol classifier
        # optional per-bit SIGMOID output head (Marco's z_k directly): m sigmoid units on the shared trunk.
        # The symbol head is then a TRAINING-ONLY auxiliary (its CE keeps the 2^m classes distinct against
        # the factorized-loss collapse); at inference only the bit head is used.
        self.bit_head = (nn.Linear(widths[-1], self.num_bits)
                         if getattr(config, "rx_bit_head", False) else None)
        self.ffe_nonlinear = getattr(config, "ffe_nonlinear", True)   # False -> purely linear FFE (no activation)

        # DUAL-NETWORK receiver (bpam-4): one INDEPENDENT network per bit -- net 1 for the amplitude
        # bit a_k (which lives in the symbol-centre sample) and net 2 for the differential phase bit
        # dphi_k (which lives in the T/2 cross-term). With a single shared trunk the two bits compete
        # for the same features and both saturate; separate trunks let each specialise.
        self.dual_network = bool(getattr(config, "rx_dual_network", False))
        # Marco (22/7): Asfand's two receivers see windows STAGGERED BY ONE 2-sps SAMPLE, each
        # centred on the quantity it estimates -- the amplitude in the middle of the pulse
        # (offset 0), the sign change at the intersection of two adjacent pulses (offset 1).
        # An earlier attempt with two trunks on the SAME window scored exactly like the single
        # trunk (1.36e-2 vs 1.31e-2 @OSNR15): without the stagger there is nothing to specialise on.
        self.phase_window_offset = int(getattr(config, "rx_phase_window_offset", 1))
        if self.dual_network:
            self.context_layer2 = nn.Linear(self.window_size, widths[0])
            self.extra_layers2 = nn.ModuleList(nn.Linear(widths[i], widths[i + 1])
                                               for i in range(len(widths) - 1))
            self.bit_head1 = nn.Linear(widths[-1], 1)      # amplitude bit, from trunk 1
            self.bit_head2 = nn.Linear(widths[-1], 1)      # phase bit, from trunk 2

    def matched_and_decimate(self, photocurrent):
        """Matched filter + downsample to the symbol rate -> (1, 1, num_symbols * sps_rx).
        "time-rect" at 1 rx sps: integrate-and-dump (one statistic per symbol).
        "time-rect" at 2 rx sps (bpam-4): the boxcar matched filter is applied as a CONVOLUTION and
        sampled at 2 sps -- rect (X) rect gives the triangular h(t) with h(+-T/2) = 1/2, so the T/2
        samples contain the adjacent-pulse interference that carries the sign (Secondini2020, footnote 2
        choice i: rectangular pulse + matched optical filter). The plain integrate-and-dump would return
        one sample/symbol and destroy the sign information (non-overlapping rects have no cross terms).
        Band-limited pulses: FIR matched filter + decimation at the symbol centre (offset 0)."""
        if self.pulse_kind == "time-rect" and self.samples_per_symbol_rx == 1:
            sps = self.samples_per_symbol_sim
            n = photocurrent.shape[-1] // sps
            return photocurrent[:n * sps].view(n, sps).mean(dim=1).view(1, 1, -1)   # integrate-and-dump, unit gain
        x = F.conv1d(photocurrent.view(1, 1, -1), self.matched, padding=self.matched.shape[-1] // 2)
        return x[:, :, ::self.decimation]

    def windows_from(self, photocurrent, offset=0, window_size=None):
        """Front-end shared by every trunk: RX Gaussian, decimation to s' sps, ADC, then one
        sliding window per symbol -> (1, num_symbols, window_size).

        `offset` shifts the window by whole 2-sps samples. Marco (22/7): Asfand's two receivers
        use windows STAGGERED BY ONE SAMPLE, each centred on the quantity it estimates -- the
        amplitude in the middle of the pulse, the sign change at the intersection of two adjacent
        pulses. offset=0 -> centred on the symbol; offset=1 -> centred on the T/2 crossing."""
        win = self.window_size if window_size is None else window_size
        if self.rx_gauss is not None:                          # digital Gaussian LPF (JLT: 10 GHz)
            photocurrent = F.conv1d(photocurrent.view(1, 1, -1), self.rx_gauss,
                                    padding=self.rx_gauss.shape[-1] // 2).view(-1)
        if self.modulation_format == "bpam-4":
            # bpam-4: feed the RAW photocurrent samples at 2 sps (pure decimation, no matched filter):
            # the optical filtering already happened on the FIELD inside the channel (where the sign
            # information is created by pulse interference), and the FFE window learns any further
            # filtering itself — the same front-end as the classical 2-sps BPAM equalizers.
            x = photocurrent.view(1, 1, -1)[:, :, ::self.decimation]
        else:
            x = self.matched_and_decimate(photocurrent)
        if self.adc_bits is not None:                          # ADC: quantize the decimated samples (STE);
            lo, hi = float(x.min()), float(x.max())            # full-scale = observed range (ideal AGC)
            x = quantize_ste(x, self.adc_bits, lo, hi)
        # sliding window of `win` samples, stride = samples_per_symbol_rx -> one window per symbol.
        # `offset` staggers the window: the left pad shrinks and the right pad grows by `offset`,
        # so the window slides forward by that many 2-sps samples while keeping one window/symbol.
        pad = (win - self.samples_per_symbol_rx) // 2
        x_pad = F.pad(x, (pad - offset, pad + offset))
        windows = x_pad.unfold(2, win, self.samples_per_symbol_rx)                # (1, 1, num_symbols, win)
        num_symbols = windows.shape[2]
        return windows.permute(0, 2, 1, 3).reshape(1, num_symbols, -1)            # (1, num_symbols, win)

    def _trunk(self, flat, context_layer, extra_layers):
        """One fully-connected trunk applied to the per-symbol windows."""
        hidden = context_layer(flat)
        if self.ffe_nonlinear:
            hidden = F.leaky_relu(hidden)                                         # nonlinear FFE (linear if False)
        for layer in extra_layers:                                                # optional depth (see __init__)
            hidden = layer(hidden)
            if self.ffe_nonlinear:
                hidden = F.leaky_relu(hidden)
        return hidden

    def features(self, photocurrent):
        """Shared FFE features per symbol: (1, num_symbols, hidden)."""
        return self._trunk(self.windows_from(photocurrent), self.context_layer, self.extra_layers)

    def dual_bit_logits(self, photocurrent):
        """Dual-network receiver: independent trunk per bit -> bit logits (num_symbols, 2).
        Trunk 1 -> amplitude bit (window centred on the symbol), trunk 2 -> differential phase
        bit (window staggered by rx_phase_window_offset samples, onto the T/2 crossing)."""
        h1 = self._trunk(self.windows_from(photocurrent, offset=0),
                         self.context_layer, self.extra_layers)
        h2 = self._trunk(self.windows_from(photocurrent, offset=self.phase_window_offset),
                         self.context_layer2, self.extra_layers2)
        return torch.cat([self.bit_head1(h1), self.bit_head2(h2)], dim=-1).squeeze(0)

    def forward(self, photocurrent):
        """photocurrent: (num_samples,) at the sim rate -> symbol logits: (num_symbols, M)."""
        return self.symbol_head(self.features(photocurrent)).squeeze(0)           # (num_symbols, M)

    def bit_and_symbol_logits(self, photocurrent):
        """One trunk pass -> (bit logits (num_symbols, m), symbol logits (num_symbols, M)).
        Requires the sigmoid bit head (config.rx_bit_head = True). With the dual-network receiver
        the bit logits come from the two independent trunks and the auxiliary symbol head from
        trunk 1 (it is a training-only anti-collapse term)."""
        if self.dual_network:
            h1 = self._trunk(self.windows_from(photocurrent, offset=0),
                             self.context_layer, self.extra_layers)
            h2 = self._trunk(self.windows_from(photocurrent, offset=self.phase_window_offset),
                             self.context_layer2, self.extra_layers2)
            bit_logits = torch.cat([self.bit_head1(h1), self.bit_head2(h2)], dim=-1).squeeze(0)
            return bit_logits, self.symbol_head(h1).squeeze(0)
        h = self.features(photocurrent)
        return self.bit_head(h).squeeze(0), self.symbol_head(h).squeeze(0)

    def bit_posteriors_direct(self, photocurrent):
        """z_k = sigmoid(bit head): per-bit posteriors, shape (num_bits, num_symbols)."""
        if self.dual_network:
            return torch.sigmoid(self.dual_bit_logits(photocurrent)).T
        h = self.features(photocurrent)
        return torch.sigmoid(self.bit_head(h)).squeeze(0).T


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

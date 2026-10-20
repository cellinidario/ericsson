"""TX and RX DSP: conventional transmitters, TX NN, TX lookup table and RX NN.

Bits have shape (blocks, 2, symbols): row 0 is the first bit of each symbol (BPAM: the
amplitude bit), row 1 the second (BPAM: the sign-change bit). TX outputs x[n] have shape
(blocks, 2*symbols): two samples per symbol, at k*T and k*T + T/2.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .parameters import PRECODERS, LinkParameters


def precoded_rows(fmt, use_precoder):
    """Rows of bits that are differentially precoded.

    True: the BPAM sign bit (row 1), and nothing for PAM-4; False: nothing. 'second' (row 1)
    and 'both' (rows 0 and 1) apply to either format.
    """
    if use_precoder not in PRECODERS:
        raise ValueError(f'{use_precoder!r} is not one of {PRECODERS}.')
    if use_precoder in ('second', 'both'):
        return (1,) if use_precoder == 'second' else (0, 1)
    return (1,) if use_precoder and fmt == 'bpam' else ()


def precoder_label(fmt, use_precoder):
    """The precoder setting as stored in checkpoints: a bool for True/False, as before, or the name."""
    return use_precoder if use_precoder in ('second', 'both') else bool(use_precoder) and fmt == 'bpam'


def precode(bits, rows=(1,)):
    """Differential precoding of the given rows of bits: d[k] = d[k-1] XOR b[k], with d[-1] = 0.

    Direct detection observes only sign changes of the field, so the BPAM sign bit (row 1) is
    sent as a change of sign. Other rows are unchanged. The receiver estimates b[k] directly.
    """
    return torch.stack([torch.cumsum(bits[:, r], dim=-1) % 2 if r in rows else bits[:, r] for r in range(2)], dim=1)


def symbol_contexts(coded, memory):
    """(blocks, 2, N) bits -> (blocks, N, 2*memory): bits of symbols k-2..k+2, first row then second row.

    Zeros are padded beyond the block edges.
    """
    x = F.pad(coded.float(), (memory//2, memory//2))
    windows = x.unfold(2, memory, 1)                                # (blocks, 2, N, memory)
    return windows.permute(0, 2, 1, 3).reshape(coded.shape[0], coded.shape[2], -1)


def conventional_levels(fmt, p=LinkParameters()):
    """Drive voltages of the conventional transmitters, before pulse shaping.

    BPAM (index 2*amplitude + sign): +-0.5 and +-1 times the peak voltage, a linear drive.
    PAM-4 (index 2*b0 + b1, Gray): voltages giving equally spaced optical intensities over
    the same peak voltage, by inverting the modulator intensity law sin^2(pi V/(2 V_pi)); with the
    modulator at quadrature (LinkParameters.mzm_bias), drive voltages around V_pi/2 that span the
    intensities reachable with +-peak_volt (all of them, from 0 to the maximum, if peak_volt = V_pi/2).
    """
    if fmt == 'bpam':
        if p.mzm_bias != 'null':
            raise ValueError('The conventional BPAM transmitter needs the modulator biased at null.')
        return p.peak_volt*torch.tensor([.5, -.5, 1., -1.])
    rank = torch.tensor([0., 1., 3., 2.])/3
    if p.mzm_bias == 'quadrature':
        low, high = (math.sin(math.pi/4+math.pi/2*s*p.peak_volt/p.vpi)**2 for s in (-1, 1))
        return 2*p.vpi/torch.pi*torch.asin(torch.sqrt(low+(high-low)*rank))-p.vpi/2
    return 2*p.vpi/torch.pi*torch.asin(math.sin(math.pi*p.peak_fraction/2)*torch.sqrt(rank))


class ConventionalTransmitter(nn.Module):
    """Fixed levels: one sample per symbol at k*T and zero at k*T + T/2, then the TX pulse."""

    def __init__(self, fmt, use_precoder=True, p=LinkParameters()):
        super().__init__()
        self.fmt, self.use_precoder = fmt, precoder_label(fmt, use_precoder)
        self.rows = precoded_rows(fmt, use_precoder)
        self.register_buffer('levels', conventional_levels(fmt, p))

    def forward(self, bits):
        coded = precode(bits, self.rows) if self.rows else bits
        levels = self.levels[2*coded[:, 0]+coded[:, 1]]
        x = torch.zeros(levels.shape[0], 2*levels.shape[1], device=levels.device)
        x[:, ::2] = levels
        return x


class TransmitterNN(nn.Module):
    """TX NN: context of five symbols (ten bits) -> hidden leaky-ReLU layer -> two samples per symbol.

    The outputs are bounded by tanh to the peak voltage. Since the input is binary, a trained
    network is exactly a table with 2^10 = 1024 rows (lookup_table()).
    """

    def __init__(self, fmt, use_precoder=True, p=LinkParameters()):
        super().__init__()
        self.fmt, self.use_precoder = fmt, precoder_label(fmt, use_precoder)
        self.rows = precoded_rows(fmt, use_precoder)
        self.memory, self.peak = p.tx_memory_symbols, p.peak_volt
        sizes = (p.tx_hidden,) if isinstance(p.tx_hidden, int) else tuple(p.tx_hidden)
        layers = []
        for width_in, width in zip((2*p.tx_memory_symbols,)+sizes[:-1], sizes):
            layers += [nn.Linear(width_in, width), nn.LeakyReLU()]
        # One hidden layer keeps the parameter names of the original network (hidden.weight, ...).
        self.hidden = layers[0] if len(sizes) == 1 else nn.Sequential(*layers[:-1])
        self.output = nn.Linear(sizes[-1], p.dsp_sps)
        self.register_buffer('levels', conventional_levels(fmt, p))   # target of the pretraining

    def contexts(self, coded):
        """(blocks, 2, N) bits -> (blocks, N, 10): bits of symbols k-2..k+2, first row then second row."""
        return symbol_contexts(coded, self.memory)

    def _volts(self, contexts):
        out = torch.tanh(self.output(F.leaky_relu(self.hidden(contexts))))
        return -self.peak+(2*self.peak)*.5*(out+1.)                # tanh range -> [-peak, +peak]

    def forward(self, bits):
        coded = precode(bits, self.rows) if self.rows else bits
        return self._volts(self.contexts(coded)).reshape(bits.shape[0], -1)

    def pretraining_loss(self, bits):
        """Mean-square error to the conventional level on the first sample and zero on the second."""
        coded = precode(bits, self.rows) if self.rows else bits
        target = torch.zeros(bits.shape[0], bits.shape[2], 2, device=bits.device)
        target[..., 0] = self.levels[2*coded[:, 0]+coded[:, 1]]
        return F.mse_loss(self(bits), target.reshape(bits.shape[0], -1))

    @torch.no_grad()
    def lookup_table(self):
        """The 1024 x 2 table equal to the trained network; row address = the ten context bits."""
        address = torch.arange(2**(2*self.memory), device=self.levels.device)
        bits = ((address[:, None] >> torch.arange(2*self.memory-1, -1, -1, device=address.device)) & 1).float()
        return self._volts(bits)


class LookupTransmitter(nn.Module):
    """TX as a trainable table with 2^10 = 1024 rows of two voltages.

    The row address is the context of TransmitterNN read as a binary number, most significant
    bit first: the five first-row bits of symbols k-2..k+2, then the five second-row bits. The
    entries are kept within the peak voltage by project(), called after every update.
    """

    def __init__(self, table, fmt, use_precoder=True, p=LinkParameters()):
        super().__init__()
        self.fmt, self.use_precoder = fmt, precoder_label(fmt, use_precoder)
        self.rows = precoded_rows(fmt, use_precoder)
        self.memory, self.peak = p.tx_memory_symbols, p.peak_volt
        if table.shape != (2**(2*self.memory), p.dsp_sps):
            raise ValueError(f'The table must have shape {(2**(2*self.memory), p.dsp_sps)}.')
        self.table = nn.Parameter(table.detach().clone())
        self.register_buffer('weights', 2**torch.arange(2*self.memory-1, -1, -1))

    @classmethod
    def from_network(cls, network, p=LinkParameters()):
        """The table equal to a trained TransmitterNN."""
        return cls(network.lookup_table(), network.fmt, network.use_precoder, p)

    def forward(self, bits):
        coded = precode(bits, self.rows) if self.rows else bits
        address = (symbol_contexts(coded, self.memory).long()*self.weights).sum(-1)
        return self.table[address].reshape(bits.shape[0], -1)

    @torch.no_grad()
    def project(self):
        self.table.clamp_(-self.peak, self.peak)


class Trunk(nn.Module):
    """Window of RX samples -> hidden leaky-ReLU layers -> logits."""

    def __init__(self, window, hidden, outputs):
        super().__init__()
        sizes = (window,)+tuple(hidden)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.head = nn.Linear(sizes[-1], outputs)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        return self.head(x)


class ReceiverNN(nn.Module):
    """RX NN on windows of the RX samples y[n], two samples per symbol.

    BPAM: two networks. The amplitude network sees the window centred on pulse k; the sign
    network sees the window shifted by one sample, centred on the crossing of pulses k-1 and
    k, where the sign change b[k] is observable. PAM-4: one network with two outputs on the
    centred window. Outputs are logits: z = sigmoid(logit) estimates P(bit = 1).
    """

    def __init__(self, fmt, p=LinkParameters()):
        super().__init__()
        self.fmt, self.window, self.sps = fmt, p.rx_window, p.adc_sps
        if fmt == 'bpam' and self.sps != 2:
            raise ValueError('BPAM needs two RX samples per symbol: the sign change is observable only between pulses.')
        if fmt == 'bpam':
            self.amplitude = Trunk(p.rx_window, p.rx_hidden, 1)
            self.sign = Trunk(p.rx_window, p.rx_hidden, 1)
        else:
            self.bits = Trunk(p.rx_window, p.rx_hidden, 2)

    def windows(self, y, offset=0):
        """(blocks, sps N) -> (blocks, N, window): one window per symbol, centred on sample sps k + offset
        (sps = adc_sps: two samples per symbol, or one with rx_sps=1)."""
        left = (self.window-1)//2-offset
        padded = F.pad(y.unsqueeze(1), (left, self.window-self.sps-left))
        return padded.unfold(2, self.window, self.sps).squeeze(1)

    def forward(self, y):
        """(blocks, 2N) RX samples -> (blocks, N, 2) bit logits."""
        if self.fmt == 'bpam':
            return torch.cat([self.amplitude(self.windows(y, 0)), self.sign(self.windows(y, -1))], dim=-1)
        return self.bits(self.windows(y, 0))


def decision_loss(logits, bits, guard):
    """Binary cross-entropy between logits and transmitted bits, block edges excluded."""
    return F.binary_cross_entropy_with_logits(logits[:, guard:-guard], bits.transpose(1, 2)[:, guard:-guard].float())


def decisions(logits):
    """(blocks, N, 2) logits -> (blocks, 2, N) bits: 1 when sigmoid(logit) >= 1/2."""
    return (torch.sigmoid(logits) >= .5).transpose(1, 2).long()

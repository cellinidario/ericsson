"""Frozen, exhaustive lookup replacement for a finite-context neural TX."""

from copy import deepcopy
import torch
import torch.nn.functional as F
from transmitter import Transmitter


class LookupTransmitter(Transmitter):
    """Keep the original waveform chain; replace only the neural mapping.

    Addresses are MSB-first, bit-stream-major: all times of bit stream 0,
    then all times of bit stream 1. Entries are volts BEFORE filtering/DAC.
    This first implementation deliberately excludes differential precoding.
    """

    @classmethod
    @torch.no_grad()
    def from_transmitter(cls, source):
        if source.equalizer != "end-to-end" or source.precode_e2e:
            raise ValueError("Expected an end-to-end TX without differential precoding")
        if source.memory % 2 != 1:
            raise ValueError("Expected an odd symbol context")
        width = source.memory * source.num_bits
        if width > 16:
            raise ValueError("Exhaustive export is limited to 16 input bits")
        device = source.context_layer.weight.device
        shifts = torch.arange(width - 1, -1, -1, device=device)
        contexts = ((torch.arange(2**width, device=device)[:, None] >> shifts) & 1)
        hidden = F.leaky_relu(source.context_layer(contexts.float()))
        for layer in source.extra_layers:
            hidden = F.leaky_relu(layer(hidden))
        raw = source.segment_layer(hidden)
        bounded = torch.tanh(raw) if source.output_activation == "tanh" else F.hardtanh(raw)
        table = source.drive_min + (source.drive_max - source.drive_min) * 0.5 * (bounded + 1)
        # Preserve all scalar settings and filter buffers exactly, without
        # reconstructing the channel or changing any source/cache fingerprints.
        result = deepcopy(source)
        result.__class__ = cls
        del result.context_layer, result.extra_layers, result.segment_layer
        result.register_buffer("table", table.detach().clone())
        result.register_buffer("address_weights", 2**shifts)
        return result.eval()

    def symbol_drive_levels(self, bits):
        windows = F.pad(bits, (self.memory // 2,) * 2).unfold(1, self.memory, 1)
        flat = windows.permute(1, 0, 2).reshape(bits.shape[1], -1).long()
        addresses = (flat * self.address_weights).sum(-1)
        levels = self.table[addresses].view(-1, self.num_segments, self.tx_subsymbols)
        return levels.permute(1, 0, 2).reshape(self.num_segments, -1)


class TrainableLookupTransmitter(LookupTransmitter):
    """Dense embedding of drive voltages, optimized jointly with the receiver.

    Project the entries onto the original voltage interval after each optimizer
    step. Integer addresses are fixed; gradients update the selected rows.
    """

    @classmethod
    def from_lookup(cls, source):
        result = deepcopy(source)
        result.__class__ = cls
        table = result.table.detach().clone()
        del result.table
        result.embedding = torch.nn.Embedding.from_pretrained(table, freeze=False)
        return result

    def symbol_drive_levels(self, bits):
        windows = F.pad(bits, (self.memory // 2,) * 2).unfold(1, self.memory, 1)
        flat = windows.permute(1, 0, 2).reshape(bits.shape[1], -1).long()
        addresses = (flat * self.address_weights).sum(-1)
        levels = self.embedding(addresses).view(-1, self.num_segments, self.tx_subsymbols)
        return levels.permute(1, 0, 2).reshape(self.num_segments, -1)

    @torch.no_grad()
    def project_voltages(self):
        self.embedding.weight.clamp_(self.drive_min, self.drive_max)

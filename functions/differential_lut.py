"""Differential-state LUT initialized exactly from a raw-bit LUT."""
from copy import deepcopy
import torch
import torch.nn.functional as F
from tx_lut import LookupTransmitter, TrainableLookupTransmitter


class DifferentialLookupTransmitter(TrainableLookupTransmitter):
    """Address [b0[k-2:k+3], c1[k-3:k+3]], with c1[k]=c1[k-1] XOR b1[k].

    Inputs and RX targets remain raw bits. Precoding is internal, not the legacy
    TX precoder flag. State starts at zero per input block. For zero raw-bit
    padding, the state extends as zero on the left and the last state on the
    right. Zero-padding the right state would introduce a spurious phase bit.
    """

    @classmethod
    def from_raw_lookup(cls, source):
        if source.num_bits != 2 or source.memory != 5 or source.precode_e2e:
            raise ValueError("Expected the raw-bit two-stream, five-symbol LUT")
        if source.embedding.weight.shape != (1024, 2):
            raise ValueError("Expected a 1024 x 2 raw-bit table")
        result = deepcopy(source)
        result.__class__ = cls
        addresses = torch.arange(2048, device=source.embedding.weight.device)
        contexts = ((addresses[:, None] >> torch.arange(10, -1, -1, device=addresses.device)) & 1)
        raw_phase = contexts[:, 5:-1] ^ contexts[:, 6:]
        raw = torch.cat((contexts[:, :5], raw_phase), dim=1)
        source_rows = (raw * source.address_weights).sum(1)
        table = source.embedding.weight.detach()[source_rows].clone()
        result.embedding = torch.nn.Embedding.from_pretrained(table, freeze=False)
        result.address_weights = 2 ** torch.arange(10, -1, -1, device=addresses.device)
        result.register_buffer("initial_source_rows", source_rows)
        return result

    def symbol_drive_levels(self, bits):
        amplitude = F.pad(bits[0], (2, 2)).unfold(0, 5, 1)
        state = torch.cumsum(bits[1], dim=0).remainder(2)
        # Six consecutive states recover the five raw phase bits by XOR.
        extended = torch.cat((state.new_zeros(3), state, state[-1:].expand(2)))
        phase = extended.unfold(0, 6, 1)
        contexts = torch.cat((amplitude, phase), dim=1).long()
        addresses = (contexts * self.address_weights).sum(-1)
        levels = self.embedding(addresses).view(-1, self.num_segments, self.tx_subsymbols)
        return levels.permute(1, 0, 2).reshape(self.num_segments, -1)


def load_raw_lut(path, device):
    """Load a completed bpam_lut checkpoint and its exact frozen digital surrogate."""
    from config import Config
    from transmitter import Transmitter
    from receiver import Receiver
    from simulated_surrogate import file_digest, frozen_channel
    from pathlib import Path
    state = torch.load(path, map_location=device, weights_only=True)
    metadata = state["metadata"]
    cfg = Config()
    cfg.__dict__.update(metadata["config"])
    for name, digest in metadata["source_sha256"].items():
        if file_digest(Path(__file__).with_name(name + ".py")) != digest:
            raise RuntimeError(f"Source code differs from checkpoint: {name}")
    if file_digest(cfg.surrogate_checkpoint) != metadata["digital_surrogate_sha256"]:
        raise RuntimeError("Digital surrogate checkpoint digest mismatch")
    tx = TrainableLookupTransmitter.from_lookup(
        LookupTransmitter.from_transmitter(Transmitter(cfg).to(device)))
    tx.load_state_dict(state["tx"])
    rx = Receiver(cfg).to(device)
    rx.load_state_dict(state["rx"])
    channel = frozen_channel(cfg, cfg.surrogate_checkpoint, device)
    return tx.eval(), channel, rx.eval(), cfg

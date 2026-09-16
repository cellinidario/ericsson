"""Joint embedding-TX and neural-RX fine-tuning through a frozen digital surrogate."""
import torch
import torch.nn.functional as F
from transmitter import bits_to_symbols
from train import link_photocurrent
from tx_lut import TrainableLookupTransmitter


def train_lut(transmitter, channel, receiver, config, num_steps, ebn0_db, seed):
    """Update voltage entries and RX weights with the original CE+BCE objective.

    Adam uses the original learning rate, without a scheduler. After each step,
    project LUT entries onto the original pre-filter voltage range. Filtering,
    DAC/ADC quantization and their straight-through gradients are unchanged.
    """
    if not isinstance(transmitter, TrainableLookupTransmitter):
        raise TypeError("Expected a trainable embedding transmitter")
    if num_steps < 1 or config.bpam_precode_e2e or config.noise_regime != "ase":
        raise ValueError("Use positive steps, raw input bits and the ASE scenario")
    if any(p.requires_grad or p.grad is not None for p in channel.parameters()):
        raise ValueError("The digital surrogate must be frozen before joint training")
    channel_before = {k: v.detach().clone() for k, v in channel.state_dict().items()}
    transmitter.train()
    receiver.train()
    channel.eval()
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(list(transmitter.parameters()) + list(receiver.parameters()),
                                 lr=config.learning_rate)
    guard = config.edge_guard_symbols
    device = transmitter.embedding.weight.device
    history = []
    for step in range(1, num_steps + 1):
        bits = torch.randint(0, 2, (config.bits_per_symbol,
                                   config.minibatch_symbols + 2 * guard), device=device)
        current = link_photocurrent(transmitter, channel, bits, ebn0_db, config, receiver)
        bit_logits, symbol_logits = receiver.bit_and_symbol_logits(current)
        n = symbol_logits.shape[0]
        probabilities = torch.sigmoid(bit_logits[guard:n-guard]).T
        loss = F.binary_cross_entropy(probabilities.clamp(1e-6, 1-1e-6),
                                      bits[:, guard:n-guard].float())
        if getattr(config, "bpam_loss", "ce+bce") != "bce":
            loss = loss + F.cross_entropy(symbol_logits[guard:n-guard],
                                          bits_to_symbols(bits, config.bits_per_symbol)[guard:n-guard])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite joint-training loss")
        if step == 1:
            for module in (transmitter, receiver):
                if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()):
                    raise RuntimeError("Missing gradient to TX or RX")
        optimizer.step()
        transmitter.project_voltages()
        if step == 1 or step % max(1, num_steps // 10) == 0 or step == num_steps:
            entry = {"step": step, "loss": float(loss.detach())}
            history.append(entry)
            print(f"LUT + RX step {step}/{num_steps}: loss={entry['loss']:.6g}", flush=True)
    if any(not torch.equal(v, channel_before[k]) for k, v in channel.state_dict().items()):
        raise RuntimeError("Digital surrogate weights changed during joint training")
    if any(p.grad is not None for p in channel.parameters()):
        raise RuntimeError("Unexpected digital surrogate parameter gradients")
    transmitter.eval()
    receiver.eval()
    return history

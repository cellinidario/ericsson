"""Training stages of the TX and RX DSP through a frozen digital surrogate.

Modes of a stage (numbers of updates in parameters.Budget):
    'rx'     RX NN trained with a fixed TX (RX initialization, RX-only reference)
    'tx'     TX NN fitted to the conventional voltages (mean-square error)
    'joint'  TX (NN or lookup table) and RX NN trained together, end to end
Every update draws new bits and new noise at the fixed training level. Loss: binary
cross-entropy on the bit logits. Adam, a fresh optimizer per stage; the entries of a TX
lookup table are projected onto the allowed voltage range after every update. A stage
resumes exactly (weights, optimizer, random generators) from its checkpoint.

On the GPU the update (bits, link, loss, backward, Adam step) is recorded once as a CUDA
graph and replayed: an update is a few hundred small kernels, and launching them one by one
from Python costs about ten times more than running them. The first update of a stage, or of
a resumed stage, runs normally and creates the gradients and the optimizer state. Values
move from the GPU to the CPU only at checkpoints, where the finiteness of loss and weights is
checked.
"""
import json
from pathlib import Path
import time

import torch

from .dsp import LookupTransmitter, decision_loss
from .files import save_json, save_torch
from .parameters import identity


def train_stage(mode, tx, rx, surrogate, path, steps, seed, regime, level_db, symbols=8448, learning_rate=1e-3,
                checkpoint_every=1000, print_every=12500, title=None, provenance=None):
    """Run (or resume) one stage and return its checkpoint path. mode: 'rx', 'tx' or 'joint'."""
    if mode not in ('rx', 'tx', 'joint'):
        raise ValueError("mode must be 'rx', 'tx' or 'joint'.")
    if any(q.requires_grad for q in surrogate.parameters()):
        raise RuntimeError('Freeze the digital surrogate before training the DSPs.')
    device = surrogate.pulse.device
    graphed = device.type == 'cuda'
    weights = list(rx.parameters()) if mode == 'rx' else list(tx.parameters())
    if mode == 'joint':
        weights += list(rx.parameters())
    optimizer = torch.optim.Adam(weights, lr=learning_rate, capturable=graphed)
    signature = json.loads(json.dumps(dict(
        mode=mode, steps=steps, seed=seed, regime=regime, level_db=level_db, symbols=symbols, learning_rate=learning_rate,
        link=identity(surrogate.p), tx_bandwidth=surrogate.tx_bandwidth, analog_sps=surrogate.sps,
        tx=type(tx).__name__, fmt=tx.fmt, precoder=tx.use_precoder, provenance=provenance)))
    path = Path(path)
    torch.manual_seed(seed)
    start, history = 0, []
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved['signature'] != signature:
            raise RuntimeError(f'{path} belongs to a different stage or configuration.')
        tx.load_state_dict(saved['tx'])
        rx.load_state_dict(saved['rx'])
        optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups:      # checkpoints written without CUDA graphs
            group['capturable'] = graphed
        if graphed:
            for state in optimizer.state.values():
                state['step'] = state['step'].to(device)
        torch.set_rng_state(saved['cpu_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        start, history = saved['step'], saved['history']
    guard = surrogate.p.guard_symbols
    if print_every and title:
        print(f"=== {title} ===" + ("  (already trained)" if start >= steps else f"  (resumed at step {start})" if start else ""),
              flush=True)
    tx.train(); rx.train()
    started, finite = time.time(), torch.ones((), dtype=torch.bool, device=device)
    projected = isinstance(tx, LookupTransmitter) and mode != 'rx'

    def update():
        bits = torch.randint(0, 2, (2, symbols), device=device)[None]
        if mode == 'tx':
            loss = tx.pretraining_loss(bits)
        else:
            with torch.set_grad_enabled(mode == 'joint'):
                y = surrogate(tx(bits), regime, level_db)
            loss = decision_loss(rx(y), bits, guard)
        finite.logical_and_(torch.isfinite(loss))
        optimizer.zero_grad(set_to_none=False)
        loss.backward()
        optimizer.step()
        if projected:
            tx.project()
        return loss

    graph = None
    for step in range(start, steps):
        if graph is not None:
            graph.replay()
        elif graphed:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                loss = update()
            torch.cuda.current_stream().wait_stream(side)
            if step+1 < steps:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    loss = update()               # recorded, not executed
        else:
            loss = update()
        if (step+1) % checkpoint_every == 0 or step+1 == steps:
            if not bool(finite) or not all(torch.isfinite(q).all() for q in weights):
                raise RuntimeError(f'Nonfinite loss or weights in stage {mode!r} before update {step+1}.')
            entry = dict(stage=mode, step=step+1, steps=steps, loss=float(loss.detach()),
                         updates_per_second=(step+1-start)/max(time.time()-started, 1e-3), updated_unix=time.time())
            history.append(entry)
            save_json(path.with_name(path.stem+'_progress.json'), entry)
            save_torch(path, dict(tx=tx.state_dict(), rx=rx.state_dict(), optimizer=optimizer.state_dict(),
                                  step=step+1, signature=signature, history=history, cpu_rng=torch.get_rng_state(),
                                  cuda_rng=torch.cuda.get_rng_state() if device.type == 'cuda' else torch.get_rng_state()))
            if print_every and ((step+1) % print_every == 0 or step+1 == steps):
                print(f'  step {step+1:6d}  loss {entry["loss"]:.8f}', flush=True)
    del graph
    tx.eval(); rx.eval()
    return path


def load_stage(path, tx, rx):
    """Load the weights saved by train_stage."""
    saved = torch.load(path, map_location=next(iter(rx.parameters())).device, weights_only=True)
    tx.load_state_dict(saved['tx'])
    rx.load_state_dict(saved['rx'])
    return saved

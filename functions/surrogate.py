"""Digital-surrogate machinery (Dario's directive, 2026-07-20).

The exact differentiable chain (`channel.OpticalChannel`) and a data-trained digital surrogate
are not alternatives but the same object in two regimes: OpticalChannel is already a
physics-structured network (Conv1d = driver/optical/PD FIR filters, analytic MZM, |.|^2 = PD)
whose taps are nn.Parameters. In the simulation results the taps are FIXED from the specs; when
measured input/output sequences of the experimental system are available (Stella's data), the
SAME structure is trained by MSE regression on them — the recipe of the reference script
MLforCPO/autoencoderJointBPAM.m (white-noise excitation, Adam+MSE, LR x0.9 schedule,
validation by recovering the true FIR taps) — then frozen and used as the E2E channel.

Swap point: `build_channel(config)` — used by train.train(). config.channel_source:
  "physics"   (default): OpticalChannel with spec-fixed parameters, exactly as before.
  "surrogate": OpticalChannel loaded from config.surrogate_checkpoint (a state_dict produced
               by fit_surrogate) and frozen. Drop-in: same forward(drive, ase_ebn0_db).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

from channel import OpticalChannel

FIR_BLOCKS = ("segment_filter", "optical_filter", "pd_filter")   # the trainable responses


def build_channel(config):
    """Channel factory for train()/notebooks. Returns a module with the OpticalChannel
    interface: forward(drive, ase_ebn0_db=None) -> photocurrent."""
    source = getattr(config, "channel_source", "physics")
    channel = OpticalChannel(config)
    if source == "physics":
        return channel
    if source == "surrogate":
        path = getattr(config, "surrogate_checkpoint", None)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"channel_source='surrogate' but surrogate_checkpoint not found: {path!r}. "
                "Train one with surrogate.fit_surrogate and save its state_dict first.")
        state = torch.load(path, map_location="cpu", weights_only=True)
        channel.load_state_dict(state)
        for p in channel.parameters():                 # frozen: the E2E must not retune physics
            p.requires_grad_(False)
        return channel
    raise ValueError(f"unknown channel_source {source!r} (use 'physics' or 'surrogate')")


def trainable_parameters(channel, blocks=FIR_BLOCKS):
    """The surrogate parameters to fit: the FIR taps of the requested blocks. Gains are
    absorbed by the taps themselves (pd_filter is linear after the square law), so no extra
    scalars are needed — same spirit as the Re/Im 1x1 weights of autoencoderJointBPAM.m."""
    params = []
    for name in blocks:
        module = getattr(channel, name, None)
        if module is None:
            continue
        params.extend(module.parameters())
    return params


def white_noise_drive(config, num_symbols, device="cpu", scale=None):
    """Format-agnostic training excitation (the .m recipe): white Gaussian noise per MZM
    segment at the simulation rate, spanning the drive range."""
    n = num_symbols * config.samples_per_symbol_sim
    lo, hi = config.drive_min_volt, config.drive_max_volt
    std = (hi - lo) / 4 if scale is None else scale
    mid = 0.5 * (hi + lo)
    return (mid + std * torch.randn(config.num_mzm_segments, n, device=device)).clamp(lo, hi)


def fit_surrogate(channel, drive, target, epochs=200, lr=0.025, lr_drop=0.9, lr_period=25,
                  val_fraction=0.2, blocks=FIR_BLOCKS, verbose=True):
    """MSE regression of the surrogate response on (drive -> photocurrent) pairs, mirroring
    the reference recipe (Adam, MSE, piecewise LR x0.9). `drive`: (num_segments, N) at the sim
    rate; `target`: (N,) measured/reference photocurrent for the same drive (noiseless or
    averaged). Returns the loss history dict."""
    n_val = int(drive.shape[-1] * val_fraction)
    d_tr, d_va = drive[:, :-n_val], drive[:, -n_val:]
    y_tr, y_va = target[:-n_val], target[-n_val:]
    params = trainable_parameters(channel, blocks)
    if not params:
        raise ValueError("no trainable blocks found on this channel")
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_period, gamma=lr_drop)
    history = {"train": [], "val": []}
    for epoch in range(epochs):
        opt.zero_grad()
        loss = torch.mean((channel(d_tr) - y_tr) ** 2)
        loss.backward()
        opt.step(); sched.step()
        with torch.no_grad():
            val = torch.mean((channel(d_va) - y_va) ** 2)
        history["train"].append(float(loss)); history["val"].append(float(val))
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  train MSE {loss:.3e}  val MSE {val:.3e}", flush=True)
    return history


def compare_taps(fitted, reference, blocks=FIR_BLOCKS):
    """Physics-level validation of the .m script (its figure 8): learned taps vs the true
    ones of a reference channel. Returns {block: (fitted_taps, reference_taps, max_abs_err)}."""
    out = {}
    for name in blocks:
        mf, mr = getattr(fitted, name, None), getattr(reference, name, None)
        if mf is None or mr is None:
            continue
        wf = mf.weight.detach().cpu().numpy().squeeze()
        wr = mr.weight.detach().cpu().numpy().squeeze()
        out[name] = (wf, wr, float(np.max(np.abs(wf - wr))))
    return out


def load_measured_io(path):
    """Loader for the experimental I/O sequences (Stella). Accepts .npz with arrays
    'drive' (num_segments, N) and 'photocurrent' (N,), or .mat with the same variable names.
    TODO when the real data arrives: confirm sample rate (resample to config.sim_sample_rate),
    TX/RX alignment (cross-correlate and shift), normalization, and whether the recorded
    output is pre or post the RX analog front end."""
    if path.endswith(".npz"):
        data = np.load(path)
        return (torch.tensor(data["drive"], dtype=torch.float32),
                torch.tensor(data["photocurrent"], dtype=torch.float32))
    if path.endswith(".mat"):
        from scipy.io import loadmat
        data = loadmat(path)
        return (torch.tensor(np.atleast_2d(data["drive"]), dtype=torch.float32),
                torch.tensor(data["photocurrent"].ravel(), dtype=torch.float32))
    raise ValueError(f"unsupported data file: {path}")

"""Fit/cache the validated hybrid simulator digital surrogate for interactive notebooks.

Only FIR responses are identified; MZM, dispersion, loss and ASE laws stay known.
This is not the experimental-data digital surrogate. Training targets are noiseless.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch

from channel import OpticalChannel
from surrogate import build_channel, trainable_parameters

RECIPE = {"steps": 5000, "learning_rate": .001, "train_blocks": 16,
          "validation_blocks": 4, "test_blocks": 4, "block_samples": 8192,
          "guard": 256, "seed": 401, "version": 1}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(cfg):
    """Conservatively invalidate cached fits when configuration or model code changes."""
    values = {k: v for k, v in vars(cfg).items()
              if k not in ("channel_source", "surrogate_checkpoint")}
    content = {"config": values, "recipe": RECIPE,
               "channel_source_sha256": file_digest(Path(__file__).with_name("channel.py"))}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(), content


def _dataset(cfg, reference, count, seed, device):
    generator = torch.Generator().manual_seed(seed)
    result = []
    with torch.no_grad():
        for _ in range(count):
            x = torch.randn((cfg.num_mzm_segments, RECIPE["block_samples"]), generator=generator)
            x = x * (cfg.drive_max_volt - cfg.drive_min_volt) / 4
            x += (cfg.drive_max_volt + cfg.drive_min_volt) / 2
            x = x.clamp(cfg.drive_min_volt, cfg.drive_max_volt).to(device)
            result.append((x, reference(x).detach()))
    return result


def _nmse(model, reference, data, noisy=False):
    guard = RECIPE["guard"]
    error, energy = 0., 0.
    with torch.no_grad():
        for i, (x, target) in enumerate(data):
            if noisy:
                torch.manual_seed(81000 + i)
                target = reference(x, ase_ebn0_db=14.)
                torch.manual_seed(81000 + i)
            prediction = model(x, ase_ebn0_db=14. if noisy else None)
            y, yp = target[guard:-guard], prediction[guard:-guard]
            error += (yp-y).double().square().sum().item()
            energy += (y-y.mean()).double().square().sum().item()
    return error / energy


def _gradient(model, reference, x):
    guard = RECIPE["guard"]
    a = x.detach().clone().requires_grad_()
    b = x.detach().clone().requires_grad_()
    ya, yb = reference(a)[guard:-guard], model(b)[guard:-guard]
    projection = torch.randn(ya.shape, generator=torch.Generator().manual_seed(9931)).to(x.device)
    ga = torch.autograd.grad((ya*projection).sum(), a)[0][..., guard:-guard].flatten()
    gb = torch.autograd.grad((yb*projection).sum(), b)[0][..., guard:-guard].flatten()
    return {"cosine": torch.nn.functional.cosine_similarity(ga, gb, dim=0).item(),
            "relative_l2": ((gb-ga).norm()/ga.norm()).item(),
            "input_gradient_norm": gb.norm().item()}


def prepare_digital_surrogate(cfg, directory, device, force_retrain=False):
    """Train once or load a validated checkpoint; use the notebook's FORCE_RETRAIN.

    The returned digest must be stored with the TX/RX checkpoints, preventing
    their reuse with a different channel. No separate force flag is introduced.
    """
    key, provenance = signature(cfg)
    folder = Path(directory) / key[:16]
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint, metadata = folder / "surrogate.pt", folder / "validation.json"
    if not force_retrain and checkpoint.exists() and metadata.exists():
        report = json.loads(metadata.read_text(encoding="utf-8"))
        if (report["signature"] != key or not report["gate_passed"]
                or report["checkpoint_sha256"] != file_digest(checkpoint)):
            raise RuntimeError("Invalid digital surrogate cache; inspect it or set FORCE_RETRAIN=True.")
        print(f"Loaded validated digital surrogate: {checkpoint}")
        return {"checkpoint": str(checkpoint), "sha256": report["checkpoint_sha256"],
                "report": report, "retrained": False}

    dev = torch.device(device)
    devices = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(RECIPE["seed"])
        reference = OpticalChannel(cfg).to(device).requires_grad_(False)
        model = OpticalChannel(cfg).to(device)
        with torch.no_grad():
            for name in ("segment_filter", "optical_filter", "pd_filter"):
                block = getattr(model, name)
                if block is not None:
                    block.weight.zero_()
                    block.weight[..., block.weight.shape[-1]//2] = 1.
        training = _dataset(cfg, reference, RECIPE["train_blocks"], 11001, device)
        validation = _dataset(cfg, reference, RECIPE["validation_blocks"], 22001, device)
        optimizer = torch.optim.Adam(trainable_parameters(model), lr=RECIPE["learning_rate"])
        guard, best, history = RECIPE["guard"], float("inf"), []
        best_state = None
        for step in range(1, RECIPE["steps"]+1):
            x, target = training[(step-1) % len(training)]
            y = target[guard:-guard]
            prediction = model(x)[guard:-guard]
            loss = (prediction-y).square().mean()/y.var(unbiased=False)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == RECIPE["steps"]:
                value = _nmse(model, reference, validation)
                history.append({"step": step, "validation_nmse": value})
                if value < best:
                    best, best_step = value, step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if step % 500 == 0 or step == RECIPE["steps"]:
                    print(f"Digital surrogate step {step}: validation NMSE={value:.3e}", flush=True)
        model.load_state_dict(best_state)
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        test = _dataset(cfg, reference, RECIPE["test_blocks"], 33001, device)
        test_nmse = _nmse(model, reference, test)
        noise_nmse = _nmse(model, reference, test, noisy=True)
        gradients = [_gradient(model, reference, x) for x, _ in test]
        passed = (test_nmse <= .001 and noise_nmse <= .01
                  and all(g["cosine"] >= .99 and g["relative_l2"] <= .10
                          and g["input_gradient_norm"] > 0 for g in gradients)
                  and all(p.grad is None and not p.requires_grad for p in model.parameters()))
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(best_state, temporary)
        temporary.replace(checkpoint)
        report = {"signature": key, "provenance": provenance, "best_step": best_step,
                  "test_nmse": test_nmse, "paired_noise_nmse": noise_nmse,
                  "gradients": gradients, "gate_passed": passed, "history": history,
                  "checkpoint_sha256": file_digest(checkpoint)}
        temporary = metadata.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(metadata)
        if not passed:
            raise RuntimeError(f"Digital surrogate validation failed; TX/RX training stopped. See {metadata}")
    print(f"Digital surrogate validated and saved: {checkpoint}")
    return {"checkpoint": str(checkpoint), "sha256": report["checkpoint_sha256"],
            "report": report, "retrained": True}


def frozen_channel(cfg, checkpoint, device):
    """Use the same factory as joint training, including the checkpoint-loading path."""
    local = deepcopy(cfg)
    local.channel_source = "surrogate"
    local.surrogate_checkpoint = str(checkpoint)
    return build_channel(local).to(device).eval()

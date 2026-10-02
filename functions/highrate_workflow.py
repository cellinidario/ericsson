"""Reproducible high-rate digital-surrogate, neural-TX and LUT workflow.

The DAC and DSP stay at two samples/symbol. Only the reconstructed waveform,
optical propagation and receiver front end use the higher simulation rate.
Historical checkpoints are read-only, hash-checked optional starting points.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from channel import OpticalChannel, supergaussian_fir
from receiver import Receiver
from scenarios import asfand_complex, bpam_classic_rx, OSNR_OFFSET_DB
from surrogate import trainable_parameters
from transmitter import Transmitter, bits_to_symbols

RECIPE = dict(version=1, fit_steps=40000, fir_l2=.001, rx_steps=100000,
              tx_steps=8000, joint_steps=100000, continuation_steps=20000,
              learning_rate=.001, training_ebn0=14., analog_sps=8,
              dac_sps=2, rx_sps=2, bandwidth_hz=20e9, converter_bits=6)

# Reviewed model code behind the October 1 validation. Refuse silent reuse if
# any of these files changes; a new run is then needed for the changed model.
VALIDATED_CODE = {
    'channel.py': '07934158bac3377e018e2874bfd434d00475d6a45bb376d2badddbf56e16cc51',
    'transmitter.py': '1d374a6b949e73243ded95e3d15f2b68219ec081d7001cc573ac7618efd23416',
    'receiver.py': '908e5a3e234b9e4171265e21b3db0a3b177493ddeb1bd142959c93887bdb5c45',
    'config.py': '7abc1aec5b65512e3ce0be1f993d5eaa5c35fd5bc73e5624575b5e67dc4e0867',
    'pulse_shaping.py': 'b0cfc99cbb2dcef65ea155cd546b267a043f42d81eb2d2a482e2f6b790c88e47',
    'utils.py': 'ce746533a07033c13a63c1dfa2eb2c1694a04b13dc0ca211153d2bc1bacebbc0',
    'scenarios.py': '318bfc1da04678f88f83ed3f25351f19a600ffd1e4fae19b4c320c3732940e4b',
}
VALIDATED_FILES = {
    'digital_surrogate': ('sim_surrogate/20261001_ASE_ridge_fresh40k/digital_surrogate_best.pt',
        '84b2d464383730731b5c81ba7016b5f39e2c35b74798ee13619b8e5682a3a9ca'),
    'initial0': ('ofc_validation/20260923/warm_bw20_seed0/joint_initial.pt',
        '7dba9f1f9fd132bbc83e11aa87498a55d6e5d75d4c4f9edd2afd8da9cd356409'),
    'initial1': ('ofc_validation/20260923/warm_bw20_seed1/joint_initial.pt',
        'e8612907a2723cff89e9d744f4b3f0e9e1139ce66476f003c19bceac01f6eee1'),
    'joint0': ('sim_surrogate/20261001_joint_transfer_100k/digital_surrogate/last_training.pt',
        'c342b630afa44a886af620c5e52fb149d270eb777246393888d879d1eb2bf8f2'),
    'joint1': ('sim_surrogate/20261001_lut_replication_seed1/digital_surrogate/last_training.pt',
        'f883352429a8538718d3f88e131e4cf1a84458cc7b2ac280c98599826f556b70'),
    'nn0': ('sim_surrogate/20261001_lut_transfer_20k/nn/last_training.pt',
        '55ab9d4d36e7d12347e2632c18cbc09c30ed35e17272e293db36912fcf8471c7'),
    'lut0': ('sim_surrogate/20261001_lut_transfer_20k/lut/last_training.pt',
        'f83b9775c248268ce0e6ba0ec49e5acb37eefde45d5cc8eb9ce74869bc3709a4'),
    'nn1': ('sim_surrogate/20261001_lut_replication_seed1/continuation/nn/last_training.pt',
        '0f87a0a6d4e0e0ea5853df9d5380a9dd0e25e1b517702a797c3b8169b9376405'),
    'lut1': ('sim_surrogate/20261001_lut_replication_seed1/continuation/lut/last_training.pt',
        '0de394973fd1709043c8cd289217fb6d77f5b420867538cf7918d1e12937778c'),
}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    for attempt in range(10):
        try:
            temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05 * (attempt + 1))


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def validated_artifacts(root, seed, force_retrain=False):
    """Return available exact historical artifacts, never silently changed ones."""
    if seed not in (0, 1):
        raise ValueError('This recipe has validated initializations 0 and 1 only')
    if force_retrain:
        return {}
    root = Path(root)
    for name, expected in VALIDATED_CODE.items():
        if file_digest(root/'functions'/name) != expected:
            raise RuntimeError(f'{name} changed: historical reuse is disabled. '
                               'Review the change and use FORCE_RETRAIN=True.')
    result = {}
    for key in ('digital_surrogate', f'initial{seed}', f'joint{seed}', f'nn{seed}', f'lut{seed}'):
        relative, expected = VALIDATED_FILES[key]
        path = root/'results'/relative
        if path.exists():
            if file_digest(path) != expected:
                raise RuntimeError(f'Historical checkpoint hash mismatch: {path}')
            result[key.rstrip('01')] = path
    return result


class BoundedTransmitter(Transmitter):
    def project_voltages(self):
        """The standard tanh output already bounds the neural drive."""


def build_system(sps=8, seed=0, fixed=False, device=None):
    """Build the validated chain without changing DAC or RX sampling rates."""
    if sps not in (8, 16, 32):
        raise ValueError('Use 8 sps for training, 16 for checks or 32 for eyes')
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = asfand_complex(adc_bits=6, dac_bits=6, precode=True, rx='asfand', sps=2)
    cfg.tx_filter, cfg.nrz_rect_cutoff_ratio = 'nrz', 1.
    cfg.equalizer = 'ffe' if fixed else 'end-to-end'
    cfg._compute_derived()
    cfg.set_modulation_format('bpam-4')
    torch.manual_seed(seed)
    tx = BoundedTransmitter(deepcopy(cfg)).to(device)
    cfg.samples_per_symbol_sim = sps
    cfg._compute_derived()
    taps = 32*sps + 1
    channel = OpticalChannel(cfg, segment_filter_taps=taps, pd_filter_taps=taps,
                             optical_filter_taps=taps).to(device).eval().requires_grad_(False)
    rx = Receiver(cfg).to(device)
    gaussian = supergaussian_fir(cfg.rx_gaussian_bw, 1, cfg.sim_sample_rate, taps)
    rx.rx_gauss = torch.tensor(gaussian[::-1].copy(), device=device,
                              dtype=torch.float32).view(1, 1, -1)
    return tx, channel, rx, cfg


def reconstruct(drive, factor, bandwidth=20e9, rate=40e9):
    """FFT reconstruction after DAC, including the original Nyquist-bin split."""
    n = drive.shape[-1]
    spectrum = torch.fft.rfft(drive)
    frequency = torch.fft.rfftfreq(n, d=1/rate, device=drive.device)
    spectrum = spectrum * (frequency <= bandwidth)
    if factor > 1 and n % 2 == 0:
        spectrum = torch.cat((spectrum[..., :-1], spectrum[..., -1:] * .5), dim=-1)
    return torch.fft.irfft(spectrum, n=n*factor) * factor


def link(tx, channel, cfg, bits, ebn0=14.):
    return channel(reconstruct(tx(bits), cfg.samples_per_symbol_sim//2), ase_ebn0_db=ebn0)


def load_pair(tx, rx, path):
    """Restore DSP parameters, retaining sample-rate-specific RX buffers."""
    state = torch.load(path, map_location=tx.rrc.device, weights_only=True)
    tx.load_state_dict(state['tx'])
    with torch.no_grad():
        for name, parameter in rx.named_parameters():
            parameter.copy_(state['rx'][name])
    return state


class CodedLookup(BoundedTransmitter):
    """1024x2 voltage embedding; cumulative differential coding stays outside."""
    def symbol_drive_levels(self, bits):
        coded = self._precode_bpam(bits) if self.precode_e2e else bits
        windows = F.pad(coded, (2, 2)).unfold(1, 5, 1)
        flat = windows.permute(1, 0, 2).reshape(bits.shape[1], 10).long()
        rows = (flat * self.address_weights).sum(-1)
        levels = self.embedding(rows).view(-1, self.num_segments, self.tx_subsymbols)
        return levels.permute(1, 0, 2).reshape(self.num_segments, -1)

    @torch.no_grad()
    def project_voltages(self):
        self.embedding.weight.clamp_(self.drive_min, self.drive_max)


@torch.no_grad()
def export_coded_lut(source):
    if (source.memory, source.num_bits, source.num_segments, source.tx_subsymbols) != (5, 2, 1, 2):
        raise ValueError('Expected five symbols, two bits, one segment and two outputs')
    weights = 2**torch.arange(9, -1, -1, device=source.rrc.device)
    contexts = ((torch.arange(1024, device=weights.device)[:, None] // weights) & 1).float()
    hidden = F.leaky_relu(source.context_layer(contexts))
    for layer in source.extra_layers:
        hidden = F.leaky_relu(layer(hidden))
    raw = source.segment_layer(hidden)
    bounded = torch.tanh(raw) if source.output_activation == 'tanh' else F.hardtanh(raw)
    table = source.drive_min + (source.drive_max-source.drive_min)*.5*(bounded+1)
    target = deepcopy(source)
    target.__class__ = CodedLookup
    del target.context_layer, target.extra_layers, target.segment_layer
    target.register_buffer('address_weights', weights)
    target.embedding = torch.nn.Embedding.from_pretrained(table, freeze=False)
    return target


@torch.no_grad()
def audit_export(source, lut):
    """Exhaustive local mapping plus waveform checks with the external precoder."""
    device = source.rrc.device
    contexts = ((torch.arange(1024, device=device)[:, None] >> torch.arange(9, -1, -1, device=device)) & 1).float()
    hidden = F.leaky_relu(source.context_layer(contexts))
    for layer in source.extra_layers:
        hidden = F.leaky_relu(layer(hidden))
    raw = source.segment_layer(hidden)
    bounded = torch.tanh(raw) if source.output_activation == 'tanh' else F.hardtanh(raw)
    table = source.drive_min + (source.drive_max-source.drive_min)*.5*(bounded+1)
    local_error = float((table-lut.embedding.weight).abs().max())
    generator = torch.Generator().manual_seed(6193)
    patterns = [torch.randint(0, 2, (2, 4096), generator=generator).to(device),
                torch.zeros((2, 64), device=device, dtype=torch.long),
                torch.ones((2, 64), device=device, dtype=torch.long),
                (torch.arange(128, device=device).repeat(2, 1) % 2)]
    checks = [dict(pre_DAC_max_error_V=float((source.symbol_drive_levels(b)-lut.symbol_drive_levels(b)).abs().max()),
                   post_DAC_max_error_V=float((source(b)-lut(b)).abs().max())) for b in patterns]
    if local_error > 1e-5 or any(r['pre_DAC_max_error_V'] > 1e-4 or r['post_DAC_max_error_V'] > 1e-4 for r in checks):
        raise RuntimeError('NN-to-LUT export failed its equivalence checks')
    return dict(rows=1024, samples_per_row=2, exhaustive_max_error_V=local_error,
                checks=checks, precoder='External cumulative XOR, unchanged')


def _dataset(cfg, reference, count, seed):
    generator = torch.Generator().manual_seed(seed)
    data = []
    with torch.no_grad():
        for _ in range(count):
            x = torch.randn((1, 8192), generator=generator)
            x = x * (cfg.drive_max_volt-cfg.drive_min_volt)/4
            x += (cfg.drive_max_volt+cfg.drive_min_volt)/2
            x = x.clamp(cfg.drive_min_volt, cfg.drive_max_volt).to(next(reference.parameters()).device)
            data.append((x, reference(x).detach()))
    return data


def _nmse(model, reference, data, noisy=False):
    error, energy = 0., 0.
    with torch.no_grad():
        for i, (x, target) in enumerate(data):
            if noisy:
                torch.manual_seed(81000+i)
                target = reference(x, ase_ebn0_db=14.)
                torch.manual_seed(81000+i)
            prediction = model(x, ase_ebn0_db=14. if noisy else None)
            y, yp = target[1024:-1024], prediction[1024:-1024]
            error += (yp-y).double().square().sum().item()
            energy += (y-y.mean()).double().square().sum().item()
    return error/energy


def validate_digital_surrogate(model, reference, cfg):
    """Held-out noiseless/paired-ASE regression and input-derivative checks."""
    model.eval().requires_grad_(False).zero_grad(set_to_none=True)
    test = _dataset(cfg, reference, 4, 33001)
    gradients = []
    for x, _ in test:
        a, b = x.detach().clone().requires_grad_(), x.detach().clone().requires_grad_()
        ya, yb = reference(a)[1024:-1024], model(b)[1024:-1024]
        projection = torch.randn(ya.shape, generator=torch.Generator().manual_seed(9931)).to(x.device)
        ga = torch.autograd.grad((ya*projection).sum(), a)[0][..., 1024:-1024].flatten()
        gb = torch.autograd.grad((yb*projection).sum(), b)[0][..., 1024:-1024].flatten()
        gradients.append(dict(cosine=float(F.cosine_similarity(ga, gb, dim=0)),
                              relative_l2=float((gb-ga).norm()/ga.norm())))
    report = dict(noiseless_nmse=_nmse(model, reference, test),
                  paired_ASE_nmse=_nmse(model, reference, test, noisy=True), input_gradients=gradients)
    report['passed'] = (report['noiseless_nmse'] <= .001 and report['paired_ASE_nmse'] <= .01
                        and all(g['cosine'] >= .99 and g['relative_l2'] <= .1 for g in gradients))
    if not report['passed']:
        raise RuntimeError(f'Digital surrogate validation failed: {report}')
    return report


def fit_digital_surrogate(reference, cfg, folder, historical=None, steps=40000):
    """Fit identity FIRs with normalized MSE plus L2; resume interrupted fits."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'digital_surrogate_best.pt'
    model = deepcopy(reference)
    if historical is not None or (folder/'complete.json').exists():
        selected = Path(historical) if historical is not None else path
        if historical is None:
            complete = json.loads((folder/'complete.json').read_text())
            if complete['steps'] != steps or complete['sha256'] != file_digest(path):
                raise RuntimeError('Incompatible digital surrogate cache')
        model.load_state_dict(torch.load(selected, map_location=next(reference.parameters()).device, weights_only=True))
        return model.eval().requires_grad_(False), selected, validate_digital_surrogate(model, reference, cfg)
    model.requires_grad_(True)
    with torch.no_grad():
        for name in ('segment_filter', 'optical_filter', 'pd_filter'):
            block = getattr(model, name)
            block.weight.zero_()
            block.weight[..., block.weight.shape[-1]//2] = 1.
    train_data = _dataset(cfg, reference, 16, 11001)
    valid_data = _dataset(cfg, reference, 4, 22001)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=.001)
    start, best, best_step = 0, float('inf'), 0
    resume = folder/'last_training.pt'
    if resume.exists():
        saved = torch.load(resume, map_location=next(reference.parameters()).device, weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start, best, best_step = saved['step'], saved['best'], saved['best_step']
    for step in range(start, steps):
        x, target = train_data[step % len(train_data)]
        y = target[1024:-1024]
        predicted = model(x)[1024:-1024]
        loss = (predicted-y).square().mean()/y.var(unbiased=False)
        loss += .001 * sum(p.square().sum() for p in trainable_parameters(model))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite digital surrogate loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1) % 100 == 0 or step+1 == steps:
            nmse = _nmse(model, reference, valid_data)
            criterion = nmse + .001*sum(float(p.detach().square().sum()) for p in trainable_parameters(model))
            if criterion < best:
                best, best_step = criterion, step+1
                save_checkpoint(path, model.state_dict())
            save_json(folder/'progress.json', dict(step=step+1, validation_nmse=nmse, objective=criterion))
        if (step+1) % 500 == 0 or step+1 == steps:
            save_checkpoint(resume, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                        step=step+1, best=best, best_step=best_step))
        if (step+1) % 1000 == 0 or step+1 == steps:
            print(f'Digital surrogate {step+1}/{steps}: validation NMSE={nmse:.6g}', flush=True)
    model.load_state_dict(torch.load(path, map_location=next(reference.parameters()).device, weights_only=True))
    report = validate_digital_surrogate(model, reference, cfg)
    save_json(folder/'complete.json', dict(steps=steps, best_step=best_step, sha256=file_digest(path), validation=report))
    return model, path, report


def train_pair(tx, rx, channel, cfg, path, steps, rng_seed, mode='joint', upstream=None, historical=None):
    """Fresh Adam per stage; resumable joint, RX-only or supervised-TX training."""
    if historical is not None:
        protocol = json.loads((Path(historical).parent.parent/'protocol.json').read_text())
        expected = dict(initial_sha256=protocol['initial_sha256'],
                        joint_sha256=protocol['initial_sha256'],
                        digital_sha256=protocol['digital_surrogate_sha256'])
        if any(value != expected.get(key) for key, value in (upstream or {}).items()):
            print('Upstream checkpoint changed; training instead of reusing the historical stage.')
            historical = None
    if historical is not None:
        state = load_pair(tx, rx, historical)
        if state.get('step', steps) != steps:
            raise RuntimeError('Historical training budget mismatch')
        print(f'Loaded {mode} checkpoint: {historical}')
        return Path(historical)
    if mode not in ('joint', 'rx', 'tx'):
        raise ValueError('Unknown training mode')
    if any(p.requires_grad or p.grad is not None for p in channel.parameters()):
        raise ValueError('Training channel must be frozen with no parameter gradients')
    signature = dict(steps=steps, rng_seed=rng_seed, mode=mode, upstream=upstream or {})
    parameters = list(tx.parameters()) if mode == 'tx' else list(rx.parameters())
    if mode == 'joint':
        parameters = list(tx.parameters()) + list(rx.parameters())
    optimizer = torch.optim.Adam(parameters, lr=.001)
    torch.manual_seed(rng_seed)
    path = Path(path)
    start, history = 0, []
    if path.exists():
        saved = torch.load(path, map_location=tx.rrc.device, weights_only=True)
        if saved['signature'] != signature:
            raise RuntimeError(f'Incompatible stage checkpoint: {path}')
        load_pair(tx, rx, path)
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['cpu_rng'].cpu())
        if tx.rrc.device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        start, history = saved['step'], saved['history']
    channel_before = {k: v.detach().clone() for k, v in channel.state_dict().items()}
    tx.train()
    rx.train()
    for step in range(start, steps):
        bits = torch.randint(0, 2, (2, 8448), device=tx.rrc.device)
        if mode == 'tx':
            coded = tx._precode_bpam(bits)
            level = tx.fixed_levels[(coded[0]*2+coded[1]).long()]
            target = torch.zeros((bits.shape[1], tx.tx_subsymbols), device=bits.device)
            target[:, 0] = level
            loss = F.mse_loss(tx.symbol_drive_levels(bits)[0], target.flatten())
        else:
            current = link(tx, channel, cfg, bits)
            logits, symbols = rx.bit_and_symbol_logits(current)
            loss = F.binary_cross_entropy(torch.sigmoid(logits[128:-128]).T.clamp(1e-6, 1-1e-6),
                                          bits[:, 128:-128].float())
            loss += F.cross_entropy(symbols[128:-128], bits_to_symbols(bits, 2)[128:-128])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite DSP loss')
        tx.zero_grad(set_to_none=True)
        rx.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        tx.project_voltages()
        if step == start or (step+1) % 1000 == 0 or step+1 == steps:
            entry = dict(mode=mode, step=step+1, loss=float(loss.detach()))
            history.append(entry)
            print(entry, flush=True)
            save_json(path.parent/(path.stem+'_progress.json'), entry)
        if (step+1) % 5000 == 0 or step+1 == steps:
            save_checkpoint(path, dict(tx=tx.state_dict(), rx=rx.state_dict(), optimizer=optimizer.state_dict(),
                step=step+1, signature=signature, history=history, cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state() if tx.rrc.device.type == 'cuda' else torch.get_rng_state()))
    if any(p.grad is not None for p in channel.parameters()) or any(
            not torch.equal(channel_before[k], v) for k, v in channel.state_dict().items()):
        raise RuntimeError('Digital surrogate changed during DSP training')
    tx.eval()
    rx.eval()
    return path


@torch.no_grad()
def evaluate_pairs(pairs, reference, cfg, blocks_by_osnr, path, upstream):
    """Fixed-budget shared test bits/ASE; no BER-based stopping or selection."""
    path = Path(path)
    signature = dict(blocks_by_osnr=blocks_by_osnr, upstream=upstream,
                     symbols_per_block=32768, guard_symbols=128, bit_seed=20000000, ASE_seed=30000000)
    points = {}
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['signature'] != json.loads(json.dumps(signature)):
            raise RuntimeError('Evaluation cache belongs to different models or test budget')
        points = saved['points']
    for osnr, blocks in blocks_by_osnr.items():
        key = str(osnr)
        point = points.setdefault(key, dict(blocks_done=0, models={n: dict(errors=0, bits=0) for n in pairs}))
        for block in range(point['blocks_done'], blocks):
            torch.manual_seed(20000000+int(osnr)*10000+block)
            bits = torch.randint(0, 2, (2, 32768), device=next(iter(pairs.values()))[0].rrc.device)
            for name, (tx, rx) in pairs.items():
                tx.eval()
                rx.eval()
                torch.manual_seed(30000000+int(osnr)*10000+block)
                current = link(tx, reference, cfg, bits, float(osnr)-OSNR_OFFSET_DB)
                decoded = rx.bit_posteriors_direct(current)[:, 128:-128] >= .5
                errors = decoded != bits[:, 128:-128]
                row = point['models'][name]
                row['errors'] += int(errors.sum())
                row['bits'] += errors.numel()
            point['blocks_done'] = block+1
            if (block+1) % 16 == 0 or block+1 == blocks:
                save_json(path, dict(signature=signature, points=points))
        for row in point['models'].values():
            row.update(ber=row['errors']/row['bits'], adequate_errors=row['errors'] >= 200)
        save_json(path, dict(signature=signature, points=points))
        print(f'OSNR {osnr}: '+', '.join(f"{n}={r['ber']:.6g} ({r['errors']} errors)" for n, r in point['models'].items()), flush=True)
    return points


@torch.no_grad()
def sampling_check(tx, rx, reference, cfg):
    """Noiseless RX-window NMSE at 8 versus 16 sps with the same DSP weights."""
    _, ref16, rx16, cfg16 = build_system(16, device=tx.rrc.device)
    for name, parameter in rx16.named_parameters():
        parameter.copy_(dict(rx.named_parameters())[name])
    torch.manual_seed(91841)
    bits = torch.randint(0, 2, (2, 4096), device=tx.rrc.device)
    a = rx.windows_from(link(tx, reference, cfg, bits, None))[0, 128:-128]
    b = rx16.windows_from(link(tx, ref16, cfg16, bits, None))[0, 128:-128]
    nmse = float((a-b).square().sum()/(b-b.mean()).square().sum())
    return dict(rx_window_nmse_8_16=nmse, passed=nmse <= .0001)


@torch.no_grad()
def plot_mzm_eyes(pairs, folder):
    """Noiseless native-32-sps Pre MZM / Re{E} / power, without renormalizing E."""
    import matplotlib.pyplot as plt
    folder = Path(folder)
    _, channel, _, _ = build_system(32, device=next(iter(pairs.values()))[0].rrc.device)
    cfg = bpam_classic_rx(adc_bits=6, dac_bits=6, vpeak=.6)
    cfg.samples_per_symbol_sim = 2
    cfg.nrz_rect_cutoff_ratio = 1.
    cfg._compute_derived()
    cfg.set_modulation_format('bpam-4')
    classic = BoundedTransmitter(cfg).to(channel.segment_filter.weight.device).eval()
    torch.manual_seed(40102026)
    bits = torch.randint(0, 2, (2, 8192), device=classic.rrc.device)
    models = [('BPAM', classic)] + [(name.upper(), tx) for name, (tx, _) in pairs.items()]
    fig, axes = plt.subplots(len(models), 3, figsize=(11, 2.65*len(models)), sharex=True, sharey='col')
    indices = np.arange(-32, 33)[:, None] + np.arange(256, 5256)*32
    for row, (label, tx) in enumerate(models):
        drive = channel.segment_filter(reconstruct(tx(bits), 16).unsqueeze(0)).sum(1, keepdim=True)
        phase = torch.pi/(2*channel.vpi)*(drive-channel.bias_volt)
        field = .5*(torch.exp(1j*phase)+channel.gamma*torch.exp(-1j*phase))
        if channel.chirp_alpha:
            field *= torch.exp(1j*channel.chirp_alpha*phase)
        values = dict(drive=drive.flatten().cpu().numpy(), field_real=field.real.flatten().cpu().numpy(),
                      field_imag=field.imag.flatten().cpu().numpy(), sps=32, vpi=channel.vpi)
        np.savez_compressed(folder/(label.lower()+'_eye32.npz'), **values)
        signals = [values['drive']/channel.vpi, values['field_real'], values['field_real']**2+values['field_imag']**2]
        for col, signal in enumerate(signals):
            ax = axes[row, col]
            ax.plot(np.arange(-32, 33)/32, signal[indices], color='#2463a6',
                    alpha=.03 if row == 0 else .05, lw=.4, rasterized=True)
            ax.set(xlim=(-1, 1), ylabel=[r'$V/V_\pi$', r'optical field Re{$E$}', r'optical power $|E|^2$'][col])
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(['Pre MZM', 'Post MZM', 'Post MZM (power)'][col])
            if row == len(models)-1:
                ax.set_xlabel('Symbol time')
        axes[row, 0].text(-.35, .5, label, rotation=90, transform=axes[row, 0].transAxes,
                          va='center', ha='center', fontsize=12)
    fig.tight_layout(rect=(.03, 0, 1, 1))
    fig.savefig(folder/'mzm_comparison.png', dpi=160)
    return fig

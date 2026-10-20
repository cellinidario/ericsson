"""One case end to end: digital surrogate, training, held-out BER, linear reference, eyes.

Results of a case go to <root>/cases/<case name>_<key>/ and its trained DSPs to
<root>/training/<case name>_<key>/; the keys are hashes of the case, the budget and (for
results) the evaluation grid. Changing any setting gives a new folder; an existing folder
is loaded instead of recomputed. Every stage is restartable.

Digital surrogates go to <root>/digital_surrogates/<key of the analog settings>/ and are
shared by all cases with the same analog link (the TX filter is digital and lies outside).

Methods:
    'linear'        least-squares linear equalizer, conventional TX
    'rx_only'       RX NN, conventional TX
    'e2e'           TX NN and RX NN trained jointly
    'nn_continued'  the E2E pair trained for Budget.continuation_steps more updates
    'lut'           the E2E TX NN exported to a lookup table; table and RX NN trained for the
                    same updates as 'nn_continued', on the same bits and noise
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import digital_surrogate, linear_equalizer as linear
from .dsp import ConventionalTransmitter, LookupTransmitter, ReceiverNN, TransmitterNN
from .evaluation import blocks_to_count, comparison, count_errors, held_out_bits, threshold_with_interval, wilson
from .files import read_json, save_json
from .link import Link
from .parameters import EVALUATION_LEVEL_DB, Budget, identity

SEEDS = dict(rx=9100000, tx=9200000, joint=9300000, rx_only=9300000,   # rx_only: same bits as joint
             initialization=9000000, continuation=9400000)
METHODS = ('linear', 'rx_only', 'e2e')
ALL_METHODS = METHODS+('nn_continued', 'lut')
NAMES = {'linear': 'linear equalizer', 'rx_only': 'RX-only NN', 'e2e': 'E2E',
         'nn_continued': 'E2E, TX NN continued', 'lut': 'E2E, TX lookup table continued'}
TRAINED = ('rx_only', 'e2e', 'nn_continued', 'lut')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:10]


def case_folder(root, case, budget, levels):
    key = _hash(dict(case=identity(case), budget=asdict(budget), levels=levels))
    return Path(root)/'cases'/f'{case.name}_{key}'


def training_folder(root, case, budget):
    return Path(root)/'training'/f'{case.name}_{_hash(dict(case=identity(case), budget=budget.training_identity()))}'


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def transmitters(case, device):
    return (ConventionalTransmitter(case.fmt, case.precode, case.link).to(device),
            TransmitterNN(case.fmt, case.precode, case.link).to(device))


def surrogate(case, root, budget, device=None, verbose=True):
    """Fit (or load) the digital surrogate of the case's analog link; return it and its audits."""
    device = device or default_device()
    folder = Path(root)/'digital_surrogates'/_hash(dict(analog=case.link.analog_part(), steps=budget.surrogate_steps))
    reference = Link(case.tx_bandwidth_ghz*1e9, case.link).to(device).freeze()
    if verbose:
        print('=== Digital surrogate: fit of the analog link ===' + ('  (already fitted)' if (folder/'fit.json').exists() else ''),
              flush=True)
    fitted, validation = digital_surrogate.fit(reference, folder, budget.surrogate_steps)
    model = Link(case.tx_bandwidth_ghz*1e9, case.link).to(device)
    with torch.no_grad():   # only the fitted FIRs: the digital filters (TX pulse, H_RX) stay those of the case
        for weight, fitted_weight in zip(digital_surrogate.fir_weights(model), digital_surrogate.fir_weights(fitted)):
            weight.copy_(fitted_weight)
    digital = _hash(dict(fmt=case.fmt, precode=case.precode, bandwidth=case.tx_bandwidth_ghz, link=identity(case.link)))
    audit_path = folder/f'audit_{case.fmt}_bw{case.tx_bandwidth_ghz:g}_{digital}.json'
    if not audit_path.exists():
        save_json(audit_path, digital_surrogate.audit(model.freeze(), reference, transmitters(case, device)[0]))
    audit = read_json(audit_path)
    if not (validation['passed'] and audit['passed']):
        print(f'  WARNING: the digital surrogate does not reproduce the reference within the tolerances '
              f'(details in {folder}); results trained through it are not reliable.', flush=True)
    return model.freeze(), dict(fit=validation, conventional_tx=audit)


def train(case, root, budget, model, methods=METHODS, device=None, verbose=True):
    """The training stages the methods need; returns {method: (TX, RX)} for the trained methods.

    RX-only: RX initialization with the conventional TX, then RX-only continuation. E2E with
    the conventional initialization: RX initialization, TX pretraining, joint training; with
    the random initialization: joint training from random weights.
    """
    from .training import train_stage
    device = device or default_device()
    out = training_folder(root, case, budget)
    wanted = set(methods)
    torch.manual_seed(SEEDS['initialization']+case.seed)   # initial weights, whatever ran before
    conventional, tx = transmitters(case, device)
    rx_only, rx = ReceiverNN(case.fmt, case.link).to(device), ReceiverNN(case.fmt, case.link).to(device)
    common = dict(regime=case.regime, level_db=case.training_level_db, symbols=budget.symbols_per_update,
                  print_every=12500 if verbose else 0)
    e2e = bool(wanted & {'e2e', 'nn_continued', 'lut'})
    from_conventional = case.initialization == 'conventional'
    if 'rx_only' in wanted or (e2e and from_conventional):
        train_stage('rx', conventional, rx_only, model, out/'rx_initialization.pt', budget.rx_steps,
                    SEEDS['rx']+case.seed, title='RX initialization with the conventional TX - BCE', **common)
    trained = {}
    if e2e:
        if from_conventional:
            rx.load_state_dict(rx_only.state_dict())         # both arms start from the same RX
            train_stage('tx', tx, rx, model, out/'tx_pretraining.pt', budget.tx_steps, SEEDS['tx']+case.seed,
                        title='TX NN fitted to the conventional voltages - MSE', **common)
            title = 'E2E: TX NN and RX NN trained jointly - BCE'
        else:
            title = 'E2E: TX NN and RX NN trained jointly from random weights - BCE'
        train_stage('joint', tx, rx, model, out/'e2e.pt', budget.joint_steps, SEEDS['joint']+case.seed,
                    title=title, **common)
        trained['e2e'] = (tx, rx)
    if 'rx_only' in wanted:
        train_stage('rx', conventional, rx_only, model, out/'rx_only.pt', budget.rx_only_steps,
                    SEEDS['rx_only']+case.seed, title='RX-only NN: RX training continued with the conventional TX - BCE',
                    **common)
        trained['rx_only'] = (conventional, rx_only)
    continuation = dict(steps=budget.continuation_steps, seed=SEEDS['continuation']+case.seed, **common)
    if 'nn_continued' in wanted:
        pair = deepcopy(tx), deepcopy(rx)
        train_stage('joint', *pair, model, out/'nn_continued.pt', title='E2E continued with the TX NN - BCE',
                    **continuation)
        trained['nn_continued'] = pair
    if 'lut' in wanted:
        pair = LookupTransmitter.from_network(tx, case.link).to(device), deepcopy(rx)
        train_stage('joint', *pair, model, out/'lut.pt', title='E2E continued with the TX lookup table - BCE',
                    **continuation)
        trained['lut'] = pair
    return trained


@torch.no_grad()
def lookup_export_error(network, case, device):
    """Largest difference (V) between a TX NN and its lookup table, on random and periodic bits."""
    table = LookupTransmitter.from_network(network, case.link).to(device)
    random = held_out_bits([8600001, 8600002], 4096, device)
    periodic = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]], device=device).repeat(1, 1024)[None]
    sequences = [random, periodic, torch.zeros_like(random), torch.ones_like(random)]
    return max(float((table(bits)-network(bits)).abs().max()) for bits in sequences)


@torch.no_grad()
def save_eye(tx, case, path, device):
    """Drive voltage and modulator field at 32 samples per symbol, 8192 symbols, no noise."""
    link = Link(case.tx_bandwidth_ghz*1e9, case.link, sps=32).to(device).freeze()
    bits = torch.randint(0, 2, (2, 8192), generator=torch.Generator().manual_seed(8500001)).to(device)[None]
    voltage = link.scaled_voltage(link.tx_waveform(tx(bits)))
    field = link.mzm(voltage)/link.p.field_loss
    np.savez_compressed(path, drive=voltage.flatten().cpu().numpy(), field_real=field.real.flatten().cpu().numpy(),
                        field_imag=field.imag.flatten().cpu().numpy(), sps=32, vpi=link.p.vpi)
    return str(path)


def linear_reference(case, folder, budget, levels, links, device, verbose=True, first_position=0):
    """Fit, calibrate and count the linear equalizer at every level, on every simulation rate.
    `first_position` shifts the grid positions that fix the seeds (see `count_errors`)."""
    conventional = transmitters(case, device)[0]
    if verbose:
        print('=== BER on the simulator: linear equalizer ===', flush=True)
    guard, n = case.link.guard_symbols, budget.symbols_per_block
    points = {label: {} for label in links}
    for index, level in enumerate(levels, first_position):
        path = folder/'linear'/f'level{level:g}.json'
        saved = read_json(path) if path.exists() else None
        if saved is None or any(label not in saved['counts'] for label in links):
            grams = rhs = None
            for start in range(0, budget.linear_fit_blocks, 8):
                ys, bs = linear.received_blocks(conventional, links['reference'], case.regime, level, index, 'fit',
                                                range(start, min(budget.linear_fit_blocks, start+8)), n)
                for y, b in zip(ys, bs):
                    g, a, _ = linear.training_moments(y, b, case.fmt, 11, guard)
                    grams = g if grams is None else [x+z for x, z in zip(grams, g)]
                    rhs = a if rhs is None else [x+z for x, z in zip(rhs, a)]
            model = linear.solve_moments(grams, rhs, case.fmt, case.link.adc_sps)
            ys, bs = linear.received_blocks(conventional, links['reference'], case.regime, level, index, 'calibration',
                                            range(budget.linear_calibration_blocks), n)
            s = [linear.scores(model, y) for y in ys]
            linear.calibrate(model, np.concatenate([a[guard:-guard] for a, _ in s]),
                             np.concatenate([p[guard:-guard] for _, p in s]) if case.fmt == 'bpam' else None,
                             np.concatenate([b[:, guard:-guard] for b in bs], axis=1))
            counts = {}
            for label, link in links.items():
                done = errors = bits = 0
                bit_errors = [0, 0]
                while (remaining := blocks_to_count(done, errors, budget.eval_blocks, budget.eval_max_blocks,
                                                    budget.eval_target_errors)):
                    ys, bs = linear.received_blocks(conventional, link, case.regime, level, index, 'test',
                                                    range(done, done+min(16, remaining)), n)
                    for y, b in zip(ys, bs):
                        wrong = linear.decisions(model, y)[:, guard:-guard] != b[:, guard:-guard]
                        errors += int(wrong.sum())
                        bits += wrong.size
                        bit_errors = [o+int(wrong[k].sum()) for k, o in enumerate(bit_errors)]
                    done += len(ys)
                counts[label] = dict(blocks_done=done, errors=errors, bits=bits, bit_errors=bit_errors, ber=errors/bits,
                                     counting_interval_95=wilson(errors, bits), at_least_200_errors=errors >= 200)
            saved = dict(model=linear.serializable(model), counts=counts)
            save_json(path, saved)
        for label in links:
            if saved['counts'][label]['errors']:              # zero counts carry no BER estimate
                points[label][str(level)] = saved['counts'][label]
        reference = saved['counts']['reference']
        if verbose and reference['errors']:
            print(f'{"OSNR" if case.regime == "optical" else "SNR":4s} {level:5.1f} dB   BER {reference["ber"]:.3e}', flush=True)
        if reference['ber'] < budget.ber_floor:
            break
    return points


def extra_points(case, root, budget=Budget(), levels=(), methods=METHODS, device=None, grid=None):
    """BER of a computed case at levels outside its evaluation grid, e.g. to extend a figure axis.

    The trained DSPs are loaded, nothing is retrained. The held-out blocks use the seeds of the
    grid positions after the last one, so they are independent of the grid blocks. Counts are
    cached in <case folder>/extra/ and restartable. `grid`: the levels the case was run with
    (default: the standard grid of its regime). Returns {method: {level: counts}}.
    """
    device = device or default_device()
    grid = list(grid or EVALUATION_LEVEL_DB[case.regime])
    folder = case_folder(root, case, budget, grid)
    if not (folder/'summary.json').exists():
        raise FileNotFoundError(f'{case.name}: run the case first (run_case).')
    levels, extra = [float(level) for level in levels], folder/'extra'
    link = Link(case.tx_bandwidth_ghz*1e9, case.link).to(device).freeze()
    points = {}
    arms = tuple(m for m in methods if m in TRAINED)
    if arms:
        model, _ = surrogate(case, root, budget, device, verbose=False)
        trained = train(case, root, budget, model, arms, device, verbose=False)
        for arm in arms:
            sender, receiver = trained[arm]
            points[arm] = count_errors(sender, receiver, link, case.regime, levels, budget.eval_blocks,
                                       extra/f'{arm}_reference_counts.json', dict(case=case.name, arm=arm, link='reference'),
                                       budget.symbols_per_block, max_blocks=budget.eval_max_blocks,
                                       target_errors=budget.eval_target_errors, first_position=len(grid))
    if 'linear' in methods:
        points['linear'] = linear_reference(case, extra, budget, levels, dict(reference=link), device, verbose=False,
                                            first_position=len(grid))['reference']
    return points


def run_case(case, root, budget=Budget(), levels=None, methods=METHODS, checks=False, eyes=True,
             device=None, verbose=True):
    """Run (or load) one case and return its summary.

    checks=True also evaluates the same DSPs on the digital surrogate and on the link simulated
    at the rates in budget.check_sps; it multiplies the evaluation time.
    """
    unknown = set(methods)-set(ALL_METHODS)
    if unknown:
        raise ValueError(f'Unknown methods {sorted(unknown)}; the methods are {ALL_METHODS}.')
    if 'linear' in methods and case.precode in ('second', 'both'):
        raise ValueError("The linear equalizer assumes the default precoder; leave 'linear' out of the methods.")
    device = device or default_device()
    levels = list(levels or EVALUATION_LEVEL_DB[case.regime])
    folder = case_folder(root, case, budget, levels)
    if (folder/'summary.json').exists():
        return read_json(folder/'summary.json')
    started = time.time()
    if verbose:
        noise = 'ASE noise' if case.regime == 'optical' else 'thermal noise'
        print(f'\n########## {case.name}: {noise}, {case.fmt.upper()}, TX filter bandwidth {case.tx_bandwidth_ghz:g} GHz'
              + (f', {case.label}' if case.label else '') + ' ##########', flush=True)
    links = {'reference': Link(case.tx_bandwidth_ghz*1e9, case.link).to(device).freeze()}
    if checks:
        links.update({f'sps{s}': Link(case.tx_bandwidth_ghz*1e9, case.link, sps=s).to(device).freeze()
                      for s in budget.check_sps if s != case.link.analog_sps})
    summary = dict(case=asdict(case), name=case.name, budget=asdict(budget), levels=levels, results={}, checks={})
    if set(methods) & set(TRAINED):
        model, surrogate_checks = surrogate(case, root, budget, device, verbose)
        summary['checks']['digital_surrogate'] = surrogate_checks
        trained = train(case, root, budget, model, methods, device, verbose)
        if 'e2e' in trained:
            summary['checks']['learned_tx'] = digital_surrogate.audit(model, links['reference'], trained['e2e'][0])
        if 'lut' in trained:
            error = lookup_export_error(trained['e2e'][0], case, device)
            summary['checks']['lookup_export'] = dict(max_difference_volt=error, passed=error <= 1e-5)
            summary['checks']['lookup_tx'] = digital_surrogate.audit(model, links['reference'], trained['lut'][0])
        evaluated = dict(links, digital_surrogate=model) if checks else links
        for arm in (m for m in TRAINED if m in methods):
            sender, receiver = trained[arm]
            if verbose:
                print(f'=== BER on the simulator: {NAMES[arm]} ===', flush=True)
            points = {label: count_errors(sender, receiver, link, case.regime, levels, budget.eval_blocks,
                                          folder/f'{arm}_{label}_counts.json', dict(case=case.name, arm=arm, link=label),
                                          budget.symbols_per_block, verbose=verbose and label == 'reference',
                                          max_blocks=budget.eval_max_blocks, target_errors=budget.eval_target_errors,
                                          floor=budget.ber_floor)
                      for label, link in evaluated.items()}
            summary['results'][arm] = dict(points=points, thresholds={k: threshold_with_interval(v) for k, v in points.items()})
            if checks:
                summary['checks'][f'{arm}_surrogate_vs_reference'] = comparison(points['digital_surrogate'], points['reference'])
                for label in points:
                    if label.startswith('sps'):
                        summary['checks'][f'{arm}_reference_vs_{label}'] = comparison(points['reference'], points[label])
                if verbose:
                    checks_text = [(f'{arm}_surrogate_vs_reference', 'same DSPs on the digital surrogate')]
                    checks_text += [(f'{arm}_reference_vs_{l}', f'same DSPs on the link simulated at {l[3:]} samples/symbol')
                                    for l in points if l.startswith('sps')]
                    for key, text in checks_text:
                        shift = summary['checks'][key].get('shift_db')
                        if key.endswith('surrogate_vs_reference') or shift is None:
                            change = 'n/a' if shift is None else f'{shift:+.2f} dB'
                        else:
                            change = f'{-shift:+.2f} dB'
                        print(f'  check, {text}: level at BER 1e-3 changes by {change}', flush=True)
            if eyes:
                (folder/'eyes').mkdir(parents=True, exist_ok=True)
                summary.setdefault('eyes', {})[arm] = save_eye(sender, case, folder/'eyes'/f'{arm}.npz', device)
    if 'linear' in methods:
        points = linear_reference(case, folder, budget, levels, links, device, verbose)
        summary['results']['linear'] = dict(points=points, thresholds={k: threshold_with_interval(v) for k, v in points.items()})
    summary['seconds'] = time.time()-started
    save_json(folder/'summary.json', summary)
    if verbose:
        text = ', '.join(f'{NAMES[m]} {t["reference"]["level_db"]:.2f} dB' if t['reference'] else f'{NAMES[m]} not reached'
                         for m, t in ((m, r['thresholds']) for m, r in summary['results'].items()))
        print(f'=== {case.name} done in {summary["seconds"]/60:.0f} min. Level at BER 1e-3: {text} ===', flush=True)
    return summary

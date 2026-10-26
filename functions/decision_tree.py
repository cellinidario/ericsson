"""Decision trees in place of the RX NNs of a computed case (DSP simplification of the receiver).

The trees read the windows of the RX NNs and are trained by the classification and regression
tree algorithm (CART, scikit-learn) on received samples, with the transmitted bits as targets.
The training samples come from the reference simulator (source='simulator', the default) or are
generated through the digital surrogate (source='surrogate'), like the training data of the NNs;
either way the trees are tested on the reference simulator. With the TX NN exported to its lookup
table, the E2E system then runs without multiplications.

Structure: 'symbol', one four-class tree deciding both bits of a symbol (for BPAM on the union of
the amplitude and sign windows), or 'bit', one tree per bit on the window of the corresponding RX
NN. By default the trees follow the RX NNs: one tree per symbol for PAM-4 (one RX NN), one per bit
for BPAM (amplitude and sign NNs).

Training windows: `training_blocks` blocks at each of five noise levels, from 4 dB below to 4 dB
above the level where the E2E RX NN reaches BER 1e-3 (bits seeded with 8800001 + 50000 j + b, noise
with 8900001 + 50000 j + b at the j-th level). Test: the held-out blocks of the RX NN evaluation
(grid levels: the same blocks; other levels: the positions after the grid, in the order given, as
in pipeline.extra_points), at least `test_blocks` blocks per level, then 128 more at a time until
`target_errors` errors or `max_blocks` blocks. For every tree: nodes, leaves, maximum depth, and the
average depth (comparisons per decision) over the first `test_blocks` blocks of each level.
Results in <case folder>/trees/ (file names ending in '_surrogate' for source='surrogate').
"""
from concurrent.futures import ThreadPoolExecutor
import json

import joblib
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier

from .evaluation import BIT_SEED, NOISE_SEED, held_out_bits, held_out_noise, threshold
from .files import read_json, save_json
from .link import Link
from .parameters import EVALUATION_LEVEL_DB, Budget, identity
from .pipeline import case_folder, default_device, surrogate, train

TRAINING_SEEDS = (8800001, 8900001)     # bits and noise of the training windows
TRAINING_OFFSETS_DB = (-4., -2., 0., 2., 4.)
BATCH, CHUNK = 16, 128                  # blocks per simulated batch; blocks added per step of the test


def _windows(link, tx, rx, regime, bit_seeds, noise_seeds, level, symbols):
    """RX NN windows (BPAM: amplitude and sign windows; PAM-4: the same window twice) and the bits."""
    device, guard = link.pulse.device, link.p.guard_symbols
    offsets = (0, -1) if rx.fmt == 'bpam' else (0, 0)
    xs, bits_out = [[], []], []
    with torch.no_grad():
        for start in range(0, len(bit_seeds), BATCH):
            bits = held_out_bits(bit_seeds[start:start+BATCH], symbols, device)
            noise = held_out_noise(noise_seeds[start:start+BATCH], link.noise_samples(regime, symbols), regime, device)
            y = link(tx(bits), regime, level, noise)
            for k, offset in enumerate(offsets):
                xs[k].append(rx.windows(y, offset)[:, guard:-guard].reshape(-1, rx.window).float().cpu().numpy())
            bits_out.append(bits[..., guard:-guard].transpose(1, 2).reshape(-1, 2).cpu().numpy())
    return [np.concatenate(x) for x in xs], np.concatenate(bits_out)


def _central(x, length):
    side = (x.shape[1]-length)//2
    return np.ascontiguousarray(x[:, side:side+length])


def _inputs(x, structure, fmt):
    """Input of the trees from the [amplitude, sign] windows: one array per tree."""
    if structure == 'bit':
        return x
    return [x[0] if fmt == 'pam4' else np.ascontiguousarray(np.concatenate([x[1][:, :1], x[0]], axis=1))]


def _errors(trees, inputs, bits, structure):
    if structure == 'bit':
        return sum(int((tree.predict(inputs[k]) != bits[:, k]).sum()) for k, tree in enumerate(trees))
    symbol = trees[0].predict(inputs[0]).astype(int)
    return int(((symbol >> 1) != bits[:, 0]).sum()+((symbol & 1) != bits[:, 1]).sum())


def decision_trees(case, root, budget=Budget(), levels=(), structure=None, leaves=1024, window=None,
                   training_blocks=200, test_blocks=128, max_blocks=2048, target_errors=100, training_center_db=None,
                   source='simulator', device=None, grid=None, verbose=True):
    """Train (or load) the RX decision trees of the E2E system of a computed case and count their BER.

    `levels`: noise levels of the test; `structure`: 'symbol', 'bit' or None (as the RX NNs);
    `leaves`: maximum number of leaves per tree (nodes = 2 leaves - 1); `window`: the central samples
    of the RX NN window used as input (None: the whole window); `grid`: the levels the case was run
    with (default: the standard grid of its regime); `training_center_db`: the centre of the training levels
    (default: the level where the E2E RX NN reaches BER 1e-3); `source`: where the training windows come
    from, 'simulator' (the reference simulator) or 'surrogate' (the digital surrogate of the case). Returns a
    dict with the points, the level at BER 1e-3 and the complexity of the trees.
    """
    if source not in ('simulator', 'surrogate'):
        raise ValueError(f"{source!r} is not one of ('simulator', 'surrogate').")
    device = device or default_device()
    grid = list(grid or EVALUATION_LEVEL_DB[case.regime])
    folder = case_folder(root, case, budget, grid)
    if not (folder/'summary.json').exists():
        raise FileNotFoundError(f'{case.name}: run the case first (run_case).')
    summary = read_json(folder/'summary.json')
    structure = structure or ('symbol' if case.fmt == 'pam4' else 'bit')
    window = window or case.link.rx_window
    levels = [float(level) for level in levels]
    outside = [level for level in levels if level not in grid]
    signature = json_ready(dict(case=identity(case), structure=structure, leaves=leaves, window=window,
                                training_blocks=training_blocks, training_offsets_db=TRAINING_OFFSETS_DB,
                                test_blocks=test_blocks, max_blocks=max_blocks, target_errors=target_errors,
                                levels=levels, symbols=budget.symbols_per_block))
    if training_center_db is not None:      # only when given, so that the default signature stays as it is
        signature['training_center_db'] = float(training_center_db)
    if source == 'surrogate':               # likewise: the trees of the simulator keep their signature and file
        signature['source'] = source
    path = folder/'trees'/f'{structure}_L{window}_leaves{leaves}{"_surrogate" if source == "surrogate" else ""}.json'
    if path.exists() and read_json(path)['signature'] == signature:
        return read_json(path)
    model, _ = surrogate(case, root, budget, device, verbose=False)
    tx, rx = train(case, root, budget, model, ('e2e',), device, verbose=False)['e2e']
    tx.eval(); rx.eval()
    link = Link(case.tx_bandwidth_ghz*1e9, case.link).to(device).freeze()
    reached = threshold(summary['results']['e2e']['points']['reference'])
    if training_center_db is None and reached is None:
        raise ValueError(f'{case.name}: the E2E RX NN does not reach BER 1e-3 on the grid; set training_center_db.')
    nn = float(training_center_db) if training_center_db is not None else reached['level_db']
    parts = [_windows(model if source == 'surrogate' else link, tx, rx, case.regime,
                      [TRAINING_SEEDS[0]+50000*j+b for b in range(training_blocks)],
                      [TRAINING_SEEDS[1]+50000*j+b for b in range(training_blocks)], nn+offset, budget.symbols_per_block)
             for j, offset in enumerate(TRAINING_OFFSETS_DB)]
    x = _inputs([_central(np.concatenate([part[0][k] for part in parts]), window) for k in range(2)], structure, case.fmt)
    bits = np.concatenate([part[1] for part in parts])
    del parts
    targets = [bits[:, 0], bits[:, 1]] if structure == 'bit' else [2*bits[:, 0]+bits[:, 1]]
    with ThreadPoolExecutor(max_workers=len(x)) as pool:
        trees = list(pool.map(lambda k: DecisionTreeClassifier(max_leaf_nodes=leaves, random_state=0).fit(x[k], targets[k]),
                              range(len(x))))
    del x, bits, targets
    points = {}
    for level in levels:
        position = grid.index(level) if level in grid else len(grid)+outside.index(level)
        errors = counted = blocks = 0
        depth = np.zeros(len(trees))
        while blocks < max_blocks and (blocks < test_blocks or errors < target_errors):
            ids = range(blocks, blocks+(test_blocks if blocks == 0 else CHUNK))
            xt, bt = _windows(link, tx, rx, case.regime, [BIT_SEED+10000*position+b for b in ids],
                              [NOISE_SEED+10000*position+b for b in ids], level, budget.symbols_per_block)
            inputs = _inputs([_central(xk, window) for xk in xt], structure, case.fmt)
            errors += _errors(trees, inputs, bt, structure)
            counted += bt.size
            if blocks == 0:
                depth = np.array([float(tree.decision_path(xk).sum(axis=1).mean()-1) for tree, xk in zip(trees, inputs)])
            blocks = ids[-1]+1
        points[str(level)] = dict(errors=errors, bits=counted, ber=errors/counted, blocks=blocks,
                                  average_depth=depth.tolist())
        if verbose:
            print(f'{case.name} trees, {"OSNR" if case.regime == "optical" else "SNR"} {level:g} dB: '
                  f'BER {errors/counted:.3e} ({errors} errors)', flush=True)
    t = threshold(points)
    result = dict(signature=signature, structure=structure, window=window, leaves=leaves, source=source, nn_level_db=nn,
                  level_db=None if t is None else t['level_db'],
                  trees=[dict(nodes=int(tree.tree_.node_count), leaves=int(tree.get_n_leaves()),
                              maximum_depth=int(tree.get_depth())) for tree in trees], points=points)
    save_json(path, result)
    joblib.dump(trees, path.with_suffix('.joblib'))
    return result


def complexity(result, bits_per_threshold=6):
    """Nodes, average depth (at the test level closest to BER 1e-3) and maximum depth of every tree, and
    comparisons and memory per information bit as in Table I of arXiv:2608.28129 (two bits per symbol)."""
    level = min(result['points'], key=lambda l: abs(float(l)-(result['level_db'] or float(l))))
    depth = result['points'][level]['average_depth']
    nodes = [tree['nodes'] for tree in result['trees']]
    return dict(nodes=nodes, average_depth=depth, maximum_depth=[tree['maximum_depth'] for tree in result['trees']],
                comparisons_per_bit=sum(depth)/2, memory_kbit_per_bit=bits_per_threshold*sum(nodes)/2/1e3)


def json_ready(value):
    """The value as it reads back from a JSON file (tuples as lists), to compare signatures."""
    return json.loads(json.dumps(value, default=str))

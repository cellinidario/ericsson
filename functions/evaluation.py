"""BER on held-out blocks, and the noise level at BER 1e-3 with its counting interval.

Held-out data: at noise level number i of the grid, block b uses bits seeded with
8200001 + 10000 i + b and noise seeded with 8300001 + 10000 i + b, so every receiver and
every simulation rate is counted on the same bits and the same noise. Every level is counted
on a minimum number of blocks and then on further blocks until a target number of errors (up
to a maximum number of blocks), so that low BERs are measured as accurately as high ones. The
sweep stops after the first level whose BER falls below a floor: higher levels would only add
zero counts. Blocks are simulated in batches on the GPU; each block keeps its own seeds,
normalizations and ADC range.
"""
import json
import math
from pathlib import Path

import torch

from .dsp import decisions
from .files import read_json, save_json
from .parameters import identity

BIT_SEED, NOISE_SEED = 8200001, 8300001
WILSON_Z = 1.959963984540054


def held_out_bits(seeds, symbols, device):
    return torch.stack([torch.randint(0, 2, (2, symbols), generator=torch.Generator().manual_seed(s))
                        for s in seeds]).to(device)


def held_out_noise(seeds, samples, regime, device):
    """Unit-variance noise per block, drawn exactly as the link draws it (real then imaginary ASE part)."""
    draws = 2 if regime == 'optical' else 1
    per_block = []
    for s in seeds:
        torch.manual_seed(s)
        per_block.append([torch.randn(samples, device=device) for _ in range(draws)])
    return [torch.stack([block[k] for block in per_block]) for k in range(draws)]


def block_batch(link, blocks_per_batch=None):
    """Blocks simulated together: fewer at high analog rates to bound GPU memory."""
    return blocks_per_batch or max(1, 32*link.p.dsp_sps//link.sps)


def blocks_to_count(done, errors, blocks, max_blocks, target_errors):
    """Blocks still to count at one level: at least `blocks`, then more until `target_errors` errors or `max_blocks`."""
    limit = blocks if done < blocks or errors >= target_errors else max_blocks
    return max(0, limit-done)


@torch.no_grad()
def count_errors(tx, rx, link, regime, levels, blocks, path=None, provenance=None, symbols=32768,
                 blocks_per_batch=None, verbose=False, max_blocks=None, target_errors=0, floor=0., first_position=0):
    """Bit errors at the levels of the grid (positions in the grid fix the seeds), in increasing
    order up to the first level with BER below `floor`; restartable. `first_position` shifts the
    positions, e.g. to count levels outside a grid on blocks independent of the grid ones."""
    device, guard = link.pulse.device, link.p.guard_symbols
    max_blocks = blocks if max_blocks is None else max_blocks
    signature = dict(levels=levels, blocks=blocks, max_blocks=max_blocks, target_errors=target_errors,
                     floor=floor, symbols=symbols, regime=regime, link=identity(link.p),
                     analog_sps=link.sps, provenance=provenance)
    if first_position:          # only when used, so that the existing restart files keep their signature
        signature['first_position'] = first_position
    signature = json.loads(json.dumps(signature))
    points = {}
    if path is not None and Path(path).exists():
        saved = read_json(path)
        if saved['signature'] != signature:
            raise RuntimeError(f'{path} holds counts of a different evaluation.')
        points = saved['points']
    tx.eval(); rx.eval()
    batch = block_batch(link, blocks_per_batch)
    for index, level in enumerate(levels, first_position):
        row = points.setdefault(str(level), dict(blocks_done=0, errors=0, bits=0, bit_errors=[0, 0]))
        while (remaining := blocks_to_count(row['blocks_done'], row['errors'], blocks, max_blocks, target_errors)):
            start = row['blocks_done']
            ids = range(start, start+min(batch, remaining))
            bits = held_out_bits([BIT_SEED+10000*index+b for b in ids], symbols, device)
            noise = held_out_noise([NOISE_SEED+10000*index+b for b in ids], link.noise_samples(regime, symbols), regime, device)
            wrong = decisions(rx(link(tx(bits), regime, level, noise)))[..., guard:-guard] != bits[..., guard:-guard]
            row['errors'] += int(wrong.sum())
            row['bits'] += wrong.numel()
            row['bit_errors'] = [old+int(wrong[:, k].sum()) for k, old in enumerate(row['bit_errors'])]
            row['blocks_done'] = ids[-1]+1
            if path is not None:
                save_json(path, dict(signature=signature, points=points))
        row.update(ber=row['errors']/row['bits'], counting_interval_95=wilson(row['errors'], row['bits']),
                   at_least_200_errors=row['errors'] >= 200)
        if path is not None:
            save_json(path, dict(signature=signature, points=points))
        if verbose and row['errors']:
            print(f'{"OSNR" if regime == "optical" else "SNR":4s} {level:5.1f} dB   BER {row["ber"]:.3e}', flush=True)
        if row['ber'] < floor:
            break
    if not all(row['errors'] for row in points.values()):     # zero counts carry no BER estimate
        points = {level: row for level, row in points.items() if row['errors']}
    return points


def wilson(errors, bits):
    """Wilson 95% binomial interval (errors are treated as independent)."""
    p, z = errors/bits, WILSON_Z
    denominator = 1+z*z/bits
    center = (p+z*z/(2*bits))/denominator
    radius = z*math.sqrt(p*(1-p)/bits+z*z/(4*bits*bits))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def threshold(points, target=1e-3):
    """Noise level at BER = target, interpolating log10(BER) linearly between the bracketing levels."""
    ordered = sorted((float(level), row) for level, row in points.items())
    for (x1, a), (x2, b) in zip(ordered[:-1], ordered[1:]):
        if a['ber'] >= target >= b['ber'] > 0:
            fraction = (math.log10(target)-math.log10(a['ber']))/(math.log10(b['ber'])-math.log10(a['ber']))
            return dict(level_db=x1+fraction*(x2-x1), bracket=[x1, x2],
                        adequate_counts=a['errors'] >= 200 and b['errors'] >= 200)
    return None


def threshold_with_interval(points):
    """Threshold plus the thresholds of the lower and upper counting-interval curves."""
    result = threshold(points)
    if result is None:
        return None
    limits = []
    for index in (0, 1):
        bound = {k: dict(v, ber=v['counting_interval_95'][index]) for k, v in points.items()}
        limit = threshold(bound)
        limits.append(None if limit is None else limit['level_db'])
    result['approximate_counting_interval_95'] = limits
    return result


def comparison(candidate, reference, tolerance_db=.1):
    """Do two BER curves cross 1e-3 within tolerance_db of each other (counting intervals included)?"""
    ct, rt = threshold_with_interval(candidate), threshold_with_interval(reference)
    if ct is None or rt is None:
        return dict(passed=False, reason='Missing measured BER1e-3 bracket.', candidate=ct, reference=rt)
    shift = ct['level_db']-rt['level_db']
    ci, ri = ct['approximate_counting_interval_95'], rt['approximate_counting_interval_95']
    interval = None if None in ci+ri else [ci[0]-ri[1], ci[1]-ri[0]]
    regressions = [k for k, v in candidate.items() if k in reference and v['errors'] >= 200
                   and reference[k]['errors'] >= 200
                   and v['counting_interval_95'][0] > 2*reference[k]['counting_interval_95'][1]]
    passed = bool(ct['adequate_counts'] and rt['adequate_counts'] and abs(shift) <= tolerance_db
                  and interval is not None and interval[0] >= -tolerance_db and interval[1] <= tolerance_db
                  and not regressions)
    return dict(passed=passed, shift_db=shift, approximate_shift_interval_95=interval,
                supported_twofold_regressions=regressions, candidate=ct, reference=rt)

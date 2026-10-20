"""Least-squares linear equalizer at two samples per symbol, the linear reference receiver.

BPAM: two fractionally spaced FIR filters of 2L+1 taps (L = 11: 23 taps). C, centred on
pulse k, estimates the intensity of symbol k (targets 0.4 and 1.6); D, centred on the
crossing of pulses k and k+1, estimates the cross term sqrt(P[k] P[k+1])/2 times the sign
change. PAM-4: one FIR filter, target the Gray rank 0..3, three thresholds; with one RX
sample per symbol (LinkParameters.rx_sps=1) the FIR is symbol spaced.
The taps solve the normal equations (no regularization). Each decision threshold minimizes
the number of errors on calibration blocks (exact search over the sorted scores).
Fit, calibration and test use disjoint blocks.
"""
import numpy as np
import torch

BPAM_POWERS = np.array([.4, 1.6], dtype=np.float64)
GRAY_RANKS = np.array([0, 1, 3, 2], dtype=np.int64)
GRAY_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.int64)

# Seeds of the received blocks: test blocks are the held-out blocks of every receiver (see
# evaluation.py); fit and calibration blocks use disjoint seeds.
STAGE_SEEDS = {'test': (8200001, 8300001), 'fit': (95000001, 95500001), 'calibration': (96000001, 96500001)}


def training_moments(samples, bits, fmt, half_length=11, guard=128):
    """Accumulate Y^T Y and Y^T target over the interior symbols of one block."""
    y = np.asarray(samples, dtype=np.float64).reshape(-1)
    b = np.asarray(bits, dtype=np.int64)
    sps = y.size//b.shape[1] if b.ndim == 2 and b.shape[1] else 0
    if b.ndim != 2 or b.shape[0] != 2 or sps not in (1, 2) or y.size != sps*b.shape[1] or (fmt == 'bpam' and sps != 2):
        raise ValueError('Expected two bit rows and two RX samples per symbol (one allowed for PAM-4).')
    memory = max(guard, int(np.ceil(half_length/sps)))
    k = np.arange(memory, b.shape[1]-memory)
    offsets = np.arange(half_length, -half_length-1, -1)
    centre = y[sps*k[:, None]+offsets[None, :]]
    target = BPAM_POWERS[b[0, k]] if fmt == 'bpam' else GRAY_RANKS[2*b[0, k]+b[1, k]].astype(float)
    grams, rhs = [centre.T@centre], [centre.T@target]
    if fmt == 'bpam':
        crossing = y[(2*k+1)[:, None]+offsets[None, :]]
        # The sample between pulses k and k+1 carries the sign change of symbol k+1.
        sign_change = 1-2*b[1, k+1]
        cross_term = .5*np.sqrt(BPAM_POWERS[b[0, k]]*BPAM_POWERS[b[0, k+1]])*sign_change
        grams.append(crossing.T@crossing)
        rhs.append(crossing.T@cross_term)
    return grams, rhs, int(k.size)


def solve_moments(grams, rhs, fmt, sps=2):
    coefficients = [np.linalg.solve(g, a) for g, a in zip(grams, rhs)]
    if not all(np.isfinite(c).all() for c in coefficients):
        raise RuntimeError('Nonfinite least-squares coefficients.')
    model = {'initial_format': fmt, 'C': coefficients[0], 'D': coefficients[1] if fmt == 'bpam' else None,
             'condition_numbers': [float(np.linalg.cond(g)) for g in grams]}
    if sps != 2:            # only then, so that the stored models keep their content
        model['sps'] = sps
    return model


def scores(model, samples):
    """Filter outputs: intensity estimate per symbol and, for BPAM, cross-term estimate per symbol."""
    y = np.asarray(samples, dtype=np.float64).reshape(-1)
    amplitude = np.convolve(y, model['C'], mode='same')[::model.get('sps', 2)]
    phase = None
    if model['initial_format'] == 'bpam':
        crossing = np.convolve(y, model['D'], mode='same')[1::2]
        phase = np.empty_like(crossing)
        phase[0] = np.nan
        phase[1:] = crossing[:-1]            # crossing of pulses k-1 and k decides bit k
    return amplitude, phase


def optimal_threshold(lower, upper):
    """Threshold minimizing count(upper < t) + count(lower > t)."""
    low = np.asarray(lower, dtype=np.float64).reshape(-1)
    high = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not len(low) or not len(high) or not np.isfinite(np.r_[low, high]).all():
        raise ValueError('Threshold calibration requires two finite, nonempty classes.')
    values = np.r_[low, high]
    upper_class = np.r_[np.zeros(len(low), dtype=np.int64), np.ones(len(high), dtype=np.int64)]
    order = np.argsort(values, kind='stable')
    values, upper_class = values[order], upper_class[order]
    positions = np.flatnonzero(values[:-1] < values[1:])
    thresholds = .5*(values[positions]+values[positions+1])
    keep = (thresholds >= low.min()) & (thresholds <= high.max())
    positions, thresholds = positions[keep], thresholds[keep]
    if not len(positions):
        raise RuntimeError('No threshold separates the calibration classes.')
    upper_below = np.cumsum(upper_class)[positions]
    errors = len(low)-(positions+1-upper_below)+upper_below
    best = np.flatnonzero(errors == errors.min())
    index = best[len(best)//2]
    return float(thresholds[index]), int(errors[index])


def calibrate(model, amplitude, phase, bits):
    b = np.asarray(bits, dtype=np.int64)
    if model['initial_format'] == 'bpam':
        amplitude_threshold, amplitude_errors = optimal_threshold(amplitude[b[0] == 0], amplitude[b[0] == 1])
        phase_threshold, phase_errors = optimal_threshold(phase[b[1] == 1], phase[b[1] == 0])
        model.update(thresholds=[amplitude_threshold], phase_threshold=phase_threshold,
                     calibration_errors=[amplitude_errors, phase_errors])
    else:
        ranks = GRAY_RANKS[2*b[0]+b[1]]
        cuts = [optimal_threshold(amplitude[ranks == j], amplitude[ranks == j+1]) for j in range(3)]
        if not np.all(np.diff([c[0] for c in cuts]) > 0):
            raise RuntimeError('PAM-4 thresholds are not ordered.')
        model.update(thresholds=[c[0] for c in cuts], calibration_errors=[c[1] for c in cuts])
    return model


def decisions(model, samples):
    amplitude, phase = scores(model, samples)
    if model['initial_format'] == 'bpam':
        return np.stack([amplitude > model['thresholds'][0], phase < model['phase_threshold']]).astype(np.int64)
    return GRAY_BITS[np.searchsorted(model['thresholds'], amplitude, side='right')].T


def serializable(model):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in model.items()}


def received_blocks(tx, link, regime, level, level_index, stage, block_ids, symbols=32768):
    """RX samples (blocks, adc_sps N) in double precision and bits (blocks, 2, N), simulated as a batch."""
    from .evaluation import held_out_bits, held_out_noise
    bit_seed, noise_seed = STAGE_SEEDS[stage]
    device = link.pulse.device
    with torch.no_grad():
        bits = held_out_bits([bit_seed+10000*level_index+b for b in block_ids], symbols, device)
        noise = held_out_noise([noise_seed+10000*level_index+b for b in block_ids], link.noise_samples(regime, symbols),
                               regime, device)
        y = link(tx(bits), regime, level, noise)
    return y.double().cpu().numpy(), bits.cpu().numpy()

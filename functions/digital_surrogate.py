"""Digital surrogate of the analog link: fit and identification audits.

The digital surrogate has the structure of the analog link: three trainable FIR (Conv1d)
layers for driver, optical filter and photodiode, around fixed operators (modulator sine
law, fiber dispersion, square law). It is fitted on input/output pairs (random voltages ->
photocurrent) by minimizing the normalized mean-square error plus a small weight penalty,
starting from identity filters. The reference can be the simulator or measured waveforms.
Its weights are then frozen and the DSPs are trained through it.

With an analog H_RX (LinkParameters.rx_filter_before_adc) the third FIR is H_RX instead of
the photodiode response, and the output is the noiseless H_RX output: H_RX models the band
limitation of photodiode and ADC electronics, so the photodiode response is kept as in the
reference (the identity at two samples per symbol), and the thermal noise, added before H_RX,
is filtered by the fitted H_RX as in the reference. The data fix only the product of the H_opt
gain squared (H_opt acts on the field) and the H_RX gain, while the thermal noise enters between
the two: the fitted H_RX is therefore normalized to unit DC gain, as the H_RX of the reference
(SNR referred to the photodiode output), and H_opt takes the inverse gain, which leaves the
noiseless and ASE-noisy outputs unchanged.
"""
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from .files import digest, read_json, save_json, save_torch

FIR_BLOCKS = ('driver', 'optical_filter', 'photodiode')
FIR_BLOCKS_ANALOG_RX = ('driver', 'optical_filter', 'rx_filter')   # with an analog H_RX
EDGE = 1024              # samples excluded at each record edge in the loss and in the checks
RECORD = 8192            # samples per training record
WEIGHT_PENALTY = .001


def fir_blocks(link):
    return FIR_BLOCKS_ANALOG_RX if link.p.rx_filter_before_adc else FIR_BLOCKS


def fir_weights(link):
    return [getattr(link, name).weight for name in fir_blocks(link)]


def normalize_gain(model):
    """Analog H_RX: unit DC gain of the fitted H_RX, the inverse gain moved to H_opt (see the module docstring)."""
    if model.p.rx_filter_before_adc:
        with torch.no_grad():
            gain = model.rx_filter.weight.sum()
            if gain <= 0:
                raise RuntimeError('The fitted H_RX has a nonpositive DC gain.')
            model.rx_filter.weight.div_(gain)
            model.optical_filter.weight.mul_(gain.sqrt())
    return model


def records(reference, count, seed):
    """Gaussian random voltages (std = peak/2), clipped at +-peak, and their raw photocurrent."""
    generator = torch.Generator().manual_seed(seed)
    peak, device = reference.p.peak_volt, reference.pulse.device
    data = []
    with torch.no_grad():
        for _ in range(count):
            x = torch.randn((1, RECORD), generator=generator)
            x = (x*(2*peak)/4).clamp(-peak, peak).to(device)
            data.append((x, reference.raw_photocurrent(x)))
    return data


def nmse(model, reference, data, noisy=False):
    """Error energy over signal energy (mean removed), over all records."""
    error = energy = 0.
    with torch.no_grad():
        for i, (x, target) in enumerate(data):
            if noisy:  # the same ASE realization through both links
                torch.manual_seed(81000+i)
                target = reference.raw_photocurrent(x, ase_ebn0_db=14.)
                torch.manual_seed(81000+i)
            prediction = model.raw_photocurrent(x, ase_ebn0_db=14. if noisy else None)
            y, yp = target[..., EDGE:-EDGE], prediction[..., EDGE:-EDGE]
            error += float((yp-y).double().square().sum())
            energy += float((y-y.mean()).double().square().sum())
    return error/energy


def validate_fit(model, reference):
    """Held-out records: noiseless NMSE <= 1e-3, noisy NMSE <= 1e-2, input gradients aligned."""
    model.eval().requires_grad_(False)
    test = records(reference, 4, 33001)
    gradients = []
    for x, _ in test:
        a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
        ya, yb = reference.raw_photocurrent(a)[..., EDGE:-EDGE], model.raw_photocurrent(b)[..., EDGE:-EDGE]
        projection = torch.randn(ya.shape[-1], generator=torch.Generator().manual_seed(9931)).to(x.device)
        ga = torch.autograd.grad((ya*projection).sum(), a)[0][..., EDGE:-EDGE].flatten()
        gb = torch.autograd.grad((yb*projection).sum(), b)[0][..., EDGE:-EDGE].flatten()
        gradients.append(dict(cosine=float(F.cosine_similarity(ga, gb, dim=0)),
                              relative_l2=float((gb-ga).norm()/ga.norm())))
    report = dict(noiseless_nmse=nmse(model, reference, test), noisy_nmse=nmse(model, reference, test, True),
                  input_gradients=gradients)
    report['passed'] = (report['noiseless_nmse'] <= .001 and report['noisy_nmse'] <= .01
                        and all(g['cosine'] >= .99 and g['relative_l2'] <= .1 for g in gradients))
    return report


def load_fitted(model, path):
    """Load a fitted digital surrogate. The digital buffers (DAC pulse, digital H_RX) are set by the link parameters,
    never fitted, and are kept from the current link: a surrogate is shared by every link with the same analog part,
    whose digital settings (e.g. the ADC rate) may differ."""
    state = torch.load(path, map_location=model.pulse.device, weights_only=True)
    state.update({k: v for k, v in model.state_dict().items() if k in ('pulse', 'h_rx')})
    model.load_state_dict(state)
    return model


def fit(reference, folder, steps=40000):
    """Fit (or load) the three FIRs; restartable; keeps the state with the best validation error."""
    folder = Path(folder)
    best_path, resume_path = folder/'digital_surrogate.pt', folder/'fit_progress.pt'
    device = reference.pulse.device
    model = deepcopy(reference)
    if (folder/'fit.json').exists():
        done = read_json(folder/'fit.json')
        if done['steps'] != steps or done['sha256'] != digest(best_path):
            raise RuntimeError(f'The digital surrogate in {folder} was fitted with other settings.')
        return normalize_gain(load_fitted(model, best_path)).freeze(), done['validation']
    model.train().requires_grad_(False)
    with torch.no_grad():
        for weight in fir_weights(model):
            weight.zero_()
            weight[..., weight.shape[-1]//2] = 1.
            weight.requires_grad_(True)
    train, valid = records(reference, 16, 11001), records(reference, 4, 22001)
    optimizer = torch.optim.Adam(fir_weights(model), lr=.001)
    start, best, best_step = 0, float('inf'), 0
    if resume_path.exists():
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start, best, best_step = saved['step'], saved['best'], saved['best_step']
    for step in range(start, steps):
        x, target = train[step % len(train)]
        y = target[..., EDGE:-EDGE]
        loss = (model.raw_photocurrent(x)[..., EDGE:-EDGE]-y).square().mean()/y.var(unbiased=False)
        loss = loss+WEIGHT_PENALTY*sum(w.square().sum() for w in fir_weights(model))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step+1) % 100 == 0 or step+1 == steps:
            score = nmse(model, reference, valid)
            criterion = score+WEIGHT_PENALTY*sum(float(w.detach().square().sum()) for w in fir_weights(model))
            if criterion < best:
                best, best_step = criterion, step+1
                save_torch(best_path, model.state_dict())
        if (step+1) % 500 == 0 or step+1 == steps:
            save_torch(resume_path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         step=step+1, best=best, best_step=best_step))
        if (step+1) % 5000 == 0 or step+1 == steps:
            print(f'  step {step+1:6d}  validation NMSE {score:.3e}', flush=True)
    load_fitted(model, best_path)
    normalize_gain(model)
    report = validate_fit(model, reference)
    save_json(folder/'fit.json', dict(steps=steps, best_step=best_step, sha256=digest(best_path), validation=report))
    return model.freeze(), report


def audit(model, reference, tx, bits_seed=8700001, noise_seed=8700101, projection_seed=8700201):
    """Compare the digital surrogate with the reference on the waveforms of one transmitter.

    Output NMSE (photocurrent, or H_RX output when H_RX is analog) without noise and with ASE
    (limits 1e-3 and 1e-2); with an analog H_RX, the thermal noise passes through the fitted H_RX,
    whose noise power gain must match the reference within 0.1 dB; gradient
    of a random projection with respect to the drive and, for a trainable TX, to its weights
    (cosine >= 0.99, relative L2 <= 0.1). The peak scale is held constant in the gradients:
    the block maximum is not differentiable where two peaks tie.
    """
    device, g = reference.pulse.device, reference.p.guard_symbols*reference.sps
    bits = torch.randint(0, 2, (2, 4096), generator=torch.Generator().manual_seed(bits_seed)).to(device)[None]
    drive = reference.tx_waveform(tx(bits)).detach()
    rows = {}
    for name, level in (('noiseless_nmse', None), ('ase_noisy_nmse', 16.05)):
        with torch.no_grad():
            torch.manual_seed(noise_seed)
            a = reference.analog_output(drive, 'optical', level)[..., g:-g]
            torch.manual_seed(noise_seed)
            b = model.analog_output(drive, 'optical', level)[..., g:-g]
        rows[name] = float((a-b).double().square().sum()/(a-a.mean()).double().square().sum())
    if reference.p.rx_filter_before_adc:
        power = [float(link.rx_filter.weight.double().square().sum()) for link in (model, reference)]
        rows['thermal_noise_gain_db'] = 10*math.log10(power[0]/power[1])
    a, b = drive.clone().requires_grad_(), drive.clone().requires_grad_()
    ya = reference.analog_output(a, 'optical', detach_peak=True)[..., g:-g]
    yb = model.analog_output(b, 'optical', detach_peak=True)[..., g:-g]
    projection = torch.randn(ya.shape[-1], generator=torch.Generator().manual_seed(projection_seed)).to(device)
    ga = torch.autograd.grad((ya*projection).sum(), a)[0][..., g:-g].flatten()
    gb = torch.autograd.grad((yb*projection).sum(), b)[0][..., g:-g].flatten()
    rows['input_gradient_relative_l2'] = float((gb-ga).norm()/ga.norm())
    rows['input_gradient_cosine'] = float(F.cosine_similarity(ga, gb, dim=0))
    weights = [q for q in tx.parameters() if q.requires_grad]
    if weights:
        ya = reference.analog_output(reference.tx_waveform(tx(bits)), 'optical', detach_peak=True)[..., g:-g]
        yb = model.analog_output(model.tx_waveform(tx(bits)), 'optical', detach_peak=True)[..., g:-g]
        ga = torch.cat([q.flatten() for q in torch.autograd.grad((ya*projection).sum(), weights)])
        gb = torch.cat([q.flatten() for q in torch.autograd.grad((yb*projection).sum(), weights)])
        rows['tx_weight_gradient_relative_l2'] = float((gb-ga).norm()/ga.norm())
        rows['tx_weight_gradient_cosine'] = float(F.cosine_similarity(ga, gb, dim=0))
    rows['passed'] = (all(rows[k] <= (.01 if 'noisy' in k else .001) for k in rows if k.endswith('nmse'))
                      and all(rows[k] >= .99 for k in rows if k.endswith('_cosine'))
                      and all(rows[k] <= .1 for k in rows if k.endswith('_relative_l2'))
                      and abs(rows.get('thermal_noise_gain_db', 0.)) <= .1)
    return rows

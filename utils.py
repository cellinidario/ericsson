"""Helpers: theoretical PAM BER, AWGN at a target Eb/N0, and BER measurement.

Eb/N0 convention (one-sided N0, as in Forestieri's BER formulas):
    Eb/N0 [dB] = SNR_per_sample [dB] + 10*log10(samples_per_symbol_sim / (2*bits_per_symbol))
The noise is real and white over the simulation band f_s = samples_per_symbol_sim*symbol_rate,
added at the ADC (after the photodiode); the matched filter then combines it optimally.
The factor 2 is the one-sided/two-sided N0 distinction: skip it and you are 3 dB off.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from scipy.special import erfc


def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))


def theoretical_ber_unipolar(ebn0_db, modulation_order):
    """Gray-mapped unipolar PAM-M BER (Forestieri et al.)."""
    ebn0 = 10 ** (np.asarray(ebn0_db) / 10)
    bits_per_symbol = np.log2(modulation_order)
    prefactor = 2 * (modulation_order - 1) / (modulation_order * bits_per_symbol)
    argument = 3 * bits_per_symbol / ((modulation_order - 1) * (2 * modulation_order - 1)) * ebn0
    return prefactor * q_function(np.sqrt(argument))


def theoretical_ber_unipolar_ase(ebn0_db, modulation_order):
    """Gray-mapped unipolar PAM-M BER in the ASE-noise-dominant regime (Forestieri eq. A.47).

    Optically-preamplified receiver, signal-spontaneous beat noise dominant, equispaced
    AMPLITUDE levels (A_m = (m-1)A), optimum thresholds. The decision variable is the field
    amplitude (Ricean), so a Rayleigh-tail exp() term replaces one of the Gaussian tails:
        BER = 1/(M*log2M) * [ exp(-(c/2)*Eb/N0) + (2M-3)*Q(sqrt(c*Eb/N0)) ],
        c = 3*log2M / ((M-1)(2M-1))   (same Q-argument as the electrical regime, A.19).
    Eb/N0 here is the photons-per-bit at the preamplifier input.
    """
    ebn0 = 10 ** (np.asarray(ebn0_db) / 10)
    bits_per_symbol = np.log2(modulation_order)
    coefficient = 3 * bits_per_symbol / ((modulation_order - 1) * (2 * modulation_order - 1))
    prefactor = 1.0 / (modulation_order * bits_per_symbol)
    rayleigh_tail = np.exp(-0.5 * coefficient * ebn0)
    gaussian_tails = (2 * modulation_order - 3) * q_function(np.sqrt(coefficient * ebn0))
    return prefactor * (rayleigh_tail + gaussian_tails)


def theoretical_ber_bipolar(ebn0_db, modulation_order):
    """Gray-mapped conventional bipolar PAM-M BER (Forestieri et al.)."""
    ebn0 = 10 ** (np.asarray(ebn0_db) / 10)
    bits_per_symbol = np.log2(modulation_order)
    prefactor = 2 * (modulation_order - 1) / (modulation_order * bits_per_symbol)
    argument = 6 * bits_per_symbol / (modulation_order ** 2 - 1) * ebn0
    return prefactor * q_function(np.sqrt(argument))


def add_awgn(photocurrent, ebn0_db, bits_per_symbol, samples_per_symbol_sim):
    """Add white Gaussian noise at the ADC for a target Eb/N0 (one-sided N0, Forestieri).

    Inverts  Eb/N0 = snr_per_sample + 10*log10(K_sim/(2k))  to get the per-sample SNR.
    The argument is the TRUE Eb/N0 (for K_sim=4, k=2 it equals snr_per_sample exactly).
    """
    signal_power = photocurrent.detach().pow(2).mean()
    snr_per_sample_db = ebn0_db - 10 * np.log10(samples_per_symbol_sim / (2 * bits_per_symbol))
    noise_power = signal_power / (10 ** (snr_per_sample_db / 10))
    noise = torch.sqrt(noise_power) * torch.randn_like(photocurrent)
    return photocurrent + noise


def measure_ber(bit_probability, transmitted_bits, edge_guard, max_shift=3):
    """Bit error rate with a small symbol-shift search (robust to filter group delay)."""
    decided = (bit_probability > 0.5).to(torch.int64)
    transmitted = transmitted_bits.to(torch.int64)
    num_symbols = decided.shape[1]
    best_ber = 1.0
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        a_start = edge_guard + max(0, shift)
        a_stop = num_symbols - edge_guard + min(0, shift)
        b_start = edge_guard - min(0, shift)
        b_stop = num_symbols - edge_guard - max(0, shift)
        errors = (decided[:, a_start:a_stop] != transmitted[:, b_start:b_stop]).float().mean().item()
        if errors < best_ber:
            best_ber = errors
            best_shift = shift
    return best_ber, best_shift


def ideal_reference_ber(ebn0_db_list, config, num_symbols=300000, device="cuda"):
    """BER of an IDEAL unipolar PAM-4 (levels 0..3, no MZM, ideal slicer) through the
    SAME noise + matched filter + decimation pipeline. Its offset from the textbook
    curve calibrates our Eb/N0 axis: any gap here is purely a definition artifact,
    so the real system's distance from THIS reference is the true impairment.
    """
    from pulse_shaping import root_raised_cosine
    import torch
    import torch.nn.functional as F

    sps = config.samples_per_symbol_sim
    tx_taps = root_raised_cosine(config.rrc_rolloff, config.rrc_span_symbols, sps)
    tx_taps = tx_taps / tx_taps.max()                       # unit-peak (same as transmitter)
    mf_taps = root_raised_cosine(config.rrc_rolloff, config.rrc_span_symbols, sps)  # unit-energy
    tx_kernel = torch.tensor(tx_taps, dtype=torch.float32, device=device).view(1, 1, -1)
    mf_kernel = torch.tensor(mf_taps, dtype=torch.float32, device=device).view(1, 1, -1)

    symbols = torch.randint(0, 4, (num_symbols,), device=device)
    levels = symbols.float()                                # ideal levels 0,1,2,3
    upsampled = torch.zeros(num_symbols * sps, device=device)
    upsampled[::sps] = levels
    clean = F.conv1d(upsampled.view(1, 1, -1), tx_kernel, padding=tx_kernel.shape[-1] // 2)

    bits = torch.stack([(symbols >> 1) & 1, symbols & 1]).to(torch.int64)
    results = []
    for ebn0_db in ebn0_db_list:
        noisy = add_awgn(clean.squeeze(), ebn0_db, config.bits_per_symbol, sps)
        matched = F.conv1d(noisy.view(1, 1, -1), mf_kernel, padding=mf_kernel.shape[-1] // 2).squeeze()
        grid = matched[:num_symbols * sps].reshape(num_symbols, sps)
        phase = grid[200:-200].var(0).argmax()
        decision = grid[:, phase]
        level_means = torch.tensor([decision[symbols == q].mean() for q in range(4)], device=device)
        thresholds = (level_means[:3] + level_means[1:]) / 2
        estimated = torch.bucketize(decision, thresholds)
        est_bits = torch.stack([(estimated >> 1) & 1, estimated & 1]).to(torch.int64)
        ber = (est_bits[:, 200:-200] != bits[:, 200:-200]).float().mean().item()
        results.append(ber)
    return np.array(results)


def decision_phase_by_separation(photocurrent, samples_per_symbol, symbols, skip=100):
    """Phase that best separates the 4 levels (symbol-center), and the level means."""
    n = len(photocurrent) // samples_per_symbol
    grid = photocurrent[:n * samples_per_symbol].reshape(n, samples_per_symbol)
    body = grid[skip:n - skip]
    sym_body = symbols[skip:n - skip]
    separations = [np.array([body[sym_body == q, p].mean() for q in range(4)]).var()
                   for p in range(samples_per_symbol)]
    phase = int(np.argmax(separations))
    levels = np.array([body[sym_body == q, phase].mean() for q in range(4)])
    return phase, levels


def calibrate_ebn0_offset(reference_ber, ebn0_db_list, modulation_order):
    """dB shift that aligns the ideal-reference curve to the textbook curve.
    This is the Eb/N0-definition offset; applying it makes the axis honest."""
    fine = np.arange(0.0, 40.0, 0.05)
    theory = theoretical_ber_unipolar(fine, modulation_order)
    offsets = []
    for ebn0, ber in zip(ebn0_db_list, reference_ber):
        if 1e-5 < ber < 2e-1:
            idx = np.argmin(np.abs(np.log10(np.maximum(theory, 1e-12)) - np.log10(ber)))
            offsets.append(ebn0 - fine[idx])
    return float(np.median(offsets)) if offsets else 0.0


if __name__ == "__main__":
    ebn0 = np.arange(6, 22, 2.0)
    print("Eb/N0 [dB]        :", ebn0)
    print("unipolar (elec)   :", np.array([f"{b:.1e}" for b in theoretical_ber_unipolar(ebn0, 4)]))
    print("unipolar (ASE)    :", np.array([f"{b:.1e}" for b in theoretical_ber_unipolar_ase(ebn0, 4)]))
    print("bipolar  (elec)   :", np.array([f"{b:.1e}" for b in theoretical_ber_bipolar(ebn0, 4)]))

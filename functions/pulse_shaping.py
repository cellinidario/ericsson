"""Root-raised-cosine (RRC) pulse shaping and matched filtering.

Fork A choice: a FIXED RRC at the transmitter and a matched RRC at the receiver.
This makes the pulse Nyquist (ISI-free at symbol instants) so the link can approach
the theoretical PAM BER, and it removes any time-multiplexing degree of freedom
(the predistorter outputs one level per symbol, not per sample).
"""

import numpy as np


def root_raised_cosine(rolloff, span_symbols, samples_per_symbol):
    """Return the unit-energy RRC impulse response sampled at samples_per_symbol.

    rolloff           : excess-bandwidth factor beta in [0, 1]
    span_symbols      : total filter length in symbols (truncation)
    samples_per_symbol: oversampling factor of the returned taps
    """
    num_taps = span_symbols * samples_per_symbol + 1
    time_in_symbols = (np.arange(num_taps) - (num_taps - 1) / 2) / samples_per_symbol
    taps = np.zeros(num_taps)

    for index, t in enumerate(time_in_symbols):
        if abs(t) < 1e-10:
            taps[index] = 1.0 - rolloff + 4 * rolloff / np.pi
        elif rolloff > 0 and abs(abs(t) - 1.0 / (4 * rolloff)) < 1e-10:
            term_a = (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
            term_b = (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            taps[index] = rolloff / np.sqrt(2) * (term_a + term_b)
        else:
            numerator = np.sin(np.pi * t * (1 - rolloff)) + 4 * rolloff * t * np.cos(np.pi * t * (1 + rolloff))
            denominator = np.pi * t * (1 - (4 * rolloff * t) ** 2)
            taps[index] = numerator / denominator

    taps = taps / np.sqrt(np.sum(taps ** 2))
    return taps


def brickwall(cutoff_hz, sample_rate, span_symbols, samples_per_symbol):
    """Unit-energy ideal-rect (brickwall) low-pass FIR: a windowed sinc with cutoff cutoff_hz.
    rect(f) in frequency -> sinc(t) in time. With TX = RX (matched), rect*rect = rect stays
    Nyquist (ISI-free at symbol instants when cutoff is a multiple of half the symbol rate).
    This is Marco's JLT model choice: H_TX = H_RX = rect at the DAC/ADC Nyquist (no RRC).
    """
    num_taps = span_symbols * samples_per_symbol + 1
    n = np.arange(num_taps) - (num_taps - 1) / 2
    fc = cutoff_hz / (sample_rate / 2.0)                  # normalized to Nyquist (1.0 == fs/2)
    taps = fc * np.sinc(fc * n)                           # ideal low-pass sinc
    taps = taps * np.hamming(num_taps)                    # window the truncation
    taps = taps / np.sqrt(np.sum(taps ** 2))             # unit energy (matched-filter consistent)
    return taps


def nrz_pulse(rolloff, span_symbols, samples_per_symbol, rect_cutoff_ratio=None):
    """The prof's band-limited NRZ pulse (trans_func.m 'NRZ'): a rectangle of duration T with
    raised-cosine edges, whose spectrum is H(f) = sinc(f/Rs) * cos(pi*ro*f/Rs)/(1-(2*ro*f/Rs)^2).

    Unlike the ideal frequency rect (brickwall -> a sinc in time, exactly zero at +-T/2), this pulse
    keeps h(+-T/2) ~= 50% of the peak. That value is exactly what a symbol contributes to its
    neighbour's T/2 crossing sample, where the differential SIGN lives (the cross term
    2 Re(x_k x*_{k+1}) under direct detection). Measured 2026-07-23: NRZ 50%, Gaussian-only 25%,
    ideal rect 0% -> the sign BER tracks this number over three orders of magnitude.

    rect_cutoff_ratio: optional ideal low-pass AFTER the NRZ shaping, cutoff in units of Rs
    (1.0 = 20 GHz at 20 GBaud). This is Stella/Marco's real TX (NRZ then 20 GHz rect); it only
    trims the sinc side-lobes and is harmless (h(T/2) 50% -> 45%). None = pure NRZ."""
    n_fft = 1 << 15
    fn = np.fft.fftfreq(n_fft, d=1.0 / samples_per_symbol)          # frequency in units of Rs
    den = 1.0 - (2 * rolloff * fn) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.sinc(fn) * np.cos(np.pi * rolloff * fn) / den
    H[np.abs(den) < 1e-9] = (rolloff / 2 * np.sin(np.pi / (2 * rolloff))) if rolloff > 0 else 0.0
    H[0] = 1.0
    if rect_cutoff_ratio is not None:
        H = H * (np.abs(fn) <= rect_cutoff_ratio)
    h = np.fft.fftshift(np.real(np.fft.ifft(H)))
    center = n_fft // 2
    num_taps = span_symbols * samples_per_symbol + 1
    idx = center + (np.arange(num_taps) - (num_taps - 1) // 2)
    return h[idx]


def upsample_and_shape(symbol_levels, rrc_taps, samples_per_symbol):
    """Zero-stuff the symbol-rate levels by samples_per_symbol, then RRC filter.

    symbol_levels: 1D array, one drive level per symbol
    returns the pulse-shaped waveform at the oversampled rate
    """
    upsampled = np.zeros(len(symbol_levels) * samples_per_symbol)
    upsampled[::samples_per_symbol] = symbol_levels
    waveform = np.convolve(upsampled, rrc_taps, mode="same")
    return waveform


def matched_filter(received_waveform, rrc_taps):
    """Apply the matched (same) RRC filter at the receiver."""
    return np.convolve(received_waveform, rrc_taps, mode="same")


if __name__ == "__main__":
    taps = root_raised_cosine(rolloff=0.85, span_symbols=16, samples_per_symbol=8)
    print(f"RRC taps: {len(taps)}, energy={np.sum(taps**2):.4f}, peak at center={taps[len(taps)//2]:.4f}")
    # Nyquist check: cascade of TX+RX RRC must be ~zero at non-zero symbol instants
    full = np.convolve(taps, taps, mode="full")
    center = len(full) // 2
    at_symbols = full[center::8][:5] / full[center]
    print(f"raised-cosine sampled at symbol instants (should be 1,0,0,..): {np.round(at_symbols, 4)}")

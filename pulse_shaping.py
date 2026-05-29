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

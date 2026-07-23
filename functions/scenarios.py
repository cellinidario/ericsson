"""Reference scenarios of the JLT invited paper — the single source of truth for the chain.

Both the notebooks and the experiment drivers must build their Config from here, so that a
figure can always be reproduced from the repository (before this module, the paper curves were
produced by throwaway drivers and the notebook had drifted to a different chain).

`jlt_chain()` is the agreed chain behind the BER-vs-OSNR / BER-vs-Eb/N0 figures sent to the
co-authors (2026-07-20):

  TX-DSP -> 20 GHz rectangular digital filter -> 10 GHz Gaussian digital filter -> DAC (6 bit)
  -> single-segment MZM (null bias, ER 30 dB) -> 10.238 km C-band SMF (beta2 = -21.7 ps^2/km)
  -> WSS optical filter (1.53 Rs flat top, 18 GHz edges) + ASE on the field
  -> square-law photodiode (25 GHz) -> 10 GHz Gaussian digital filter
  -> 2 samples/symbol -> ADC (N_ADC bit, ideal AGC) -> RX-DSP (per-bit sigmoid head)

Modulation: BPAM-4 discovered end to end (no imposed level spacing / precoding / pulse shape).
Axis convention (Luca, 2026-07-16): OSNR = Eb/N0 + 10log10(2) + 10log10(Rs/(2 Bref))
                                        = Eb/N0 + 2.04 dB   (Rs = 20 GBaud, Bref = 12.5 GHz).
"""

from config import Config

# OSNR = Eb/N0 + OSNR_OFFSET_DB. The measured grids use 2.05; the exact value is 10log10(1.6) =
# 2.041 dB. The 0.01 dB difference is far below the run-to-run spread and is kept for continuity
# with the already published points.
OSNR_OFFSET_DB = 2.05

TRAIN_EBN0_DB = 14.0        # fixed training point: training at 20 dB lands in a bad basin with
                            # the narrow filters; 14 dB is the agreed operating point
TRAIN_STEPS = 100_000


def jlt_chain(width=16, depth=1, adc_bits=None, dac_bits=6, band="cband",
              rx_window_symbols=11, sps=2):
    """Config of the agreed JLT chain. Defaults reproduce the paper curve (W=16, D=1, N_DAC=6,
    ideal ADC). Set adc_bits=8/6/5 for the ADC-resolution family.

    width/depth apply to both the TX-DSP (DPD) and the RX-DSP (FFE). The 5-symbol memory limit
    of the TX-DSP (Ericsson LUT constraint, 2^10 entries) is part of the Config default and is
    NOT touched here; rx_window_symbols is free (measured neutral between 11 and 31).

    sps = samples_per_symbol_sim (internal simulation rate). Default 2: the study of 2026-07-22
    measured the E2E BER unchanged from 2 to 16 sps, so 2 (the true system oversampling) is
    enough for the end-to-end case. The classical-BPAM RX-only case needs 4 to resolve the T/2
    sign cross-term -- use bpam_classic_rx(), which sets it.
    """
    cfg = Config()
    cfg.set_wavelength_band(band)
    cfg.noise_regime = "ase"
    cfg.modulator = "mzm"                    # null-biased MZM, ER 30 dB (Config default)

    # digital filters of the agreed chain
    cfg.tx_filter = "freq-rect"              # 20 GHz brick-wall on the drive
    cfg.rx_filter = "freq-rect"
    cfg.tx_gaussian_bw = None                # NO TX Gaussian: the 20 GHz rect drive goes straight to the
                                             # MZM. Removing it recovers the bandwidth the group's baseline
                                             # has (they filter at 10 GHz, we filtered at ~7.1 equivalent).
                                             # NB: this does NOT make our TX equal to the baseline's -- the
                                             # baseline pulse is NRZ (time-rectangular, roll-off 0.85), ours
                                             # is a 20 GHz brick-wall. Harmless for the E2E, which learns its
                                             # own waveform on top of this basis; NOT harmless for a FIXED
                                             # transmitter -- see bpam_classic_rx, which uses NRZ instead.
    cfg.rx_gaussian_bw = 10e9                # Gaussian on the photocurrent, before decimation

    # converters (straight-through estimator -> quantization-aware training)
    cfg.dac_bits = dac_bits
    cfg.adc_bits = adc_bits

    # end-to-end autoencoder
    cfg.equalizer = "end-to-end"
    cfg.dpd_hidden_width = width
    cfg.dpd_hidden_layers = depth
    cfg.ffe_hidden_width = width
    cfg.ffe_hidden_layers = depth
    cfg.ffe_memory_symbols = rx_window_symbols

    cfg.samples_per_symbol_sim = sps          # internal simulation rate (see docstring)
    cfg._compute_derived()
    cfg.set_modulation_format("bpam-4")      # recomputes the derived quantities
    return cfg


def asfand_complex(adc_bits=None, precode=False, dac_bits=6, band="cband", sps=2):
    """The 'Asfand-comparable' configuration (Luca, mail 2026-07-20): same JLT chain, but the
    RX-DSP is Asfand's 'complex' NN — three fully-connected hidden layers (32, 64, 16) with a
    sigmoid output, context window of 5 symbols. The TX-DSP stays ours (Luca: 'or better if
    you even use TX optimization').

    sps = samples_per_symbol_sim (internal simulation rate). Default 2: the study of 2026-07-22
    showed the E2E BER is unchanged from 2 to 16 sps, so 2 (the true system oversampling) is
    enough. (Classical-BPAM RX-only needs 4 to resolve the T/2 sign cross-term -- handled
    separately when needed.)

    precode=True applies Marco's proposal (meeting 2026-07-20): the DPD input is the
    DIFFERENTIALLY ENCODED bit stream while loss and BER stay on the raw bits — the raw phase
    bit is then carried by the sign TRANSITION between adjacent symbols, which a finite RX
    window can see (the absolute sign, which needs infinite memory, no longer matters).
    """
    cfg = jlt_chain(width=16, depth=1, adc_bits=adc_bits, dac_bits=dac_bits, band=band,
                    rx_window_symbols=5)
    cfg.ffe_hidden_widths = [32, 64, 16]
    cfg.bpam_precode_e2e = bool(precode)
    cfg.samples_per_symbol_sim = sps
    cfg._compute_derived()
    cfg.set_modulation_format("bpam-4")      # recompute derived quantities after the overrides
    return cfg


def bpam_classic_rx(adc_bits=None, dac_bits=6, band="cband", rx_window_symbols=6, vpeak=0.6):
    """Standard BPAM transmitter (fixed levels + differential precoder, LINEAR MZM drive at the
    prof's Vpeak=0.6) with the NN receiver only -- the RX-only / receiver-side case, to compare
    against the E2E. This reproduces in the PyTorch chain the result validated on the prof's MATLAB
    simulator: RX-only NN on Asfand's curve (2026-07-23).

    TX pulse = NRZ (band-limited, roll-off 0.85), which keeps h(T/2) ~ 50% so the differential SIGN
    survives in the T/2 cross-term of adjacent pulses. The E2E's ideal frequency rect nulls h(T/2)
    and destroys the sign for a FIXED transmitter (the E2E gets away with it because the DPD learns
    its own waveform, writing the crossing sample directly). Any TX filter after the NRZ pulse only
    hurts: an ideal 20 GHz rect is harmless (h(T/2) 50% -> 45%, cfg.nrz_rect_cutoff_ratio=1.0), a
    10 GHz Gaussian costs ~0.5 dB.

    Receiver: one NN 64-128-32 with sigmoid outputs, single context window of 6 whole symbols
    (12 samples at 2 sps) -- the union of Asfand's two staggered 11-sample windows. Measured
    equivalent to two independent nets and to staggered windows (within 3%), so the simplest form.

    Uses sps=4: the sign lives in the T/2 cross-term, which needs >2 samples/symbol to be resolved.
    """
    cfg = jlt_chain(width=16, depth=1, adc_bits=adc_bits, dac_bits=dac_bits, band=band,
                    rx_window_symbols=rx_window_symbols)
    cfg.ffe_hidden_widths = [64, 128, 32]          # doubled-neuron RX (Stella), single window (Marco)
    cfg.equalizer = "ffe"                          # RX-only: fixed standard BPAM at the TX
    cfg.tx_filter = "nrz"                           # band-limited NRZ pulse (keeps the sign; see above)
    cfg.tx_gaussian_bw = None                       # no TX Gaussian (it halves h(T/2))
    cfg.bpam_classic_drive_swing = vpeak           # 0.6 = the reference simulator (linear-eq optimum)
    cfg.dac_fullscale_from_signal = True            # DAC full-scale = actual drive peak (= the prof)
    cfg.samples_per_symbol_sim = 4                 # 4 sps to resolve the T/2 sign cross-term
    cfg._compute_derived()
    cfg.set_modulation_format("bpam-4")
    return cfg


def checkpoint_name(cfg, width, depth, adc_bits, seed, steps, band="cband"):
    """Checkpoint file name that encodes the architecture too. The legacy notebook name only
    encoded band/regime/format, so two runs with different W/D/N_ADC silently overwrote each
    other -- a known trap of this project."""
    adc = "idealADC" if adc_bits is None else f"adc{adc_bits}"
    dac = "idealDAC" if cfg.dac_bits is None else f"dac{cfg.dac_bits}"
    widths = getattr(cfg, "ffe_hidden_widths", None)
    arch = ("W" + "-".join(str(w) for w in widths)) if widths else f"W{width}_D{depth}"
    win = getattr(cfg, "ffe_memory_symbols", 11)
    pre = "_diffpre" if getattr(cfg, "bpam_precode_e2e", False) else ""
    sps = getattr(cfg, "samples_per_symbol_sim", 4)      # encode sps: a 2-sps run must not load a 4-sps ckpt
    return (f"results/jlt_{band}_bpam4_{arch}_win{win}_sps{sps}_{dac}_{adc}{pre}"
            f"_seed{seed}_{steps // 1000}k.pt")

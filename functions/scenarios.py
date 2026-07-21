"""Reference scenarios of the JLT invited paper — the single source of truth for the chain.

Both the notebooks and the experiment drivers must build their Config from here, so that a
figure can always be reproduced from the repository (before this module, the paper curves were
produced by throwaway drivers and the notebook had drifted to a different chain).

`jlt_chain()` is the agreed chain behind the BER-vs-OSNR / BER-vs-Eb/N0 figures sent to the
co-authors (2026-07-20):

  TX-DSP -> 20 GHz rectangular digital filter -> 10 GHz Gaussian digital filter -> DAC (6 bit)
  -> single-segment MZM (null bias, ER 25 dB) -> 10.238 km C-band SMF (beta2 = -21.7 ps^2/km)
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
              rx_window_symbols=11):
    """Config of the agreed JLT chain. Defaults reproduce the paper curve (W=16, D=1, N_DAC=6,
    ideal ADC). Set adc_bits=8/6/5 for the ADC-resolution family.

    width/depth apply to both the TX-DSP (DPD) and the RX-DSP (FFE). The 5-symbol memory limit
    of the TX-DSP (Ericsson LUT constraint, 2^10 entries) is part of the Config default and is
    NOT touched here; rx_window_symbols is free (measured neutral between 11 and 31).
    """
    cfg = Config()
    cfg.set_wavelength_band(band)
    cfg.noise_regime = "ase"
    cfg.modulator = "mzm"                    # null-biased MZM, ER 25 dB (Config default)

    # digital filters of the agreed chain
    cfg.tx_filter = "freq-rect"              # 20 GHz brick-wall on the drive
    cfg.rx_filter = "freq-rect"
    cfg.tx_gaussian_bw = 10e9                # Gaussian cascaded after the TX pulse filter
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

    cfg.set_modulation_format("bpam-4")      # recomputes the derived quantities
    return cfg


def asfand_complex(adc_bits=None, precode=False, dac_bits=6, band="cband"):
    """The 'Asfand-comparable' configuration (Luca, mail 2026-07-20): same JLT chain, but the
    RX-DSP is Asfand's 'complex' NN — three fully-connected hidden layers (32, 64, 16) with a
    sigmoid output, context window of 5 symbols. The TX-DSP stays ours (Luca: 'or better if
    you even use TX optimization').

    precode=True applies Marco's proposal (meeting 2026-07-20): the DPD input is the
    DIFFERENTIALLY ENCODED bit stream while loss and BER stay on the raw bits — the raw phase
    bit is then carried by the sign TRANSITION between adjacent symbols, which a finite RX
    window can see (the absolute sign, which needs infinite memory, no longer matters).
    """
    cfg = jlt_chain(width=16, depth=1, adc_bits=adc_bits, dac_bits=dac_bits, band=band,
                    rx_window_symbols=5)
    cfg.ffe_hidden_widths = [32, 64, 16]
    cfg.bpam_precode_e2e = bool(precode)
    cfg.set_modulation_format("bpam-4")      # recompute derived quantities after the overrides
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
    return (f"results/jlt_{band}_bpam4_{arch}_win{win}_{dac}_{adc}{pre}"
            f"_seed{seed}_{steps // 1000}k.pt")

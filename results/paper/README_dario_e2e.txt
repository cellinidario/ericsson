E2E (Dario) — BER curves for fig_results_e2e
=============================================
Chain: 20 GBd, TX 20 GHz rect digital filter (+10 GHz Gaussian cascade where noted),
RX 10 GHz Gaussian digital filter, WSS optical filter, MZM (ER 25 dB), 10.238 km,
N_DAC = 6 available via knob (curves below: ideal converters unless *_ndac6).
TX-DSP: FC, context window 2L'+1 = 5 symbols, hidden 64x2, tanh out, s = 2 samples/symbol.
RX-DSP: FC, window 2L''+1 = 11 symbols at s' = 2 (22 taps), hidden 64x2, softmax ->
        marginalized per-bit posteriors z_k. Loss: BCE + auxiliary symbol-CE.
Training: Adam 1e-3, 1e5 x 8192-symbol online minibatches, at Eb/N0 = 14 dB.

AXIS NOTE (IMPORTANT): col2 uses the NOMINAL physical mapping
    OSNR = Eb/N0 + 10*log10(Rb/Bref) = Eb/N0 + 5.05 dB   (Rb = 40 Gb/s, Bref = 12.5 GHz, single pol.)
Our attempts to calibrate this offset by replicating the "ideal AE" / "BPAM+AE" reference
curves failed because the AE receiver structure is not specified enough for a faithful
replica (a plain linear equalizer floors on BPAM in our chain). Once the OSNR convention
(Bref / polarizations) or the AE spec is agreed, the axis is a single constant shift;
the BER values (col3) do not change.

Files (columns: EbN0[dB]  OSNR_nominal[dB]  BER  theory_unipolar_ASE):
  bpam_b2b_ber.txt, bpam_oband_ber.txt, bpam_cband_ber.txt   (E2E bipolar, discovered signaling)
  upam_b2b_ber.txt, upam_oband_ber.txt, upam_cband_ber.txt   (E2E unipolar, same chain)
Deep-tail points (18-20 dB) evaluated with 12-24M symbols.

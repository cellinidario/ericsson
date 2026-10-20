"""End-to-end optimization of IM/DD links through a digital surrogate.

Modules:
    parameters         link settings (LinkParameters), procedures (Budget), cases (Case)
    filters            filter designs and quantizers
    link               the link from the TX DSP output to the RX DSP input
    dsp                conventional transmitters, TX NN, TX lookup table, RX NN, loss and decisions
    digital_surrogate  fit of the analog FIRs and identification audits
    training           training stages, restartable
    evaluation         BER on held-out blocks, level at BER 1e-3, comparisons
    linear_equalizer   least-squares linear equalizer (reference receiver)
    pipeline           one case end to end
    files              atomic writes of JSON and checkpoint files
"""

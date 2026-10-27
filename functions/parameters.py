"""Configuration of the simulated link, of the DSP architectures and of the procedures.

Every block of the link reads its settings from LinkParameters; the defaults describe a
20-GBd, 40-Gbps IM/DD link over 10.238 km of standard single-mode fiber in the C-band.
"""
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Optional

PULSES = ('nrz', 'rrc')
TX_FILTERS = ('rectangular', 'gaussian', 'none')
OPTICAL_FILTERS = ('waveshaper', 'gaussian', 'supergaussian', 'none')
RX_FILTERS = ('gaussian', 'none')
REGIMES = ('electrical', 'optical')
FORMATS = ('pam4', 'bpam')
INITIALIZATIONS = ('conventional', 'receiver', 'random')
PRECODERS = (True, False, 'second', 'both')
POWER_REFERENCES = ('mean', 'laser')
MZM_BIASES = ('null', 'quadrature')

# Settings added after results were stored, with the values that reproduce the earlier behavior:
# at those values they are left out of the keys, so earlier folders and checkpoints keep their names.
LEGACY_VALUES = {'power_reference': (None, 'mean'), 'optical_supergaussian_order': (4,),
                 'rx_filter_before_adc': (False,), 'mzm_bias': ('null',), 'rx_sps': (None,),
                 'tx_filter_after_dac': (False,), 'tx_filter_in_surrogate': (False,)}


def identity(settings):
    """asdict() of a Case, LinkParameters or Budget for keys and signatures, without the settings of
    LEGACY_VALUES at their earlier behavior (nested dictionaries included)."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if not (k in LEGACY_VALUES and v in LEGACY_VALUES[k])}
        return value
    return strip(asdict(settings))


@dataclass(frozen=True)
class LinkParameters:
    # Format and rates
    symbol_rate: float = 20e9            # 20 GBd; two bits per symbol give 40 Gbps
    bits_per_symbol: int = 2
    dsp_sps: int = 2                     # samples per symbol of the TX and RX DSP (and of DAC and ADC)
    rx_sps: Optional[int] = None         # samples per symbol of the ADC and RX DSP if different from dsp_sps
                                         # (1: the ADC samples only the pulse centres; PAM-4 only)
    analog_sps: int = 2                  # samples per symbol of the analog simulation
    guard_symbols: int = 128             # symbols excluded at each block edge from losses and BER counts

    # Transmitter
    tx_pulse: str = 'nrz'                # 'nrz': rectangle with raised-cosine edges; 'rrc': root raised cosine
    tx_pulse_rolloff: float = .85
    tx_pulse_span_symbols: int = 16
    tx_filter: str = 'rectangular'       # H_TX: 'rectangular', 'gaussian' (3-dB bandwidth) or 'none'
    tx_filter_after_dac: bool = True     # True: analog H_TX after the DAC, the band limitation of DAC and driver
                                         # electronics; False: digital H_TX before the DAC
    tx_filter_in_surrogate: bool = True  # the analog H_TX is part of the analog link, merged with the driver response,
                                         # and the digital surrogate refits it; False: known and applied exactly,
                                         # outside the digital surrogate. Always False for a digital H_TX
    dac_bits: Optional[int] = 6          # None: ideal DAC. Full scale = peak of each block (no clipping)
    peak_fraction: float = .6            # peak voltage at the modulator input, in units of V_pi

    # Driver and modulator (push-pull Mach-Zehnder modulator)
    driver_bandwidth: float = 30e9
    vpi: float = 5.
    extinction_ratio_db: float = 30.
    mzm_bias: str = 'null'               # 'null': drive +-peak_fraction V_pi around the null (BPAM, the default);
                                         # 'quadrature' (PAM-4): bias V_pi/2, drive +-min(peak_fraction, 1/2) V_pi,
                                         # i.e. the full intensity range within the same peak-to-peak driver swing

    # Fiber: chromatic dispersion and loss, no Kerr nonlinearity
    fiber_length_km: float = 10.
    beta2_ps2_per_km: float = -21.7      # C-band standard single-mode fiber
    fiber_loss_db_per_km: float = .2

    # Optical filter H_opt before the photodiode
    optical_filter: str = 'waveshaper'   # 'waveshaper' (flat top with Gaussian edges), 'gaussian', 'supergaussian'
                                         # or 'none'
    waveshaper_width: float = 1.53       # flat-top full width, in units of the symbol rate (30.6 GHz)
    waveshaper_edge_bandwidth: float = 18e9  # 3-dB width of the Gaussian that shapes the edges
    optical_gaussian_bandwidth: float = 30e9  # full 3-dB width of the 'gaussian' and 'supergaussian' options
    optical_supergaussian_order: int = 4  # order of the 'supergaussian' option (1: Gaussian)

    # Photodiode and receiver front end
    photodiode_bandwidth: float = 25e9
    adc_bits: Optional[int] = 6          # None: ideal ADC. Samples the photodiode output (no anti-aliasing);
                                         # range = observed range of each block
    rx_filter: str = 'gaussian'          # H_RX: 'gaussian' or 'none'
    rx_filter_bandwidth: float = 10e9
    rx_filter_before_adc: bool = False   # False: digital H_RX after the ADC; True: analog H_RX before the ADC,
                                         # the band limitation of photodiode and ADC electronics, with the
                                         # thermal noise added before it, refitted by the digital surrogate
    osnr_reference_bandwidth: float = 12.5e9  # OSNR in 0.1 nm, ASE on two polarizations

    # Reference of the noise level. None: set by the regime of the Case ('laser' for thermal noise,
    # 'mean' for ASE noise); a LinkParameters used without a Case takes 'mean'.
    # 'mean':  drive scaled to the peak voltage at the modulator input and field scaled to unit mean
    #          optical power, for every TX (simulator of Secondini; with an optical amplifier).
    # 'laser': the DAC full scale is the peak of the TX samples before H_TX (with a digital H_TX the
    #          filtered signal is not rescaled; a rectangular H_TX can clip), the driver
    #          gain is fixed (DAC full scale -> peak voltage) and the laser power is fixed: the field is
    #          divided by the mean optical power of the conventional PAM-4 TX on the same link, so a TX
    #          that pre-emphasizes or lowers its mean power pays for it in SNR.
    power_reference: Optional[str] = None

    # DSP architectures
    tx_memory_symbols: int = 5           # TX NN context: symbols k-2 ... k+2, two bits each
    tx_hidden: int = 16                  # neurons of the hidden layer; a tuple gives several layers
    rx_window: int = 11                  # RX NN context: 11 samples at two samples per symbol
    rx_hidden: tuple = (32, 64, 16)

    def __post_init__(self):
        for value, allowed in ((self.tx_pulse, PULSES), (self.tx_filter, TX_FILTERS),
                               (self.optical_filter, OPTICAL_FILTERS), (self.rx_filter, RX_FILTERS),
                               (self.mzm_bias, MZM_BIASES)):
            if value not in allowed:
                raise ValueError(f'{value!r} is not one of {allowed}.')
        if self.power_reference not in (None,)+POWER_REFERENCES:
            raise ValueError(f'{self.power_reference!r} is not one of {POWER_REFERENCES}.')
        if self.analog_sps % self.dsp_sps or self.analog_sps % self.adc_sps:
            raise ValueError('The analog rate must be a multiple of the DSP and ADC rates.')
        if not self.tx_filter_after_dac:             # only an analog H_TX can be refitted
            object.__setattr__(self, 'tx_filter_in_surrogate', False)

    @property
    def adc_sps(self):
        """Samples per symbol of the ADC and RX DSP."""
        return self.dsp_sps if self.rx_sps is None else self.rx_sps

    def sample_rate(self, sps=None):
        return (self.analog_sps if sps is None else sps)*self.symbol_rate

    def fir_taps(self, sps=None):
        """Every FIR spans 32 samples per symbol on each side of its centre tap."""
        return 32*(self.analog_sps if sps is None else sps)+1

    @property
    def peak_volt(self):
        """Peak drive voltage around the bias point (at quadrature, beyond V_pi/2 the intensity stops growing)."""
        return (min(self.peak_fraction, .5) if self.mzm_bias == 'quadrature' else self.peak_fraction)*self.vpi

    @property
    def mzm_gamma(self):
        """Arm imbalance setting the finite extinction ratio: E = (exp(j phi) + gamma exp(-j phi))/2."""
        er = 10**(self.extinction_ratio_db/10)
        return (math.sqrt(er)-1)/(math.sqrt(er)+1)

    @property
    def field_loss(self):
        return math.sqrt(10**(-self.fiber_loss_db_per_km*self.fiber_length_km/10))

    @property
    def dispersion_coefficient(self):
        """beta2 L / 2 in s^2: the fiber multiplies the field spectrum by exp(j coefficient omega^2)."""
        return .5*(self.beta2_ps2_per_km*1e-27)*(self.fiber_length_km*1e3)

    def analog_part(self):
        """Settings of the analog link only (what a digital surrogate has to learn)."""
        keys = ('symbol_rate', 'bits_per_symbol', 'analog_sps', 'driver_bandwidth', 'vpi', 'extinction_ratio_db',
                'fiber_length_km', 'beta2_ps2_per_km', 'fiber_loss_db_per_km', 'optical_filter', 'waveshaper_width',
                'waveshaper_edge_bandwidth', 'optical_gaussian_bandwidth', 'photodiode_bandwidth')
        if self.optical_filter == 'supergaussian':   # only then, so that earlier surrogates keep their keys
            keys += ('optical_supergaussian_order',)
        if self.rx_filter_before_adc:                # the analog H_RX is part of the digital surrogate
            keys += ('rx_filter', 'rx_filter_bandwidth', 'rx_filter_before_adc')
        if self.mzm_bias != 'null':                  # bias and drive range of the modulator
            keys += ('mzm_bias',)
        if self.tx_filter_in_surrogate:              # H_TX belongs to the analog link (its bandwidth is the Case's)
            keys += ('tx_filter', 'tx_filter_in_surrogate')
        return {k: getattr(self, k) for k in keys}


@dataclass(frozen=True)
class Budget:
    """Number of updates, symbols and blocks of every procedure."""
    surrogate_steps: int = 40000
    rx_steps: int = 100000               # RX NN with the conventional TX
    tx_steps: int = 8000                 # TX NN fitted to the conventional voltages
    joint_steps: int = 100000            # TX NN and RX NN together (E2E)
    rx_only_steps: int = 100000          # RX-only reference: the RX continues with the conventional TX
    continuation_steps: int = 20000      # E2E continued with the TX NN, or with the TX lookup table
    symbols_per_update: int = 8448
    eval_blocks: int = 128               # held-out blocks per noise level, at least
    eval_max_blocks: int = 8192          # ... and more, up to this number, until eval_target_errors errors
    eval_target_errors: int = 200
    ber_floor: float = 1e-7              # levels above the first one with a BER below this are not counted
                                         # (one decade below the 1e-6 bottom of the BER figures)
    symbols_per_block: int = 32768
    linear_fit_blocks: int = 32
    linear_calibration_blocks: int = 8
    check_sps: tuple = (4, 16)           # analog rates of the sampling check

    def __post_init__(self):
        if not self.eval_blocks <= self.eval_max_blocks <= 10000:
            raise ValueError('Held-out blocks: eval_blocks <= eval_max_blocks <= 10000 (10000 seeds per level).')

    def training_identity(self):
        """The settings that define the trained DSPs (evaluation settings excluded)."""
        keys = ('surrogate_steps', 'rx_steps', 'tx_steps', 'joint_steps', 'rx_only_steps', 'continuation_steps',
                'symbols_per_update')
        return {k: getattr(self, k) for k in keys}


# Noise levels for training (OSNR in dB for ASE, SNR in dB for thermal noise) and evaluation.
TRAINING_LEVEL_DB = {'optical': 16.05, 'electrical': 18.92}
EVALUATION_LEVEL_DB = {
    'optical': [12., 13., 14., 14.5, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.5, 19., 20., 21., 22., 24.],
    'electrical': [14., 15., 16., 17., 17.5, 18., 18.5, 19., 19.5, 20., 20.5, 21., 22., 23., 24., 26.]}


@dataclass(frozen=True)
class Case:
    """One simulated configuration: noise regime, format, TX filter bandwidth and link settings."""
    regime: str                          # 'electrical' (thermal noise) or 'optical' (ASE noise)
    fmt: str                             # 'pam4' or 'bpam'
    tx_bandwidth_ghz: float
    precode: object = True               # differential precoding: True = BPAM sign bit only (none for PAM-4),
                                         # False = none, 'second' or 'both' = those bits, either format
    training_level_db: Optional[float] = None
    link: LinkParameters = field(default_factory=LinkParameters)
    seed: int = 0
    label: str = ''                      # optional suffix of the name
    initialization: str = 'conventional'  # E2E start: 'conventional' transceiver, 'receiver' (RX initialized with the
                                          # conventional TX, TX NN from random weights) or 'random' weights

    def __post_init__(self):
        if self.regime not in REGIMES or self.fmt not in FORMATS:
            raise ValueError(f'Unknown regime {self.regime!r} or format {self.fmt!r}.')
        if self.initialization not in INITIALIZATIONS:
            raise ValueError(f'{self.initialization!r} is not one of {INITIALIZATIONS}.')
        if self.precode not in PRECODERS:
            raise ValueError(f'{self.precode!r} is not one of {PRECODERS}.')
        if self.training_level_db is None:
            object.__setattr__(self, 'training_level_db', TRAINING_LEVEL_DB[self.regime])
        if self.link.power_reference is None:
            reference = 'laser' if self.regime == 'electrical' else 'mean'
            object.__setattr__(self, 'link', replace(self.link, power_reference=reference))

    @property
    def name(self):
        return f'{self.regime}_{self.fmt}_bw{self.tx_bandwidth_ghz:g}' + (f'_{self.label}' if self.label else '')

    def key(self):
        """Short hash of everything that defines the case: changes of any setting give new results."""
        text = json.dumps(identity(self), sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()[:10]

    def with_link(self, **changes):
        return replace(self, link=replace(self.link, **changes))

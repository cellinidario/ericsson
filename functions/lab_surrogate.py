"""Learned DAC-drive -> acquired RX-DSP-input boundary, separate from paper physics.

Input units are V/Vpi, NOT volts or integer codes. Output retains acquisition units
and rate. No extra noise, ADC, RX filter, centering or sample-rate conversion.
The model predicts the conditional mean, not the random noise realization.
Learned blocks are an effective cascade, not uniquely identified lab components.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LabSurrogateSpec:
    input_taps: int = 17
    field_taps: int = 65
    output_taps: int = 33
    samples_per_symbol: int = 4
    sample_rate_hz: float = 80e9
    symbol_rate_hz: float = 20e9
    dac_bits: int = 6
    drive_limit_vpi: float = 0.6

    def __post_init__(self):
        if any(k < 1 or k % 2 != 1 for k in
               (self.input_taps, self.field_taps, self.output_taps)):
            raise ValueError("FIR lengths must be positive odd integers")
        if self.samples_per_symbol < 1 or not math.isclose(
                self.sample_rate_hz / self.symbol_rate_hz, self.samples_per_symbol):
            raise ValueError("Sample and symbol rates are inconsistent")
        if self.dac_bits < 1 or self.drive_limit_vpi <= 0:
            raise ValueError("Invalid DAC specification")


def codes_to_drive(codes, bits=6, limit_vpi=0.6):
    """Endpoint mapping with fixed full-scale; never subtract the empirical mean."""
    x = torch.as_tensor(codes)
    if not torch.isfinite(x).all() or (x < 0).any() or (x > 2**bits - 1).any():
        raise ValueError("DAC codes must be finite and within the specified range")
    if not torch.equal(x, x.round()):
        raise ValueError("Recorded DAC codes must be integers")
    return -limit_vpi + 2 * limit_vpi * x.float() / (2**bits - 1)


def _identity_fir(taps):
    layer = nn.Conv1d(1, 1, taps, padding=taps // 2, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[0, 0, taps // 2] = 1
    return layer


class LabSurrogate(nn.Module):
    """Physics-structured finite-memory regression at the measured I/O boundary.

    Centered FIRs implement the already delay-aligned channel. The phase starts at
    a transmission minimum but is learned; a lab bias setting is not hard-coded.
    The imaginary branch starts small, representing finite extinction, and the
    complex FIR can learn effective dispersive memory. No physical ER is inferred.
    Padding is for shape preservation only: discard `halo` samples at record edges.
    """

    def __init__(self, spec=None):
        super().__init__()
        self.spec = spec or LabSurrogateSpec()
        self.input_fir = _identity_fir(self.spec.input_taps)
        self.phase = nn.Parameter(torch.tensor(0.0))
        self.quadrature_gain = nn.Parameter(torch.tensor(0.03))
        self.field_real = _identity_fir(self.spec.field_taps)
        self.field_imag = _identity_fir(self.spec.field_taps)
        with torch.no_grad():
            self.field_imag.weight.zero_()
        self.output_fir = _identity_fir(self.spec.output_taps)
        self.output_gain = nn.Parameter(torch.tensor(4.0))
        self.output_offset = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    @property
    def halo(self):
        return sum((k - 1) // 2 for k in
                   (self.spec.input_taps, self.spec.field_taps, self.spec.output_taps))

    def forward(self, drive_vpi):
        """Accept (N,), (1,N), or (batch,1,N), returning the same shape."""
        shape = drive_vpi.shape
        if drive_vpi.ndim not in (1, 2, 3) or (drive_vpi.ndim > 1 and shape[-2] != 1):
            raise ValueError("Expected (N,), (1,N), or (batch,1,N)")
        x = drive_vpi.reshape(-1, 1, shape[-1])
        phase = math.pi / 2 * self.input_fir(x) + self.phase
        re = torch.sin(phase)
        im = self.quadrature_gain * torch.cos(phase)
        er = self.field_real(re) - self.field_imag(im)
        ei = self.field_real(im) + self.field_imag(re)
        power = er.square() + ei.square()
        y = self.output_gain * self.output_fir(power) + self.output_offset
        return y.reshape(shape)

    def freeze(self):
        """Freeze weights, NOT autograd through the input."""
        self.eval()
        self.requires_grad_(False)
        return self

    def checkpoint(self, metadata):
        return {"format": "lab_surrogate_v1", "spec": asdict(self.spec),
                "state_dict": self.state_dict(), "metadata": metadata}


def load_lab_surrogate(path, device="cpu", frozen=True):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "lab_surrogate_v1":
        raise ValueError("Not a laboratory-surrogate checkpoint")
    model = LabSurrogate(LabSurrogateSpec(**checkpoint["spec"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if frozen:
        model.freeze()
    return model, checkpoint["metadata"]


def make_lab_receiver(config, spec):
    """Reuse the RX classifier without repeating the acquired-signal front end.

    Preserve the context duration in symbols, not the old number of samples.
    Existing 2-sps receiver weights are not compatible with the new input layer.
    This does not modify the caller's config or the paper's default Receiver.
    """
    from receiver import Receiver
    cfg = deepcopy(config)
    if cfg.modulation_format != "bpam-4":
        raise ValueError("This adapter currently supports the BPAM receiver only")
    cfg.symbol_rate = spec.symbol_rate_hz
    cfg.samples_per_symbol_sim = spec.samples_per_symbol
    cfg.samples_per_symbol_rx = spec.samples_per_symbol
    cfg.rx_gaussian_bw = None
    cfg.adc_bits = None
    cfg.rx_decimation_phase = 0
    if getattr(cfg, "rx_dual_network", False):
        # A half-symbol stagger means TWO samples at 4 sps, not one.
        cfg.rx_phase_window_offset = spec.samples_per_symbol // 2
    return Receiver(cfg)


class LabDSPLink(nn.Module):
    """Explicit future DSP training path; no legacy simulated noise wrapper.

    TX supplies post-DAC drive in volts at 4 sps and the fixed calibrated range.
    RX is made with make_lab_receiver. This helper does not train either DSP.
    Hardware transfer still needs surrogate validation on new TX waveforms.
    """

    def __init__(self, transmitter, surrogate, receiver, vpi_volt, tx_sample_rate_hz):
        super().__init__()
        if vpi_volt <= 0 or not math.isclose(tx_sample_rate_hz, surrogate.spec.sample_rate_hz):
            raise ValueError("Explicit positive Vpi and matching TX sample rate required")
        if receiver.rx_gauss is not None or receiver.adc_bits is not None or receiver.decimation != 1:
            raise ValueError("RX must consume acquired samples without another front end")
        if receiver.samples_per_symbol_rx != surrogate.spec.samples_per_symbol:
            raise ValueError("RX sample rate mismatch")
        self.transmitter = transmitter
        self.surrogate = surrogate.freeze()
        self.receiver = receiver
        self.vpi_volt = float(vpi_volt)

    def forward(self, bits):
        drive = self.transmitter(bits) / self.vpi_volt
        if drive.detach().abs().max() > self.surrogate.spec.drive_limit_vpi + 1e-5:
            raise ValueError("TX drive exceeds the recorded DAC full-scale")
        received = self.surrogate(drive).reshape(-1)
        return self.receiver.bit_and_symbol_logits(received)

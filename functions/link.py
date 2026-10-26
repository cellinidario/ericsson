"""The link between the TX DSP output and the RX DSP input, block by block, in signal order.

    x[n] -> pulse shaping -> H_TX -> DAC -> reconstruction -> driver -> peak scaling -> MZM
         -> unit power -> fiber -> + ASE -> H_opt -> |.|^2 -> photodiode -> sampling at the ADC
         rate -> + thermal noise -> ADC -> H_RX -> y[n]
    (with LinkParameters.rx_filter_before_adc: ... -> photodiode -> + thermal noise -> analog H_RX
     -> sampling -> quantization -> y[n], H_RX modeling the photodiode and ADC electronics;
     with LinkParameters.tx_filter_after_dac: x[n] -> pulse shaping -> DAC -> reconstruction -> analog
     H_TX -> driver -> ..., H_TX modeling the DAC and driver electronics)

Pulse shaping, H_TX, DAC and ADC are known exactly, and so is the digital H_RX after the ADC; an
analog H_TX after the DAC is known too and applied exactly, outside the digital surrogate. The analog
part (driver, modulator, fiber, H_opt, photodiode, and H_RX when it is analog) is what a digital
surrogate replaces during training: the surrogate is this same module with its analog FIRs refitted
on input/output data (driver, optical_filter and photodiode; driver, optical_filter and rx_filter
when H_RX is analog, the photodiode response then being kept as in the reference: the identity at
two samples per symbol).

Conventions (power reference 'mean', the default for ASE noise):
- the drive of every transmitter is scaled to the same peak voltage, over the whole block;
- the modulator output is normalized to unit mean optical power, over the whole block;
With the power reference 'laser' (the default for thermal noise, LinkParameters.power_reference)
the DAC full scale is set by the TX samples before H_TX, and the driver gain and the laser power
are fixed instead.
- ASE: complex white Gaussian noise on the field before H_opt; the OSNR is referred to
  12.5 GHz and to the ASE of two polarizations;
- the ADC samples the photodiode output at two samples per symbol (or one, rx_sps=1: the
  pulse centres), without an anti-aliasing filter: the photodiode response is the only band
  limitation, as at a simulation rate of two samples per symbol;
- thermal noise: real white Gaussian noise on the ADC samples, with variance
  0.5 adc_sps 10^(-SNR/10) at unit optical power (SNR referred to the optical power), the
  same at every simulation rate;
- the DAC full scale is the peak of each block (no clipping); the ADC range is the
  observed range of each block.

Signals have shape (blocks, samples). Noise comes from the global torch generator unless
explicit unit-variance noise tensors are passed (held-out evaluation does this to fix the
noise of every block).
"""
import copy
import math
from dataclasses import replace

import torch
import torch.nn as nn

from .filters import (apply_fir, convolve, fir_module, gaussian_fir, gaussian_lowpass, identity_taps, ideal_lowpass,
                      lowpass_fir, nrz_pulse, quantize, rrc_pulse, supergaussian_fir, waveshaper_fir,
                      waveshaper_insertion_loss)
from .parameters import LinkParameters


class Link(nn.Module):
    def __init__(self, tx_bandwidth_hz, parameters=LinkParameters(), sps=None):
        super().__init__()
        p = self.p = parameters
        self.sps = p.analog_sps if sps is None else sps
        if self.sps % p.dsp_sps:
            raise ValueError('The analog rate must be a multiple of the DSP rate.')
        self.tx_bandwidth = float(tx_bandwidth_hz)
        if not 0 < self.tx_bandwidth <= p.sample_rate(p.dsp_sps)/2:
            raise ValueError('The TX filter bandwidth must lie between 0 and the DAC Nyquist frequency.')
        rate, taps = p.sample_rate(self.sps), p.fir_taps(self.sps)
        nyquist = rate/2
        # Analog FIRs. A low-pass with cutoff above Nyquist is the identity on the simulated band.
        self.driver = fir_module(identity_taps(taps) if p.driver_bandwidth >= nyquist
                                 else lowpass_fir(p.driver_bandwidth, rate, taps))
        if p.optical_filter == 'waveshaper':
            width = p.waveshaper_width*p.symbol_rate
            self.optical_filter = fir_module(waveshaper_fir(width, p.waveshaper_edge_bandwidth, rate, taps),
                                             scale=waveshaper_insertion_loss(width, p.waveshaper_edge_bandwidth))
        elif p.optical_filter == 'gaussian':
            self.optical_filter = fir_module(gaussian_fir(p.optical_gaussian_bandwidth/2, rate, taps))
        elif p.optical_filter == 'supergaussian':
            self.optical_filter = fir_module(supergaussian_fir(p.optical_gaussian_bandwidth/2,
                                                               p.optical_supergaussian_order, rate, taps))
        else:
            self.optical_filter = fir_module(identity_taps(taps))
        self.photodiode = fir_module(identity_taps(taps) if p.photodiode_bandwidth >= nyquist
                                     else lowpass_fir(p.photodiode_bandwidth, rate, taps))
        # Digital filters: TX pulse at the DSP rate, H_RX at the ADC rate.
        pulse = nrz_pulse if p.tx_pulse == 'nrz' else rrc_pulse
        self.register_buffer('pulse', torch.tensor(pulse(p.tx_pulse_rolloff, p.tx_pulse_span_symbols, p.dsp_sps)))
        adc_rate = p.sample_rate(p.adc_sps)
        self.register_buffer('h_rx', torch.tensor(gaussian_fir(p.rx_filter_bandwidth, adc_rate, p.fir_taps(p.adc_sps))
                                                  if p.rx_filter == 'gaussian' else identity_taps(p.fir_taps(p.adc_sps))))
        if p.rx_filter_before_adc:
            # Analog H_RX at the simulation rate (band limitation of photodiode and ADC electronics): a fourth
            # FIR of the analog link, which the digital surrogate refits instead of the photodiode response
            self.rx_filter = fir_module(gaussian_fir(p.rx_filter_bandwidth, rate, taps) if p.rx_filter == 'gaussian'
                                        else identity_taps(taps))
        self._dispersion, self._reference_power = {}, None

    @property
    def rate(self):
        return self.p.sample_rate(self.sps)

    def freeze(self):
        """No gradient reaches the link: the simulator and a trained surrogate are never updated by DSP training."""
        return self.eval().requires_grad_(False)

    # ---------------------------------------------------------------- transmitter (digital)
    def tx_waveform(self, x):
        """TX DSP samples x[n] (two per symbol) -> analog drive voltage at the simulation rate."""
        p, dac_rate = self.p, self.p.sample_rate(self.p.dsp_sps)
        shaped = torch.nn.functional.conv1d(x.unsqueeze(1), self.pulse.view(1, 1, -1),
                                            padding=self.pulse.numel()//2).squeeze(1)
        if p.tx_filter_after_dac:
            # analog H_TX after the DAC, known and applied exactly at the simulation rate (outside the digital
            # surrogate): the DAC converts the shaped samples, full scale = their peak
            peak = shaped.detach().abs().amax(-1, keepdim=True).clamp_min(1e-12)
            if p.dac_bits:
                shaped = quantize(shaped, p.dac_bits, -peak, peak)
            drive = self.reconstruct(shaped/peak if self.laser else shaped, self.sps//p.dsp_sps, dac_rate)
            return self.tx_filter(drive, self.rate)
        if self.laser:
            # DAC full scale fixed by the TX samples before H_TX, not by the filtered signal: a filter
            # (or a pre-emphasis) that lowers the swing lowers the DAC output; output normalized to +-1
            peak = shaped.detach().abs().amax(-1, keepdim=True).clamp_min(1e-12)
            shaped = self.tx_filter(shaped, dac_rate)
            if p.dac_bits:
                shaped = quantize(shaped, p.dac_bits, -peak, peak)
            return self.reconstruct(shaped/peak, self.sps//p.dsp_sps, dac_rate)
        shaped = self.tx_filter(shaped, dac_rate)
        if p.dac_bits:
            peak = shaped.detach().abs().amax(-1, keepdim=True).clamp_min(1e-12)
            shaped = quantize(shaped, p.dac_bits, -peak, peak)
        return self.reconstruct(shaped, self.sps//p.dsp_sps, dac_rate)

    def tx_filter(self, x, rate):
        """The TX filter H_TX at the DAC rate."""
        if self.p.tx_filter == 'rectangular':
            return ideal_lowpass(x, self.tx_bandwidth, rate)
        if self.p.tx_filter == 'gaussian':
            return gaussian_lowpass(x, self.tx_bandwidth, rate)
        return x

    @property
    def laser(self):
        """True for the 'laser' power reference (see LinkParameters.power_reference)."""
        return self.p.power_reference == 'laser'

    @staticmethod
    def reconstruct(samples, factor, dac_rate):
        """Ideal band-limited interpolation of the DAC samples by an integer factor."""
        n = samples.shape[-1]
        spectrum = torch.fft.rfft(samples)
        f = torch.fft.rfftfreq(n, d=1/dac_rate, device=samples.device)
        spectrum = spectrum*(f <= dac_rate/2)
        if factor > 1 and n % 2 == 0:
            spectrum = torch.cat((spectrum[..., :-1], spectrum[..., -1:]*.5), dim=-1)  # split the Nyquist bin
        return torch.fft.irfft(spectrum, n=n*factor)*factor

    # ---------------------------------------------------------------- analog link
    def scaled_voltage(self, drive, detach_peak=False):
        """Driver response, then the whole block scaled to the configured peak voltage ('mean'), or a
        fixed driver gain that maps the DAC full scale to the peak voltage ('laser')."""
        voltage = apply_fir(self.driver, drive)
        if self.laser:
            return voltage*self.p.peak_volt
        peak = voltage.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        return voltage*(self.p.peak_volt/(peak.detach() if detach_peak else peak))

    def mzm(self, voltage):
        """Push-pull Mach-Zehnder modulator biased at null or at quadrature (LinkParameters.mzm_bias), finite
        extinction ratio; complex field."""
        p = self.p
        bias = p.vpi/2 if p.mzm_bias == 'quadrature' else 0.
        angle = torch.pi/(2*p.vpi)*(voltage+bias-p.vpi)
        return .5*(torch.exp(1j*angle)+p.mzm_gamma*torch.exp(-1j*angle))*p.field_loss

    def fiber(self, field):
        """Chromatic dispersion exp(j beta2 L omega^2 / 2) applied to the field spectrum (circular)."""
        n = field.shape[-1]
        key = (n, field.device)
        if key not in self._dispersion:
            omega = 2*math.pi*torch.fft.fftfreq(n, d=1/self.rate, device=field.device, dtype=torch.float32)
            self._dispersion[key] = torch.exp(1j*self.p.dispersion_coefficient*omega**2)
        return torch.fft.ifft(torch.fft.fft(field)*self._dispersion[key])

    def noise_std(self, regime, level_db):
        ratio = 10**(float(level_db)/10)
        if regime == 'optical':      # per quadrature
            return math.sqrt(self.rate/(4*self.p.osnr_reference_bandwidth*ratio))
        if regime == 'electrical':  # on the ADC samples, or at the simulation rate before an analog H_RX
            return math.sqrt(.5*(self.sps if self.p.rx_filter_before_adc else self.p.adc_sps)/ratio)
        raise ValueError("Choose 'optical' or 'electrical' noise.")

    def noise_samples(self, regime, symbols):
        """Noise samples per block: ASE on the field and thermal noise before an analog H_RX at the simulation
        rate, thermal noise on the ADC samples otherwise."""
        analog = regime == 'optical' or self.p.rx_filter_before_adc
        return symbols*(self.sps if analog else self.p.adc_sps)

    def reference_power(self):
        """Fixed laser power of the 'laser' power reference: the mean optical power at the modulator
        output of the conventional PAM-4 TX on this link (computed once), so that the conventional
        PAM-4 keeps its SNR axis and any TX with less mean power pays for it. The reference is the
        null-biased conventional PAM-4 for every bias, so that the laser power is the same."""
        if self._reference_power is None and self.p.mzm_bias != 'null':
            null = copy.deepcopy(self)
            null.p, null._reference_power = replace(self.p, mzm_bias='null'), None
            self._reference_power = null.reference_power()
        if self._reference_power is None:
            from .dsp import ConventionalTransmitter
            device = self.pulse.device
            bits = torch.randint(0, 2, (1, 2, 16384), generator=torch.Generator().manual_seed(7)).to(device)
            with torch.no_grad():
                x = ConventionalTransmitter('pam4', True, self.p).to(device)(bits)
                field = self.mzm(self.scaled_voltage(self.tx_waveform(x)))
            self._reference_power = float(field.abs().square().mean())
        return self._reference_power

    def photocurrent(self, drive, regime, level_db=None, noise=None, detach_peak=False):
        """Drive voltage -> photocurrent at the simulation rate; ASE in the optical regime."""
        field = self.mzm(self.scaled_voltage(drive, detach_peak))
        if self.laser:
            field = field/math.sqrt(self.reference_power())
        else:
            field = field*torch.rsqrt(field.abs().square().mean(-1, keepdim=True))
        field = self.fiber(field)
        real, imag = field.real, field.imag
        draw = (lambda like: torch.randn_like(like)) if noise is None else (lambda like, it=iter(noise): next(it))
        if regime == 'optical' and level_db is not None:
            std = self.noise_std(regime, level_db)
            real = real+std*draw(real)
            imag = imag+std*draw(imag)
        real, imag = apply_fir(self.optical_filter, real), apply_fir(self.optical_filter, imag)
        return apply_fir(self.photodiode, real.square()+imag.square())

    def raw_photocurrent(self, voltage, ase_ebn0_db=None):
        """Photocurrent of a raw voltage, without peak or power normalization (surrogate training data),
        filtered by H_RX when H_RX is analog (rx_filter_before_adc).

        Optional ASE at an Eb/N0 relative to the mean optical power of the block, one polarization.
        """
        field = self.fiber(self.mzm(apply_fir(self.driver, voltage)))
        real, imag = field.real, field.imag
        if ase_ebn0_db is not None:
            power = (real.square()+imag.square()).mean(-1, keepdim=True).detach()
            std = torch.sqrt(power*(self.sps/self.p.bits_per_symbol)/(2*10**(ase_ebn0_db/10)))
            real = real+std*torch.randn_like(real)
            imag = imag+std*torch.randn_like(imag)
        real, imag = apply_fir(self.optical_filter, real), apply_fir(self.optical_filter, imag)
        current = apply_fir(self.photodiode, real.square()+imag.square())
        return apply_fir(self.rx_filter, current) if self.p.rx_filter_before_adc else current

    def analog_output(self, drive, regime, level_db=None, noise=None, detach_peak=False):
        """What the digital surrogate reproduces, without thermal noise: the photocurrent, filtered by H_RX
        when H_RX is analog (rx_filter_before_adc)."""
        current = self.photocurrent(drive, regime, level_db, noise, detach_peak)
        return apply_fir(self.rx_filter, current) if self.p.rx_filter_before_adc else current

    # ---------------------------------------------------------------- receiver
    def adc_samples(self, current, regime=None, level_db=None, noise=None):
        """Photocurrent -> ADC output. H_RX after the ADC (default): sampling at two samples per symbol ->
        thermal noise -> quantization. H_RX before the ADC (rx_filter_before_adc): thermal noise at the
        simulation rate -> analog H_RX (photodiode and ADC electronics) -> sampling -> quantization."""
        p = self.p
        thermal = regime == 'electrical' and level_db is not None
        if p.rx_filter_before_adc:
            if thermal:
                draw = torch.randn_like(current) if noise is None else noise[0]
                current = current+self.noise_std(regime, level_db)*draw
            current = apply_fir(self.rx_filter, current)[..., ::self.sps//p.adc_sps]
        else:
            current = current[..., ::self.sps//p.adc_sps]
            if thermal:
                draw = torch.randn_like(current) if noise is None else noise[0]
                current = current+self.noise_std(regime, level_db)*draw
        if p.adc_bits:
            current = quantize(current, p.adc_bits, current.amin(-1, keepdim=True), current.amax(-1, keepdim=True))
        return current

    def rx_samples(self, current, regime=None, level_db=None, noise=None):
        """Photocurrent -> RX DSP input: the ADC output, then the digital H_RX unless H_RX is analog."""
        y = self.adc_samples(current, regime, level_db, noise)
        return y if self.p.rx_filter_before_adc else convolve(y, self.h_rx)

    def forward(self, x, regime, level_db=None, noise=None):
        """TX DSP samples -> RX DSP samples: the whole link."""
        return self.rx_samples(self.photocurrent(self.tx_waveform(x), regime, level_db, noise), regime, level_db, noise)

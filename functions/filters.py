"""Filter designs and the operations that apply filters and quantizers to blocks of samples.

Every signal has shape (blocks, samples): each row is an independent block. Analog filters
are FIRs designed by frequency sampling of the transfer function, truncated and windowed.
"""
import numpy as np
from scipy.special import erf
import torch
import torch.nn.functional as F


def _frequency_sampled(transfer, num_taps):
    """Centre num_taps of the impulse response of a sampled transfer function, Hamming window, unit DC."""
    n_fft = transfer.size
    impulse = np.fft.fftshift(np.fft.ifft(transfer).real)
    half = num_taps//2
    taps = impulse[n_fft//2-half:n_fft//2+half+1]*np.hamming(num_taps)
    return (taps/np.sum(taps)).astype(np.float32)


def lowpass_fir(cutoff_hz, sample_rate, num_taps):
    """Windowed-sinc low-pass with unit DC gain."""
    normalized = cutoff_hz/(sample_rate/2)
    n = np.arange(num_taps)-(num_taps-1)/2
    taps = normalized*np.sinc(normalized*n)*np.hamming(num_taps)
    return (taps/np.sum(taps)).astype(np.float32)


def gaussian_fir(bandwidth_3db_hz, sample_rate, num_taps):
    """Gaussian low-pass H(f) = exp(-ln2/2 (f/B)^2), B the 3-dB bandwidth."""
    f = np.fft.fftfreq(8192, d=1/sample_rate)
    return _frequency_sampled(np.exp(-.5*np.log(2)*(np.abs(f)/bandwidth_3db_hz)**2), num_taps)


def supergaussian_fir(bandwidth_3db_hz, order, sample_rate, num_taps):
    """Super-Gaussian low-pass H(f) = exp(-ln2/2 (f/B)^(2 order)), B the 3-dB bandwidth; order 1 is the
    Gaussian (shape 'G' with filter.ord in Secondini's trans_func.m)."""
    f = np.fft.fftfreq(8192, d=1/sample_rate)
    return _frequency_sampled(np.exp(-.5*np.log(2)*(np.abs(f)/bandwidth_3db_hz)**(2*order)), num_taps)


def waveshaper_fir(width_hz, edge_bandwidth_hz, sample_rate, num_taps):
    """Flat-top optical bandpass with Gaussian edges, as a baseband filter on the field.

    H(f) = [erf((W/2 - f)/s) - erf((-W/2 - f)/s)]/2 with s = B_edge/(2 sqrt(ln2)): a rectangle
    of full width W convolved with a Gaussian of 3-dB width B_edge. Unit DC; the insertion
    loss of the device is applied separately (waveshaper_insertion_loss).
    """
    f = np.fft.fftfreq(8192, d=1/sample_rate)
    s = edge_bandwidth_hz/(2*np.sqrt(np.log(2)))
    return _frequency_sampled(.5*(erf((width_hz/2-f)/s)-erf((-width_hz/2-f)/s)), num_taps)


def waveshaper_insertion_loss(width_hz, edge_bandwidth_hz):
    """DC gain of the flat-top response, erf(W/(2 s)) (below one when the edges overlap)."""
    s = edge_bandwidth_hz/(2*np.sqrt(np.log(2)))
    return float(erf(width_hz/(2*s)))


def nrz_pulse(rolloff, span_symbols, sps):
    """Rectangle of one symbol with raised-cosine edges, band-limited to the symbol rate, unit peak.

    H(f) = sinc(f/Rs) cos(pi r f/Rs)/(1 - (2 r f/Rs)^2) for |f| <= Rs.
    """
    n_fft = 1 << 15
    f = np.fft.fftfreq(n_fft, d=1/sps)                    # in units of the symbol rate
    denominator = 1-(2*rolloff*f)**2
    with np.errstate(divide='ignore', invalid='ignore'):
        transfer = np.sinc(f)*np.cos(np.pi*rolloff*f)/denominator
    transfer[np.abs(denominator) < 1e-9] = rolloff/2*np.sin(np.pi/(2*rolloff))
    transfer[0] = 1.
    transfer = transfer*(np.abs(f) <= 1.)
    h = np.fft.fftshift(np.real(np.fft.ifft(transfer)))
    num_taps = span_symbols*sps+1
    h = h[n_fft//2+np.arange(num_taps)-(num_taps-1)//2]
    return (h/h.max()).astype(np.float32)


def rrc_pulse(rolloff, span_symbols, sps):
    """Root-raised-cosine pulse, unit peak."""
    t = (np.arange(span_symbols*sps+1)-span_symbols*sps/2)/sps   # in symbol periods
    h = np.empty_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-12:
            h[i] = 1+rolloff*(4/np.pi-1)
        elif rolloff > 0 and abs(abs(x)-1/(4*rolloff)) < 1e-12:
            h[i] = rolloff/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*rolloff))+(1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
        else:
            h[i] = (np.sin(np.pi*x*(1-rolloff))+4*rolloff*x*np.cos(np.pi*x*(1+rolloff)))/(np.pi*x*(1-(4*rolloff*x)**2))
    return (h/h.max()).astype(np.float32)


def fir_module(taps, scale=1.):
    """Conv1d convolving every row with `taps` (the weights a digital surrogate can refit)."""
    taps = np.asarray(taps, dtype=np.float32)
    conv = torch.nn.Conv1d(1, 1, kernel_size=taps.size, padding=taps.size//2, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor(taps[::-1].copy()).view(1, 1, -1)*scale)
    return conv


def identity_taps(num_taps):
    taps = np.zeros(num_taps, dtype=np.float32)
    taps[num_taps//2] = 1.
    return taps


def apply_fir(conv, x):
    """(blocks, samples) through a single-channel Conv1d, same length."""
    return conv(x.unsqueeze(1)).squeeze(1)


def convolve(x, taps):
    """Same-length convolution of every row with fixed taps (odd length)."""
    return F.conv1d(x.unsqueeze(1), taps.flip(0).view(1, 1, -1), padding=taps.numel()//2).squeeze(1)


def ideal_lowpass(x, cutoff_hz, sample_rate):
    """Rectangular response over the whole block: keep |f| <= cutoff."""
    spectrum = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(x.shape[-1], d=1/sample_rate, device=x.device)
    return torch.fft.irfft(spectrum*(f <= cutoff_hz), n=x.shape[-1])


def gaussian_lowpass(x, bandwidth_3db_hz, sample_rate):
    """Gaussian response exp(-ln2/2 (f/B)^2) over the whole block."""
    spectrum = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(x.shape[-1], d=1/sample_rate, device=x.device)
    return torch.fft.irfft(spectrum*torch.exp(-.5*np.log(2)*(f/bandwidth_3db_hz)**2), n=x.shape[-1])


def quantize(x, bits, low, high):
    """Uniform quantizer with 2^bits levels over [low, high] and a straight-through gradient.

    low and high are (blocks, 1) tensors: each block has its own range. Forward: quantized
    values; backward: identity. The step is computed in double precision and rounded once.
    """
    step = ((high.double()-low.double())/(2**bits-1)).to(x.dtype)
    q = torch.round((torch.minimum(torch.maximum(x, low), high)-low)/step)*step+low
    return x+(q-x).detach()

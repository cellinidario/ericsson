"""Reaching the A.19 bound: a self-contained study (slicer, then E2E).

Reproduces the overnight finding (see OVERNIGHT_A19_LOG.md):
  the whole gap to Forestieri's A.19 is NONLINEAR ISI from pulse OVERLAP through the
  square-law / MZM intensity nonlinearity. A.19 (eq. A.1) explicitly assumes a pulse
  g(t) confined to one symbol ("vanishes outside T to avoid nonlinear ISI after
  photodetection"). With that time-limited pulse, BOTH a genie slicer and the learned
  E2E reach A.19; with band-limited RRC a residual ~3x floor remains (capacity-
  independent), which is the overlap nonlinear ISI a 1-sps / Fork-A link cannot cancel.

No fudge factors: optimum (genie) thresholds, Gray, equiprobable symbols -- all Forestieri
assumptions. The E2E levels are LEARNED by symbol cross-entropy (no equispacing prior).

Usage (CPU is fine; the channel/nets are small):
    python a19_study.py slicer        # genie slicer: NRZ -> A.19, RRC -> overlap floor
    python a19_study.py e2e_rrc        # E2E on realistic band-limited RRC (residual floor)
    python a19_study.py e2e_nrz        # E2E with time-limited pulse -> A.19
    python a19_study.py all
"""
import os, sys, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "functions"))
import numpy as np
import torch
import torch.nn.functional as F
from config import Config
from channel import OpticalChannel
from transmitter import Transmitter, bits_to_symbols
from receiver import Receiver, bit_posteriors
from utils import add_awgn, measure_ber, theoretical_ber_unipolar
from pulse_shaping import root_raised_cosine, brickwall

DEVICE = "cpu"
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
GRAY = torch.tensor([0, 1, 3, 2])                       # symbol -> intensity rank (adjacent ranks: 1 bit)
SNRS = [12, 14, 16, 18]


# ----------------------------------------------------------------------------- configs
def base_cfg(front_end_ghz=None):
    """MZM / oband / thermal PAM-4. front_end_ghz widens segment-MZM and PD (None = realistic)."""
    c = Config()
    c.wavelength_band = "oband"; c._apply_wavelength_band()
    c.noise_regime = "thermal"; c.modulator = "mzm"
    c.tx_filter = "rrc"; c.rx_filter = "rrc"
    if front_end_ghz is not None:
        c.mzm_bandwidth = front_end_ghz * 1e9
        c.photodiode_bandwidth = front_end_ghz * 1e9
    c.dpd_hidden_width = 16; c.ffe_hidden_width = 16
    c.minibatch_symbols = 4096; c.edge_guard_symbols = 64
    c._compute_derived()
    return c


# ----------------------------------------------------------------------------- pulses
def tx_matched_kernels(kind, c):
    """(unit-peak TX pulse, unit-energy matched filter). 'nrz' is Forestieri's time-limited pulse."""
    sps = c.samples_per_symbol_sim
    if kind == "nrz":
        taps = np.ones(sps, dtype=np.float32)                       # sample-and-hold, confined to T
    elif kind == "rect":
        taps = brickwall(c.rect_filter_bandwidth, c.sim_sample_rate, c.rrc_span_symbols, sps)
    else:
        taps = root_raised_cosine(c.rrc_rolloff, c.rrc_span_symbols, sps)
    t = torch.tensor(taps, dtype=torch.float32, device=DEVICE)
    return (t / t.max()).view(1, 1, -1), (t / torch.sqrt((t ** 2).sum())).view(1, 1, -1)


# ----------------------------------------------------------------------------- genie slicer
def genie_slicer_ber(c, kind, num_symbols=2_000_000):
    """Ideal slicer: genie equispaced-INTENSITY levels, optimum thresholds, Gray. No NN."""
    ch = OpticalChannel(c).to(DEVICE)
    sps = c.samples_per_symbol_sim
    symbols = torch.randint(0, 4, (num_symbols,), device=DEVICE)
    ranks = GRAY.to(DEVICE)[symbols]
    bits = torch.stack([(symbols >> 1) & 1, symbols & 1]).to(torch.int64)
    # map Gray rank -> equispaced optical power, invert the MZM transfer
    target = ranks.float() / 3.0
    vv = torch.linspace(c.drive_min_volt, c.drive_max_volt, 4001, device=DEVICE)
    arg = (np.pi / (2 * c.mzm_vpi_volt)) * (vv - ch.bias_volt)
    inten = (0.5 * (torch.exp(1j * arg) + ch.gamma * torch.exp(-1j * arg))).abs() ** 2
    inten = (inten - inten.min()) / (inten.max() - inten.min())
    levels = vv[torch.bucketize(target.clamp(0, 1), inten).clamp(max=vv.numel() - 1)]
    pk, mf = tx_matched_kernels(kind, c)
    up = torch.zeros(num_symbols * sps, device=DEVICE); up[::sps] = levels
    drive = F.conv1d(up.view(1, 1, -1), pk, padding=pk.shape[-1] // 2).view(1, -1)
    with torch.no_grad():
        clean = ch(drive).detach()
    out = []
    for ebn0 in SNRS:
        noisy = add_awgn(clean, ebn0, c.bits_per_symbol, sps)
        m = F.conv1d(noisy.view(1, 1, -1), mf, padding=mf.shape[-1] // 2).squeeze()
        grid = m[:num_symbols * sps].reshape(num_symbols, sps)
        dec = grid[:, grid[200:-200].var(0).argmax()]              # best sub-symbol phase (ideal timing)
        best = 1.0
        for sh in range(-2, 3):                                    # integer-symbol group delay only
            r = torch.roll(ranks, sh)
            lm = torch.stack([dec[r == a].mean() for a in range(4)])
            if not torch.all(lm[1:] > lm[:-1]):
                continue
            th = (lm[:3] + lm[1:]) / 2
            est = GRAY.to(DEVICE)[torch.bucketize(dec, th)]
            eb = torch.stack([(est >> 1) & 1, est & 1]).to(torch.int64)
            best = min(best, (eb[:, 200:-200] != torch.roll(bits, sh, 1)[:, 200:-200]).float().mean().item())
        out.append(best)
    return np.array(out)


# ----------------------------------------------------------------------------- E2E
def _rx_logits(rx, photo, matched, offset, sps):
    x = F.conv1d(photo.view(1, 1, -1), matched, padding=matched.shape[-1] // 2)[:, :, offset::sps]
    win = rx.window_size; pad = (win - 1) // 2
    w = F.pad(x, (pad, pad)).unfold(2, win, 1); ns = w.shape[2]
    h = F.leaky_relu(rx.context_layer(w.permute(0, 2, 1, 3).reshape(1, ns, -1)))
    return rx.symbol_head(h).squeeze(0)


def _best_offset(tx, ch, matched, c):
    sps = c.samples_per_symbol_sim
    with torch.no_grad():
        bits = torch.randint(0, 2, (c.bits_per_symbol, 20000), device=DEVICE)
        m = F.conv1d(ch(tx(bits)).view(1, 1, -1), matched, padding=matched.shape[-1] // 2).squeeze()
        n = m.numel() // sps
        return int(m[:n * sps].reshape(n, sps)[100:-100].var(0).argmax())


def train_eval_e2e(c, kind, num_steps=40000):
    """Joint DPD+FFE, symbol cross-entropy, cosine LR decay. Returns BER over SNRS."""
    sps = c.samples_per_symbol_sim
    tx = Transmitter(c).to(DEVICE); ch = OpticalChannel(c).to(DEVICE); rx = Receiver(c).to(DEVICE)
    pk, matched = tx_matched_kernels(kind, c)
    tx.rrc = pk; rx.matched = matched                              # swap in the chosen pulse
    offset = _best_offset(tx, ch, matched, c)                      # decimation phase (ideal timing)
    guard = c.edge_guard_symbols; window = c.minibatch_symbols + 2 * guard
    opt = torch.optim.Adam(list(tx.parameters()) + list(rx.parameters()), lr=c.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps, eta_min=c.learning_rate * 0.01)
    t0 = time.time()
    for step in range(num_steps):
        ebn0 = float(np.random.uniform(9.0, 21.0))
        bits = torch.randint(0, 2, (c.bits_per_symbol, window), device=DEVICE)
        symbols = bits_to_symbols(bits, c.bits_per_symbol)
        logits = _rx_logits(rx, add_awgn(ch(tx(bits)), ebn0, c.bits_per_symbol, sps), matched, offset, sps)
        n = logits.shape[0]
        loss = F.cross_entropy(logits[guard:n - guard], symbols[guard:n - guard])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print(f"  trained {kind} in {time.time()-t0:.0f}s (offset {offset}/{sps})", flush=True)
    tx.eval(); rx.eval(); out = []
    with torch.no_grad():
        for e in SNRS:
            accs = []
            for s in range(3):
                torch.manual_seed(1000 + s)
                bits = torch.randint(0, 2, (c.bits_per_symbol, 1_000_000), device=DEVICE)
                logits = _rx_logits(rx, add_awgn(ch(tx(bits)), e, c.bits_per_symbol, sps), matched, offset, sps)
                ber, _ = measure_ber(bit_posteriors(logits, c.bits_per_symbol), bits, guard)
                accs.append(ber)
            out.append(float(np.mean(accs)))
    return np.array(out)


def _table(title, cols):
    th = theoretical_ber_unipolar(np.array(SNRS), 4)
    print(f"\n=== {title} ===")
    head = " Eb/N0 |   A.19   | " + " | ".join(f"{n:>16}" for n in cols)
    print(head)
    for i, e in enumerate(SNRS):
        row = f"  {e:4.0f} | {th[i]:.2e} |"
        for _, ber in cols.items():
            row += f"  {ber[i]:.2e} ({ber[i]/th[i]:5.1f}) |"
        print(row)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"device={DEVICE} threads={torch.get_num_threads()}  task={what}")

    if what in ("slicer", "all"):
        nrz = genie_slicer_ber(base_cfg(front_end_ghz=200), "nrz")    # time-limited + wide -> A.19
        rrc = genie_slicer_ber(base_cfg(front_end_ghz=200), "rrc")    # overlap -> nonlinear ISI floor
        _table("genie slicer (wide front-end): time-limited NRZ vs band-limited RRC",
               {"NRZ (->A.19)": nrz, "RRC (overlap)": rrc})

    if what in ("e2e_rrc", "all"):
        e = train_eval_e2e(base_cfg(front_end_ghz=None), "rrc")       # realistic band-limited
        _table("E2E, realistic RRC (residual overlap floor, capacity-independent)", {"E2E RRC": e})

    if what in ("e2e_nrz", "all"):
        e = train_eval_e2e(base_cfg(front_end_ghz=200), "nrz")        # time-limited -> A.19
        _table("E2E, time-limited pulse (-> A.19, learned levels)", {"E2E NRZ": e})

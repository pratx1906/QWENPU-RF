"""Synthetic IQ captures for 25 modulation classes (PROTOCOL Amendment 15; host-side data generation only).

The RadioML-2018 class list (24) plus LoRa CSS. Digital linear classes: 8 samples/symbol, RRC roll-off 0.35
(span 8 symbols), random symbols. Ring/cross constellations are DVB-S2-like approximations, not standard tables.
Impairments: random carrier phase, random timing offset (0..7 samples), random gain, complex AWGN at the given
SNR (per-sample signal power / noise power). Not modelled: carrier frequency offset, multipath, unknown symbol rate.
Output: N (I, Q) Q15 integer pairs.
"""
import numpy as np

CLASSES = ["OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK", "16APSK", "32APSK", "64APSK",
           "128APSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
           "AM-DSB-SC", "FM", "GMSK", "OQPSK", "LoRa"]
N, SPS, BETA, SPAN = 1024, 8, 0.35, 8


def rrc(sps=SPS, beta=BETA, span=SPAN):
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.empty_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(4 * beta * x) - 1) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * x * (1 - beta)) + 4 * beta * x * np.cos(np.pi * x * (1 + beta))) / (np.pi * x * (1 - (4 * beta * x) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


H_RRC = rrc()


def psk(m):
    return np.exp(2j * np.pi * np.arange(m) / m)


def qam(m):
    k = int(round(np.sqrt(m)))
    if k * k == m:
        g = np.arange(k) - (k - 1) / 2
        return (g[:, None] + 1j * g[None, :]).ravel()
    k = int(np.ceil(np.sqrt(m * 9 / 8)))  # cross: square grid minus equal corner blocks
    k += k % 2
    g = np.arange(k) - (k - 1) / 2
    pts = (g[:, None] + 1j * g[None, :]).ravel()
    c = int(round(np.sqrt((k * k - m) / 4)))
    lim = (k - 1) / 2 - c + 1e-9
    keep = ~((np.abs(pts.real) > lim) & (np.abs(pts.imag) > lim))
    pts = pts[keep]
    assert len(pts) == m, (m, len(pts))
    return pts


def apsk(counts, radii):
    return np.concatenate([r * np.exp(1j * (2 * np.pi * np.arange(n) / n + np.pi / n * (i % 2)))
                           for i, (n, r) in enumerate(zip(counts, radii))])


CONST = {
    "OOK": np.array([0, 1.]), "4ASK": np.arange(4.), "8ASK": np.arange(8.), "BPSK": psk(2), "QPSK": psk(4) * np.exp(1j * np.pi / 4),
    "8PSK": psk(8), "16PSK": psk(16), "32PSK": psk(32),
    "16APSK": apsk([4, 12], [1, 2.85]), "32APSK": apsk([4, 12, 16], [1, 2.84, 5.27]),
    "64APSK": apsk([4, 12, 20, 28], [1, 2.4, 4.3, 7.0]), "128APSK": apsk([8, 8, 16, 16, 16, 16, 24, 24], np.arange(1, 9.)),
    "16QAM": qam(16), "32QAM": qam(32), "64QAM": qam(64), "128QAM": qam(128), "256QAM": qam(256),
}


def _shaped(sym, rng):
    up = np.zeros(len(sym) * SPS, complex)
    up[::SPS] = sym
    return np.convolve(up, H_RRC, mode="same")


def _message(rng, n):
    """Band-limited random message (0.002..0.04 cycles/sample), unit peak."""
    f = np.fft.fft(rng.normal(size=n))
    k = np.fft.fftfreq(n)
    lo, hi = 0.002, rng.uniform(0.02, 0.04)
    f[(np.abs(k) < lo) | (np.abs(k) > hi)] = 0
    m = np.fft.ifft(f).real
    return m / np.max(np.abs(m))


def _analytic(m):
    f = np.fft.fft(m)
    n = len(m)
    h = np.zeros(n)
    h[0] = 1
    h[1:n // 2] = 2
    h[n // 2] = 1
    return np.fft.ifft(f * h)


def baseband(cls, rng, n=N):
    """Clean complex baseband of `cls`, length n, unit mean power."""
    pad = 4 * n
    if cls in CONST:
        c = CONST[cls]
        x = _shaped(c[rng.integers(len(c), size=pad // SPS)], rng)
    elif cls == "OQPSK":
        b = rng.integers(2, size=(2, pad // SPS)) * 2 - 1.
        i, q = _shaped(b[0].astype(complex), rng).real, _shaped(b[1].astype(complex), rng).real
        x = i + 1j * np.roll(q, SPS // 2)
    elif cls == "GMSK":
        bt, nb = 0.3, pad // SPS
        bits = rng.integers(2, size=nb) * 2 - 1.
        t = np.arange(-2 * SPS, 2 * SPS + 1) / SPS
        g = np.exp(-2 * (np.pi * bt * t) ** 2 / np.log(2))
        g /= g.sum()
        f = np.convolve(np.repeat(bits, SPS), g, mode="same")
        x = np.exp(1j * np.pi * 0.5 * np.cumsum(f) / SPS)
    elif cls == "FM":
        x = np.exp(2j * np.pi * np.cumsum(rng.uniform(0.03, 0.08) * _message(rng, pad)))
    elif cls.startswith("AM"):
        m = _message(rng, pad)
        mu = rng.uniform(0.5, 0.9)
        x = {"AM-DSB-WC": 1 + mu * m, "AM-DSB-SC": m + 0j, "AM-SSB-SC": _analytic(m),
             "AM-SSB-WC": 1 + mu * _analytic(m)}[cls].astype(complex)
    elif cls == "LoRa":
        sf = 7
        M = 1 << sf
        syms = rng.integers(M, size=pad // M + 1)
        k = np.arange(M)
        x = np.concatenate([np.exp(2j * np.pi * np.cumsum(((s + k) % M) / M - 0.5)) for s in syms])
    else:
        raise ValueError(cls)
    off = n + rng.integers(128 if cls == "LoRa" else SPS)  # random timing offset, away from the edges
    x = x[off:off + n]
    return x / np.sqrt(np.mean(np.abs(x) ** 2))


def capture(cls, rng, snr_db, n=N):
    """Impaired capture as Q15 integer pairs."""
    x = baseband(cls, rng, n) * np.exp(2j * np.pi * rng.uniform())
    sigma = np.sqrt(10 ** (-snr_db / 10) / 2)
    x = x + sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))
    x *= rng.uniform(0.12, 0.3) / np.sqrt(np.mean(np.abs(x) ** 2))
    q = np.clip(np.round(np.stack((x.real, x.imag), -1) * 32768), -32768, 32767).astype(int)
    return [tuple(map(int, v)) for v in q]

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from rfpu_v15.cpu import spectral as S  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402

EDGE = [(-32768, -32768), (32767, 32767), (-32768, 32767), (0, 0), (1, -1), (-1, 1)]
X = [-32768, 32767]


def stream(seed, length, n):
    rng = random.Random(seed)
    s = list(EDGE)
    s += [(-32768, -32768)] * n                                  # DC at full scale
    s += [((-32768, 32767)[i % 2], (32767, -32768)[i % 2]) for i in range(n)]  # Nyquist, full scale
    s += [(rng.choice(X), rng.choice(X)) for _ in range(n)]      # random extremes
    s += [(round(20000 * math.cos(2 * math.pi * 3 * t / n)), round(20000 * math.sin(2 * math.pi * 3 * t / n)))
          for t in range(2 * n)]                                 # tone on bin 3 (CFAR target)
    s += [(rng.randrange(-32768, 32768), rng.randrange(-32768, 32768)) for _ in range(max(0, length - len(s)))]
    return s[:max(length, len(s))]


def window(seed, n):
    rng = random.Random(seed)
    w = [rng.randrange(-32768, 32768) for _ in range(n)]
    w[0], w[-1] = -32768, 32767
    return w


@pytest.fixture(scope="module")
def runner():
    return Runner(vx=False, batch=4)


def check(runner, kernel, blocks, length, seeds=(1, 2, 3)):
    streams = [stream(s, length, kernel.n) for s in seeds]
    results = {}
    for block in blocks:
        rs = runner.run_many(kernel, streams, block=block, neural=False)
        assert all(r.exact_outputs for r in rs), (kernel.name, block)
        results[block] = rs
    a, b = (results[k] for k in blocks)
    assert [r.outputs for r in a] == [r.outputs for r in b]
    assert any(o is not None for o in a[0].outputs)
    return a[0]


@pytest.mark.parametrize("n,hop", [(8, 2), (16, 4), (64, 16)])
def test_fft_power(runner, n, hop):
    check(runner, S.FFTPower(n, window(n, n), hop), (max(hop, 8) * 2, 64), 5 * n + 16)


@pytest.mark.parametrize("n,hop", [(8, 8), (16, 2), (64, 64)])
def test_fft_complex_matches_numpy(runner, n, hop):
    k = S.FFTPower(n, window(n + 1, n), hop, output="complex")
    r = check(runner, k, (hop * 2 if hop >= 8 else 16, 64), 5 * n + 16)
    x = stream(1, 5 * n + 16, n)
    w = np.array(k.window) / 32768.0
    for e, bins in enumerate(r.outputs):
        if bins is None:
            continue
        seg = np.array([complex(a, b) for a, b in x[e - n + 1:e + 1]])
        want = np.fft.fft(w * seg) / n
        got = np.array([complex(a, b) for a, b in bins])
        assert np.max(np.abs(got - want)) < 2 * k.L + 2 + 4e-4 * np.max(np.abs(want)), (n, e)


def test_fft_256_once(runner):
    k = S.FFTPower(256, None, 256)
    assert k.describe()["max_block"] >= 256
    r = runner.run_many(k, [stream(5, 600, 256)], block=256, neural=False)[0]
    assert r.exact_outputs


@pytest.mark.parametrize("n,hop,frames", [(8, 4, 4), (16, 8, 2), (64, 32, 2)])
def test_welch(runner, n, hop, frames):
    k = S.WelchPSD(n, window(n + 2, n), hop, frames)
    check(runner, k, (k.hop, 2 * k.hop if 2 * k.hop >= 64 else 64), 4 * n + 2 * k.hop)


@pytest.mark.parametrize("n,hop,train,guard,scale", [(8, 4, 1, 1, 768), (16, 8, 3, 1, 2000),
                                                     (64, 32, 8, 2, 1500), (64, 64, 1, 0, 65535)])
def test_cfar(runner, n, hop, train, guard, scale):
    k = S.CFARDetect(n, window(n + 3, n) if train != 3 else None, hop, train, guard, scale)  # None: Hann
    r = check(runner, k, (hop * 2 if hop >= 16 else 16, 128), 5 * n + 16)
    if scale < 4096:
        assert any(o and o[0] for o in r.outputs)  # some detections occur


def test_cfar_ties_and_zero(runner):
    k = S.CFARDetect(16, [0] * 16, 16, 2, 1, 256)  # zero window: all powers 0 -> peak 0, no detections
    r = runner.run(k, stream(4, 64, 16), block=32, neural=False)
    assert r.exact_outputs and all(o == (0, 0) for o in r.outputs if o is not None)


@pytest.mark.parametrize("n,k,hop", [(8, 1, 2), (16, 5, 4), (64, 63, 16)])
def test_dft_bin(runner, n, k, hop):
    kern = S.DFTBin(n, k, hop)
    r = check(runner, kern, (32, 64), 5 * n + 16)
    x = stream(1, 5 * n + 16, n)
    for e, p in enumerate(r.outputs):
        if p is not None:
            seg = np.array([complex(a, b) for a, b in x[e - n + 1:e + 1]])
            want = abs(np.fft.fft(seg)[k]) ** 2
            assert abs(p - want) <= 1e-3 * want + 4 * n * np.sqrt(want) + 4 * n * n


def test_layout_limits():
    for n in (128, 256):
        k = S.FFTPower(n, None, n)
        assert k.describe()["max_block"] >= n
    assert S.FFTPower(64, None, 64).max_block() >= 256
    assert S.CFARDetect(64, None, 64, 4, 2).max_block() >= 256
    with pytest.raises(ValueError):
        S.CFARDetect(8, None, 8, 3, 1)


# ---------------------------------------------------------------- block-floating-point FFT (Amendment 4)
def test_block_reference_forced_scaling_equals_fixed():
    """With every stage forced to scale and r = 15, the iterative block reference is the fixed FFT."""
    n = 16
    x = stream(3, 40, n)
    w = window(7, n)
    fixed = S.ref_frame(x, 30, n, w)
    L = n.bit_length() - 1
    p = [(re * v, im * v) for (re, im), v in zip(x[30 - n + 1:31], w)]
    buf = [None] * n
    for i, (re, im) in enumerate(p):
        buf[S.bitrev(i, L)] = (S.round_half_up(re, 15), S.round_half_up(im, 15))
    for s in range(L):
        h = 1 << s
        for g in range(0, n, 2 * h):
            for k in range(h):
                wr, wi = S.twiddle(k * (n // (2 * h)), n)
                (ar, ai), (br, bi) = buf[g + k], buf[g + k + h]
                tr, ti = S.round_half_up(wr * br - wi * bi, 15), S.round_half_up(wr * bi + wi * br, 15)
                buf[g + k] = ((ar + tr + 1) >> 1, (ai + ti + 1) >> 1)
                buf[g + k + h] = ((ar - tr + 1) >> 1, (ai - ti + 1) >> 1)
    assert buf == fixed


@pytest.mark.parametrize("n,hop,output", [(8, 8, "power"), (16, 4, "complex"), (64, 64, "power"), (64, 32, "complex")])
def test_fft_block(runner, n, hop, output):
    k = S.FFTPower(n, window(n + 3, n), hop, output=output, scaling="block")
    check(runner, k, (max(hop, 8) * 2, 64), 5 * n + 16)


@pytest.mark.parametrize("level", [1, 30, 3000, 32767])
def test_fft_block_accuracy_any_level(runner, level):
    """Bins scaled by 2^-exponent match numpy within a few LSB of the block scale at every input level."""
    n = 64
    rng = random.Random(level)
    x = [(rng.randint(-level, level), rng.randint(-level, level)) for _ in range(2 * n)]
    k = S.FFTPower(n, S.hann(n), n, output="complex", scaling="block")
    r = runner.run_many(k, [x], block=64, neural=False)[0]
    assert r.exact_outputs
    w = np.array(k.window) / 32768.0
    for e, out in enumerate(r.outputs):
        if out is None:
            continue
        exponent, bins = out
        seg = np.array([complex(a, b) for a, b in x[e - n + 1:e + 1]])
        want = np.fft.fft(w * seg) / n * 2.0 ** exponent
        got = np.array([complex(a, b) for a, b in bins])
        assert np.max(np.abs(want)) > 2 ** 12          # the block scale uses the word
        assert np.max(np.abs(got - want)) < 2 * k.L + 4, (level, e)

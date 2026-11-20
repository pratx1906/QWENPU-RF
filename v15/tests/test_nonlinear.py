import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from rfpu_v15.cpu import nonlinear as N  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402

EDGE = [(-32768, -32768), (32767, 32767), (-32768, 32767), (32767, -32768), (0, 0), (0, 0),
        (1, -1), (-1, 1), (-32768, 0), (0, -32768), (32767, 0), (0, 32767), (-1, 0), (0, 1)]
ANGLE_TOL = 2 ** 17  # 2^-14 * pi rad in binary angle units (2^31 == pi)


def stream(seed, n):
    rng = random.Random(seed)
    return EDGE + [(rng.randrange(-32768, 32768), rng.randrange(-32768, 32768)) for _ in range(n - len(EDGE))]


def tone(n, step, amp=20000, seed=0, noise=0):
    rng = random.Random(seed)
    return [(max(-32768, min(32767, round(amp * math.cos(step * k) + rng.gauss(0, noise)))),
             max(-32768, min(32767, round(amp * math.sin(step * k) + rng.gauss(0, noise)))))
            for k in range(n)]


def wrap_diff(a, b):
    return (a - b + 2 ** 31) % 2 ** 32 - 2 ** 31


@pytest.fixture(scope="module")
def cpu():
    return Runner(vx=False, batch=4)


def test_atan_table_and_gain():
    assert N.ATAN[0] == 1 << 29
    assert abs(N.GAIN - 1.6467602581) < 1e-9


def test_cordic_reference_against_numpy():
    rng = random.Random(3)
    pts = [(rng.randrange(-N.LIMIT + 1, N.LIMIT), rng.randrange(-N.LIMIT + 1, N.LIMIT)) for _ in range(3000)]
    pts += [(N.LIMIT - 1, 0), (-(N.LIMIT - 1), 0), (0, N.LIMIT - 1), (0, -(N.LIMIT - 1)),
            (-(N.LIMIT - 1), -(N.LIMIT - 1)), (N.LIMIT - 1, N.LIMIT - 1), (-(N.LIMIT - 1), N.LIMIT - 1)]
    for x, y in pts:
        mag, ang = N.cordic_vector(x, y)
        r = math.hypot(x, y)
        if r < 2 ** 16:
            continue
        want = round(np.angle(complex(x, y)) / np.pi * 2 ** 31)
        assert abs(wrap_diff(ang, want)) <= ANGLE_TOL, (x, y, ang, want)
        assert abs(mag / N.GAIN / r - 1) < 1e-3, (x, y)
    with pytest.raises(ValueError):
        N.cordic_vector(N.LIMIT, 0)


def check_magphase_accuracy(samples, out):
    for (re, im), (mag, ang) in zip(samples, out):
        r = abs(complex(re, im))
        if r < 64:
            continue
        assert abs(mag / N.GAIN / 2 ** 13 / r - 1) < 1e-3
        assert abs(wrap_diff(ang, round(np.angle(complex(re, im)) / np.pi * 2 ** 31))) <= ANGLE_TOL


@pytest.mark.parametrize("kernel", [N.MagPhase(), N.AMEnvelope(), N.AMEnvelope(10), N.FMDemod()], ids=lambda k: k.name)
def test_per_sample_kernels_exact(cpu, kernel):
    x = stream(5, 70)
    outs = []
    for block in (8, 32):
        r = cpu.run(kernel, x, block=block, neural=False)
        assert r.exact_outputs, (kernel.name, block)
        outs.append(r.outputs)
    assert outs[0] == outs[1]
    if isinstance(kernel, N.MagPhase):
        check_magphase_accuracy(x, outs[0])
    if isinstance(kernel, N.AMEnvelope):
        for (re, im), mag in zip(x, outs[0]):
            if abs(complex(re, im)) >= 64:
                assert abs(mag / N.GAIN / 2 ** 13 / abs(complex(re, im)) - 1) < 1e-3


def test_run_many_streams(cpu):
    streams = [stream(s, n) for s, n in ((11, 40), (12, 17), (13, 33))]
    for kernel in (N.MagPhase(), N.FMDemod(), N.CFOEstimate(lag=2, window=8)):
        res = cpu.run_many(kernel, streams, block=16, neural=False)
        assert all(r.exact_outputs for r in res), kernel.name


def test_fm_demod_tracks_frequency(cpu):
    step = 0.3  # rad/sample
    x = tone(48, step)
    r = cpu.run(N.FMDemod(), x, block=16, neural=False)
    assert r.exact_outputs
    want = round(step / math.pi * 2 ** 31)
    for a in r.outputs[1:]:
        assert abs(wrap_diff(a, want)) < 2 ** 31 * 2e-3  # Q15 quantization of the tone dominates
    # sign of frequency
    r = cpu.run(N.FMDemod(), tone(24, -2.5), block=8, neural=False)
    assert all(abs(wrap_diff(a, round(-2.5 / math.pi * 2 ** 31))) < 2 ** 31 * 2e-3 for a in r.outputs[1:])


@pytest.mark.parametrize("lag,window", [(1, 8), (4, 16), (16, 32)])
def test_cfo_estimate_exact(cpu, lag, window):
    k = N.CFOEstimate(lag, window)
    x = stream(21 + lag, 100)
    outs = []
    for block in (window, 2 * window):
        r = cpu.run(k, x, block=block, neural=False)
        assert r.exact_outputs, (k.name, block)
        outs.append(r.outputs)
    assert outs[0] == outs[1]
    assert sum(o is not None for o in outs[0]) == 100 // window


def test_cfo_estimate_extremes_and_accuracy(cpu):
    k = N.CFOEstimate(lag=1, window=16)
    x = [(-32768, -32768)] * 32 + [(32767, -32768), (-32768, 32767)] * 8 + [(0, 0)] * 16
    r = cpu.run(k, x, block=16, neural=False)
    assert r.exact_outputs
    step = 0.05
    k = N.CFOEstimate(lag=4, window=64)
    x = tone(256, step, amp=15000, seed=1, noise=300)
    r = cpu.run(k, x, block=128, neural=False)
    assert r.exact_outputs
    xs = np.array([complex(a, b) for a, b in x])
    for e, o in enumerate(r.outputs):
        if o is None:
            continue
        ang, mag = o
        n = np.arange(e - 63, e + 1)
        prev = np.where(n - 4 >= 0, xs[np.maximum(n - 4, 0)], 0)
        R = np.sum(xs[n] * np.conj(prev))
        assert abs(wrap_diff(ang, round(np.angle(R) / np.pi * 2 ** 31))) <= ANGLE_TOL
        assert abs(mag / N.GAIN * 2 ** k.shift / abs(R) - 1) < 1e-3
        if e >= 127:  # full lag history: angle / lag estimates the per-sample rotation
            assert abs(ang / 2 ** 31 * math.pi / 4 - step) < 2e-3


def test_block_256_fits():
    for k in (N.MagPhase(), N.AMEnvelope(), N.FMDemod(), N.CFOEstimate(16, 256)):
        b = k.build(256)
        assert b.layout.top <= (1 << 15) - 1024
    with pytest.raises(ValueError):
        N.CFOEstimate(1, 48)
    with pytest.raises(ValueError):
        N.CFOEstimate(17, 64)


def test_programs_have_no_data_branches():
    for k in (N.MagPhase(), N.FMDemod(), N.CFOEstimate(3, 16)):
        branches = [ln for ln in k.build(32).source.splitlines()
                    if ln.split() and ln.split()[0] in ("beq", "bne", "blt", "bge", "bltu", "bgeu", "beqz",
                                                        "bltz", "bgez", "blez", "bgtz", "bgt", "ble")]
        assert all(ln.startswith("bnez a") for ln in branches), branches


def test_tokens_per_sample(cpu):
    x = stream(31, 64)
    for k in (N.MagPhase(), N.AMEnvelope(), N.FMDemod(), N.CFOEstimate(4, 64)):
        r = cpu.run(k, x, block=64, neural=False)
        assert r.exact_outputs
        print(k.name, round(r.stats["tokens_per_sample"], 1))

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from rfpu_v15.cpu import filters as F  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402

EDGE = [(-32768, -32768), (32767, 32767), (-32768, 32767), (32767, -32768), (0, 0), (1, -1), (-1, 1),
        (32767, 32767), (32767, 32767), (-32768, -32768), (-32768, -32768)]


def stream(seed, n):
    rng = random.Random(seed)
    body = []
    while len(body) < n - len(EDGE):
        if rng.random() < 0.2:  # runs of rails drive filters into saturation
            v = (rng.choice((-32768, 32767)), rng.choice((-32768, 32767)))
            body += [v] * rng.randrange(1, 6)
        else:
            body.append((rng.randrange(-32768, 32768), rng.randrange(-32768, 32768)))
    return (EDGE + body)[:n]


def q15(rng, lo=-32768, hi=32768):
    return rng.randrange(lo, hi)


@pytest.fixture(scope="module")
def cpu_runner():
    return Runner(vx=False, batch=4)


def decim_cases():
    rng = random.Random(77)
    yield F.DecimatingFIR([(16384, 0), (8192, 0), (4096, 0)], 2, "decim2_narrow")
    yield F.DecimatingFIR([(32767, -32768), (-32768, 32767), (32767, 32767)], 4, "decim4_wide")
    yield F.DecimatingFIR([(q15(rng, -3000, 3000), q15(rng, -3000, 3000)) for _ in range(12)], 8, "decim8_t12")
    yield F.DecimatingFIR([(q15(rng), q15(rng)) for _ in range(7)], 4, "decim4_t7_rand")
    yield F.DecimatingFIR([(q15(rng), q15(rng))], 1, "decim1")


def stateful_cases():
    rng = random.Random(1234)
    yield F.IIRBiquad(4096, 8192, 4096, -16384, 6554, "iir_lowpass")
    yield F.IIRBiquad(32767, -32768, 32767, -32768, 32767, "iir_extreme")
    yield F.IIRBiquad(*[q15(rng) for _ in range(5)], name="iir_random")
    yield F.IIRBiquad(4096, 8192, 4096, -16384, 4096, wide=True)
    yield F.IIRBiquad(16384, 0, 0, -32768, 16384, "iir_wide_resonant", wide=True)
    yield F.IIRBiquad(4096, 8192, 4096, -16384, 4096, wide=True, out_frac=4)
    yield F.DCBlocker(32440, "dc_default")
    yield F.DCBlocker(32767, "dc_max")
    yield F.DCBlocker(-32768, "dc_min")
    yield F.DCBlocker(q15(rng), "dc_random")
    yield F.NCOMixer(0x12345679, "nco_fixed")
    yield F.NCOMixer(rng.randrange(1 << 32), "nco_random")
    yield F.NCOMixer(-(1 << 28), "nco_negative")
    yield F.NCOMixer(0, "nco_dc")
    yield F.NCOMixer(0x12345679, "nco_interp_fixed", interpolate=True)
    yield F.NCOMixer(rng.randrange(1 << 32), "nco_interp_random", interpolate=True)
    yield F.NCOMixer(-(1 << 28) - 12345, "nco_interp_negative", interpolate=True)
    yield F.NCOMixer((1 << 21) - 1, "nco_interp_half_step", interpolate=True)


@pytest.mark.parametrize("kernel", list(decim_cases()), ids=lambda k: k.name)
def test_decimating_fir(cpu_runner, kernel):
    x = stream(5, 75)
    outs = []
    for block in (8, 32):
        r = cpu_runner.run(kernel, x, block=block, neural=False)
        assert r.exact_outputs, (kernel.name, block)
        outs.append(r.outputs)
    assert outs[0] == outs[1]
    D = kernel.decim
    assert all((o is None) == (n % D != D - 1) for n, o in enumerate(outs[0]))


def test_decim_precision_selection():
    assert not F.DecimatingFIR([(16384, 0), (8192, 0)], 2).wide
    assert F.DecimatingFIR([(32767, 32767), (32767, 32767)], 2).wide


@pytest.mark.parametrize("kernel", list(stateful_cases()), ids=lambda k: k.name)
def test_stateful_block_invariance_and_run_many(cpu_runner, kernel):
    x = stream(11, 70)
    outs = []
    for block in (8, 32):
        r = cpu_runner.run(kernel, x, block=block, neural=False)
        assert r.exact_outputs, (kernel.name, block)
        outs.append(r.outputs)
    assert outs[0] == outs[1] == kernel.reference(x)
    streams = [stream(20 + k, n) for k, n in enumerate((37, 64, 9, 50))]
    rs = cpu_runner.run_many(kernel, streams, block=16, neural=False)
    for s, r in zip(streams, rs):
        assert r.exact_outputs
        assert r.outputs == kernel.reference(s) == cpu_runner.run(kernel, s, block=8, neural=False).outputs


def test_saturation_is_exercised():
    k = F.IIRBiquad(32767, -32768, 32767, -32768, 32767)
    y = k.reference(stream(11, 70))
    assert any(v in (-32768, 32767) for p in y for v in p)
    d = F.DCBlocker(32767).reference(stream(11, 70))
    assert any(v in (-32768, 32767) for p in d for v in p)


def test_sine_table():
    t = F.sine_table()
    assert len(t) == 1024 and t[0] == 0 and t[256] == 32767 and t[768] == -32768 and max(t) == 32767
    assert all(abs(v - 32768 * math.sin(2 * math.pi * i / 1024)) <= 1 for i, v in enumerate(t))


@pytest.mark.parametrize("kernel", [F.DecimatingFIR([(1, 2)] * 64, 4), F.IIRBiquad(1, 2, 3, 4, 5),
                                    F.DCBlocker(), F.NCOMixer(12345), F.NCOMixer(12345, interpolate=True)], ids=lambda k: k.name)
def test_block_256_fits_and_branch_uniform(cpu_runner, kernel):
    """block=256 fits RAM; and the instruction count is data-independent (same pc trace)."""
    built, prog = cpu_runner.program(kernel, 256)
    counts = set()
    for seed in (1, 2):
        x = stream(seed, 256)
        ctx, _ = Runner.context(kernel, x, 0, 256)
        words = kernel.init_words(built, ctx, 0)
        if kernel.state_words:
            kernel.put_state(built, words, [random.Random(seed).randrange(-32768, 32768)
                                            for _ in range(kernel.state_words)])
        m = cpu_runner.machine(prog, words)
        m.run(10**8)
        assert m.halted and m.fault is None and m.exit_code == 0
        counts.add(m.steps)
    assert len(counts) == 1


def test_nco_interpolation_accuracy():
    """The interpolated NCO's oscillator is within ~1 LSB of exact sin/cos at every phase (unit input)."""
    fw = 0x0123457
    k = F.NCOMixer(fw, interpolate=True)
    y = k.reference([(32767, 0)] * 4096)
    ph = 2 * math.pi * fw / 2**32 * np.arange(4096)
    err = np.abs(np.array([complex(a, b) for a, b in y]) - 32767 * np.exp(1j * ph))
    assert err.max() < 2  # output rounding 0.71 + table rounding 0.71 + Taylor remainder 0.15
    plain = F.NCOMixer(fw).reference([(32767, 0)] * 4096)
    assert np.abs(np.array([complex(a, b) for a, b in plain]) - 32767 * np.exp(1j * ph)).max() > 50


def test_iir_out_frac():
    """Amendment 12: out_frac f keeps f extra output bits; the rounding error vs float drops ~4^f."""
    from scipy.signal import lfilter
    with pytest.raises(ValueError):
        F.IIRBiquad(4096, 8192, 4096, -16384, 4096, out_frac=4)
    b = (4096, 8192, 4096, -16384, 4096)
    rng = random.Random(9)
    x = [(rng.randrange(-3000, 3000), rng.randrange(-3000, 3000)) for _ in range(400)]
    fl = lfilter([v / 16384 for v in b[:3]], [1, b[3] / 16384, b[4] / 16384], [complex(*v) for v in x])
    err = []
    for f in (0, 4):
        y = F.IIRBiquad(*b, wide=True, out_frac=f).reference(x)
        err.append(sum(abs(complex(*v) / 2 ** f - w) ** 2 for v, w in zip(y, fl)))
    assert err[1] < err[0] / 100

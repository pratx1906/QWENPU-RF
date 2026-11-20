import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from rfpu_v15.cpu import kernels as K  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402

EDGE = [(-32768, -32768), (32767, 32767), (-32768, 32767), (0, 0), (1, -1), (-1, 1)]


def stream(seed, n):
    rng = random.Random(seed)
    return EDGE + [(rng.randrange(-32768, 32768), rng.randrange(-32768, 32768)) for _ in range(n - len(EDGE))]


def cases():
    rng = random.Random(1505)
    yield K.FIR([(16384, 0), (8192, 0)], "gate_fir")
    yield K.FIR([(16384, 0), (-8192, 0)], "gate_matched")
    yield K.FIR([(32767, -32768), (-32768, 32767), (32767, 32767)], "wide_fir3")  # forces 64-bit
    yield K.FIR([(rng.randrange(-3000, 3000), rng.randrange(-3000, 3000)) for _ in range(16)])
    yield K.FFT4Power()
    yield K.FFT4Power(window=(32767, -32768, 12345, 0), hop=1)


@pytest.fixture(scope="module")
def cpu_runner():
    return Runner(vx=False, batch=4)


@pytest.mark.parametrize("kernel", list(cases()), ids=lambda k: k.name)
def test_rv32_program_matches_independent_reference(cpu_runner, kernel):
    x = stream(7, 70)
    for block in (8, 32):
        r = cpu_runner.run(kernel, x, block=block, neural=False)
        assert r.exact_outputs, (kernel.name, block)


def test_chunking_is_invariant(cpu_runner):
    k, x = K.FIR([(1000, -2000), (3000, 4000), (-5000, 600)]), stream(9, 64)
    a = cpu_runner.run(k, x, block=8, neural=False).outputs
    b = cpu_runner.run(k, x, block=32, neural=False).outputs
    assert a == b == k.reference(x)


def test_precision_selection():
    assert not K.FIR([(16384, 0), (8192, 0)]).wide
    assert K.FIR([(32767, 32767), (32767, 32767)]).wide


@pytest.mark.skipif(os.environ.get("RFPU_V15_GPU") != "1", reason="set RFPU_V15_GPU=1 for neural execution")
def test_neural_execution_exact():
    r = Runner(vx=False, batch=4)
    for k in (K.FIR([(16384, 0), (8192, 0)]), K.FFT4Power()):
        res = r.run(k, stream(11, 40), block=8)
        assert res.exact_tokens and res.exact_outputs

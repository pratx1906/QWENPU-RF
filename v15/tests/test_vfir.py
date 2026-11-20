import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from chipformer.rv import asm  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(asm, "VX2_ASM"), reason="needs chipformer from the rfpu-vfir branch")


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("ntaps", [1, 2, 4, 8, 16])
def test_vx_fir_matches_reference(ntaps, packed):
    from rfpu_v15.cpu.runner import Runner
    from rfpu_v15.cpu.vfir import VXFir
    rng = random.Random(ntaps)
    k = VXFir([(rng.randrange(-32768, 32768), rng.randrange(-32768, 32768)) for _ in range(ntaps)], packed=packed)
    x = [(-32768, -32768), (32767, 32767)] + [(rng.randrange(-32768, 32768), rng.randrange(-32768, 32768)) for _ in range(90)]
    r = Runner(vx=True, batch=4)
    for block in (8, 32):
        assert r.run(k, x, block=block, neural=False).exact_outputs, block


@pytest.mark.parametrize("ntaps", [1, 4, 16])
def test_vx_fir_guard_matches_reference(ntaps):
    """guard: taps x4, a chained rounding constant, rd = sat16(y >> 2); vs the independent reference and
    closer to exact accumulation than the per-tap-rounding contract."""
    from rfpu_v15.cpu.kernels import FIR
    from rfpu_v15.cpu.runner import Runner
    from rfpu_v15.cpu.vfir import VXFir
    rng = random.Random(50 + ntaps)
    taps = [(rng.randrange(-8192, 8192), rng.randrange(-8192, 8192)) for _ in range(ntaps)]
    x = [(-32768, -32768), (32767, 32767)] + [(rng.randrange(-9000, 9000), rng.randrange(-9000, 9000)) for _ in range(120)]
    r = Runner(vx=True, batch=4)
    for block in (8, 32):
        assert r.run(VXFir(taps, guard=True), x, block=block, neural=False).exact_outputs, block
    exact = FIR(taps).reference(x)
    err = lambda out: sum((a - c) ** 2 + (b - d) ** 2 for (a, b), (c, d) in zip(out[2:], exact[2:]))  # noqa: E731
    if ntaps >= 4:
        assert err(VXFir(taps, guard=True).reference(x)) < err(VXFir(taps).reference(x))


def test_vx_fir_scale():
    """Amendment 12: scale s stores taps x 2^s (non-guard): rd = s extra output bits; contract checks."""
    from rfpu_v15.cpu.runner import Runner
    from rfpu_v15.cpu.vfir import VXFir
    rng = random.Random(12)
    taps = [(rng.randrange(-6000, 6000), rng.randrange(-6000, 6000)) for _ in range(8)]
    x = [(rng.randrange(-3000, 3000), rng.randrange(-3000, 3000)) for _ in range(100)]
    k = VXFir(taps, packed=False, scale=2)
    assert Runner(vx=True, batch=4).run(k, x, block=32, neural=False).exact_outputs
    assert k.reference(x) == VXFir([(4 * a, 4 * b) for a, b in taps], packed=False).reference(x)
    with pytest.raises(ValueError):
        VXFir(taps, guard=True, scale=2)
    with pytest.raises(ValueError):
        VXFir([(9000, 0)], scale=2)

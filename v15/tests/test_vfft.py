import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from chipformer.rv import asm  # noqa: E402

pytestmark = pytest.mark.skipif("vbfly" not in getattr(asm, "VX2_ASM", {}), reason="needs chipformer from the rfpu-vfir branch (VBFLY)")


@pytest.mark.parametrize("scaling", ["block", "fixed"])
@pytest.mark.parametrize("amp", [40, 9000, 32767])
def test_vx_fft64_matches_reference_and_numpy(scaling, amp):
    from rfpu_v15.cpu.runner import Runner
    from rfpu_v15.cpu.vfft import HANN, VXFFT64
    rng = random.Random(amp)
    x = [(rng.randint(-amp, amp), rng.randint(-amp, amp)) for _ in range(128)]
    k = VXFFT64(scaling)
    res = Runner(vx=True, batch=2).run(k, x, block=64, neural=False)
    assert res.exact_outputs
    w = np.array(HANN) / 32768
    for e, out in enumerate(res.outputs):
        if out is None:
            continue
        ex, pw = out
        want = np.abs(np.fft.fft(np.array([complex(*v) for v in x[e - 63:e + 1]]) * w) / 64) ** 2
        got = np.array(pw, float) / 4.0 ** ex
        rel = np.sum((got - want) ** 2) / np.sum(want ** 2)
        if scaling == "block" or amp >= 9000:  # fixed scaling on a weak signal is the point of the ablation
            assert rel < 1e-5, (scaling, amp, rel)

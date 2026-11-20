import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from rfpu_v15.cpu.vx_kernels import (VXEnergy, ScalarEnergy, VXDense, VXDenseFrames,  # noqa: E402
                                     VXSpectralMLP, vq_ref)
from rfpu_v15.cpu.detect import EnergyDetector  # noqa: E402
from rfpu_v15.cpu.infer import SpectralMLP  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402

TOKENS = {}


@pytest.fixture(scope="module")
def runner():
    return Runner(vx=True, batch=4)


def random_stream(seed, n):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a = rng.choice([200, 3000, 32768]) if (i // 64) % 3 else 20000
        out.append((max(-32768, min(32767, rng.randrange(-a, a + 1))),
                    max(-32768, min(32767, rng.randrange(-a, a + 1)))))
    return out


def extreme_stream(n):
    vals = [-32768, 32767, -1, 0, 1, -128, 127, -129, 128, -256, 255, -32640, 32639]
    return [(vals[i % len(vals)], vals[(3 * i + 1) % len(vals)]) for i in range(n - 8)] + \
        [(-32768, -32768)] * 4 + [(32767, 32767)] * 4


def streams():
    return [random_stream(1, 300), extreme_stream(260)]


def test_vq_ref_matches_contract():
    assert vq_ref(-32768, 7, 16384) == -127 and vq_ref(32767, 7, 16384) == 127
    assert [vq_ref(v, 0, 32767) for v in range(128)] == list(range(128))
    assert vq_ref(-129, 7, 16384) == -1 and vq_ref(-1, 7, 16384) == -1 and vq_ref(127, 7, 16384) == 0


@pytest.mark.parametrize("window", [64, 32])
def test_vx_energy(runner, window):
    k = VXEnergy(window=window)
    for x in streams():
        for block in (window, 4 * window if window == 32 else 128):
            r = runner.run(k, x, block=block, neural=False)
            assert r.exact_outputs, (window, block)
            assert any(v for v in r.outputs if v is not None)
    TOKENS[f"vx_energy W={window}"] = r.stats["tokens_per_sample"]


def test_scalar_energy_baselines(runner):
    x = random_stream(1, 256)
    r = runner.run(ScalarEnergy(64), x, block=128, neural=False)
    assert r.exact_outputs
    TOKENS["scalar_energy W=64"] = r.stats["tokens_per_sample"]
    r = runner.run(VXEnergy(64), x, block=128, neural=False)
    TOKENS["vx_energy W=64 (same stream)"] = r.stats["tokens_per_sample"]
    r = runner.run(EnergyDetector(window=64, memory=8), x, block=128, neural=False)
    assert r.exact_outputs
    TOKENS["energy_detector W=64"] = r.stats["tokens_per_sample"]


def int4(rng, n):
    return [rng.randrange(-8, 8) for _ in range(n)]


@pytest.mark.parametrize("H,C", [(1, 2), (5, 3), (8, 8)])
def test_vx_spectral_mlp(runner, H, C):
    rng = random.Random(H * 10 + C)
    k = VXSpectralMLP([int4(rng, 4) for _ in range(H)], [rng.randrange(1, 1 << 16) for _ in range(H)],
                      int4(rng, H), [int4(rng, H) for _ in range(C)],
                      [rng.randrange(1, 1 << 16) for _ in range(C)], int4(rng, C),
                      fshift=rng.randrange(20, 30), sh1=rng.randrange(4, 14), r1=rng.randrange(1, 1 << 15),
                      bias_act=rng.randrange(128))
    for x in streams():
        for block in (16, 64):
            r = runner.run(k, x, block=block, neural=False)
            assert r.exact_outputs, (H, C, block)
    outs = [v for v in r.outputs if v is not None]
    assert len({o[1] for o in outs}) >= 1 and any(any(o[0]) for o in outs) or H == 1
    TOKENS[f"vx_spectral_mlp H={H} C={C}"] = r.stats["tokens_per_sample"]
    s = SpectralMLP([[rng.randrange(-32768, 32768) for _ in range(4)] for _ in range(H)],
                    [rng.randrange(-32768, 32768) for _ in range(H)],
                    [[rng.randrange(-32768, 32768) for _ in range(H)] for _ in range(C)],
                    [rng.randrange(-32768, 32768) for _ in range(C)], fshift=16)
    rs = runner.run(s, x, block=64, neural=False)
    assert rs.exact_outputs
    TOKENS[f"spectral_mlp(scalar) H={H} C={C}"] = rs.stats["tokens_per_sample"]


@pytest.mark.parametrize("N,R", [(4, 3), (8, 8), (2, 1)])
def test_vx_dense_frames(runner, N, R):
    rng = random.Random(N * 7 + R)
    d = VXDense([int4(rng, N) for _ in range(R)], [rng.randrange(1 << 16) for _ in range(R)], int4(rng, R))
    k = VXDenseFrames(d, sh_in=rng.randrange(6, 9), sh_out=rng.randrange(6, 16), r_out=rng.randrange(1 << 15))
    for x in streams():
        for block in (8, 32):
            assert runner.run(k, x, block=block, neural=False).exact_outputs, (N, R, block)


def test_zzz_report_tokens():
    for name, t in sorted(TOKENS.items()):
        print(f"{name:40s} {t:8.1f} tokens/sample")

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from rfpu_v15.cpu.detect import EnergyDetector, PeakTracker  # noqa: E402
from rfpu_v15.cpu.infer import SpectralMLP  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402


@pytest.fixture(scope="module")
def runner():
    return Runner(vx=False, batch=4)


def bursty(seed, n, amp_noise=200, amp_burst=20000):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a = amp_burst if (i // 64) % 5 == 3 else amp_noise
        out.append((rng.randrange(-a, a + 1), rng.randrange(-a, a + 1)))
    return out + [(-32768, -32768), (32767, 32767)]


def test_energy_detector_matches_reference_and_finds_bursts(runner):
    k = EnergyDetector(window=32, memory=8, alpha_q8=2048)
    x = bursty(1, 640)
    for block in (64, 128):
        r = runner.run(k, x, block=block, neural=False)
        assert r.exact_outputs
    edges = [v for v in r.outputs if v is not None and v[3]]
    assert edges, "bursts should produce start edges"


def test_energy_detector_streams_independent_of_chunking(runner):
    k = EnergyDetector(window=16, memory=4, alpha_q8=700)
    xs = [bursty(s, 300) for s in range(3)]
    a = runner.run_many(k, xs, block=32, neural=False)
    b = runner.run_many(k, xs, block=96, neural=False)
    assert [r.outputs for r in a] == [r.outputs for r in b] == [k.reference(x) for x in xs]


def test_peak_tracker(runner):
    rng = random.Random(3)
    x = [(rng.randrange(-30000, 30000), rng.randrange(-30000, 30000)) for _ in range(200)]
    x[150] = (-32768, -32768)
    r = runner.run(PeakTracker(), x, block=32, neural=False)
    assert r.exact_outputs and r.outputs[-1] == 150


def test_spectral_mlp_random_weights(runner):
    rng = random.Random(5)
    for H, C in ((1, 2), (8, 3), (4, 8)):
        k = SpectralMLP([[rng.randrange(-32768, 32768) for _ in range(4)] for _ in range(H)],
                        [rng.randrange(-32768, 32768) for _ in range(H)],
                        [[rng.randrange(-32768, 32768) for _ in range(H)] for _ in range(C)],
                        [rng.randrange(-32768, 32768) for _ in range(C)], fshift=rng.randrange(4, 24))
        x = bursty(H, 120) + [(-32768, -32768)] * 4
        for block in (16, 64):
            assert runner.run(k, x, block=block, neural=False).exact_outputs


def test_tech_classifier_matches_reference(runner):
    from rfpu_v15.cpu.classify import TechClassifier
    rng = random.Random(17)
    r16 = lambda: rng.randrange(-32768, 32768)  # noqa: E731
    for H, F in ((4, 16), (16, 64), (1, 8)):
        k = TechClassifier([[r16() for _ in range(17)] for _ in range(H)], [r16() for _ in range(H)],
                           [[r16() for _ in range(H)] for _ in range(3)], [r16() for _ in range(3)], frames=F)
        x = bursty(H, 2048) + [(-32768, -32768)] * 1000 + [(32767, 32767)] * 50
        r = runner.run(k, x, block=1024, neural=False)
        assert r.exact_outputs, (H, F)


def test_block_floating_point_front_end(runner):
    from rfpu_v15.cpu.bfp import AutoScale, BlockMax, shift_for
    rng = random.Random(23)
    for peak in (5, 32767, 32768, 2**24 * 3, 2**30):
        x = [(rng.randrange(-peak, peak + 1), rng.randrange(-peak, peak + 1)) for _ in range(150)] + [(-peak, peak)]
        m = runner.run(BlockMax(), x, block=32, neural=False)
        assert m.exact_outputs and m.outputs[-1] == peak
        a = runner.run(AutoScale(m.outputs[-1]), x, block=64, neural=False)
        assert a.exact_outputs
        assert max(max(abs(v) for v in s) for s in a.outputs) <= 32768 and (shift_for(peak) == 0 or
                                                                             max(max(abs(v) for v in s) for s in a.outputs) >= 16384)


def test_tech_classifier_bandwidth_two_stage():
    """bandwidth=True: 81 features; stage A's FFT64 powers copied in by the host (here from the FFTPower
    reference); the program equals the reference (features and MLP)."""
    import random as _r
    from rfpu_v15.cpu.classify import TechClassifier
    from rfpu_v15.cpu.runner import Runner
    from rfpu_v15.cpu.spectral import FFTPower
    rng = _r.Random(9)
    H, C = 8, 3
    w1 = [[rng.randint(-3000, 3000) for _ in range(81)] for _ in range(H)]
    k = TechClassifier(w1, [rng.randint(-500, 500) for _ in range(H)],
                       [[rng.randint(-3000, 3000) for _ in range(H)] for _ in range(C)], [0] * C, frames=16, bandwidth=True)
    x = [(rng.randint(-9000, 9000), rng.randint(-9000, 9000)) for _ in range(1024)]
    frames = [o for o in FFTPower(64, None, 64).reference(x) if o is not None]
    k.set_fft(0, frames)
    r = Runner(vx=False, batch=1).run(k, x, block=1024, neural=False)
    assert r.exact_outputs
    assert len(k.all_features(x, 0)) == 81

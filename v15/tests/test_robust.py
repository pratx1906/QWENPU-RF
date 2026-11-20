"""Robustness: firmware tool-call parsing and adversarial signals; property fuzzing of every kernel."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15"), str(ROOT / "v15/tests")]
from rfpu_v15.cpu import kernels as K  # noqa: E402
from rfpu_v15.cpu.detect import EnergyDetector, PeakTracker  # noqa: E402
from rfpu_v15.cpu.filters import DCBlocker, DecimatingFIR, IIRBiquad, NCOMixer  # noqa: E402
from rfpu_v15.cpu.infer import SpectralMLP  # noqa: E402
from rfpu_v15.cpu.nonlinear import AMEnvelope, CFOEstimate, FMDemod, MagPhase  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402
from rfpu_v15.cpu.spectral import CFARDetect, DFTBin, FFTPower, WelchPSD  # noqa: E402
from rfpu_v15.stage2.firmware import N, analyze, source  # noqa: E402
from test_stage2_firmware import parse, rows  # noqa: E402

# ------------------------------------------------------------------ firmware


def adversarial():
    t = np.arange(N)
    alt = np.where(t % 2, 1, -1)
    sigs = {"full-scale negative corner": np.full(N, -1 - 1j),
            "full-scale positive": np.full(N, (32767 + 32767j) / 32768),
            "DC": np.full(N, .5 + 0j),
            "Nyquist": .7 * alt + 0j,
            "impulse at 777": np.where(t == 777, .9 + .9j, 0),
            "alternating full scale I/Q": alt * (1 + 1j) * (32767 / 32768),
            "tone at bin 8 (-0.5 cycles)": .6 * np.exp(1j * np.pi * t)}
    out = []
    for label, z in sigs.items():
        q = np.clip(np.round(np.stack((z.real, z.imag), -1) * 32768), -32768, 32767).astype(int)
        out.append((label, [tuple(map(int, v)) for v in q]))
    return out


@pytest.fixture(scope="module")
def fw():
    from chipformer.rv import ref
    from chipformer.rv.asm import assemble
    caps = adversarial()
    return assemble(source(caps)), caps, ref


def call(fw, text, token_bytes=9):
    prog, caps, ref = fw
    m = ref.Machine.from_program(prog)
    m.run(10**6); m.halted = False
    b = text.encode()
    toks = [b[i:i + token_bytes] for i in range(0, len(b), token_bytes)]
    r = [bytes(16)] + [t.ljust(16, b"\0") for t in toks]
    m.text = r + [bytes(16)] * (128 - len(r))
    m.run(10**7)
    assert m.halted and m.fault is None, m.fault
    return m.out.decode()


WRAP = "\n<function=rf_analyze>\n<parameter=capture>\n{}\n</parameter>\n</function>\n"


@pytest.mark.parametrize("value,expect", [
    ("3", 3), (" 3 ", 3), ("003", 3), ("6", 6), ("0", 0),
    ("7", None), ("99", None), ("-1", None), ("abc", None), ("", None),
    ("4294967299", None), ("18446744073709551619", None), ("99999", None)])
def test_index_parsing(fw, value, expect):
    out = call(fw, WRAP.format(value))
    if expect is None:
        assert "error: capture must be an integer" in out and "mean_power_dbfs" not in out, out[:120]
    else:
        assert f"capture={expect} " in out, out[:120]


@pytest.mark.parametrize("token_bytes", [1, 2, 3, 5, 16])
def test_token_boundaries(fw, token_bytes):
    assert "capture=5 " in call(fw, WRAP.format("5"), token_bytes)


def test_no_parameter_and_garbage(fw):
    assert "error" in call(fw, "\n<function=rf_analyze>\n</function>\n")
    assert "error" in call(fw, "garbage without brackets " * 3)


@pytest.mark.parametrize("index", range(7))
def test_adversarial_signals(fw, index):
    out = call(fw, WRAP.format(index))
    got, want = parse(out), analyze(fw[1][index][1])
    for key, tol in (("mean_power_dbfs", .07), ("peak_power_dbfs", .07), ("noise_floor_dbfs", .07),
                     ("loudest_window_dbfs", .07)):
        assert abs(got[key] - want[key]) <= tol, (key, got[key], want[key])
    for key in ("at_sample", "active_windows"):
        assert got[key] == want[key], (key, got[key], want[key])
    # spectrum and CFO are undefined for DC/impulse-like inputs only through ties; compare when defined
    spec_ok = want["power_share_percent"] == want["power_share_percent"]
    if spec_ok and index not in (4,):
        assert got["dominant_dft16_bin"] == want["dominant_dft16_bin"]
    if index in (2, 3, 6):  # pure carriers: CFO well defined
        assert abs((got["cfo_cycles_per_sample"] - want["cfo_cycles_per_sample"] + .5) % 1 - .5) <= 2e-4


# ------------------------------------------------------------------ kernel fuzzing

def taps(rng, n, a):
    return [(rng.randrange(-a, a + 1), rng.randrange(-a, a + 1)) for _ in range(n)]


def random_kernel(rng):
    c = rng.randrange(17)
    if c == 0: return K.FIR(taps(rng, rng.randrange(1, 65), rng.choice([100, 4000, 32767])))
    if c == 1: return DecimatingFIR(taps(rng, rng.randrange(1, 13), rng.choice([2000, 32767])), rng.choice([1, 2, 4, 8]))
    if c == 2: return IIRBiquad(*[rng.randrange(-32768, 32768) for _ in range(5)])
    if c == 3: return DCBlocker(rng.randrange(-32768, 32768))
    if c == 4: return NCOMixer(rng.randrange(-2**31, 2**31))
    if c == 5: return K.FFT4Power([rng.randrange(-32768, 32768) for _ in range(4)], rng.choice([1, 2, 4]))
    n = rng.choice([8, 16, 32, 64])
    win = [rng.randrange(-32768, 32768) for _ in range(n)]
    if c == 6: return FFTPower(n, win, rng.choice([h for h in (1, 2, 4, 8, 16, 32, 64) if h <= n]), output=rng.choice(["power", "complex"]))
    if c == 7: return WelchPSD(n, win, rng.choice([h for h in (2, 4, 8) if h <= n]), rng.choice([1, 2, 4]))
    if c == 8:
        tr = rng.randrange(1, 5); g = rng.randrange(0, 3)
        n = max(n, 2 * (tr + g) + 1 if 2 * (tr + g) + 1 <= 16 else 32)
        return CFARDetect(n, [rng.randrange(-32768, 32768) for _ in range(n)], rng.choice([4, 8]), tr, g, rng.randrange(1, 65536))
    if c == 9: return DFTBin(n, rng.randrange(n), rng.choice([h for h in (1, 4, 8) if h <= n]))
    if c == 10: return EnergyDetector(rng.choice([4, 8, 16, 32]), rng.randrange(1, 17), rng.randrange(0, 4096))
    if c == 11: return PeakTracker()
    if c == 12: return MagPhase()
    if c == 13: return AMEnvelope()
    if c == 14: return FMDemod()
    if c == 15: return CFOEstimate(rng.randrange(1, 17), rng.choice([4, 8, 16, 32, 64]))
    H, C = rng.randrange(1, 9), rng.randrange(2, 9)
    r16 = lambda: rng.randrange(-32768, 32768)  # noqa: E731
    return SpectralMLP([[r16() for _ in range(4)] for _ in range(H)], [r16() for _ in range(H)],
                       [[r16() for _ in range(H)] for _ in range(C)], [r16() for _ in range(C)], rng.randrange(4, 32))


def stream(rng, n):
    kind = rng.randrange(4)
    out = []
    for i in range(n):
        if kind == 0: out.append((rng.randrange(-32768, 32768), rng.randrange(-32768, 32768)))
        elif kind == 1: out.append(rng.choice([(-32768, -32768), (32767, 32767), (-32768, 32767), (0, 0)]))
        elif kind == 2: out.append((rng.randrange(-8, 9), rng.randrange(-8, 9)))
        else: out.append((32767 if (i // 3) % 2 else -32768, 0))
    return out


@pytest.fixture(scope="module")
def cpu():
    return Runner(vx=False, batch=4)


@pytest.mark.parametrize("seed", range(60))
def test_fuzz_kernel(cpu, seed):
    rng = random.Random(9000 + seed)
    k = random_kernel(rng)
    hop = k.hop
    block = hop * rng.choice([1, 2, 4, 8]) * (1 if hop > 4 else 4)
    block = min(block, k.describe().get("max_block", 256))
    block -= block % hop
    for n in sorted({1, 2, 3, k.history, block - 1, block, block + 1, 2 * block + 3}):
        if n < 1:
            continue
        xs = [stream(rng, n), stream(rng, n + rng.randrange(0, 5))]
        res = cpu.run_many(k, xs, block=block, neural=False)
        assert all(r.exact_outputs for r in res), (k.describe(), block, n)

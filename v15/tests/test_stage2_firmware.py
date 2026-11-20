import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from rfpu_v15.stage2.firmware import N, analyze, sample_rate, source  # noqa: E402

CALL = "\n<function=rf_analyze>\n<parameter=capture>\n{}\n</parameter>\n</function>\n"


def captures():
    rng = np.random.default_rng(1)
    t = np.arange(N)
    noise = lambda s: (rng.normal(size=N) + 1j * rng.normal(size=N)) * s  # noqa: E731
    burst = noise(.003); burst[320:640] += .3 * np.exp(2j * np.pi * .19 * t[320:640])
    caps = {"tone at 0.0625 + noise fs=1MS/s": .5 * np.exp(2j * np.pi * .0625 * t) + noise(.01),
            "burst 320..640 at 0.19 fs=120MS/s": burst,
            "full-scale noise fs=200MS/s": noise(.4),
            "silence": np.zeros(N, complex)}
    out = []
    for label, z in caps.items():
        q = np.clip(np.round(np.stack((z.real, z.imag), -1) * 32768), -32768, 32767).astype(int)
        out.append((label, [tuple(map(int, v)) for v in q]))
    return out


def rows(text):
    """Text-buffer rows as the bridge writes them: row i = the i-th token's bytes (<= 16)."""
    b = text.encode()
    toks = [b[i:i + 9] for i in range(0, len(b), 9)]  # arbitrary token boundaries
    r = [bytes(16)] + [t.ljust(16, b"\0") for t in toks]
    return r + [bytes(16)] * (128 - len(r))


@pytest.fixture(scope="module")
def machine_and_caps():
    from chipformer.rv import ref
    from chipformer.rv.asm import assemble
    caps = captures()
    prog = assemble(source(caps))
    return prog, caps, ref


def run(prog, ref, text):
    m = ref.Machine.from_program(prog)
    m.run(10**6)
    assert m.halted and m.fault is None
    m.halted = False
    m.text = rows(text)
    toks = m.run(10**7)
    assert m.halted and m.fault is None, m.fault
    return m.out.decode(), len(toks)


def parse(report):
    v = {}
    for key in ("mean_power_dbfs", "peak_power_dbfs", "at_sample", "noise_floor_dbfs", "loudest_window_dbfs",
                "active_windows", "dominant_dft16_bin", "freq_cycles_per_sample", "power_share_percent",
                "cfo_cycles_per_sample"):
        m = re.search(key + r"=(-?[0-9.]+)", report)
        v[key] = float(m.group(1))
    return v


@pytest.mark.parametrize("index", range(4))
def test_report_matches_host_model(machine_and_caps, index):
    prog, caps, ref = machine_and_caps
    out, ntok = run(prog, ref, CALL.format(index))
    assert out.startswith("\n\n") and f"capture={index}" in out, out[:80]
    got, want = parse(out), analyze(caps[index][1])
    if index == 3:  # silence: power reported as -99.9
        assert got["mean_power_dbfs"] == -99.9
        return
    for key, tol in (("mean_power_dbfs", .07), ("peak_power_dbfs", .07), ("noise_floor_dbfs", .07),
                     ("loudest_window_dbfs", .07), ("power_share_percent", 1.01), ("cfo_cycles_per_sample", 2e-4)):
        assert abs(got[key] - want[key]) <= tol, (key, got[key], want[key])
    for key in ("at_sample", "active_windows", "dominant_dft16_bin"):
        assert got[key] == want[key], (key, got[key], want[key])


def test_bad_index_reports_error(machine_and_caps):
    prog, caps, ref = machine_and_caps
    out, _ = run(prog, ref, CALL.format(99))
    assert "error: capture must be an integer" in out


def test_second_call_after_park(machine_and_caps):
    prog, caps, ref = machine_and_caps
    m = ref.Machine.from_program(prog)
    m.run(10**6); m.halted = False
    for i in (1, 0):
        m.text = rows(CALL.format(i)); m.out = bytearray()
        m.run(10**7); assert m.halted and m.fault is None
        m.halted = False
        assert f"capture={i}" in m.out.decode()


# ---------------------------------------------------------------- rf_levels: compute only what's asked
CALL_L = "\n<function=rf_levels>\n<parameter=capture>\n{}\n</parameter>\n</function>\n"
LEVEL_KEYS = ("mean_power_dbfs", "peak_power_dbfs", "at_sample", "noise_floor_dbfs", "loudest_window_dbfs",
              "active_windows")


@pytest.mark.parametrize("index", range(3))
def test_levels_report_is_the_same_subset(machine_and_caps, index):
    prog, caps, ref = machine_and_caps
    full, n_full = run(prog, ref, CALL.format(index))
    lev, n_lev = run(prog, ref, CALL_L.format(index))
    assert "dominant_dft16_bin" not in lev and "cfo_cycles_per_sample" not in lev
    assert "</tool_response>" in lev or lev.count("\n") > 3
    f, g = parse(full), {}
    for key in LEVEL_KEYS:
        m = re.search(key + r"=(-?[0-9.]+)", lev)
        g[key] = float(m.group(1))
        assert g[key] == f[key], key           # identical numbers to rf_analyze's
    assert n_lev < 0.45 * n_full, (n_lev, n_full)


def test_levels_bad_index_and_alternation(machine_and_caps):
    prog, caps, ref = machine_and_caps
    assert "error: capture must be an integer" in run(prog, ref, CALL_L.format(99))[0]
    m = ref.Machine.from_program(prog)
    m.run(10**6); m.halted = False
    for call in (CALL_L, CALL, CALL_L):
        m.text = rows(call.format(1)); m.out = bytearray()
        m.run(10**7); assert m.halted and m.fault is None
        m.halted = False
        assert ("cfo_cycles_per_sample" in m.out.decode()) == (call is CALL)


@pytest.mark.parametrize("index", range(4))
def test_v3_hz_fields_and_method_notes(machine_and_caps, index):
    """Amendment 12 P12.3: freq_hz exact, cfo_hz within fs 2^-12 of the host model, method notes; no
    Hz fields when the label has no sample rate."""
    prog, caps, ref = machine_and_caps
    out, _ = run(prog, ref, CALL.format(index))
    fs = sample_rate(caps[index][0])
    assert "(averaged 16-point DFT, 8 frames)" in out and "(phase of the lag-1 autocorrelation)" in out
    if not fs:
        assert "freq_hz" not in out and "cfo_hz" not in out
        return
    want = analyze(caps[index][1], fs=fs)
    got = {k: int(re.search(k + r"=(-?[0-9]+)", out).group(1)) for k in ("freq_hz", "cfo_hz")}
    assert got["freq_hz"] == want["freq_hz"]
    assert abs(got["cfo_hz"] - want["cfo_hz"]) <= fs * 2 ** -12, (got, want["cfo_hz"])


@pytest.mark.parametrize("index", range(2))
def test_v4_not_measured_line(machine_and_caps, index):
    """Amendment 14 P14.1: both reports end with the not_measured line; the numbers are unchanged."""
    from rfpu_v15.stage2.firmware import NOT_MEASURED, NOT_MEASURED_LEVELS
    prog, caps, ref = machine_and_caps
    full, _ = run(prog, ref, CALL.format(index))
    lev, _ = run(prog, ref, CALL_L.format(index))
    assert "not_measured=" + NOT_MEASURED + " (" in full and "spectrum and frequency offset" not in full
    assert "not_measured=" + NOT_MEASURED_LEVELS in lev
    assert "error: capture" not in full and "error: capture" not in lev


def test_v3_build_has_no_not_measured_line():
    from chipformer.rv import ref
    from chipformer.rv.asm import assemble
    caps = captures()
    prog = assemble(source(caps, not_measured=False))
    out, _ = run(prog, ref, CALL.format(0))
    assert "not_measured" not in out and "cfo_cycles_per_sample" in out


def test_rf_modulation_tool_matches_host_model():
    """Amendment 15: rf_modulation runs ModFeatures + IntMLP inline in the firmware; the printed class, runner-up
    and margin equal the independent host model; rf_analyze still works in the same firmware."""
    import random
    from chipformer.rv import ref
    from chipformer.rv.asm import assemble
    from rfpu_v15.cpu.modclass import NF, IntMLP
    from rfpu_v15.modgen import capture
    from rfpu_v15.stage2.firmware import modulation_report
    r = random.Random(4)
    mlp = IntMLP([[r.randint(-3000, 3000) for _ in range(NF)] for _ in range(8)], [r.randint(-999, 999) for _ in range(8)],
                 [[r.randint(-32768, 32767) for _ in range(8)] for _ in range(25)], [r.randint(-99, 99) for _ in range(25)], s1=10)
    rng = np.random.default_rng(6)
    caps = [("synthetic a", capture("FM", rng, 20)), ("synthetic b", capture("16QAM", rng, 15))]
    prog = assemble(source(caps, modulation=mlp))
    for i, (_, x) in enumerate(caps):
        out, _ = run(prog, ref, f"\n<function=rf_modulation>\n<parameter=capture>\n{i}\n</parameter>\n</function>\n")
        c, ru, mg = modulation_report(x, mlp)
        assert f"modulation={c} runner_up={ru} score_margin={mg}\n" in out, out
    out, _ = run(prog, ref, CALL.format(1))
    assert "mean_power_dbfs=" in out and "modulation class (use rf_modulation)" in out

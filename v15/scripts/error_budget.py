"""Error budget and PROTOCOL Amendment 4 precision targets.

  python v15/scripts/error_budget.py              # neural (every token verified); writes the gate
  python v15/scripts/error_budget.py --reference  # development only: integer reference, no neural run

Arithmetic SQNR: kernel output vs a float64 model on the same quantized input, with user parameters
(taps, IIR coefficients, window codes) at their integer values and internal constants (twiddles, sine
table, CORDIC angles and gain) exact. Design SQNR: float64 with quantized user parameters vs the ideal
design. Writes v15/reports/error-budget.json and v15/reports/ERROR_BUDGET.md.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
import dsp_metrics as DM  # noqa: E402
from rfpu_v15.cpu import kernels as K  # noqa: E402
from rfpu_v15.cpu.filters import DCBlocker, DecimatingFIR, IIRBiquad, NCOMixer  # noqa: E402
from rfpu_v15.cpu.nonlinear import GAIN, MagPhase  # noqa: E402
from rfpu_v15.cpu.pipeline import lowpass_taps  # noqa: E402
from rfpu_v15.cpu.spectral import FFTPower  # noqa: E402
from rfpu_v15.cpu.vx_kernels import ScalarEnergy, VXEnergy  # noqa: E402

OUT = ROOT / "v15/reports/error-budget.json"
MD = ROOT / "v15/reports/ERROR_BUDGET.md"
N = 2048
LEVELS = list(range(-60, 1, 6))
REFERENCE_ONLY = "--reference" in sys.argv


def run(kernel, x, vx=False, block=256):
    if REFERENCE_ONLY:
        return kernel.reference(x), {"kernel_config": kernel.describe(), "samples": len(x), "reference_only": True}
    y, st = DM.neural(kernel, [x], block=block, vx=vx)
    return y[0], st


def cx(pairs):
    return np.array([complex(a, b) for a, b in pairs])


def signal(level):
    z = DM.cgauss(np.random.default_rng(1), N, level)
    x, clipped = DM.quantize(z, 15)
    return x, cx(x), clipped


def ideal_lowpass(n=16, cutoff=0.25):
    """lowpass_taps before integer rounding (same scale)."""
    mid = (n - 1) / 2
    h = [(math.sin(2 * math.pi * cutoff * (k - mid)) / (math.pi * (k - mid)) if k != mid else 2 * cutoff)
         * (0.54 - 0.46 * math.cos(2 * math.pi * k / (n - 1))) for k in range(n)]
    scale = 32767 / sum(abs(v) for v in h)
    return np.array(h) * scale


def conv(X, taps):
    return np.convolve(X, np.array([complex(a, b) for a, b in taps]) / 32768)[:len(X)]


def row(name, level, y, model, st, design=None, **extra):
    r = {"kernel": name, "level_dbfs": level, **{k: v for k, v in DM.sqnr(y, model).items()}, **st, **extra}
    if design is not None:
        r["design_sqnr_db"] = DM.sqnr(*design)["sqnr_db"]
    if st.get("tokens"):
        r["tokens_per_sample"] = st["tokens"] / st["samples"]
    print({k: r[k] for k in ("kernel", "level_dbfs", "sqnr_db")}, flush=True)
    return r


def fft_rows(rows, scaling, level, x, X):
    k = FFTPower(64, None, 64, scaling=scaling)
    y, st = run(k, x)
    w = np.array(k.window) / 32768
    ends = list(range(63, N, 64))
    model = np.array([np.abs(np.fft.fft(X[e - 63:e + 1] * w) / 64) ** 2 for e in ends])
    if scaling == "block":
        got = np.array([np.array(y[e][1], float) / 4.0 ** y[e][0] for e in ends])
        exps = sorted({y[e][0] for e in ends})
    else:
        got = np.array([np.array(y[e], float) for e in ends])
        exps = None
    rows.append(row(f"FFT64 power ({scaling} scaling)", level, got, model, st, exponents=exps))


def main():
    rows = []
    x, X, _ = signal(-12)
    # filters
    taps = lowpass_taps()
    y, st = run(K.FIR(taps, "lowpass16"), x)
    rows.append(row("lowpass16 FIR", -12, cx(y), conv(X, taps), st,
                    design=(conv(X, taps), np.convolve(X, ideal_lowpass() / 32768)[:N])))
    rng = random.Random(4)
    for n in (16, 64):
        t = [(rng.randint(-5999, 5999), rng.randint(-5999, 5999)) for _ in range(n)]
        k = K.FIR(t, f"fir{n}_random")
        y, st = run(k, x)
        rows.append(row(f"complex FIR {n} taps ({k.describe()['accumulator_bits']}-bit acc)", -12, cx(y), conv(X, t), st))
    t8 = lowpass_taps(8, 0.125)
    k = DecimatingFIR(t8, 4)
    y, st = run(k, x)
    idx = list(range(3, N, 4))
    rows.append(row("decimating FIR 8 taps /4", -12, cx([y[i] for i in idx]), conv(X, t8)[idx], st))
    from scipy.signal import lfilter
    b = (4096, 8192, 4096, -16384, 4096)
    for level in (-12, -24):
        xl, Xl, _ = signal(level)
        y, st = run(IIRBiquad(*b), xl)
        model = lfilter(np.array(b[:3]) / 16384, [1, b[3] / 16384, b[4] / 16384], Xl)
        rows.append(row("IIR biquad (+12 dB DC gain)", level, cx(y), model, st,
                        note="saturates at -12 dBFS: filter gain exceeds Q15 headroom" if level == -12 else None))
    y, st = run(DCBlocker(32440), x)
    rows.append(row("DC blocker", -12, cx(y), lfilter([1, -1], [1, -32440 / 32768], X), st))
    # oscillator
    fw = 0x12345678
    ph = 2 * math.pi * fw / 2**32 * np.arange(N)
    for interp in (False, True):
        y, st = run(NCOMixer(fw, interpolate=interp), x)
        rows.append(row(f"NCO mixer ({'Taylor-interpolated' if interp else 'table only'})", -12, cx(y),
                        X * np.exp(1j * ph), st))
    # CORDIC
    y, st = run(MagPhase(), x)
    rows.append(row("CORDIC magnitude", -12, np.array([v[0] for v in y]) / (GAIN * 8192), np.abs(X), st))
    ang = np.array([v[1] for v in y]) / 2**31 * math.pi
    err = np.angle(np.exp(1j * (ang - np.angle(X))))
    rows[-1]["rms_phase_error_rad"] = float(np.sqrt(np.mean(err ** 2)))
    # energy
    ends = list(range(63, N, 64))
    ef = np.array([np.sum(np.abs(X[e - 63:e + 1]) ** 2) for e in ends])
    y, st = run(ScalarEnergy(64), x, vx=True)
    rows.append(row("energy W64, scalar", -12, np.array([y[e] for e in ends], float), ef, st))
    y, st = run(VXEnergy(64, 7), x, vx=True)
    rows.append(row("energy W64, VX int8", -12, np.array([y[e] for e in ends], float) * 2**16, ef, st))
    # FFT: both scalings over the level sweep
    for level in LEVELS:
        xl, Xl, clipped = signal(level)
        for scaling in ("fixed", "block"):
            fft_rows(rows, scaling, level, xl, Xl)
            rows[-1]["input_clipped_components"] = clipped

    def sq(name, level):
        return next(r["sqnr_db"] for r in rows if r["kernel"] == name and r["level_dbfs"] == level)
    block = "FFT64 power (block scaling)"
    sweep = {lv: sq(block, lv) for lv in LEVELS}
    gates = {
        "P4.1": {"target": "block FFT64 power arithmetic SQNR >= 65 dB at every level -60..0 dBFS",
                 "min_db": min(sweep.values()), "passed": min(sweep.values()) >= 65},
        "P4.2": {"target": "block FFT64 power arithmetic SQNR >= 70 dB at -12 dBFS",
                 "value_db": sweep[-12], "baseline_fixed_db": sq("FFT64 power (fixed scaling)", -12),
                 "passed": sweep[-12] >= 70},
        "P4.3": {"target": "interpolated NCO arithmetic SQNR >= 75 dB at -12 dBFS",
                 "value_db": sq("NCO mixer (Taylor-interpolated)", -12),
                 "baseline_table_db": sq("NCO mixer (table only)", -12),
                 "passed": sq("NCO mixer (Taylor-interpolated)", -12) >= 75},
        "P4.4": {"target": "all rows neural-exact and reported",
                 "passed": not REFERENCE_ONLY, "rows": len(rows)},
    }
    rep = {"protocol": "PROTOCOL.md Amendment 4", "reference_only": REFERENCE_ONLY,
           "input": "complex Gaussian, seed 1, 2048 samples, Q15", "gates": gates, "rows": rows}
    if REFERENCE_ONLY:
        print(json.dumps(gates, indent=1))
        return
    OUT.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    write_md(rep)
    print(json.dumps(gates, indent=1))
    if not all(g["passed"] for g in gates.values()):
        raise SystemExit(1)


def write_md(rep):
    f = lambda v: "inf" if v == float("inf") else f"{v:.1f}"  # noqa: E731
    lines = ["# Error budget (PROTOCOL Amendment 4)", "",
             "Generated by `v15/scripts/error_budget.py`. Every output is neural (every token equal to the RV32 "
             "reference, outputs equal to the independent integer reference). Arithmetic SQNR compares with float64 "
             "on the same quantized input, user parameters at their integer values, internal constants exact. "
             "Design SQNR compares quantized user parameters with the ideal design.", "",
             "## Gates", "", "| ID | Target | Result | Pass |", "|---|---|---|---|"]
    for gid, g in rep["gates"].items():
        val = ", ".join(f"{k} {f(v) if isinstance(v, float) else v}" for k, v in g.items() if k not in ("target", "passed"))
        lines.append(f"| {gid} | {g['target']} | {val} | {'PASS' if g['passed'] else 'FAIL'} |")
    lines += ["", "## Per-kernel budget (-12 dBFS)", "",
              "| Kernel | Arithmetic SQNR dB | ENOB | Design SQNR dB | Tokens/sample | ms/sample |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in rep["rows"]:
        if r["level_dbfs"] == -12 or not r["kernel"].startswith("FFT"):
            enob = "" if r["enob_bits"] is None else f"{r['enob_bits']:.1f}"
            lines.append(f"| {r['kernel']} ({r['level_dbfs']} dBFS) | {f(r['sqnr_db'])} | {enob} | "
                         f"{f(r['design_sqnr_db']) if 'design_sqnr_db' in r else ''} | "
                         f"{r.get('tokens_per_sample', 0):.0f} | {r.get('ms_per_sample', 0):.1f} |")
    lines += ["", "## FFT64 power: arithmetic SQNR vs input level", "", "| Level dBFS | Fixed scaling dB | Block scaling dB |",
              "|---:|---:|---:|"]
    for lv in LEVELS:
        a = next(r for r in rep["rows"] if r["kernel"] == "FFT64 power (fixed scaling)" and r["level_dbfs"] == lv)
        b = next(r for r in rep["rows"] if r["kernel"] == "FFT64 power (block scaling)" and r["level_dbfs"] == lv)
        lines.append(f"| {lv} | {f(a['sqnr_db'])} | {f(b['sqnr_db'])} |")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

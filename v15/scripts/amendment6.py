"""PROTOCOL Amendment 6, neural (needs PYTHONPATH=upstream/QwenPU-rf: VBFLY and VFIR).

  P6.1  FFT64 power, VBFLY + block floating point, arithmetic SQNR at -60..0 dBFS (>= 68 dB every level),
        tokens and ms per sample vs the CPU block FFT (FFTPower scaling="block") on the same circuit
  P6.2  IIR biquad wide=True at -12 and -6 dBFS (>= 65 dB), int16 version alongside
  P6.3  VFIR 16 taps (seed 4, |c| < 6000), per-tap rounding, arithmetic SQNR at -12 dBFS (target >= 70 dB)
Same definitions as Amendment 4 (error_budget.py). Writes v15/reports/amendment6.json.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
import dsp_metrics as DM  # noqa: E402
from error_budget import N, conv, signal  # noqa: E402
from rfpu_v15.cpu.filters import IIRBiquad  # noqa: E402
from rfpu_v15.cpu.spectral import FFTPower  # noqa: E402

OUT = ROOT / "v15/reports/amendment6.json"
LEVELS = list(range(-60, 1, 6))


def main():
    import chipformer
    if "QwenPU-rf" not in chipformer.__file__:
        raise SystemExit("set PYTHONPATH=upstream/QwenPU-rf")
    from rfpu_v15.cpu.vfft import HANN, VXFFT64
    from rfpu_v15.cpu.vfir import VXFir
    from scipy.signal import lfilter
    rows = []

    def add(name, level, y, model, st, **kw):
        r = {"kernel": name, "level_dbfs": level, **DM.sqnr(y, model), **st,
             "tokens_per_sample": st["tokens"] / st["samples"], **kw}
        rows.append(r)
        print({k: r[k] for k in ("kernel", "level_dbfs", "sqnr_db", "tokens_per_sample", "ms_per_sample")}, flush=True)
        return r

    # P6.1
    w = np.array(HANN) / 32768
    for level in LEVELS:
        x, X, clipped = signal(level)
        ends = list(range(63, N, 64))
        model = np.array([np.abs(np.fft.fft(X[e - 63:e + 1] * w) / 64) ** 2 for e in ends])
        y, st = DM.neural(VXFFT64(), [x], vx=True)
        got = np.array([np.array(y[0][e][1], float) / 4.0 ** y[0][e][0] for e in ends])
        add("FFT64 power, VBFLY block floating point", level, got, model, st, input_clipped_components=clipped)
        k = FFTPower(64, None, 64, scaling="block")
        y, st = DM.neural(k, [x], vx=True)
        got = np.array([np.array(y[0][e][1], float) / 4.0 ** y[0][e][0] for e in ends])
        add("FFT64 power, CPU block floating point", level, got, model, st)
    # P6.2
    b = (4096, 8192, 4096, -16384, 4096)
    for level in (-12, -6):
        x, X, _ = signal(level)
        model = lfilter(np.array(b[:3]) / 16384, [1, b[3] / 16384, b[4] / 16384], X)
        for wide in (True, False):
            y, st = DM.neural(IIRBiquad(*b, wide=wide), [x], vx=True)
            add(f"IIR biquad (+12 dB DC gain), {'wide int32' if wide else 'int16 saturating'}", level,
                np.array([complex(*v) for v in y[0]]), model, st)
    # P6.3
    rng = random.Random(4)
    taps = [(rng.randint(-5999, 5999), rng.randint(-5999, 5999)) for _ in range(16)]
    x, X, _ = signal(-12)
    y, st = DM.neural(VXFir(taps, packed=False), [x], vx=True)
    add("VFIR 16 taps (per-tap rounding)", -12, np.array([complex(*v) for v in y[0]]), conv(X, taps), st)

    def sq(name, level):
        return next(r["sqnr_db"] for r in rows if r["kernel"] == name and r["level_dbfs"] == level)
    vb = "FFT64 power, VBFLY block floating point"
    sweep = {lv: sq(vb, lv) for lv in LEVELS}
    gates = {
        "P6.1": {"target": ">= 68 dB at every level -60..0 dBFS", "min_db": min(sweep.values()), "passed": min(sweep.values()) >= 68},
        "P6.2": {"target": ">= 65 dB at -12 and -6 dBFS (wide)", "values_db": [sq("IIR biquad (+12 dB DC gain), wide int32", lv) for lv in (-12, -6)],
                 "passed": all(sq("IIR biquad (+12 dB DC gain), wide int32", lv) >= 65 for lv in (-12, -6))},
        "P6.3": {"target": ">= 70 dB at -12 dBFS (measurement)", "value_db": sq("VFIR 16 taps (per-tap rounding)", -12),
                 "passed": sq("VFIR 16 taps (per-tap rounding)", -12) >= 70},
    }
    OUT.write_text(json.dumps({"protocol": "PROTOCOL.md Amendment 6", "chipformer": chipformer.__file__,
                               "gates": gates, "rows": rows}, indent=2, default=float), encoding="utf-8")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()

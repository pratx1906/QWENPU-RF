"""PROTOCOL Amendment 8, neural (needs PYTHONPATH=upstream/QwenPU-rf).

  P8.1  VFIR guard-bit mode, 16 taps (seed 4, |c| < 6000), arithmetic SQNR at -12 dBFS (>= 75 dB), with the
        per-tap-rounding VFIR and the exact-accumulation scalar FIR16 alongside (same circuit)
  P8.6  VXFFT64 tracked block floating point vs untracked fixed scaling, -60..0 dBFS (ablation)
Same definitions as Amendment 4. Writes v15/reports/amendment8.json.
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
from rfpu_v15.cpu import kernels as K  # noqa: E402

OUT = ROOT / "v15/reports/amendment8.json"
LEVELS = list(range(-60, 1, 6))


def main():
    import chipformer
    if "QwenPU-rf" not in chipformer.__file__:
        raise SystemExit("set PYTHONPATH=upstream/QwenPU-rf")
    from rfpu_v15.cpu.vfft import HANN, VXFFT64
    from rfpu_v15.cpu.vfir import VXFir
    rows = []

    def add(name, level, y, model, st, **kw):
        r = {"kernel": name, "level_dbfs": level, **DM.sqnr(y, model), **st,
             "tokens_per_sample": st["tokens"] / st["samples"], **kw}
        rows.append(r)
        print({k: r[k] for k in ("kernel", "level_dbfs", "sqnr_db", "tokens_per_sample", "ms_per_sample")}, flush=True)

    rng = random.Random(4)
    taps = [(rng.randint(-5999, 5999), rng.randint(-5999, 5999)) for _ in range(16)]
    x, X, _ = signal(-12)
    for name, k in (("VFIR16 guard (taps x4, one final rounding)", VXFir(taps, packed=False, guard=True)),
                    ("VFIR16 per-tap rounding", VXFir(taps, packed=False)),
                    ("scalar FIR16 exact accumulation", K.FIR(taps, "fir16"))):
        y, st = DM.neural(k, [x], vx=True)
        add(name, -12, np.array([complex(*v) for v in y[0]]), conv(X, taps), st)
    w = np.array(HANN) / 32768
    for level in LEVELS:
        x, X, clipped = signal(level)
        ends = list(range(63, N, 64))
        model = np.array([np.abs(np.fft.fft(X[e - 63:e + 1] * w) / 64) ** 2 for e in ends])
        for scaling in ("block", "fixed"):
            y, st = DM.neural(VXFFT64(scaling), [x], vx=True)
            got = np.array([np.array(y[0][e][1], float) / 4.0 ** y[0][e][0] for e in ends])
            add(f"VBFLY FFT64 {scaling}", level, got, model, st)

    def sq(name, level):
        return next(r["sqnr_db"] for r in rows if r["kernel"] == name and r["level_dbfs"] == level)
    g = sq("VFIR16 guard (taps x4, one final rounding)", -12)
    gates = {"P8.1": {"target": ">= 75 dB at -12 dBFS", "value_db": g, "per_tap_db": sq("VFIR16 per-tap rounding", -12),
                      "exact_fir_db": sq("scalar FIR16 exact accumulation", -12), "passed": g >= 75},
             "P8.6": {"block_db": [sq("VBFLY FFT64 block", lv) for lv in LEVELS],
                      "fixed_db": [sq("VBFLY FFT64 fixed", lv) for lv in LEVELS], "levels": LEVELS, "passed": True}}
    OUT.write_text(json.dumps({"protocol": "PROTOCOL.md Amendment 8", "chipformer": chipformer.__file__,
                               "gates": gates, "rows": rows}, indent=2, default=float), encoding="utf-8")
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()

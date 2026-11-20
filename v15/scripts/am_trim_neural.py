"""PROTOCOL Amendment 5, P5.2: AM envelope with a magnitude-only CORDIC, 16 and 10 iterations, neural.
Arithmetic SQNR vs float64 |x| of the same quantized input (gain(k) exact), -12 dBFS complex Gaussian."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
import dsp_metrics as DM  # noqa: E402
from rfpu_v15.cpu.nonlinear import AMEnvelope, gain  # noqa: E402

BEFORE = 450  # tokens/sample, 16 iterations with angle accumulation (README capability matrix)


def main():
    x, _ = DM.quantize(DM.cgauss(np.random.default_rng(1), 2048, -12), 15)
    X = np.array([complex(a, b) for a, b in x])
    rows = []
    for k in (16, 10):
        y, st = DM.neural(AMEnvelope(k), [x])
        r = {"iterations": k, **DM.sqnr(np.array(y[0], float) / (gain(k) * 8192), np.abs(X)),
             "tokens_per_sample": st["tokens"] / st["samples"], "ms_per_sample": st["ms_per_sample"]}
        rows.append(r)
        print(r, flush=True)
    p52 = rows[1]["sqnr_db"] >= 90
    rep = {"protocol": "PROTOCOL.md Amendment 5 P5.2", "before_tokens_per_sample": BEFORE, "rows": rows,
           "P5.2": {"target": "k=10 arithmetic SQNR >= 90 dB", "value_db": rows[1]["sqnr_db"], "passed": p52}}
    (ROOT / "v15/reports/am-trim-neural.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    if not p52:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

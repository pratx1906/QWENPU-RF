"""PROTOCOL Amendment 12 P12.5 / P12.6 (with 12a): IIR extra output bits and VFIR adaptive output scale
on the Amendment 10b battery inputs (133 recordings, 256 samples, Q2.13). Needs PYTHONPATH=upstream/QwenPU-rf.
Writes v15/reports/a12-battery.json.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
OUT = ROOT / "v15/reports/a12-battery.json"
SEL = ROOT / "v15/reports/real-iq-selection-expanded.json"
N = 256


def sqnr(a, r):
    a, r = np.asarray(a, complex).ravel(), np.asarray(r, complex).ravel()
    e = float(np.sum(abs(a - r) ** 2))
    return 10 * math.log10(float(np.sum(abs(r) ** 2)) / e) if e > 0 else float("inf")


def stats(v):
    return {"median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "min": float(min(v))}


def main():
    import chipformer
    if "QwenPU-rf" not in chipformer.__file__:
        raise SystemExit("set PYTHONPATH=upstream/QwenPU-rf")
    from scipy.signal import lfilter
    sys.argv = [sys.argv[0], "--headroom"]
    import real_iq_q15 as R
    from rfpu_v15.cpu.bfp import BlockMax
    from rfpu_v15.cpu.filters import IIRBiquad
    from rfpu_v15.cpu.runner import Runner
    from rfpu_v15.cpu.vfir import VXFir
    rows = json.loads(SEL.read_text(encoding="utf-8"))["captures"]
    streams = [R.inputs(r)[1][:N] for r in rows]
    names = [f"{r['dataset']}:{Path(r['path']).parent.name}/{Path(r['path']).name}" for r in rows]
    Xs = [np.array([complex(*v) for v in q]) for q in streams]
    runner = Runner(vx=True, batch=16)
    rng = random.Random(4)
    vtaps = [(rng.randint(-5999, 5999), rng.randint(-5999, 5999)) for _ in range(16)]
    H = np.array([complex(*t) for t in vtaps]) / 32768
    b = (4096, 8192, 4096, -16384, 4096)
    report = {"protocol": "PROTOCOL.md Amendment 12 (P12.5, P12.6 with 12a)", "recordings": len(rows), "names": names}

    def run(k, ss):
        res = runner.run_many(k, ss, block=256)
        if not all(r.exact_tokens and r.exact_outputs for r in res):
            raise SystemExit(f"{k.name}: not exact")
        return res

    # P12.5
    k = IIRBiquad(*b, wide=True, out_frac=4)
    res = run(k, streams)
    v = [sqnr(np.array([complex(*y) for y in r.outputs]) / 16,
              lfilter(np.array(b[:3]) / 16384, [1, b[3] / 16384, b[4] / 16384], X)) for r, X in zip(res, Xs)]
    report["P12.5"] = {"kernel": k.describe(), "exact": f"{len(res)}/{len(res)}", "tokens": sum(r.tokens for r in res),
                       "sqnr_db": stats(v), "per_recording": v, "pass": min(v) >= 70}
    print({"P12.5": report["P12.5"]["sqnr_db"], "pass": report["P12.5"]["pass"]}, flush=True)
    # P12.6: guard pass, BlockMax of its output (in weights), s in {0, 2}, scaled pass where s = 2
    g = run(VXFir(vtaps, packed=False, guard=True), streams)
    bm = run(BlockMax(), [r.outputs for r in g])
    ms = [r.outputs[-1] for r in bm]
    s = [2 if (m + 1) * 4 + 16 <= 32767 else 0 for m in ms]
    idx = [i for i in range(len(rows)) if s[i] == 2]
    sc = run(VXFir(vtaps, packed=False, scale=2), [streams[i] for i in idx]) if idx else []
    outs = [np.array([complex(*y) for y in r.outputs]) for r in g]
    for i, r in zip(idx, sc):
        outs[i] = np.array([complex(*y) for y in r.outputs]) / 4
    sat = sum(int(abs(y) >= 32767) for i, r in zip(idx, sc) for z in r.outputs for y in z) + \
        sum(int(abs(y) >= 32767) for i, r in enumerate(g) if s[i] == 0 for z in r.outputs for y in z)
    v = [sqnr(o, np.convolve(X, H)[:N]) for o, X in zip(outs, Xs)]
    report["P12.6"] = {"block_max": ms, "scale_bits": s, "scaled_recordings": len(idx), "saturated_samples": sat,
                       "exact": "all passes token-exact (guard, BlockMax, scaled)", "sqnr_db": stats(v), "per_recording": v,
                       "pass": min(v) >= 70 and sat == 0}
    print({"P12.6": report["P12.6"]["sqnr_db"], "scaled": len(idx), "sat": sat, "pass": report["P12.6"]["pass"]}, flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

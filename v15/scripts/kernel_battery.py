"""PROTOCOL Amendment 10b: the main kernels on every expanded real recording (needs PYTHONPATH=upstream/QwenPU-rf).

Each kernel runs neurally over all recordings as batched streams (256 samples each, Q2.13 inputs as the
pipeline). Pass = every kernel on every recording token-exact with outputs equal to its independent
reference. Arithmetic SQNR (Amendment 4 definition) per recording where a float model exists.
Writes v15/reports/kernel-battery.json.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
OUT = ROOT / "v15/reports/kernel-battery.json"
SEL = ROOT / "v15/reports/real-iq-selection-expanded.json"
N = 256


def sqnr(a, r):
    a, r = np.asarray(a, complex).ravel(), np.asarray(r, complex).ravel()
    e, p = float(np.sum(abs(a - r) ** 2)), float(np.sum(abs(r) ** 2))
    return 10 * math.log10(p / e) if e > 0 and p > 0 else (float("inf") if p > 0 else None)


def main():
    smoke = "--smoke" in sys.argv  # development check: 3 recordings, reference machine only (not a result)
    import chipformer
    if "QwenPU-rf" not in chipformer.__file__:
        raise SystemExit("set PYTHONPATH=upstream/QwenPU-rf")
    from scipy.signal import lfilter
    sys.argv = [sys.argv[0], "--headroom"] + (["--smoke"] if smoke else [])
    import real_iq_q15 as R
    from rfpu_v15.cpu.filters import IIRBiquad, NCOMixer
    from rfpu_v15.cpu.nonlinear import GAIN, CFOEstimate, FMDemod, MagPhase
    from rfpu_v15.cpu.runner import Runner
    from rfpu_v15.cpu.spectral import CFARDetect, FFTPower, WelchPSD
    from rfpu_v15.cpu.vfft import HANN, VXFFT64
    from rfpu_v15.cpu.vfir import VXFir
    rows = json.loads(SEL.read_text(encoding="utf-8"))["captures"][:3 if smoke else None]
    streams, names, levels = [], [], []
    for r in rows:
        _, q, _ = R.inputs(r)
        q = q[:N]
        streams.append(q)
        names.append(f"{r['dataset']}:{Path(r['path']).parent.name}/{Path(r['path']).name}")
        X = np.array([complex(*v) for v in q])
        levels.append(10 * math.log10(max(np.mean(abs(X) ** 2), 1e-12) / 2**26))  # dBFS of Q2.13 full scale
    Xs = [np.array([complex(*v) for v in q]) for q in streams]
    runner = Runner(vx=True, batch=16)
    rng = random.Random(4)
    vtaps = [(rng.randint(-5999, 5999), rng.randint(-5999, 5999)) for _ in range(16)]
    b = (4096, 8192, 4096, -16384, 4096)
    fw = 0x12345678
    ph = 2 * math.pi * fw / 2**32 * np.arange(N)
    w = np.array(HANN) / 32768

    def fft_model(X):
        return [np.abs(np.fft.fft(X[e - 63:e + 1] * w) / 64) ** 2 for e in range(63, N, 64)]

    kernels = [
        ("FFT64 power, block floating point", FFTPower(64, None, 64, scaling="block"),
         lambda y, X: sqnr([np.array(y[e][1], float) / 4.0 ** y[e][0] for e in range(63, N, 64)], fft_model(X))),
        ("Welch PSD 64x4", WelchPSD(64, None, 32, 4), None),
        ("CA-CFAR 64", CFARDetect(64, None, 64), None),
        ("CORDIC magnitude/phase", MagPhase(), lambda y, X: sqnr(np.array([v[0] for v in y]) / (GAIN * 8192), np.abs(X))),
        ("FM discriminator", FMDemod(), None),
        ("CFO estimate lag 1 / 64", CFOEstimate(lag=1, window=64), None),
        ("NCO, Taylor-interpolated", NCOMixer(fw, interpolate=True), lambda y, X: sqnr([complex(*v) for v in y], X * np.exp(1j * ph))),
        ("IIR biquad, wide", IIRBiquad(*b, wide=True),
         lambda y, X: sqnr([complex(*v) for v in y], lfilter(np.array(b[:3]) / 16384, [1, b[3] / 16384, b[4] / 16384], X))),
        ("VFIR16 guard", VXFir(vtaps, packed=False, guard=True),
         lambda y, X: sqnr([complex(*v) for v in y], np.convolve(X, np.array([complex(*t) for t in vtaps]) / 32768)[:N])),
        ("VBFLY FFT64, block floating point", VXFFT64(),
         lambda y, X: sqnr([np.array(y[e][1], float) / 4.0 ** y[e][0] for e in range(63, N, 64)], fft_model(X))),
    ]
    report = {"protocol": "PROTOCOL.md Amendment 10b", "recordings": len(rows), "samples_per_recording": N,
              "input_format": "Q2.13", "kernels": []}
    for name, k, model in kernels:
        res = runner.run_many(k, streams, block=256, neural=not smoke)
        exact = [r.exact_tokens and r.exact_outputs for r in res]
        row = {"kernel": name, "config": k.describe() if hasattr(k, "describe") else None,
               "exact_recordings": int(sum(exact)), "recordings": len(res), "tokens": res[0].tokens,
               "neural_seconds": res[0].neural_seconds}
        if model:
            vals = [model(r.outputs, X) for r, X in zip(res, Xs)]
            finite = [v for v in vals if v is not None and v != float("inf")]
            row["sqnr_db"] = {"per_recording": vals, "median": float(np.median(finite)) if finite else None,
                              "p5": float(np.percentile(finite, 5)) if finite else None, "min": min(finite) if finite else None}
        report["kernels"].append(row)
        print({k_: v for k_, v in row.items() if k_ not in ("config",)} if not model else
              {"kernel": name, "exact": f"{row['exact_recordings']}/{row['recordings']}",
               "sqnr_median": row["sqnr_db"]["median"], "sqnr_min": row["sqnr_db"]["min"]}, flush=True)
        if not smoke:
            OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    fft = next(r for r in report["kernels"] if r["kernel"] == "FFT64 power, block floating point")
    report["names"], report["input_level_dbfs_q213"] = names, levels
    report["gates"] = {"all_exact": all(r["exact_recordings"] == r["recordings"] for r in report["kernels"]),
                       "fft64_block_ge_65db_every_recording": fft["sqnr_db"]["min"] is not None and fft["sqnr_db"]["min"] >= 65}
    if not smoke:
        OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report["gates"]))
    if not report["gates"]["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

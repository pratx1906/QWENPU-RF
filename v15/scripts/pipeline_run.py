"""One command: full RF DSP-AI pipeline on QwenPU's CPU in weights, with pass/fail.

  python v15/scripts/pipeline_run.py                 # all 14 real captures, 2048 samples each
  python v15/scripts/pipeline_run.py --captures 0 3  # selected captures
  python v15/scripts/pipeline_run.py --synthetic     # synthetic bursts with known ground truth

Input format Q2.13 (full scale +-4, no clipping; see PROTOCOL Amendment 1). PASS means every stage
of every stream ran free-running with all tokens equal to the RV32 reference and all outputs equal
to the independent integer reference.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
from rfpu_v15.cpu.pipeline import run_pipeline  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402


def synthetic(seed, n=2048):
    """Noise with known bursts: a tone burst at [512, 1024) and a wideband burst at [1536, 1792)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 150
    t = np.arange(n)
    x[512:1024] += 9000 * np.exp(1j * 0.6 * t[512:1024])
    x[1536:1792] += (rng.normal(size=256) + 1j * rng.normal(size=256)) * 6000
    q = np.clip(np.round(np.stack((x.real, x.imag), -1)), -32768, 32767).astype(int)
    return [tuple(map(int, s)) for s in q], {"tone_burst": [512, 1024], "wideband_burst": [1536, 1792]}


def summarize(name, k, stages, outs, x2):
    S = stages
    energy = [v for v in outs["energy"][k].outputs if v is not None]
    starts = [64 * (i + 1) - 1 for i, v in enumerate(energy) if v[3]]
    ends = [64 * (i + 1) - 1 for i, v in enumerate(energy) if v[4]]
    cfar = [v for v in outs["cfar"][k].outputs if v is not None]
    classes = Counter(v[1] for v in outs["classifier"][k].outputs if v is not None)
    cfo = [v[0] for v in outs["cfo"][k].outputs if v is not None]
    import math
    return {
        "stream": name,
        "all_stages_exact": all(outs[s][k].exact_tokens and outs[s][k].exact_outputs for s in outs),
        "burst_start_samples": starts, "burst_end_samples": ends,
        "noise_floor_last": energy[-1][1] if energy else None,
        "cfar_frames_with_detection": sum(1 for m, _ in cfar if m), "cfar_frames": len(cfar),
        "dominant_fft16_bin": Counter(p for _, p in cfar).most_common(1)[0][0] if cfar else None,
        "class_histogram": {["noise", "tone", "wideband"][c]: n for c, n in sorted(classes.items())},
        "cfo_median_rad_per_sample": sorted(cfo)[len(cfo) // 2] / 2**31 * math.pi if cfo else None,
        "peak_index": outs["peak"][k].outputs[-1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=int, nargs="*")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "v15/reports/pipeline.json"))
    ap.add_argument("--selection", default=str(ROOT / "data/real-iq-selection.json"),
                    help="capture list (Amendment 10: v15/reports/real-iq-selection-expanded.json)")
    a = ap.parse_args()
    mlp = json.loads((ROOT / "v15/configs/spectral-mlp.json").read_text())
    names, streams, truth = [], [], {}
    if a.synthetic:
        for s in range(4):
            x, t = synthetic(s, a.samples)
            names.append(f"synthetic-{s}"); streams.append(x); truth[f"synthetic-{s}"] = t
    else:
        sys.argv = [sys.argv[0], "--headroom"]  # Q2.13 inputs (Amendment 1)
        import real_iq_q15
        rows = json.loads(Path(a.selection).read_text(encoding="utf-8"))["captures"]
        for i in (a.captures if a.captures else range(len(rows))):
            _, q, _ = real_iq_q15.inputs(rows[i])
            names.append(f"{rows[i]['dataset']}:{rows[i]['emitter']}:{Path(rows[i]['path']).parent.name}/{Path(rows[i]['path']).name}")
            streams.append(q[:a.samples])
    runner = Runner(vx=False, batch=a.batch)
    t = time.perf_counter()
    stages, x1, x2, outs = run_pipeline(runner, streams, mlp)
    wall = time.perf_counter() - t
    results = [summarize(n, k, stages, outs, x2) for k, n in enumerate(names)]
    tokens = sum(outs[s][0].tokens for s in outs)
    neural = sum(outs[s][0].neural_seconds for s in outs)
    report = {"status": "PASS" if all(r["all_stages_exact"] for r in results) else "FAIL",
              "backend": "QwenPU RV32 CPU in weights, chipformer fusedexec.FreeRun3",
              "input_format": "Q2.13" if not a.synthetic else "Q15 synthetic",
              "stages": {k: v.describe() for k, v in stages.items()}, "streams": len(streams),
              "samples_per_stream": a.samples, "tokens": tokens, "neural_seconds": neural, "wall_seconds": wall,
              "tokens_per_sample": tokens / sum(len(s) for s in streams),
              "neural_ms_per_sample": 1000 * neural / sum(len(s) for s in streams),
              "results": results, "ground_truth": truth}
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['stream'][:60]:60s} exact={r['all_stages_exact']} bursts={r['burst_start_samples'][:4]} "
              f"classes={r['class_histogram']} peak={r['peak_index']}", flush=True)
    print(f"{report['status']}: {len(streams)} streams, {tokens:,} tokens, {report['neural_ms_per_sample']:.1f} ms/sample neural")
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

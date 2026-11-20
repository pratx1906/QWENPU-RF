"""Primary v1.5 fidelity gate: 14 real captures, Q15, every sample executed on the CPU in weights."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15")]
from rfpu_v15.cpu import kernels as K  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402
from rfpu_v15.paths import local  # noqa: E402

CRITERIA = {"FIR_NMSE_max": .01, "spectral_power_NMSE_max": .01, "matched_peak_displacement_max": 1}
BFP = "--bfp" in sys.argv  # Amendment 2: host writes Q24 int32 codes; the CPU picks the Q15 scale
FRAC = 24 if BFP else 13 if "--headroom" in sys.argv else 15  # Amendment 1: Q2.13 secondary diagnostic
SCALE = 1 << FRAC
OUT = ROOT / ("v15/reports/real-iq-bfp.json" if BFP else "v15/reports/real-iq-q13-headroom.json" if FRAC == 13 else "v15/reports/real-iq-q15.json")
LIMIT = 2**31 if BFP else 32768
SELECTION = Path(sys.argv[sys.argv.index("--selection") + 1]) if "--selection" in sys.argv else ROOT / "data/real-iq-selection.json"
if "--out" in sys.argv:  # Amendment 9: expanded screen with its own selection and report
    OUT = ROOT / sys.argv[sys.argv.index("--out") + 1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(actual, reference):
    import numpy as np
    a, r = np.asarray(actual), np.asarray(reference)
    error, energy = float(np.sum(abs(a - r) ** 2)), float(np.sum(abs(r) ** 2))
    return {"nmse": error / energy if energy else None, "rmse": float(np.sqrt(np.mean(abs(a - r) ** 2))),
            "max_absolute_error": float(np.max(abs(a - r))), "reference_energy": energy}


def inputs(row):
    """v1.4 preprocessing (v14/scripts/real_iq.py) with Q15 instead of Q3 input codes."""
    import numpy as np
    path = local(row["path"])
    assert sha256(path) == row["sha256"], "dataset bytes changed"
    if row["format"] == "ci16_le":
        raw = np.memmap(path, dtype="<i2", mode="r").reshape(-1, 2)

        def read(a, b):
            z = raw[a:b].astype(float) / row["ADC_divisor"]
            return z[:, 0] + 1j * z[:, 1]
    elif row["format"] == "cf32_le":
        raw = np.memmap(path, dtype="<c8", mode="r")

        def read(a, b):
            return np.asarray(raw[a:b], dtype=np.complex128)
    else:
        raise ValueError("unsupported recorded datatype")
    count = len(raw)
    annotations = row.get("metadata", {}).get("annotations", [])
    start = max(0, int(annotations[0].get("core:sample_start", 0))) if annotations else 0
    if start + 528 > count:
        start = 0
    prefix = read(start, start + 512)
    scale = float(np.percentile(np.abs(np.concatenate((prefix.real, prefix.imag))), 95))
    gain = .75 / scale if scale > 0 else 1.
    offset = start + 512
    z = read(offset, min(offset + 4096, count)) * gain
    scaled = np.stack((z.real, z.imag), axis=1) * SCALE
    rounded = np.sign(scaled) * np.floor(abs(scaled) + .5)
    q = np.clip(rounded, -LIMIT, LIMIT - 1).astype(np.int64)
    meta = {"source_sample_start": offset, "gain": gain, "samples": len(z),
            "component_clipping_count": int(np.count_nonzero((rounded < -LIMIT) | (rounded > LIMIT - 1))),
            "input_quantization": metrics(q[:, 0] / SCALE + 1j * q[:, 1] / SCALE, z)}
    return z, [tuple(map(int, s)) for s in q], meta


def main():
    import numpy as np
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))["captures"]
    h = np.array([.5, .25], dtype=complex)
    fir = K.FIR([(16384, 0), (8192, 0)], "gate_fir")
    matched = K.FIR([(16384, 0), (8192, 0)], "gate_matched")  # conj(h) == h for real taps
    spectral = K.FFT4Power(window=(0, 24576, 24576, 0), hop=2)
    runner = Runner(vx=False, batch=16)
    report = {"status": "running", "criteria": CRITERIA, "protocol": "v15/research/PROTOCOL.md", "input_fractional_bits": FRAC,
              "analysis": "BFP front end in weights (Amendment 2)" if BFP else "secondary post-hoc headroom diagnostic (Amendment 1)" if FRAC == 13 else "primary gate",
              "backend": "QwenPU RV32 CPU compiled into weights; chipformer fusedexec.FreeRun3, free-running",
              "kernels": [k.describe() for k in (fir, matched, spectral)], "recordings": []}

    def save():
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    try:
        for row in selection:
            try:
                z, q, meta = inputs(row)
            except (OSError, ValueError, AssertionError) as e:  # an unreadable recording is reported, not dropped
                report["recordings"].append({"dataset": row["dataset"], "emitter": row["emitter"], "file": Path(row["path"]).name,
                                             "unreadable": repr(e), "screen_pass": False})
                save()
                print(f"{row['dataset']:12s} UNREADABLE {e!r}", flush=True)
                continue
            runs, scale = {}, SCALE
            if BFP:
                from rfpu_v15.cpu.bfp import AutoScale, BlockMax, shift_for
                bm = runner.run(BlockMax(), q, block=256)
                assert bm.exact_tokens and bm.exact_outputs
                m = bm.outputs[-1]
                au = runner.run(AutoScale(m), q, block=256)
                assert au.exact_tokens and au.exact_outputs
                q, s = au.outputs, shift_for(m)
                scale = 1 << (FRAC - s)
                meta.update(block_max=m, cpu_shift=s, block_exponent=FRAC - s,
                            q15_quantization=metrics(np.array([complex(*v) for v in q]) / scale, z))
            for name, kernel in (("fir", fir), ("matched", matched), ("spectral", spectral)):
                r = runner.run(kernel, q, block=256)
                assert r.exact_tokens and r.exact_outputs, (row["path"], name)
                runs[name] = r
            y = np.array([complex(*v) / scale for v in runs["fir"].outputs])
            fir_m = metrics(y, np.convolve(z, h)[:len(z)])
            mm = np.array([complex(*v) / scale for v in runs["matched"].outputs])
            mf = np.convolve(z, np.conj(h))[:len(z)]
            peak = abs(int(np.argmax(abs(mm) ** 2)) - int(np.argmax(abs(mf) ** 2)))
            ends = [i for i, v in enumerate(runs["spectral"].outputs) if v is not None]
            p_actual = np.array([[p / scale**2 for p in runs["spectral"].outputs[e]] for e in ends])
            frames = z[np.array(ends)[:, None] - 3 + np.arange(4)]
            p_float = abs(np.fft.fft(frames * np.array([0, .75, .75, 0]), axis=1)) ** 2
            spec_m = metrics(p_actual, p_float)
            passed = fir_m["nmse"] <= .01 and spec_m["nmse"] <= .01 and peak <= 1
            report["recordings"].append({
                "dataset": row["dataset"], "emitter": row["emitter"], "file": Path(row["path"]).name,
                "in_14_capture_screen": row.get("in_14_capture_screen"),
                "sha256": row["sha256"], "sample_rate": row.get("metadata", {}).get("sample_rate"),
                "center_frequency_hz": row.get("metadata", {}).get("center_frequency_hz"),
                "preprocessing": meta, "FIR": fir_m, "spectral_power": spec_m,
                "matched_peak_displacement_samples": peak, "screen_pass": bool(passed),
                "neural_samples_executed": len(q), "cpu_only_samples": 0,
                "execution": {k: {"tokens": r.tokens, "chunks": r.chunks, "neural_seconds": r.neural_seconds,
                                  "reference_trace_seconds": r.reference_seconds, **r.stats}
                              for k, r in runs.items()}})
            save()
            print(f"{row['dataset']:12s} {row['emitter']:16s} FIR {fir_m['nmse']:.2e} SPEC {spec_m['nmse']:.2e} "
                  f"PEAK {peak} {'PASS' if passed else 'FAIL'}", flush=True)
        recs = report["recordings"]
        report.update(status="completed", screen_pass_recordings=sum(r["screen_pass"] for r in recs),
                      recordings_total=len(recs), neural_samples_total=sum(r["neural_samples_executed"] for r in recs))
    except BaseException as e:
        report.update(status="failed", error=repr(e))
        raise
    finally:
        save()
    print(f"PASS {report['screen_pass_recordings']}/{report['recordings_total']}", flush=True)


if __name__ == "__main__":
    main()

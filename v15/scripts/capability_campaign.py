"""Neural verification and cost of every v1.5 kernel on QwenPU's CPU in weights (fused executor).

For each kernel: a synthetic stream (edge values + random full-scale) and a real-IQ stream (first
capture, Q15, no clipping) run free-running with every token checked against the RV32 reference and
outputs checked against the independent integer reference. Reports tokens/sample and ms/sample.
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
from rfpu_v15.cpu import kernels as K  # noqa: E402
from rfpu_v15.cpu.detect import EnergyDetector, PeakTracker  # noqa: E402
from rfpu_v15.cpu.filters import DCBlocker, DecimatingFIR, IIRBiquad, NCOMixer  # noqa: E402
from rfpu_v15.cpu.infer import SpectralMLP  # noqa: E402
from rfpu_v15.cpu.nonlinear import AMEnvelope, CFOEstimate, FMDemod, MagPhase  # noqa: E402
from rfpu_v15.cpu.pipeline import lowpass_taps  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402
from rfpu_v15.cpu.spectral import CFARDetect, DFTBin, FFTPower, WelchPSD  # noqa: E402

OUT = ROOT / "v15/reports/capability-campaign.json"


def kernels():
    rng = random.Random(1515)
    mlp = json.loads((ROOT / "v15/configs/spectral-mlp.json").read_text())
    taps = lambda n, a: [(rng.randrange(-a, a), rng.randrange(-a, a)) for _ in range(n)]  # noqa: E731
    return [
        ("filtering", K.FIR(taps(4, 8000), "fir4")), ("filtering", K.FIR(taps(16, 2000), "fir16")),
        ("filtering", K.FIR(taps(32, 1000), "fir32")), ("filtering", K.FIR(taps(64, 30000), "fir64_wide")),
        ("filtering", K.FIR(lowpass_taps(), "lowpass16")), ("filtering", DecimatingFIR(taps(8, 4000), 4)),
        ("filtering", IIRBiquad(4096, 8192, 4096, -16384, 4096)), ("filtering", DCBlocker(32440)),
        ("frequency", NCOMixer(0x12345678)),
        ("spectral", K.FFT4Power()), ("spectral", FFTPower(16, None, 16)), ("spectral", FFTPower(64, None, 32)),
        ("spectral", FFTPower(64, None, 64, output="complex")), ("spectral", FFTPower(256, None, 256)),
        ("spectral", WelchPSD(64, None, 32, 4)), ("spectral", DFTBin(64, 5, 16)),
        ("detection", CFARDetect(64, None, 32, 8, 2, 1024)), ("detection", EnergyDetector(64, 8, 1024)),
        ("detection", PeakTracker()),
        ("demodulation", MagPhase()), ("demodulation", AMEnvelope()), ("demodulation", FMDemod()),
        ("synchronization", CFOEstimate(1, 64)), ("synchronization", CFOEstimate(8, 256)),
        ("inference", SpectralMLP(mlp["w1"], mlp["b1"], mlp["w2"], mlp["b2"], mlp["fshift"])),
    ]


def main():
    from real_iq_q15 import inputs
    rng = random.Random(7)
    edge = [(-32768, -32768), (32767, 32767), (-32768, 32767), (0, 0), (1, -1)]
    synthetic = edge + [(rng.randrange(-32768, 32768), rng.randrange(-32768, 32768)) for _ in range(1024 - len(edge))]
    row = json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"][0]
    _, real, meta = inputs(row)
    real = real[:2048]
    runner = Runner(vx=False, batch=16)
    report = {"status": "running", "backend": "QwenPU RV32 CPU in weights, chipformer fusedexec.FreeRun3, batch 16",
              "real_capture": {"file": Path(row["path"]).name, "sha256": row["sha256"],
                               "clipped_components": meta["component_clipping_count"]}, "kernels": []}
    try:
        for group, k in kernels():
            block = min(256, k.describe().get("max_block", 256))
            block -= block % k.hop
            entry = {"group": group, **k.describe(), "block": block}
            for label, x in (("synthetic", synthetic), ("real_iq", real)):
                t = time.perf_counter()
                r = runner.run(k, x, block=block)
                assert r.exact_tokens and r.exact_outputs, (k.name, label)
                entry[label] = {"samples": len(x), "tokens": r.tokens, "tokens_per_sample": r.tokens / len(x),
                                "neural_seconds": r.neural_seconds, "ms_per_sample": 1000 * r.neural_seconds / len(x),
                                "reference_trace_seconds": r.reference_seconds, "wall_seconds": time.perf_counter() - t,
                                "exact_tokens": r.exact_tokens, "outputs_equal_reference": r.exact_outputs}
            report["kernels"].append(entry)
            OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
            s = entry["real_iq"]
            print(f"{group:15s} {k.name:22s} EXACT {s['tokens_per_sample']:7.1f} tok/sample {s['ms_per_sample']:7.2f} ms/sample", flush=True)
        report["status"] = "passed"
        report["kernels_verified"] = len(report["kernels"])
    except BaseException as e:
        report.update(status="failed", error=repr(e))
        raise
    finally:
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

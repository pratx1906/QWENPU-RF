"""Final Stage 1 check on the actual published QwenPU checkpoint (Qwen3.5-4B with the compiled CPU).

Each representative v1.5 kernel runs on a real-IQ chunk through the checkpoint's own saved chip
layers (stock HF Qwen3_5DecoderLayer modules, free-running, every token checked), and its outputs
must equal the kernel's independent integer reference. Chip-only: Qwen's language layers are not
loaded here (Stage 2). Short chunks keep the slow HF path (~20 ms/token) tractable.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
from rfpu_v15.cpu import kernels as K  # noqa: E402
from rfpu_v15.cpu.detect import EnergyDetector, PeakTracker  # noqa: E402
from rfpu_v15.cpu.filters import IIRBiquad, NCOMixer  # noqa: E402
from rfpu_v15.cpu.infer import SpectralMLP  # noqa: E402
from rfpu_v15.cpu.nonlinear import CFOEstimate, MagPhase  # noqa: E402
from rfpu_v15.cpu.pipeline import lowpass_taps  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402

OUT = ROOT / "v15/reports/qwen-checkpoint-final.json"


def main():
    from real_iq_q15 import inputs
    from rfpu.backends.qwenpu_checkpoint.published import PublishedCheckpoint
    mlp = json.loads((ROOT / "v15/configs/spectral-mlp.json").read_text())
    row = json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"][0]
    _, q, meta = inputs(row)
    cases = [
        ("gate_fir", K.FIR([(16384, 0), (8192, 0)], "gate_fir"), 16),
        ("fft4_power", K.FFT4Power(), 8),
        ("lowpass16", K.FIR(lowpass_taps(), "lowpass16"), 4),
        ("iir_biquad", IIRBiquad(4096, 8192, 4096, -16384, 4096), 8),
        ("nco_mixer", NCOMixer(0x10000000), 8),
        ("energy_detector", EnergyDetector(window=8, memory=4, alpha_q8=1024), 16),
        ("peak_tracker", PeakTracker(), 16),
        ("cfo_estimate", CFOEstimate(lag=1, window=8), 8),
        ("spectral_mlp", SpectralMLP(mlp["w1"], mlp["b1"], mlp["w2"], mlp["b2"], mlp["fshift"]), 8),
        ("mag_phase_cordic", MagPhase(), 2),
    ]
    import random
    from rfpu_v15.cpu.spectral import CFARDetect, FFTPower
    from vx_neural import vx_mlp
    from rfpu_v15.cpu.vx_kernels import VXEnergy
    cases += [("fft16_power", FFTPower(16, [32767] * 16, 16), 16),
              ("cfar16", CFARDetect(16, [32767] * 16, 16, 2, 1, 1024), 16),
              ("vx_energy_npu", VXEnergy(8, 7), 16),
              ("vx_spectral_mlp_npu", vx_mlp(random.Random(1555)), 8)]
    runner = Runner(vx=True, batch=1)
    report = {"status": "running", "backend": "D.published_checkpoint_chip_only (QwenPU-published, HF layers)",
              "capture": {"file": Path(row["path"]).name, "sha256": row["sha256"], **{k: meta[k] for k in ("source_sample_start", "gain")}},
              "cases": []}

    def save():
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save()
    ckpt = PublishedCheckpoint()
    report["loading_seconds"] = ckpt.loading_seconds
    try:
        for name, kernel, block in cases:
            built, prog = runner.program(kernel, block)
            ctx, n = runner.context(kernel, q, 0, block)
            words = kernel.init_words(built, ctx, 0)
            if kernel.state_words:
                kernel.put_state(built, words, kernel.initial_state())
            expected = runner.machine(prog, words)
            expected.run(10**7)
            t = time.perf_counter()
            res, _ = ckpt.check_program(prog, lambda: runner.machine(prog, words), max_tokens=10**6)
            seconds = time.perf_counter() - t
            outputs = kernel.decode(built, expected.ram, 0)[:n]
            ref = kernel.reference(q[:block])[:n]
            ok = res["exact_trace_match"] and outputs == ref
            report["cases"].append({"kernel": name, **kernel.describe(), "samples": block, "trace_tokens": res["trace_tokens"],
                                    "mode": res["mode"], "exact_trace": res["exact_trace_match"], "outputs_equal_reference": outputs == ref,
                                    "seconds": seconds, "ms_per_token": 1000 * seconds / max(res["trace_tokens"], 1)})
            save()
            print(f"QWEN CHECKPOINT {name}: {res['trace_tokens']} tokens {'EXACT' if ok else 'FAIL'} {seconds:.1f}s", flush=True)
            assert ok, name
        report["status"] = "passed"
    except BaseException as e:
        report.update(status="failed", error=repr(e))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

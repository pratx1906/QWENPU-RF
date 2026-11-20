"""VX (QwenPU's in-weights vector unit) kernels vs scalar equivalents, neural, every token checked."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
from rfpu_v15.cpu.infer import SpectralMLP  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402
from rfpu_v15.cpu.vx_kernels import ScalarEnergy, VXEnergy, VXSpectralMLP  # noqa: E402

OUT = ROOT / "v15/reports/vx-neural.json"


def vx_mlp(rng, H=8, C=8):
    i4 = lambda n: [rng.randrange(-8, 8) for _ in range(n)]  # noqa: E731
    return VXSpectralMLP([i4(4) for _ in range(H)], [rng.randrange(1, 1 << 16) for _ in range(H)], i4(H),
                         [i4(H) for _ in range(C)], [rng.randrange(1, 1 << 16) for _ in range(C)], i4(C),
                         fshift=24, sh1=8, r1=16384, bias_act=64)


def main():
    from real_iq_q15 import inputs
    rng = random.Random(1555)
    row = json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"][0]
    real = inputs(row)[1][:2048]
    scalar_mlp = SpectralMLP([[rng.randrange(-32768, 32768) for _ in range(4)] for _ in range(8)],
                             [rng.randrange(-32768, 32768) for _ in range(8)],
                             [[rng.randrange(-32768, 32768) for _ in range(8)] for _ in range(8)],
                             [rng.randrange(-32768, 32768) for _ in range(8)], fshift=16)
    pairs = [("energy_W64", VXEnergy(64, 7), ScalarEnergy(64)), ("mlp_H8_C8", vx_mlp(rng), scalar_mlp)]
    runner = Runner(vx=True, batch=16)
    report = {"backend": "QwenPU RV32+VX CPU in weights (vx build), fusedexec.FreeRun3, batch 16",
              "capture": Path(row["path"]).name, "samples": len(real), "pairs": []}
    for name, vx, scalar in pairs:
        entry = {"workload": name}
        for label, k in (("vx", vx), ("scalar", scalar)):
            r = runner.run(k, real, block=256)
            assert r.exact_tokens and r.exact_outputs, (name, label)
            entry[label] = {"kernel": k.describe(), "tokens_per_sample": r.tokens / len(real),
                            "ms_per_sample": 1000 * r.neural_seconds / len(real), "exact": True}
        entry["token_ratio_scalar_over_vx"] = entry["scalar"]["tokens_per_sample"] / entry["vx"]["tokens_per_sample"]
        report["pairs"].append(entry)
        print(name, json.dumps(entry), flush=True)
    report["status"] = "passed"
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

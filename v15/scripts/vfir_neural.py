"""VFIR (new VX instruction) vs scalar RV32 complex FIR on real IQ, neural, every token checked.
Run with the rfpu-vfir branch first: PYTHONPATH=upstream/QwenPU-rf python v15/scripts/vfir_neural.py
Both kernels run on the same branch circuit (core.build(vx=True)) and the fused executor, batch 16."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
import chipformer  # noqa: E402
from rfpu_v15.cpu import kernels as K  # noqa: E402
from rfpu_v15.cpu.runner import Runner  # noqa: E402
from rfpu_v15.cpu.vfir import VXFir  # noqa: E402


def main():
    assert "QwenPU-rf" in chipformer.__file__
    from real_iq_q15 import inputs
    rows = json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"]
    streams = [inputs(r)[1][:512] for r in rows[:16]]            # 14 captures x 512 samples
    rng = random.Random(4)
    runner = Runner(vx=True, batch=16)
    out = {"chipformer": chipformer.__file__, "samples": sum(map(len, streams)), "rows": []}
    for n in (4, 8, 16):
        taps = [(rng.randrange(-6000, 6000), rng.randrange(-6000, 6000)) for _ in range(n)]
        row = {"taps": n}
        for label, k in (("scalar_rv32", K.FIR(taps, f"fir{n}")), ("vfir", VXFir(taps))):
            res = runner.run_many(k, streams, block=256)
            assert all(r.exact_tokens and r.exact_outputs for r in res), (label, n)
            total = sum(map(len, streams))
            row[label] = {"tokens_per_sample": res[0].tokens / total, "ms_per_sample": 1000 * res[0].neural_seconds / total,
                          "exact": True, "kernel": k.describe()}
        row["token_gain"] = row["scalar_rv32"]["tokens_per_sample"] / row["vfir"]["tokens_per_sample"]
        row["time_gain"] = row["scalar_rv32"]["ms_per_sample"] / row["vfir"]["ms_per_sample"]
        out["rows"].append(row)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items() if k in ("taps", "token_gain", "time_gain")}),
              {l: round(row[l]["ms_per_sample"], 2) for l in ("scalar_rv32", "vfir")}, flush=True)
    (ROOT / "v15/reports/vfir-neural.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

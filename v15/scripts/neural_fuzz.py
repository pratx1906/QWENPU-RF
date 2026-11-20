"""Neural fuzzing: random kernel configurations and awkward streams on QwenPU's CPU in weights.

Each case: a random kernel (any of the 17 families, random runtime parameters), random block, and
several streams of random lengths (1, < history, non-multiples of the block) with full-scale,
corner-value and tiny-value content. Every token must equal the RV32 reference and every output
the independent integer reference. Report: v15/reports/neural-fuzz.json.
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/tests")]
from rfpu_v15.cpu.runner import Runner  # noqa: E402
from test_robust import random_kernel, stream  # noqa: E402

CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def main():
    rng = random.Random(15151)
    runner = Runner(vx=False, batch=16)
    rows, t0 = [], time.perf_counter()
    for case in range(CASES):
        k = random_kernel(rng)
        block = k.hop * rng.choice([2, 4, 8]) * (1 if k.hop > 4 else 4)
        block = min(block, k.describe().get("max_block", 256)); block -= block % k.hop
        lengths = [1, max(1, k.history), block - 1 if block > 1 else 1, block + 1, 3 * block + rng.randrange(1, block)]
        xs = [stream(rng, n) for n in lengths]
        res = runner.run_many(k, xs, block=block)
        ok = all(r.exact_tokens and r.exact_outputs for r in res)
        rows.append({"case": case, "kernel": k.describe(), "block": block, "stream_lengths": lengths,
                     "tokens": res[0].tokens, "exact": ok})
        print(f"case {case:2d} {k.name:22s} block {block:4d} lengths {lengths} tokens {res[0].tokens:8d} {'EXACT' if ok else 'FAIL'}", flush=True)
        assert ok, case
    out = {"cases": len(rows), "all_exact": all(r["exact"] for r in rows), "tokens": sum(r["tokens"] for r in rows),
           "seconds": time.perf_counter() - t0, "families": sorted({r["kernel"]["kernel"].rstrip("0123456789_") for r in rows}),
           "rows": rows}
    (ROOT / "v15/reports" / (sys.argv[2] if len(sys.argv) > 2 else "neural-fuzz.json")).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

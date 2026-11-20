"""Amendment 3 timing study: one process = one experimental unit. Usage (driver):
  PYTHONPATH=upstream/QwenPU-rf python v15/scripts/timing_study.py --processes 8
The driver launches each unit as a fresh subprocess (--unit k) and analyses the results."""
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
OUT = ROOT / "v15/reports/timing-study.json"


def setup():
    from real_iq_q15 import inputs
    from rfpu_v15.cpu import kernels as K
    from rfpu_v15.cpu.vfir import VXFir
    rows = json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"][:4]
    streams = [inputs(r)[1][:256] for r in rows]
    rng = random.Random(4)
    taps = [(rng.randrange(-6000, 6000), rng.randrange(-6000, 6000)) for _ in range(16)]
    digest = hashlib.sha256(json.dumps([streams, taps]).encode()).hexdigest()
    return streams, {"A_scalar": K.FIR(taps, "fir16"), "B_vfir": VXFir(taps)}, digest


def unit(k):
    import chipformer
    from rfpu_v15.cpu.runner import Runner
    assert "QwenPU-rf" in chipformer.__file__
    streams, arms, digest = setup()
    runner = Runner(vx=True, batch=16)
    n = sum(map(len, streams))
    times, tokens = {a: [] for a in arms}, {}
    for a in arms:                                   # warm-up, excluded
        runner.run_many(arms[a], streams, block=256)
    for a in ("A_scalar", "B_vfir", "B_vfir", "A_scalar"):
        res = runner.run_many(arms[a], streams, block=256)
        assert all(r.exact_tokens and r.exact_outputs for r in res), a
        times[a].append(res[0].neural_seconds / n)
        tokens[a] = res[0].tokens / n
    print(json.dumps({"unit": k, "input_sha256": digest, "seconds_per_sample": {a: sum(v) / len(v) for a, v in times.items()},
                      "runs": times, "tokens_per_sample": tokens}), flush=True)


def driver(p):
    py = sys.executable
    units = []
    for k in range(p):
        t = time.perf_counter()
        out = subprocess.run([py, __file__, "--unit", str(k)], capture_output=True, text=True, cwd=ROOT)
        line = [l for l in out.stdout.splitlines() if l.startswith('{"unit"')]
        if out.returncode or not line:
            raise RuntimeError(out.stderr[-2000:])
        units.append(json.loads(line[0])); units[-1]["wall_seconds"] = time.perf_counter() - t
        print(f"unit {k}: A {units[-1]['seconds_per_sample']['A_scalar']*1e3:.2f} ms  B {units[-1]['seconds_per_sample']['B_vfir']*1e3:.2f} ms", flush=True)
    logs = [math.log(u["seconds_per_sample"]["A_scalar"] / u["seconds_per_sample"]["B_vfir"]) for u in units]
    m = sum(logs) / len(logs); sd = math.sqrt(sum((x - m) ** 2 for x in logs) / (len(logs) - 1))
    from scipy.stats import t as tdist
    h = tdist.ppf(.975, len(logs) - 1) * sd / math.sqrt(len(logs))
    # exact two-sided sign-flip test on the mean log ratio
    import itertools
    ge = sum(1 for s in itertools.product((1, -1), repeat=len(logs)) if abs(sum(a * b for a, b in zip(s, logs))) >= abs(sum(logs)) - 1e-15)
    out = {"protocol": "PROTOCOL.md Amendment 3", "processes": len(units), "units": units,
           "speedup_geometric_mean": math.exp(m), "speedup_95ci_t": [math.exp(m - h), math.exp(m + h)],
           "exact_sign_flip_p_two_sided": ge / 2 ** len(logs), "log_ratio_sd": sd,
           "tokens_per_sample": units[0]["tokens_per_sample"]}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "units"}, indent=2))


if __name__ == "__main__":
    if "--unit" in sys.argv:
        unit(int(sys.argv[sys.argv.index("--unit") + 1]))
    else:
        driver(int(sys.argv[sys.argv.index("--processes") + 1]) if "--processes" in sys.argv else 8)

"""VPOW on the fused GPU executor, free-running: the branch's random VPOW test programs (vpow_program)
run batched on core.build(vx=True); every token must equal the RV32 reference machine trace.
Needs PYTHONPATH=upstream/QwenPU-rf."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "v15/reports/vpow-executor.json"


def main():
    import chipformer
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(chipformer.__file__).resolve().parents[1] / "tests"))
    from chipformer.rv import core, fusedexec, ref
    from chipformer.rv import spec as S
    from test_rv_vx import vpow_program
    if "QwenPU-rf" not in chipformer.__file__:
        raise SystemExit("set PYTHONPATH=upstream/QwenPU-rf")
    seeds = list(range(8))
    progs = [vpow_program(s) for s in seeds]
    c = core.build(vx=True)
    traces = []
    for p in progs:
        m = ref.Machine.from_program(p)
        traces.append(m.run(10**7))
        assert m.halted and m.fault is None, (m.fault, m.exit_code)
    fr = fusedexec.FreeRun3(c, len(progs), "cuda")
    for row, p in enumerate(progs):
        core.snapshot(c, ref.Machine.from_program(p))
        fr.load(row, S.tok("GO"))
    fr.capture()
    n = max(len(t) for t in traces) - 1
    torch.cuda.synchronize(); t0 = time.perf_counter()
    got = np.asarray(fr.run(n))
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    rows = []
    for s, tr, g in zip(seeds, traces, got):
        m_ = len(tr) - 1
        ok = g[:m_].tolist() == tr[1:]
        vpow_calls = sum(1 for x in tr[1:] if S.family(x)[0] == "VC")
        rows.append({"seed": s, "tokens": m_, "exact": ok, "vc_tokens": vpow_calls})
        print(rows[-1], flush=True)
    rep = {"chipformer": chipformer.__file__, "programs": len(progs), "rows": rows,
           "all_exact": all(r["exact"] for r in rows), "seconds": dt}
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if not rep["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

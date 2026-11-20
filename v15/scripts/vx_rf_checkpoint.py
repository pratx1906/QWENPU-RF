"""VFIR, VPOW and VBFLY on the saved layers of a merged checkpoint built from the rfpu-vfir branch
(chipformer.rv.merge --vx at commit b72623e: Qwen3.5-4B + VX + the three RF instructions).

  PYTHONPATH=upstream/QwenPU-rf python v15/scripts/vx_rf_checkpoint.py

Writes the checkpoint's rfpu-provenance.json (shard SHA256), loads the embedding and chip layers onto
the GPU (PublishedCheckpoint: verifies every shard hash and that the branch's circuit matches the
checkpoint metadata) and runs, free-running through the HF layers with every token checked against the
reference machine: the branch's VPOW and VBFLY circuit-test programs, VXFir on 16 real IQ samples and
VXFFT64 (VBFLY block floating point) on one 64-sample frame of real IQ (outputs vs the independent model).
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v15")); from rfpu_v15.paths import DATA  # noqa: E402  (RFPU_DATA, see paths.py)
CK = DATA / "checkpoints/rfpu-v15/qwen3.5-4b-vx-rf"
OUT = ROOT / "v15/reports/vx-rf-checkpoint.json"


def main():
    sys.path[:0] = [str(ROOT), str(ROOT / "upstream/QwenPU-rf/tests"), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
    import chipformer
    assert "QwenPU-rf" in chipformer.__file__, chipformer.__file__
    import vfir_checkpoint as VC
    VC.CK = CK
    pre = VC.provenance()
    (CK / "rfpu-provenance.json").write_text((CK / "rfpu-provenance.json").read_text().replace(
        "branch rfpu-vfir over", "branch rfpu-vfir (b72623e: VFIR, VPOW, VBFLY) over"))
    from rfpu.backends.qwenpu_checkpoint.published import PublishedCheckpoint
    import test_rv_vx
    test_rv_vx.EXIT = "\nli a0, 0\nli a7, 93\necall"
    from test_rv_vx import vbfly_program, vpow_program
    from chipformer.rv import ref
    from chipformer.rv.asm import assemble
    from rfpu_v15.cpu.vfft import VXFFT64
    from rfpu_v15.cpu.vfir import VXFir
    from real_iq_q15 import inputs
    q = inputs(json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"][0])[1]

    def kernel_case(kern, samples, block):
        built = kern.build(block)
        prog = assemble(built.source)
        ctx = [(0, 0)] * kern.history + samples
        words = kern.init_words(built, ctx, 0)

        def machine():
            m = ref.Machine.from_program(prog)
            for adr, v in words.items():
                m.ram[adr >> 2] = v
            return m
        m = machine(); m.run(10**7)
        ok = kern.decode(built, m.ram, 0)[:len(samples)] == kern.reference(samples)
        return prog, machine, ok

    ck = PublishedCheckpoint(path=str(CK))
    rows = []
    rng = random.Random(4)
    cases = []
    p, mk, ok = kernel_case(VXFFT64(), q[:64], 64)
    cases.append(("VXFFT64 (VBFLY block floating point), one 64-sample real IQ frame", p, mk, ok))
    p, mk, ok = kernel_case(VXFir([(rng.randrange(-6000, 6000), rng.randrange(-6000, 6000)) for _ in range(16)]), q[:16], 16)
    cases.append(("VXFir 16 taps, 16 real IQ samples", p, mk, ok))
    for seed in (0, 1):
        prog = vbfly_program(seed)
        cases.append((f"vbfly_program seed {seed}", prog, (lambda prog=prog: ref.Machine.from_program(prog)), None))
    prog = vpow_program(0)
    cases.append(("vpow_program seed 0", prog, (lambda: ref.Machine.from_program(prog)), None))
    for name, prog, mk, ok in cases:
        t = time.perf_counter()
        res, _ = ck.check_program(prog, mk, max_tokens=10**6)
        rows.append({"program": name, "trace_tokens": res["trace_tokens"], "exact_trace": res["exact_trace_match"],
                     "outputs_equal_reference": ok, "mode": res.get("mode"), "seconds": time.perf_counter() - t})
        print(rows[-1], flush=True)
        OUT.write_text(json.dumps({"checkpoint": str(CK), "programs": rows}, indent=2), encoding="utf-8")
    out = {"checkpoint": str(CK), "chipformer": chipformer.__file__, "preflight": pre,
           "loading_seconds": ck.loading_seconds, "programs": rows,
           "all_exact": all(r["exact_trace"] and r["outputs_equal_reference"] is not False for r in rows)}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not out["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

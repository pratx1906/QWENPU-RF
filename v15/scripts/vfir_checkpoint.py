"""VFIR on the saved layers of a merged checkpoint built from the rfpu-vfir QwenPU branch.

Run with the branch first on the path:
  PYTHONPATH=upstream/QwenPU-rf python v15/scripts/vfir_checkpoint.py

Writes rfpu-provenance.json for the checkpoint (shard SHA256 and chip-only memory figures computed
from the safetensors headers), then loads only the embedding and the 13 chip layers onto the GPU
(rfpu.backends.qwenpu_checkpoint.published, which verifies every shard hash and checks that the
circuit built from the branch matches the checkpoint metadata), and runs the branch's VFIR test
programs free-running through the HF layers, every token checked against the reference machine.
"""
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v15")); from rfpu_v15.paths import DATA  # noqa: E402  (RFPU_DATA, see paths.py)
CK = DATA / "checkpoints/rfpu-v15/qwen3.5-4b-vx-vfir"
BYTES = {"BF16": 2, "F32": 4, "F16": 2, "I64": 8}


def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def provenance():
    meta = json.loads((CK / "chipformer.json").read_text())
    n_chip = meta["chip_layers"]
    files, chip, emb = [], 0, 0
    for p in sorted(CK.glob("model-*.safetensors")):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        files.append({"name": p.name, "sha256": h.hexdigest()})
        for k, v in header(p).items():
            if k == "__metadata__":
                continue
            size = BYTES[v["dtype"]] * int(__import__("math").prod(v["shape"]))
            if k == "model.embed_tokens.weight":
                emb += size
            elif k.startswith("model.layers.") and int(k.split(".")[2]) < n_chip:
                chip += size
    for name in ("config.json", "chipformer.json", "tokenizer.json"):
        files.append({"name": name, "sha256": hashlib.sha256((CK / name).read_bytes()).hexdigest()})
    pre = {"chip_only_admitted": True, "chip_tensor_bytes": chip, "embedding_bytes": emb,
           "cache_estimate_bytes": 2 << 30, "temporary_allowance_bytes": 2 << 30,
           "chip_only_peak_estimate_bytes": chip + emb + (4 << 30)}
    (CK / "rfpu-provenance.json").write_text(json.dumps(
        {"source": "chipformer.rv.merge --vx from branch rfpu-vfir over Qwen/Qwen3.5-4B@851bf6e",
         "gpu_memory_preflight": pre, "verified_files": files}, indent=2))
    return pre


def main():
    sys.path[:0] = [str(ROOT), str(ROOT / "upstream/QwenPU-rf/tests")]
    import chipformer
    assert "QwenPU-rf" in chipformer.__file__, chipformer.__file__
    pre = provenance()
    from rfpu.backends.qwenpu_checkpoint.published import PublishedCheckpoint
    import test_rv_vx
    from test_rv_vx import vfir_program
    from chipformer.rv import ref
    from chipformer.rv.asm import assemble
    test_rv_vx.EXIT = "\nli a0, 0\nli a7, 93\necall"   # exit code 0 (the loader's contract); a0 only
    sys.path[:0] = [str(ROOT / "v15"), str(ROOT / "v15/scripts")]
    from rfpu_v15.cpu.vfir import VXFir
    from real_iq_q15 import inputs
    import random
    rng = random.Random(4)
    fir = VXFir([(rng.randrange(-6000, 6000), rng.randrange(-6000, 6000)) for _ in range(16)])
    q = inputs(json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"][0])[1][:16]
    built = fir.build(16)
    fir_prog = assemble(built.source)
    ctx = [(0, 0)] * fir.history + q
    words = fir.init_words(built, ctx, 0)

    def fir_machine():
        m = ref.Machine.from_program(fir_prog)
        for adr, v in words.items():
            m.ram[adr >> 2] = v
        return m
    ck = PublishedCheckpoint(path=str(CK))
    rows = []
    m = fir_machine(); m.run(10**6)
    fir_ok = fir.decode(built, m.ram, 0) == fir.reference(q)
    t = time.perf_counter()
    res, _ = ck.check_program(fir_prog, fir_machine, max_tokens=10**6)
    rows.append({"program": "VXFir 16 taps, 16 real IQ samples", "trace_tokens": res["trace_tokens"],
                 "exact_trace": res["exact_trace_match"], "outputs_equal_reference": fir_ok, "seconds": time.perf_counter() - t})
    print(rows[-1], flush=True)
    for seed in range(4):
        prog = vfir_program(seed)
        t = time.perf_counter()
        res, _ = ck.check_program(prog, lambda: ref.Machine.from_program(prog), max_tokens=10**6)
        rows.append({"program": f"vfir_program seed {seed}", "trace_tokens": res["trace_tokens"], "exact_trace": res["exact_trace_match"],
                     "mode": res["mode"], "seconds": time.perf_counter() - t})
        print(rows[-1], flush=True)
    out = {"checkpoint": str(CK), "chipformer": chipformer.__file__, "preflight": pre,
           "loading_seconds": ck.loading_seconds, "programs": rows, "all_exact": all(r["exact_trace"] for r in rows)}
    (ROOT / "v15/reports/vfir-checkpoint.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

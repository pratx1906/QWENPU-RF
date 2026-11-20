"""Speculative (parallel) verification of computer tokens on the saved checkpoint layers.

The RV32 reference machine drafts the whole trace; the published QwenPU checkpoint's chip layers
(stock HF modules) then process the drafted tokens C at a time in one forward pass (as a prompt is
read) and every position's prediction must equal the next drafted token. If all match, free-running
decoding would produce the identical trace (induction over positions), so verification costs one
parallel pass per chunk instead of one sequential step per token. Whether the chunked (parallel)
Gated-DeltaNet kernels stay bit-exact is the empirical question tested here; free-running
per-token decoding is run on the same program as the baseline.

This is an audit / verification mode: the host also computes the trace (the draft).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
OUT = ROOT / "v15/reports/speculative-verify.json"


def args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--taps", type=int, default=16, help="16: lowpass16; otherwise a seeded random FIR")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--chunks", default="64,256,1024")
    p.add_argument("--no-free-running", action="store_true", help="skip the slow free-running baseline")
    p.add_argument("--executor", action="store_true", help="also time the program on the fused executor")
    p.add_argument("--batched-decode", action="store_true", help="decode each chunk with one device-to-host copy")
    p.add_argument("--gpu-compare", action="store_true",
                   help="compare each position's output code with the drafted token's code on the GPU (same rule)")
    p.add_argument("--corrupt", type=int, default=None, help="mutation test: alter the drafted token at this position")
    p.add_argument("--out", default=str(OUT))
    return p.parse_args()


def main():
    opt = args()
    out = Path(opt.out)
    import torch
    from transformers import DynamicCache
    from chipformer.rv import core, hf, ref
    from chipformer.rv import spec as S_
    from chipformer.rv.asm import assemble
    from rfpu.backends.qwenpu_checkpoint.published import PublishedCheckpoint
    from rfpu_v15.cpu import kernels as K
    from rfpu_v15.cpu.pipeline import lowpass_taps
    from real_iq_q15 import inputs
    q = inputs(json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"][0])[1]
    if opt.taps == 16:
        kern = K.FIR(lowpass_taps(), "lowpass16")
    else:
        import random
        rng = random.Random(4)
        kern = K.FIR([(rng.randint(-5999, 5999), rng.randint(-5999, 5999)) for _ in range(opt.taps)], f"fir{opt.taps}")
    built = kern.build(opt.samples)
    prog = assemble(built.source)
    words = kern.init_words(built, [(0, 0)] * kern.history + q[:opt.samples], 0)

    def machine():
        m = ref.Machine.from_program(prog)
        for a, v in words.items():
            m.ram[a >> 2] = v
        return m
    m = machine(); want = m.run(10**8)
    if opt.corrupt is not None:
        nm, pl = S_.family(want[opt.corrupt])
        want[opt.corrupt] = S_.tok(nm, pl ^ 1) if nm not in S_.CONTROLS else S_.tok("NT")
    import importlib.util
    report = {"program": f"{kern.name} FIR, {opt.samples} real IQ samples (scalar RV32)", "trace_tokens": len(want) - 1,
              "flash_linear_attention": importlib.util.find_spec("fla") is not None, "corrupted_position": opt.corrupt,
              "rows": []}
    print(report, flush=True)
    if opt.executor:
        from chipformer.rv import fusedexec
        from chipformer.rv import spec as S
        import numpy as np
        c = core.build(vx=True)
        fr = fusedexec.FreeRun3(c, 1, "cuda")
        core.snapshot(c, machine()); fr.load(0, S.tok("GO")); fr.capture()
        torch.cuda.synchronize(); t = time.perf_counter()
        got = np.asarray(fr.run(len(want) - 1))
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        report["executor_batch1"] = {"exact": got[0].tolist() == want[1:], "seconds": dt, "tokens_per_second": (len(want) - 1) / dt}
        print("executor", report["executor_batch1"], flush=True)
        del fr, c
        torch.cuda.empty_cache()
    ck = PublishedCheckpoint()
    if not opt.no_free_running:
        # baseline: free-running, one token per forward (v1.0 check_program)
        t = time.perf_counter()
        res, _ = ck.check_program(prog, machine, max_tokens=10**8)
        report["free_running"] = {"exact": res["exact_trace_match"], "seconds": time.perf_counter() - t,
                                  "tokens_per_second": res["trace_tokens"] / (time.perf_counter() - t)}
        print("free-running", report["free_running"], flush=True)
    prompt = ck.tokenizer.apply_chat_template([{"role": "user", "content": "Hi"}], add_generation_prompt=True,
                                              enable_thinking=False, tokenize=True)
    if not isinstance(prompt, list):
        prompt = prompt["input_ids"]
    nf = ck.meta["nfam"]

    def decode_chunk(hc):
        """ck.decode for every position of a chunk with one device-to-host copy (same rule, same result)."""
        codes = hc[:, ck.code_dims].float()
        owned = (codes != 0).any(1).cpu().tolist()
        bits = (codes > 0).cpu().numpy()
        out = []
        for row, ok in zip(bits, owned):
            if not ok:
                raise RuntimeError("checkpoint did not own this output token")
            fid = sum(1 << i for i in range(nf) if row[i])
            pay = sum(1 << (i - nf) for i in range(nf, len(row)) if row[i])
            token = core.decode_code(fid, pay, ck.text)
            if token < 0:
                raise RuntimeError("checkpoint emitted invalid output code")
            out.append(token - S_.TEXT_TOKEN if S_.is_text(token) else ck.offset + token)
        return out

    report["batched_decode"] = opt.batched_decode
    report["gpu_compare"] = opt.gpu_compare
    if opt.gpu_compare:
        # Expected code of every drafted token: family id bits, payload bits (masked for controls, whose
        # payload decode ignores). A position matches iff it is owned (some code dim nonzero) and its sign
        # bits equal the expected code on the mask, which is exactly ck.decode(...) == the drafted token.
        import numpy as np
        D = len(ck.meta["code_dims"])
        E = np.zeros((len(want), D), bool)
        Mk = np.ones((len(want), D), bool)
        for t_i, tk in enumerate(want):
            nm, pl = S_.family(tk)
            fid = core.OUT_NAMES.index(nm)
            E[t_i, :nf] = [(fid >> b) & 1 for b in range(nf)]
            if nm in S_.CONTROLS:
                Mk[t_i, nf:] = False
            else:
                E[t_i, nf:] = [(pl >> b) & 1 for b in range(D - nf)]
        E, Mk = torch.tensor(E, device="cuda"), torch.tensor(Mk, device="cuda")
    for C in map(int, opt.chunks.split(",")):
        with torch.no_grad():
            cache = DynamicCache(config=ck.cfg)
            h = ck.embedding(torch.tensor([prompt], device="cuda"))
            for layer in ck.layers:
                h = layer(h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=cache)
            core.snapshot(ck.c, machine()); hf.inject(ck.c, cache)
            torch.cuda.synchronize(); t = time.perf_counter()
            ids = [ck.offset + x for x in want]
            mism, first = 0, None
            for i in range(0, len(ids) - 1, C):
                chunk = ids[i:min(i + C, len(ids) - 1)]
                h = ck.embedding(torch.tensor([chunk], device="cuda"))
                for layer in ck.layers:
                    h = layer(h, position_embeddings=None, attention_mask=None, position_ids=None, past_key_values=cache)
                if opt.gpu_compare:
                    codes = h[0][:, ck.code_dims]
                    lo, hi = i + 1, i + 1 + len(chunk)
                    bad = ((codes > 0) != E[lo:hi]) & Mk[lo:hi]
                    bad = bad.any(1) | ~(codes != 0).any(1)
                    nb = int(bad.sum())
                    if nb:
                        mism += nb
                        first = first if first is not None else i + int(bad.nonzero()[0, 0])
                    continue
                if opt.batched_decode:
                    got = decode_chunk(h[0])
                else:
                    got = [ck.decode(h[0, j]) for j in range(len(chunk))]
                for j, g in enumerate(got):
                    if g != ids[i + j + 1]:
                        mism += 1
                        first = first if first is not None else i + j
            torch.cuda.synchronize(); dt = time.perf_counter() - t
        row = {"chunk": C, "mismatches": mism, "first_mismatch": first, "seconds": dt, "tokens_per_second": (len(ids) - 1) / dt}
        if "free_running" in report:
            row["speedup_vs_free_running"] = report["free_running"]["seconds"] / dt
        if "executor_batch1" in report:
            row["speedup_vs_executor_batch1"] = report["executor_batch1"]["seconds"] / dt
        report["rows"].append(row)
        print(row, flush=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

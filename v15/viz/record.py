"""SPIKE (throwaway): record a real Stage 2 run's internal activations for the visualizer.

Runs one question on the 19 GB merge (rv1, every layer on the GPU) with ChunkedRvChat, the reference
lock on, exactly as stage2_chat.py --full-gpu --chunk 1024 does, except that the CUDA graph for the
single-token chip step is disabled so PyTorch forward hooks fire on every forward. Hooks record, for
every layer the token passes through:
  * MLP neuron activations (the input of down_proj = silu(gate)*up; the chip layers: their used units)
  * per value head output norm (input of out_proj / o_proj, split per head)
  * residual stream norm after the layer
and, after the chip layers, the circuit's predicted next token at every position (verification).
Names: every chip MLP unit and head gets its compiled IR name (emit.emit_layers order).

  python v15/viz/record.py [--q 0] [--out $RFPU_DATA/viz/levels-q0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]

from rfpu_v15.paths import DATA  # noqa: E402
RV1 = str(DATA / "checkpoints/rfpu-v15/qwen3.5-4b-rv1")


def circuit_names(chat):
    """Per chip layer: MLP unit names (emit order), value-head names, and IR node edges."""
    from chipformer.q35.emit import allocate, reads
    from chipformer.q35.ir import FFN, Head
    c = chat.c
    inter = chat.model.config.intermediate_size
    keep = set(c.keep)
    alloc = allocate(c, None, keep, inter)
    exit_level = max(n.level for n in c.nodes if n.name.startswith("exit."))
    live = alloc.final
    exit_clear = [d for d in live if d not in keep]
    dn = lambda d: c.dims[d].name
    layers = []
    for L in range(c.n_layers()):
        units = []
        for n in c.nodes:
            if not (isinstance(n, FFN) and n.level == 2 * L + 2):
                continue
            units += [f"{n.name}[{t}]" for t in range(len(n.terms))]
            if n.out in alloc.subtract:
                units.append(f"handover:{dn(alloc.subtract[n.out])}")
        units += [f"clear:{dn(o)}" for o in alloc.clears.get(2 * L + 2, [])]
        if exit_level == 2 * L + 2:
            units += [f"exit.clear:{dn(o)}" for o in exit_clear]
        heads = [n for n in c.nodes if isinstance(n, Head) and n.level == 2 * L + 1]
        layers.append({"units": units, "heads": [h.name for h in heads],
                       "head_vheads": [len(h.vheads()) for h in heads],
                       "head_used": [any(vh.out for vh in h.vheads()) for h in heads]})
    # node graph: writer -> reader through IR dims (the real dataflow of the compiled CPU)
    writer = {}
    nodes = []
    for i, n in enumerate(c.nodes):
        kind = "head" if isinstance(n, Head) else "ffn"
        outs = [d for vh in n.vheads() for _, d, _ in vh.out] if kind == "head" else [n.out]
        nodes.append({"name": n.name, "kind": kind, "level": n.level})
        for d in outs:
            writer.setdefault(d, []).append(i)
    edges = set()
    for i, n in enumerate(c.nodes):
        for d in reads(n):
            for w in writer.get(d, []):
                if w != i and c.nodes[w].level < n.level:
                    edges.add((w, i))
    return layers, {"nodes": nodes, "edges": sorted(edges)}


class Rec:
    def __init__(self, chat):
        self.chat = chat
        self.on_chip = False
        self.on_qwen = True
        self.mlp = {}      # layer -> list of [T, U] fp16
        self.heads = {}    # layer -> list of [T, H] fp16 (per value head output norm)
        self.resid = {}    # layer -> list of [T] fp32
        self.pred = []     # circuit code prediction per chip-stream position
        self.hooks = []
        n_chip = chat.n_chip
        for i, lay in enumerate(chat.lm.layers):
            chip = i < n_chip
            dp = lay.mlp.down_proj
            sl = slice(dp.c0, dp.c1) if hasattr(dp, "c0") else slice(None)
            att = lay.linear_attn if hasattr(lay, "linear_attn") and lay.linear_attn is not None else lay.self_attn
            op = att.out_proj if hasattr(att, "out_proj") else att.o_proj
            hd = att.head_v_dim if hasattr(att, "head_v_dim") else att.head_dim
            self.hooks.append(dp.register_forward_pre_hook(self._mk_mlp(i, chip, sl)))
            self.hooks.append(op.register_forward_pre_hook(self._mk_head(i, chip, hd)))
            self.hooks.append(lay.register_forward_hook(self._mk_res(i, chip)))
        self.mlp_slice = {}
        for i, lay in enumerate(chat.lm.layers):
            dp = lay.mlp.down_proj
            self.mlp_slice[i] = (dp.c0, dp.c1) if hasattr(dp, "c0") else (0, dp.in_features)

    def _active(self, chip):
        return self.on_chip if chip else self.on_qwen

    def _mk_mlp(self, i, chip, sl):
        def f(mod, args):
            if self._active(chip):
                self.mlp.setdefault(i, []).append(args[0][0, :, sl].detach().to(torch.float32 if chip else torch.float16).cpu())  # chip: exit ballast units exceed fp16
        return f

    def _mk_head(self, i, chip, hd):
        def f(mod, args):
            if self._active(chip):
                x = args[0][0].float()
                self.heads.setdefault(i, []).append(x.view(x.shape[0], -1, hd).norm(dim=-1).half().cpu())
        return f

    def _mk_res(self, i, chip):
        def f(mod, args, out):
            if not self._active(chip):
                return
            h = out[0] if isinstance(out, tuple) else out
            self.resid.setdefault(i, []).append(h[0].float().norm(dim=-1).cpu())
            if chip and i == self.chat.n_chip - 1:
                self.pred += self.chat._decode_codes(h[0][:, self.chat.code_dims])
        return f


def main():
    from rfpu_v15.textio import utf8_stdout
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=0, help="index into stage2_chat.LEVELS")
    ap.add_argument("--out", default=str(DATA / "viz/levels-q0"))
    ap.add_argument("--max-new", type=int, default=1000)
    a = ap.parse_args()
    import stage2_chat as S2
    from chipformer.rv import bridge as BR, ref
    from chipformer.rv import spec as S
    from chipformer.rv.asm import assemble
    from rfpu_v15.stage2.chunked_chat import ChunkedRvChat
    from rfpu_v15.stage2.firmware import RF_LEVELS_TOOL, RF_TOOL, source

    class RecChat(ChunkedRvChat):
        def _capture(self):            # no CUDA graph: hooks must fire on every chip step
            pass

        def _chip_cache(self):         # the full-model prefill is over: record the chip view from here
            cc = super()._chip_cache()
            self.rec.on_chip = True
            return cc

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    chat = RecChat(RV1, fast=True, executor=False, chunk=1024)
    print(f"loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    names, graph = circuit_names(chat)
    caps = S2.capture_set()
    prog = assemble(source(caps, not_measured=True))
    listing = "\n".join(f"{i}: {label}" for i, (label, _) in enumerate(caps))
    system = ("You can analyze stored radio IQ captures with the rf_analyze tool, which runs exact signal "
              "processing on the processor built into this model." + S2.RULE + " Available captures:\n" + listing)
    cap, q = S2.LEVELS[a.q]
    m = ref.Machine.from_program(prog); BR.park(m)
    m2 = ref.Machine.from_program(prog); BR.park(m2)
    ids = chat.prompt_ids([{"role": "system", "content": system}, {"role": "user", "content": q}],
                          tools=(RF_TOOL, RF_LEVELS_TOOL))
    lock = BR.Session(chat.T, m2)
    lock.history(ids)
    chat.rec = Rec(chat)
    t = time.perf_counter()
    r = chat.generate(ids, m, a.max_new, max_chip=2_000_000, check_head=True, verbose=True, lock=lock)
    secs = time.perf_counter() - t
    rec = chat.rec
    for h in rec.hooks:
        h.remove()
    gen = r["tokens"]
    text = chat.text_of(gen)
    print(text, flush=True)
    print(f"{r['text_tokens']} text + {r['chip_tokens']} chip tokens, {secs:.0f}s, lock error {r['lock_error']}")

    # streams: chip view sees ids + gen[:-1] (every input); Qwen sees the prompt + text inputs
    n_rows = sum(x.shape[0] for x in rec.mlp[0])          # positions the chip view actually ran
    chip_in = (ids + gen)[:n_rows]                        # gen[-1] is fed only if the run hit max_new
    v0 = chat.v0

    def tname(t):
        return S.name(t - v0) if t >= v0 else chat.tok.decode([t])
    toks = [{"id": int(t), "s": tname(t), "chip": bool(t >= v0)} for t in chip_in]
    qwen_pos = [i for i, t in enumerate(chip_in) if i < len(ids) or t < v0]
    n_chip = chat.n_chip
    meta = {"question": q, "capture": cap, "prompt_tokens": len(ids), "n_tokens": len(chip_in),
            "text_tokens": r["text_tokens"], "chip_tokens": r["chip_tokens"], "seconds": secs,
            "lock_error": r["lock_error"], "head_mismatch": r["head_mismatch"], "answer": text,
            "n_chip_layers": n_chip, "n_layers": len(chat.lm.layers), "qwen_positions": qwen_pos,
            "layer_types": list(chat.model.config.layer_types), "layers": []}
    # verification: circuit prediction at chip position i vs the token actually emitted at i+1
    pred = rec.pred
    full = ids + gen
    nxt = [full[i + 1] if i + 1 < len(full) else None for i in range(len(chip_in))]
    owned = [p is not None for p in pred]
    ok = [p is None or n is None or p == n for p, n in zip(pred, nxt)]
    # prompt positions (all but the last) are given, not generated: the circuit may still fire there
    # (e.g. on literal "</tool_call>" text in the tool instructions) -> status 2, not checked
    P = len(ids)
    st = [0 if p is None else (2 if i < P - 1 else (1 if k else -1)) for i, (p, k) in enumerate(zip(pred, ok))]
    g = st[P - 1:]
    meta["verify"] = {"positions": len(st), "prompt_positions_where_circuit_fired": st[:P - 1].count(2),
                      "generated_positions": len(g), "circuit_owned": sum(x != 0 for x in g),
                      "circuit_equal_emitted": g.count(1), "mismatch": g.count(-1)}
    np.save(out / "pred_ok.npy", np.array(st, np.int8))
    for i in range(len(chat.lm.layers)):
        chip = i < n_chip
        mlp = torch.cat(rec.mlp[i]).numpy()
        hd = torch.cat(rec.heads[i]).numpy()
        rs = torch.cat(rec.resid[i]).numpy()
        exp = len(chip_in) if chip else len(qwen_pos)
        assert mlp.shape[0] == exp, (i, mlp.shape, exp)
        np.save(out / f"mlp_{i:02d}.npy", mlp)
        np.save(out / f"head_{i:02d}.npy", hd)
        np.save(out / f"resid_{i:02d}.npy", rs)
        info = {"i": i, "chip": chip, "type": meta["layer_types"][i], "mlp_range": rec.mlp_slice[i],
                "units": mlp.shape[1], "heads": hd.shape[1],
                "mlp_absmax": float(np.abs(mlp).max()) if mlp.size else 0.0, "head_max": float(hd.max()) if hd.size else 0.0}
        if chip:
            nm = names[i]
            c0 = rec.mlp_slice[i][0]
            info["unit_names"] = [nm["units"][c0 + u] if c0 + u < len(nm["units"]) else ("exit.clear:ballast" if i == n_chip - 1 else "unused")
                                  for u in range(mlp.shape[1])]
            kept = [j for j, u in enumerate(nm["head_used"]) if u] or [0]
            vh = []
            for j in kept:
                for x in range(2):
                    vh.append(nm["heads"][j] + (f".v{x}" if x else "") if j < len(nm["heads"]) else "unused")
            info["head_names"] = vh if len(vh) == hd.shape[1] else [f"vhead{k}" for k in range(hd.shape[1])]
        meta["layers"].append(info)
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (out / "tokens.json").write_text(json.dumps(toks), encoding="utf-8")
    (out / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    print(json.dumps(meta["verify"]), flush=True)


if __name__ == "__main__":
    main()

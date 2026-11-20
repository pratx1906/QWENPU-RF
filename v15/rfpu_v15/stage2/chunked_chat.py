"""RvChat with chunked (speculative) verification of the computer's tokens inside a conversation.

During a tool call the reference lock (BR.Session) already knows the computer's upcoming tokens. A
run of consecutive chip-input tokens is fed through the saved chip layers C at a time (one parallel
forward per chunk, the chip view's cache), and at every position the circuit's predicted token must
equal the reference's next token: by induction over positions this is the token-by-token run, so the
chip layers verify every computer token. A mismatch stops generation with a lock error, as the
sequential path would. Tokens that are text inputs (Qwen's own tokens, the report characters the
firmware prints) take the unchanged one-token path through the chip view and Qwen's layers.

Audit mode, stated plainly: the RV32 reference computes the draft; the model's layers verify it.
The upstream RvChat is not modified (subclass; `generate` follows chipformer.rv.chat line by line
apart from the marked chunk block)."""
import time

import numpy as np
import torch

from chipformer.rv import bridge as BR
from chipformer.rv import core, hf
from chipformer.rv import spec as S
from chipformer.rv.chat import RvChat


class ChunkedRvChat(RvChat):
    def __init__(self, *args, chunk: int = 1024, **kw):
        super().__init__(*args, **kw)
        assert self.fast and not self.executor, "chunked verification runs on the fast (saved-layer) path"
        self.chunk = chunk
        self.n_chunked = 0

    def _state_refs(self):
        return [(self.ccache.layers[L].conv_states[0], self.ccache.layers[L].recurrent_states[0])
                for L in range(self.n_chip)]

    def _keep_static(self, refs):
        """The CUDA graph reads the cache's original state buffers: if a chunked forward replaced
        them, copy the new states back into the originals and restore them."""
        for L, (cs, rs) in enumerate(refs):
            lay = self.ccache.layers[L]
            if lay.conv_states[0] is not cs:
                cs.copy_(lay.conv_states[0])
                lay.conv_states[0] = cs
            if lay.recurrent_states[0] is not rs:
                rs.copy_(lay.recurrent_states[0])
                lay.recurrent_states[0] = rs

    def _decode_codes(self, codes):
        """code_token for every row of a chunk, one device-to-host copy (same rule)."""
        c = codes.float()
        owned = (c != 0).any(1).cpu().numpy()
        bits = (c > 0).cpu().numpy().astype(np.int64)
        nf = core.NFAM
        fid = bits[:, :nf] @ (1 << np.arange(nf))
        pay = bits[:, nf:] @ (1 << np.arange(bits.shape[1] - nf))
        out = []
        for ok, f, p in zip(owned, fid, pay):
            if not ok:
                out.append(None)
                continue
            t = core.decode_code(int(f), int(p), self.T)
            out.append(-1 if t < 0 else (t - S.TEXT_TOKEN if S.is_text(t) else self.v0 + t))
        return out

    def _mid(self, st):
        """A stream token (chip token or TEXT_TOKEN | qwen id) as a model id."""
        return self.v0 + st if st < S.VOCAB else st - S.TEXT_TOKEN

    @torch.no_grad()
    def _verify_run(self, seq, refs):
        """Feed seq[0..m-1] (chip tokens) in chunks; position i must predict seq[i + 1]. Returns
        (number verified, first bad index or None, got)."""
        m = len(seq) - 1
        done = 0
        while done < m:
            n = min(self.chunk, m - done)
            x = torch.tensor([[self._mid(t) for t in seq[done:done + n]]], device=self.dev)
            h = self.cm.embed_tokens(x)
            for L in range(self.n_chip):
                h = self.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                      position_ids=None, past_key_values=self.ccache)
            self._keep_static(refs)
            got = self._decode_codes(h[0][:, self.code_dims])
            self.n_chunked += 1
            for j, gt in enumerate(got):
                if gt != self._mid(seq[done + j + 1]):
                    return done + j, done + j, gt
            done += n
        return m, None, None

    @torch.no_grad()
    def generate(self, ids, machine, max_new=4000, max_chip=60000, check_head=False, verbose=False,
                 lock=None, on_token=None):
        x = torch.tensor([ids], device=self.dev)
        out = self.model(input_ids=x, use_cache=True)
        cache = out.past_key_values
        core.snapshot(self.c, machine)
        ccache = self._chip_cache()
        h = self.cm.embed_tokens(x)
        for L in range(self.n_chip):
            h = self.cm.layers[L](h, position_embeddings=None, attention_mask=None,
                                  position_ids=None, past_key_values=ccache)
        hf.inject(self.c, ccache)
        if self.graph is None:
            self._capture()
        refs = self._state_refs()
        g = len(ids)
        logits = out.logits[0, -1]
        pos = len(ids)
        gen, n_text, n_chip_tok = [], 0, 0
        nxt = int(torch.argmax(logits[: self.v0]))
        t0 = time.time()
        mismatch = 0
        t_chip = t_text = 0.0
        expect, lock_error = [], None
        while n_text < max_new and n_chip_tok < max_chip:
            gen.append(nxt)
            if on_token is not None:
                on_token(nxt)
            if lock is not None:
                st = nxt - self.v0 if nxt >= self.v0 else BR.text_token(nxt)
                if expect:
                    e = expect.pop(0)
                    if st != e:
                        lock_error = {"at": len(gen) - 1, "got": S.name(st) if st < S.VOCAB
                                      else self.tok.decode([nxt]),
                                      "want": S.name(e) if e < S.VOCAB
                                      else self.tok.decode([e - S.TEXT_TOKEN])}
                        break
                elif st < S.VOCAB:
                    lock_error = {"at": len(gen) - 1, "got": S.name(st), "want": "text"}
                    break
                else:
                    k = len(lock.toks)
                    try:
                        lock.qwen([nxt])
                    except RuntimeError as e:
                        lock_error = {"at": len(gen) - 1, "got": str(e), "want": "a halt"}
                        break
                    expect = list(lock.toks[k + 1:])
            if nxt in self.eos:
                break
            if verbose and len(gen) % 500 == 0:
                print(f"  {len(gen)} tokens ({n_text} text, {n_chip_tok} chip) {time.time() - t0:.0f}s", flush=True)
            ts = time.time()
            chip_in = nxt >= self.v0
            # ---- chunk block: a run of chip-input tokens, each with a known next token, in parallel
            if chip_in and lock is not None and expect:
                seq = [nxt - self.v0] + expect
                m = 1
                while m < len(seq) - 1 and seq[m] < S.VOCAB:
                    m += 1                       # inputs seq[0..m-1] are chip tokens; targets seq[1..m]
                m = min(m, max_chip - n_chip_tok)
                if m >= 2:
                    ok, bad, got = self._verify_run(seq[:m + 1], refs)
                    g += ok if bad is None else bad + 1
                    if bad is not None:
                        gen.extend(self._mid(t) for t in seq[1:bad + 1])
                        lock_error = {"at": len(gen), "got": got, "want": self._mid(seq[bad + 1]),
                                      "chunked": True}
                        break
                    run_ids = [self._mid(t) for t in seq[1:m]]
                    gen.extend(run_ids)
                    if on_token is not None:
                        for t in run_ids:
                            on_token(t)
                    expect = expect[m - 1:]
                    n_chip_tok += m
                    nxt = self._mid(seq[m])
                    t_chip += time.time() - ts
                    continue
            # ---- end chunk block
            hc = self._chip_step(nxt, g, ccache)
            g += 1
            if chip_in:
                n_chip_tok += 1
                nxt = self.circuit_token(hc)
                t_chip += time.time() - ts
                if nxt is None:
                    raise RuntimeError("the circuit gave up a chip position")
            else:
                n_text += 1
                logits = self._qwen_step(hc, cache, pos)
                pos += 1
                ct = self.circuit_token(hc)
                q = int(torch.argmax(logits))
                if ct is not None and check_head and q != ct:
                    mismatch += 1
                nxt = ct if ct is not None else int(torch.argmax(logits[: self.v0]))
                t_text += time.time() - ts
        dt = time.time() - t0
        return {"tokens": gen, "text_tokens": n_text, "chip_tokens": n_chip_tok,
                "seconds": dt, "head_mismatch": mismatch, "lock_error": lock_error,
                "verify_mismatch": None, "chunks": self.n_chunked,
                "chip_seconds": round(t_chip, 2), "text_seconds": round(t_text, 2)}

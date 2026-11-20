"""SPIKE: serve a recorded run (record.py output) to the browser viewer.

  python v15/viz/serve.py $RFPU_DATA/viz/levels-q0 [--port 8765]

Endpoints: / (viewer), /meta, /tokens, /graph (json); /frame?t=N (binary: per layer int8 MLP
activations scaled by the layer's absmax, then uint8 head norms scaled by the layer's max; layers
the token does not pass through are zeros and flagged in a leading presence byte per layer);
/series?l=L&kind=mlp|head&u=U&t0=A&t1=B (float32 values over chip-stream tokens A..B).
"""
import argparse
import json
import sys
from bisect import bisect_left
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent


class Run:
    def __init__(self, d):
        d = Path(d)
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.tokens = (d / "tokens.json").read_bytes()
        self.graph = (d / "graph.json").read_bytes()
        self.L = self.meta["n_layers"]
        self.nc = self.meta["n_chip_layers"]
        self.mlp = [np.load(d / f"mlp_{i:02d}.npy", mmap_mode="r") for i in range(self.L)]
        self.head = [np.load(d / f"head_{i:02d}.npy", mmap_mode="r") for i in range(self.L)]
        self.qpos = self.meta["qwen_positions"]
        self.pred_ok = np.load(d / "pred_ok.npy")
        self.resid = [np.load(d / f"resid_{i:02d}.npy") for i in range(self.L)]
        # per-layer scales: robust (99.9th pct of |x|) so a few huge values do not wash out the rest
        self.ms, self.hs = [], []
        for i in range(self.L):
            a = np.abs(np.nan_to_num(np.asarray(self.mlp[i][:: max(1, len(self.mlp[i]) // 2000)], dtype=np.float32), posinf=65504, neginf=-65504))
            self.ms.append(float(np.percentile(a[a > 0], 99.9)) if (a > 0).any() else 1.0)
            h = np.asarray(self.head[i][:: max(1, len(self.head[i]) // 2000)], dtype=np.float32)
            self.hs.append(float(np.percentile(h[h > 0], 99.9)) if (h > 0).any() else 1.0)
        for i, info in enumerate(self.meta["layers"]):
            info["mlp_scale"], info["head_scale"] = self.ms[i], self.hs[i]
        # overview: per token, mean |act| of chip layers / qwen layers, resid of last chip layer
        n = self.meta["n_tokens"]
        self.meta["overview"] = {
            "pred_ok": self.pred_ok.tolist(),
            "chip_resid": np.asarray(self.resid[self.nc - 1], dtype=np.float32)[:n].round(1).tolist(),
        }
        self.quantize(d)

    def row(self, i, t):
        """Index of chip-stream token t in layer i's arrays, or None."""
        if i < self.nc:
            return t
        k = bisect_left(self.qpos, t)
        return k if k < len(self.qpos) and self.qpos[k] == t else None

    def quantize(self, d):
        """int8 MLP / uint8 head copies, scaled per layer, built once next to the run."""
        self.qm, self.qh = [], []
        for i in range(self.L):
            fm, fh = d / f"q8mlp_{i:02d}.npy", d / f"q8head_{i:02d}.npy"
            if not fm.exists() or not fh.exists():
                m = np.nan_to_num(np.asarray(self.mlp[i], dtype=np.float32), posinf=65504, neginf=-65504) / self.ms[i] * 127  # fp16 overflow (exit ballast units)
                np.save(fm, np.clip(np.round(m), -127, 127).astype(np.int8))
                h = np.asarray(self.head[i], dtype=np.float32) / self.hs[i] * 255
                np.save(fh, np.clip(np.round(h), 0, 255).astype(np.uint8))
            self.qm.append(np.load(fm, mmap_mode="r"))
            self.qh.append(np.load(fh, mmap_mode="r"))

    def frame(self, t):
        """presence byte per layer, then for each present layer: int8[U] MLP, uint8[H] heads."""
        pres = bytearray(self.L)
        parts = [pres]
        for i in range(self.L):
            r = self.row(i, t)
            if r is None:
                continue
            pres[i] = 1
            parts.append(self.qm[i][r].tobytes())
            parts.append(self.qh[i][r].tobytes())
        return b"".join(bytes(p) for p in parts)

    def frames(self, t, n):
        out = []
        for k in range(t, min(t + n, self.meta["n_tokens"])):
            f = self.frame(k)
            out.append(len(f).to_bytes(4, "little") + f)
        return b"".join(out)

    def series(self, i, kind, u, t0, t1):
        arr = self.mlp[i] if kind == "mlp" else self.head[i]
        if i < self.nc:
            return np.asarray(arr[t0:t1, u], dtype=np.float32)
        out = np.full(t1 - t0, np.nan, np.float32)
        for k in range(bisect_left(self.qpos, t0), bisect_left(self.qpos, t1)):
            out[self.qpos[k] - t0] = arr[k, u]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    run = Run(a.run)
    def finite(o):  # JSON has no Infinity/NaN: the browser's JSON.parse rejects them
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [finite(v) for v in o]
        return o
    meta = json.dumps(finite(run.meta), allow_nan=False).encode()

    class H(SimpleHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    self.send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/meta":
                    self.send(meta, "application/json")
                elif u.path == "/tokens":
                    self.send(run.tokens, "application/json")
                elif u.path == "/graph":
                    self.send(run.graph, "application/json")
                elif u.path == "/frames":
                    self.send(run.frames(int(q["t"]), int(q.get("n", 32))), "application/octet-stream")
                elif u.path == "/frame":
                    self.send(run.frame(int(q["t"])), "application/octet-stream")
                elif u.path == "/series":
                    self.send(run.series(int(q["l"]), q["kind"], int(q["u"]), int(q["t0"]), int(q["t1"])).tobytes(),
                              "application/octet-stream")
                else:
                    self.send_error(404)
            except Exception as e:  # noqa: BLE001 - spike
                self.send_error(500, str(e))

    print(f"http://localhost:{a.port}/  ({run.meta['n_tokens']} tokens, {run.L} layers)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main())

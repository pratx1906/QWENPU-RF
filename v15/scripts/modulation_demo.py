"""PROTOCOL Amendment 15: blind 25-class modulation classifier on the CPU in weights.

  modulation_demo.py train    host: synthetic train (seed 1500) / dev (seed 1501) sets, reference features,
                              float MLP -> int16 codes; writes reports/modclass-model.json (dev accuracy only)
  modulation_demo.py neural   GPU: the pre-declared test set (seed 1515, 10 captures per class at SNR 10..28 dB),
                              every capture through ModFeatures then IntMLP on the CPU in weights, every token
                              checked against the RV32 reference, outputs against the independent references;
                              writes reports/modclass-neural.json
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "v15")]
from rfpu_v15.cpu.modclass import FEATURE_NAMES, NF, IntMLP, features  # noqa: E402
from rfpu_v15.modgen import CLASSES, capture  # noqa: E402

REPORTS = ROOT / "v15/reports"
MODEL = REPORTS / "modclass-model.json"
TEST_SNRS = list(range(10, 30, 2))
H = 32


def _job(args):
    cls, seed, snr = args
    x = capture(CLASSES[cls], np.random.default_rng(seed), snr)
    return cls, snr, features(x)


def dataset(seed, per_class, snrs=None):
    rng = np.random.default_rng(seed)
    jobs = []
    for c in range(len(CLASSES)):
        for i in range(per_class):
            snr = snrs[i % len(snrs)] if snrs else float(rng.uniform(8, 30))
            jobs.append((c, int(rng.integers(2**62)), snr))
    with Pool(12) as p:
        return jobs, p.map(_job, jobs, chunksize=32)


def train_float(X, y, seed=0, epochs=3000, lr=3e-3, l2=1e-4):
    rng = np.random.default_rng(seed)
    C = len(CLASSES)
    W1 = rng.normal(scale=np.sqrt(2 / NF), size=(H, NF)); b1 = np.zeros(H)
    W2 = rng.normal(scale=np.sqrt(1 / H), size=(C, H)); b2 = np.zeros(C)
    P = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
    Y = np.eye(C)[y]
    for t in range(1, epochs + 1):
        Z = X @ W1.T + b1; A = np.maximum(Z, 0); S = A @ W2.T + b2
        S -= S.max(1, keepdims=True); E = np.exp(S); Pr = E / E.sum(1, keepdims=True)
        G = (Pr - Y) / len(X)
        gW2 = G.T @ A + l2 * W2; gb2 = G.sum(0)
        GA = G @ W2 * (Z > 0)
        gW1 = GA.T @ X + l2 * W1; gb1 = GA.sum(0)
        for i, g in enumerate((gW1, gb1, gW2, gb2)):
            m[i] = .9 * m[i] + .1 * g; v[i] = .999 * v[i] + .001 * g * g
            P[i] -= lr * (m[i] / (1 - .9 ** t)) / (np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
    return P


def quantize(P, mu, sd, X):
    W1, b1, W2, b2 = P
    We, be = W1 / sd, b1 - (W1 / sd) @ mu                    # folded standardisation, raw integer features
    hmax = np.percentile(np.maximum(X @ We.T + be, 0), 99.9) or 1.
    A = 4096 / hmax                                          # hidden activations ~ 0..4096
    s1 = int(np.floor(np.log2(32767 / (A * np.abs(We).max()))))
    W1q = np.round(We * A * 2 ** s1).astype(int); b1q = np.round(be * A).astype(int)
    B = 32767 / (np.abs(W2).max() * 4096 / A)
    W2q = np.round(W2 * 4096 * B / A).astype(int); b2q = np.round(b2 * B).astype(int)
    return IntMLP(W1q.tolist(), b1q.tolist(), W2q.tolist(), b2q.tolist(), s1=s1)


def train():
    t0 = time.perf_counter()
    _, tr = dataset(1500, 600)
    jd, dv = dataset(1501, 10, TEST_SNRS)
    print(f"features {time.perf_counter() - t0:.0f}s", flush=True)
    Xtr = np.array([r[2] for r in tr], float); ytr = np.array([r[0] for r in tr])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    P = train_float((Xtr - mu) / sd, ytr)
    mlp = quantize(P, mu, sd, Xtr)
    acc = lambda rows: float(np.mean([mlp.forward(list(r[2]))[1] == r[0] for r in rows]))  # noqa: E731
    W1, b1, W2, b2 = P
    fl = lambda rows: float(np.mean(np.argmax(np.maximum((np.array([r[2] for r in rows]) - mu) / sd @ W1.T + b1, 0) @ W2.T + b2, 1) == [r[0] for r in rows]))  # noqa: E731
    by_snr = {s: acc([r for r in dv if r[1] == s]) for s in TEST_SNRS}
    out = {"protocol": "PROTOCOL.md Amendment 15", "classes": CLASSES, "features": FEATURE_NAMES, "hidden": H,
           "train": {"seed": 1500, "per_class": 600, "snr_db": "U[8, 30]"}, "dev": {"seed": 1501, "per_class": 10, "snr_db": TEST_SNRS},
           "train_accuracy_int": acc(tr), "dev_accuracy_float": fl(dv), "dev_accuracy_int": acc(dv), "dev_accuracy_by_snr": by_snr,
           "dev_confusion": confusion([r[0] for r in dv], [mlp.forward(list(r[2]))[1] for r in dv]),
           "mlp": {"w1": mlp.w1, "b1": mlp.b1, "w2": mlp.w2, "b2": mlp.b2, "s1": mlp.s1}, "seconds": time.perf_counter() - t0}
    MODEL.write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("train_accuracy_int", "dev_accuracy_float", "dev_accuracy_int", "dev_accuracy_by_snr")}, indent=1))
    per = {CLASSES[c]: out["dev_confusion"][c][c] for c in range(len(CLASSES))}
    print("dev correct per class (of 10):", per)


def confusion(y, p):
    C = len(CLASSES)
    M = [[0] * C for _ in range(C)]
    for a, b in zip(y, p):
        M[a][b] += 1
    return M


def load_mlp():
    d = json.loads(MODEL.read_text(encoding="utf-8"))["mlp"]
    return IntMLP(d["w1"], d["b1"], d["w2"], d["b2"], s1=d["s1"])


def neural():
    from rfpu_v15.cpu.modclass import ModFeatures
    from rfpu_v15.cpu.runner import Runner
    rng = np.random.default_rng(1515)
    jobs = [(c, int(rng.integers(2**62)), s) for c in range(len(CLASSES)) for s in TEST_SNRS]
    caps = [capture(CLASSES[c], np.random.default_rng(seed), snr) for c, seed, snr in jobs]
    mlp = load_mlp()
    R = Runner(vx=False, batch=8)
    t0 = time.perf_counter()
    fk = ModFeatures()
    rows, tokens, t_neural = [], 0, 0.0
    for i in range(0, len(caps), 50):
        part = R.run_many(fk, caps[i:i + 50], block=1024)
        tokens += part[0].tokens; t_neural += part[0].neural_seconds
        for j, r in enumerate(part):
            rows.append({"capture": i + j, "class": CLASSES[jobs[i + j][0]], "snr_db": jobs[i + j][2],
                         "features": list(r.outputs[-1]), "features_exact": r.exact_outputs})
        print(f"features {len(rows)}/{len(caps)} exact {all(r['features_exact'] for r in rows)} "
              f"tokens {tokens} {time.perf_counter() - t0:.0f}s", flush=True)
    R2 = Runner(vx=False, batch=8)
    streams = [[(v, 0) for v in r["features"]] + [(0, 0)] * (32 - NF) for r in rows]
    res = R2.run_many(mlp, streams, block=32)
    mlp_tokens = res[0].tokens
    for r, m in zip(rows, res):
        scores, cls = m.outputs[-1]
        r.update(predicted=CLASSES[cls], correct=CLASSES[cls] == r["class"], mlp_exact=m.exact_outputs, scores=list(scores))
    acc = float(np.mean([r["correct"] for r in rows]))
    out = {"protocol": "PROTOCOL.md Amendment 15", "test": {"seed": 1515, "per_class": len(TEST_SNRS), "snr_db": TEST_SNRS},
           "captures": len(rows), "all_features_exact": all(r["features_exact"] for r in rows),
           "all_mlp_exact": all(r["mlp_exact"] for r in rows), "accuracy": acc,
           "accuracy_by_snr": {s: float(np.mean([r["correct"] for r in rows if r["snr_db"] == s])) for s in TEST_SNRS},
           "correct_per_class": {c: sum(r["correct"] for r in rows if r["class"] == c) for c in CLASSES},
           "confusion": confusion([CLASSES.index(r["class"]) for r in rows], [CLASSES.index(r["predicted"]) for r in rows]),
           "tokens_features": tokens, "tokens_mlp": mlp_tokens, "neural_seconds_features": t_neural,
           "seconds": time.perf_counter() - t0, "rows": rows}
    (REPORTS / "modclass-neural.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "confusion")}, indent=1))


if __name__ == "__main__":
    {"train": train, "neural": neural}[sys.argv[1]]()

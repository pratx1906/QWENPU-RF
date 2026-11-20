"""Real-data technology classifier (WiFi / LTE / DVB-T) for the CPU in weights.

Data: IDLab "IQ samples of LTE, WiFi and DVB-T in a real environment" (data/idlab-iq-manifest.json),
1 MS/s recordings, six locations. Split: the manifest's calibration recordings train, held_out
recordings test (recording-level split; no window of a test recording is seen in training).
Per recording: gain 0.25/p95 from its first 512 samples, then 16 windows of 1024 samples evenly
spaced over the rest, Q15. Features are the exact integer features of TechClassifier (computed by
its independent reference, identical to the CPU's by test). A float MLP is trained on them,
quantized to int16 codes, and evaluated exactly in integer arithmetic. --neural additionally runs one
window of every held-out recording on the CPU in weights and checks it equals the integer model.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15")]
from rfpu_v15.paths import local  # noqa: E402
from rfpu_v15.cpu.classify import WIN, TechClassifier  # noqa: E402

CLASSES = ["WiFi", "LTE", "DVB-T"]
L2 = float(sys.argv[sys.argv.index("--l2") + 1]) if "--l2" in sys.argv else 1e-4
H, F, WINDOWS = int(sys.argv[sys.argv.index("--hidden") + 1]) if "--hidden" in sys.argv else 12, int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 16, 16


def recordings():
    d = json.loads((ROOT / "data/idlab-iq-manifest.json").read_text())
    items = next(v for v in d.values() if isinstance(v, list))
    return [i for i in items if i["sample_rate"] == 1e6 and i["technology"] in CLASSES and local(i["path"]).exists()]


def windows(item):
    raw = np.memmap(local(item["path"]), dtype="<i2", mode="r").reshape(-1, 2)
    z = (raw[:, 0].astype(float) + 1j * raw[:, 1]) / 32768
    p95 = float(np.percentile(np.abs(np.concatenate((z[:512].real, z[:512].imag))), 95)) or 1.
    z = z * (.25 / p95)
    starts = np.linspace(512, len(z) - WIN, WINDOWS).astype(int)
    out = []
    for s in starts:
        q = np.clip(np.round(np.stack((z[s:s + WIN].real, z[s:s + WIN].imag), -1) * 32768), -32768, 32767).astype(int)
        out.append([tuple(map(int, v)) for v in q])
    return out


def main():
    t0 = time.perf_counter()
    probe = TechClassifier([[0] * 17] * H, [0] * H, [[0] * H] * 3, [0] * 3, frames=F)
    data = {"train": [], "test": []}
    for item in recordings():
        split = "train" if item["split"] == "calibration" else "test"
        y = CLASSES.index(item["technology"])
        for w in windows(item):
            data[split].append((item["name"], item["location"], y, w, probe.features(w)))
    print({k: len(v) for k, v in data.items()}, f"features {time.perf_counter() - t0:.0f}s", flush=True)
    Xtr = np.array([d[4] for d in data["train"]], float) / 256; ytr = np.array([d[2] for d in data["train"]])
    Xte = np.array([d[4] for d in data["test"]], float) / 256; yte = np.array([d[2] for d in data["test"]])
    rng = np.random.default_rng(1)
    A = rng.normal(scale=.3, size=(H, 17)); a = np.zeros(H); B = rng.normal(scale=.3, size=(3, H)); b = np.zeros(3)
    w_cls = len(ytr) / (3 * np.bincount(ytr, minlength=3))
    Y, sw = np.eye(3)[ytr], w_cls[ytr][:, None]
    m = [np.zeros_like(v) for v in (A, a, B, b)]; v2 = [np.zeros_like(v) for v in (A, a, B, b)]
    for step in range(1, 6001):  # Adam, class-balanced softmax cross-entropy, small L2
        z = Xtr @ A.T + a; h = np.maximum(z, 0); s = h @ B.T + b
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        g = (p - Y) * sw / len(Xtr)
        grads = [None, None, g.T @ h + L2 * B, g.sum(0)]
        gh = g @ B * (z > 0)
        grads[0], grads[1] = gh.T @ Xtr + L2 * A, gh.sum(0)
        for i, (prm, gr) in enumerate(zip((A, a, B, b), grads)):
            m[i] = .9 * m[i] + .1 * gr; v2[i] = .999 * v2[i] + .001 * gr * gr
            prm -= .01 * (m[i] / (1 - .9 ** step)) / (np.sqrt(v2[i] / (1 - .999 ** step)) + 1e-8)
        np.clip(A, -7.9, 7.9, out=A); np.clip(B, -7.9, 7.9, out=B)
    w1 = np.clip(np.round(A * 4096), -32768, 32767).astype(int).tolist()
    b1 = np.clip(np.round(a * 256), -32768, 32767).astype(int).tolist()
    w2 = np.clip(np.round(B * 4096), -32768, 32767).astype(int).tolist()
    b2 = np.clip(np.round(b * 256), -32768, 32767).astype(int).tolist()
    model = TechClassifier(w1, b1, w2, b2, frames=F)
    fp = (np.maximum(Xte @ A.T + a, 0) @ B.T + b).argmax(1)
    ip = np.array([model.forward(d[4])[1] for d in data["test"]])
    itr = np.array([model.forward(d[4])[1] for d in data["train"]])
    recs = {}
    for d, pr in zip(data["test"], ip):
        recs.setdefault(d[0], [d[2], []])[1].append(pr)
    rec_acc = np.mean([np.bincount(p, minlength=3).argmax() == y for y, p in recs.values()])
    conf = [[int(np.sum((yte == i) & (ip == j))) for j in range(3)] for i in range(3)]
    per_loc = {}
    for d, pr in zip(data["test"], ip):
        per_loc.setdefault(d[1], []).append(pr == d[2])
    out = {"classes": CLASSES, "hidden": H, "frames": F, "w1": w1, "b1": b1, "w2": w2, "b2": b2,
           "train_windows": len(ytr), "test_windows": len(yte), "train_recordings": len({d[0] for d in data["train"]}),
           "test_recordings": len(recs), "train_accuracy_integer": float(np.mean(itr == ytr)),
           "test_accuracy_float": float(np.mean(fp == yte)), "test_accuracy_integer": float(np.mean(ip == yte)),
           "test_recording_accuracy_majority": float(rec_acc), "confusion_rows_true_cols_pred": conf,
           "test_accuracy_by_location": {k: float(np.mean(v)) for k, v in per_loc.items()},
           "chance_majority_class": float(np.bincount(yte).max() / len(yte)),
           "scope": "IDLab 1 MS/s, recording-level split from the manifest; single dataset/receiver; technology labels from the publisher"}
    if "--neural" in sys.argv:
        from rfpu_v15.cpu.runner import Runner
        firsts = {}
        for d in data["test"]:
            firsts.setdefault(d[0], d)
        streams = [d[3] for d in firsts.values()]
        t = time.perf_counter()
        res = Runner(vx=False, batch=16).run_many(model, streams, block=WIN)
        neural_pred = [r.outputs[-1][1] for r in res]
        out["neural"] = {"windows": len(streams), "exact_tokens": all(r.exact_tokens for r in res),
                         "outputs_equal_integer_model": all(r.exact_outputs for r in res),
                         "accuracy": float(np.mean([p == d[2] for p, d in zip(neural_pred, firsts.values())])),
                         "tokens": res[0].tokens, "neural_seconds": res[0].neural_seconds,
                         "ms_per_window": 1000 * res[0].neural_seconds / len(streams), "wall_seconds": time.perf_counter() - t}
        assert out["neural"]["exact_tokens"] and out["neural"]["outputs_equal_integer_model"]
    (ROOT / "v15/configs/tech-classifier.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("w1", "b1", "w2", "b2")}, indent=2))


if __name__ == "__main__":
    main()

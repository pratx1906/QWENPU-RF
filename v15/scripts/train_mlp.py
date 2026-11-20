"""Train the SpectralMLP on synthetic frames, quantize to Q12 codes, evaluate the exact integer model,
then run a held-out subset on the CPU in weights (every token verified).

Classes: 0 = receiver noise, 1 = narrowband tone + noise, 2 = wideband burst. Synthetic task only;
it demonstrates a learned model executing exactly, not field classification accuracy.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15")]
from rfpu_v15.cpu.infer import SpectralMLP  # noqa: E402

FSHIFT, H, C = 16, 8, 3


def frames(rng, count):
    labels = rng.integers(0, C, count)
    x = np.zeros((count, 4), dtype=complex)
    amp = rng.uniform(.4, 1., count) * 12000
    for i, y in enumerate(labels):
        noise = (rng.normal(size=4) + 1j * rng.normal(size=4)) * rng.uniform(100, 600)
        if y == 1:
            f, ph = rng.uniform(-np.pi, np.pi), rng.uniform(0, 2 * np.pi)
            x[i] = amp[i] * np.exp(1j * (f * np.arange(4) + ph)) + noise
        elif y == 2:
            x[i] = (rng.normal(size=4) + 1j * rng.normal(size=4)) * amp[i] / 2 + noise
        else:
            x[i] = noise
    q = np.clip(np.round(np.stack((x.real, x.imag), -1)), -32768, 32767).astype(int)
    return [[tuple(map(int, s)) for s in fr] for fr in q], labels


def main():
    rng = np.random.default_rng(1505)
    probe = SpectralMLP([[0] * 4] * H, [0] * H, [[0] * H] * C, [0] * C, FSHIFT)
    train_x, train_y = frames(rng, 6000)
    test_x, test_y = frames(rng, 2000)
    U = np.array([probe.features(f) for f in train_x], float) / 32768
    A = rng.normal(scale=.5, size=(H, 4)); a = np.zeros(H); B = rng.normal(scale=.5, size=(C, H)); b = np.zeros(C)
    onehot = np.eye(C)[train_y]
    for step in range(4000):  # full-batch gradient descent, softmax cross-entropy
        z = U @ A.T + a; h = np.maximum(z, 0); s = h @ B.T + b
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        g = (p - onehot) / len(U)
        gB, gb = g.T @ h, g.sum(0); gh = g @ B * (z > 0)
        gA, ga = gh.T @ U, gh.sum(0)
        lr = 2.
        A -= lr * gA; a -= lr * ga; B -= lr * gB; b -= lr * gb
        A, B = np.clip(A, -60, 60), np.clip(B, -7.9, 7.9)
    w1 = np.clip(np.round(A * 512), -32768, 32767).astype(int).tolist()
    b1 = np.clip(np.round(a * 4096), -32768, 32767).astype(int).tolist()
    w2 = np.clip(np.round(B * 4096), -32768, 32767).astype(int).tolist()
    b2 = np.clip(np.round(b * 4096), -32768, 32767).astype(int).tolist()
    model = SpectralMLP(w1, b1, w2, b2, FSHIFT)
    float_test = np.array([probe.features(f) for f in test_x], float) / 32768
    float_pred = (np.maximum(float_test @ A.T + a, 0) @ B.T + b).argmax(1)
    int_pred = np.array([model.forward(model.features(f))[1] for f in test_x])
    out = {"classes": ["noise", "tone", "wideband"], "fshift": FSHIFT, "w1": w1, "b1": b1, "w2": w2, "b2": b2,
           "train_frames": len(train_x), "test_frames": len(test_x),
           "float_test_accuracy": float((float_pred == test_y).mean()),
           "integer_test_accuracy": float((int_pred == test_y).mean()),
           "scope": "Synthetic 4-sample frames; demonstrates exact learned inference, not field accuracy."}
    neural = "--neural" in sys.argv
    if neural:
        from rfpu_v15.cpu.runner import Runner
        n = 512
        stream = [s for f in test_x[:n] for s in f]
        r = Runner(vx=False, batch=16).run(model, stream, block=256)
        preds = [v[1] for v in r.outputs if v is not None]
        out["neural"] = {"frames": n, "exact_tokens": r.exact_tokens, "exact_outputs": r.exact_outputs,
                         "accuracy": float((np.array(preds) == test_y[:n]).mean()), "tokens": r.tokens,
                         "neural_seconds": r.neural_seconds, "frames_per_second": n / r.neural_seconds}
        assert r.exact_tokens and r.exact_outputs
    (ROOT / "v15/configs").mkdir(exist_ok=True)
    (ROOT / "v15/configs/spectral-mlp.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("w1", "b1", "w2", "b2")}, indent=2))


if __name__ == "__main__":
    main()

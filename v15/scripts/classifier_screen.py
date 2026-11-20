"""PROTOCOL Amendment 6, P6.4 (exploratory): screen OFDM cyclic-prefix lag-correlation features for the
real-data technology classifier. Selection by leave-one-location-out cross-validation on the calibration
(training) recordings only; the held-out recordings are evaluated once, for the selected feature set.
Host float screening (not an in-weights result). Writes v15/reports/classifier-screen.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "v15"), str(ROOT / "v15/scripts")]
from rfpu_v15.cpu.classify import TechClassifier  # noqa: E402
from rfpu_v15.paths import local  # noqa: E402
from train_tech_classifier import CLASSES, recordings, windows  # noqa: E402

LAGS = [64, 67, 128, 224, 448]
H, FRAMES, L2 = 8, 64, 1e-3  # the recorded model's settings (configs/tech-classifier.json)


def lag_features(w):
    x = np.array([complex(a, b) for a, b in w])
    r0 = float(np.sum(np.abs(x) ** 2)) or 1.0
    return [4 * abs(np.sum(x[L:] * np.conj(x[:-L]))) / r0 for L in LAGS]


def train(X, y, seed=1, integer=False, bandwidth=False):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    A = rng.normal(scale=.3, size=(H, d)); a = np.zeros(H); B = rng.normal(scale=.3, size=(3, H)); b = np.zeros(3)
    w_cls = len(y) / (3 * np.maximum(np.bincount(y, minlength=3), 1))
    Y, sw = np.eye(3)[y], w_cls[y][:, None]
    m = [np.zeros_like(v) for v in (A, a, B, b)]; v2 = [np.zeros_like(v) for v in (A, a, B, b)]
    for step in range(1, 6001):
        z = X @ A.T + a; h = np.maximum(z, 0); s = h @ B.T + b
        p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        g = (p - Y) * sw / len(X)
        gh = g @ B * (z > 0)
        grads = [gh.T @ X + L2 * A, gh.sum(0), g.T @ h + L2 * B, g.sum(0)]
        for i, (prm, gr) in enumerate(zip((A, a, B, b), grads)):
            m[i] = .9 * m[i] + .1 * gr; v2[i] = .999 * v2[i] + .001 * gr * gr
            prm -= .01 * (m[i] / (1 - .9 ** step)) / (np.sqrt(v2[i] / (1 - .999 ** step)) + 1e-8)
        np.clip(A, -7.9, 7.9, out=A); np.clip(B, -7.9, 7.9, out=B)
    fl = lambda Z: (np.maximum(Z @ A.T + a, 0) @ B.T + b).argmax(1)  # noqa: E731
    if not integer:
        return fl
    q = lambda M, s: np.clip(np.round(M * s), -32768, 32767).astype(int).tolist()  # noqa: E731
    return fl, TechClassifier(q(A, 4096), q(a, 256), q(B, 4096), q(b, 256), frames=FRAMES, bandwidth=bandwidth)


def scores(pred, y, rec):
    by = {}
    for p, t, r in zip(pred, y, rec):
        by.setdefault(r, [t, []])[1].append(p)
    return float(np.mean(pred == y)), float(np.mean([np.bincount(p, minlength=3).argmax() == t for t, p in by.values()]))


def lolo6(rows):
    """P6.4b: leave-one-location-out over all six locations; every variant reported."""
    allr = rows["train"] + rows["test"]
    locs = sorted({r[1] for r in allr})
    global H
    out = {}
    for feats in ("baseline", "baseline+lags"):
        f = (lambda r: r[3]) if feats == "baseline" else (lambda r: np.concatenate((r[3], r[4])))
        for h in (8, 16):
            H = h
            preds, ys, recs, per_loc = [], [], [], {}
            for loc in locs:
                fit = [r for r in allr if r[1] != loc]; ev = [r for r in allr if r[1] == loc]
                model = train(np.array([f(r) for r in fit]), np.array([r[2] for r in fit]))
                p = model(np.array([f(r) for r in ev]))
                preds += list(p); ys += [r[2] for r in ev]; recs += [r[0] for r in ev]
                per_loc[loc] = scores(p, np.array([r[2] for r in ev]), [r[0] for r in ev])[1]
            w_acc, r_acc = scores(np.array(preds), np.array(ys), recs)
            yt, pr = np.array(ys), np.array(preds)
            key = f"{feats}, H={h}"
            out[key] = {"window_accuracy": w_acc, "recording_accuracy": r_acc, "recording_accuracy_by_location": per_loc,
                        "confusion_rows_true_cols_pred": [[int(np.sum((yt == i) & (pr == j))) for j in range(3)] for i in range(3)]}
            print(key, {k: v for k, v in out[key].items() if k != "confusion_rows_true_cols_pred"}, flush=True)
    H = 8
    return {"protocol": "PROTOCOL.md Amendment 6, P6.4b (host float, leave-one-location-out over 6 locations)",
            "locations": locs, "variants": out, "primary": "baseline+lags, H=8",
            "primary_beats_baseline": out["baseline+lags, H=8"]["recording_accuracy"] > out["baseline, H=8"]["recording_accuracy"]}


def bandwidth_features(w):
    """P8.2: occupied-bandwidth shape: FFT64 power (Hann, hop 64; the in-weights FFTPower integer
    reference) averaged over the window's 16 frames, log2 in Q8 (classify.log2_q8), mean removed."""
    from rfpu_v15.cpu.classify import log2_q8
    from rfpu_v15.cpu.spectral import FFTPower
    out = [o for o in FFTPower(64, None, 64).reference(w) if o is not None]
    avg = [sum(f[k] for f in out) // len(out) for k in range(64)]
    lg = [log2_q8(v >> 6) for v in avg]
    m = sum(lg) // 64
    return [v - m for v in lg]


def ten_msps():
    """P6.4c: 10 MS/s recordings, leave-one-recording-out (host float)."""
    import json as _j
    d = _j.loads((ROOT / "data/idlab-iq-manifest.json").read_text())
    items = [i for i in next(v for v in d.values() if isinstance(v, list))
             if i["sample_rate"] == 1e7 and i["technology"] in CLASSES and local(i["path"]).exists()]
    probe = TechClassifier([[0] * 17] * H, [0] * H, [[0] * H] * 3, [0] * 3, frames=FRAMES)
    # recordings keyed by path (two share a file name across locations)
    bw = "--bandwidth" in sys.argv
    rows = [(i["path"], i["location"], CLASSES.index(i["technology"]),
             np.concatenate((np.array(probe.features(w), float) / 256,
                             np.array(bandwidth_features(w), float) / 256 if bw else [])), w)
            for i in items for w in windows(i)]
    assert len({i["path"] for i in items}) == len(items), "recordings must be keyed uniquely"
    names = sorted({r[0] for r in rows})
    preds, ipreds, ys, recs = [], [], [], []
    for nm in names:
        fit = [r for r in rows if r[0] != nm]; ev = [r for r in rows if r[0] == nm]
        model, kern = train(np.array([r[3] for r in fit]), np.array([r[2] for r in fit]), integer=True, bandwidth=bw)
        preds += list(model(np.array([r[3] for r in ev]))); ys += [r[2] for r in ev]; recs += [nm] * len(ev)
        # the in-weights integer model (bandwidth: its features from the FFT64 integer reference, which
        # stage A reproduces exactly in weights)
        ipreds += [kern.forward(list(probe.features(r[4])) + (list(bandwidth_features(r[4])) if bw else []))[1] for r in ev]
    yt, pr, ip = np.array(ys), np.array(preds), np.array(ipreds)
    w_acc, r_acc = scores(pr, yt, recs)
    iw_acc, ir_acc = scores(ip, yt, recs)
    final, fk = train(np.array([r[3] for r in rows]), np.array([r[2] for r in rows]), integer=True, bandwidth=bw)
    firsts = {}
    for r in rows:
        firsts.setdefault(r[0], r)
    neural = None
    if "--neural" in sys.argv and bw:
        # two stages in weights per window: FFT64 power (stage A), its verified powers copied by the host
        # into the classifier (stage B); every token of both checked
        from rfpu_v15.cpu.runner import Runner
        from rfpu_v15.cpu.spectral import FFTPower
        run_ = Runner(vx=False, batch=1)
        ok_t = ok_o = True
        agree = 0
        for r in firsts.values():
            a_ = run_.run(FFTPower(64, None, 64), r[4], block=1024)
            fk.set_fft(0, [o for o in a_.outputs if o is not None])
            b_ = run_.run(fk, r[4], block=1024)
            ok_t &= a_.exact_tokens and b_.exact_tokens
            ok_o &= a_.exact_outputs and b_.exact_outputs
            agree += b_.outputs[-1][1] == fk.forward(list(probe.features(r[4])) + list(bandwidth_features(r[4])))[1]
        neural = {"windows": len(firsts), "stages": "FFT64 power (A) -> host copy of verified words -> classifier (B)",
                  "exact_tokens": ok_t, "outputs_equal_references": ok_o, "predictions_equal_integer_model": agree}
        assert ok_t and ok_o and agree == len(firsts)
    elif "--neural" in sys.argv:
        from rfpu_v15.cpu.runner import Runner
        res = Runner(vx=False, batch=16).run_many(fk, [r[4] for r in firsts.values()], block=1024)
        neural = {"windows": len(res), "exact_tokens": all(r_.exact_tokens for r_ in res),
                  "outputs_equal_integer_model": all(r_.exact_outputs for r_ in res), "tokens": res[0].tokens}
        assert neural["exact_tokens"] and neural["outputs_equal_integer_model"]
    return {"protocol": "PROTOCOL.md Amendment 6, P6.4c (host float, 10 MS/s, leave-one-recording-out)" if not bw
            else "PROTOCOL.md Amendment 8, P8.2 (10 MS/s + FFT64 bandwidth shape, leave-one-recording-out)",
            "recordings": len(names), "windows": len(rows),
            "by_class_recordings": {c: sum(1 for i in items if i["technology"] == c) for c in CLASSES},
            "float": {"window_accuracy": w_acc, "recording_accuracy": r_acc,
                      "confusion_rows_true_cols_pred": [[int(np.sum((yt == i) & (pr == j))) for j in range(3)] for i in range(3)]},
            "integer_in_weights_model": {"window_accuracy": iw_acc, "recording_accuracy": ir_acc,
                                         "confusion_rows_true_cols_pred": [[int(np.sum((yt == i) & (ip == j))) for j in range(3)] for i in range(3)]},
            "beats_1msps_lolo6": r_acc > 0.744, "neural_final_model_one_window_per_recording": neural,
            "bandwidth_features": bw, "baseline_recording_accuracy_P6.4c": 0.846 if bw else None,
            "note": "first run keyed recordings by file name (25 units: two recordings share a name); rerun keyed by path"}


def main():
    if "--10msps" in sys.argv:
        res = ten_msps()
        (ROOT / ("v15/reports/classifier-10msps-bandwidth.json" if "--bandwidth" in sys.argv else "v15/reports/classifier-10msps.json")).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps(res, indent=1))
        return
    probe = TechClassifier([[0] * 17] * H, [0] * H, [[0] * H] * 3, [0] * 3, frames=FRAMES)
    rows = {"train": [], "test": []}
    for item in recordings():
        split = "train" if item["split"] == "calibration" else "test"
        y = CLASSES.index(item["technology"])
        for w in windows(item):
            rows[split].append((item["name"], item["location"], y, np.array(probe.features(w), float) / 256,
                                np.array(lag_features(w))))
    print({k: len(v) for k, v in rows.items()}, flush=True)
    if "--lolo6" in sys.argv:
        res = lolo6(rows)
        (ROOT / "v15/reports/classifier-lolo6.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "variants"}, indent=1))
        return
    sets = {"baseline": lambda r: r[3], "baseline+lags": lambda r: np.concatenate((r[3], r[4]))}
    tr = rows["train"]
    locs = sorted({r[1] for r in tr})
    cv = {}
    for name, f in sets.items():
        preds, ys, recs = [], [], []
        for loc in locs:
            fit = [r for r in tr if r[1] != loc]; ev = [r for r in tr if r[1] == loc]
            model = train(np.array([f(r) for r in fit]), np.array([r[2] for r in fit]))
            preds += list(model(np.array([f(r) for r in ev]))); ys += [r[2] for r in ev]; recs += [r[0] for r in ev]
        w_acc, r_acc = scores(np.array(preds), np.array(ys), recs)
        cv[name] = {"window_accuracy": w_acc, "recording_accuracy": r_acc}
        print(name, cv[name], flush=True)
    chosen = max(cv, key=lambda k: (cv[k]["recording_accuracy"], k == "baseline"))
    f = sets[chosen]
    model = train(np.array([f(r) for r in tr]), np.array([r[2] for r in tr]))
    te = rows["test"]
    pred = model(np.array([f(r) for r in te])); yt = np.array([r[2] for r in te])
    w_acc, r_acc = scores(pred, yt, [r[0] for r in te])
    conf = [[int(np.sum((yt == i) & (pred == j))) for j in range(3)] for i in range(3)]
    out = {"protocol": "PROTOCOL.md Amendment 6, P6.4 (host float screening)", "lags": LAGS,
           "cv_leave_one_location_out_calibration": cv, "cv_locations": locs, "chosen": chosen,
           "held_out_once": {"window_accuracy": w_acc, "recording_accuracy": r_acc, "confusion_rows_true_cols_pred": conf},
           "recorded_in_weights_baseline": {"window_accuracy": 0.670, "recording_accuracy": 0.703},
           "improves_over_recorded": r_acc > 0.703}
    (ROOT / "v15/reports/classifier-screen.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

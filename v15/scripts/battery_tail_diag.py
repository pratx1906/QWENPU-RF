import json, math, random, sys
import numpy as np
from pathlib import Path
ROOT = Path(r".")
sys.path[:0] = [str(ROOT), str(ROOT/"v15"), str(ROOT/"v15/scripts")]
sys.argv = [sys.argv[0], "--headroom"]
import real_iq_q15 as R
from scipy.signal import lfilter
from rfpu_v15.cpu.filters import IIRBiquad
from rfpu_v15.cpu.vfir import VXFir
N = 256
rows = json.loads((ROOT/"v15/reports/real-iq-selection-expanded.json").read_text(encoding="utf-8"))["captures"]
bat = json.loads((ROOT/"v15/reports/kernel-battery.json").read_text(encoding="utf-8"))
rng = random.Random(4); vtaps = [(rng.randint(-5999, 5999), rng.randint(-5999, 5999)) for _ in range(16)]
b = (4096, 8192, 4096, -16384, 4096)
iir, vf = IIRBiquad(*b, wide=True), VXFir(vtaps, packed=False, guard=True)
H = np.array([complex(*t) for t in vtaps]) / 32768
out = []
for i, r in enumerate(rows):
    _, q, _ = R.inputs(r); q = q[:N]
    X = np.array([complex(*v) for v in q])
    rec = {"name": bat["names"][i], "in_pow": float(np.mean(abs(X)**2)), "in_max": int(max(max(abs(a), abs(c)) for a, c in q))}
    for tag, k, fl in (("iir", iir, lfilter(np.array(b[:3])/16384, [1, b[3]/16384, b[4]/16384], X)),
                       ("vfir", vf, np.convolve(X, H)[:N])):
        y = np.array([complex(*v) for v in k.reference(q)])
        e = y - fl
        rec[tag] = {"sqnr": 10*math.log10(np.sum(abs(fl)**2)/np.sum(abs(e)**2)), "out_pow": float(np.mean(abs(fl)**2)),
                    "err_pow": float(np.mean(abs(e)**2)), "gain_db": 10*math.log10(np.mean(abs(fl)**2)/rec["in_pow"]),
                    "out_max": float(max(np.max(abs(fl.real)), np.max(abs(fl.imag)))),
                    "sat": int(sum(abs(v) >= 32767 for z in k.reference(q) for v in z)),
                    "err_max": float(np.max(abs(e)))}
    out.append(rec)
json.dump(out, open(Path(sys.argv[0]).with_name("tail_diag.json"), "w"), indent=1)
for tag in ("iir", "vfir"):
    o = sorted(out, key=lambda r: r[tag]["sqnr"])
    print(f"== {tag}: worst 5, median, best")
    for r in o[:5] + [o[len(o)//2], o[-1]]:
        t = r[tag]
        print(f"{r['name'][:52]:52s} sqnr {t['sqnr']:6.1f}  inPow {r['in_pow']:.3g}  gain {t['gain_db']:6.1f} dB  outPow {t['out_pow']:.3g}  errPow {t['err_pow']:.3f}  sat {t['sat']}  inMax {r['in_max']}  outMax {t['out_max']:.0f}")
    s = np.array([r[tag]["sqnr"] for r in out]); op = np.log10([r[tag]["out_pow"] for r in out]); ep = np.array([r[tag]["err_pow"] for r in out])
    print("corr(sqnr, log out_pow)", round(np.corrcoef(s, op)[0, 1], 3), " err_pow median/min/max", np.median(ep), ep.min(), ep.max(), " total sat", sum(r[tag]["sat"] for r in out))
    # check against battery numbers
    bv = next(x for x in bat["kernels"] if x["kernel"].startswith("IIR" if tag == "iir" else "VFIR"))["sqnr_db"]["per_recording"]
    print("max |diff vs battery|", max(abs(a - r[tag]["sqnr"]) for a, r in zip(bv, out)))

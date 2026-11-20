import json, math, random, sys
import numpy as np
from pathlib import Path
ROOT = Path(r"."); sys.path[:0] = [str(ROOT), str(ROOT/"v15"), str(ROOT/"v15/scripts")]
sys.argv = [sys.argv[0], "--headroom"]
import real_iq_q15 as R
from scipy.signal import lfilter
from rfpu_v15.cpu.filters import IIRBiquad
from rfpu_v15.cpu.vfir import VXFir
rows = json.loads((ROOT/"v15/reports/real-iq-selection-expanded.json").read_text(encoding="utf-8"))["captures"]
rng = random.Random(4); vt = [(rng.randint(-5999, 5999), rng.randint(-5999, 5999)) for _ in range(16)]
b = (4096, 8192, 4096, -16384, 4096); H = np.array([complex(*t) for t in vt]) / 32768
sq = lambda y, f: 10*math.log10(np.sum(abs(f)**2)/np.sum(abs(y-f)**2))
for name in ("DJI_matrice_210_5G_2of2.bin", "DJI_phantom_4_pro_plus_5G_2of2.bin", "wf_g76_gentbrugge_f5240MHz_r4.bin"):
    r = next(r for r in rows if r["path"].endswith(name)); _, q, _ = R.inputs(r); q = q[:256]
    X = np.array([complex(*v) for v in q])
    fl = lfilter(np.array(b[:3])/16384, [1, b[3]/16384, b[4]/16384], X)
    res = [name[:28]]
    for f in (0, 4):
        y = np.array([complex(*v) for v in IIRBiquad(*b, wide=True, out_frac=f).reference(q)]) / 2**f
        res.append(f"iir f{f} {sq(y, fl):.1f}")
    fv = np.convolve(X, H)[:256]
    g = VXFir(vt, packed=False, guard=True).reference(q); m = max(max(abs(a), abs(c)) for a, c in g)
    s = max([k for k in (0, 1, 2) if (m + 1) * 2**k + 16 <= 32767])
    y = VXFir(vt, packed=False, guard=s == 0, scale=s).reference(q)
    sat = sum(abs(v) >= 32767 for z in y for v in z)
    res.append(f"vfir guard {sq(np.array([complex(*v) for v in g]), fv):.1f} -> s{s} {sq(np.array([complex(*v) for v in y])/2**s, fv):.1f} sat {sat} max {m}")
    print(" | ".join(res))

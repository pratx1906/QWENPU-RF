"""Regenerate the README graphics from the recorded evidence (QwenPU / Qwenputer palette).

    python docs/make_figures.py                 # charts + diagram (needs only this repo)
    python docs/make_figures.py --banner        # also the banner (needs one IDLab capture under $RFPU_DATA)

Every number drawn comes from v15/reports/*.json, data/real-iq-selection.json or the recorded Stage 2 run
(token strings in the banner strip are the real trace of the rf_levels demo, 2026-11-20).
"""
import json
import sys
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
REP = ROOT / "v15/reports"
sys.path[:0] = [str(ROOT / "v15")]

BG, GRID, INK, SUB, MUTE = "#fbfaf7", "#f0eee8", "#1b1b1b", "#3f3f3b", "#8f8f8a"
TEAL, TEAL_D, BLUE, CORAL, PALE = "#0d9488", "#0b7a70", "#2060b0", "#d9534f", "#e7ecf3"
SANS, MONO = "Arial, Helvetica, sans-serif", "Consolas, Menlo, monospace"


def load(name):
    return json.loads((REP / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- charts (matplotlib, SVG)
def charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 12,
        "axes.edgecolor": "#b9b7b0", "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG,
        "svg.fonttype": "none", "axes.titleweight": "bold", "axes.titlesize": 14, "axes.titlecolor": INK,
        "axes.titlelocation": "left", "axes.axisbelow": True})

    # 1. real IQ: design ladder + expanded screen per dataset
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"wspace": 0.55})
    steps = [("v1.4 (4-bit processor)", 0), ("Q15, fixed gain", 8), ("Q2.13 headroom", 13), ("Block floating point\nin the weights", 14)]
    y = np.arange(len(steps))[::-1]
    a.barh(y, [s[1] for s in steps], color=[CORAL, "#9cc7c2", "#5fb3aa", TEAL], height=0.62)
    for yi, (_, v) in zip(y, steps):
        a.text(v + 0.25, yi, f"{v}/14", va="center", color=INK, fontsize=12, fontweight="bold" if v == 14 else None)
    a.set_yticks(y, [s[0] for s in steps]); a.set_xlim(0, 16); a.set_xticks([0, 7, 14])
    a.set_title("Same 14 real captures, pre-declared criteria")
    a.grid(axis="x", color=GRID)
    ds = [("IDLab Wi-Fi / LTE / DVB-T", 53, 53), ("EPFL LoRa", 32, 32), ("NTU LoRa (over the air)", 27, 27),
          ("Tampere drone controllers", 18, 19), ("CAMRAS radio astronomy", 2, 2)]
    y = np.arange(len(ds))[::-1]
    b.barh(y, [p / n * 100 for _, p, n in ds], color=[TEAL if p == n else BLUE for _, p, n in ds], height=0.62)
    for yi, (_, p, n) in zip(y, ds):
        b.text(p / n * 100 + 1.5, yi, f"{p}/{n}", va="center", color=INK)
    b.set_yticks(y, [d[0] for d in ds]); b.set_xlim(0, 118); b.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
    b.set_title("Expanded screen: 132/133 recordings, 5 datasets")
    b.grid(axis="x", color=GRID)
    fig.savefig(OUT / "fig-real-iq.svg", bbox_inches="tight"); plt.close(fig)

    # 2. precision: kernel battery on 133 recordings (median and minimum SQNR)
    kb = [("CORDIC", 150, 143), ("IIR biquad\n(out_frac=4)", 102.6, 80.0), ("NCO mixer", 81.1, 74.1), ("FFT64, block FP", 75.7, 72.7),
          ("VFIR (guard bit)", 73.4, 65.4), ("VBFLY FFT64", 71.5, 68.6)]
    fig, ax = plt.subplots(figsize=(13, 3.9))
    x = np.arange(len(kb))
    ax.bar(x - 0.19, [k[1] for k in kb], 0.36, color=TEAL, label="median")
    ax.bar(x + 0.19, [k[2] for k in kb], 0.36, color="#9cc7c2", label="minimum")
    for xi, (_, med, mn) in zip(x, kb):
        ax.text(xi - 0.19, med + 2, f"{med:g}", ha="center", fontsize=10, color=INK)
        ax.text(xi + 0.19, mn + 2, f"{mn:g}", ha="center", fontsize=10, color=SUB)
    ax.axhline(70, color=CORAL, lw=1.2, ls=(0, (4, 3)))
    ax.text(-0.55, 73, "70 dB", color=CORAL, ha="left", fontsize=10)
    ax.set_xticks(x, [k[0] for k in kb]); ax.set_ylabel("SQNR (dB)"); ax.set_ylim(0, 165)
    ax.set_title("Precision on 133 real recordings, every token exact (selected kernels)")
    ax.legend(frameon=False, loc="upper right"); ax.grid(axis="y", color=GRID)
    fig.savefig(OUT / "fig-precision.svg", bbox_inches="tight"); plt.close(fig)

    # 3. modulation classifier (Amendment 15)
    m = load("modclass-neural.json")
    snr = sorted(m["accuracy_by_snr"], key=float)
    cls = m["correct_per_class"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 2.3], "wspace": 0.18})
    a.plot([float(s) for s in snr], [m["accuracy_by_snr"][s] * 100 for s in snr], color=TEAL, lw=2.4, marker="o")
    a.axhline(4, color=MUTE, lw=1, ls=":"); a.text(27.5, 6, "chance 4%", color=MUTE, ha="right", fontsize=10)
    a.set_ylim(0, 105); a.set_xlabel("SNR (dB)"); a.set_ylabel("accuracy (%)"); a.grid(color=GRID)
    a.set_title(f"25 classes: {m['accuracy'] * 100:.1f}% overall")
    names = list(cls)
    b.bar(range(len(names)), [cls[n] for n in names], color=[TEAL if cls[n] >= 9 else BLUE if cls[n] >= 7 else CORAL for n in names])
    b.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=9.5)
    b.set_ylim(0, 10.8); b.set_yticks([0, 5, 7, 10]); b.grid(axis="y", color=GRID)
    b.set_title("Correct out of 10 per class (250/250 runs token-exact in the weights)")
    fig.savefig(OUT / "fig-modulation.svg", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- shared SVG bits
def chip(x, y, text, kind, size=17, h=34, op=1.0):
    """A token chip: 'cpu' = teal filled (computer token), 'txt' = outlined (Qwen text token)."""
    w = int(len(text) * size * 0.56 + 22)
    if kind == "cpu":
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{TEAL}" fill-opacity="{op:.2f}"/>'
        c = "#ffffff"
    else:
        r = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="#dcdad3" stroke-opacity="{op:.2f}"/>'
        c = INK if op > 0.5 else "#b9b7b0"
    t = (f'<text x="{x + w / 2:.1f}" y="{y + h * 0.68:.1f}" text-anchor="middle" font-family="{MONO}" '
         f'font-size="{size}" fill="{c}">{escape(text)}</text>')
    return r + t, w


def strip(y, items, x0=34, gap=8, xmax=1570):
    out, x = [], x0
    for text, kind in items:
        s, w = chip(x, y, text, kind)
        if x + w > xmax:
            break
        out.append(s); x += w + gap
    return "".join(out)


# ---------------------------------------------------------------- banner (1600 x 500)
def banner():
    sys.path.insert(0, str(ROOT / "v15/scripts"))
    from rfpu_v15.paths import local
    sel = json.loads((ROOT / "data/real-iq-selection.json").read_text(encoding="utf-8"))["captures"]
    row = next(r for r in sel if r["dataset"] == "IDLab" and r["emitter"] == "LTE")
    raw = np.memmap(local(row["path"]), dtype="<i2", mode="r").reshape(-1, 2)
    z = (raw[4096:4096 + 4096, 0] + 1j * raw[4096:4096 + 4096, 1]).astype(complex) / 32768
    w = z[:120]
    spec = np.abs(np.fft.fftshift(np.fft.fft(z[:64 * 64].reshape(64, 64), axis=1), axes=1)) ** 2
    p = 10 * np.log10(spec.mean(0) + 1e-12)
    p = (p - p.min()) / (p.max() - p.min())

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="500" viewBox="0 0 1600 500" role="img" aria-labelledby="title desc">',
         '<title id="title">QwenPU-RF</title>',
         '<desc id="desc">A radio signal processor, compiled into the weights of a language model. Real LTE IQ samples and '
         'their 64-bin power spectrum; the strip is the real token trace of Qwen calling its in-weights RF tool.</desc>',
         f'<defs><pattern id="grid" width="36" height="36" patternUnits="userSpaceOnUse"><path d="M36 0H0V36" fill="none" stroke="{GRID}"/></pattern>'
         f'<linearGradient id="fade" x1="0" x2="1"><stop offset="0" stop-color="{BG}"/><stop offset="0.06" stop-color="{BG}" stop-opacity="0"/>'
         f'<stop offset="0.94" stop-color="{BG}" stop-opacity="0"/><stop offset="1" stop-color="{BG}"/></linearGradient></defs>',
         f'<rect width="1600" height="500" fill="{BG}"/><rect width="1600" height="500" fill="url(#grid)"/>',
         f'<g font-family="{SANS}"><text x="84" y="196" font-size="126" fill="{INK}" letter-spacing="-4">Qwen<tspan fill="{TEAL}">PU</tspan>'
         f'<tspan fill="{CORAL}">-RF</tspan></text>'
         f'<text x="90" y="268" font-size="38" fill="{SUB}">A radio signal processor, compiled</text>'
         f'<text x="90" y="318" font-size="38" fill="{SUB}">into the weights of a language model.</text></g>']
    # IQ waveform
    X0, X1, YC, A = 900, 1500, 128, 46
    xs = np.linspace(X0, X1, len(w))
    sc = A / max(1e-9, np.abs(np.r_[w.real, w.imag]).max())
    s.append(f'<text x="{X0}" y="62" font-family="{MONO}" font-size="17" fill="{SUB}">real LTE capture · I/Q · {row['metadata']['sample_rate'] / 1e6:g} MS/s</text>')
    s.append(f'<line x1="{X0}" y1="{YC}" x2="{X1}" y2="{YC}" stroke="#dcdad3"/>')
    for comp, col in ((w.real, TEAL), (w.imag, BLUE)):
        pts = " ".join(f"{x:.1f},{YC - v * sc:.1f}" for x, v in zip(xs, comp))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8" stroke-opacity="0.9"/>')
    # FFT64 power bars, VMAT-style cells
    s.append(f'<text x="{X0}" y="212" font-family="{MONO}" font-size="17" fill="{SUB}">FFT64 power · 64 bins</text>')
    k_pk = int(np.argmax(p))
    bw, base = (X1 - X0) / 64, 340
    for k, v in enumerate(p):
        hgt = 8 + v * 104
        col = TEAL if v > 0.55 else BLUE
        s.append(f'<rect x="{X0 + k * bw + 1:.1f}" y="{base - hgt:.1f}" width="{bw - 2.2:.1f}" height="{hgt:.1f}" rx="2" '
                 f'fill="{col}" fill-opacity="{0.25 + 0.7 * v:.2f}"/>')
    xk = X0 + k_pk * bw
    s.append(f'<rect x="{xk - 2:.1f}" y="{base - 118}" width="{bw + 2.2:.1f}" height="122" rx="4" fill="none" stroke="{CORAL}" stroke-width="2"/>')
    s.append(f'<text x="{X0}" y="372" font-family="{MONO}" font-size="17" fill="{SUB}">exact, inside Qwen3.5-4B, every token checked</text>')
    # token strip: the real rf_levels trace (2026-11-20 run)
    items = [("</tool_call>", "txt"), ("GO", "cpu"), ("V(12)", "cpu"), ("F1(8)", "cpu"), ("J(6)", "cpu"), ("V(32752)", "cpu"),
             ("F1(0)", "cpu"), ("ST1", "cpu"), ("…", "txt"), ("F0(93)", "cpu"), ("HALT", "cpu"), ("noise", "txt"),
             ("floor", "txt"), ("-48.9", "txt"), ("dBFS", "txt")]
    s.append(strip(428, items))
    s.append('<rect x="0" y="418" width="1600" height="56" fill="url(#fade)"/>')
    s.append("</svg>")
    (OUT / "banner.svg").write_text("\n".join(s), encoding="utf-8")


# ---------------------------------------------------------------- how it works (1600 x 640)
def diagram():
    v = json.loads((ROOT / "docs/demo-run.json").read_text(encoding="utf-8"))
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="640" viewBox="0 0 1600 640" role="img" aria-labelledby="t d">',
         '<title id="t">How QwenPU-RF answers a radio question</title>',
         '<desc id="d">Qwen reads the question and emits a tool call; the RF program then runs as CPU trace tokens through the 11 '
         'compiled layers, each token checked against the RV32 reference machine; the printed report flows back into Qwen, which answers.</desc>',
         f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0 0L10 5L0 10z" fill="{MUTE}"/></marker>'
         f'<pattern id="g" width="36" height="36" patternUnits="userSpaceOnUse"><path d="M36 0H0V36" fill="none" stroke="{GRID}"/></pattern></defs>',
         f'<rect width="1600" height="640" fill="{BG}"/><rect width="1600" height="640" fill="url(#g)"/>',
         f'<text x="60" y="64" font-family="{SANS}" font-size="30" fill="{INK}">One question, one model: Qwen and the computer share the same weights</text>']

    def box(x, y, w, h, title, lines, stroke, fill="#ffffff", tcol=INK):
        o = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
             f'<text x="{x + 18}" y="{y + 34}" font-family="{SANS}" font-size="21" font-weight="bold" fill="{tcol}">{escape(title)}</text>']
        for i, ln in enumerate(lines):
            o.append(f'<text x="{x + 18}" y="{y + 64 + i * 25}" font-family="{MONO}" font-size="15.5" fill="{SUB}">{escape(ln)}</text>')
        return "".join(o)

    def arrow(x1, y1, x2, y2, label=None, dy=-10):
        o = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTE}" stroke-width="2" marker-end="url(#ar)"/>'
        if label:
            o += (f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 + dy}" text-anchor="middle" font-family="{MONO}" '
                  f'font-size="14" fill="{SUB}">{escape(label)}</text>')
        return o

    s.append(box(60, 110, 300, 150, "1 · Question", ['"What is the noise floor', ' of capture 14 in dBFS,', ' and its loudest', ' 64-sample window?"'], "#dcdad3"))
    s.append(box(430, 110, 330, 150, "2 · Qwen3.5-4B", ["32 language layers", "decides to call a tool:", "<tool_call>", " rf_levels(capture=14)"], BLUE))
    s.append(box(830, 110, 360, 150, "3 · RV32 CPU in the weights", ["11 compiled layers in front", "fetch · decode · registers ·", "ALU · RAM in Gated DeltaNet", "state; no new heads"], TEAL, "#f2faf9", TEAL_D))
    s.append(box(1260, 110, 290, 150, "4 · Reference check", ["every CPU token compared", "with the RV32 reference", f"{v['circuit_equal_emitted']:,} / {v['circuit_owned']:,} equal", f"{v['mismatch']} mismatches"], "#dcdad3"))
    s.append(arrow(360, 185, 428, 185))
    s.append(arrow(760, 185, 828, 185))
    s.append(arrow(1190, 185, 1258, 185))
    # loop: report back to Qwen
    s.append(f'<path d="M1010 262 C1010 312, 600 312, 600 266" fill="none" stroke="{MUTE}" stroke-width="2" marker-end="url(#ar)"/>')
    s.append(f'<text x="805" y="340" text-anchor="middle" font-family="{MONO}" font-size="14" fill="{SUB}">printed report (computed in the weights) returns as text</text>')
    s.append(box(430, 358, 760, 96, "5 · Answer", ["The noise floor of capture 14 is -48.9 dBFS, and its", "loudest 64-sample window is -10.4 dBFS."], CORAL))
    # token budget bar
    tot = v["text_tokens"] + v["chip_tokens"]
    x0, x1, y = 60, 1540, 520
    xt = x0 + (x1 - x0) * v["text_tokens"] / tot
    s.append(f'<text x="{x0}" y="{y - 18}" font-family="{SANS}" font-size="19" fill="{INK}">Where the tokens of this one question go '
             f'({tot:,} tokens, {v["seconds"]} s on one laptop GPU)</text>')
    s.append(f'<rect x="{x0}" y="{y}" width="{xt - x0:.1f}" height="30" rx="4" fill="{BLUE}"/>')
    s.append(f'<rect x="{xt:.1f}" y="{y}" width="{x1 - xt:.1f}" height="30" rx="4" fill="{TEAL}"/>')
    s.append(f'<text x="{x0}" y="{y + 58}" font-family="{MONO}" font-size="15" fill="{BLUE}">Qwen text: {v["text_tokens"]:,}</text>')
    s.append(f'<text x="{x1}" y="{y + 58}" text-anchor="end" font-family="{MONO}" font-size="15" fill="{TEAL_D}">'
             f'CPU instructions executed inside the weights: {v["chip_tokens"]:,} ({100 * v["chip_tokens"] / tot:.0f}%)</text>')
    s.append("</svg>")
    (OUT / "how-it-works.svg").write_text("\n".join(s), encoding="utf-8")


if __name__ == "__main__":
    charts()
    diagram()
    if "--banner" in sys.argv:
        banner()
    print("written to", OUT)

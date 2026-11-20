"""PROTOCOL Amendment 9: deterministic stratified selection of real-IQ recordings from every dataset on D:.

Writes v15/reports/real-iq-selection-expanded.json in the schema of data/real-iq-selection.json
(dataset, emitter, format, ADC_divisor, metadata{..., annotations}, path, sha256), consumed by
real_iq_q15.py --bfp --selection ... --out ...
"""
import hashlib
import sys
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v15")); from rfpu_v15.paths import DATA, local  # noqa: E402  (RFPU_DATA, see paths.py)
D = DATA / "datasets"
OUT = ROOT / "v15/reports/real-iq-selection-expanded.json"
SEED = 2026


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def row(dataset, emitter, path, fmt, meta, adc=None, start=None):
    md = dict(meta)
    if start is not None:
        md["annotations"] = [{"core:sample_start": int(start)}]
    r = {"dataset": dataset, "emitter": emitter, "format": fmt, "metadata": md, "path": str(path), "sha256": sha256(path)}
    if adc:
        r["ADC_divisor"] = adc
    return r


def main():
    rng = random.Random(SEED)
    rows = []
    # IDLab: up to 2 per (technology, sample rate, location)
    man = json.loads((ROOT / "data/idlab-iq-manifest.json").read_text())
    items = [i for i in next(v for v in man.values() if isinstance(v, list)) if local(i["path"]).exists()]
    groups = defaultdict(list)
    for i in items:
        groups[(i["technology"], i["sample_rate"], i["location"])].append(i)
    for key in sorted(groups):
        for i in rng.sample(sorted(groups[key], key=lambda x: x["path"]), min(2, len(groups[key]))):
            meta = {k: i[k] for k in ("name", "location", "technology", "sample_rate", "center_frequency_hz", "split")}
            rows.append(row("IDLab", i["technology"], i["path"], "ci16_le", meta, adc=32768))
    # EPFL LoRaIQ: one SigMF recording per (session, receiver), up to 32 groups
    eg = defaultdict(list)
    for p in sorted((D / "loraiq-20341802").rglob("*.sigmf-data")):
        eg[(p.parent.parent.name, p.parent.name)].append(p)
    keys = sorted(eg)
    for key in sorted(rng.sample(keys, min(32, len(keys)))):
        p = rng.choice(eg[key])
        g = json.loads(p.with_suffix(".sigmf-meta").read_text()).get("global", {})
        rows.append(row("EPFL-LoRaIQ", "LoRa", p, g.get("core:datatype", "cf32_le"),
                        {"session": key[0], "receiver": key[1], "sample_rate": g.get("core:sample_rate"),
                         "center_frequency_mhz": g.get("core:frequency")}))
    # NTU RFSR over-the-air: 9 experiments x 3 receiver gains (lowest, middle, highest)
    ng = defaultdict(list)
    for p in sorted((D / "rfsr-ota-multicondition").glob("*.cfile")):
        exp, rest = p.name.split("_", 1)
        gain = int(rest.split("rxg")[1].split("_")[0])
        ng[exp].append((gain, p))
    for exp in sorted(ng):
        g = sorted(ng[exp])
        for gain, p in (g[0], g[len(g) // 2], g[-1]):
            rows.append(row("NTU-RFSR", "LoRa", p, "cf32_le", {"experiment": exp, "rx_gain_db": gain, "sample_rate": 2000000.0}))
    # Tampere: every file, one segment at a seeded offset (avoid the first and last 10%)
    for p in sorted((D / "tampere-drone-rf-4264467").glob("*.bin")):
        n = p.stat().st_size // 4
        start = rng.randrange(n // 10, n - n // 10 - 5000)
        fs = 200e6 if "5G" in p.name else 120e6
        rows.append(row("Tampere", "drone-controller", p, "ci16_le", {"name": p.name, "sample_rate": fs}, adc=32768, start=start))
    # CAMRAS / Dwingeloo: both recordings
    for p in sorted((D / "camras-eve2026").glob("*.sigmf-data")):
        g = json.loads(p.with_suffix(".sigmf-meta").read_text()).get("global", {})
        rows.append(row("CAMRAS-Dwingeloo", "radio-astronomy (Venus radar echo)", p, g.get("core:datatype", "ci16_le"),
                        {"name": p.name, "sample_rate": g.get("core:sample_rate")}, adc=32768))
    v12 = {c["path"].lower() for c in json.loads((ROOT / "data/real-iq-selection.json").read_text())["captures"]}
    for r in rows:
        r["in_14_capture_screen"] = r["path"].lower() in v12
    OUT.write_text(json.dumps({"protocol": "PROTOCOL.md Amendment 9", "seed": SEED, "count": len(rows),
                               "by_dataset": {d: sum(r["dataset"] == d for r in rows) for d in sorted({r["dataset"] for r in rows})},
                               "captures": rows}, indent=1), encoding="utf-8")
    print(len(rows), {d: sum(r["dataset"] == d for r in rows) for d in sorted({r["dataset"] for r in rows})})


if __name__ == "__main__":
    main()

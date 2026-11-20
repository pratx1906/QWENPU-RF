"""PROTOCOL Amendment 12 P12.4 (descriptive): check every Hz / kHz / MHz value in Qwen's answers against
the verified report (cfo_hz / freq_hz, or cycles-per-sample x the label's sample rate, within 1%).
Usage: hz_audit.py <bank.json> [...]"""
import json
import re
import sys

UNIT = {"": 1, "k": 1e3, "M": 1e6, "G": 1e9}


def audit(path):
    runs = json.load(open(path, encoding="utf-8"))["runs"]
    ok, bad = 0, []
    for x in runs:
        ans = x["answer"].split("</tool_response>")[-1].replace("\u2212", "-")
        rep = x["report_check"]["report"] if x["report_check"] else ""
        fs = re.search(r"fs=([0-9.]+)MS/s", rep)
        truth = [float(v) for v in re.findall(r"(?:cfo_hz|freq_hz)=(-?\d+)", rep)]
        if fs:
            truth += [float(c) * float(fs.group(1)) * 1e6 for c in
                      re.findall(r"(?:cfo_cycles_per_sample|freq_cycles_per_sample)=(-?[0-9.]+)", rep)]
        for num, unit in re.findall(r"(-?[\d,]*\.?\d+)\s*(k|M|G)?Hz", ans):
            v = float(num.replace(",", "")) * UNIT[unit]
            if any(abs(v - t) <= max(.01 * abs(t), 2) for t in truth):
                ok += 1
            else:
                bad.append({"capture": x["capture"], "claim": f"{num} {unit}Hz", "report_hz": truth[:2]})
    return {"file": path, "hz_claims": ok + len(bad), "correct": ok, "wrong": bad}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(audit(p)))

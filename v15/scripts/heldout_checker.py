"""PROTOCOL Amendment 13: checker for the held-out Stage 2 set (P13.2-P13.5).

Usage: heldout_checker.py [--v3] [--set a13|a14] [transcript.json]
Writes <input>-checked.json (--v3: -checked-v3.json) next to the input and prints the gates. Fields and numbers
use checker v2 rules, or v3 (Amendment 14: list markers ignored, simple derivations and unit rescalings
of report values supported). --set a14 grades the Amendment 14 questions (P14.2-P14.5).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v15/scripts"))
from answer_checker import check, norm  # noqa: E402

DECLINE = re.compile(
    r"not (provided|reported|included|available|measured|given|possible|contain|determin|estimat|identif|specif)"
    r"|does(n't| not) (provide|report|include|measure|give|contain|tell)|cannot|can't|unable"
    r"|no (information|data|measurement)|outside|beyond|isn't (provided|reported|available|included)|not part of", re.I)


def A14_LISTED(question):
    """The first 8 Amendment 14 unanswerable questions ask for a quantity named on the not_measured line."""
    from stage2_chat import A14
    return question in [q for _, q, _, k in A14 if k == "unans"][:8]


def final_answer(text):
    a = text.split("</tool_response>")[-1]
    return norm(re.sub(r"<think>.*?</think>", "", a, flags=re.S).replace("<|im_end|>", "")).strip()


def main():
    from stage2_chat import A14, HELDOUT
    argv = sys.argv[1:]
    v3 = "--v3" in argv
    qset = argv[argv.index("--set") + 1] if "--set" in argv else "a13"
    pos = [x for i, x in enumerate(argv) if not x.startswith("--") and (i == 0 or argv[i - 1] != "--set")]
    default = "fullgpu-chunked-stage2-heldout.json" if qset == "a13" else "fullgpu-chunked-stage2-a14.json"
    src = Path(pos[0] if pos else ROOT / "v15/reports/stage2" / default)
    meta = {q: (f, kind) for _, q, f, kind in HELDOUT + A14}
    runs = json.loads(src.read_text(encoding="utf-8"))["runs"]
    rows = []
    for r in runs:
        f, kind = meta[r["question"]]
        row = check(r, f, v2=True, v3=v3) if r["tool_responses"] else {
            "capture": r["capture"], "question": r["question"], "tools_called": r["tools_called"], "tool_answered": False,
            "exact": r["reference_lock_error"] is None and not r["head_mismatch"], "seconds": r["seconds"],
            "fields": {x: {"report": None, "quoted": False} for x in f}, "numbers": 0, "unsupported": [],
            "answer": final_answer(r["answer"])}
        row.update(kind=kind, tool_called="<tool_call>" in r["answer"], chip_tokens=r["chip_tokens"],
                   declines=bool(DECLINE.search(final_answer(r["answer"]))))
        rows.append(row)
    by = {k: [r for r in rows if r["kind"] == k] for k in ("para", "notool", "unans")}
    para, fq = by["para"], [v["quoted"] for r in by["para"] for v in r["fields"].values()]
    nums, uns = sum(r["numbers"] for r in para), sum(len(r["unsupported"]) for r in para)
    gates = {
        "questions": len(rows), "expected": len(HELDOUT), "counts": {k: len(v) for k, v in by.items()},
        "P13.2_all_exact": all(r["exact"] for r in rows if r["chip_tokens"]),
        "P13.3_tool_answered_rate": sum(r["tool_answered"] for r in para) / len(para),
        "P13.3_fields_quoted_rate": sum(fq) / len(fq),
        "P13.3_numbers_supported_rate": (nums - uns) / nums if nums else None, "numbers": nums, "unsupported": uns,
        "P13.4_no_tool": sum(not r["tool_called"] for r in by["notool"]),
        "P13.5_declines": sum(r["declines"] for r in by["unans"]),
        "P13.5_unsupported_numbers": sum(len(r["unsupported"]) for r in by["unans"]),
        "tool_choice": {t: sum(t in r["tools_called"] for r in para) for t in ("rf_levels", "rf_analyze")},
        "seconds_median_para": sorted(r["seconds"] for r in para)[len(para) // 2],
        "checker": "v3" if v3 else "v2", "set": qset}
    if v3:
        der = [n for r in rows for n in r.get("supported_by_derivation", [])]
        gates["supported_by_derivation"] = len(der)
        gates["perturbed_x1.07_also_pass"] = sum(r.get("perturbed_also_pass", 0) for r in rows)
        gates["perturbed_x1.07_tested"] = sum(r.get("perturbed_tested", 0) for r in rows)
    if qset == "a14":
        listed = [r for r in by["unans"] if A14_LISTED(r["question"])]
        gates["unans_declined_listed"] = f"{sum(r['declines'] for r in listed)}/{len(listed)}"
        gates["unans_declined_unlisted"] = f"{sum(r['declines'] for r in by['unans'] if r not in listed)}/{len(by['unans']) - len(listed)}"
    nu = len(by["unans"])
    gates["pass"] = {"P13.2": gates["P13.2_all_exact"],
                     "P13.3": gates["P13.3_tool_answered_rate"] >= .95 and gates["P13.3_fields_quoted_rate"] >= .90
                     and gates["P13.3_numbers_supported_rate"] is not None and gates["P13.3_numbers_supported_rate"] >= .95,
                     "P13.4": gates["P13.4_no_tool"] >= len(by["notool"]) - 1,
                     "P13.5": gates["P13.5_declines"] >= nu - nu // 8 and gates["P13.5_unsupported_numbers"] == 0}
    if qset == "a14":  # same tests, Amendment 14 names: P14.2 exact, P14.3 unanswerable, P14.4 answerable, P14.5 no-tool
        p = gates["pass"]
        gates["pass"] = {"P14.2": p["P13.2"], "P14.3": p["P13.5"], "P14.4": p["P13.3"], "P14.5": p["P13.4"]}
    out = src.with_name(src.stem + ("-checked-v3.json" if v3 else "-checked.json"))
    out.write_text(json.dumps({"protocol": "PROTOCOL.md Amendment " + ("14" if qset == "a14" or v3 else "13"), "gates": gates, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(gates, indent=1))
    for r in rows:
        missed = [f for f, v in r["fields"].items() if not v["quoted"]]
        if r["kind"] != "para" or missed or r["unsupported"] or not r["exact"]:
            print(f"[{r['kind']}] cap {r['capture']} tool={r['tool_called']} declines={r['declines']} missed {missed} "
                  f"unsupported {r['unsupported']} :: {r['answer'][:220]!r}")


if __name__ == "__main__":
    main()

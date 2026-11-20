"""PROTOCOL Amendment 11: deterministic answer checker for Stage 2 transcripts.

Usage: answer_checker.py [--v2] [v15/reports/stage2/fullgpu-chunked-stage2-bank.json]
Writes <input>-checked.json (v2: -checked-v2.json) next to the input and prints the P11 gates.
--v2 (PROTOCOL Amendment 12 P12.1): numbers compare by value (trailing zeros ignored, "-0.5" = "-0.5000")
and "120,000,000" is one number.
v3 (PROTOCOL Amendment 14, used through check(..., v3=True)): v2 plus (a) list markers at a line start ("1. ",
"2) ") are not numbers; (b) a number is also supported if it equals, at the answer's precision, a simple
derivation of report/question numbers: a unit rescaling r * 10^k (k = +-3, +-6, +-9), a difference r1 - r2
(either order), or a percentage 100 * r1 / r2; differences and percentages (r1 != r2) count only for answer
numbers with a decimal point or >= 3 significant digits, since small integers are derivable by chance; (c) a field is quoted if a number equals its value times
10^k. derivable() reports how many perturbed numbers (x 1.07) would also pass, as a permissiveness check.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NUM = re.compile(r"-?\d+(?:\.\d+)?")
NUM2 = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
HZ = re.compile(r"(-?[\d.,]+)\s*(?:k|M|G)?Hz")


def numbers(text, v2):
    return [n.replace(",", "") for n in NUM2.findall(text)] if v2 else NUM.findall(text)


LIST = re.compile(r"(?m)^\s*(?:[*-]\s*)?\d+[.)]\s")
SCALES = [10.0 ** k for k in range(-9, 10, 3)]


def derived(pool):
    pool = sorted({float(x) for x in pool})
    scaled = {a * s for a in pool for s in SCALES}
    combos = {a - b for a in pool for b in pool if a != b} | {100 * a / b for a in pool for b in pool if b and a != b}
    return scaled, combos


def derivable(n, dset):
    scaled, combos = dset if isinstance(dset, tuple) else (dset, set())
    dec = len(n.split(".")[1]) if "." in n else 0
    x = float(n)
    sig = len(n.lstrip("-").replace(".", "").strip("0"))
    pool = scaled | combos if "." in n or sig >= 3 else scaled
    return any(abs(round(d, dec) - x) < 1e-9 * max(1.0, abs(x)) for d in pool)


def norm(t):
    return t.replace("−", "-").replace("–", "-")


def answer_and_report(text):
    reports = re.findall(r"<tool_response>(.*?)</tool_response>", text, re.S)
    answer = re.sub(r"<think>.*?</think>", "", text.split("</tool_response>")[-1], flags=re.S) if reports else ""
    return norm(answer.replace("<|im_end|>", "")), norm("\n".join(reports))


def supported(n, report_nums, question_nums, v2=False):
    if n in report_nums or n in question_nums:
        return True
    if v2 and any(float(n) == float(r) for r in report_nums | question_nums):
        return True
    dec = len(n.split(".")[1]) if "." in n else 0
    return any(round(float(r), dec) == float(n) for r in report_nums)


def check(run, fields, v2=False, v3=False):
    v2 = v2 or v3
    answer, report = answer_and_report(run["answer"])
    rnums, qnums = set(numbers(report, v2)), set(numbers(norm(run["question"]), v2))
    nums = numbers(LIST.sub(" ", answer) if v3 else answer, v2)
    unsupported = [n for n in nums if not supported(n, rnums, qnums, v2)]
    extra = {}
    if v3:
        dset = derived(rnums | qnums)
        by_derivation = [n for n in unsupported if derivable(n, dset)]
        unsupported = [n for n in unsupported if n not in by_derivation]
        bumped = [f"{float(n) * 1.07:.{len(n.split('.')[1]) if '.' in n else 0}f}" for n in by_derivation]
        bumped = [b for b, n in zip(bumped, by_derivation) if float(b) != float(n)]
        extra = {"supported_by_derivation": by_derivation, "perturbed_tested": len(bumped),
                 "perturbed_also_pass": sum(derivable(b, dset) for b in bumped)}
    quoted = {}
    for f in fields:
        v = re.search(f + r"=(-?[0-9.]+)", report)
        if v3:
            hit = bool(v) and any(abs(float(n) - float(v.group(1)) * s) <= 1e-9 * max(1.0, abs(float(n))) or
                                  (abs(float(v.group(1)) * s) >= 1 and derivable(n, {float(v.group(1)) * s}))
                                  for n in nums for s in SCALES)
        elif v2:
            hit = bool(v) and any(float(n) == float(v.group(1)) for n in nums)
        else:
            hit = bool(v) and re.search(r"(?<![\d.])" + re.escape(v.group(1)) + r"(?![\d])", answer) is not None
        quoted[f] = {"report": v.group(1) if v else None, "quoted": hit}
    return {"capture": run["capture"], "question": run["question"], "tools_called": run["tools_called"],
            "tool_answered": run["tool_responses"] >= 1, "exact": run["reference_lock_error"] is None and not run["head_mismatch"],
            "seconds": run["seconds"], "fields": quoted, "numbers": len(nums), "unsupported": unsupported,
            "hz_claims": HZ.findall(answer), "hz_in_report": "freq_hz=" in report or "cfo_hz=" in report,
            "cfo_attributed_to_bin": "carrier frequency offset" in run["question"] and bool(
                re.search(r"(spectral bin|dominant bin|16-point spectrum)", answer)),
            "answer": answer.strip(), **extra}


def main():
    sys.path.insert(0, str(ROOT / "v15/scripts"))
    from stage2_chat import BANK, REAL
    args = [a for a in sys.argv[1:] if a != "--v2"]
    v2 = "--v2" in sys.argv
    src = Path(args[0] if args else ROOT / "v15/reports/stage2/fullgpu-chunked-stage2-bank.json")
    fields = {q: f for _, q, f in REAL + BANK}
    runs = json.loads(src.read_text(encoding="utf-8"))["runs"]
    rows = [check(r, fields[r["question"]], v2) for r in runs]
    n = len(rows)
    fq = [v["quoted"] for r in rows for v in r["fields"].values()]
    nums, uns = sum(r["numbers"] for r in rows), sum(len(r["unsupported"]) for r in rows)
    kinds = [("level" if i % 6 < 3 else "spectral") for i in range(n)]
    choice = {k: {t: sum(1 for r, kk in zip(rows, kinds) if kk == k and t in r["tools_called"]) for t in ("rf_levels", "rf_analyze")}
              for k in ("level", "spectral")}
    gates = {"questions": n, "expected": 96,
             "P11.1_all_exact": all(r["exact"] for r in rows),
             "P11.2_tool_answered_rate": sum(r["tool_answered"] for r in rows) / n,
             "P11.3_fields_quoted_rate": sum(fq) / len(fq),
             "P11.4_numbers_supported_rate": (nums - uns) / nums if nums else None,
             "numbers": nums, "unsupported": uns, "tool_choice": choice,
             "seconds_median": sorted(r["seconds"] for r in rows)[n // 2], "checker": "v2" if v2 else "v1",
             "hz_claims": sum(len(r["hz_claims"]) for r in rows),
             "cfo_attributed_to_bin": sum(r["cfo_attributed_to_bin"] for r in rows)}
    gates["pass"] = {"P11.1": gates["P11.1_all_exact"], "P11.2": gates["P11.2_tool_answered_rate"] >= .95,
                     "P11.3": gates["P11.3_fields_quoted_rate"] >= .90,
                     "P11.4": gates["P11.4_numbers_supported_rate"] is not None and gates["P11.4_numbers_supported_rate"] >= .95}
    out = src.with_name(src.stem + ("-checked-v2.json" if v2 else "-checked.json"))
    out.write_text(json.dumps({"protocol": "PROTOCOL.md Amendment 11" + (" / 12 (checker v2)" if v2 else ""), "gates": gates, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(gates, indent=1))
    for r in rows:
        missed = [f for f, v in r["fields"].items() if not v["quoted"]]
        if missed or r["unsupported"] or not r["exact"]:
            print(f"cap {r['capture']}: missed {missed} unsupported {r['unsupported']} :: {r['answer'][:160]!r}")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v15/scripts"))
import answer_checker as A  # noqa: E402

REPORT = "<tool_response>\ncfo_cycles_per_sample=-0.5000 cfo_hz=-62600\nfs=120MS/s\n</tool_response>"


def run(answer, q="Estimate the carrier frequency offset of capture 5 in cycles per sample."):
    return {"answer": REPORT + answer, "question": q, "capture": 5, "tools_called": ["rf_analyze"], "tool_responses": 1,
            "reference_lock_error": None, "head_mismatch": None, "seconds": 1.0}


def test_v2_trailing_zeros():
    """Amendment 12 P12.1: "-0.5" equals the report's "-0.5000" under v2, not under v1."""
    f = ("cfo_cycles_per_sample",)
    assert not A.check(run("The CFO is -0.5 cycles per sample."), f)["fields"][f[0]]["quoted"]
    assert A.check(run("The CFO is -0.5 cycles per sample."), f, v2=True)["fields"][f[0]]["quoted"]


def test_v2_thousands_separator():
    r1 = A.check(run("That is -62,600 Hz."), ())
    r2 = A.check(run("That is -62,600 Hz."), (), v2=True)
    assert r1["numbers"] == 2 and r2["numbers"] == 1 and r2["unsupported"] == []
    assert A.check(run("About 120,000,000 samples/s."), (), v2=True)["unsupported"] == ["120000000"]


def test_wrong_hz_conversion_is_unsupported():
    r = A.check(run("This corresponds to approximately -62.6 Hz."), (), v2=True)
    assert r["unsupported"] == ["-62.6"] and r["hz_claims"]


def test_cfo_attribution_flag():
    assert A.check(run("Derived from the dominant spectral bin analysis."), ())["cfo_attributed_to_bin"]
    assert not A.check(run("From the lag-1 autocorrelation phase."), ())["cfo_attributed_to_bin"]


def test_v3_derivations_rescaling_and_list_markers():
    from answer_checker import check
    report = "<tool_response>mean_power_dbfs=-11.9 noise_floor_dbfs=-13.0 active_windows=3/16 cfo_hz=17594</tool_response>"
    run = {"question": "q", "capture": 2, "tools_called": ["rf_analyze"], "tool_responses": 1,
           "reference_lock_error": None, "head_mismatch": False, "seconds": 1,
           "answer": report + "1. The gap is 1.1 dB.\n2. Active share 18.75%.\n3. Offset 17.594 kHz, about 17.6 kHz. Also 42.0."}
    v2 = check(run, ("cfo_hz",), v2=True)
    v3 = check(run, ("cfo_hz",), v3=True)
    assert "1.1" in v2["unsupported"] and not v2["fields"]["cfo_hz"]["quoted"]
    assert v3["unsupported"] == ["42.0"]                    # an underivable number stays unsupported
    assert set(v3["supported_by_derivation"]) >= {"1.1", "18.75", "17.594", "17.6"}
    assert v3["fields"]["cfo_hz"]["quoted"]                 # quoted after a kHz rescaling
    assert v3["numbers"] == 5                               # list markers 1. 2. 3. are not numbers

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from heldout_checker import DECLINE, final_answer  # noqa: E402


def test_decline_phrases():
    for s in ("The analyzer does not report modulation.", "This is not provided by the tool.",
              "I cannot determine the distance from these measurements.", "The report doesn't include bandwidth."):
        assert DECLINE.search(s), s
    assert not DECLINE.search("The mean power is -16.3 dBFS.")


def test_final_answer_strips_report_and_think():
    t = "<tool_response>x=1</tool_response><think>hmm</think>It is -3 dBFS.<|im_end|>"
    assert final_answer(t) == "It is -3 dBFS."

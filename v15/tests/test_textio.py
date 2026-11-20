import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = "import sys; sys.path.insert(0, r'{}'); {}print('\u2248 -62.6 kHz')"


def _run(guard):
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    pre = "from rfpu_v15.textio import utf8_stdout; utf8_stdout(); " if guard else ""
    return subprocess.run([sys.executable, "-c", CODE.format(ROOT / "v15", pre)], env=env, capture_output=True)


def test_utf8_guard_prints_model_symbols():
    """Amendment 12 P12.2: without the guard a cp1252 stdout raises; with it the text is printed."""
    assert _run(False).returncode != 0
    r = _run(True)
    assert r.returncode == 0 and "≈".encode() in r.stdout


def test_run_py_installs_the_guard():
    assert "utf8_stdout()" in (ROOT / "v15/scripts/run.py").read_text(encoding="utf-8")

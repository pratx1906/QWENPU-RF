"""Console text output (PROTOCOL Amendment 12 P12.2)."""
import sys


def utf8_stdout():
    """Make stdout/stderr UTF-8 with replacement, so model text such as "≈" cannot crash a run on a
    cp1252 Windows console or pipe (the Amendment 11 bank crashed in run.py's echo of the log)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

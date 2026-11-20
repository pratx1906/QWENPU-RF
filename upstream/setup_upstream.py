"""Fetch the code this project runs on: QwenPU (chipformer), pinned, plus its RF instruction branch.

    python upstream/setup_upstream.py

Creates (both git-ignored):
  upstream/QwenPU      https://github.com/divagr18/QwenPU at 5af1fc0 (the version every result used)
  upstream/QwenPU-rf   the same commit with rfpu-vx.patch applied: VFIR, VPOW, VBFLY and the VFIR guard bit
                       (see COMMITS.md). Only the VX/VFIR/VPOW/VBFLY scripts need it.

Then install chipformer from upstream/QwenPU, e.g.  pip install -e "upstream/QwenPU[hf]"
"""
import subprocess
import sys
from pathlib import Path

URL = "https://github.com/divagr18/QwenPU"
PIN = "5af1fc0e0722ad1cc0d983394410c7fc331e08f2"
HERE = Path(__file__).resolve().parent


def git(*args, cwd=None):
    print("+ git", *args, flush=True)
    subprocess.run(["git", *args], cwd=cwd, check=True)


def checkout(dest: Path):
    if not (dest / ".git").exists():
        git("clone", URL, str(dest))
    git("fetch", "--quiet", "origin", cwd=dest)
    git("-c", "advice.detachedHead=false", "checkout", PIN, cwd=dest)


def main():
    base, rf = HERE / "QwenPU", HERE / "QwenPU-rf"
    checkout(base)
    checkout(rf)
    if subprocess.run(["git", "diff", "--quiet"], cwd=rf).returncode == 0:
        git("apply", "--whitespace=nowarn", str(HERE / "rfpu-vx.patch"), cwd=rf)
    else:
        print("QwenPU-rf already has local changes; leaving it as is")
    print("\nDone. Install chipformer with:  pip install -e \"upstream/QwenPU[hf]\"")
    print("For the VX instruction scripts set PYTHONPATH=upstream/QwenPU-rf (see README).")


if __name__ == "__main__":
    sys.exit(main())

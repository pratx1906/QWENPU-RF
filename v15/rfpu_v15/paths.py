"""Where the large local assets live: datasets, checkpoints, caches. They are not in git.

Set the environment variable RFPU_DATA to that folder. Without it, the first existing folder of
<repo>/rfpu-data, ~/RFPU-data, C:/RFPU-data and D:/RFPU-data is used (else <repo>/rfpu-data).
Expected layout: datasets/<name>/..., checkpoints/<name>/..., cache/..., viz/...
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    env = os.environ.get("RFPU_DATA")
    if env:
        return Path(env)
    for p in (REPO / "rfpu-data", Path.home() / "RFPU-data", Path("C:/RFPU-data"), Path("D:/RFPU-data")):
        if p.is_dir():
            return p
    return REPO / "rfpu-data"


DATA = data_root()


def local(path) -> Path:
    """Map a recorded dataset/checkpoint path (selections and manifests store the original
    D:/RFPU-data/... location) onto this machine's RFPU_DATA folder."""
    s = str(path).replace("\\", "/")
    for prefix in ("D:/RFPU-data", "d:/RFPU-data", "$RFPU_DATA"):
        if s.startswith(prefix):
            return DATA / s[len(prefix):].lstrip("/")
    return Path(s)

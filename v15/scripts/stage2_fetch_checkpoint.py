"""Complete the local QwenPU-published checkpoint (Qwen3.5-4B + compiled CPU) for Stage 2.

Every shard is hashed and compared with the publisher's LFS SHA256 (pinned revision). Missing or
incomplete shards (v1.0 fetched only chip byte ranges) are downloaded to a staging folder,
verified, and only then moved into place. Report: v15/reports/stage2-checkpoint.json.
"""
import hashlib
import sys
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v15")); from rfpu_v15.paths import DATA  # noqa: E402  (RFPU_DATA, see paths.py)
CK = DATA / "checkpoints/QwenPU-published"
STAGE = DATA / "cache/qwenpu-stage2"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    from huggingface_hub import hf_hub_download
    meta = json.loads((CK / "hf-file-metadata.json").read_text())
    rev, rows = meta["revision"], {}
    for name, (size, want) in meta["files"].items():
        if not name.startswith("model-") and name != "tokenizer.json":
            continue
        p = CK / name
        have = sha(p) if p.exists() and p.stat().st_size == size else None
        status = "verified" if have == want else "fetched"
        if have != want:
            STAGE.mkdir(parents=True, exist_ok=True)
            got = Path(hf_hub_download("divagr1925/QwenPU", name, revision=rev, local_dir=STAGE))
            if sha(got) != want:
                raise ValueError(f"downloaded {name} does not match the publisher SHA256")
            shutil.move(str(got), str(p))
        rows[name] = {"bytes": size, "sha256": want, "status": status}
        print(name, status, flush=True)
    out = {"repo": "divagr1925/QwenPU", "revision": rev, "files": rows, "all_verified": True}
    (ROOT / "v15/reports/stage2-checkpoint.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

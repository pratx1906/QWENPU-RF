# Upstream: QwenPU and the RF instruction patch

Everything here runs on [QwenPU](https://github.com/divagr18/QwenPU) (`chipformer`) by Divyansh Agrawal
(MIT), pinned at `5af1fc0`: every result in this repository was measured on that commit.

| File | What it is |
|---|---|
| `setup_upstream.py` | Clones QwenPU at the pin into `upstream/QwenPU`, and a second copy with the patch applied into `upstream/QwenPU-rf` (both git-ignored) |
| `rfpu-vx.patch` | The `rfpu-vfir` branch as one diff on `5af1fc0`: the VFIR, VPOW and VBFLY vector instructions and the VFIR guard bit, with reference semantics, decoder, circuit and tests (10 files, +1147 / −20) |
| `COMMITS.md` | The branch's 11 commits, in order, with their messages |

The new instructions add no layers or heads: they reuse the VX unit's MAC chain inside the existing
13-layer circuit. Design and results: [VX_RF_INSTRUCTIONS.md](../v15/research/VX_RF_INSTRUCTIONS.md).

Use the patched copy only for the VX/VFIR/VPOW/VBFLY scripts and tests:

```bash
PYTHONPATH=upstream/QwenPU-rf python -m pytest v15/tests/test_vfir.py -q
cd upstream/QwenPU-rf && PYTHONPATH=. python -m pytest tests/test_rv_vx.py tests/test_rv_vbfly_ref.py \
    tests/test_rv_vfir_ref.py tests/test_rv_vpow_ref.py -q
```

Checked on 2026-11-20: the patch applies cleanly to `5af1fc0` and reproduces the branch file for file
(line endings aside).

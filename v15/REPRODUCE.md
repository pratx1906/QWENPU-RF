# Reproducing the v1.5 results

Every result in [README.md](README.md) comes from a script in `v15/scripts/`, run through the append-only ledger `v15/scripts/run.py`. The ledger writes `v15/results/raw/*` (the log, a JSON record and a source snapshot) and refreshes `v15/results.md`.

Pre-declared gates are in [research/PROTOCOL.md](research/PROTOCOL.md) and their outcomes in [reports/gates.json](reports/gates.json). Every run, including failed ones, is in [research/RESEARCH_LOG.md](research/RESEARCH_LOG.md).

## 1. Environment (as measured)

**Hardware:** one NVIDIA GeForce RTX 5090 Laptop GPU (24,463 MiB, driver 610.88), 63 GB RAM, Windows 11.

**Main environment** (`.venv`)
- Python 3.12.10, torch 2.14.1+cu130, transformers 5.22.1, triton-windows.
- Exact pins: [configs/env-main.txt](configs/env-main.txt).
- chipformer: editable install of `upstream/QwenPU` at `5af1fc0e0722ad1cc0d983394410c7fc331e08f2`.

**Flash-linear-attention environment**
- `$RFPU_DATA/environments/rfpu-v15-fla`: the same pins plus flash-linear-attention 0.6.0 and accelerate.
- Exact list: [configs/env-fla.txt](configs/env-fla.txt).

**New-instruction branch**
- `upstream/QwenPU-rf`, branch `rfpu-vfir` at `b72623e3ca25daef0a790e9bee0c4bc543b2bfdb`. It holds VFIR, VPOW and VBFLY.
- Use it with `PYTHONPATH=upstream/QwenPU-rf`; scripts assert that `chipformer.__file__` points into it.

**Interpreter path:** must be absolute for `run.py` on Windows.

```bash
PY='.venv\Scripts\python.exe'
FPY='$RFPU_DATA\environments\rfpu-v15-fla\Scripts\python.exe'
RF="upstream/QwenPU-rf"
```

## 2. Assets and checksums

| Asset | Where | Integrity record |
|---|---|---|
| QwenPU published checkpoint (26.7 GB) | `$RFPU_DATA/checkpoints/QwenPU-published` | `rfpu-download.json` (SHA-256 per shard), verified on every load |
| Qwen3.5-4B base (revision `851bf6e`) | `$RFPU_DATA/checkpoints/qwen3.5-4b` | `rfpu-download.json` |
| Plain-CPU merge (19 GB, fits the GPU) | `$RFPU_DATA/checkpoints/rfpu-v15/qwen3.5-4b-rv1` | No hash file was written at merge time; the shard SHA-256s are published with the Hugging Face upload. Functional check 2026-11-20: Stage 2 `rf_levels` run, 27,144/27,144 chip tokens equal to the reference |
| VX + VFIR merge | `.../qwen3.5-4b-vx-vfir` (not kept; rebuild with `merge --vx` on the branch at `c5d55c1`) | [reports/checkpoint-provenance/](reports/checkpoint-provenance/) (shard SHA-256 of the original) |
| VX + VFIR + VPOW + VBFLY merge | `.../qwen3.5-4b-vx-rf` (not kept; rebuild with `merge --vx` at `b72623e`) | [reports/checkpoint-provenance/](reports/checkpoint-provenance/) (shard SHA-256 of the original) |
| IDLab, Tampere, NTU LoRa, EPFL datasets | `$RFPU_DATA/datasets` | `data/idlab-iq-manifest.json` and per-dataset manifests (SHA-256) |
| 14 gate captures | listed in `data/real-iq-selection.json` | — |

**Merges**
```bash
"$PY" -m chipformer.rv.merge $RFPU_DATA/checkpoints/qwen3.5-4b --out .../qwen3.5-4b-rv1
PYTHONPATH="$RF" "$PY" -m chipformer.rv.merge $RFPU_DATA/checkpoints/qwen3.5-4b --out .../qwen3.5-4b-vx-rf --vx
```

## 3. Tests (CPU, ~2 min; branch tests ~20 min)

```bash
"$PY" -m pytest v15/tests -q -p no:cacheprovider                      # 218 passed, 21 skipped (branch-only)
PYTHONPATH="$RF" "$PY" -m pytest v15/tests/test_vfir.py -q            # 14 passed
cd upstream/QwenPU-rf && PYTHONPATH=. "$PY" -m pytest tests/test_rv_vx.py tests/test_rv_vbfly_ref.py \
    tests/test_rv_vfir_ref.py tests/test_rv_vpow_ref.py -q            # every token through the circuit
```

## 4. Results

Run one GPU job at a time: timings from overlapping runs are invalid. Each command is `"$PY" v15/scripts/run.py <name> -- <interpreter> v15/scripts/<script>.py [args]`; the table gives the name, the script and its arguments.

| Result | Ledger name / script and args | Interp. | Approx. time | Output |
|---|---|---|---|---|
| 14-capture real-IQ gate (primary / headroom / BFP) | `real-iq-*` / `real_iq_q15.py [--headroom \| --bfp]` | `$PY` | 10–20 min | `reports/real-iq-*.json` |
| Full pipeline | `pipeline-14-captures` / `pipeline_run.py` (`--synthetic`) | `$PY` | ~20 min | `reports/pipeline*.json` |
| DSP figures of merit | `metrics-*` / `dsp_metrics.py <groups> summary` | `$PY` | 10–40 min | `reports/DSP_METRICS.md` |
| Error budget (Amendment 4) | `error-budget` / `error_budget.py` | `$PY` | ~30 min | `reports/ERROR_BUDGET.md` |
| Token reductions (Amendment 5) | `fft-opt-neural`, `am-trim-neural` | `$PY` | ~5 min each | `reports/fft-opt-neural.json`, `am-trim-neural.json` |
| Chunked verification, 107k tokens | `speculative_verify.py --taps 64 --samples 48 --no-free-running --gpu-compare --chunks 512,1024,2048,4096` (`--out v15/reports/speculative-fir64-fla-gpucmp.json`) | `$FPY` | ~2 min | `reports/speculative-fir64-*.json` |
| VFIR timing study (Amendment 3) | `timing-study` / `timing_study.py --processes 8` (with `PYTHONPATH="$RF"`) | `$PY` | ~90 min | `reports/timing-study.json` |
| VBFLY, IIR wide, VFIR precision (Amendment 6) | `amendment6` / `amendment6.py` (with `PYTHONPATH="$RF"`) | `$PY` | ~20 min | `reports/amendment6.json` |
| VFIR / VPOW / VBFLY on saved weights | `vx-rf-checkpoint` / `vx_rf_checkpoint.py` (with `PYTHONPATH="$RF"`) | `$PY` | ~30 min | `reports/vx-rf-checkpoint.json` |
| Stage 2: Qwen calls the RF tool, one-token path | `stage2-fullgpu-*` / `stage2_chat.py --full-gpu $RFPU_DATA/checkpoints/rfpu-v15/qwen3.5-4b-rv1 --max-new 1500` | `$PY` | ~15 min | `reports/stage2/fullgpu-stage2-chat.json` |
| Stage 2, chunked verification (token-identical) | same + `--chunk 1024` | `$PY` | ~3 min | `reports/stage2/fullgpu-chunked-stage2-chat.json` |
| Stage 2, `rf_levels` / real recordings | same + `--suite levels` / `--suite real` | `$PY` | ~3–6 min | `reports/stage2/fullgpu-chunked-stage2-{levels,real}.json` |
| Classifier: fixed split / 6 locations / 10 MS/s | `train_tech_classifier.py --frames 64 --hidden 8 --l2 1e-3 --neural`; `classifier_screen.py [--lolo6 \| --10msps --neural]` | `$PY` | 10–30 min | `configs/tech-classifier.json`, `reports/classifier-*.json` |
| Expanded selection, 133 recordings (seed 2026) | `build_expanded_selection.py` (CPU) | `$PY` | ~10 min (hashing) | `reports/real-iq-selection-expanded.json` |
| Expanded real-IQ screen (Amendment 9) | `real-iq-expanded-133` / `real_iq_q15.py --bfp --selection reports/real-iq-selection-expanded.json --out reports/real-iq-expanded.json` | `$PY` | ~3.7 h | `reports/real-iq-expanded.json` |
| Full pipeline on 133 (Amendment 10a) | `pipeline-expanded-133` / `pipeline_run.py --selection ... --out reports/pipeline-expanded.json` | `$PY` | ~3.2 h | `reports/pipeline-expanded.json` |
| 10-kernel battery on 133 (Amendment 10b) | `kernel-battery-133` / `kernel_battery.py` (with `PYTHONPATH="$RF"`) | `$PY` | ~1.7 h | `reports/kernel-battery.json`; tail diagnosis `battery_tail_diag.py` (CPU) |
| IIR extra bits / VFIR adaptive scale (Amendment 12) | `a12-battery` / `a12_battery.py` (with `PYTHONPATH="$RF"`) | `$PY` | ~20 min | `reports/a12-battery.json` |
| Firmware v3 Hz fields (reference machine) | `firmware_v3_check.py` (CPU) | `$PY` | ~1 min | `reports/firmware-v3-check.json` |
| Stage 2 question bank (Amendments 11, 12) | `stage2_chat.py --suite bank --full-gpu $RFPU_DATA/checkpoints/rfpu-v15/qwen3.5-4b-rv1 --chunk 1024` (`--resume` after a crash), then `answer_checker.py [--v2] <file>` | `$PY` | ~65–80 min | `reports/stage2/fullgpu-chunked-stage2-bank*.json` (Amendment 12 re-run used `--max-new 1000`) |

## 5. What "exact" means here

A neural result counts only if:
1. every token the network produced equals the RV32 reference machine's trace; and
2. every output equals an independent integer reference.

CPU-only and reference-only results are labelled as such in the research log and are never reported as neural.

Chunked verification is an audit mode: the reference drafts the trace and the layers verify every position.

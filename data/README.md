# Data

The recordings are public datasets. They are **not redistributed here**; each keeps its publisher's
licence. This folder holds what is needed to fetch them and to prove you have the same bytes.

| Dataset | Used for | Source | Licence |
|---|---|---|---|
| IDLab LTE / Wi-Fi / DVB-T IQ (Ghent University – imec) | 14-capture gate, expanded screen, technology classifier | [idlab.ugent.be](https://idlab.ugent.be/resources/iq-samples-of-lte-wifi-and-dvb-t-in-a-real-environment), [GitHub](https://github.com/JaronFontaine/Technology-Recognition-dataset-of-real-life-LTE-WiFi-and-DVB-T) | CC-BY-NC-SA-4.0 |
| LoRaIQ, EPFL (Tapparel, Burg) | gate, expanded screen | [10.5281/zenodo.20341802](https://doi.org/10.5281/zenodo.20341802) | CC-BY-4.0 |
| NTU RFSR LoRa over the air | gate, expanded screen | [10.21979/N9/C6ABM3](https://doi.org/10.21979/N9/C6ABM3) | see the DOI page |
| Tampere drone RF (Vuorenmaa et al.) | gate, expanded screen | [10.5281/zenodo.4264467](https://doi.org/10.5281/zenodo.4264467) | CC-BY-4.0 |
| CAMRAS Dwingeloo EVE2026 | expanded screen | [data.camras.nl/eve2026/iq](https://data.camras.nl/eve2026/iq/) | CC-BY-SA-4.0 |

The modulation classifier (Amendment 15) uses synthetic signals from
[modgen.py](../v15/rfpu_v15/modgen.py) and needs no download.

## Files here

| File | What it is |
|---|---|
| `real-iq-selection.json` | The 14 gate captures, frozen before any neural run, with SHA-256 per file |
| `idlab-iq-manifest.json` | All 190 IDLab files: location, technology, rate, gain, SHA-256 |
| `manifests/<dataset>.manifest.json` | Per-dataset download manifests: source URL, size, SHA-256 (and publisher checksums where given) |
| `LARGE_FILES.json` | Files too large for git (the 75–100 MB Stage 2 transcripts and the LoRaIQ archive index): path, size, SHA-256. They are in the Hugging Face dataset [pratx1/QwenPU-RF-evidence](https://huggingface.co/datasets/pratx1/QwenPU-RF-evidence) at the same paths. Each transcript has a `-summary.json` in git with everything except the raw token list |

The expanded 133-recording selection is [v15/reports/real-iq-selection-expanded.json](../v15/reports/real-iq-selection-expanded.json).

## Layout

Put the data in one folder and point `RFPU_DATA` at it (see [paths.py](../v15/rfpu_v15/paths.py)):

```
$RFPU_DATA/
  datasets/idlab-technology/<location>/<file>.bin
  datasets/loraiq-20341802/sigmfs/...
  datasets/rfsr-ota-multicondition/...
  datasets/tampere-drone-rf-4264467/<file>.bin
  datasets/camras-eve2026/<file>.sigmf-data
  checkpoints/qwen3.5-4b/                   # Qwen/Qwen3.5-4B @ 851bf6e
  checkpoints/QwenPU-published/             # divagr1925/QwenPU
  checkpoints/rfpu-v15/qwen3.5-4b-rv1/      # the 19 GB merge (build or download)
```

Paths recorded in the selections and manifests (`$RFPU_DATA/...`) are mapped onto this folder, and
every capture's SHA-256 is checked before it is used, so a different download fails loudly instead of
giving different numbers.

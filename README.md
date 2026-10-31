# ZEN & NoiFi — Reproduction and Extension

## Purpose

This repository reproduces key experiments from two security/ML papers and adds
small, focused extensions. It is structured for clarity and academic
credibility.

| Paper | Domain | Core Idea |
|-------|--------|-----------|
| **ZEN** (CyFI Lab) | Memory forensics + DNN | Recover custom neural-network layers from a Linux memory dump, attribute the base model, and patch the model so it runs again. |
| **NoiFi** (Shahriar et al.) | WiFi sensing security | Detect adversarial attacks on WiFi CSI classifiers using randomized-smoothing–based noise-feature analysis. |

## Repository Layout

```
.
├── third_party/
│   ├── ZEN/              ← official ZEN repo (git submodule)
│   └── NoiFi/            ← official NoiFi repo (git submodule)
├── scripts/              ← our reproduction & extension scripts
├── environments/         ← pip/conda requirements (Mac-adapted)
├── reports/              ← upstream commit hashes, findings, PDFs
├── results/              ← experiment outputs (small CSVs, plots)
├── logs/                 ← execution logs
└── data/                 ← datasets (gitignored — download locally)
```

## Upstream Attribution

| Project | Authors | Repository | License |
|---------|---------|------------|---------|
| ZEN | CyFI Lab, Georgia Tech | [CyFI-Lab-Public/ZEN](https://github.com/CyFI-Lab-Public/ZEN) | See upstream |
| NoiFi | Shahriar et al. | [shahriar0651/NoiFi](https://github.com/shahriar0651/NoiFi) | See upstream |

Pinned upstream commits are recorded in [`reports/upstream-commits.txt`](reports/upstream-commits.txt).

## Hardware & Environment

| Spec | Value |
|------|-------|
| Machine | MacBook Air (M5, arm64) |
| RAM | 16 GB |
| macOS | 26.6.2 |
| Python | 3.9.6 (system), 3.10.x (venv for NoiFi) |
| Git | 2.54.0 |

> **ZEN** requires a compatible Linux x86-64 environment for the full forensic
> pipeline. The Mac is used for NoiFi and for the ZEN output-consistency
> checker.

## Experiment Status

| # | Experiment | Status | Evidence |
|---|-----------|--------|----------|
| 1 | NoiFi environment setup | ⬜ Pending | |
| 2 | NoiFi smoke pipeline (2 epochs) | ⬜ Pending | |
| 3 | NoiFi baseline reproduction | ⬜ Pending | |
| 4 | NoiFi N-vs-stability extension | ⬜ Pending | |
| 5 | ZEN environment setup (Linux) | ⬜ Pending | |
| 6 | ZEN supplied-dump recovery | ⬜ Pending | |
| 7 | ZEN patched inference | ⬜ Pending | |
| 8 | ZEN output consistency checker | ⬜ Pending | |

## Quick Start

### Clone with submodules

```bash
git clone --recurse-submodules \
  https://github.com/khaledhasanirfan/zen-noifi-reproduction.git
```

### NoiFi (Mac)

```bash
# Create venv
python3.10 -m venv .venv-noifi
source .venv-noifi/bin/activate
pip install -r environments/noifi-mac-initial.txt

# Download dataset
git clone https://github.com/SiamiLab/RoboFiSense.git data/RoboFiSense

# Smoke test
bash scripts/run_noifi_smoke.sh
```

### ZEN (Linux)

```bash
# On a Linux x86-64 machine
conda env create -f third_party/ZEN/Environments/py_38.yml
conda activate yolov5_py_38

# Run supplied-dump demo
bash scripts/run_zen_demo.sh
```

## Extensions

### NoiFi: Randomization Count (N) vs Detection Stability

Tests how the number of randomized references affects detection AUROC,
recall, and latency. See [`scripts/run_noifi_n_study.sh`](scripts/run_noifi_n_study.sh).

### ZEN: Output Consistency Checker

Numerically compares original model outputs against ZEN-recovered model
outputs. See [`scripts/compare_outputs.py`](scripts/compare_outputs.py).

## Limitations

- NoiFi runs on CPU (no CUDA on Mac); training is slower but numerically equivalent.
- ZEN full pipeline requires Linux x86-64 with matching kernel symbols.
- Results here are from our reproduction runs, not copied from the papers.

## License

Our scripts are MIT-licensed. Upstream code retains its original license.
See each submodule for details.

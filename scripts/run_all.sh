#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
.venv-noifi/bin/python scripts/verify_environment.py
bash scripts/download_data.sh
bash scripts/run_noifi_baseline.sh "$@"
.venv-noifi/bin/python scripts/run_zen_attribution.py
.venv-noifi/bin/python scripts/run_zen_patch_demo.py
.venv-noifi/bin/python scripts/compare_outputs.py results/zen-patch-demo/reference.npy results/zen-patch-demo/recovered.npy --output-json results/zen-patch-demo/consistency.json
.venv-noifi/bin/python scripts/generate_report.py

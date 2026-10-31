#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
.venv-noifi/bin/python -u scripts/run_noifi.py --epochs 2 --ae-epochs 2 --per-class 16 --n-values 1 5 --score-seeds 7 --output results/noifi-smoke "$@" 2>&1 | tee logs/noifi-smoke.log

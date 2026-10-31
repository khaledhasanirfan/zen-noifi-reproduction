#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
# Same trained checkpoints, held-out recordings and attacks; only scoring is repeated.
.venv-noifi/bin/python -u scripts/run_noifi.py --reuse "$@" 2>&1 | tee logs/noifi-n-study.log

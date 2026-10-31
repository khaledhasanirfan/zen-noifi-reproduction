#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
.venv-noifi/bin/python -u scripts/run_noifi.py "$@" 2>&1 | tee logs/noifi-basic.log

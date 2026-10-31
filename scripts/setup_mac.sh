#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if ! command -v python3.10 >/dev/null; then
  HOMEBREW_NO_AUTO_UPDATE=1 brew install python@3.10
fi
python3.10 -m venv .venv-noifi
.venv-noifi/bin/python -m pip install -r environments/noifi-mac-initial.txt
.venv-noifi/bin/python scripts/verify_environment.py

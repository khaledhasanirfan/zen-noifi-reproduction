#!/usr/bin/env bash
# Supply the actual patch path explicitly; do not silently select a bundled artifact.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="${1:?Usage: bash scripts/run_zen_inference.sh /absolute/path/to/recovered_patch.pkl}"
python -c 'import sys; assert sys.version_info[:3] == (3, 8, 18), "Activate Python 3.8.18 for recovered bytecode"'
test -s "$PATCH"
mkdir -p "$ROOT/logs/zen" "$ROOT/results/zen"
cd "$ROOT/third_party/ZEN/Models/YoloV5s/yolov5"
WEIGHTS='../yolov5-CBAM-Involution-Visdrone/visdrone.pt'
test -s "$WEIGHTS"
if python detect.py --weights "$WEIGHTS" > "$ROOT/logs/zen/inference-without-patch.log" 2>&1; then
  echo 'Unexpected success without patch; inspect the baseline environment.' >&2
  exit 1
fi
if ! rg 'AttributeError.*(CBAM|Involution)|ModuleNotFoundError.*(CBAM|[Ii]nvolution|models)' "$ROOT/logs/zen/inference-without-patch.log" >/dev/null; then
  echo 'Baseline failed for a different reason; inspect the log.' >&2
  exit 1
fi
python detect.py --weights "$WEIGHTS" --patch "$PATCH" --project "$ROOT/results/zen" --name with-patch --exist-ok 2>&1 | tee "$ROOT/logs/zen/inference-with-patch.log"

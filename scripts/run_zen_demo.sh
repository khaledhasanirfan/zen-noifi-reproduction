#!/usr/bin/env bash
# Official supplied-dump recovery; separate from the Mac artifact attribution run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DUMP="${1:-$ROOT/data/zen/yolov5-involution.lime}"
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  echo 'Use the documented Linux x86-64 / Python 3.8.18 demo environment.' >&2
  exit 1
fi
python -c 'import sys; assert sys.version_info[:3] == (3, 8, 18), "Activate Python 3.8.18"'
test -s "$DUMP"
mkdir -p "$ROOT/logs/zen" "$ROOT/results/zen" "$ROOT/reports"
sha256sum "$DUMP" > "$ROOT/reports/zen-memory-dump.sha256"
cd "$ROOT/third_party/ZEN/volatility3"
python vol.py --help > /dev/null
RUN_MARKER="$(mktemp)"
trap 'rm "$RUN_MARKER"' EXIT
python vol.py -f "$DUMP" linux.pytorch_cmr_recovery --pid 4009375 --PyVersion 3_8_18 --BaseModelCMR 0 --ModelName yolov5-involution --CompareOnly 0 2>&1 | tee "$ROOT/logs/zen/recovery.log"
# A successful process is insufficient; require actual recovered outputs.
PATCH='PR/yolov5-involution_3_8_18/pickle_dict_4009375_3_8_18_yolov5-involution.pkl'
test -s "$PATCH"
if [ ! "$PATCH" -nt "$RUN_MARKER" ]; then
  echo 'Recovery did not produce a fresh patch; a bundled artifact cannot establish recovery.' >&2
  exit 1
fi
printf '%s\n' 'Recovery command finished. Inspect the log and compare output artifacts before making recovery claims.'

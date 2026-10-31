#!/usr/bin/env bash
# ============================================================
# ZEN Demo: Supplied memory dump recovery
# Purpose: Run the official ZEN quick-start demo on a Linux
#          machine using the supplied yolov5-involution dump.
#
# Prerequisites:
#   - Linux x86-64 machine
#   - Conda environment from third_party/ZEN/Environments/py_38.yml
#   - Downloaded .lime file in data/zen/
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DUMP="$ROOT/data/zen/yolov5-involution.lime"
ZEN_DIR="$ROOT/third_party/ZEN"

mkdir -p "$ROOT/logs/zen" "$ROOT/results/zen"

# Record dump hash for provenance
echo "Recording memory dump hash..."
sha256sum "$DUMP" > "$ROOT/reports/zen-memory-dump.sha256"
cat "$ROOT/reports/zen-memory-dump.sha256"

echo ""
echo "=========================================="
echo " ZEN Recovery Demo"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Step 1: Verify Volatility3 plugin
echo "→ Checking Volatility3 installation..."
cd "$ZEN_DIR/volatility3"
python vol.py --help > /dev/null 2>&1 && echo "  ✅ Volatility3 OK" || echo "  ❌ Volatility3 failed"

# Step 2: Run CMR Recovery
echo ""
echo "→ Running CMR Recovery..."
python vol.py \
  -f "$DUMP" \
  linux.pytorch_cmr_recovery \
  --pid 4009375 \
  --PyVersion 3_8_18 \
  --BaseModelCMR 0 \
  --ModelName yolov5-involution \
  --CompareOnly 0 \
  2>&1 | tee "$ROOT/logs/zen/recovery.log"

echo ""
echo "→ Recovery phase complete. Checking output..."

# Step 3: Record generated artifacts
find "$ZEN_DIR" -name "pickle_dict_*" -type f \
  2>/dev/null | head -5 | tee "$ROOT/results/zen/recovered-artifacts.txt"

echo ""
echo "=========================================="
echo " ZEN Recovery Demo COMPLETED"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

#!/usr/bin/env bash
# ============================================================
# NoiFi Extension: N vs Stability Study
# Purpose: Test how randomization count (N) affects detection
#          AUROC, recall, and latency.
#
# Prerequisite: run_noifi_baseline.sh must have completed
#               (trained models and generated attacks are reused).
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE="$ROOT/results/noifi-baseline"
RUN="$ROOT/results/noifi-n-study"
mkdir -p "$RUN" "$ROOT/logs/noifi-n-study"

cd "$ROOT/third_party/NoiFi/src"

COMMON=(
  dataset=robofi
  device=cpu
  "dataset_dir=$ROOT/data"
  "models_dir=$BASELINE/models"
  "scaler_dir=$BASELINE/scaler"
  dataset.batch_size=16
  "environments=[clean]"
  "adv_attacks=[RN,FGSM,BIM,PGD]"
  basetype=both
  evalenv=BenPerturb
  baseline_noise=BP
  n=100
  retrain=False
)

N_VALUES=(1 3 5 10 20)

echo "NoiFi N-vs-Stability Study — started at $(date)"
echo "Testing N values: ${N_VALUES[*]}"
echo ""

for N in "${N_VALUES[@]}"; do
  echo ""
  echo "============================================"
  echo " N = $N"
  echo " $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================"

  N_DIR="$RUN/N_${N}"
  mkdir -p "$N_DIR"

  # Step 3: Re-analyze noise features with this N
  echo "  → Step 3: Analyzing noise features (N=$N)..."
  python -u step_3_analyze_noise_features.py \
    "${COMMON[@]}" \
    "results_dir=$N_DIR/outputs" \
    "randomization_repeats=$N" \
    2>&1 | tee "$ROOT/logs/noifi-n-study/step3_N${N}.log"

  # Step 4: Detect adversarial examples with this N
  echo "  → Step 4: Detecting adversarial examples (N=$N)..."
  START_TIME=$(python3 -c "import time; print(time.time())")
  python -u step_4_detect_adversarial_examples_simple.py \
    "${COMMON[@]}" \
    "results_dir=$N_DIR/outputs" \
    "randomization_repeats=$N" \
    2>&1 | tee "$ROOT/logs/noifi-n-study/step4_N${N}.log"
  END_TIME=$(python3 -c "import time; print(time.time())")

  ELAPSED=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
  echo "  ⏱  N=$N detection took ${ELAPSED}s"
  echo "$N,$ELAPSED" >> "$RUN/latency_results.csv"
done

echo ""
echo "=========================================="
echo " 🎉 N-vs-Stability Study COMPLETED"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo "Results: $RUN"
echo "Latency data: $RUN/latency_results.csv"

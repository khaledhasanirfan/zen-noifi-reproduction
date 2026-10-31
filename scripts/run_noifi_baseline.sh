#!/usr/bin/env bash
# ============================================================
# NoiFi Baseline Reproduction — Full training + evaluation
# Purpose: Reproduce paper results with proper training.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/results/noifi-baseline"
mkdir -p "$RUN" "$ROOT/logs/noifi-baseline"

cd "$ROOT/third_party/NoiFi/src"

COMMON=(
  dataset=robofi
  device=cpu
  "dataset_dir=$ROOT/data"
  "models_dir=$RUN/models"
  "results_dir=$RUN/outputs"
  "scaler_dir=$RUN/scaler"
  dataset.batch_size=16
  "environments=[clean]"
  "adv_attacks=[RN,FGSM,BIM,PGD]"
)

run_step() {
  local script="$1"
  shift
  echo ""
  echo "=========================================="
  echo " Running: $script"
  echo " Time:    $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================="
  python -u "$script" "${COMMON[@]}" "$@" \
    2>&1 | tee "$ROOT/logs/noifi-baseline/${script%.py}.log"
  echo " ✅ $script completed at $(date '+%H:%M:%S')"
}

echo "NoiFi Baseline Reproduction — started at $(date)"
echo "Root: $ROOT"
echo "Output: $RUN"

# Step 0a: Train classifiers (proper epochs)
run_step step_0_train_classifiers.py \
  dataset.epochs=50 retrain=True

# Step 0b: Train autoencoders (proper epochs)
run_step step_0_train_autoencoders.py \
  dataset.ae_epochs=50 retrain=True

# Step 1: Generate adversarial examples
run_step step_1_generate_adversarial_examples_simple.py \
  n=100

# Step 2: Reconstruct and extract noise
run_step step_2_reconstruct_and_extract_noise.py \
  n=100

# Step 3: Analyze noise features
run_step step_3_analyze_noise_features.py \
  n=100 randomization_repeats=10

# Step 4: Detect adversarial examples
run_step step_4_detect_adversarial_examples_simple.py \
  n=100 randomization_repeats=10 \
  basetype=both evalenv=BenPerturb baseline_noise=BP

echo ""
echo "=========================================="
echo " 🎉 NoiFi Baseline Reproduction COMPLETED"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

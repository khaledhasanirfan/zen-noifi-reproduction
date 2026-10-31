#!/usr/bin/env bash
# ============================================================
# NoiFi Smoke Test — 2-epoch pipeline validation
# Purpose: Verify environment, data loading, and end-to-end
#          pipeline connectivity. NOT for research results.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/results/noifi-smoke"
mkdir -p "$RUN" "$ROOT/logs/noifi-smoke"

cd "$ROOT/third_party/NoiFi/src"

COMMON=(
  dataset=robofi
  device=cpu
  "dataset_dir=$ROOT/data"
  "models_dir=$RUN/models"
  "results_dir=$RUN/outputs"
  "scaler_dir=$RUN/scaler"
  dataset.batch_size=8
  "environments=[clean]"
  "adv_attacks=[RN,FGSM]"
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
    2>&1 | tee "$ROOT/logs/noifi-smoke/${script%.py}.log"
  echo " ✅ $script completed at $(date '+%H:%M:%S')"
}

echo "NoiFi Smoke Test — started at $(date)"
echo "Root: $ROOT"
echo "Output: $RUN"
echo ""

# Step 0a: Train classifiers (2 epochs)
run_step step_0_train_classifiers.py \
  dataset.epochs=2 retrain=True

# Step 0b: Train autoencoders (2 epochs)
run_step step_0_train_autoencoders.py \
  dataset.ae_epochs=2 retrain=True

# Step 1: Generate adversarial examples
run_step step_1_generate_adversarial_examples_simple.py \
  n=10

# Step 2: Reconstruct and extract noise
run_step step_2_reconstruct_and_extract_noise.py \
  n=10

# Step 3: Analyze noise features
run_step step_3_analyze_noise_features.py \
  n=10 randomization_repeats=5

# Step 4: Detect adversarial examples
run_step step_4_detect_adversarial_examples_simple.py \
  n=10 randomization_repeats=5 \
  basetype=both evalenv=BenPerturb baseline_noise=BP

echo ""
echo "=========================================="
echo " 🎉 NoiFi Smoke Test COMPLETED"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo "Outputs: $RUN"
echo "Logs:    $ROOT/logs/noifi-smoke/"

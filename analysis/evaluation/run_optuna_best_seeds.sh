#!/usr/bin/env bash
################################################################################
# Train mref-ad (scripts/mref-ad/train_moe.py) with **fixed** hyperparameters from
# results/optuna_moe_seed_<seed>.json — this is **not** Optuna tuning; it is the
# step **after** you have chosen best params (e.g. from a seed-7 Optuna study).
#
# Usage (from repo root): bash analysis/evaluation/run_optuna_best_seeds.sh
################################################################################

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}" || exit 1
export PYTHONPATH="${ROOT}:${ROOT}/scripts"

# Configuration
# SEEDS="7 13 42 99 123 555 999 1234 1337 2027"
SEEDS="7"
EXPERTS_CONFIG="configs/freesurfer_lastvisit_experts_files.yaml"
SPLITS_PATTERN="configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json"
OUTPUT_DIR="results"

echo "=================================="
echo "TRAINING WITH OPTUNA BEST HYPERPARAMETERS"
echo "=================================="
echo "Experts config: $EXPERTS_CONFIG"
echo "Seeds: $SEEDS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Train with each seed
for SEED in $SEEDS; do
    echo ""
    echo "=========================================="
    echo "Training with seed $SEED"
    echo "=========================================="
    
    CHECKPOINT="${OUTPUT_DIR}/moe_seed_${SEED}_full.pt"
    OUT_JSON="${OUTPUT_DIR}/moe_seed_${SEED}.json"
    SEED_SPLITS="${SPLITS_PATTERN//\{seed\}/${SEED}}"
    BEST_PARAMS_JSON="${OUTPUT_DIR}/optuna_moe_seed_${SEED}.json"

    if [ ! -f "$SEED_SPLITS" ]; then
        echo "[ERROR] Splits file not found: $SEED_SPLITS"
        exit 1
    fi
    
    # Check if best params file exists for this seed
    if [ ! -f "$BEST_PARAMS_JSON" ]; then
        echo "[WARNING] Best params file not found: $BEST_PARAMS_JSON"
        echo "[INFO] Training will use default hyperparameters from config"
    else
        echo "[INFO] Using best params from: $BEST_PARAMS_JSON"
    fi
    
    # Build command with optional best params
    CMD="python3 scripts/mref-ad/train_moe.py \
        --experts_config $EXPERTS_CONFIG \
        --splits $SEED_SPLITS \
        --split_type train_val_test \
        --retrain_only \
        --retrain_on_full \
        --no_early_stopping \
        --out_json $OUT_JSON \
        --save_checkpoint $CHECKPOINT"
    
    # Add best params if file exists
    if [ -f "$BEST_PARAMS_JSON" ]; then
        PARAMS=$(python3 - "$BEST_PARAMS_JSON" << 'PYTHON'
import json
import sys

if len(sys.argv) < 2:
    raise SystemExit("Missing path to best-params JSON")

with open(sys.argv[1]) as f:
    data = json.load(f)

key = list(data.keys())[0]
params = data[key].get('best_params', {})

args = []
for k, v in params.items():
    if k == "gumbel_hard":
        if v:
            args.append("--gumbel_hard")
        continue
    args.append(f"--{k} {v}")

print(" ".join(args))
PYTHON
        )
        CMD="$CMD $PARAMS"
    fi
    
    echo "[INFO] Running: $CMD"
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Completed seed $SEED"
    else
        echo "[ERROR] Failed for seed $SEED"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "ALL SEEDS COMPLETED"
echo "=========================================="
echo ""
echo "Output files:"
for SEED in $SEEDS; do
    echo "  - ${OUTPUT_DIR}/moe_seed_${SEED}_full.pt"
    echo "  - ${OUTPUT_DIR}/moe_seed_${SEED}_full_final_per_subject.json"
done

echo ""
echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Generate plots:"
echo "   python3 analysis/clinical_interpretability/scripts/plot_moe_interpretability.py"
echo ""
echo "2. Use per-subject data for analysis:"
echo "   ${OUTPUT_DIR}/moe_seed_*_full_final_per_subject.json"
echo ""

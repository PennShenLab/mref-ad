#!/bin/bash

# Base config
EXPERTS_CONFIG="configs/freesurfer_lastvisit_cv10_experts_files.yaml"
OUT_BASE="results/optuna_moe"

# Toggle: when true, disable the automatic retrain step after tuning and append
# a `_no_retrain` suffix to the out_base for bookkeeping.
NO_RETRAIN=true

# Toggle: when true, skip seeds where the output file already exists.
# Set to "false" to re-run even if output exists.
SKIP_EXISTING=true

TUNE_TRIALS=200
TUNE_EPOCHS=50
NUM_WORKERS=16
TRIALS_PER_GPU=10
GPU_DEVICES="0,1,2,3"
SELECT_METRIC="val_f1"

# Seeds to run
#SEEDS=(7 13 42 1234 2027)
# SEEDS=(99, 123, 555, 999, 1337)
# SEEDS=(99 123 555)
#SEEDS=(999 1337)
SEEDS=(999)

for SEED in "${SEEDS[@]}"; do
  SPLITS_FILE="configs/splits/splits_by_ptid_80_10_10_seed_${SEED}.json"


  echo "============================================"
  echo "Running MOE Optuna tuning for seed ${SEED}"
  echo "Splits file: ${SPLITS_FILE}"
  echo "============================================"

  # compute optional suffix/flag
  if [[ "${NO_RETRAIN}" == "true" ]]; then
    OUT_BASE_RUN="${OUT_BASE}_seed_${SEED}_no_retrain"
    DB_PATH="sqlite:////home/fzhuang/mref-ad/multimodal-imaging-agents/results/optuna_moe_seed_${SEED}_no_retrain.db"
    NO_RETRAIN_FLAG="--no_auto_retrain"
  else
    OUT_BASE_RUN="${OUT_BASE}_seed_${SEED}"
    DB_PATH="sqlite:////home/fzhuang/mref-ad/multimodal-imaging-agents/results/optuna_moe_seed_${SEED}.db"
    NO_RETRAIN_FLAG=""
  fi

  # Construct output file path and skip if exists (when SKIP_EXISTING=true)
  OUT_FILE="${OUT_BASE_RUN}.json"
  if [[ "${SKIP_EXISTING}" == "true" ]] && [[ -f "${OUT_FILE}" ]]; then
    echo "[SKIP] ${OUT_FILE} already exists (SKIP_EXISTING=true)"
    continue
  fi

  echo "Output file: ${OUT_FILE}"

  python3 scripts/tune_moe.py \
    --experts_config ${EXPERTS_CONFIG} \
    --splits ${SPLITS_FILE} \
    --tune_trials ${TUNE_TRIALS} \
    --tune_epochs ${TUNE_EPOCHS} \
    --num_workers ${NUM_WORKERS} \
    --trials_per_gpu ${TRIALS_PER_GPU} \
    --gpu_devices "${GPU_DEVICES}" \
    --out_base ${OUT_BASE_RUN} \
    --storage ${DB_PATH} \
    --select_metric ${SELECT_METRIC} \
    ${NO_RETRAIN_FLAG}

done

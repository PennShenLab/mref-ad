#!/usr/bin/env bash
set -euo pipefail
#
# Optuna hyperparameter search for tabular baselines (train_baselines.py) and,
# optionally, Flex-MoE (tune_flex_moe_seed7_optuna.py).
#
# Add the special name **flex_moe** to BASELINES to run Flex-MoE tuning for each
# entry in SEEDS_TO_RUN (SQLite study + optional multi-GPU workers).
#
# Usage: bash scripts/optuna/hyperparameter_tune_baselines.sh

# =======================
# Shared
# =======================
EXPERTS_CONFIG="configs/freesurfer_lastvisit_experts_files.yaml"
SPLIT_TYPE="train_val_test"
SELECT_METRIC="val_f1"
OUT_DIR="results/"

NO_RETRAIN=false

# =======================
# Flex-MoE Optuna (used only when BASELINES contains flex_moe)
# =======================
SPLITS_TEMPLATE="configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json"
FLEX_MOE_TOTAL_TRIALS=200
FLEX_MOE_WORKERS_PER_GPU=10
# true: fan out workers across GPUs (requires nvidia-smi). false: one process runs all trials (CPU or single GPU).
FLEX_MOE_PARALLEL=true
FLEX_MOE_ROOT="third_party/flex-moe"
FLEX_MOE_DATALOADER_WORKERS=4

mkdir -p "${OUT_DIR}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}" || exit 1
export PYTHONPATH="${ROOT}:${ROOT}/scripts"

# =======================
# Baselines to run (flex_moe is special — not train_baselines.py)
# =======================
BASELINES=(xgb_all lr_all rf_all mlp_concat ftt_all flex_moe)
# BASELINES=(flex_moe)  # Flex-MoE only
# BASELINES=(ftt_all)

# =======================
# Split files by seed
# =======================
declare -A SPLITS
SPLITS[7]="configs/splits/splits_by_ptid_80_10_10_seed_7.json"
SPLITS[13]="configs/splits/splits_by_ptid_80_10_10_seed_13.json"
SPLITS[42]="configs/splits/splits_by_ptid_80_10_10_seed_42.json"
SPLITS[1234]="configs/splits/splits_by_ptid_80_10_10_seed_1234.json"
SPLITS[2027]="configs/splits/splits_by_ptid_80_10_10_seed_2027.json"

SPLITS[99]="configs/splits/splits_by_ptid_80_10_10_seed_99.json"
SPLITS[123]="configs/splits/splits_by_ptid_80_10_10_seed_123.json"
SPLITS[555]="configs/splits/splits_by_ptid_80_10_10_seed_555.json"
SPLITS[999]="configs/splits/splits_by_ptid_80_10_10_seed_999.json"
SPLITS[1337]="configs/splits/splits_by_ptid_80_10_10_seed_1337.json"

# =======================
# Seeds to run
# =======================
SEEDS_TO_RUN=(7)
# SEEDS_TO_RUN=(7 13 42 1234 2027)

# --- Flex-MoE Optuna for one split seed ---
run_flex_moe_optuna() {
  local seed=$1
  local study_name="optuna_flex_moe_seed_${seed}_${FLEX_MOE_TOTAL_TRIALS}"
  local storage="sqlite:///results/${study_name}.db"
  local out_json="results/${study_name}_best_trial.json"

  if [[ -f "${out_json}" ]]; then
    echo "[SKIP] ${out_json} already exists"
    return 0
  fi

  echo "============================================================"
  echo "[RUN] flex_moe Optuna | seed=${seed}"
  echo "      study=${study_name}"
  echo "      storage=${storage}"
  echo "      parallel=${FLEX_MOE_PARALLEL}"
  echo "============================================================"

  if [[ "${FLEX_MOE_PARALLEL}" == "true" ]]; then
    local GPU_COUNT
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${GPU_COUNT}" -lt 1 ]]; then
      echo "[ERROR] FLEX_MOE_PARALLEL=true but no GPU detected. Set FLEX_MOE_PARALLEL=false for single-process tuning."
      exit 1
    fi
    local TOTAL_WORKERS=$((GPU_COUNT * FLEX_MOE_WORKERS_PER_GPU))
    local TRIALS_PER_WORKER=$(((FLEX_MOE_TOTAL_TRIALS + TOTAL_WORKERS - 1) / TOTAL_WORKERS))

    python3 - <<PY
import optuna
study_name = "${study_name}"
storage = "${storage}"
optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)
print(f"[INIT] Study ready: {study_name}")
PY

    local g w
    for ((g = 0; g < GPU_COUNT; g++)); do
      for ((w = 0; w < FLEX_MOE_WORKERS_PER_GPU; w++)); do
        local LOG="results/${study_name}.gpu${g}.worker${w}.log"
        echo "[LAUNCH] GPU=${g} worker=${w} -> ${LOG}"
        CUDA_VISIBLE_DEVICES=${g} python3 scripts/optuna/tune_flex_moe_seed7_optuna.py \
          --experts_config "${EXPERTS_CONFIG}" \
          --splits_template "${SPLITS_TEMPLATE}" \
          --seed "${seed}" \
          --device 0 \
          --study_name "${study_name}" \
          --storage "${storage}" \
          --n_trials "${TRIALS_PER_WORKER}" \
          --total_trials "${FLEX_MOE_TOTAL_TRIALS}" \
          --select_metric "${SELECT_METRIC}" \
          --flex_moe_root "${FLEX_MOE_ROOT}" \
          --num_workers "${FLEX_MOE_DATALOADER_WORKERS}" \
          >"${LOG}" 2>&1 &
      done
    done
    wait

    python3 - <<PY
import json
import optuna
study_name = "${study_name}"
storage = "${storage}"
study = optuna.load_study(study_name=study_name, storage=storage)
out = {
    "study_name": study_name,
    "storage": storage,
    "best_trial_number": study.best_trial.number,
    "best_value": study.best_value,
    "best_params": study.best_trial.params,
    "n_trials": len(study.trials),
}
with open("${out_json}", "w") as f:
    json.dump(out, f, indent=2)
print(f"[DONE] wrote ${out_json}")
PY
  else
    python3 scripts/optuna/tune_flex_moe_seed7_optuna.py \
      --experts_config "${EXPERTS_CONFIG}" \
      --splits_template "${SPLITS_TEMPLATE}" \
      --seed "${seed}" \
      --device 0 \
      --study_name "${study_name}" \
      --storage "${storage}" \
      --n_trials "${FLEX_MOE_TOTAL_TRIALS}" \
      --total_trials "${FLEX_MOE_TOTAL_TRIALS}" \
      --select_metric "${SELECT_METRIC}" \
      --flex_moe_root "${FLEX_MOE_ROOT}" \
      --num_workers "${FLEX_MOE_DATALOADER_WORKERS}"
  fi
}

# =======================
# Main loop
# =======================
for baseline in "${BASELINES[@]}"; do
  if [[ "${baseline}" == "flex_moe" ]]; then
    for seed in "${SEEDS_TO_RUN[@]}"; do
      run_flex_moe_optuna "${seed}"
    done
    continue
  fi

  if [[ "${baseline}" == "mlp_concat" || "${baseline}" == "ftt_all" ]]; then
    TUNE_TRIALS=200
  else
    TUNE_TRIALS=100
  fi

  for seed in "${SEEDS_TO_RUN[@]}"; do

    SPLIT_FILE="${SPLITS[$seed]}"
    if [[ "${NO_RETRAIN}" == "true" ]]; then
      OUT_SUFFIX="_no_retrain"
      SKIP_RETRAIN_FLAG="--skip_retrain"
    else
      OUT_SUFFIX=""
      SKIP_RETRAIN_FLAG=""
    fi

    OUT_FILE="${OUT_DIR}/optuna_${baseline}_seed_${seed}${OUT_SUFFIX}.json"

    if [[ -f "${OUT_FILE}" ]]; then
      echo "[SKIP] ${OUT_FILE} already exists"
      continue
    fi

    echo "============================================================"
    echo "[RUN] baseline=${baseline} | seed=${seed}"
    echo "      splits=${SPLIT_FILE}"
    echo "      out=${OUT_FILE}"
    echo "============================================================"

    python -u scripts/baselines/train_baselines.py \
      --experts_config "${EXPERTS_CONFIG}" \
      --splits "${SPLIT_FILE}" \
      --baseline "${baseline}" \
      --split_type "${SPLIT_TYPE}" \
      --tune_trials "${TUNE_TRIALS}" \
      --select_metric "${SELECT_METRIC}" \
      --out "${OUT_FILE}" \
      ${SKIP_RETRAIN_FLAG}

  done
done

echo "All Optuna tuning runs finished."

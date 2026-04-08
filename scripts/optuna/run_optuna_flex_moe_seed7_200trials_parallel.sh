#!/usr/bin/env bash
set -euo pipefail

# 200 total Optuna trials for Flex-MoE on seed-7 split,
# running 10 worker processes per GPU in parallel.

EXPERTS_CONFIG="configs/freesurfer_lastvisit_cv10_experts_files.yaml"
SPLITS_TEMPLATE="configs/splits_by_ptid_80_10_10_seed_{seed}.json"
SEED=7
TOTAL_TRIALS=200
WORKERS_PER_GPU=10
SELECT_METRIC="val_f1"
STUDY_NAME="optuna_flex_moe_seed7_200"
STORAGE="sqlite:////home/fzhuang/mref-ad/multimodal-imaging-agents/results/optuna_flex_moe_seed7_200.db"
FLEX_MOE_ROOT="third_party/flex-moe"
NUM_WORKERS_DATALOADER=4

GPU_COUNT=$(nvidia-smi -L | wc -l | tr -d ' ')
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "[ERROR] No GPU detected."
  exit 1
fi

TOTAL_WORKERS=$((GPU_COUNT * WORKERS_PER_GPU))
TRIALS_PER_WORKER=$(((TOTAL_TRIALS + TOTAL_WORKERS - 1) / TOTAL_WORKERS))

mkdir -p results

echo "============================================"
echo "Flex-MoE Optuna tuning (seed=${SEED})"
echo "GPUs: ${GPU_COUNT}, workers/GPU: ${WORKERS_PER_GPU}, total workers: ${TOTAL_WORKERS}"
echo "Total trials cap: ${TOTAL_TRIALS}, per worker: ${TRIALS_PER_WORKER}"
echo "Study: ${STUDY_NAME}"
echo "Storage: ${STORAGE}"
echo "============================================"

# Initialize storage/study once to avoid sqlite schema race across workers.
python3 - <<'PY'
import optuna
study_name = "optuna_flex_moe_seed7_200"
storage = "sqlite:////home/fzhuang/mref-ad/multimodal-imaging-agents/results/optuna_flex_moe_seed7_200.db"
optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)
print(f"[INIT] Study ready: {study_name}")
PY

for ((g=0; g<GPU_COUNT; g++)); do
  for ((w=0; w<WORKERS_PER_GPU; w++)); do
    LOG="results/${STUDY_NAME}.gpu${g}.worker${w}.log"
    echo "[LAUNCH] GPU=${g} worker=${w} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=${g} python3 scripts/tune_flex_moe_seed7_optuna.py \
      --experts_config "${EXPERTS_CONFIG}" \
      --splits_template "${SPLITS_TEMPLATE}" \
      --seed "${SEED}" \
      --device 0 \
      --study_name "${STUDY_NAME}" \
      --storage "${STORAGE}" \
      --n_trials "${TRIALS_PER_WORKER}" \
      --total_trials "${TOTAL_TRIALS}" \
      --select_metric "${SELECT_METRIC}" \
      --flex_moe_root "${FLEX_MOE_ROOT}" \
      --num_workers "${NUM_WORKERS_DATALOADER}" \
      > "${LOG}" 2>&1 &
  done
done

wait

python3 - <<'PY'
import json
import optuna
study_name = "optuna_flex_moe_seed7_200"
storage = "sqlite:////home/fzhuang/mref-ad/multimodal-imaging-agents/results/optuna_flex_moe_seed7_200.db"
study = optuna.load_study(study_name=study_name, storage=storage)
out = {
    "study_name": study_name,
    "storage": storage,
    "best_trial_number": study.best_trial.number,
    "best_value": study.best_value,
    "best_params": study.best_trial.params,
    "n_trials": len(study.trials),
}
with open("results/optuna_flex_moe_seed7_200_best_trial.json", "w") as f:
    json.dump(out, f, indent=2)
print("[DONE] wrote results/optuna_flex_moe_seed7_200_best_trial.json")
PY

echo "[DONE] All workers finished."

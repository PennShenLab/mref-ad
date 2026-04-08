#!/usr/bin/env bash
set -euo pipefail

# Evaluate seed-7 tuned MoE hyperparameters across all 10 split seeds.

SEEDS=(7 13 42 1234 2027 99 123 555 999 1337)
EXPERTS_CONFIG="configs/freesurfer_lastvisit_cv10_experts_files.yaml"
SPLITS_PATTERN="configs/splits_by_ptid_80_10_10_seed_{seed}.json"
PARAMS_JSON="results/optuna_moe_seed_7_best_trial.json"
OUT_DIR="results/seed7_bestparams_all10"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${PARAMS_JSON}" ]]; then
  echo "[ERROR] Missing params json: ${PARAMS_JSON}"
  exit 1
fi

PARAMS=$(python3 - "${PARAMS_JSON}" << 'PYTHON'
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)

params = data.get("params", {})
if not params:
    raise SystemExit("No 'params' found in JSON")

args = []
for k, v in params.items():
    if k == "gumbel_hard":
        if bool(v):
            args.append("--gumbel_hard")
        continue
    args.append(f"--{k} {v}")

print(" ".join(args))
PYTHON
)

echo "======================================================="
echo "Seed-7 fixed hyperparameters across 10 seeds"
echo "Params source: ${PARAMS_JSON}"
echo "Output dir: ${OUT_DIR}"
echo "Seeds: ${SEEDS[*]}"
echo "======================================================="

for SEED in "${SEEDS[@]}"; do
  SPLITS_FILE="${SPLITS_PATTERN//\{seed\}/${SEED}}"
  OUT_JSON="${OUT_DIR}/moe_seed_${SEED}.json"
  CKPT="${OUT_DIR}/moe_seed_${SEED}.pt"
  LOG_FILE="${OUT_DIR}/moe_seed_${SEED}.log"

  if [[ ! -f "${SPLITS_FILE}" ]]; then
    echo "[ERROR] Missing splits file: ${SPLITS_FILE}"
    exit 1
  fi

  echo "[RUN] seed=${SEED}"
  CMD="python3 scripts/train_moe.py \
    --experts_config ${EXPERTS_CONFIG} \
    --splits ${SPLITS_FILE} \
    --split_type train_val_test \
    --retrain_only \
    --retrain_on_full \
    --no_early_stopping \
    --out_json ${OUT_JSON} \
    --save_checkpoint ${CKPT} \
    ${PARAMS}"
  echo "[CMD] ${CMD}"
  eval "${CMD}" 2>&1 | tee "${LOG_FILE}"
done

echo "[DONE] All seeds completed."

#!/usr/bin/env bash
# =============================================================================
# MoE ablations: fixed seed-7 Optuna best hyperparameters, evaluated on 10 split seeds
# =============================================================================
# Same recipe as run_seed7_bestparams_all10.sh (PARAMS_JSON, retrain_on_full, etc.).
#
# - Modality: swap experts YAML (drop modality or single-modality configs).
# - Expert sparsity: --topk on the full CV10 expert graph.
# - Gating architecture: --gate_ablation region_only | modality_only | random
#
# Outputs: results/ablations_seed7_bestparams_all10/<tag>/moe_seed_<seed>.json
#           + moe_aggregated.json per tag.
#
# Per-split-seed Optuna hyperparameters: use run_ablation_seeds.sh instead.
# =============================================================================

set -euo pipefail

RUN_MODALITY_ABLATION=true
RUN_TOPK_ABLATION=true
RUN_GATE_ABLATION=true

SEEDS=(7 13 42 1234 2027 99 123 555 999 1337)
SPLITS_PATTERN="configs/splits_by_ptid_80_10_10_seed_{seed}.json"
PARAMS_JSON="results/optuna_moe_seed_7_best_trial.json"
FULL_EXPERTS_CONFIG="configs/freesurfer_lastvisit_cv10_experts_files.yaml"
OUT_ROOT="results/ablations_seed7_bestparams_all10"

RUN_AGGREGATION=true

MODALITY_ABLATIONS=(
  "modality_no_amyloid:configs/freesurfer_lastvisit_cv10_no_amyloid.yaml"
  "modality_no_mri:configs/freesurfer_lastvisit_cv10_no_mri.yaml"
  "modality_no_demographic:configs/freesurfer_lastvisit_cv10_no_demographic.yaml"
  "modality_mri_only:configs/freesurfer_lastvisit_cv10_mri_only.yaml"
  "modality_amyloid_only:configs/freesurfer_lastvisit_cv10_amyloid_only.yaml"
)

TOPK_VALUES=(1 3 5)

GATE_ABLATIONS=("gate_region_only:region_only" "gate_modality_only:modality_only" "gate_random:random")

extract_moe_cli_from_json() {
  python3 - "${PARAMS_JSON}" << 'PYTHON'
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)

params = data.get("params", {})
if not params:
    raise SystemExit("No 'params' in JSON (expected optuna_moe_seed_7_best_trial.json)")

args = []
for k, v in params.items():
    if k == "gumbel_hard":
        if bool(v):
            args.append("--gumbel_hard")
        continue
    args.append(f"--{k} {v}")

print(" ".join(args))
PYTHON
}

run_ablation_loop() {
  local label=$1
  local experts_config=$2
  local extra_cli=$3

  local out_dir="${OUT_ROOT}/${label}"
  mkdir -p "${out_dir}"

  echo ""
  echo "-----------------------------------"
  echo "[ABLATION] ${label}"
  echo "  experts_config=${experts_config}"
  echo "  extra: ${extra_cli:-<none>}"
  echo "-----------------------------------"

  for seed in "${SEEDS[@]}"; do
    local splits_file="${SPLITS_PATTERN//\{seed\}/${seed}}"
    local out_json="${out_dir}/moe_seed_${seed}.json"
    local ckpt="${out_dir}/moe_seed_${seed}.pt"
    local log_file="${out_dir}/moe_seed_${seed}.log"

    if [[ ! -f "${splits_file}" ]]; then
      echo "[ERROR] Missing splits: ${splits_file}"
      exit 1
    fi

    echo "[RUN] ${label} seed=${seed}"
    # shellcheck disable=SC2086
    python3 scripts/train_moe.py \
      --experts_config "${experts_config}" \
      --splits "${splits_file}" \
      --split_type train_val_test \
      --retrain_only \
      --retrain_on_full \
      --no_early_stopping \
      --out_json "${out_json}" \
      --save_checkpoint "${ckpt}" \
      ${PARAMS} \
      ${extra_cli} \
      2>&1 | tee "${log_file}"
  done
}

mkdir -p "${OUT_ROOT}"

if [[ ! -f "${PARAMS_JSON}" ]]; then
  echo "[ERROR] Missing ${PARAMS_JSON}"
  exit 1
fi

PARAMS=$(extract_moe_cli_from_json)

echo "============================================================"
echo "Ablations | seed-7 best hyperparameters | 10 split seeds"
echo "PARAMS_JSON=${PARAMS_JSON}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "MODALITY=${RUN_MODALITY_ABLATION} TOPK=${RUN_TOPK_ABLATION} GATE=${RUN_GATE_ABLATION}"
echo "============================================================"

if [[ "${RUN_MODALITY_ABLATION}" == "true" ]]; then
  for entry in "${MODALITY_ABLATIONS[@]}"; do
    tag="${entry%%:*}"
    yaml="${entry##*:}"
    run_ablation_loop "${tag}" "${yaml}" ""
  done
fi

if [[ "${RUN_TOPK_ABLATION}" == "true" ]]; then
  for k in "${TOPK_VALUES[@]}"; do
    run_ablation_loop "topk_${k}" "${FULL_EXPERTS_CONFIG}" "--topk ${k}"
  done
fi

if [[ "${RUN_GATE_ABLATION}" == "true" ]]; then
  for entry in "${GATE_ABLATIONS[@]}"; do
    tag="${entry%%:*}"
    gate_arg="${entry##*:}"
    run_ablation_loop "${tag}" "${FULL_EXPERTS_CONFIG}" "--gate_ablation ${gate_arg}"
  done
fi

if [[ "${RUN_AGGREGATION}" == "true" ]]; then
  echo ""
  echo "============================================================"
  echo "[AGGREGATE]"
  echo "============================================================"

  aggregate_dir() {
    local name=$1
    local d="${OUT_ROOT}/${name}"
    [[ -d "${d}" ]] || return
    local n
    n=$(find "${d}" -maxdepth 1 -name 'moe_seed_*.json' 2>/dev/null | wc -l)
    [[ "${n}" -gt 0 ]] || return
    python3 scripts/aggregate_train_test_val_seeds.py \
      --seed_jsons "${d}"/moe_seed_*.json \
      --out "${d}/moe_aggregated.json" \
      --is_moe
    echo "  -> ${d}/moe_aggregated.json"
  }

  if [[ "${RUN_MODALITY_ABLATION}" == "true" ]]; then
    for entry in "${MODALITY_ABLATIONS[@]}"; do
      aggregate_dir "${entry%%:*}"
    done
  fi
  if [[ "${RUN_TOPK_ABLATION}" == "true" ]]; then
    for k in "${TOPK_VALUES[@]}"; do
      aggregate_dir "topk_${k}"
    done
  fi
  if [[ "${RUN_GATE_ABLATION}" == "true" ]]; then
    for entry in "${GATE_ABLATIONS[@]}"; do
      aggregate_dir "${entry%%:*}"
    done
  fi
fi

echo ""
echo "[DONE] ${OUT_ROOT}"

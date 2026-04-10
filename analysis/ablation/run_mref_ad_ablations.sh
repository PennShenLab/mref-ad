#!/usr/bin/env bash
# =============================================================================
# mref-ad ablations using fixed best hyperparameters running for 10 data splits
# =============================================================================
#
# - Modality: swap experts YAML (drop modality or single-modality configs).
# - Expert sparsity: --topk on the full last-visit expert graph.
# - Gating architecture: flat region-only and flat modality-only (--gate_ablation), plus full
#   hierarchical gating (modality + region gates) via --use_hierarchical_gate.
#
# Outputs: ${OUT_ROOT}/<tag>/moe_seed_<seed>.json + moe_aggregated.json per tag.
#
# =============================================================================

set -euo pipefail

RUN_MODALITY_ABLATION=true
RUN_TOPK_ABLATION=true
RUN_GATE_ABLATION=true

SEEDS=(7 13 42 1234 2027 99 123 555 999 1337)
SPLITS_PATTERN="configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json"
PARAMS_JSON="${PARAMS_JSON:-configs/best_hyperparameters/mref_ad_best_trial.json}"
FULL_EXPERTS_CONFIG="configs/freesurfer_lastvisit_experts_files.yaml"
OUT_ROOT="${OUT_ROOT:-results/ablations_mref_ad}"

RUN_AGGREGATION=true

MODALITY_ABLATIONS=(
  "modality_no_amyloid:configs/ablation/freesurfer_lastvisit_no_amyloid.yaml"
  "modality_no_mri:configs/ablation/freesurfer_lastvisit_no_mri.yaml"
  "modality_no_demographic:configs/ablation/freesurfer_lastvisit_no_demographic.yaml"
  "modality_mri_only:configs/ablation/freesurfer_lastvisit_mri_only.yaml"
  "modality_amyloid_only:configs/ablation/freesurfer_lastvisit_amyloid_only.yaml"
)

TOPK_VALUES=(1 3 5)

# Exactly 3 gating runs: output tag | extra CLI (pipe separates tag from args).
GATE_RUNS=(
  "gate_region_only|--gate_ablation region_only"
  "gate_modality_only|--gate_ablation modality_only"
  "gate_hierarchical|--use_hierarchical_gate"
)

extract_moe_cli_from_json() {
  python3 - "${PARAMS_JSON}" << 'PYTHON'
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)

params = data.get("params", {})
if not params:
    raise SystemExit("No 'params' in JSON (expected Optuna export with top-level 'params')")

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
    local out_json="${out_dir}/mref_ad_seed_${seed}.json"
    local ckpt="${out_dir}/mref_ad_seed_${seed}.pt"
    local log_file="${out_dir}/mref_ad_seed_${seed}.log"

    if [[ ! -f "${splits_file}" ]]; then
      echo "[ERROR] Missing splits: ${splits_file}"
      exit 1
    fi

    echo "[RUN] ${label} seed=${seed}"
    # shellcheck disable=SC2086
    python3 scripts/mref-ad/train_moe.py \
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
echo "mref-ad ablations | PARAMS_JSON=${PARAMS_JSON}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "SEEDS=${SEEDS[*]}"
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
  for entry in "${GATE_RUNS[@]}"; do
    tag="${entry%%|*}"
    extra_cli="${entry#*|}"
    run_ablation_loop "${tag}" "${FULL_EXPERTS_CONFIG}" "${extra_cli}"
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
    n=$(find "${d}" -maxdepth 1 -name 'mref_ad_seed_*.json' 2>/dev/null | wc -l)
    [[ "${n}" -gt 0 ]] || return
    python3 analysis/utils/aggregate_train_test_val_seeds.py \
      --seed_jsons "${d}"/mref_ad_seed_*.json \
      --out "${d}/mref_ad_aggregated.json" \
      --is_moe
    echo "  -> ${d}/mref_ad_aggregated.json"
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
    for entry in "${GATE_RUNS[@]}"; do
      aggregate_dir "${entry%%|*}"
    done
  fi
fi

echo ""
echo "[DONE] ${OUT_ROOT}"

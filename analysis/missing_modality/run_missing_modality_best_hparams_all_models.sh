#!/usr/bin/env bash
# Train/eval missing-modality protocol (train_test_val) for MREF-AD MoE, Flex-MoE,
# FT-Transformer, MLP, and logistic regression using fixed best hyperparameters
# from mref-ad/configs/best_hyperparameters (override with HYPERPARAM_DIR).
#
# For each scenario: train on complete data, test with masked PET (fraction 1.0 on
# amy experts) or masked MRI (fraction 1.0 on mri experts), plus full test (fraction 0).
#
# Usage:
#   ./run_missing_modality_best_hparams_all_models.sh
#   SEEDS=7,42 HYPERPARAM_DIR=/path/to/best_hyperparameters ./run_missing_modality_best_hparams_all_models.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

HYPERPARAM_DIR="${HYPERPARAM_DIR:-${HOME}/mref-ad/mref-ad/configs/best_hyperparameters}"
EXPERTS_CONFIG="${EXPERTS_CONFIG:-configs/freesurfer_lastvisit_cv10_experts_files.yaml}"
# Default must not use ${VAR:-literal_{seed}.json}: bash brace expansion turns {seed}.json into {seed.json}.
DEFAULT_SPLITS_PATTERN='configs/splits_by_ptid_80_10_10_seed_{seed}.json'
SPLITS_PATTERN="${SPLITS_PATTERN:-$DEFAULT_SPLITS_PATTERN}"
OUT_DIR="${OUT_DIR:-results/missingness}"
SEEDS="${SEEDS:-7,13,42,1234,2027,99,123,555,999,1337}"
FRACTIONS_STR="${FRACTIONS:-0,1.0}"
TAG_PET="${TAG_PET:-pet_missing_train_test_val_bestparams}"
TAG_MRI="${TAG_MRI:-mri_missing_train_test_val_bestparams}"
PLOT_OUT="${PLOT_OUT:-${OUT_DIR}/plots/grouped_bar_test_f1_bestparams_10seeds}"

if [[ ! -d "$HYPERPARAM_DIR" ]]; then
  echo "[ERROR] HYPERPARAM_DIR not found: $HYPERPARAM_DIR" >&2
  exit 1
fi

run_models_for_tag () {
  local DROP_EXPERTS="$1"
  local TAG="$2"
  echo ""
  echo "===================================================="
  echo "[RUN] tag=${TAG} drop_experts=${DROP_EXPERTS}"
  echo "===================================================="
  for spec in "moe" "flex_moe" "baseline:ftt_all" "baseline:mlp_concat" "baseline:lr_all"; do
    local MODE EXTRA=()
    if [[ "$spec" == baseline:* ]]; then
      MODE="baseline"
      EXTRA=(--baseline "${spec#baseline:}")
    else
      MODE="$spec"
    fi
    echo ""
    echo "--- mode=${MODE} ${EXTRA[*]:-}"
    python -u scripts/missing_modality.py \
      --split_mode train_test_val \
      --experts_config "$EXPERTS_CONFIG" \
      --splits "$SPLITS_PATTERN" \
      --mode "$MODE" \
      "${EXTRA[@]}" \
      --seeds "$SEEDS" \
      --hyperparams_dir "$HYPERPARAM_DIR" \
      --drop_experts "$DROP_EXPERTS" \
      --fractions "$FRACTIONS_STR" \
      --out_dir "$OUT_DIR" \
      --tag "$TAG"
  done
}

run_models_for_tag "amy" "$TAG_PET"
run_models_for_tag "mri" "$TAG_MRI"

mkdir -p "$(dirname "$PLOT_OUT")"
echo ""
echo "[PLOT] Writing ${PLOT_OUT}.png / .pdf"
python scripts/plot_missingness.py \
  --plot_type bar \
  --group_labels "Full,PET missing,MRI missing" \
  --metric test_f1 \
  --title "" \
  --inputs \
    "${OUT_DIR}/${TAG_PET}/moe_moe_p0p00_aggregated.json" \
    "${OUT_DIR}/${TAG_PET}/moe_moe_p1p00_aggregated.json" \
    "${OUT_DIR}/${TAG_MRI}/moe_moe_p1p00_aggregated.json" \
  --label "MREF-AD (MoE)" \
  --inputs \
    "${OUT_DIR}/${TAG_PET}/flex_moe_p0p00_aggregated.json" \
    "${OUT_DIR}/${TAG_PET}/flex_moe_p1p00_aggregated.json" \
    "${OUT_DIR}/${TAG_MRI}/flex_moe_p1p00_aggregated.json" \
  --label "Flex-MoE" \
  --inputs \
    "${OUT_DIR}/${TAG_PET}/baseline_ftt_all_p0p00_aggregated.json" \
    "${OUT_DIR}/${TAG_PET}/baseline_ftt_all_p1p00_aggregated.json" \
    "${OUT_DIR}/${TAG_MRI}/baseline_ftt_all_p1p00_aggregated.json" \
  --label "FT-Transformer" \
  --inputs \
    "${OUT_DIR}/${TAG_PET}/baseline_mlp_concat_p0p00_aggregated.json" \
    "${OUT_DIR}/${TAG_PET}/baseline_mlp_concat_p1p00_aggregated.json" \
    "${OUT_DIR}/${TAG_MRI}/baseline_mlp_concat_p1p00_aggregated.json" \
  --label "MLP" \
  --inputs \
    "${OUT_DIR}/${TAG_PET}/baseline_lr_all_p0p00_aggregated.json" \
    "${OUT_DIR}/${TAG_PET}/baseline_lr_all_p1p00_aggregated.json" \
    "${OUT_DIR}/${TAG_MRI}/baseline_lr_all_p1p00_aggregated.json" \
  --label "Logistic regression" \
  --out "$PLOT_OUT"

echo ""
echo "[DONE] Aggregates under ${OUT_DIR}/${TAG_PET}/ and ${OUT_DIR}/${TAG_MRI}/"
echo "       Plot: ${PLOT_OUT}.png"

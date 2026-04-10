#!/usr/bin/env bash
# Quick checks for the Optuna tuning entrypoints (no full training).
# Run from the repository root:
#   bash scripts/optuna/smoke_test.sh

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

export PYTHONPATH="${ROOT}:${ROOT}/scripts"

failures=0
run() {
  local name=$1
  shift
  echo "=== ${name} ==="
  if "$@"; then
    echo "[OK] ${name}"
  else
    echo "[FAIL] ${name}"
    failures=$((failures + 1))
  fi
  echo
}

run "import optuna" python3 -c "import optuna; print('optuna', optuna.__version__)"

run "tune_flex_moe_seed7_optuna.py --help" \
  python3 scripts/optuna/tune_flex_moe_seed7_optuna.py --help

echo "=== train_baselines.py --help ==="
if err=$(python3 scripts/baselines/train_baselines.py --help 2>&1); then
  echo "[OK] train_baselines.py --help"
else
  ec=$?
  if echo "${err}" | grep -q "No module named 'utils'"; then
    echo "[SKIP] train_baselines.py — shared top-level utils/ not in repo (see REPRODUCIBILITY.md)."
  else
    echo "${err}"
    echo "[FAIL] train_baselines.py --help (exit ${ec})"
    failures=$((failures + 1))
  fi
fi
echo

echo "=== third_party/flex-moe (Flex-MoE training) ==="
if [[ -d third_party/flex-moe ]]; then
  echo "[OK] third_party/flex-moe present"
else
  echo "[WARN] third_party/flex-moe missing — clone Flex-MoE here or pass --flex_moe_root to train_flex_moe / tune_flex_moe"
fi
echo

echo "=== Example split file for seed 7 (Flex-MoE / tuning) ==="
S7="configs/splits/splits_by_ptid_80_10_10_seed_7.json"
if [[ -f "${S7}" ]]; then
  echo "[OK] found ${S7}"
else
  echo "[WARN] missing ${S7} — generate with data_preprocessing/make_splits.py (per seed)"
fi
echo

if [[ "${failures}" -ne 0 ]]; then
  echo "Completed with ${failures} failing step(s). Often the first fix is restoring shared utils/ (see REPRODUCIBILITY.md)."
  exit 1
fi
echo "All Optuna smoke checks passed."

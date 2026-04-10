# Evaluation (fixed hyperparameters, 10 seeds)

This folder is the right place for **post-tuning evaluation**: load **fixed best hyperparameters** (e.g. from Optuna, often tuned on one split RNG seed), then **train on train+val and report test metrics** on **each of 10 split seeds** so you get mean ± std across different PTID partitions.

## Workflow order

You can run **evaluation first** whenever you already ship committed JSON under **`configs/best_hyperparameters/`** (from a prior Optuna run). You do **not** need to re-run tuning.

1. **Splits** — `configs/splits/splits_by_ptid_80_10_10_seed_<id>.json` for all seeds you evaluate (default list is in each script).
2. **Params** — either:
   - **`configs/best_hyperparameters/*.json`** (defaults in the scripts below), or
   - **`results/optuna_*.json`** from your own tuning (override with `--params-file` where supported).
3. **Run** — from the **repository root** (with venv + deps). Prefer a one-time **editable install** so you do not need `PYTHONPATH`:

```bash
pip install -e .
# optional instead of pip install -e .:
# export PYTHONPATH="$(pwd):$(pwd)/scripts"

python analysis/evaluation/eval_rf.py
python analysis/evaluation/eval_lr.py
python analysis/evaluation/eval_mlp.py
python analysis/evaluation/eval_xgb.py
```

Default `--out-dir` values are `results/eval_rf`, `results/eval_lr`, `results/eval_mlp`, and `results/eval_xgb` (override as needed). Each run writes `summary.json` plus one JSON per split seed.

## mref-ad (MoE; different JSON format)

`run_optuna_best_seeds.sh` calls **`scripts/mref-ad/train_moe.py`** with **`results/optuna_moe_seed_<seed>.json`**-style files. The committed file **`configs/best_hyperparameters/mref_ad_best_trial.json`** uses a different schema (`params` / `value`); use **`analysis/evaluation/run_seed7_bestparams_all10.sh`** (or adapt JSON) to feed that schema into `train_moe.py`—see comments in `run_optuna_best_seeds.sh`.

## Parity with `train_baselines.py` (train_val_test)

- **Data** — Same as tuning: `load_experts_from_yaml`, `baselines.preprocessing._build_xy`, and the same split JSON keys (`train_ptids` / `val_ptids` / `test_ptids`). Each eval run fits on **train+val** and scores **test**, matching the default retrain path after Optuna in `baselines/sklearn_baselines.py` / `baselines/mlp.py`.
- **RandomForest** — `baselines.sklearn_baselines.make_random_forest_classifier` (same as Optuna / refit in `sklearn_baselines.train_val_test`); `random_state=utils.SEED` (falls back to `42` in eval if absent).
- **Logistic regression** — `make_logistic_regression` (shared with Optuna refit and `LogisticRegressionRunner`); eval fixes `C` from JSON while tuning searches `C`.
- **XGBoost** — `make_xgb_classifier`; eval passes **`tree_method`** (default `hist`) and `random_state=utils.SEED`. The Optuna objective omits `tree_method` (library default), so eval can differ slightly from tuning unless you align `tree_method`.
- **MLP** — `mlp_config_for_retrain` + `retrain_mlp_on_full`, same as the refit block in `mlp.train_val_test`. JSON **`epochs`** should be the final retrain epoch count from tuning.

## Requirements

Shared top-level **`utils`** (`load_experts_from_yaml`, `eval_multiclass_metrics`, …) must be importable. If imports fail, see **`REPRODUCIBILITY.md`**.

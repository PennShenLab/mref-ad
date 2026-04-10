# Hyperparameter tuning (Optuna)

After **local data** are in place, **experts YAML** points at your CSVs, and you have **80/10/10 split JSONs** (see `data_preprocessing/README.md` and `make_splits.py`), the usual workflow is: **tune on one reference split (e.g. seed 7)**, save best hyperparameters to JSON / SQLite, then **evaluate on other split seeds** with those fixed params (separate eval scripts or `analysis/evaluation/run_optuna_best_seeds.sh` for **mref-ad** with fixed Optuna JSONs).

## One driver script: `hyperparameter_tune_baselines.sh`

`scripts/optuna/hyperparameter_tune_baselines.sh` runs Optuna for:

- **Tabular baselines** — names like `xgb_all`, `lr_all`, `rf_all`, `mlp_concat`, `ftt_all` → `train_baselines.py` (`train_val_test`, per-seed split file).
- **Flex-MoE** — use the special name **`flex_moe`** in `BASELINES`. For each `SEEDS_TO_RUN` entry it runs `tune_flex_moe_seed7_optuna.py` with SQLite storage and writes `results/optuna_flex_moe_seed_<seed>_<trials>_best_trial.json`.

Edit the variables at the top of the script:

| Block | Role |
|--------|------|
| `BASELINES` | Include or omit **`flex_moe`**; reorder so Flex-MoE runs when you want (e.g. after CPU baselines). |
| `SEEDS_TO_RUN` | Same seeds as for baselines (e.g. `(7)` for seed-7-only search). |
| `FLEX_MOE_PARALLEL` | `true` = multi-GPU worker pool (needs `nvidia-smi`). `false` = one process, all trials (debug / CPU). |
| `FLEX_MOE_TOTAL_TRIALS`, `FLEX_MOE_WORKERS_PER_GPU`, `FLEX_MOE_ROOT`, … | Flex-MoE study size and paths. |

Example:

```bash
bash scripts/optuna/hyperparameter_tune_baselines.sh
```

## Other shell scripts

| Script | What it does |
|--------|----------------|
| **`analysis/evaluation/run_optuna_best_seeds.sh`** | **Not tuning** — trains **mref-ad** (`train_moe.py`) with fixed `results/optuna_moe_seed_<seed>.json` (retrain / checkpoints). Runs **after** mref-ad hyperparameters are chosen. |

## Environment

From the repository root (venv activated):

```bash
export PYTHONPATH="$(pwd):$(pwd)/scripts"
```

Shared helpers live in a top-level **`utils`** module (`set_seed`, `load_experts_from_yaml`, `load_splits`, …). If `python scripts/baselines/train_baselines.py --help` fails with `No module named 'utils'`, restore that module from your full project checkout (see **`REPRODUCIBILITY.md`**).

## Baselines CLI (single command)

Use one split JSON per run, e.g. `configs/splits/splits_by_ptid_80_10_10_seed_7.json`:

```bash
python scripts/baselines/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_80_10_10_seed_7.json \
  --baseline rf_all \
  --split_type train_val_test \
  --tune_trials 20 \
  --select_metric val_f1 \
  --out results/optuna_rf_smoke.json
```

Full example blocks (including MLP / XGB) are in the module docstring at the top of `train_baselines.py`. Downstream eval scripts may expect specific `results/` filenames—align `--out` or update eval defaults.

## Flex-MoE (manual / advanced)

Dependencies: **`third_party/flex-moe`**, split template with `{seed}`, GPU recommended when `FLEX_MOE_PARALLEL=true`.

You normally use **`flex_moe` in `hyperparameter_tune_baselines.sh`**. For ad-hoc runs:

```bash
python scripts/optuna/tune_flex_moe_seed7_optuna.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits_template 'configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json' \
  --seed 7 \
  --study_name optuna_flex_moe_smoke \
  --storage 'sqlite:///results/optuna_flex_moe_smoke.db' \
  --n_trials 2 \
  --total_trials 2 \
  --device 0
```

## Smoke test

```bash
bash scripts/optuna/smoke_test.sh
```

See **`REPRODUCIBILITY.md`** for known gaps (missing `utils`, local `configs/paths.yaml`, etc.).

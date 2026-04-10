# mref-ad

Multimodal modeling code and analysis helpers for ADNI-style imaging + amyloid + demographics workflows. **Cohort data are not included**; configure local paths and agreements separately.

## Documentation map

| Topic | Location |
|--------|----------|
| Environment, frozen deps, known import gaps | **`REPRODUCIBILITY.md`** |
| Exploration scripts, FreeSurfer mapping, **per-seed split JSON** | **`data_preprocessing/README.md`** |
| **Evaluation** (fixed params from `configs/best_hyperparameters/`, 10 seeds) | **`analysis/evaluation/README.md`** |
| **Optuna tuning** (baselines + Flex-MoE), prerequisites, smoke test | **`scripts/optuna/README.md`** |

Typical order: prepare **`data/`** → experts YAML → **`make_splits.py`** → (optional) Optuna tuning **or** use committed best params → **`analysis/evaluation/`** scripts for multi-seed test metrics.

## Quick start (environment)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The **editable install** (`pip install -e .`) registers the `baselines` package and the shared **`utils`** module at the repo root (`utils.py`). You do **not** need `export PYTHONPATH="$(pwd):$(pwd)/scripts"` for normal imports. If you skip `pip install -e .`, set `PYTHONPATH` to the repo root (and `scripts/` if needed for legacy paths).

Then run preprocessing smoke tests (`bash data_preprocessing/smoke_test.sh`) and Optuna entrypoint checks (`bash scripts/optuna/smoke_test.sh`) as described in the docs above.

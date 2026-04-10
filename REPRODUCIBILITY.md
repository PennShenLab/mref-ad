# Reproducibility notes

## Where should shared `utils` live?

Pipelines (`scripts/mref-ad/`, `scripts/baselines/`, `analysis/evaluation/`, Optuna drivers, etc.) all do `import utils` for **`load_experts_from_yaml`**, **`eval_multiclass_metrics`**, **`set_seed`**, splits helpers, and similar.

**Layout in this repo:** shared code lives at the repository root as **`utils.py`** (importable as `utils`). That keeps training (`scripts/`), analysis (`analysis/`), and preprocessing (`data_preprocessing/`) on equal footing. `pip install -e .` and `analysis/evaluation/paths.py` (repo root on `sys.path`) both resolve it.

| Alternative | Notes |
|-------------|--------|
| **`utils/` package** at root | Fine if you split the module later; add `utils/__init__.py` and drop `utils.py`. |
| **`scripts/utils.py`** | Avoids a top-level file but blurs “library” vs “CLI”; not used here anymore. |

**Longer-term:** a real project package (e.g. `mref_ad/`) under `src/` avoids the generic name `utils` clashing with other environments and scales better—but it requires changing imports to `from mref_ad...`.

## Data preparation

Raw cohort data are **not** included in this repository. To reproduce the analysis, obtain data under applicable use agreements, populate **`data/`**, and configure paths (see **`configs/paths.yaml`** if you use `make_splits.py`). A concise step-by-step for exploration scripts, expert YAML preparation, and **per-seed 80/10/10 split JSON** generation is in **`data_preprocessing/README.md`**.

## Environment used for smoke tests (2026-04-08)

- **OS**: Linux x86_64  
- **Python**: 3.12.3  
- **Local venv**: `python3 -m venv .venv` at the repository root (not committed; listed in `.gitignore`).  
- **Install**: `source .venv/bin/activate && pip install -r requirements.txt && pip install -e .`  
  - `pip install -e .` registers `baselines` and root `utils` (`utils.py` or `utils/`) so `PYTHONPATH` is optional for imports.  
  - `requirements.txt` was produced with `pip freeze` after installing the dependency set used across training, baselines, Optuna, and analysis scripts.

**Torch**: The frozen `torch==2.11.0` build on Linux pulled CUDA 13–related wheels from PyPI (`nvidia-*`, `cuda-*`, `triton`). If you need a smaller CPU-only install, install `torch` from [pytorch.org](https://pytorch.org/) and adjust or trim those lines in a local constraints file.

**Conda**: `environment.yml` runs `pip install -e .` then `pip install -r requirements.txt`. Run `conda env create -f environment.yml` from the repo root.

## Smoke tests (CLI import / `--help`)

| Command | Result |
|--------|--------|
| `bash scripts/optuna/smoke_test.sh` | **OK** — Optuna import, `tune_flex_moe_seed7_optuna.py --help`; `train_baselines.py --help` is **SKIP** if shared `utils` is missing. Warns if `third_party/flex-moe` or seed-7 split JSON is absent. |
| `python analysis/utils/compare_models_paired_stats.py --help` | **OK** — argparse help prints. |
| `PYTHONPATH=<repo>:<repo>/scripts python scripts/mref-ad/train_moe.py --help` | **FAIL** if root **`utils.py`** (or `utils/`) is missing — mref-ad (`train_moe.py`) needs `set_seed`, `load_experts_from_yaml`, `load_splits`, metrics, `SEED`, etc. |
| `python scripts/baselines/train_baselines.py --help` | **FAIL** — `ModuleNotFoundError: No module named 'utils'` until you restore the **shared** top-level `utils` module (`set_seed`, `load_experts_from_yaml`, `load_splits`, metrics, …). The old name clash with `scripts/baselines/utils.py` is removed: that file is now `scripts/baselines/device_util.py` (device helper only). |

## What you need for full training reproduction

1. **Restore the shared `utils` module** used by mref-ad (`train_moe.py`), `train_baselines.py`, eval scripts, and Flex-MoE helpers (see comments in those files).  
2. **`train_baselines.py` path setup** — the repo root and `scripts/` are prepended to `sys.path` before importing the shared `utils` module.  
3. **Data and split JSONs** under `configs/` and paths referenced in YAML experts configs (e.g. `data/...`) are local to your study and subject to data-use agreements; this repository does not ship cohort data.

## Suggested verification after restoring `utils`

```bash
source .venv/bin/activate
pip install -e .   # if not already (optional: export PYTHONPATH="$(pwd):$(pwd)/scripts" instead)
python scripts/mref-ad/train_moe.py --help
python scripts/baselines/train_baselines.py --help
```

Then rerun a single short training command you used for the paper (one fold, few epochs) to confirm end-to-end behavior.

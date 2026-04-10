# mref-ad ablation sweeps

This folder holds drivers that retrain **mref-ad** (`scripts/mref-ad/train_moe.py`) under fixed hyperparameters while changing one axis at a time (modality graph, top‑k routing, or gating architecture).

## Main script: `run_mref_ad_ablations.sh`

From the **repository root**:

```bash
bash analysis/ablation/run_mref_ad_ablations.sh
```

### What stays fixed

- **Hyperparameters** come from **`PARAMS_JSON`** (default: `configs/best_hyperparameters/mref_ad_best_trial.json`). The script turns the JSON `params` object into `train_moe.py` CLI flags.
- **Training recipe**: `train_val_test` splits, `--retrain_only`, `--retrain_on_full`, `--no_early_stopping` (same idea as the multi-seed eval driver under `analysis/evaluation/`).

### Split seeds

Each ablation tag is run for **10** PTID split files:

`7 13 42 1234 2027 99 123 555 999 1337`

Pattern: `configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json`  
Those JSONs are **local** (patient assignments); generate them yourself and keep them out of git (see repo `.gitignore`).

### Output layout

Default **`OUT_ROOT`**: `results/ablations_mref_ad` (override with env).

For each ablation **tag**:

- `${OUT_ROOT}/<tag>/mref_ad_seed_<seed>.json` — metrics per split seed  
- `${OUT_ROOT}/<tag>/mref_ad_seed_<seed>.pt` / `.log` — checkpoint and log (logs are gitignored if present)  
- `${OUT_ROOT}/<tag>/mref_ad_aggregated.json` — aggregated across seeds (`analysis/utils/aggregate_train_test_val_seeds.py --is_moe`) when aggregation is enabled

### Ablation families (edit toggles in the script)

At the top of `run_mref_ad_ablations.sh`, **`RUN_MODALITY_ABLATION`**, **`RUN_TOPK_ABLATION`**, **`RUN_GATE_ABLATION`**, and **`RUN_AGGREGATION`** turn whole blocks on or off.

| Block | Tags | What changes |
|--------|------|----------------|
| **Modality** | `modality_no_amyloid`, `modality_no_mri`, `modality_no_demographic`, `modality_mri_only`, `modality_amyloid_only` | Experts YAML under `configs/ablation/` (drop a modality or single-modality subsets). |
| **Top‑k** | `topk_1`, `topk_3`, `topk_5` | `--topk` on the **full** last-visit expert graph (`configs/freesurfer_lastvisit_experts_files.yaml`). |
| **Gating** (3 runs) | `gate_region_only`, `gate_modality_only`, `gate_hierarchical` | `gate_region_only` / `gate_modality_only`: flat MoE with `--gate_ablation …`. `gate_hierarchical`: `--use_hierarchical_gate` (modality gate + region gates). |

### Environment overrides

| Variable | Default | Meaning |
|----------|---------|---------|
| `PARAMS_JSON` | `configs/best_hyperparameters/mref_ad_best_trial.json` | Optuna-style JSON with top-level `params`. |
| `OUT_ROOT` | `results/ablations_mref_ad` | Root directory for all tags above. |

Example:

```bash
PARAMS_JSON=configs/best_hyperparameters/mref_ad_hidden145_trial.json \
OUT_ROOT=results/ablations_mref_ad_h145 \
bash analysis/ablation/run_mref_ad_ablations.sh
```

### See also

- Trainer behavior and ablation semantics: `scripts/mref-ad/train_moe.py`  
- Multi-seed eval without these ablation grids: `analysis/evaluation/eval_mref_ad.sh` (and `analysis/evaluation/README.md`)

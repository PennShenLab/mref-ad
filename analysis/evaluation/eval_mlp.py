#!/usr/bin/env python3
"""Evaluate MLP (concat) with fixed hyperparameters across 10 split seeds (train+val → test).

How to run (from the repository root)::

    source .venv/bin/activate   # optional
    pip install -e .            # once per venv; wires ``baselines`` / ``utils`` without PYTHONPATH
    python analysis/evaluation/eval_mlp.py

If you do not use the editable install, use ``export PYTHONPATH="$(pwd):$(pwd)/scripts"`` instead.

Requires **PyTorch**, the shared ``utils`` module, local data for ``--experts-config``, and one split JSON
per seed (default template ``configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json``). Hyperparameters
are read from ``--params-file`` (default ``configs/best_hyperparameters/mlp_best_trial.json``, key
``mlp_concat_all.best_params``). For each split seed the model is trained on **train+val** with
``baselines.mlp.retrain_mlp_on_full`` (same helper as the refit step in ``baselines.mlp.train_val_test``),
then evaluated on **test**.

**Outputs** (under ``--out-dir``, default ``results/eval_mlp/``; directory is created if missing):

* ``summary.json`` — mean/pstdev of test acc, F1, AUC across seeds, plus ``per_seed`` metrics and
  ``fixed_best_params`` / ``params_source``.
* ``mlp_seed_<split_seed>.json`` — one file per split RNG seed (same 10 seeds as ``SEEDS`` in this module),
  each with ``mlp_concat_all.best_params`` and ``test_metrics`` for that split.

Stdout logs each seed run and prints the aggregate ``mean_std`` JSON block (same structure as in
``summary.json``). ``mlp_config_for_retrain`` builds the same ``MLPConfig`` as ``train_val_test``’s refit
step; ``utils.SEED`` is passed through; ``retrain_mlp_on_full`` seeds PyTorch and batch shuffling so two
runs with the same inputs match on CPU (GPU may still vary slightly).

Example with overrides::

    python analysis/evaluation/eval_mlp.py \\
      --params-file results/tuning_mlp_concat.json \\
      --out-dir results/eval_mlp

Use ``python analysis/evaluation/eval_mlp.py --help`` for all flags. See ``analysis/evaluation/README.md``.
"""
import argparse

_ARGPARSE_DESCRIPTION = (
    "Evaluate MLP (concat) with fixed hyperparameters across 10 split seeds (train+val → test)."
)
import json
import statistics
from pathlib import Path

from paths import ensure_repo_imports

ensure_repo_imports()

SEEDS = [7, 13, 42, 1234, 2027, 99, 123, 555, 999, 1337]


def main():
    ap = argparse.ArgumentParser(
        description=_ARGPARSE_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run from repo root: pip install -e . && python analysis/evaluation/eval_mlp.py\n"
            "(Or set PYTHONPATH to repo:scripts instead of pip install -e .)\n"
            "Defaults: --params-file configs/best_hyperparameters/mlp_best_trial.json"
        ),
    )
    ap.add_argument(
        "--params-file",
        type=str,
        default="configs/best_hyperparameters/mlp_best_trial.json",
        help="JSON with mlp_concat_all.best_params (e.g. Optuna export).",
    )
    ap.add_argument("--experts-config", type=str, default="configs/freesurfer_lastvisit_experts_files.yaml")
    ap.add_argument(
        "--splits-template",
        type=str,
        default="configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="results/eval_mlp",
        help="Directory for outputs: summary.json plus mlp_seed_<seed>.json per split (created if missing).",
    )
    args = ap.parse_args()

    import numpy as np

    from utils import SEED, eval_multiclass_metrics, load_experts_from_yaml
    from baselines.preprocessing import _build_xy
    from baselines.mlp import (
        mlp_config_for_retrain,
        retrain_mlp_on_full,
        load_mlp_from_state,
        predict_proba_mlp,
    )

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.params_file, "r") as f:
        best = json.load(f)["mlp_concat_all"]["best_params"]

    df, groups, _ = load_experts_from_yaml(args.experts_config)
    cols = [c for _, feat in groups.items() for c in feat]

    retrain_epochs = int(best["epochs"])
    cfg = mlp_config_for_retrain(
        best,
        retrain_epochs=retrain_epochs,
        seed=int(SEED),
        early_stop_metric="val_loss",
        patience=10,
    )

    accs, f1s, aucs = [], [], []
    per_seed = {}
    for seed in SEEDS:
        print(f"[RUN] seed={seed}", flush=True)
        with open(args.splits_template.format(seed=seed), "r") as f:
            splits = json.load(f)
        ptid_col = "PTID" if "PTID" in df.columns else "ptid"
        tr_idx = df.index[df[ptid_col].isin(splits["train_ptids"])].tolist()
        va_idx = df.index[df[ptid_col].isin(splits["val_ptids"])].tolist()
        te_idx = df.index[df[ptid_col].isin(splits["test_ptids"])].tolist()

        Xtr, ytr, scaler = _build_xy(df, cols, tr_idx, None)
        Xva, yva, _ = _build_xy(df, cols, va_idx, scaler)
        Xte, yte, _ = _build_xy(df, cols, te_idx, scaler)

        X_all = np.vstack([Xtr, Xva])
        y_all = np.concatenate([ytr, yva])

        state = retrain_mlp_on_full(X_all, y_all, config=cfg, device=None, epochs=retrain_epochs)
        model = load_mlp_from_state(Xte.shape[1], state, config=cfg)
        proba = predict_proba_mlp(model, Xte)
        m = eval_multiclass_metrics(yte, proba)

        item = {"auc": float(m["auc"]), "acc": float(m["acc"]), "f1": float(m["f1"])}
        print(f"[DONE] seed={seed} acc={item['acc']:.4f} f1={item['f1']:.4f} auc={item['auc']:.4f}", flush=True)
        per_seed[str(seed)] = item
        accs.append(item["acc"])
        f1s.append(item["f1"])
        aucs.append(item["auc"])

        with open(OUT_DIR / f"mlp_seed_{seed}.json", "w") as f:
            json.dump({"mlp_concat_all": {"best_params": best, "test_metrics": item}}, f, indent=2)

    summary = {
        "params_source": str(Path(args.params_file).resolve()),
        "fixed_best_params": best,
        "per_seed": per_seed,
        "mean_std": {
            "acc_mean": sum(accs) / len(accs),
            "acc_std": statistics.pstdev(accs),
            "f1_mean": sum(f1s) / len(f1s),
            "f1_std": statistics.pstdev(f1s),
            "auc_mean": sum(aucs) / len(aucs),
            "auc_std": statistics.pstdev(aucs),
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["mean_std"], indent=2))


if __name__ == "__main__":
    main()

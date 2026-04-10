#!/usr/bin/env python3
"""Evaluate random forest (concat) with fixed hyperparameters across 10 split seeds (train+val → test).

How to run (from the repository root)::

    source .venv/bin/activate   # optional
    pip install -e .            # once per venv; wires ``baselines`` / ``utils`` without PYTHONPATH
    python analysis/evaluation/eval_rf.py

If you do not use the editable install, use ``export PYTHONPATH="$(pwd):$(pwd)/scripts"`` instead.

Requires the shared ``utils`` module, local data for ``--experts-config``, and one split JSON per seed
(default template ``configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json``). Hyperparameters are read
from ``--params-file`` (default ``configs/best_hyperparameters/rf_best_trial.json``, key
``rf_concat_all.best_params``). For each split seed, ``baselines.sklearn_baselines.make_random_forest_classifier``
builds the same forest as ``train_val_test`` / ``train_baselines`` (``random_state=utils.SEED``), fit on
**train+val**, then evaluated on **test**.

**Outputs** (under ``--out-dir``, default ``results/eval_rf/``; directory is created if missing):

* ``summary.json`` — mean/pstdev of test acc, F1, AUC across seeds, plus ``per_seed`` metrics and
  ``fixed_best_params`` / ``params_source``.
* ``rf_seed_<split_seed>.json`` — one file per split RNG seed (same 10 seeds as ``SEEDS`` in this module),
  each with ``rf_concat_all.best_params`` and ``test_metrics`` for that split.

Stdout prints the aggregate ``mean_std`` JSON block (same structure as in ``summary.json``).

Example with overrides::

    python analysis/evaluation/eval_rf.py \\
      --params-file results/tuning_rf_concat.json \\
      --out-dir results/eval_rf

Use ``python analysis/evaluation/eval_rf.py --help`` for all flags. See ``analysis/evaluation/README.md``.
"""
import argparse

_ARGPARSE_DESCRIPTION = (
    "Evaluate random forest (concat) with fixed hyperparameters across 10 split seeds (train+val → test)."
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
            "Run from repo root: pip install -e . && python analysis/evaluation/eval_rf.py\n"
            "(Or set PYTHONPATH to repo:scripts instead of pip install -e .)\n"
            "Defaults: --params-file configs/best_hyperparameters/rf_best_trial.json"
        ),
    )
    ap.add_argument(
        "--params-file",
        type=str,
        default="configs/best_hyperparameters/rf_best_trial.json",
        help="JSON with rf_concat_all.best_params (e.g. Optuna export).",
    )
    ap.add_argument(
        "--experts-config",
        type=str,
        default="configs/freesurfer_lastvisit_experts_files.yaml",
    )
    ap.add_argument(
        "--splits-template",
        type=str,
        default="configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="results/eval_rf",
        help="Directory for outputs: summary.json plus rf_seed_<seed>.json per split (created if missing).",
    )
    args = ap.parse_args()

    import numpy as np

    import utils
    from baselines.preprocessing import _build_xy
    from baselines.sklearn_baselines import make_random_forest_classifier
    from utils import eval_multiclass_metrics, load_experts_from_yaml

    _rf_seed = int(getattr(utils, "SEED", 42))

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.params_file, "r") as f:
        best = json.load(f)["rf_concat_all"]["best_params"]

    df, groups, _ = load_experts_from_yaml(args.experts_config)
    cols = [c for _, feat in groups.items() for c in feat]

    accs, f1s, aucs = [], [], []
    per_seed = {}

    for seed in SEEDS:
        with open(args.splits_template.format(seed=seed), "r") as f:
            splits = json.load(f)
        ptid_col = "PTID" if "PTID" in df.columns else "ptid"
        tr_idx = df.index[df[ptid_col].isin(splits["train_ptids"])].tolist()
        va_idx = df.index[df[ptid_col].isin(splits["val_ptids"])].tolist()
        te_idx = df.index[df[ptid_col].isin(splits["test_ptids"])].tolist()

        Xtr, ytr, scaler = _build_xy(df, cols, tr_idx, None)
        Xva, yva, _ = _build_xy(df, cols, va_idx, scaler)
        Xte, yte, _ = _build_xy(df, cols, te_idx, scaler)

        Xfit = np.vstack([Xtr, Xva])
        yfit = np.concatenate([ytr, yva])

        clf = make_random_forest_classifier(
            n_estimators=int(best["n_estimators"]),
            max_depth=int(best["max_depth"]),
            min_samples_split=int(best["min_samples_split"]),
            n_jobs=-1,
            seed=int(_rf_seed),
        )
        clf.fit(Xfit, yfit)
        proba = clf.predict_proba(Xte)
        m = eval_multiclass_metrics(yte, proba)
        item = {"auc": float(m["auc"]), "acc": float(m["acc"]), "f1": float(m["f1"])}
        per_seed[str(seed)] = item
        accs.append(item["acc"])
        f1s.append(item["f1"])
        aucs.append(item["auc"])

        with open(OUT_DIR / f"rf_seed_{seed}.json", "w") as f:
            json.dump({"rf_concat_all": {"best_params": best, "test_metrics": item}}, f, indent=2)

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

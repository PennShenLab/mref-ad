#!/usr/bin/env python3
"""Evaluate XGBoost (concat) with fixed hyperparameters across 10 split seeds (train+val → test).

How to run (from the repository root)::

    source .venv/bin/activate   # optional
    pip install -e .            # once per venv; wires ``baselines`` / ``utils`` without PYTHONPATH
    python analysis/evaluation/eval_xgb.py

If you do not use the editable install, use ``export PYTHONPATH="$(pwd):$(pwd)/scripts"`` instead.

Requires **xgboost**, the shared ``utils`` module, local data for ``--experts-config``, and one split JSON
per seed (default template ``configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json``). Hyperparameters
are read from ``--params-file`` (default ``configs/best_hyperparameters/xgb_best_trial.json``, key
``xgb_concat_all.best_params``). For each split seed, ``XGBClassifier`` is fit on **train+val** and
scored on **test** (``objective=multi:softprob``, ``tree_method`` from ``--tree-method``, default
``hist``). The worker uses ``make_xgb_classifier`` from ``baselines.sklearn_baselines`` with ``random_state=utils.SEED``.
Tuned ``subsample`` / ``colsample_bytree`` below 1.0 use RNG noise; that noise is fixed given ``random_state`` and the training matrix.
For stricter reproducibility across machines and XGBoost builds, pass ``--deterministic`` to force ``n_jobs=1`` inside each fit (avoids parallel ``hist`` ordering effects). Fully deterministic greedy boosting would require ``subsample=1`` and ``colsample_bytree=1``, which is a different model than the tuned hyperparameters.

**Outputs** (under ``--out-dir``, default ``results/eval_xgb/``; directory is created if missing):

* ``summary.json`` — mean/pstdev of test acc, F1, AUC, ``per_seed`` metrics, ``fixed_best_params``,
  ``params_source``, plus ``tree_method``, ``n_jobs``, ``effective_n_jobs_per_model``, and ``workers``.
* ``xgb_seed_<split_seed>.json`` — one file per split RNG seed (same 10 seeds as ``SEEDS``), each with
  ``xgb_concat_all.best_params`` and ``test_metrics``.

By default all seeds run in parallel (``--workers`` defaults to 10). With multiple workers, if
``--n-jobs`` is ``-1``, each subprocess uses ``n_jobs=1`` for XGBoost to limit CPU oversubscription;
set ``--n-jobs`` explicitly to override.

Stdout prints per-seed ``[DONE]`` lines (from workers), ``[PREP]`` / ``[RUN]`` from the parent, and the
aggregate ``mean_std`` JSON block.

Example with overrides::

    python analysis/evaluation/eval_xgb.py \\
      --params-file results/tuning_xgb_concat.json \\
      --out-dir results/eval_xgb \\
      --workers 1

Use ``python analysis/evaluation/eval_xgb.py --help`` for all flags. See ``analysis/evaluation/README.md``
for parity notes vs ``train_baselines`` / Optuna (e.g. ``tree_method``).
"""
import argparse

_ARGPARSE_DESCRIPTION = (
    "Evaluate XGBoost (concat) with fixed hyperparameters across 10 split seeds (train+val → test)."
)
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from paths import ensure_repo_imports

ensure_repo_imports()

SEEDS = [7, 13, 42, 1234, 2027, 99, 123, 555, 999, 1337]


def _xgb_train_eval_one(args_tuple):
    """Picklable worker: fit XGB on train+val, evaluate on test; write per-seed JSON."""
    from baselines.sklearn_baselines import make_xgb_classifier
    from utils import eval_multiclass_metrics

    (
        seed,
        Xfit,
        yfit,
        Xte,
        yte,
        best,
        tree_method,
        n_jobs,
        out_dir_str,
        booster_seed,
    ) = args_tuple

    clf = make_xgb_classifier(
        n_estimators=int(best["n_estimators"]),
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        tree_method=tree_method,
        n_jobs=n_jobs,
        random_state=int(booster_seed),
    )
    clf.fit(Xfit, yfit)
    proba = clf.predict_proba(Xte)
    m = eval_multiclass_metrics(yte, proba)
    item = {"auc": float(m["auc"]), "acc": float(m["acc"]), "f1": float(m["f1"])}
    out_dir = Path(out_dir_str)
    with open(out_dir / f"xgb_seed_{seed}.json", "w") as f:
        json.dump({"xgb_concat_all": {"best_params": best, "test_metrics": item}}, f, indent=2)
    print(
        f"[DONE] seed={seed} acc={item['acc']:.4f} f1={item['f1']:.4f} auc={item['auc']:.4f}",
        flush=True,
    )
    return seed, item


def main():
    parser = argparse.ArgumentParser(
        description=_ARGPARSE_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Run from repo root: pip install -e . && python analysis/evaluation/eval_xgb.py\n"
            "(Or set PYTHONPATH to repo:scripts instead of pip install -e .)\n"
            "Defaults: --params-file configs/best_hyperparameters/xgb_best_trial.json"
        ),
    )
    parser.add_argument(
        "--params-file",
        type=str,
        default="configs/best_hyperparameters/xgb_best_trial.json",
        help="JSON with xgb_concat_all.best_params (e.g. Optuna export or results/optuna_*.json).",
    )
    parser.add_argument(
        "--experts-config",
        type=str,
        default="configs/freesurfer_lastvisit_experts_files.yaml",
    )
    parser.add_argument(
        "--splits-template",
        type=str,
        default="configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/eval_xgb",
        help="Directory for outputs: summary.json plus xgb_seed_<seed>.json per split (created if missing).",
    )
    parser.add_argument(
        "--tree-method",
        type=str,
        default="hist",
        help="XGBoost tree_method (default: hist — fast on CPU).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="n_jobs for each XGBoost fit (-1 = all cores). With --workers>1, default -1 is "
        "remapped to 1 per process to avoid CPU oversubscription; set explicitly to override.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Force n_jobs=1 for every XGBoost fit (single-thread). Reduces run-to-run variance from "
        "parallel histogram aggregation with tree_method=hist; slower but more reproducible.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel seed processes (default: all %d seeds at once)." % len(SEEDS),
    )
    args = parser.parse_args()

    import numpy as np

    from baselines.preprocessing import _build_xy
    from utils import SEED, load_experts_from_yaml

    params_path = Path(args.params_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(params_path, "r") as f:
        best = json.load(f)["xgb_concat_all"]["best_params"]

    workers = args.workers if args.workers is not None else len(SEEDS)
    workers = max(1, min(workers, len(SEEDS)))

    xgb_n_jobs = args.n_jobs
    if args.deterministic:
        xgb_n_jobs = 1
        print(
            "[INFO] --deterministic: using n_jobs=1 per XGBoost fit for reproducibility.",
            flush=True,
        )
    elif workers > 1 and args.n_jobs == -1:
        xgb_n_jobs = 1
        print(
            "[INFO] --workers>1 and --n-jobs=-1: using n_jobs=1 per XGBoost fit "
            "(set --n-jobs explicitly to use more threads per model).",
            flush=True,
        )

    df, groups, _ = load_experts_from_yaml(args.experts_config)
    cols = [c for _, feat in groups.items() for c in feat]
    ptid_col = "PTID" if "PTID" in df.columns else "ptid"

    bundles = []
    for seed in SEEDS:
        print(f"[PREP] seed={seed}", flush=True)
        with open(args.splits_template.format(seed=seed), "r") as f:
            splits = json.load(f)
        tr_idx = df.index[df[ptid_col].isin(splits["train_ptids"])].tolist()
        va_idx = df.index[df[ptid_col].isin(splits["val_ptids"])].tolist()
        te_idx = df.index[df[ptid_col].isin(splits["test_ptids"])].tolist()

        Xtr, ytr, scaler = _build_xy(df, cols, tr_idx, None)
        Xva, yva, _ = _build_xy(df, cols, va_idx, scaler)
        Xte, yte, _ = _build_xy(df, cols, te_idx, scaler)

        Xfit = np.vstack([Xtr, Xva])
        yfit = np.concatenate([ytr, yva])
        bundles.append(
            (
                seed,
                Xfit,
                yfit,
                Xte,
                yte,
                best,
                args.tree_method,
                xgb_n_jobs,
                str(out_dir.resolve()),
                int(SEED),
            )
        )

    per_seed = {}
    if workers == 1:
        for b in bundles:
            print(f"[RUN] seed={b[0]}", flush=True)
            seed, item = _xgb_train_eval_one(b)
            per_seed[str(seed)] = item
    else:
        print(
            f"[RUN] parallel workers={workers} (cpu_count={os.cpu_count()})",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_xgb_train_eval_one, b): b[0] for b in bundles}
            for fut in as_completed(futures):
                seed, item = fut.result()
                per_seed[str(seed)] = item

    accs = [per_seed[str(s)]["acc"] for s in SEEDS]
    f1s = [per_seed[str(s)]["f1"] for s in SEEDS]
    aucs = [per_seed[str(s)]["auc"] for s in SEEDS]

    summary = {
        "params_source": str(params_path.resolve()),
        "tree_method": args.tree_method,
        "n_jobs": args.n_jobs,
        "deterministic_single_thread": bool(args.deterministic),
        "effective_n_jobs_per_model": xgb_n_jobs,
        "workers": workers,
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
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["mean_std"], indent=2))


if __name__ == "__main__":
    main()

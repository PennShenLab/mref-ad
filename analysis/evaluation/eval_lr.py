#!/usr/bin/env python3
"""Evaluate logistic regression (fixed C) across 10 split seeds (train+val → test).

How to run (from the repository root)::

    source .venv/bin/activate   # optional
    pip install -e .            # once per venv; wires ``baselines`` / ``utils`` without PYTHONPATH
    python analysis/evaluation/eval_lr.py

If you do not use the editable install, use ``export PYTHONPATH="$(pwd):$(pwd)/scripts"`` instead.

Requires the shared top-level ``utils`` module, local data for ``--experts-config``, and one split JSON
per seed (default template ``configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json``). Hyperparameter ``C``
is read from ``--params-file`` (default ``configs/best_hyperparameters/logreg_best_trial.json``, key
``lr_concat_all.best_params``).

**Outputs** (under ``--out-dir``, default ``results/eval_lr/``; directory is created if missing):

* ``summary.json`` — mean/std of test acc, F1, AUC across seeds, plus ``per_seed`` metrics and
  ``fixed_best_params`` / ``params_source``.
* ``lr_seed_<split_seed>.json`` — one file per split RNG seed (same 10 seeds as ``SEEDS`` in this module),
  each with ``lr_concat_all.best_params`` and ``test_metrics`` for that split.

Stdout prints the same aggregate block as the ``mean_std`` field in ``summary.json``, and start/end lines
that list the output directory and paths written.

Example with overrides::

    python analysis/evaluation/eval_lr.py \\
      --params-file results/optuna_lr_all_seed_7.json \\
      --out-dir results/eval_lr

Use ``python analysis/evaluation/eval_lr.py --help`` for all flags. See
``analysis/evaluation/README.md`` for parity with ``train_baselines.py``.
"""
import argparse

_ARGPARSE_DESCRIPTION = "Evaluate logistic regression (fixed C) across 10 split seeds (train+val → test)."
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
            "Run from repo root: pip install -e . && python analysis/evaluation/eval_lr.py\n"
            "(Or set PYTHONPATH to repo:scripts instead of pip install -e .)\n"
            "Defaults: --params-file configs/best_hyperparameters/logreg_best_trial.json"
        ),
    )
    ap.add_argument(
        "--params-file",
        type=str,
        default="configs/best_hyperparameters/logreg_best_trial.json",
        help="JSON with lr_concat_all.best_params (e.g. Optuna export).",
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
        default="results/eval_lr",
        help="Directory for outputs: summary.json plus lr_seed_<seed>.json per split (created if missing).",
    )
    args = ap.parse_args()

    import numpy as np

    from baselines.preprocessing import _build_xy
    from baselines.sklearn_baselines import make_logistic_regression
    from utils import SEED, eval_multiclass_metrics, load_experts_from_yaml

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_abs = OUT_DIR.resolve()
    print(f"[eval_lr] Writing results under: {out_abs}", flush=True)
    print(
        f"[eval_lr] Per-seed files: {out_abs / 'lr_seed_<seed>.json'} "
        f"({len(SEEDS)} seeds); summary: {out_abs / 'summary.json'}",
        flush=True,
    )

    with open(args.params_file, "r") as f:
        params = json.load(f)["lr_concat_all"]["best_params"]
    cval = float(params["C"])

    df, groups, _ = load_experts_from_yaml(args.experts_config)
    cols = [c for _, feat in groups.items() for c in feat]

    per_seed = {}
    accs, f1s, aucs = [], [], []

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

        # Same estimator as baselines/sklearn_baselines (Optuna refit + LogisticRegressionRunner).
        clf = make_logistic_regression(C=cval, seed=SEED)
        clf.fit(Xfit, yfit)
        proba = clf.predict_proba(Xte)
        m = eval_multiclass_metrics(yte, proba)
        item = {"auc": float(m["auc"]), "acc": float(m["acc"]), "f1": float(m["f1"])}
        per_seed[str(seed)] = item
        aucs.append(item["auc"])
        accs.append(item["acc"])
        f1s.append(item["f1"])

        per_path = OUT_DIR / f"lr_seed_{seed}.json"
        with open(per_path, "w") as f:
            json.dump({"lr_concat_all": {"best_params": {"C": cval}, "test_metrics": item}}, f, indent=2)
        print(f"[eval_lr] Wrote {per_path.resolve()}", flush=True)

    summary = {
        "params_source": str(Path(args.params_file).resolve()),
        "fixed_best_params": {"C": cval},
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
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval_lr] Wrote {summary_path.resolve()}", flush=True)
    print("[eval_lr] Aggregate test metrics (mean / pstdev across split seeds):", flush=True)
    print(json.dumps(summary["mean_std"], indent=2), flush=True)
    print(
        f"[eval_lr] Done. {len(SEEDS)}× lr_seed_*.json + 1× summary.json in {out_abs}",
        flush=True,
    )


if __name__ == "__main__":
    main()

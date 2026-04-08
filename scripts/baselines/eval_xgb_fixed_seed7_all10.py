#!/usr/bin/env python3
import argparse
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xgboost as xgb

import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils import load_experts_from_yaml, eval_multiclass_metrics  # noqa: E402
from baselines.preprocessing import _build_xy  # noqa: E402


SEEDS = [7, 13, 42, 1234, 2027, 99, 123, 555, 999, 1337]
EXPERTS_CONFIG = "configs/freesurfer_lastvisit_cv10_experts_files.yaml"
SPLITS_TEMPLATE = "configs/splits_by_ptid_80_10_10_seed_{seed}.json"


def _xgb_train_eval_one(args_tuple):
    """Picklable worker: fit XGB on train+val, evaluate on test; write per-seed JSON."""
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
    ) = args_tuple
    clf = xgb.XGBClassifier(
        n_estimators=int(best["n_estimators"]),
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
        subsample=float(best["subsample"]),
        colsample_bytree=float(best["colsample_bytree"]),
        tree_method=tree_method,
        n_jobs=n_jobs,
        use_label_encoder=False,
        verbosity=0,
        objective="multi:softprob",
        random_state=42,
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
        description="Train XGBoost on train+val per split seed; report test metrics (10 seeds)."
    )
    parser.add_argument(
        "--params-file",
        type=str,
        default="results/optuna_xgb_all_seed_7_no_retrain.json",
        help="JSON from Optuna with xgb_concat_all.best_params (default: lighter no-retrain study).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/xgb_seed7_no_retrain_best_all10",
        help="Directory for per-seed JSON and summary.json.",
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
        "--workers",
        type=int,
        default=None,
        help="Number of parallel seed processes (default: all %d seeds at once)." % len(SEEDS),
    )
    args = parser.parse_args()

    params_path = Path(args.params_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(params_path, "r") as f:
        best = json.load(f)["xgb_concat_all"]["best_params"]

    workers = args.workers if args.workers is not None else len(SEEDS)
    workers = max(1, min(workers, len(SEEDS)))

    xgb_n_jobs = args.n_jobs
    if workers > 1 and args.n_jobs == -1:
        xgb_n_jobs = 1
        print(
            "[INFO] --workers>1 and --n-jobs=-1: using n_jobs=1 per XGBoost fit "
            "(set --n-jobs explicitly to use more threads per model).",
            flush=True,
        )

    df, groups, _ = load_experts_from_yaml(EXPERTS_CONFIG)
    cols = [c for _, feat in groups.items() for c in feat]
    ptid_col = "PTID" if "PTID" in df.columns else "ptid"

    bundles = []
    for seed in SEEDS:
        print(f"[PREP] seed={seed}", flush=True)
        with open(SPLITS_TEMPLATE.format(seed=seed), "r") as f:
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
            (seed, Xfit, yfit, Xte, yte, best, args.tree_method, xgb_n_jobs, str(out_dir.resolve()))
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
        "effective_n_jobs_per_model": xgb_n_jobs,
        "workers": workers,
        "best_params_seed7": best,
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

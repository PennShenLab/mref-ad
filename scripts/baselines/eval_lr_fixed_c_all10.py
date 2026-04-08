#!/usr/bin/env python3
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils import load_experts_from_yaml, eval_multiclass_metrics  # noqa: E402
from baselines.preprocessing import _build_xy  # noqa: E402


SEEDS = [7, 13, 42, 1234, 2027, 99, 123, 555, 999, 1337]
EXPERTS_CONFIG = "configs/freesurfer_lastvisit_cv10_experts_files.yaml"
SPLITS_TEMPLATE = "configs/splits_by_ptid_80_10_10_seed_{seed}.json"
PARAMS_FILE = "results/optuna_lr_all_seed_7.json"
OUT_DIR = Path("results/lr_seed7_bestparams_all10")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(PARAMS_FILE, "r") as f:
        params = json.load(f)["lr_concat_all"]["best_params"]
    cval = float(params["C"])

    df, groups, _ = load_experts_from_yaml(EXPERTS_CONFIG)
    cols = [c for _, feat in groups.items() for c in feat]

    per_seed = {}
    accs, f1s, aucs = [], [], []

    for seed in SEEDS:
        with open(SPLITS_TEMPLATE.format(seed=seed), "r") as f:
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

        clf = LogisticRegression(C=cval, max_iter=2000, n_jobs=1, random_state=42)
        clf.fit(Xfit, yfit)
        proba = clf.predict_proba(Xte)
        m = eval_multiclass_metrics(yte, proba)
        item = {"auc": float(m["auc"]), "acc": float(m["acc"]), "f1": float(m["f1"])}
        per_seed[str(seed)] = item
        aucs.append(item["auc"])
        accs.append(item["acc"])
        f1s.append(item["f1"])

        with open(OUT_DIR / f"lr_seed_{seed}.json", "w") as f:
            json.dump({"lr_concat_all": {"best_params": {"C": cval}, "test_metrics": item}}, f, indent=2)

    summary = {
        "params_source": PARAMS_FILE,
        "best_params_seed7": {"C": cval},
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

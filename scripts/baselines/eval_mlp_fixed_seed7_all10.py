#!/usr/bin/env python3
import json
import statistics
from pathlib import Path

import numpy as np

import sys
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils import load_experts_from_yaml, eval_multiclass_metrics  # noqa: E402
from baselines.preprocessing import _build_xy  # noqa: E402
from baselines.mlp import MLPConfig, retrain_mlp_on_full, load_mlp_from_state, predict_proba_mlp  # noqa: E402


SEEDS = [7, 13, 42, 1234, 2027, 99, 123, 555, 999, 1337]
EXPERTS_CONFIG = "configs/freesurfer_lastvisit_cv10_experts_files.yaml"
SPLITS_TEMPLATE = "configs/splits_by_ptid_80_10_10_seed_{seed}.json"
PARAMS_FILE = "results/optuna_mlp_concat_seed_7.json"
OUT_DIR = Path("results/mlp_seed7_bestparams_all10")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_FILE, "r") as f:
        best = json.load(f)["mlp_concat_all"]["best_params"]

    df, groups, _ = load_experts_from_yaml(EXPERTS_CONFIG)
    cols = [c for _, feat in groups.items() for c in feat]

    cfg = MLPConfig(
        hidden=int(best["hidden"]),
        drop=float(best["drop"]),
        epochs=int(best["epochs"]),
        batch_size=int(best["batch"]),
        lr=float(best["lr"]),
        weight_decay=float(best["wd"]),
        patience=int(best["epochs"]) + 1,
        early_stop_metric="val_loss",
    )

    accs, f1s, aucs = [], [], []
    per_seed = {}
    for seed in SEEDS:
        print(f"[RUN] seed={seed}", flush=True)
        with open(SPLITS_TEMPLATE.format(seed=seed), "r") as f:
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

        state = retrain_mlp_on_full(X_all, y_all, config=cfg, device=None, epochs=int(best["epochs"]))
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
        "params_source": PARAMS_FILE,
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
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["mean_std"], indent=2))


if __name__ == "__main__":
    main()

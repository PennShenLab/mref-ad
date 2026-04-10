#!/usr/bin/env python3
"""
usage:
 python -u scripts/train_baselines.py \
  --experts_config configs/experts_files.yaml \
  --baseline mlp_all \
  --splits configs/splits/splits_by_ptid.json

 python -u scripts/train_baselines.py \
  --experts_config configs/freesurfer_experts_files.yaml \
  --baseline mlp_all \
  --splits configs/splits/splits_by_ptid.json

# 5-fold CV
python scripts/train_baselines.py \
  --experts_config configs/freesurfer_experts_files.yaml \
  --splits configs/splits/splits_by_ptid.json \
  --baseline mlp_all --split_type cv5

# 5-fold CV with last-visit-only splits
python scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_lastvisit.json \
  --baseline mlp_all \
  --split_type cv5 \
  --out results/baselines_lastvisit_cv_folds.json

# 10-fold CV with last-visit-only splits
python scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_lastvisit.json \
  --baseline mlp_all \
  --split_type cv5 \
  --out results/baselines_lastvisit_folds.json \
  --early_stop_metric val_loss

# Random Forest
python scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_lastvisit.json \
  --baseline rf_all \
  --split_type cv5 \
  --out results/baselines_rf_lastvisit.json \
    --save_sklearn_models

# XGBoost
python scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_lastvisit.json \
  --baseline xgb_all \
  --split_type cv5 \
  --out results/baselines_xgb_lastvisit.json \
  --save_sklearn_models

# Logistic regression
python scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_lastvisit.json \
  --baseline lr_all \
  --split_type cv5 \
  --out results/baselines_lr_lastvisit.json \
    --save_sklearn_models

# FT Transformer
python -u scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_lastvisit.json \
  --baseline ftt \
  --split_type cv5 \
  --early_stop_metric val_loss \
  --out results/baselines_ftt_cv.json

# Hyperparameter tuning with Optuna
# 100 trials for classical ML and 200 trials for NN based models (MLP, FTT)

python scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_80_10_10.json \
  --baseline rf_all \
  --split_type train_val_test\
  --tune_trials 100 \
  --select_metric val_f1 \
  --out results/optuna_rf.json

  python scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_80_10_10.json \
  --baseline xgb_all \
  --split_type train_val_test\
  --tune_trials 100 \
  --select_metric val_f1 \
  --out results/optuna_xgb.json

python scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_80_10_10.json \
  --baseline mlp_concat \
  --early_stop_metric val_loss \
  --split_type train_val_test\
  --tune_trials 200 \
  --select_metric val_f1 \
  --out results/optuna_mlp.json

# Get result of best hyperparameters
python scripts/train_baselines.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_80_10_10.json \
  --baseline lr_all \
  --split_type train_val_test \
  --retrain_on_full \
  --retrain_params scripts/best_baselines_params.json \
  --out results/retrain_lr_results.json

"""

import argparse, os, re, json, random
import sys
from pathlib import Path

# Path order: repo root first (shared `utils`), then `scripts/` (package `baselines`).
_scripts_dir = Path(__file__).resolve().parent.parent
_repo_root = _scripts_dir.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Additional imports for new baselines
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import pickle

import utils
from utils import (
    set_seed, load_experts_from_yaml, load_splits
)
# Use the canonical, shared metric implementations from repo-root `utils.py`
from utils import eval_multiclass_metrics, eval_confusion_report, macro_auroc

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# When the script is run as `python scripts/train_baselines.py` the
# interpreter sets sys.path[0] to the `scripts/` folder. That means
# sibling top-level packages (like `baselines/`) are not discoverable
import baselines
from baselines import registry as baselines_registry
from baselines import runners as baselines_runners

# Adapter to use baselines/sklearn_baselines.py
def train_eval_sklearn_baselines(baseline_name: str, df: pd.DataFrame, cols: List[str], tr_idx: List[int], va_idx: List[int], *, seed: int = 42, n_jobs: int = 1) -> dict:
    """Backward-compatible thin wrapper for sklearn baselines.

    Delegates to `baselines.runners.train_eval_sklearn_baselines` so the
    script itself contains no sklearn training logic.
    """
    return baselines_runners.train_eval_sklearn_baselines(baseline_name, df, cols, tr_idx, va_idx, seed=seed, n_jobs=n_jobs)

# ---------- MLP baselines (capacity-matched to mref-ad MoE experts) ----------
# class MLP(nn.Module):
#     def __init__(self, d, hidden=128, drop=0.2, n_classes=3):
#         super().__init__()
#         self.fc1 = nn.Linear(d, hidden)
#         self.do  = nn.Dropout(drop)
#         self.fc2 = nn.Linear(hidden, n_classes)
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.do(x)
#         return self.fc2(x)

# The canonical `MLP` implementation lives in `baselines.mlp.MLP` and training
# helpers are exposed via `baselines.mlp.fit_mlp`. We removed the duplicate
# definition here to avoid drift and ensure a single source of truth.


# Preprocessing helpers (median impute + StandardScaler) are provided by
# `baselines.preprocessing`. The implementations were moved there to keep
# `scripts/train_baselines.py` a thin orchestration layer.

# ----------------------------
# Helper: Concatenate feature columns (with optional has_{expert} flags)
# ----------------------------
def _concat_cols(groups, df=None, include_has_flags: bool = False):
    """Return concatenated feature columns across all experts.

    If include_has_flags=True, also append `has_{expert}` indicator columns when present.
    """
    cols = [c for _, feat in groups.items() for c in feat]
    if include_has_flags:
        for name in groups.keys():
            h = f"has_{name}"
            if df is None or (hasattr(df, "columns") and h in df.columns):
                cols.append(h)
    # de-duplicate while preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def train_eval_mlp(df, cols, tr_idx, va_idx, hidden=128, drop=0.2, epochs=50, lr=1e-3, wd=1e-4, batch=128, device=None, patience=5, early_stop_metric="val_auc"):
    return baselines_runners.train_eval_mlp(df, cols, tr_idx, va_idx, hidden=hidden, drop=drop, epochs=epochs, lr=lr, wd=wd, batch=batch, device=device, patience=patience, early_stop_metric=early_stop_metric)

def run_single_modality_MLP(df, groups, mod, tr_idx, va_idx, **kw):
    return baselines_runners.run_single_modality_MLP(df, groups, mod, tr_idx, va_idx, **kw)

def run_concat_MLP(df, groups, tr_idx, va_idx, **kw):
    return baselines_runners.run_concat_MLP(df, groups, tr_idx, va_idx, **kw)

def run_latefusion_MLP(df, groups, tr_idx, va_idx, early_stop_metric=None):
    return baselines_runners.run_latefusion_MLP(df, groups, tr_idx, va_idx, early_stop_metric=early_stop_metric)
# --------------------------------------------------------------------

def run_single_modality_LR(df, groups, mod, tr_idx, va_idx):
    return baselines_runners.run_single_modality_LR(df, groups, mod, tr_idx, va_idx)

def run_concat_LR(df, groups, tr_idx, va_idx):
    return baselines_runners.run_concat_LR(df, groups, tr_idx, va_idx)

def run_latefusion(df, groups, tr_idx, va_idx):
    return baselines_runners.run_latefusion(df, groups, tr_idx, va_idx)

# No in-script FT-Transformer training helpers: use `baselines.ftt.fit_ftt`
# through the registry/runners. The script is intentionally thin.


# The custom, in-script FT-Transformer trainer was removed in favor of the
# canonical implementation in `baselines.ftt.fit_ftt`. Use that
# function (exposed via the runners registry) or dispatch through
# the runtime registry which maps 'ftt' to the official trainer.
# --------------------------------------------------------------------

def _attach_confusion(r: dict, class_names=None) -> dict:
    """Attach confusion_report to a baseline result dict if proba/yva are present."""
    if class_names is None:
        class_names = ["CN", "MCI", "AD"]
    try:
        if r is None:
            return r
        proba = r.get("proba", None)
        yva = r.get("yva", None)
        if proba is None or yva is None:
            return r
        r["confusion_report"] = eval_confusion_report(yva, proba, class_names=class_names)
    except Exception as e:
        print(f"[WARN] could not compute confusion_report: {e}")
    return r

def _save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def _run_holdout(args, df, groups, classes):
    """Run the 'holdout' split_type branch (single train/val split).

    This extracts the previous inline logic and keeps behavior identical.
    """
    tr, va = load_splits(args.splits, df)
    print(f"[INFO] using fixed splits: train={len(tr)} | test={len(va)}")

    mods = list(groups.keys())
    print(f"[INFO] train={len(tr)} | val={len(va)} | modalities: {', '.join(mods)}")

    results = {}

    import baselines as baselines_pkg
    from baselines import registry as baselines_registry

    run_baselines = baselines_pkg.build_run_baselines(args.baseline, mods)

    for key, btype, mod in run_baselines:
        runner = baselines_registry.get_runner(btype)
        if runner is None:
            raise ValueError(f"Unknown baseline type: {btype}")
        if mod is not None:
            r = runner(df, groups, tr, va, mod=mod, early_stop_metric=args.early_stop_metric, seed=utils.SEED)
        else:
            r = runner(df, groups, tr, va, early_stop_metric=args.early_stop_metric, seed=utils.SEED)

        r = _attach_confusion(r, classes)
        results[key] = {
            "auc": r.get("val_auc", r.get("auc", float("nan"))),
            "acc": r.get("val_acc", float("nan")),
            "bacc": r.get("val_bacc", float("nan")),
            "f1": r.get("val_f1", float("nan")),
            "confusion_report": r.get("confusion_report", None),
        }
        print(f"[{key}] AUC={results[key]['auc']:.4f}")

    out_path = args.out if args.out != "results/baselines.json" else f"results/baselines_{args.baseline}.json"
    _save_json(out_path, results)
    print(f"[INFO] wrote {out_path}\n{json.dumps(results, indent=2)}")


def _run_train_val_test(args, df, groups, classes):
    """Run hyperparameter tuning on train/val, retrain on train+val, evaluate on test.

    Implements Optuna-based tuning for MLP and FT-Transformer baselines.
    For other baselines this falls back to a simple holdout evaluation.
    """
    try:
        import optuna
    except Exception:
        raise RuntimeError("optuna is required for --split_type train_val_test tuning. Install with `pip install optuna`.")

    if args.splits is None:
        raise ValueError("--split_type train_val_test requires --splits pointing to a splits JSON file")
    with open(args.splits, 'r') as sf:
        splits = json.load(sf)

    for k in ("train_ptids", "val_ptids", "test_ptids"):
        if k not in splits:
            raise ValueError(f"splits JSON missing required key: {k}")

    ptid_col = "PTID" if "PTID" in df.columns else "ptid"
    tr_ptids = splits["train_ptids"]
    va_ptids = splits["val_ptids"]
    test_ptids = splits["test_ptids"]

    tr_idx = df.index[df[ptid_col].isin(tr_ptids)].tolist()
    va_idx = df.index[df[ptid_col].isin(va_ptids)].tolist()
    test_idx = df.index[df[ptid_col].isin(test_ptids)].tolist()

    import baselines as baselines_pkg
    run_baselines = baselines_pkg.build_run_baselines(args.baseline, list(groups.keys()))

    results = {}
    n_trials = int(getattr(args, "tune_trials", 20))
    timeout = None if getattr(args, "tune_timeout", None) is None else float(args.tune_timeout)

    # Parse optional retrain parameters (JSON string or path to JSON file)
    retrain_params_obj = None
    if getattr(args, "retrain_params", None):
        rp = args.retrain_params
        import os
        # If rp is a path to a file, load it; else interpret as JSON string
        if os.path.isfile(rp):
            with open(rp, 'r') as rf:
                retrain_params_obj = json.load(rf)
        else:
            retrain_params_obj = json.loads(rp)
    for key, btype, mod in run_baselines:
        print(f"\n[INFO] Processing baseline {key} (type={btype})")
        # Prefer per-baseline train_val_test implementations inside baselines.* modules.
        # Map baseline type to a baselines module that may implement train_val_test.
        module_name = None
        if btype.startswith("mlp") or key.startswith("mlp"):
            module_name = "mlp"
        elif btype.startswith("ftt") or key.startswith("ftt"):
            module_name = "ftt"
        elif btype in {"rf", "xgb", "lr"} or key in {"rf_all", "xgb_all", "lr_all"}:
            module_name = "sklearn_baselines"

        handled = False
        # Eval-only support for sklearn joblib models: load saved model + meta and evaluate on test
        if getattr(args, "eval_only", False) and module_name == "sklearn_baselines":
            import os, joblib, pickle
            from baselines.preprocessing import _build_xy
            # resolve checkpoint path: prefer args.ckpt, else default final model path
            ckpt_path = getattr(args, "ckpt", None)
            if ckpt_path is None:
                ckpt_path = f"results/models/{args.baseline}/{key}/final/model.joblib"
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"eval_only requested but model not found at {ckpt_path}")
            
            loaded = joblib.load(ckpt_path)
            # unwrap SklearnFitResult if present
            clf = getattr(loaded, "model", loaded)
            
            # locate meta
            meta_path = None
            cand1 = os.path.splitext(ckpt_path)[0] + ".meta.pkl"
            cand2 = os.path.join(os.path.dirname(ckpt_path), "model.meta.pkl")
            if os.path.isfile(cand1):
                meta_path = cand1
            elif os.path.isfile(cand2):
                meta_path = cand2

            cols_meta = None
            scaler_meta = None
            if meta_path is not None and os.path.isfile(meta_path):
                with open(meta_path, "rb") as mf:
                    meta = pickle.load(mf)
                cols_meta = meta.get("cols", None)
                scaler_meta = meta.get("scaler", None)

            # build Xte using saved preprocessing or fallback
            if cols_meta is None:
                cols_meta = _concat_cols(groups)
            Xte, yte, _ = _build_xy(df, cols_meta, test_idx, scaler_meta)
            
            proba = clf.predict_proba(Xte)
            m = eval_multiclass_metrics(yte, proba)
            test_metrics = {"auc": float(m.get("auc")), "acc": float(m.get("acc")), "f1": float(m.get("f1"))}

            results[key] = {"eval_only": True, "test_metrics": test_metrics, "model_path": ckpt_path, "meta_path": meta_path}
            handled = True
        
        if module_name is not None:
            modpkg = __import__(f"baselines.{module_name}", fromlist=[module_name])
            fn = getattr(modpkg, "train_val_test", None)
            if fn is not None:
                cols = _concat_cols(groups)
                out_dir = f"results/models/{args.baseline}/{key}/final"
                # pass baseline_name for sklearn helper which needs to know which estimator to tune
                if module_name == "sklearn_baselines":
                    # Use exact-match semantics for retrain params: prefer a dict keyed by the
                    # CLI baseline alias (e.g. 'lr_all'). If the user supplied an un-keyed
                    # params dict and we're running a single baseline, accept it as well.
                    params_for_key = None
                    # DEBUG: show retrain params and current baseline/run
                    print(f"[DEBUG_RETRAIN] retrain_params_keys={list(retrain_params_obj.keys()) if retrain_params_obj is not None else None} args.baseline={args.baseline} run_key={key} btype={btype}")
                    if retrain_params_obj is not None and isinstance(retrain_params_obj, dict):
                        # Exact match on the CLI baseline alias (required).
                        if args.baseline in retrain_params_obj:
                            params_for_key = retrain_params_obj[args.baseline]

                    # If user requested strict retrain-on-full mode, fail early when
                    # no matching params are found for this baseline/run.
                    if getattr(args, "retrain_on_full", False) and params_for_key is None:
                        raise RuntimeError(
                            f"--retrain_on_full set but no matching params found in --retrain_params for baseline '{args.baseline}' run '{key}'. Provide a JSON keyed by the CLI baseline name (e.g. '{args.baseline}': {{...}})."
                        )

                    if params_for_key is not None:
                        # retrain on TRAIN+VAL and evaluate on TEST using provided params
                        p = params_for_key or {}
                        skip_retrain = getattr(args, "skip_retrain", False)
                        train_mode = "TRAIN only (no retrain)" if skip_retrain else "TRAIN+VAL (retrain mode)"
                        print(f"[INFO] Using fixed sklearn params: training on {train_mode} with C={p.get('C', 1.0)}")
                        from baselines.sklearn_baselines import (
                            LOGISTIC_REGRESSION_MAX_ITER,
                            LogisticRegressionRunner,
                            RandomForestRunner,
                            XGBoostRunner,
                        )
                        from baselines.preprocessing import _build_xy
                        from utils import eval_multiclass_metrics

                        Xtr, ytr, scaler = _build_xy(df, cols, tr_idx, None)
                        Xva, yva, _ = _build_xy(df, cols, va_idx, scaler)
                        Xte, yte, _ = _build_xy(df, cols, test_idx, scaler)
                        
                        # If skip_retrain, train on TRAIN only; otherwise retrain on TRAIN+VAL
                        if skip_retrain:
                            X_fit, y_fit = Xtr, ytr
                        else:
                            X_fit = np.vstack([Xtr, Xva])
                            y_fit = np.concatenate([ytr, yva])

                        p = params_for_key or {}
                        runner = None
                        if btype.startswith("lr") or key.startswith("lr"):
                            runner = LogisticRegressionRunner(
                                C=p.get("C", 1.0),
                                max_iter=p.get("max_iter", LOGISTIC_REGRESSION_MAX_ITER),
                                n_jobs=p.get("n_jobs", 1),
                                seed=utils.SEED,
                            )
                        elif btype.startswith("rf") or key.startswith("rf"):
                            runner = RandomForestRunner(
                                n_estimators=p.get("n_estimators", 100),
                                max_depth=p.get("max_depth", None),
                                min_samples_split=p.get("min_samples_split", 2),
                                min_samples_leaf=p.get("min_samples_leaf", 1),
                                n_jobs=p.get("n_jobs", 1),
                                seed=utils.SEED,
                            )
                        elif btype.startswith("xgb") or key.startswith("xgb"):
                            runner = XGBoostRunner(
                                n_estimators=p.get("n_estimators", 100),
                                n_jobs=p.get("n_jobs", 1),
                                seed=utils.SEED,
                            )

                        if runner is None:
                            raise RuntimeError(f"Could not construct runner for baseline {key}; skipping retrain_params path.")

                        fit_res = runner.fit(X_fit, y_fit)
                        proba = runner.predict_proba(fit_res, Xte)
                        m = eval_multiclass_metrics(yte, proba)

                        import joblib, pickle
                        os.makedirs(out_dir, exist_ok=True)
                        model_path = os.path.join(out_dir, "model.joblib")
                        joblib.dump(fit_res.model, model_path)
                        meta = {"cols": cols, "scaler": scaler, "n_classes": int(len(np.unique(y_fit)))}
                        meta_path = os.path.join(out_dir, "model.meta.pkl")
                        with open(meta_path, "wb") as mf:
                            pickle.dump(meta, mf)

                        # Normalize params_for_key into explicit best/second-best fields.
                        best_params_val = None
                        second_best_params_val = None
                        if isinstance(params_for_key, dict):
                            # Common case: params_for_key is the best-params dict itself.
                            # Also support an object shaped like {"best_params":..., "second_best_params":...}
                            if "best_params" in params_for_key or "second_best_params" in params_for_key:
                                best_params_val = params_for_key.get("best_params")
                                second_best_params_val = params_for_key.get("second_best_params") or params_for_key.get("second_best")
                            elif "best" in params_for_key or "second_best" in params_for_key:
                                best_params_val = params_for_key.get("best")
                                second_best_params_val = params_for_key.get("second_best")
                            else:
                                best_params_val = params_for_key
                        elif isinstance(params_for_key, (list, tuple)) and len(params_for_key) > 0:
                            best_params_val = params_for_key[0]
                            if len(params_for_key) > 1:
                                second_best_params_val = params_for_key[1]
                        else:
                            best_params_val = params_for_key

                        res = {
                            "best_params": best_params_val,
                            "second_best_params": second_best_params_val,
                            "best_val_auc": float("nan"),
                            "best_val_metric_name": None,
                            "best_val_metric": float("nan"),
                            "best_val_loss": float("nan"),
                            "best_val_acc": float("nan"),
                            "best_val_bacc": float("nan"),
                            "best_val_f1": float("nan"),
                            "test_metrics": {"auc": float(m.get("auc", float("nan"))), "acc": float(m.get("acc", float("nan"))), "f1": float(m.get("f1", float("nan")))},
                            "model_path": model_path,
                            "meta_path": meta_path,
                            "n_trials": 0,
                            "retrain_only": True,
                        }
                        results[key] = res
                        handled = True
                    if not handled:
                        # Forward skip_retrain from the CLI so sklearn
                        # train_val_test can honor the user's intent to
                        # evaluate the TRAIN-only snapshot on TEST
                        # instead of refitting on TRAIN+VAL.
                        res = fn(
                            df,
                            cols,
                            tr_idx,
                            va_idx,
                            test_idx,
                            n_trials=n_trials,
                            timeout=timeout,
                            out_dir=out_dir,
                            baseline_name=key,
                            seed=utils.SEED,
                            select_metric=args.select_metric,
                            skip_retrain=(not getattr(args, "skip_retrain", False)) if False else getattr(args, "skip_retrain", False),
                        )
                        results[key] = res
                        handled = True
                else:
                    # Check if retrain_params provided for MLP/FTT:
                    # if so, train with fixed params and no early stopping instead of tuning
                    params_for_key = None
                    if retrain_params_obj is not None and isinstance(retrain_params_obj, dict):
                        if args.baseline in retrain_params_obj:
                            params_for_key = retrain_params_obj[args.baseline]
                    
                    if params_for_key is not None and (module_name == "mlp" or module_name == "ftt"):
                        # Train with fixed params, no tuning, no early stopping
                        print(f"[INFO] Using fixed {module_name.upper()} params: training on TRAIN only (no retrain, no early stopping, exactly epochs={params_for_key.get('epochs', 'N/A')})")
                        
                        if module_name == "mlp":
                            import torch
                            from baselines.preprocessing import _build_xy
                            from baselines.mlp import MLPConfig, fit_mlp
                            from utils import eval_multiclass_metrics
                            
                            Xtr, ytr, scaler = _build_xy(df, cols, tr_idx, None)
                            Xva, yva, _ = _build_xy(df, cols, va_idx, scaler)
                            Xte, yte, _ = _build_xy(df, cols, test_idx, scaler)
                            
                            # Extract params, use epochs from params, set patience >= epochs to disable early stopping
                            p = params_for_key or {}
                            epochs = int(p.get("epochs", 50))
                            cfg = MLPConfig(
                                hidden=int(p.get("hidden", 128)),
                                drop=float(p.get("drop", 0.2)),
                                epochs=epochs,
                                batch_size=int(p.get("batch", 128)),
                                lr=float(p.get("lr", 1e-3)),
                                weight_decay=float(p.get("wd", 1e-4)),
                                patience=epochs + 1,  # Set patience > epochs to disable early stopping: train for exactly epochs
                                early_stop_metric=args.early_stop_metric,
                            )
                            out = fit_mlp(Xtr, ytr, Xva, yva, config=cfg, metric_fn=eval_multiclass_metrics, device=None, verbose=True)
                            proba = out.get("proba_test", None)
                            if proba is None:
                                # If proba_test not available, evaluate on test manually
                                from baselines.mlp import load_mlp_from_state
                                model = load_mlp_from_state(Xte.shape[1], out.get("best_state_dict"), config=cfg, device=torch.device("cpu"))
                                proba = baselines.mlp.predict_proba_mlp(model, Xte, batch_size=256, device=torch.device("cpu"))
                            m = eval_multiclass_metrics(yte, proba)
                        else:  # ftt
                            import torch
                            from baselines.ftt import FTTConfig, fit_ftt
                            from baselines.preprocessing import _build_xy
                            from utils import eval_multiclass_metrics
                            
                            p = params_for_key or {}
                            epochs = int(p.get("epochs", 80))
                            cfg = FTTConfig(
                                n_classes=3,
                                epochs=epochs,
                                batch_size=int(p.get("batch", 128)),
                                lr=float(p.get("lr", 3e-4)),
                                weight_decay=float(p.get("weight_decay", 1e-3)),
                                patience=epochs + 1,  # Set patience > epochs to disable early stopping: train for exactly epochs
                                early_stop_metric=args.early_stop_metric,
                            )
                            # fit_ftt expects df, cols, tr_idx, va_idx (not preprocessed arrays)
                            out = fit_ftt(df, cols, tr_idx, va_idx, config=cfg, metric_fn=eval_multiclass_metrics, device=None, verbose=True)
                            
                            # Evaluate on test set
                            Xte, yte, _ = _build_xy(df, cols, test_idx, out.get("scaler"))
                            proba = out.get("proba_test", None)
                            if proba is None:
                                from baselines.ftt import load_ftt_from_state, predict_proba_ftt
                                model = load_ftt_from_state(Xte.shape[1], out.get("best_state_dict"), config=cfg, device=torch.device("cpu"))
                                proba = predict_proba_ftt(model, Xte, batch_size=256, device=torch.device("cpu"))
                            m = eval_multiclass_metrics(yte, proba)
                        
                        res = {
                            "best_params": params_for_key,
                            "best_val_auc": float(out.get("best_val_auc", float("nan"))),
                            "best_val_metric_name": "train_fixed",
                            "best_val_metric": float(out.get("best_val_metric", float("nan"))),
                            "best_val_loss": float(out.get("best_val_loss", float("nan"))),
                            "best_val_acc": float(out.get("best_val_acc", float("nan"))),
                            "best_val_bacc": float(out.get("best_val_bacc", float("nan"))),
                            "best_val_f1": float(out.get("best_val_f1", float("nan"))),
                            "test_metrics": {"auc": float(m.get("auc", float("nan"))), "acc": float(m.get("acc", float("nan"))), "f1": float(m.get("f1", float("nan")))},
                            "n_trials": 0,
                            "train_fixed_params": True,
                        }
                        results[key] = res
                        handled = True
                    else:
                        # Call per-baseline train_val_test without attempting to
                        # inject `retrain_on_full` which may not be supported by
                        # all baseline implementations (e.g. `baselines.ftt`).
                        print(f"[INFO] Running Optuna tuning with early stopping on TRAIN/VAL ({n_trials} trials, retrain_on_full={not getattr(args, 'skip_retrain', False)})")
                        res = fn(
                            df,
                            cols,
                            tr_idx,
                            va_idx,
                            test_idx,
                            n_trials=n_trials,
                            timeout=timeout,
                            out_dir=out_dir,
                            seed=utils.SEED,
                            early_stop_metric=args.early_stop_metric,
                            select_metric=args.select_metric,
                            retrain_on_full=(not getattr(args, "skip_retrain", False))
                        )
                        results[key] = res
                        handled = True

        if not handled:
            raise RuntimeError(f"No train_val_test implementation found for baseline type '{btype}' (key='{key}'). Ensure baselines.{module_name}.train_val_test exists.")

    out_path = args.out if args.out else f"results/tuning_{args.baseline}.json"
    _save_json(out_path, results)
    print(f"[INFO] wrote tuning results to {out_path}")
    return results


def _run_cv5(args, df, groups, classes):
    """Run 5-fold cross-validation branch extracted from main()."""
    with open(args.splits, "r") as f:
        splits_data = json.load(f)
    if "cv_splits_ptid" not in splits_data:
        raise ValueError("splits JSON file does not contain 'cv_splits_ptid' for cv5 split_type.")

    cv_splits_ptid = splits_data["cv_splits_ptid"]
    n_folds = len(cv_splits_ptid)
    mods = list(groups.keys())
    print(f"[INFO] Running 5-fold cross-validation with {n_folds} folds and modalities: {', '.join(mods)}")

    if args.only_fold is not None:
        if args.only_fold < 0 or args.only_fold >= n_folds:
            raise ValueError(f"--only_fold must be in [0, {n_folds-1}], got {args.only_fold}")
        fold_iter = [args.only_fold]
        print(f"[INFO] --only_fold set: running only fold {args.only_fold}")
    else:
        fold_iter = list(range(n_folds))

    fold_results = {}
    metrics_to_collect = ["val_auc", "val_acc", "val_bacc", "val_f1"]

    import baselines as baselines_pkg
    run_baselines = baselines_pkg.build_run_baselines(args.baseline, mods)

    metrics_storage = {key: {metric: [] for metric in metrics_to_collect} for key, _, _ in run_baselines}
    pooled_storage = {key: {"y": [], "proba": []} for key, _, _ in run_baselines}

    def _call_runner_and_get_result(btype, mod, runner, df, groups, tr_idx, va_idx, args):
        """Call the given runner and return the standardized result dict.

        This centralizes the eval-only logic for FT-Transformer checkpoints and
        falls back to normal runner invocation for other baselines.
        """
        # If eval-only requested and this is the FT-Transformer, load checkpoint and evaluate
        # without training. For all other cases (including non-eval runs) delegate to the
        # runner function and normalize the return value.
        if args.eval_only and btype == "ftt":
            if args.ckpt is None:
                raise ValueError("--eval_only requires --ckpt to be set to a checkpoint file path")
            # FT-Transformer eval-only path (saved torch state + .meta.pkl).
            import pickle, torch
            from baselines.ftt import load_ftt_from_state, predict_proba_ftt, transform_with_train_stats
            # Resolve ckpt: allow providing a base path and try baseline-suffixed variants
            def _resolve_ckpt_path(ckpt_path, baseline_name):
                import os
                # direct path
                if ckpt_path and os.path.isfile(ckpt_path):
                    return ckpt_path
                # try inserting baseline before extension: foldN -> foldN.<baseline>.pt
                if ckpt_path:
                    base, ext = os.path.splitext(ckpt_path)
                    if baseline_name:
                        candidate = f"{base}.{baseline_name}{ext}"
                        if os.path.isfile(candidate):
                            return candidate
                return ckpt_path

            ckpt_path = _resolve_ckpt_path(args.ckpt, getattr(args, "baseline", None))
            meta_path = os.path.splitext(ckpt_path)[0] + ".meta.pkl"
            if not os.path.isfile(ckpt_path):
                raise ValueError(f"Checkpoint not found: {ckpt_path}")
            meta = {}
            if os.path.isfile(meta_path):
                with open(meta_path, "rb") as mf:
                    meta = pickle.load(mf)

            cols = meta.get("cols", None)
            scaler = meta.get("scaler", None)
            train_means = meta.get("train_means", None)

            # build Xva using saved preprocessing; fallback to simple selection
            if cols is None or scaler is None or train_means is None:
                cols = [c for _, feat in groups.items() for c in feat]
                Xva = df.iloc[va_idx][cols].astype(float).to_numpy()
                yva = df.iloc[va_idx]["y"].values
            else:
                Xva = transform_with_train_stats(df, cols, va_idx, scaler=scaler, train_means=train_means)
                yva = df.iloc[va_idx]["y"].astype(int).values

            state = torch.load(args.ckpt, map_location="cpu")
            cfg = meta.get("model_config", None)
            model = load_ftt_from_state(Xva.shape[1], state, config=cfg, device=torch.device("cpu"))
            proba = predict_proba_ftt(model, Xva, batch_size=256, device=torch.device("cpu"))

            m = eval_multiclass_metrics(yva, proba)
            return {
                "val_auc": m.get("auc", float("nan")),
                "val_acc": m.get("acc", float("nan")),
                "val_bacc": m.get("bacc", float("nan")),
                "val_f1": m.get("f1", float("nan")),
                "proba": proba,
                "yva": yva,
            }

        # Eval-only support for MLP baselines: load checkpoint + meta and run prediction
        if args.eval_only and btype.startswith("mlp"):
            if args.ckpt is None:
                raise ValueError("--eval_only requires --ckpt to be set to a checkpoint file path")
            import pickle, torch
            from baselines.mlp import load_mlp_from_state, predict_proba_mlp, MLPConfig
            from baselines import preprocessing as preprocessing
            # Resolve ckpt with baseline-suffixed fallback
            def _resolve_ckpt_path(ckpt_path, baseline_name):
                import os
                if ckpt_path and os.path.isfile(ckpt_path):
                    return ckpt_path
                if ckpt_path:
                    base, ext = os.path.splitext(ckpt_path)
                    if baseline_name:
                        candidate = f"{base}.{baseline_name}{ext}"
                        if os.path.isfile(candidate):
                            return candidate
                return ckpt_path

            ckpt_path = _resolve_ckpt_path(args.ckpt, getattr(args, "baseline", None))
            meta_path = os.path.splitext(ckpt_path)[0] + ".meta.pkl"
            if not os.path.isfile(ckpt_path):
                raise ValueError(f"Checkpoint not found: {ckpt_path}")
            meta = {}
            if os.path.isfile(meta_path):
                with open(meta_path, "rb") as mf:
                    meta = pickle.load(mf)

            cols = meta.get("cols", None)
            scaler = meta.get("scaler", None)
            # build Xva using saved preprocessing; fallback to simple selection
            if cols is None or scaler is None:
                cols = [c for _, feat in groups.items() for c in feat]
                Xva = df.iloc[va_idx][cols].astype(float).to_numpy()
                yva = df.iloc[va_idx]["y"].values
            else:
                Xva, yva, _ = preprocessing._build_xy(df, cols, va_idx, scaler=scaler)

            state = torch.load(args.ckpt, map_location="cpu")
            cfg = meta.get("model_config", None)
            if cfg is None:
                cfg = MLPConfig()
            model = load_mlp_from_state(Xva.shape[1], state, config=cfg, device=torch.device("cpu"))
            proba = predict_proba_mlp(model, Xva, batch_size=256, device=torch.device("cpu"))

            m = eval_multiclass_metrics(yva, proba)
            return {
                "val_auc": m.get("auc", float("nan")),
                "val_acc": m.get("acc", float("nan")),
                "val_bacc": m.get("bacc", float("nan")),
                "val_f1": m.get("f1", float("nan")),
                "proba": proba,
                "yva": yva,
            }

        # Default: call the runner (training or non-FTT eval-only fallback).
        # Forward save_checkpoint and retrain_on_full where supported by runners.
        runner_kwargs = dict(early_stop_metric=args.early_stop_metric, seed=utils.SEED)
        if hasattr(args, "save_checkpoint") and args.save_checkpoint is not None:
            runner_kwargs["save_checkpoint"] = args.save_checkpoint
        if hasattr(args, "retrain_on_full") and getattr(args, "retrain_on_full", False):
            runner_kwargs["retrain_on_full"] = True

        if mod is not None:
            r = runner(df, groups, tr_idx, va_idx, mod=mod, **runner_kwargs)
        else:
            r = runner(df, groups, tr_idx, va_idx, **runner_kwargs)

        # Defensive: runners must return a dict-like result. If they return None,
        # raise a clear error pointing to the offending baseline type/modality so the
        # user can inspect the runner implementation or data causing the early exit.
        if r is None:
            raise RuntimeError(f"Runner for baseline type='{btype}' modality='{mod}' returned None. Expected a result dict.\n"
                               f"Context: fold tr_idx_len={len(tr_idx)} va_idx_len={len(va_idx)} eval_only={args.eval_only} ckpt={args.ckpt}")

        return r

    for fold_idx in fold_iter:
        fold_split = cv_splits_ptid[fold_idx]
        print(f"\n[INFO] Fold {fold_idx+1}/{n_folds}")
        tr_ptids = fold_split.get("train_ptids")
        va_ptids = fold_split.get("val_ptids")
        if tr_ptids is None or va_ptids is None:
            raise KeyError(
                f"Fold {fold_idx+1} in cv_splits_ptid missing 'train'/'val' keys. Expected 'train_ptids' and 'val_ptids'."
            )

        ptid_col = "PTID" if "PTID" in df.columns else "ptid"
        tr_idx = df.index[df[ptid_col].isin(tr_ptids)].tolist()
        va_idx = df.index[df[ptid_col].isin(va_ptids)].tolist()

        fold_results[fold_idx] = {}

        for key, btype, mod in run_baselines:
            from baselines import registry as baselines_registry
            runner = baselines_registry.get_runner(btype)
            if runner is None:
                raise ValueError(f"Unknown baseline type: {btype}")

            # Delegate to helper that handles eval-only & FT-Transformer checkpoint eval
            r = _call_runner_and_get_result(btype, mod, runner, df, groups, tr_idx, va_idx, args)

            r = _attach_confusion(r, classes)

            auc = r.get("val_auc", r.get("auc", float("nan")))
            acc = r.get("val_acc", float("nan"))
            bacc = r.get("val_bacc", float("nan"))
            f1 = r.get("val_f1", float("nan"))

            print(f"[{key}] Fold {fold_idx+1} AUC={auc:.4f}")

            fold_results[fold_idx][key] = {
                "auc": auc,
                "acc": acc,
                "bacc": bacc,
                "f1": f1,
                "confusion_report": r.get("confusion_report", None),
            }

            # Optionally save sklearn model objects and a small meta file per-fold.
            # The runners may return trained estimators under keys like 'model',
            # 'estimator', 'best_estimator' or 'clf'. If --save_sklearn_models is set
            # we persist them with joblib and write a companion .meta.pkl containing
            # columns and preprocessing objects (if provided by the runner result).
            if getattr(args, "save_sklearn_models", False):
                model_obj = r.get("model") or r.get("estimator") or r.get("best_estimator") or r.get("clf")
                if model_obj is not None:
                    out_dir = Path(f"results/models/{args.baseline}/{key}/fold{fold_idx}")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    model_path = out_dir / f"fold{fold_idx}.joblib"
                    try:
                        joblib.dump(model_obj, str(model_path))
                    except Exception as e:  # pragma: no cover
                        print(f"[WARN] failed to joblib.dump model for {key} fold{fold_idx}: {e}")

                    meta = {
                        "cols": _concat_cols(groups),
                        "scaler": r.get("scaler", None),
                        "model_config": r.get("model_config", None),
                        "n_classes": int(len(classes)) if classes is not None else None,
                    }
                    meta_path = out_dir / f"fold{fold_idx}.meta.pkl"
                    try:
                        with open(meta_path, "wb") as mf:
                            pickle.dump(meta, mf)
                    except Exception as e:  # pragma: no cover
                        print(f"[WARN] failed to write meta for {key} fold{fold_idx}: {e}")
                    print(f"[INFO] saved sklearn model to {model_path} and meta to {meta_path}")

            metrics_storage[key]["val_auc"].append(auc)
            metrics_storage[key]["val_acc"].append(acc)
            metrics_storage[key]["val_bacc"].append(bacc)
            metrics_storage[key]["val_f1"].append(f1)

            if r.get("proba", None) is not None and r.get("yva", None) is not None:
                pooled_storage[key]["proba"].append(np.asarray(r["proba"]))
                pooled_storage[key]["y"].append(np.asarray(r["yva"]))

    if args.only_fold is not None:
        out_data = {"only_fold": int(args.only_fold), "fold_results": fold_results}
        _save_json(args.out, out_data)
        print(f"[INFO] wrote {args.out}\n{json.dumps(out_data, indent=2)[:2000]}")
        return

    summary = {}
    print("\n[INFO] Cross-validation summary (mean ± std):")
    for key in metrics_storage:
        summary[key] = {}
        for metric in metrics_to_collect:
            arr = np.array(metrics_storage[key][metric])
            mean = np.nanmean(arr)
            std = np.nanstd(arr)
            summary[key][metric] = {"mean": mean, "std": std}
            print(f"{key} {metric}: {mean:.4f} ± {std:.4f}")

    pooled_metrics = {}
    for bkey in metrics_storage.keys():
        ys = pooled_storage[bkey]["y"]
        ps = pooled_storage[bkey]["proba"]
        if len(ys) == 0 or len(ps) == 0:
            pooled_metrics[bkey] = None
            continue
        y_all = np.concatenate(ys, axis=0)
        p_all = np.concatenate(ps, axis=0)
        m = eval_multiclass_metrics(y_all, p_all)
        pooled_metrics[bkey] = {
            "val_auc": float(m["auc"]),
            "val_acc": float(m["acc"]),
            "val_bacc": float(m["bacc"]),
            "val_f1": float(m["f1"]),
            "n": int(len(y_all)),
        }

    print("\n[INFO] Pooled (all-fold) metrics on concatenated out-of-fold predictions:")
    for bkey, pm in pooled_metrics.items():
        if pm is None:
            continue
        print(
            f"{bkey}: AUC={pm['val_auc']:.4f} | ACC={pm['val_acc']:.4f} | "
            f"BACC={pm['val_bacc']:.4f} | F1={pm['val_f1']:.4f} | n={pm['n']}"
        )

    confusion_summary = {}
    for bkey in metrics_storage.keys():
        cms = []
        for fidx in fold_results.keys():
            cr = fold_results[fidx].get(bkey, {}).get("confusion_report", None)
            if cr is None:
                continue
            cm_counts = cr.get("cm_counts", None)
            if cm_counts is None:
                continue
            cms.append(np.array(cm_counts, dtype=int))

        if len(cms) == 0:
            confusion_summary[bkey] = None
            continue

        cm_total = np.sum(np.stack(cms, axis=0), axis=0).astype(int)
        row_sums = cm_total.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0.0] = 1.0
        cm_row = (cm_total / row_sums).astype(float)

        tp = np.diag(cm_total).astype(float)
        col_sums = cm_total.sum(axis=0).astype(float)
        prec = np.divide(tp, col_sums, out=np.zeros_like(tp), where=(col_sums > 0))
        rec = np.divide(tp, row_sums.squeeze(1), out=np.zeros_like(tp), where=(row_sums.squeeze(1) > 0))
        f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(tp), where=((prec + rec) > 0))

        class_names = classes if isinstance(classes, list) else ["CN", "MCI", "AD"]
        confusion_summary[bkey] = {
            "cm_counts": cm_total.tolist(),
            "cm_row_norm": cm_row.tolist(),
            "per_class": {
                "precision": {class_names[i]: float(prec[i]) for i in range(len(class_names))},
                "recall": {class_names[i]: float(rec[i]) for i in range(len(class_names))},
                "f1": {class_names[i]: float(f1[i]) for i in range(len(class_names))},
            },
        }

    out_data = {
        "fold_results": fold_results,
        "summary": summary,
        "pooled_metrics": pooled_metrics,
        "confusion_report_all_folds": confusion_summary,
    }

    if args.baseline == "ftt" and args.out == "results/baselines.json":
        out_path = "results/baselines_ftt.json"
    elif args.baseline == "lr_all" and args.out == "results/baselines.json":
        out_path = "results/baselines_lr.json"
    elif args.baseline == "rf_all" and args.out == "results/baselines.json":
        out_path = "results/baselines_rf.json"
    elif args.baseline == "xgb_all" and args.out == "results/baselines.json":
        out_path = "results/baselines_xgb.json"
    else:
        out_path = args.out
    _save_json(out_path, out_data)
    print(f"[INFO] wrote cross-validation results to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts_config", default=None,
                    help="YAML mapping of expert_name -> CSV path (same as mref-ad train_moe).")
    ap.add_argument("--baseline", default="all",
                    choices=[
                        "all","single","concat","latefusion",
                        "mlp_single","mlp_concat","mlp_latefusion","mlp_all",
                        "rf_all","xgb_all","lr_all",
                        "ftt_all", "ftt",
                    ])
    ap.add_argument("--train_size", type=float, default=0.8)
    ap.add_argument("--splits", default=None, help="Optional path to fixed splits.json")
    ap.add_argument("--out", default="results/baselines.json")
    ap.add_argument("--save_sklearn_models", action="store_true", help="If set, save sklearn models (joblib) and a .meta.pkl per-fold under results/models/<baseline>/<key>/fold{N}/")
    ap.add_argument("--save_checkpoint", default=None, help="Optional path to save best model state_dict (torch).")
    ap.add_argument("--retrain_on_full", action="store_true", help="If set, retrain the best model on train+val for best_epoch before saving the checkpoint.")
    ap.add_argument("--skip_retrain", action="store_true", help="If set, do NOT retrain the tuned hyperparameter on train+val; use the early-stopped best model for test evaluation instead.")
    ap.add_argument("--eval_only", action="store_true", help="If set, do not train; load --ckpt and evaluate on the validation set for the specified --only_fold.")
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint file (torch .pt) to load when --eval_only is set.")
    ap.add_argument("--split_type", choices=["holdout", "cv5", "train_val_test"], default="cv5")
    ap.add_argument("--early_stop_metric", choices=["val_loss", "val_auc"], default="val_loss",
                    help="Metric to use for early stopping in MLP baselines (default: val_auc)")
    ap.add_argument("--select_metric", choices=["val_auc","val_acc","val_bacc","val_f1","val_loss"], default="val_auc",
                    help="Which validation metric to use to pick the best hyperparameter trial after tuning (default: val_auc)")
    ap.add_argument("--tune_trials", type=int, default=20, help="Number of Optuna trials to run for tuning (train_val_test mode)")
    ap.add_argument("--tune_timeout", type=int, default=None, help="Timeout (seconds) for Optuna tuning; overrides tune_trials if set")
    ap.add_argument(
        "--retrain_params",
        default=None,
        help=(
            "Optional JSON string or path to a JSON file containing hyperparameters to "
            "skip tuning and directly retrain the specified baseline on train+val and evaluate on test. "
            "Either supply a dict keyed by baseline name (e.g. {\"lr_all\": {\"C\":0.02}}) or a single dict "
            "of params when running a single baseline."
        ),
    )
    ap.add_argument(
        "--only_fold",
        type=int,
        default=None,
        help="If set (with --split_type cv5), run only this fold index (0-based) and write a single-fold JSON.",
    )
    args = ap.parse_args()

    set_seed()

    df, groups, classes = utils.load_experts_from_yaml(args.experts_config)
    mods_str = ", ".join([f"{m}={len(cols)}" for m, cols in groups.items()])
    print(f"[INFO] rows={len(df)} | feats: {mods_str}")

    # Strict dispatch based on requested split_type. When a splits JSON is
    # provided we require it to match the requested mode to avoid surprising
    # behavior. In particular, `train_val_test` enforces a train/val/test
    # manifest (no CV folds) and `cv5` enforces cv_splits_ptid with >=2 folds.
    if args.split_type == "holdout":
        return _run_holdout(args, df, groups, classes)

    if args.split_type == "train_val_test":
        # Require a splits JSON with explicit train/val/test
        if args.splits is None:
            raise ValueError("--split_type train_val_test requires --splits pointing to a splits JSON file")
        with open(args.splits, 'r') as sf:
            splits_data = json.load(sf)
        # Must have explicit train/val/test keys and must NOT include cv_splits_ptid
        if not all(k in splits_data for k in ("train_ptids", "val_ptids", "test_ptids")):
            raise ValueError("splits JSON must contain 'train_ptids', 'val_ptids' and 'test_ptids' for train_val_test mode")
        if "cv_splits_ptid" in splits_data and len(splits_data.get("cv_splits_ptid", [])) > 0:
            raise ValueError("splits JSON contains CV folds but split_type is 'train_val_test' (strict mode). Remove cv_splits_ptid or choose split_type=cv5.")
    return _run_train_val_test(args, df, groups, classes)

    if args.split_type == "cv5":
        # Require a splits JSON containing CV folds
        if args.splits is None:
            raise ValueError("--split_type cv5 requires --splits pointing to a splits JSON file containing 'cv_splits_ptid'")
        with open(args.splits, 'r') as sf:
            splits_data = json.load(sf)
        if "cv_splits_ptid" not in splits_data or len(splits_data.get("cv_splits_ptid", [])) < 2:
            raise ValueError("splits JSON must contain 'cv_splits_ptid' with >=2 folds for cv5 split_type")
        return _run_cv5(args, df, groups, classes)

    raise ValueError(f"Unsupported split_type: {args.split_type}")


if __name__ == "__main__":
    main()
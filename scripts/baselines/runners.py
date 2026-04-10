"""Orchestration / runner helpers for baselines.

This module contains the orchestration functions that were previously
embedded inside `scripts/train_baselines.py`. They are small adapters that
call models/trainers in `baselines.*` and shared helpers in `scripts.utils`
or `baselines.preprocessing`.

Keeping these here avoids circular imports with the CLI and makes unit
testing easier.
"""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch

import utils
from utils import eval_multiclass_metrics, macro_auroc

from baselines.sklearn_baselines import build_sklearn_runner
from baselines.mlp import fit_mlp, MLPConfig
from baselines.ftt import fit_ftt as baselines_fit_ftt, FTTConfig
from baselines.preprocessing import _build_xy, median_impute_and_scale


def train_eval_sklearn_baselines(
    baseline_name: str,
    df: pd.DataFrame,
    cols: List[str],
    tr_idx: List[int],
    va_idx: List[int],
    *,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict:
    """Run a sklearn baseline via baselines/sklearn_baselines.py.

    Falls back to local implementations if the baselines package import is unavailable.
    """
    if build_sklearn_runner is None:
        raise RuntimeError("baselines.sklearn_baselines is not available")

    # Use the same preprocessing as other baselines: robust impute + StandardScaler.
    Xtr, ytr, sc = _build_xy(df, cols, tr_idx, None)
    Xva, yva, _ = _build_xy(df, cols, va_idx, sc)

    runner = build_sklearn_runner(baseline_name, seed=seed, n_jobs=n_jobs)
    fit = runner.fit(Xtr, ytr)
    proba = runner.predict_proba(fit, Xva)

    m = eval_multiclass_metrics(yva, proba)
    out = {
        "val_auc": m["auc"],
        "val_acc": m["acc"],
        "val_bacc": m["bacc"],
        "val_f1": m["f1"],
        "proba": proba,
        "yva": yva,
        # expose the fitted estimator and scaler so callers can persist them
        "estimator": fit,
        "scaler": sc,
    }
    return out


def train_eval_mlp(df, cols, tr_idx, va_idx,
                   hidden=128, drop=0.2, epochs=50, lr=1e-3, wd=1e-4, batch=128,
                   device=None, patience=5, early_stop_metric="val_auc", save_checkpoint: str = None, retrain_on_full: bool = False,
                   seed: int = None, **kwargs):
    # Delegate to baselines.mlp.fit_mlp which implements training with early stopping
    Xtr, ytr, sc = _build_xy(df, cols, tr_idx, None)
    Xva, yva, _ = _build_xy(df, cols, va_idx, sc)

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    cfg = MLPConfig(
        n_classes=3,
        hidden=hidden,
        drop=drop,
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        weight_decay=wd,
        patience=patience,
        early_stop_metric=early_stop_metric,
    )

    out = fit_mlp(
        Xtr, ytr, Xva, yva,
        config=cfg,
        metric_fn=eval_multiclass_metrics,
        device=device,
        verbose=True,
    )

    # Optionally save checkpoint: either the best_state_dict (selected by val)
    # or a model retrained on the combined train+val for best_epoch epochs.
    if save_checkpoint is not None:
        try:
            # Ensure directory
            import os
            os.makedirs(os.path.dirname(save_checkpoint) or '.', exist_ok=True)
            if retrain_on_full and out.get("best_epoch") is not None:
                # retrain on full data for best_epoch epochs
                X_all = np.vstack([Xtr, Xva])
                y_all = np.concatenate([ytr, yva])
                state = retrain_mlp_on_full(X_all, y_all, config=cfg, device=device, epochs=int(out["best_epoch"]))
            else:
                state = out.get("best_state_dict")
            if state is not None:
                torch.save(state, save_checkpoint)
                # Confirm that the checkpoint file was written
                try:
                    if os.path.isfile(save_checkpoint):
                        print(f"[INFO] saved checkpoint to {save_checkpoint}")
                    else:
                        print(f"[WARN] attempted to save checkpoint but file not found at {save_checkpoint}")
                except Exception:
                    print(f"[WARN] could not verify checkpoint file at {save_checkpoint}")
                # Confirm that the checkpoint file was written
                try:
                    if os.path.isfile(save_checkpoint):
                        print(f"[INFO] saved checkpoint to {save_checkpoint}")
                    else:
                        print(f"[WARN] attempted to save checkpoint but file not found at {save_checkpoint}")
                except Exception:
                    print(f"[WARN] could not verify checkpoint file at {save_checkpoint}")
                # also save preprocessing artifacts if present in `out`
                try:
                    import pickle
                    meta = {}
                    # Prefer explicit objects returned by the trainer, but fall
                    # back to the local variables available here (scaler/cols)
                    # to ensure we always persist the preprocessing contract.
                    if out.get("scaler", None) is not None:
                        meta["scaler"] = out.get("scaler")
                    else:
                        # 'sc' is the scaler fit on TRAIN earlier in this function
                        try:
                            meta["scaler"] = sc
                        except Exception:
                            pass
                    if out.get("train_means", None) is not None:
                        meta["train_means"] = out.get("train_means")
                    if out.get("cols", None) is not None:
                        meta["cols"] = out.get("cols")
                    else:
                        try:
                            meta["cols"] = cols
                        except Exception:
                            pass
                    if out.get("model_config", None) is not None:
                        meta["model_config"] = out.get("model_config")
                    if meta:
                        meta_path = os.path.splitext(save_checkpoint)[0] + ".meta.pkl"
                        with open(meta_path, "wb") as mf:
                            pickle.dump(meta, mf)
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] could not save checkpoint to {save_checkpoint}: {e}")

    # fit_mlp returns keys similar to previous contract: val_loss/val_auc/... proba/yva
    res = {k: out.get(k) for k in ("val_loss", "val_auc", "val_acc", "val_bacc", "val_f1", "proba", "yva")}
    # pass through optional artifacts for downstream usage and debugging
    if out.get("history") is not None:
        res["history"] = out.get("history")
    if out.get("scaler") is not None:
        res["scaler"] = out.get("scaler")
    if out.get("cols") is not None:
        res["cols"] = out.get("cols")
    if out.get("model_config") is not None:
        res["model_config"] = out.get("model_config")
    # include best_state_dict for checkpointing callers
    if out.get("best_state_dict") is not None:
        res["best_state_dict"] = out.get("best_state_dict")
    return res


def run_single_modality_MLP(df, groups, mod, tr_idx, va_idx, **kw):
    has_tr = df.iloc[tr_idx][f"has_{mod}"].to_numpy().astype(bool) if f"has_{mod}" in df.columns else np.ones(len(tr_idx), bool)
    has_va = df.iloc[va_idx][f"has_{mod}"].to_numpy().astype(bool) if f"has_{mod}" in df.columns else np.ones(len(va_idx), bool)

    tr_sub_idx = np.array(tr_idx)[has_tr]
    va_sub_idx = np.array(va_idx)[has_va]
    if len(tr_sub_idx) < 2 or len(va_sub_idx) < 1:
        return {"val_auc": float("nan"), "proba": None, "yva": df.iloc[va_idx]["y"].values}

    r = train_eval_mlp(df, groups[mod], tr_sub_idx, va_sub_idx, **kw)

    # expand back to full val set (zeros where modality missing)
    P_full = np.zeros((len(va_idx), 3), dtype=float)
    P_full[has_va] = r["proba"]
    yva_full = df.iloc[va_idx]["y"].values

    # Compute all multiclass metrics at once
    m = eval_multiclass_metrics(yva_full, P_full)

    # Merge metrics into one dictionary
    out = {
        "modality": mod,
        **{f"val_{k}": v for k, v in m.items()},  # prefix all metrics with val_
        "proba": P_full,
        "yva": yva_full
    }
    return out


def run_concat_MLP(df, groups, tr_idx, va_idx, **kw):
    cols = [c for _, feat in groups.items() for c in feat]
    return train_eval_mlp(df, cols, tr_idx, va_idx, **kw)


def run_latefusion_MLP(df, groups, tr_idx, va_idx, early_stop_metric=None):
    mods = list(groups.keys())
    if not mods:
        return {"val_auc": float("nan"), "proba": None, "yva": None}

    # Run each modality’s single-modality MLP (not LR)
    runs = {m: run_single_modality_MLP(df, groups, m, tr_idx, va_idx, early_stop_metric=early_stop_metric) for m in mods}

    # Pick first valid yva from the runs
    yva = next((r["yva"] for r in runs.values() if r.get("yva") is not None), None)
    if yva is None:
        return {"val_auc": float("nan"), "proba": None, "yva": None}

    n_val, K = len(va_idx), 3
    proba_sum = np.zeros((n_val, K), dtype=float)
    cnt = np.zeros((n_val, 1), dtype=float)

    # Average probabilities across modalities that exist for each subject
    for m, r in runs.items():
        if r.get("proba") is None:
            continue
        mask_col = f"has_{m}"
        mask = df.iloc[va_idx][mask_col].to_numpy().astype(bool) if mask_col in df.columns else np.ones(n_val, dtype=bool)
        proba_sum[mask] += r["proba"][mask]
        cnt[mask] += 1.0

    cnt = np.clip(cnt, 1.0, None)
    P = proba_sum / cnt

    # Compute all metrics from final fused probabilities
    m = eval_multiclass_metrics(yva, P)

    # Return metrics and outputs
    out = {
        **{f"val_{k}": v for k, v in m.items()},
        "proba": P,
        "yva": yva
    }
    return out


def run_single_modality_LR(df, groups, mod, tr_idx, va_idx):
    cols = groups[mod]
    Xtr = df.iloc[tr_idx][cols].astype(float).to_numpy()
    Xva = df.iloc[va_idx][cols].astype(float).to_numpy()
    ytr = df.iloc[tr_idx]["y"].values
    yva = df.iloc[va_idx]["y"].values
    has_tr = df.iloc[tr_idx][f"has_{mod}"].values.astype(bool)
    has_va = df.iloc[va_idx][f"has_{mod}"].values.astype(bool)

    Xtr, Xva = median_impute_and_scale(Xtr, Xva)

    # train only on samples that actually have this modality
    Xtr2, ytr2 = Xtr[has_tr], ytr[has_tr]
    if len(np.unique(ytr2)) < 2:
        return {"mod": mod, "val_auc": float("nan"), "proba": None}

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(Xtr2, ytr2)

    proba = np.zeros((len(yva), 3), dtype=float)
    if has_va.any():
        proba[has_va] = clf.predict_proba(Xva[has_va])
    auc = macro_auroc(yva, proba, 3)
    return {"mod": mod, "val_auc": auc, "proba": proba, "yva": yva}


def run_concat_LR(df, groups, tr_idx, va_idx):
    cols = [c for _, feat in groups.items() for c in feat]
    Xtr = df.iloc[tr_idx][cols].astype(float).to_numpy()
    Xva = df.iloc[va_idx][cols].astype(float).to_numpy()
    ytr = df.iloc[tr_idx]["y"].values
    yva = df.iloc[va_idx]["y"].values

    Xtr, Xva = median_impute_and_scale(Xtr, Xva)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300, solver="lbfgs")
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xva)
    m = eval_multiclass_metrics(yva, proba)
    out = {
        "val_auc": m["auc"],
        "val_acc": m["acc"],
        "val_bacc": m["bacc"],
        "val_f1": m["f1"],
        "proba": proba,
        "yva": yva
    }
    return out


def run_latefusion(df, groups, tr_idx, va_idx):
    results = []
    mods = list(groups.keys())
    for m in mods:
        if m in groups:
            results.append(run_single_modality_MLP(df, groups, m, tr_idx, va_idx))

    # pick yva from the first result that has it
    yva = next((r["yva"] for r in results if r.get("yva") is not None), None)
    if yva is None:
        return {"val_auc": float("nan"), "proba": None, "yva": None}

    # average only across available probabilities
    P_list = [r["proba"] for r in results if r.get("proba") is not None]
    if not P_list:
        return {"val_auc": float("nan"), "proba": None, "yva": yva}

    proba = np.mean(np.stack(P_list, axis=0), axis=0)
    m = eval_multiclass_metrics(yva, proba)
    return {
        "val_auc": m["auc"],
        "val_acc": m["acc"],
        "val_bacc": m["bacc"],
        "val_f1": m["f1"],
        "proba": proba,
        "yva": yva,
    }


def train_eval_ftt(
    df: pd.DataFrame,
    cols: List[str],
    tr_idx: List[int],
    va_idx: List[int],
    d_token: int = 192,
    n_layers: int = 3,
    n_heads: int = 8,
    d_ffn_factor: float = 4.0,
    attention_dropout: float = 0.2,
    ffn_dropout: float = 0.0,
    residual_dropout: float = 0.0,
    activation: str = "geglu",
    prenormalization: bool = True,
    initialization: str = "xavier",
    token_bias: bool = True,
    epochs: int = 80,
    lr: float = 3e-4,
    wd: float = 1e-3,
    batch: int = 128,
    device=None,
    patience: int = 10,
    early_stop_metric: str = "val_auc",
    save_checkpoint: str = None,
    retrain_on_full: bool = False,
):
    # Delegate to baselines.ftt.fit_ftt which implements the official FT-Transformer training
    cfg = FTTConfig(
        n_classes=3,
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        weight_decay=wd,
        patience=patience,
        early_stop_metric=early_stop_metric,
    )

    out = baselines_fit_ftt(
        df,
        cols,
        tr_idx,
        va_idx,
        config=cfg,
        metric_fn=eval_multiclass_metrics,
        device=device,
        verbose=True,
    )

    # Optionally save checkpoint for FT-Transformer as for MLPs
    if save_checkpoint is not None:
        try:
            import torch, os
            os.makedirs(os.path.dirname(save_checkpoint) or '.', exist_ok=True)
            if retrain_on_full and out.get("best_epoch") is not None:
                # build combined train+val and retrain
                Xtr, ytr, Xva, yva, scaler, mu = None, None, None, None, None, None
                # Use the same preproc as fit_ftt: caller returns scaler and train_means
                Xtr, ytr, Xva, yva, scaler, mu = None, None, None, None, None, None
                # We'll reconstruct Xtr/Xva using the preprocessing helpers from baselines.ftt
                from baselines.ftt import build_xy_mean_from_train, retrain_ftt_on_full
                # build train and val arrays
                Xtr, ytr, Xva, yva, scaler, mu = build_xy_mean_from_train(df, cols, tr_idx, va_idx)
                X_all = np.vstack([Xtr, Xva])
                y_all = np.concatenate([ytr, yva])
                state = retrain_ftt_on_full(X_all, y_all, config=cfg, device=device, epochs=int(out.get("best_epoch", 0)))
            else:
                state = out.get("best_state_dict")
            if state is not None:
                torch.save(state, save_checkpoint)
            # Save preprocessing artifacts (scaler/train_means/cols) when available
            try:
                import pickle
                meta = {}
                if out.get("scaler", None) is not None:
                    meta["scaler"] = out.get("scaler")
                if out.get("train_means", None) is not None:
                    meta["train_means"] = out.get("train_means")
                if out.get("cols", None) is not None:
                    meta["cols"] = out.get("cols")
                if out.get("model_config", None) is not None:
                    meta["model_config"] = out.get("model_config")
                if meta:
                    meta_path = os.path.splitext(save_checkpoint)[0] + ".meta.pkl"
                    with open(meta_path, "wb") as mf:
                        pickle.dump(meta, mf)
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] could not save FTT checkpoint to {save_checkpoint}: {e}")

    res = {
        "val_loss": out.get("val_loss", float("nan")),
        "val_auc": out.get("val_auc", float("nan")),
        "val_acc": out.get("val_acc", float("nan")),
        "val_bacc": out.get("val_bacc", float("nan")),
        "val_f1": out.get("val_f1", float("nan")),
        "proba": out.get("proba"),
        "yva": out.get("yva"),
    }
    if out.get("history") is not None:
        res["history"] = out.get("history")
    if out.get("scaler") is not None:
        res["scaler"] = out.get("scaler")
    if out.get("cols") is not None:
        res["cols"] = out.get("cols")
    if out.get("model_config") is not None:
        res["model_config"] = out.get("model_config")
    if out.get("best_state_dict") is not None:
        res["best_state_dict"] = out.get("best_state_dict")
    return res

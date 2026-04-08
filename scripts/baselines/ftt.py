# baselines/ftt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rtdl_revisiting_models import FTTransformer

from sklearn.preprocessing import StandardScaler


# -----------------------------
# Config
# -----------------------------
@dataclass
class FTTConfig:
    n_classes: int = 3

    # Training hyperparameters (tunable)
    epochs: int = 80
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    patience: int = 10
    early_stop_metric: str = "val_loss"  # "val_loss" or "val_auc"
    improvement_tol: float = 1e-4

    seed: int = 42
    num_workers: int = 0

    seed: int = 42
    num_workers: int = 0


# -----------------------------
# Device
# -----------------------------
from .utils import get_default_device


def _class_weights(y: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=n_classes) + 1
    w = counts.sum() / counts
    return torch.tensor(w, dtype=torch.float32, device=device)


def _call_model(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """Call FTTransformer with x_num (continuous) and x_cat=None (no categorical features)."""
    return model(xb, None)




# -----------------------------
# Data building: official-like mean impute from TRAIN + standardize
# -----------------------------
def build_xy_mean_from_train(
    df: pd.DataFrame,
    cols: List[str],
    tr_idx: List[int],
    va_idx: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.Series]:
    """
    Official-like policy: compute means on TRAIN, apply to train/val.
    Returns:
      Xtr, ytr, Xva, yva, scaler, train_means
    """
    Xtr_df = df.iloc[tr_idx][cols].astype(float)
    Xva_df = df.iloc[va_idx][cols].astype(float)

    mu = Xtr_df.mean(numeric_only=True).fillna(0.0)
    Xtr_df = Xtr_df.fillna(mu).fillna(0.0)
    Xva_df = Xva_df.fillna(mu).fillna(0.0)

    scaler = StandardScaler().fit(Xtr_df)
    Xtr = scaler.transform(Xtr_df).astype(np.float32)
    Xva = scaler.transform(Xva_df).astype(np.float32)

    ytr = df.iloc[tr_idx]["y"].astype(int).values
    yva = df.iloc[va_idx]["y"].astype(int).values
    return Xtr, ytr, Xva, yva, scaler, mu


# -----------------------------
# Training
# -----------------------------


def fit_ftt(
    df: pd.DataFrame,
    cols: List[str],
    tr_idx: List[int],
    va_idx: List[int],
    *,
    config: FTTConfig,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train official FTT on df[cols] for indices.
    Returns:
      - best val metrics
      - best val proba + yva
      - best_state_dict
      - scaler + train_means (so you can reproduce the exact preprocessing later)
    """
    if device is None:
        device = get_default_device()
    if config.early_stop_metric not in ("val_loss", "val_auc"):
        raise ValueError("early_stop_metric must be 'val_loss' or 'val_auc'")

    Xtr, ytr, Xva, yva, scaler, mu = build_xy_mean_from_train(df, cols, tr_idx, va_idx)

    # Official package FT-Transformer with official default hyperparameters
    model = FTTransformer(
        n_cont_features=Xtr.shape[1],
        cat_cardinalities=None,  # numeric-only
        d_out=config.n_classes,
        **FTTransformer.get_default_kwargs(),
    ).to(device)

    weights = _class_weights(ytr, config.n_classes, device)
    crit = nn.CrossEntropyLoss(weight=weights)

    # Use the official optimizer configuration from the package.
    # This matches the reference implementation more closely than custom WD grouping.
    opt = model.make_default_optimizer()

    g = torch.Generator()
    g.manual_seed(config.seed)

    ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    ds_va = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))

    dl_tr = DataLoader(ds_tr, batch_size=config.batch_size, shuffle=True, generator=g, num_workers=config.num_workers)
    dl_va = DataLoader(ds_va, batch_size=config.batch_size, shuffle=False, generator=g, num_workers=config.num_workers)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_proba: Optional[np.ndarray] = None

    best = {
        "val_loss": float("inf"),
        "val_auc": -np.inf,
        "val_acc": float("nan"),
        "val_bacc": float("nan"),
        "val_f1": float("nan"),
    }
    bad = 0

    # history collector
    history = []

    for ep in range(1, config.epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = _call_model(model, xb)
            loss = crit(logits, yb)
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()

        model.eval()
        val_loss = 0.0
        P_list, Y_list = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = _call_model(model, xb)
                loss = crit(logits, yb)
                val_loss += loss.item() * xb.size(0)
                P_list.append(torch.softmax(logits, dim=1).cpu().numpy())
                Y_list.append(yb.cpu().numpy())

        val_loss /= len(dl_va.dataset)
        P = np.vstack(P_list)
        Y = np.concatenate(Y_list)

        m: Dict[str, float] = {}
        if metric_fn is not None:
            m = metric_fn(Y, P)

        if verbose:
            auc = float(m.get("auc", float("nan")))
            acc = float(m.get("acc", float("nan")))
            bacc = float(m.get("bacc", float("nan")))
            f1 = float(m.get("f1", float("nan")))
            print(
                f"Epoch {ep:03d} | val_loss={val_loss:.4f} "
                f"| AUC={auc:.4f} | Acc={acc:.4f} | BAcc={bacc:.4f} | F1={f1:.4f}"
            )

        # compute train metrics (inference on train set)
        train_loss = 0.0
        P_tr_list, Y_tr_list = [], []
        with torch.no_grad():
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                logits = _call_model(model, xb)
                loss = crit(logits, yb)
                train_loss += loss.item() * xb.size(0)
                P_tr_list.append(torch.softmax(logits, dim=1).cpu().numpy())
                Y_tr_list.append(yb.cpu().numpy())

        train_loss /= len(dl_tr.dataset)
        P_tr = np.vstack(P_tr_list) if P_tr_list else np.zeros((len(ytr), config.n_classes), dtype=float)
        Y_tr = np.concatenate(Y_tr_list) if Y_tr_list else np.array([])
        train_metrics: Dict[str, float] = {}
        if metric_fn is not None and Y_tr.size > 0:
            train_metrics = metric_fn(Y_tr, P_tr)

        history.append({
            "epoch": int(ep),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_metrics": {k: float(v) for k, v in (train_metrics or {}).items()},
            "val_metrics": {k: float(v) for k, v in (m or {}).items()},
        })

        if config.early_stop_metric == "val_loss":
            improved = val_loss < best["val_loss"] - config.improvement_tol
        else:
            cur_auc = float(m.get("auc", -np.inf))
            improved = cur_auc > best["val_auc"] + config.improvement_tol

        if improved:
            best["val_loss"] = float(val_loss)
            if "auc" in m:
                best["val_auc"] = float(m["auc"])
            if "acc" in m:
                best["val_acc"] = float(m["acc"])
            if "bacc" in m:
                best["val_bacc"] = float(m["bacc"])
            if "f1" in m:
                best["val_f1"] = float(m["f1"])

            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            best_proba = P.copy()
            bad = 0
        else:
            bad += 1
            if bad >= config.patience:
                if verbose:
                    if config.early_stop_metric == "val_loss":
                        print(f"Early stopping at epoch {ep} | best val_loss={best['val_loss']:.4f}")
                    else:
                        print(f"Early stopping at epoch {ep} | best val_auc={best['val_auc']:.4f}")
                break

    # fallback
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_proba is None:
        # compute once from final model
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(Xva).to(device)
            logits = _call_model(model, xb)
            best_proba = torch.softmax(logits, dim=1).cpu().numpy()

    return {
        **best,
        "proba": best_proba,
        "yva": yva,
        "best_state_dict": best_state,
        "best_epoch": locals().get("best_epoch", None),
        # preprocessing artifacts (important for reproducible missing-modality inference)
        "scaler": scaler,
        "train_means": mu,
        "cols": list(cols),
        "model_config": config,
        "history": history,
    }


def retrain_ftt_on_full(
    X_all: np.ndarray,
    y_all: np.ndarray,
    *,
    config: FTTConfig,
    device: Optional[torch.device] = None,
    epochs: int = 0,
    Xte: Optional[np.ndarray] = None,
    yte: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """Retrain an FTTransformer on combined data for `epochs` epochs and return state_dict.

    If Xte and yte provided, print test metrics at each epoch.
    Note: this uses the same model constructor as fit_ftt and trains without validation.
    """
    if device is None:
        device = get_default_device()

    # Use official FT-Transformer defaults for retraining (architecture unchanged)
    model = FTTransformer(
        n_cont_features=X_all.shape[1],
        cat_cardinalities=None,
        d_out=config.n_classes,
        **FTTransformer.get_default_kwargs(),
    ).to(device)

    weights = _class_weights(y_all, config.n_classes, device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = model.make_default_optimizer()

    ds = TensorDataset(torch.from_numpy(X_all.astype(np.float32)), torch.from_numpy(y_all.astype(np.int64)))
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    for ep in range(1, max(1, epochs) + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = _call_model(model, xb)
            loss = crit(logits, yb)
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()

        # Optionally evaluate on test set at each epoch
        if Xte is not None and yte is not None:
            model.eval()
            with torch.no_grad():
                xte_t = torch.from_numpy(Xte.astype(np.float32)).to(device)
                logits_te = _call_model(model, xte_t)
                proba_te = torch.softmax(logits_te, dim=1).cpu().numpy()
            from utils import eval_multiclass_metrics
            m_te = eval_multiclass_metrics(yte, proba_te)
            if m_te:
                print(f"[RETRAIN] Epoch {ep:03d} | test_auc={m_te.get('auc', float('nan')):.4f} | test_acc={m_te.get('acc', float('nan')):.4f} | test_f1={m_te.get('f1', float('nan')):.4f}")

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# -----------------------------
# Inference helpers
# -----------------------------
def transform_with_train_stats(
    df: pd.DataFrame,
    cols: List[str],
    idx: List[int],
    *,
    scaler: StandardScaler,
    train_means: pd.Series,
) -> np.ndarray:
    Xdf = df.iloc[idx][cols].astype(float)
    Xdf = Xdf.fillna(train_means).fillna(0.0)
    X = scaler.transform(Xdf).astype(np.float32)
    return X


@torch.no_grad()
def predict_proba_ftt(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X_t = torch.from_numpy(X.astype(np.float32, copy=False))
    dl = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False, num_workers=0)

    probs = []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = _call_model(model, xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def load_ftt_from_state(
    d_in: int,
    state_dict: Dict[str, torch.Tensor],
    *,
    config: FTTConfig,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if device is None:
        device = get_default_device()

    model = FTTransformer(
        n_cont_features=d_in,
        cat_cardinalities=None,
        d_out=config.n_classes,
        **FTTransformer.get_default_kwargs(),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_val_test(
    df,
    cols,
    tr_idx,
    va_idx,
    test_idx,
    *,
    n_trials: int = 20,
    timeout: float = None,
    out_dir: str = None,
    seed: int = 42,
    early_stop_metric: str = "val_auc",
    select_metric: str = "val_auc",
    retrain_on_full: bool = True,
):
    """Perform Optuna tuning for FT-Transformer, retrain on train+val and evaluate on test.

    Returns a dict with best_params, best_val_auc, test_metrics, model_path, meta_path, n_trials.
    """
    import optuna

    Xtr, ytr, Xva, yva, scaler, mu = build_xy_mean_from_train(df, cols, tr_idx, va_idx)
    Xte = transform_with_train_stats(df, cols, test_idx, scaler=scaler, train_means=mu)
    yte = df.iloc[test_idx]["y"].astype(int).values

    # Store trial outputs so we can retrieve them later without re-running
    trial_outputs = {}
    
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        cfg = FTTConfig(
            epochs=50,
            batch_size=128,
            lr=lr,
            weight_decay=weight_decay,
            patience=10,
            early_stop_metric=early_stop_metric
        )
        from utils import eval_multiclass_metrics
        out = fit_ftt(df, cols, tr_idx, va_idx, config=cfg, metric_fn=eval_multiclass_metrics)
        # Store the full output for later retrieval
        trial_outputs[trial.number] = out
        return float(out.get(select_metric, float("nan")))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_trial = study.best_trial
    
    # Get best_epoch and validation metrics from the stored trial output (no re-run needed)
    best_epoch = None
    val_metrics_best = {k: float("nan") for k in ("val_loss", "val_auc", "val_acc", "val_bacc", "val_f1")}
    out_best = None
    
    if best_trial.number in trial_outputs:
        out_best = trial_outputs[best_trial.number]
        best_epoch = out_best.get("best_epoch", None)
        val_metrics_best = {
            "val_loss": float(out_best.get("val_loss", float("nan"))),
            "val_auc": float(out_best.get("val_auc", float("nan"))),
            "val_acc": float(out_best.get("val_acc", float("nan"))),
            "val_bacc": float(out_best.get("val_bacc", float("nan"))),
            "val_f1": float(out_best.get("val_f1", float("nan"))),
        }

    # Create config for retraining (used regardless of retrain_on_full, for epoch tracking)
    X_all = np.vstack([Xtr, Xva])
    y_all = np.concatenate([ytr, yva])
    cfg = FTTConfig(
        epochs=int(best_params.get("epochs", 50)),
        batch_size=128,
        lr=best_params.get("lr", 3e-4),
        weight_decay=best_params.get("weight_decay", 1e-3),
        patience=10,
        early_stop_metric=early_stop_metric
    )
    
    # Retrain (or not) according to retrain_on_full
    if retrain_on_full:
        # Use best_epoch from the trial for retraining
        retrain_epochs = int(best_epoch) if best_epoch is not None else cfg.epochs
        print(f"[INFO] Retraining best trial on TRAIN+VAL for {retrain_epochs} epochs (best_epoch={best_epoch}) params={best_params}")
        print(f"[INFO] Training on combined TRAIN+VAL data (n={len(X_all)}) without validation")
        state = retrain_ftt_on_full(X_all, y_all, config=cfg, device=None, epochs=retrain_epochs, Xte=Xte, yte=yte)
    else:
        # skip retrain: use the TRAIN-only snapshot from the best trial (already available from out_best)
        print(f"[INFO] retrain_on_full=False: using TRAIN-only snapshot from Optuna trial")
        if out_best is not None:
            state = out_best.get("best_state_dict")
            print(f"[INFO] Best trial snapshot has best_epoch={best_epoch}, val_auc={val_metrics_best.get('val_auc', float('nan')):.4f}")
        else:
            state = None

    model_path = None
    meta_path = None
    if out_dir is not None:
        import os, pickle
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "model.ftt.state.pkl")
        with open(model_path, "wb") as mf:
            pickle.dump(state, mf)
        meta = {"cols": cols, "scaler": scaler, "train_means": mu}
        meta_path = os.path.join(out_dir, "model.meta.pkl")
        with open(meta_path, "wb") as mf:
            pickle.dump(meta, mf)

    # load model from state and compute proba on test
    print(f"[INFO] Evaluating on TEST set (n={len(Xte)})")
    if state is None:
        raise RuntimeError("state is None, cannot load model")
    model = load_ftt_from_state(Xte.shape[1], state, config=cfg)
    proba = predict_proba_ftt(model, Xte)

    from utils import eval_multiclass_metrics
    m_test = eval_multiclass_metrics(yte, proba) if proba is not None else {}
    
    # Print test metrics with clear labels
    if m_test:
        print(f"[INFO] TEST metrics: test_auc={m_test.get('auc', float('nan')):.4f} | test_acc={m_test.get('acc', float('nan')):.4f} | test_f1={m_test.get('f1', float('nan')):.4f}")
    # --- Second-best trial analysis: find the second-best set of params (if any)
    second = None
    finished = [t for t in study.trials if t.value is not None]
    finished_sorted = sorted(finished, key=lambda t: float(t.value), reverse=True)
    if len(finished_sorted) > 1:
        second = finished_sorted[1]

    second_params = None
    second_val_metrics = {k: float('nan') for k in ("val_loss", "val_auc", "val_acc", "val_bacc", "val_f1")}
    second_test_metrics = {k: float('nan') for k in ("auc", "acc", "f1")}
    second_val_metric_val = float('nan')
    out_sec = None
    if second is not None:
        second_params = dict(second.params)
        # Try to get the stored output from the Optuna trials
        if second.number in trial_outputs:
            out_sec = trial_outputs[second.number]
        else:
            # Fallback: this shouldn't happen if Optuna was run properly
            out_sec = None
        
        if out_sec is not None:
            second_val_metrics = {
                "val_loss": float(out_sec.get("val_loss", float("nan"))),
                "val_auc": float(out_sec.get("val_auc", float("nan"))),
                "val_acc": float(out_sec.get("val_acc", float("nan"))),
                "val_bacc": float(out_sec.get("val_bacc", float("nan"))),
                "val_f1": float(out_sec.get("val_f1", float("nan"))),
            }
            second_val_metric_val = float(second.value) if getattr(second, 'value', None) is not None else float('nan')
        else:
            second_val_metric_val = float(second.value) if getattr(second, 'value', None) is not None else float('nan')

        # Decide whether to retrain on TRAIN+VAL or use the TRAIN-only snapshot
        second_best_epoch = out_sec.get('best_epoch', None) if out_sec is not None else None
        state2 = None
        if retrain_on_full:
            # Build a config for second-best trial (use sensible defaults when keys missing)
            cfg_second = FTTConfig(
                epochs=int(second_params.get('epochs', 50)),
                batch_size=int(second_params.get('batch_size', 128)),
                lr=second_params.get('lr', 3e-4),
                weight_decay=second_params.get('weight_decay', 1e-3),
                patience=10,
                early_stop_metric=early_stop_metric,
            )
            retrain_epochs_sec = int(second_best_epoch) if second_best_epoch is not None else cfg_second.epochs
            print(f"[INFO] Retraining second-best trial on TRAIN+VAL for {retrain_epochs_sec} epochs (best_epoch={second_best_epoch}) params={second_params}")
            print(f"[INFO] Training on combined TRAIN+VAL data (n={len(X_all)}) without validation")
            state2 = retrain_ftt_on_full(X_all, y_all, config=cfg_second, device=None, epochs=retrain_epochs_sec, Xte=Xte, yte=yte)
        else:
            # skip retrain: use the TRAIN-only snapshot from the stored Optuna trial output
            if out_sec is not None:
                state2 = out_sec.get('best_state_dict')
                print(f"[INFO] Using TRAIN-only snapshot from Optuna trial for second-best (best_epoch={second_best_epoch})")
            else:

                state2 = None

        # Evaluate second-best on TEST if we have a state
        if state2 is not None:
            print(f"[INFO] Evaluating second-best trial on TEST set")
            model2 = load_ftt_from_state(Xte.shape[1], state2, config=FTTConfig())
            proba_test2 = predict_proba_ftt(model2, Xte)
            from utils import eval_multiclass_metrics
            m_test2 = eval_multiclass_metrics(yte, proba_test2) if proba_test2 is not None else {}
            second_test_metrics = {
                "auc": float(m_test2.get('auc', float('nan'))),
                "acc": float(m_test2.get('acc', float('nan'))),
                "f1": float(m_test2.get('f1', float('nan'))),
            }
            print(f"[INFO] Second-best TEST metrics: test_auc={second_test_metrics['auc']:.4f} | test_acc={second_test_metrics['acc']:.4f} | test_f1={second_test_metrics['f1']:.4f}")
    # Include epoch budgets in reported hyperparameters so outputs explicitly
    # document how long training and retraining ran.
    best_params_out = dict(best_params or {})

    # cfg_best may be defined from the re-run on TRAIN/VAL; fall back to cfg
    train_epochs = None
    if "cfg_best" in locals() and getattr(cfg_best, "epochs", None) is not None:
        train_epochs = int(cfg_best.epochs)
    elif getattr(cfg, "epochs", None) is not None:
        train_epochs = int(cfg.epochs)
    else:
        train_epochs = int(FTTConfig().epochs)

    # retrain_epochs may have been set earlier; if not, default to train_epochs
    retrain_epochs_val = int(locals().get("retrain_epochs", train_epochs))

    # capture epoch information for best and second-best (if available)
    best_epoch_val = None
    if 'out_best' in locals() and out_best is not None:
        best_epoch_val = int(out_best.get('best_epoch')) if out_best.get('best_epoch', None) is not None else None

    second_best_epoch_val = None
    if 'out_sec' in locals() and out_sec is not None:
        second_best_epoch_val = int(out_sec.get('best_epoch')) if out_sec.get('best_epoch', None) is not None else None

    # Set best params 'epochs' to the best epoch observed during the
    # re-run on TRAIN/VAL when available; otherwise fall back to the
    # configured train_epochs.
    best_params_out["epochs"] = int(best_epoch_val) if best_epoch_val is not None else int(train_epochs)

    # Ensure second_best_params contains an 'epochs' entry (set to the
    # second-best best_epoch when available).
    if second_params is not None and isinstance(second_params, dict):
        if second_best_epoch_val is not None:
            second_params["epochs"] = int(second_best_epoch_val)
        else:
            second_params.setdefault("epochs", int(second_params.get("epochs", 50)))

    return {
        "best_params": best_params_out,
        # populate best_val_auc from the re-run validation snapshot so AUC is
        # always available regardless of the selection metric.
        "best_val_auc": float(val_metrics_best.get("val_auc", float("nan"))),
        "best_val_metric_name": select_metric,
        "best_val_metric": float(study.best_value),
        # expose full validation metrics for best hyperparameters
        "best_val_loss": float(val_metrics_best.get("val_loss", float("nan"))),
        "best_val_acc": float(val_metrics_best.get("val_acc", float("nan"))),
        "best_val_bacc": float(val_metrics_best.get("val_bacc", float("nan"))),
        "best_val_f1": float(val_metrics_best.get("val_f1", float("nan"))),
        "test_metrics": {"auc": float(m_test.get("auc", float("nan"))), "acc": float(m_test.get("acc", float("nan"))), "f1": float(m_test.get("f1", float("nan")))},
        # second-best trial information (may be None / NaN when unavailable)
        "second_best_params": second_params,
        "second_best_val_auc": float(second_val_metrics.get("val_auc", float("nan"))),
        "second_best_val_metric_name": select_metric,
        "second_best_val_metric": float(second_val_metric_val),
        "second_best_val_loss": float(second_val_metrics.get("val_loss", float("nan"))),
        "second_best_val_acc": float(second_val_metrics.get("val_acc", float("nan"))),
        "second_best_val_bacc": float(second_val_metrics.get("val_bacc", float("nan"))),
        "second_best_val_f1": float(second_val_metrics.get("val_f1", float("nan"))),
        "second_best_test_metrics": {"auc": float(second_test_metrics.get("auc", float("nan"))), "acc": float(second_test_metrics.get("acc", float("nan"))), "f1": float(second_test_metrics.get("f1", float("nan")))},
        "model_path": model_path,
        "meta_path": meta_path,
        # epoch of best validation during the re-run on TRAIN/VAL (may be None)
        "best_epoch": best_epoch_val,
        # epoch of best validation for the second-best re-run (may be None)
        "second_best_epoch": second_best_epoch_val,
        "n_trials": len(study.trials),
    }
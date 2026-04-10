# baselines/mlp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Config
# -----------------------------
@dataclass
class MLPConfig:
    n_classes: int = 3
    hidden: int = 256
    drop: float = 0.1

    epochs: int = 80
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    patience: int = 10
    early_stop_metric: str = "val_loss"  # "val_loss" or "val_auc"
    improvement_tol: float = 1e-4

    seed: int = 42
    num_workers: int = 0


def mlp_config_for_retrain(
    best_params: Dict[str, Any],
    *,
    retrain_epochs: int,
    seed: int,
    early_stop_metric: str = "val_loss",
    patience: int = 10,
) -> MLPConfig:
    """Build ``MLPConfig`` for ``retrain_mlp_on_full`` (shared by tuning and ``analysis/evaluation/eval_mlp``).

    ``patience`` / ``early_stop_metric`` are included for consistency with ``fit_mlp`` tooling; the refit
    helper only uses architecture, optimizer, batch, seed, and the separate ``epochs=`` argument.
    """
    return MLPConfig(
        hidden=int(best_params.get("hidden", 128)),
        drop=float(best_params.get("drop", 0.1)),
        epochs=int(retrain_epochs),
        batch_size=int(best_params.get("batch", 128)),
        lr=float(best_params.get("lr", 1e-3)),
        weight_decay=float(best_params.get("wd", 1e-4)),
        patience=int(patience),
        early_stop_metric=early_stop_metric,
        seed=int(seed),
    )


# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    """
    Simple MLP for numeric tabular classification.
    Matches your current "3 hidden layers + dropout + ReLU" style.
    """
    def __init__(self, d_in: int, hidden: int = 256, drop: float = 0.1, n_classes: int = 3):
        super().__init__()
        half = max(1, hidden // 2)

        self.fc1 = nn.Linear(d_in, half)
        self.fc2 = nn.Linear(half, half)
        self.fc3 = nn.Linear(half, half)
        self.do = nn.Dropout(drop)
        self.out = nn.Linear(half, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.do(x)
        x = F.relu(self.fc2(x))
        x = self.do(x)
        x = F.relu(self.fc3(x))
        x = self.do(x)
        return self.out(x)


# -----------------------------
# Utils
# -----------------------------
from .device_util import get_default_device


def _class_weights(y: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    # +1 smoothing so no division-by-zero; same idea as your current code.
    counts = np.bincount(y.astype(int), minlength=n_classes) + 1
    w = counts.sum() / counts
    return torch.tensor(w, dtype=torch.float32, device=device)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    X_t = torch.from_numpy(X.astype(np.float32, copy=False))
    y_t = torch.from_numpy(y.astype(np.int64, copy=False))
    ds = TensorDataset(X_t, y_t)

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        pin_memory=False,
    )


@torch.no_grad()
def predict_proba_mlp(
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
        logits = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def fit_mlp(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    config: MLPConfig,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train with early stopping and return:
      - best val metrics (val_loss/val_auc/val_acc/val_bacc/val_f1 if metric_fn provides them)
      - best proba on val and yva
      - best_state_dict (so caller can re-load for test-time missing-modality inference)
    """
    if device is None:
        device = get_default_device()

    if config.early_stop_metric not in ("val_loss", "val_auc"):
        raise ValueError(f"early_stop_metric must be 'val_loss' or 'val_auc', got: {config.early_stop_metric}")

    model = MLP(
        d_in=Xtr.shape[1],
        hidden=config.hidden,
        drop=config.drop,
        n_classes=config.n_classes,
    ).to(device)

    weights = _class_weights(ytr, config.n_classes, device)
    crit = nn.CrossEntropyLoss(weight=weights)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dl_tr = _make_loader(
        Xtr, ytr,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    dl_va = _make_loader(
        Xva, yva,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    # history collector: per-epoch train/val losses and metrics
    history = []

    # tracking
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

    for ep in range(1, config.epochs + 1):
        # ---- train ----
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
        # ---- compute train metrics (run inference on train set) ----
        model.eval()
        train_loss = 0.0
        P_tr_list, Y_tr_list = [], []
        with torch.no_grad():
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                train_loss += loss.item() * xb.size(0)
                P_tr_list.append(torch.softmax(logits, dim=1).cpu().numpy())
                Y_tr_list.append(yb.cpu().numpy())

        train_loss /= len(dl_tr.dataset)
        P_tr = np.vstack(P_tr_list) if P_tr_list else np.zeros((len(ytr), config.n_classes), dtype=float)
        Y_tr = np.concatenate(Y_tr_list) if Y_tr_list else np.array([])
        train_metrics: Dict[str, float] = {}
        if metric_fn is not None and Y_tr.size > 0:
            try:
                train_metrics = metric_fn(Y_tr, P_tr)
            except Exception:
                train_metrics = {}

        # ---- val ----
        model.eval()
        val_loss = 0.0
        P_list, Y_list = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
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

        # record epoch history
        history.append({
            "epoch": int(ep),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_metrics": {k: float(v) for k, v in (train_metrics or {}).items()},
            "val_metrics": {k: float(v) for k, v in (m or {}).items()},
        })

        # print
        if verbose:
            # use whatever metric_fn provides; fall back to NaNs
            auc = float(m.get("auc", float("nan")))
            acc = float(m.get("acc", float("nan")))
            bacc = float(m.get("bacc", float("nan")))
            f1 = float(m.get("f1", float("nan")))
            print(
                f"Epoch {ep:03d} | val_loss={val_loss:.4f} "
                f"| AUC={auc:.4f} | Acc={acc:.4f} | BAcc={bacc:.4f} | F1={f1:.4f}"
            )

        # early stop decision
        if config.early_stop_metric == "val_loss":
            improved = val_loss < best["val_loss"] - config.improvement_tol
        else:
            # require metric_fn to provide auc
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

    # fallback if never improved (should be rare)
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_proba is None:
        best_proba = predict_proba_mlp(model, Xva, batch_size=config.batch_size, device=device)

    return {
        **best,
        "proba": best_proba,
        "yva": yva,
        "best_state_dict": best_state,
        "best_epoch": locals().get("best_epoch", None),
        "model_config": config,
        "history": history,
    }


def load_mlp_from_state(
    d_in: int,
    state_dict: Dict[str, torch.Tensor],
    *,
    config: MLPConfig,
    device: Optional[torch.device] = None,
) -> nn.Module:
    if device is None:
        device = get_default_device()
    model = MLP(d_in=d_in, hidden=config.hidden, drop=config.drop, n_classes=config.n_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def retrain_mlp_on_full(
    X_all: np.ndarray,
    y_all: np.ndarray,
    *,
    config: MLPConfig,
    device: Optional[torch.device] = None,
    epochs: int = 0,
    Xte: Optional[np.ndarray] = None,
    yte: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """Train an MLP on the combined train+val data for a given number of epochs.

    Returns the final model.state_dict() (CPU tensors).
    If Xte and yte are provided, prints test metrics at each epoch.

    Uses ``config.seed`` for ``torch.manual_seed`` and the training ``DataLoader`` shuffle
    generator so repeated calls with the same data and config match (CPU). Some GPU ops may
    still be nondeterministic unless the environment enforces deterministic algorithms.
    """
    if device is None:
        device = get_default_device()

    # Match fit_mlp reproducibility: fixed init + seeded shuffle (shuffle=True without a generator
    # makes batch order and prior global RNG state vary across processes/runs).
    s = int(config.seed)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    model = MLP(d_in=X_all.shape[1], hidden=config.hidden, drop=config.drop, n_classes=config.n_classes).to(device)
    weights = _class_weights(y_all, config.n_classes, device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    ds = TensorDataset(torch.from_numpy(X_all.astype(np.float32)), torch.from_numpy(y_all.astype(np.int64)))
    g = torch.Generator()
    g.manual_seed(s)
    dl = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=g,
    )

    for ep in range(1, max(1, epochs) + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            if config.grad_clip and config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()

        if Xte is not None and yte is not None:
            model.eval()
            with torch.no_grad():
                xte_t = torch.from_numpy(Xte.astype(np.float32)).to(device)
                logits_te = model(xte_t)
                proba_te = torch.softmax(logits_te, dim=1).cpu().numpy()
            from utils import eval_multiclass_metrics
            m_te = eval_multiclass_metrics(yte, proba_te)
            if m_te:
                print(
                    f"[RETRAIN] Epoch {ep:03d} | "
                    f"test_auc={m_te.get('auc', float('nan')):.4f} | "
                    f"test_acc={m_te.get('acc', float('nan')):.4f} | "
                    f"test_f1={m_te.get('f1', float('nan')):.4f}"
                )

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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
    """Do Optuna tuning for MLP, retrain on train+val and evaluate on test.

    Returns a dict with keys similar to other train_val_test helpers:
      - best_params, best_val_auc, test_metrics, model_path (if saved), meta_path (if saved), n_trials
    """
    try:
        import optuna
    except Exception:
        raise RuntimeError("optuna is required for MLP train_val_test tuning")

    Xtr, ytr, scaler = None, None, None
    # build arrays using simple mean-impute + scaler compatible with runners
    try:
        from baselines.preprocessing import _build_xy
        Xtr, ytr, scaler = _build_xy(df, cols, tr_idx, None)
        Xva, yva, _ = _build_xy(df, cols, va_idx, scaler)
        Xte, yte, _ = _build_xy(df, cols, test_idx, scaler)
    except Exception:
        raise

    def objective(trial):
        hidden = trial.suggest_int("hidden", 64, 512)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
        drop = trial.suggest_float("drop", 0.0, 0.5)
        batch = trial.suggest_categorical("batch", [64, 128, 256])
        epochs = 50
        # compute validation metrics during fit so Optuna receives a numeric objective
        from baselines.mlp import MLPConfig, fit_mlp
        from utils import eval_multiclass_metrics
        cfg = MLPConfig(
            hidden=hidden,
            drop=drop,
            epochs=epochs,
            batch_size=batch,
            lr=lr,
            weight_decay=wd,
            patience=10,
            early_stop_metric=early_stop_metric,
            seed=seed,
        )
        out = fit_mlp(Xtr, ytr, Xva, yva, config=cfg, metric_fn=eval_multiclass_metrics)
        # record the trial-level best_epoch so we can reuse it later without
        # needing to re-run the best hyperparams. Also, if an out_dir was
        # provided, persist the trial's best_state_dict and out metadata so the
        # winning trial can be inspected/loaded after the study.
        try:
            trial.set_user_attr("best_epoch", int(out.get("best_epoch")) if out.get("best_epoch") is not None else None)
        except Exception:
            trial.set_user_attr("best_epoch", out.get("best_epoch"))

        # persist per-trial artifacts when out_dir is provided
        if out_dir is not None:
            try:
                import os, pickle
                trials_dir = os.path.join(out_dir, "mlp_tune_trials")
                os.makedirs(trials_dir, exist_ok=True)
                tname = f"trial_{trial.number}"
                state_path = os.path.join(trials_dir, f"{tname}.state.pkl")
                out_path = os.path.join(trials_dir, f"{tname}.out.pkl")
                # save model state
                with open(state_path, "wb") as sf:
                    pickle.dump(out.get("best_state_dict"), sf)
                # save out metadata (metrics, best_epoch, history)
                with open(out_path, "wb") as of:
                    pickle.dump({k: v for k, v in out.items() if k != "best_state_dict"}, of)
                trial.set_user_attr("trial_state_path", state_path)
                trial.set_user_attr("trial_out_path", out_path)
            except Exception:
                # don't fail the trial on IO problems; just continue
                pass
        # Return the requested selection metric (e.g., 'val_auc', 'val_acc', 'val_f1', 'val_bacc', 'val_loss')
        return float(out.get(select_metric, float("nan")))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # make a mutable copy of the best params so we can add extra metadata
    # (some callers expect 'epochs' to live in best_params)
    best_params = dict(study.best_params)

    # Try to obtain full validation metrics and the early-stopping epoch from
    # the winning trial's recorded artifacts to avoid re-running the best
    # hyperparameters. If per-trial artifacts aren't available, fall back to
    # re-running the best hyperparameters (previous behavior).
    from utils import eval_multiclass_metrics
    out_best = None
    val_metrics_best = {k: float("nan") for k in ("val_loss", "val_auc", "val_acc", "val_bacc", "val_f1")}
    try:
        best_trial = study.best_trial
        # prefer trial user-attrs if they include a saved state and out
        trial_state_path = best_trial.user_attrs.get("trial_state_path") if best_trial is not None else None
        trial_out_path = best_trial.user_attrs.get("trial_out_path") if best_trial is not None else None
        if trial_state_path and trial_out_path:
            try:
                import pickle
                with open(trial_out_path, "rb") as of:
                    out_best_meta = pickle.load(of)
                # load the saved state and compute val proba/metrics without retraining
                with open(trial_state_path, "rb") as sf:
                    state = pickle.load(sf)
                cfg_best = MLPConfig(
                    hidden=best_params.get("hidden", 128),
                    drop=best_params.get("drop", 0.1),
                    epochs=int(best_params.get("epochs", 50)),
                    batch_size=int(best_params.get("batch", 128)),
                    lr=best_params.get("lr", 1e-3),
                    weight_decay=best_params.get("wd", 1e-4),
                    patience=10,
                    early_stop_metric=early_stop_metric,
                    seed=seed,
                )
                # build model from saved state and score on validation set
                model = load_mlp_from_state(Xtr.shape[1], state, config=cfg_best)
                proba_va = predict_proba_mlp(model, Xva)
                m = eval_multiclass_metrics(yva, proba_va)
                val_metrics_best = {
                    "val_loss": float(out_best_meta.get("val_loss", float("nan"))) if out_best_meta is not None else float("nan"),
                    "val_auc": float(m.get("auc", float("nan"))),
                    "val_acc": float(m.get("acc", float("nan"))),
                    "val_bacc": float(m.get("bacc", float("nan"))),
                    "val_f1": float(m.get("f1", float("nan"))),
                }
                out_best = {**(out_best_meta or {}), "proba": proba_va}
            except Exception:
                out_best = None

        # If we couldn't load saved trial artifacts, fall back to the old behaviour
        if out_best is None:
            cfg_best = MLPConfig(
                hidden=best_params.get("hidden", 128),
                drop=best_params.get("drop", 0.1),
                epochs=int(best_params.get("epochs", 50)),
                batch_size=int(best_params.get("batch", 128)),
                lr=best_params.get("lr", 1e-3),
                weight_decay=best_params.get("wd", 1e-4),
                patience=10,
                early_stop_metric=early_stop_metric,
                seed=seed,
            )
            out_best = fit_mlp(Xtr, ytr, Xva, yva, config=cfg_best, metric_fn=eval_multiclass_metrics)
            val_metrics_best = {
                "val_loss": float(out_best.get("val_loss", float("nan"))),
                "val_auc": float(out_best.get("val_auc", float("nan"))),
                "val_acc": float(out_best.get("val_acc", float("nan"))),
                "val_bacc": float(out_best.get("val_bacc", float("nan"))),
                "val_f1": float(out_best.get("val_f1", float("nan"))),
            }
    except Exception as e:  # pragma: no cover - defensive logging for debugging
        import traceback
        print(f"[WARN] failed to compute full validation metrics for best hyperparams: {e}")
        traceback.print_exc()

    # Retrain on train+val (or use early-stopped best_state when requested)
    # Use the best epoch found during the re-run on (train, val) if available
    # (out_best is produced above when recomputing full validation metrics). If
    # that information is not available, fall back to the tuned 'epochs'
    # hyperparameter.
    tuned_epochs = int(best_params.get("epochs", 50))
    best_epoch = None
    if "out_best" in locals() and out_best is not None:
        try:
            be = out_best.get("best_epoch", None)
            if be is not None:
                best_epoch = int(be)
        except Exception:
            best_epoch = None

    retrain_epochs = best_epoch if best_epoch is not None else tuned_epochs

    cfg = mlp_config_for_retrain(
        best_params,
        retrain_epochs=int(retrain_epochs),
        seed=seed,
        early_stop_metric=early_stop_metric,
        patience=10,
    )

    X_all = np.vstack([Xtr, Xva])
    y_all = np.concatenate([ytr, yva])

    # By default we retrain on TRAIN+VAL to build the final model used for TEST.
    # When retrain_on_full==False we instead use the early-stopped best_state
    # produced during the validation snapshot (out_best) or any saved trial
    # state that was loaded earlier.
    state = None
    if retrain_on_full:
        state = retrain_mlp_on_full(
            X_all,
            y_all,
            config=cfg,
            device=None,
            epochs=retrain_epochs,
            Xte=Xte,
            yte=yte,
        )
    else:
        # prefer saved state from out_best (fit_mlp returns best_state_dict)
        if "out_best" in locals() and out_best is not None and out_best.get("best_state_dict") is not None:
            state = out_best.get("best_state_dict")
        else:
            # fallback to any trial-loaded state in locals (when trial artifacts were used)
            state = locals().get("state", None)
        # final fallback: if no saved state is available, retrain anyway to ensure we can evaluate
        if state is None:
            state = retrain_mlp_on_full(X_all, y_all, config=cfg, device=None, epochs=retrain_epochs)

    # Expose the actual epoch information in best_params so callers and
    # external wrappers (e.g., tuning scripts) can see which epoch was used
    # when retraining on train+val
    try:
        best_params["epochs"] = int(retrain_epochs) if retrain_epochs is not None else None
    except Exception:
        best_params["epochs"] = retrain_epochs

    model_path = None
    meta_path = None
    if out_dir is not None:
        import os, pickle
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "model.mlp.state.pt")
        with open(os.path.join(out_dir, "model.mlp.state.pkl"), "wb") as mf:
            pickle.dump(state, mf)
        meta = {"cols": cols, "scaler": scaler}
        meta_path = os.path.join(out_dir, "model.meta.pkl")
        with open(meta_path, "wb") as mf:
            pickle.dump(meta, mf)

    # Evaluate on test: load final model state into model and predict (use module-level helpers)
    cfg_for_load = cfg
    model = load_mlp_from_state(Xte.shape[1], state, config=cfg_for_load)
    proba = predict_proba_mlp(model, Xte)
    # compute metrics using shared utils
    from utils import eval_multiclass_metrics
    m_test = eval_multiclass_metrics(yte, proba)
    # --- Second-best trial analysis: try to locate the runner's second-best trial
    second = None
    try:
        finished = [t for t in study.trials if t.value is not None]
        finished_sorted = sorted(finished, key=lambda t: float(t.value), reverse=True)
        if len(finished_sorted) > 1:
            second = finished_sorted[1]
    except Exception:
        second = None

    second_params = None
    second_val_metrics = {k: float('nan') for k in ("val_loss", "val_auc", "val_acc", "val_bacc", "val_f1")}
    second_test_metrics = {k: float('nan') for k in ("auc", "acc", "f1")}
    second_val_metric_val = float('nan')
    if second is not None:
        try:
            second_params = dict(second.params)
            # Build MLPConfig for second-best
            cfg_second = mlp_config_for_retrain(
                second_params,
                retrain_epochs=int(second_params.get("epochs", 50)),
                seed=seed,
                early_stop_metric=early_stop_metric,
                patience=10,
            )
            # Compute validation snapshot for second-best by fitting on TRAIN only
            try:
                out_sec = fit_mlp(Xtr, ytr, Xva, yva, config=cfg_second, metric_fn=eval_multiclass_metrics)
                second_val_metrics = {
                    "val_loss": float(out_sec.get("val_loss", float("nan"))),
                    "val_auc": float(out_sec.get("val_auc", float("nan"))),
                    "val_acc": float(out_sec.get("val_acc", float("nan"))),
                    "val_bacc": float(out_sec.get("val_bacc", float("nan"))),
                    "val_f1": float(out_sec.get("val_f1", float("nan"))),
                }
                second_val_metric_val = float(second.value) if getattr(second, 'value', None) is not None else float('nan')
            except Exception:
                # keep NaNs if the validation snapshot fails
                pass

            # Retrain second-best on TRAIN+VAL and evaluate on TEST
            try:
                # choose retrain epochs from the validation snapshot if available
                retrain_epochs = int(out_sec.get("best_epoch")) if (out_sec is not None and out_sec.get("best_epoch") is not None) else int(cfg_second.epochs)
            except Exception:
                retrain_epochs = int(cfg_second.epochs)

            # persist the retrain epoch into second_params for provenance
            try:
                if second_params is not None:
                    second_params["epochs"] = int(retrain_epochs)
            except Exception:
                pass

            try:
                state2 = retrain_mlp_on_full(X_all, y_all, config=cfg_second, device=None, epochs=retrain_epochs)
                model2 = load_mlp_from_state(Xte.shape[1], state2, config=cfg_second)
                proba_test2 = predict_proba_mlp(model2, Xte)
                from utils import eval_multiclass_metrics as _eval2
                m_test2 = _eval2(yte, proba_test2)
                second_test_metrics = {"auc": float(m_test2.get('auc', float('nan'))), "acc": float(m_test2.get('acc', float('nan'))), "f1": float(m_test2.get('f1', float('nan')))}
            except Exception:
                # keep NaNs on failure
                pass
        except Exception:
            second_params = None

    return {
        "best_params": best_params,
        # populate best_val_auc from the re-run validation snapshot so the field is
        # always the full-validation AUC (not tied to selection metric). This keeps
        # backwards compatibility while avoiding NaNs when select_metric != 'val_auc'.
        "best_val_auc": float(val_metrics_best.get("val_auc", float("nan"))),
        "best_val_metric_name": select_metric,
        "best_val_metric": float(study.best_value),
        # also expose full validation metrics computed for the best hyperparameters
        "best_val_loss": float(val_metrics_best.get("val_loss", float("nan"))),
        "best_val_acc": float(val_metrics_best.get("val_acc", float("nan"))),
        "best_val_bacc": float(val_metrics_best.get("val_bacc", float("nan"))),
        "best_val_f1": float(val_metrics_best.get("val_f1", float("nan"))),
        # test metrics from the model retrained on TRAIN+VAL using best params
        "test_metrics": {"auc": float(m_test.get("auc", float("nan"))), "acc": float(m_test.get("acc", float("nan"))), "f1": float(m_test.get("f1", float("nan")))},
        # second-best trial information (may be None / NaN when unavailable)
    "second_best_params": second_params,
    "second_best_epoch": int(second_params.get("epochs")) if isinstance(second_params, dict) and second_params.get("epochs") is not None else None,
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
        "n_trials": len(study.trials),
    }
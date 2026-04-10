"""Scikit-learn baselines.

This module implements classical tabular baselines (LogReg / RF / XGBoost if available)
behind a small, consistent interface so `scripts/train_baselines.py` can call them via the registry.

Design goals:
- Keep this file torch-free.
- Accept already-constructed numpy arrays (X_train, y_train, X_val, y_val, ...).
- Return class probabilities when possible.

Notes
-----
* For multiclass AUC we typically need per-class probabilities.
* Use ``make_logistic_regression``, ``make_random_forest_classifier``, and ``make_xgb_classifier``
  for tuning, refit, and ``analysis/evaluation/eval_*.py`` so constructors stay aligned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Canonical max_iter for project LogisticRegression (Optuna trials, refit, eval, runner).
LOGISTIC_REGRESSION_MAX_ITER = 1000


def make_logistic_regression(
    *, C: float, max_iter: Optional[int] = None, seed: int = 0
) -> LogisticRegression:
    """Project-standard `lbfgs` multinomial logistic regression (tuning, refit, eval, runner)."""
    mi = LOGISTIC_REGRESSION_MAX_ITER if max_iter is None else max_iter
    return LogisticRegression(C=C, max_iter=mi, solver="lbfgs", random_state=seed)


def make_random_forest_classifier(
    *,
    n_estimators: int,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    n_jobs: int = -1,
    seed: int = 42,
) -> RandomForestClassifier:
    """Project-standard ``RandomForestClassifier`` (Optuna, refit, eval, runner)."""
    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        n_jobs=n_jobs,
        random_state=seed,
    )


def make_xgb_classifier(
    *,
    n_estimators: int,
    max_depth: int = 6,
    learning_rate: float = 0.3,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    tree_method: Optional[str] = None,
    n_jobs: int = -1,
    random_state: int = 42,
    eval_metric: Optional[str] = None,
):
    """Project-standard ``XGBClassifier`` (multiclass). If ``tree_method`` is None, XGBoost library default is used."""
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:
        raise RuntimeError("xgboost is required for make_xgb_classifier") from e
    kw: Dict[str, Any] = {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "use_label_encoder": False,
        "verbosity": 0,
        "objective": "multi:softprob",
        "n_jobs": int(n_jobs),
        "random_state": int(random_state),
    }
    if tree_method is not None:
        kw["tree_method"] = str(tree_method)
    if eval_metric is not None:
        kw["eval_metric"] = eval_metric
    return xgb.XGBClassifier(**kw)


def _as_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


@dataclass
class SklearnFitResult:
    model_name: str
    model: Any


class SklearnRunner:
    """Minimal runner interface for sklearn-style estimators."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SklearnFitResult:
        raise NotImplementedError

    def predict_proba(self, fit: SklearnFitResult, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape (n, n_classes)."""
        raise NotImplementedError


class LogisticRegressionRunner(SklearnRunner):
    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = LOGISTIC_REGRESSION_MAX_ITER,
        n_jobs: int = 1,
        seed: int = 0,
    ):
        super().__init__("lr_all")
        self.C = C
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SklearnFitResult:
        X_train = _as_2d(X_train)
        clf = make_logistic_regression(C=self.C, max_iter=self.max_iter, seed=self.seed)
        clf.fit(X_train, y_train)
        return SklearnFitResult(self.model_name, clf)

    def predict_proba(self, fit: SklearnFitResult, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        return fit.model.predict_proba(X)


class RandomForestRunner(SklearnRunner):
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        super().__init__("rf_all")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SklearnFitResult:
        X_train = _as_2d(X_train)
        clf = make_random_forest_classifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            seed=self.seed,
        )
        clf.fit(X_train, y_train)
        return SklearnFitResult(self.model_name, clf)

    def predict_proba(self, fit: SklearnFitResult, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        return fit.model.predict_proba(X)


class XGBoostRunner(SklearnRunner):
    """XGBoost baseline (optional).

    Only works if xgboost is installed. We keep it optional so environments without xgboost
    can still run the other baselines.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        super().__init__("xgb_all")
        self.kw = dict(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=seed
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> SklearnFitResult:
        X_train = _as_2d(X_train)
        try:
            clf = make_xgb_classifier(
                n_estimators=int(self.kw["n_estimators"]),
                n_jobs=int(self.kw["n_jobs"]),
                random_state=int(self.kw["random_state"]),
                eval_metric="mlogloss",
            )
        except Exception as e:
            raise RuntimeError(
                "xgboost is not available but XGBoostRunner was requested. "
                "Install xgboost or choose a different baseline."
            ) from e

        clf.fit(X_train, y_train)
        return SklearnFitResult(self.model_name, clf)

    def predict_proba(self, fit: SklearnFitResult, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        return fit.model.predict_proba(X)


def build_sklearn_runner(name: str, *, seed: int = 0, n_jobs: int = 1) -> SklearnRunner:
    """Factory used by the baselines registry."""
    name = name.lower()

    if name in {"lr_all", "logreg", "logistic", "logistic_regression"}:
        return LogisticRegressionRunner(seed=seed, n_jobs=n_jobs)

    if name in {"rf_all", "random_forest", "rf"}:
        return RandomForestRunner(seed=seed, n_jobs=n_jobs)

    if name in {"xgb_all", "xgboost", "xgb"}:
        return XGBoostRunner(seed=seed, n_jobs=n_jobs)

    raise KeyError(f"Unknown sklearn baseline '{name}'.")


def train_val_test(
    df,
    cols,
    tr_idx,
    va_idx,
    test_idx,
    *,
    baseline_name: str,
    n_trials: int = 20,
    timeout: float = None,
    out_dir: str = None,
    seed: int = 42,
    select_metric: str = "val_auc",
    skip_retrain: bool = False,
):
    """Tune a sklearn baseline (rf_all/xgb_all/lr_all), retrain on train+val and evaluate on test.

    Returns dict with best_params, best_val_auc, test_metrics, model_path, meta_path, n_trials.
    """
    try:
        import optuna
    except Exception:
        raise RuntimeError("optuna is required for sklearn train_val_test tuning")

    # build X/y using baselines.preprocessing to match training pipeline
    try:
        from baselines.preprocessing import _build_xy
        Xtr, ytr, scaler = _build_xy(df, cols, tr_idx, None)
        Xva, yva, _ = _build_xy(df, cols, va_idx, scaler)
        Xte, yte, _ = _build_xy(df, cols, test_idx, scaler)
    except Exception:
        raise

    baseline = baseline_name.lower()

    def objective_rf(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 3, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        clf = make_random_forest_classifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            seed=seed,
        )
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xva)
        from utils import eval_multiclass_metrics
        m = eval_multiclass_metrics(yva, proba)
        metric_key = select_metric[4:] if select_metric.startswith("val_") else select_metric
        return float(m.get(metric_key, float("nan")))

    def objective_xgb(trial):
        try:
            clf = make_xgb_classifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 500),
                max_depth=trial.suggest_int("max_depth", 3, 12),
                learning_rate=trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                tree_method=None,
                n_jobs=-1,
                random_state=seed,
            )
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xva)
        except Exception:
            return float("nan")
        from utils import eval_multiclass_metrics
        m = eval_multiclass_metrics(yva, proba)
        metric_key = select_metric[4:] if select_metric.startswith("val_") else select_metric
        return float(m.get(metric_key, float("nan")))

    def objective_lr(trial):
        C = trial.suggest_float('C', 1e-4, 1e2, log=True)
        clf = make_logistic_regression(C=C, seed=seed)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xva)
        from utils import eval_multiclass_metrics
        m = eval_multiclass_metrics(yva, proba)
        metric_key = select_metric[4:] if select_metric.startswith("val_") else select_metric
        return float(m.get(metric_key, float("nan")))

    study = optuna.create_study(direction='maximize')
    if baseline.startswith('rf'):
        study.optimize(objective_rf, n_trials=n_trials, timeout=timeout)
        best = study.best_params
        best_clf = make_random_forest_classifier(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            min_samples_split=best["min_samples_split"],
            n_jobs=-1,
            seed=seed,
        )
    elif baseline.startswith('xgb'):
        study.optimize(objective_xgb, n_trials=n_trials, timeout=timeout)
        best = study.best_params
        try:
            best_clf = make_xgb_classifier(
                n_estimators=best["n_estimators"],
                max_depth=best["max_depth"],
                learning_rate=best["learning_rate"],
                subsample=best["subsample"],
                colsample_bytree=best["colsample_bytree"],
                tree_method=None,
                n_jobs=-1,
                random_state=seed,
            )
        except Exception:
            best_clf = None
    elif baseline.startswith('lr'):
        study.optimize(objective_lr, n_trials=n_trials, timeout=timeout)
        best = study.best_params
        best_clf = make_logistic_regression(C=best['C'], seed=seed)
    else:
        raise ValueError(f"Unsupported sklearn baseline for tuning: {baseline_name}")

    # Before retraining on train+val, fit the selected best classifier on TRAIN
    # to compute full validation metrics for the chosen hyperparameters and
    # expose them in the returned result JSON.
    val_metrics_best = {"val_loss": float("nan"), "val_auc": float("nan"), "val_acc": float("nan"), "val_f1": float("nan"), "val_bacc": float("nan")}
    # Compute a validation snapshot for the chosen hyperparameters by
    # fitting a classifier on TRAIN only. We intentionally perform this
    # fit so we can expose validation metrics for the chosen
    # hyperparameters (val_metrics_best). However, to avoid accidentally
    # reusing a fitted estimator object (which could have side-effects for
    # estimators that support warm-start or hold state), clone the
    # estimator and use the clone for the final retrain on TRAIN+VAL below.
    try:
        if best_clf is not None:
            # Fit on TRAIN-only to compute the validation snapshot
            best_clf.fit(Xtr, ytr)
            proba_val = best_clf.predict_proba(Xva)
            from utils import eval_multiclass_metrics
            m_val = eval_multiclass_metrics(yva, proba_val)
            val_metrics_best = {
                "val_loss": float("nan"),
                "val_auc": float(m_val.get("auc", float("nan"))),
                "val_acc": float(m_val.get("acc", float("nan"))),
                "val_bacc": float(m_val.get("bacc", float("nan"))),
                "val_f1": float(m_val.get("f1", float("nan"))),
            }
    except Exception:
        # keep NaNs on failure
        pass

    # Retrain on TRAIN+VAL for final evaluation. We intentionally reuse the
    # selected estimator object and call fit() on the combined TRAIN+VAL set.
    # For the sklearn baselines used in this project (LogisticRegression,
    # RandomForest with warm_start=False), calling fit() reinitializes
    # learned state and produces an equivalent fresh fit, so cloning is not
    # necessary. Reusing the estimator keeps the code simple and consistent.
    X_all = np.vstack([Xtr, Xva])
    y_all = np.concatenate([ytr, yva])
    # Fit the selected estimator on TRAIN+VAL and keep using the same
    # `best_clf` variable as the final model. This keeps naming simple and
    # matches earlier behavior where fit() reinitializes the estimator.
 

    # Decide whether to retrain on TRAIN+VAL or keep the TRAIN-only snapshot
    if best_clf is not None:
        if skip_retrain:
            # Use TRAIN-only snapshot for test evaluation
            print(f"[INFO] skip_retrain=True: evaluating TEST using TRAIN-only snapshot for baseline {baseline_name}")
            proba_test = best_clf.predict_proba(Xte)
        else:
            # Fit final model on TRAIN+VAL
            print(f"[INFO] Retraining on TRAIN+VAL (n={len(X_all)}) with best params: {best}")
            best_clf.fit(X_all, y_all)
            proba_test = best_clf.predict_proba(Xte)
    from utils import eval_multiclass_metrics
    m_test = eval_multiclass_metrics(yte, proba_test) if proba_test is not None else {}
    print(f"[TEST RESULTS] AUC={m_test.get('auc', float('nan')):.4f} | ACC={m_test.get('acc', float('nan')):.4f} | F1={m_test.get('f1', float('nan')):.4f} | BACC={m_test.get('bacc', float('nan')):.4f}")

    # --- Second-best trial analysis: find the second-best set of params (if any)
    second = None
    try:
        # consider only finished trials with numeric values
        finished = [t for t in study.trials if t.value is not None]
        finished_sorted = sorted(finished, key=lambda t: float(t.value), reverse=True)
        if len(finished_sorted) > 1:
            second = finished_sorted[1]
    except Exception:
        second = None

    second_params = None
    second_val_metrics = {"val_loss": float("nan"), "val_auc": float("nan"), "val_acc": float("nan"), "val_bacc": float("nan"), "val_f1": float("nan")}
    second_test_metrics = {"auc": float("nan"), "acc": float("nan"), "f1": float("nan")}
    second_val_metric_val = float("nan")
    if second is not None:
        try:
            second_params = dict(second.params)
            # Build classifier for second params following same logic as above
            second_clf = None
            if baseline.startswith('rf'):
                second_clf = make_random_forest_classifier(
                    n_estimators=second_params.get('n_estimators', 100),
                    max_depth=second_params.get('max_depth', None),
                    min_samples_split=second_params.get('min_samples_split', 2),
                    n_jobs=-1,
                    seed=seed,
                )
            elif baseline.startswith('xgb'):
                try:
                    second_clf = make_xgb_classifier(
                        n_estimators=second_params.get('n_estimators', 100),
                        max_depth=second_params.get('max_depth', 3),
                        learning_rate=second_params.get('learning_rate', 0.01),
                        subsample=second_params.get('subsample', 1.0),
                        colsample_bytree=second_params.get('colsample_bytree', 1.0),
                        tree_method=None,
                        n_jobs=-1,
                        random_state=seed,
                    )
                except Exception:
                    second_clf = None
            elif baseline.startswith('lr'):
                second_clf = make_logistic_regression(C=second_params.get('C', 1.0), seed=seed)

            # Compute validation snapshot for second-best by fitting on TRAIN only
            if second_clf is not None:
                try:
                    second_clf.fit(Xtr, ytr)
                    proba_val2 = second_clf.predict_proba(Xva)
                    from utils import eval_multiclass_metrics
                    m_val2 = eval_multiclass_metrics(yva, proba_val2)
                    second_val_metrics = {
                        "val_loss": float("nan"),
                        "val_auc": float(m_val2.get("auc", float("nan"))),
                        "val_acc": float(m_val2.get("acc", float("nan"))),
                        "val_bacc": float(m_val2.get("bacc", float("nan"))),
                        "val_f1": float(m_val2.get("f1", float("nan"))),
                    }
                    second_val_metric_val = float(second.value) if getattr(second, 'value', None) is not None else float("nan")
                except Exception:
                    # keep NaNs on failure of validation snapshot
                    pass

            # Either evaluate TEST using the TRAIN-only snapshot or retrain on TRAIN+VAL
            try:
                # Debug info: ensure we have shapes and predict_proba available
                try:
                    print(f"[DEBUG_SECOND] second_clf={type(second_clf)} has_predict_proba={hasattr(second_clf, 'predict_proba')} Xtr.shape={getattr(Xtr, 'shape', None)} Xva.shape={getattr(Xva, 'shape', None)} Xte.shape={getattr(Xte, 'shape', None)}")
                except Exception:
                    pass
                if skip_retrain:
                    # Use the TRAIN-only snapshot already fit above
                    proba_test2 = second_clf.predict_proba(Xte)
                else:
                    try:
                        second_clf.fit(X_all, y_all)
                        proba_test2 = second_clf.predict_proba(Xte)
                    except Exception as e:
                        # If retrain on TRAIN+VAL fails for any reason,
                        # fall back to evaluating the TRAIN-only snapshot
                        # to ensure we at least provide comparable metrics.
                        try:
                            proba_test2 = second_clf.predict_proba(Xte)
                            print(f"[WARN] retrain on TRAIN+VAL failed for second-best; falling back to TRAIN-only snapshot: {e}")
                        except Exception:
                            raise

                m_test2 = eval_multiclass_metrics(yte, proba_test2)
                second_test_metrics = {
                    "auc": float(m_test2.get('auc', float('nan'))),
                    "acc": float(m_test2.get('acc', float('nan'))),
                    "f1": float(m_test2.get('f1', float('nan'))),
                }
            except Exception as e:
                # Log the failure so the user can see why second-best TEST
                # metrics are missing instead of silently swallowing.
                import traceback
                print(f"[WARN] failed to compute second_best_test_metrics for baseline {baseline_name}: {e}")
                traceback.print_exc()
        except Exception:
            # keep NaNs on any failure
            second_params = None


    model_path = None
    meta_path = None
    # Save the final model (the one trained on TRAIN+VAL) if requested.
    if out_dir is not None and best_clf is not None:
        import os, joblib, pickle
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, 'model.joblib')
        joblib.dump(best_clf, model_path)
        meta = {'cols': cols, 'scaler': scaler, 'n_classes': int(len(np.unique(y_all)))}
        meta_path = os.path.join(out_dir, 'model.meta.pkl')
        with open(meta_path, 'wb') as mf:
            pickle.dump(meta, mf)

    return {
        'best_params': study.best_params,
        # Always populate best_val_auc from the re-run validation snapshot so AUC
        # is available even when another metric was selected for tuning.
        'best_val_auc': float(val_metrics_best.get('val_auc', float('nan'))),
        'best_val_metric_name': select_metric,
        'best_val_metric': float(study.best_value),
        # also expose validation metrics computed for the best hyperparameters
        'best_val_loss': float(val_metrics_best.get('val_loss', float('nan'))),
        'best_val_acc': float(val_metrics_best.get('val_acc', float('nan'))),
        'best_val_bacc': float(val_metrics_best.get('val_bacc', float('nan'))),
        'best_val_f1': float(val_metrics_best.get('val_f1', float('nan'))),
        'test_metrics': {'auc': float(m_test.get('auc', float('nan'))), 'acc': float(m_test.get('acc', float('nan'))), 'f1': float(m_test.get('f1', float('nan')) )},
        # second-best trial information (may be None / NaN when unavailable)
        'second_best_params': second_params,
        'second_best_val_auc': float(second_val_metrics.get('val_auc', float('nan'))),
        'second_best_val_metric_name': select_metric,
        'second_best_val_metric': float(second_val_metric_val),
        'second_best_val_loss': float(second_val_metrics.get('val_loss', float('nan'))),
        'second_best_val_acc': float(second_val_metrics.get('val_acc', float('nan'))),
        'second_best_val_bacc': float(second_val_metrics.get('val_bacc', float('nan'))),
        'second_best_val_f1': float(second_val_metrics.get('val_f1', float('nan'))),
        'second_best_test_metrics': {'auc': float(second_test_metrics.get('auc', float('nan'))), 'acc': float(second_test_metrics.get('acc', float('nan'))), 'f1': float(second_test_metrics.get('f1', float('nan')) )},
        'model_path': model_path,
        'meta_path': meta_path,
        'n_trials': len(study.trials),
    }


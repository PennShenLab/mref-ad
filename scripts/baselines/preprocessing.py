"""Preprocessing helpers extracted from `scripts/train_baselines.py`.

Provides median-impute + StandardScaler helpers and the "mean-from-train" policy
used by the FT-Transformer official wrapper.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def median_impute_and_scale(Xtr: np.ndarray, Xva: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Median-impute numpy arrays and fit/apply a StandardScaler on TRAIN.

    Parameters
    ----------
    Xtr, Xva : np.ndarray
        Numeric arrays with shape (n_samples, n_features) possibly containing NaNs.

    Returns
    -------
    Xtr_scaled, Xva_scaled : np.ndarray
        Scaled outputs as float32.
    """
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xva = np.where(np.isnan(Xva), med, Xva)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr).astype(np.float32), sc.transform(Xva).astype(np.float32)


def _build_xy(df: pd.DataFrame, cols: List[str], idx: List[int], scaler: StandardScaler | None = None):
    """Build (X, y, scaler) with median imputation and StandardScaler.

    This preserves the original `_build_xy` contract used in the CLI so the
    minimal changes are required when replacing the in-file implementation.
    """
    Xdf = df.iloc[idx][cols].astype(float)

    # Column-wise medians; if a column is entirely NaN, median is NaN.
    med = Xdf.median(numeric_only=True)
    med = med.fillna(0.0)

    # Fill NaNs with median (or 0.0 fallback for all-NaN columns)
    Xdf = Xdf.fillna(med)

    # Defensive: if anything is still NaN (should be rare), fill with 0.0
    Xdf = Xdf.fillna(0.0)

    if scaler is None:
        scaler = StandardScaler().fit(Xdf)

    X = scaler.transform(Xdf).astype(np.float32)
    y = df.iloc[idx]["y"].astype(int).values
    return X, y, scaler


def _build_xy_mean_from_train(df: pd.DataFrame, cols: List[str], tr_idx: List[int], va_idx: List[int]):
    """Official-like num_nan_policy='mean': compute means on TRAIN, apply to train/val.

    Returns Xtr, ytr, Xva, yva (X arrays are float32).
    """
    Xtr_df = df.iloc[tr_idx][cols].astype(float)
    Xva_df = df.iloc[va_idx][cols].astype(float)

    mu = Xtr_df.mean(numeric_only=True).fillna(0.0)
    Xtr_df = Xtr_df.fillna(mu).fillna(0.0)
    Xva_df = Xva_df.fillna(mu).fillna(0.0)

    sc = StandardScaler().fit(Xtr_df)
    Xtr = sc.transform(Xtr_df).astype(np.float32)
    Xva = sc.transform(Xva_df).astype(np.float32)

    ytr = df.iloc[tr_idx]["y"].astype(int).values
    yva = df.iloc[va_idx]["y"].astype(int).values
    return Xtr, ytr, Xva, yva, sc, mu

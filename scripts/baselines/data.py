# baselines/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularXY:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xva: np.ndarray
    yva: np.ndarray
    cols: List[str]
    tr_mask: Optional[np.ndarray] = None  # (Ntr, 1 + D) bool, True = mask token
    va_mask: Optional[np.ndarray] = None  # (Nva, 1 + D) bool, True = mask token


def concat_cols(
    groups: Dict[str, List[str]],
    df: Optional[pd.DataFrame] = None,
    include_has_flags: bool = False,
) -> List[str]:
    """
    Concatenate feature columns across all experts.

    If include_has_flags=True, append `has_{expert}` columns when present in df.
    (If df is None, we append the flags unconditionally.)

    Deduplicates while preserving order.
    """
    cols: List[str] = [c for _, feats in groups.items() for c in feats]
    if include_has_flags:
        for name in groups.keys():
            h = f"has_{name}"
            if df is None or (hasattr(df, "columns") and h in df.columns):
                cols.append(h)

    # de-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _impute_numeric(
    Xtr_df: pd.DataFrame,
    Xva_df: pd.DataFrame,
    nan_policy: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute NaNs consistently.

    - nan_policy="mean": compute mean on TRAIN only and apply to train/val (official-style)
    - nan_policy="median": compute median on TRAIN only and apply to train/val
    - nan_policy="zero": fill NaNs with 0.0 in both

    For columns that are all-NaN in train, mean/median becomes NaN -> we fall back to 0.0.
    """
    nan_policy = nan_policy.lower()
    if nan_policy not in ("mean", "median", "zero"):
        raise ValueError(f"nan_policy must be one of ['mean','median','zero'], got '{nan_policy}'")

    if nan_policy == "zero":
        return Xtr_df.fillna(0.0), Xva_df.fillna(0.0)

    if nan_policy == "mean":
        fill = Xtr_df.mean(numeric_only=True).fillna(0.0)
    else:
        fill = Xtr_df.median(numeric_only=True).fillna(0.0)

    Xtr_df = Xtr_df.fillna(fill).fillna(0.0)
    Xva_df = Xva_df.fillna(fill).fillna(0.0)
    return Xtr_df, Xva_df


def _scale_train_only(
    Xtr_df: pd.DataFrame,
    Xva_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize using train-only fit, then apply to val.
    """
    sc = StandardScaler().fit(Xtr_df)
    Xtr = sc.transform(Xtr_df).astype(np.float32)
    Xva = sc.transform(Xva_df).astype(np.float32)
    return Xtr, Xva


def _ftt_feature_owner(cols: List[str], groups: Dict[str, List[str]]) -> List[Optional[str]]:
    """
    For each column in `cols`, return the expert name that owns it, or None.

    - Feature cols in groups[expert] map to that expert.
    - Indicator cols like has_{expert} map to None (never masked).
    """
    feat_to_exp: Dict[str, str] = {}
    for exp, feats in groups.items():
        for c in feats:
            feat_to_exp[c] = exp

    owners: List[Optional[str]] = []
    for c in cols:
        if c.startswith("has_"):
            owners.append(None)
        else:
            owners.append(feat_to_exp.get(c, None))
    return owners


def build_token_key_padding_mask(
    df: pd.DataFrame,
    idx: List[int],
    cols: List[str],
    groups: Dict[str, List[str]],
) -> np.ndarray:
    """
    Build FT-Transformer-style key_padding_mask.

    Returns bool array (N, 1 + D) where True means "mask/ignore".
    Masks feature-tokens that belong to experts missing for that sample.
    CLS (pos 0) is never masked.

    If has_{expert} is missing from df, we treat that expert as always present.
    """
    owners = _ftt_feature_owner(cols, groups)  # length D
    N = len(idx)
    D = len(cols)
    mask = np.zeros((N, 1 + D), dtype=bool)

    # availability per expert
    avail: Dict[str, np.ndarray] = {}
    for exp in groups.keys():
        h = f"has_{exp}"
        if h in df.columns:
            avail[exp] = df.iloc[idx][h].to_numpy().astype(bool)
        else:
            avail[exp] = np.ones(N, dtype=bool)

    for j, exp in enumerate(owners):
        if exp is None:
            continue
        mask[:, 1 + j] = ~avail[exp]

    return mask


def build_tabular_xy(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    tr_idx: List[int],
    va_idx: List[int],
    *,
    cols: Optional[List[str]] = None,
    include_has_flags: bool = False,
    nan_policy: str = "mean",
    standardize: bool = True,
    return_ftt_mask: bool = False,
) -> TabularXY:
    """
    Convert df -> (Xtr,ytr,Xva,yva) with consistent preprocessing.

    Defaults are chosen to match FT-Transformer official-style preprocessing:
      - nan_policy="mean" on TRAIN
      - StandardScaler fit on TRAIN

    If return_ftt_mask=True, also returns token-level masks for FT-Transformer.
    """
    if cols is None:
        cols = concat_cols(groups, df=df, include_has_flags=include_has_flags)

    Xtr_df = df.iloc[tr_idx][cols].astype(float)
    Xva_df = df.iloc[va_idx][cols].astype(float)

    Xtr_df, Xva_df = _impute_numeric(Xtr_df, Xva_df, nan_policy=nan_policy)

    if standardize:
        Xtr, Xva = _scale_train_only(Xtr_df, Xva_df)
    else:
        Xtr = Xtr_df.to_numpy(dtype=np.float32)
        Xva = Xva_df.to_numpy(dtype=np.float32)

    ytr = df.iloc[tr_idx]["y"].astype(int).to_numpy()
    yva = df.iloc[va_idx]["y"].astype(int).to_numpy()

    tr_mask = va_mask = None
    if return_ftt_mask:
        tr_mask = build_token_key_padding_mask(df, tr_idx, cols, groups)
        va_mask = build_token_key_padding_mask(df, va_idx, cols, groups)

    return TabularXY(
        Xtr=Xtr,
        ytr=ytr,
        Xva=Xva,
        yva=yva,
        cols=cols,
        tr_mask=tr_mask,
        va_mask=va_mask,
    )
#!/usr/bin/env python3
import os, re, json, random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score

SEED = 42

# ----------------------------
# Seeds
# ----------------------------
def set_seed(seed=42):
    import os, random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # determinism:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# I/O + schema helpers
# ----------------------------
def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "SCANDATE" in df.columns:
        df["SCANDATE"] = pd.to_datetime(df["SCANDATE"], errors="coerce").dt.date.astype(str)
    if "VISCODE" in df.columns:
        df["VISCODE"] = df["VISCODE"].astype(str).str.strip()
    return df

def read_csv(path: str) -> pd.DataFrame:
    if not path:
        raise FileNotFoundError("Missing path")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for k in ["PTID.x", "PTID.y"]:
        if k in df.columns and "PTID" not in df.columns:
            df = df.rename(columns={k: "PTID"})
    return normalize_dates(df)

def pick_cols(df: pd.DataFrame, suffixes: Tuple[str, ...]) -> List[str]:
    pat = re.compile(rf"({'|'.join([re.escape(s) for s in suffixes])})$", re.I)
    return [c for c in df.columns if pat.search(c)]

def infer_diag_col(df: pd.DataFrame) -> str:
    for cand in ["DIAGNOSIS","DX","DX_bl","DXCHANGE"]:
        if cand in df.columns: return cand
    raise ValueError("No diagnosis/label column found.")

def _common_keys(a: pd.DataFrame, b: pd.DataFrame) -> List[str]:
    for keys in (["PTID","SCANDATE", "VISCODE"], ["PTID","SCANDATE"], ["PTID","VISCODE"], ["PTID"]):
        if all(k in a.columns for k in keys) and all(k in b.columns for k in keys):
            return keys
    raise ValueError("No common join keys among PTID/SCANDATE/VISCODE.")

def safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(left, right, how="outer", on=_common_keys(left, right), suffixes=("", "_dup"))

# ----------------------------
# Dataset builder (optional modalities)
# ----------------------------
# utils.py
def load_experts_from_yaml(cfg_path: str):
    """Wrapper: read YAML config of experts, then call build_experts."""
    import yaml, os
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    exp_map = cfg.get("experts", {})
    if not exp_map:
        raise ValueError("experts_config has no 'experts' mapping.")
    for name, path in exp_map.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}: {path}")
        
    return build_experts(exp_map)

def build_experts(expert_paths: dict[str, str]):
    """
    Load multiple expert CSVs (one per expert), merge on PTID+SCANDATE,
    and return (df, groups, classes).

    expert_paths: dict of {expert_name -> csv_path}
    - Each CSV must have PTID, SCANDATE, DIAGNOSIS plus the expert's feature columns.
    - Features are all non-key, non-label columns in that CSV.
    """
    import os
    import pandas as pd

    def _read_and_normalize(path: str, expert_name: str = "") -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        d = pd.read_csv(path)

        # Normalize PTID name
        if "PTID" not in d.columns:
            if "PTID.x" in d.columns:
                d = d.rename(columns={"PTID.x": "PTID"})
            elif "PTID.y" in d.columns:
                d = d.rename(columns={"PTID.y": "PTID"})
        if "PTID" not in d.columns:
            raise ValueError(f"{path} missing PTID")

        d["PTID"] = d["PTID"].astype(str).str.strip()

        # DEMOGRAPHIC expert may not have SCANDATE (static info)
        if expert_name.lower() != "demographic":
            if "SCANDATE" not in d.columns:
                raise ValueError(f"{path} missing SCANDATE")
            d["SCANDATE"] = pd.to_datetime(d["SCANDATE"], errors="coerce").dt.date.astype(str)
        else:
            if "SCANDATE" not in d.columns:
                d["SCANDATE"] = np.nan  # placeholder, will be filled later

        # Standardize diagnosis column
        if "DIAGNOSIS" not in d.columns:
            for alt in ["DX", "DX_bl", "DXCHANGE"]:
                if alt in d.columns:
                    d = d.rename(columns={alt: "DIAGNOSIS"})
                    break

        return d

    dfs, groups = [], {}

    for name, path in expert_paths.items():
        d = _read_and_normalize(path, name)

        # pick features for this expert (all non-key, non-label columns)
        keep_keys = [c for c in ["PTID", "SCANDATE", "DIAGNOSIS", "VISCODE"] if c in d.columns]
        feats = [c for c in d.columns if c not in keep_keys]

        if name.lower() == "demographic":
            feats = [f for f in feats if f not in ["PTDOB", "PTDOB_parsed", "PTID.1"]]


        groups[name] = feats
        dfs.append(d[keep_keys + feats])
    
    # After reading all experts, compute AGE_AT_VISIT if DEMOGRAPHIC present
    if "demographic" in expert_paths:
        print("[INFO] Found DEMOGRAPHIC expert – computing AGE_AT_VISIT using imaging SCANDATE...")
        df_demo = pd.read_csv(expert_paths["demographic"])
        df_demo["PTID"] = df_demo["PTID"].astype(str).str.strip()

        # Pick a reference imaging expert that has SCANDATE per visit
        ref_name = next((k for k in expert_paths.keys() if k != "demographic"), None)
        if ref_name:
            ref_df = pd.read_csv(expert_paths[ref_name])

            # Compute AGE_AT_VISIT and keep all other demographic columns
            df_demo = _compute_age_at_visit(df_demo, ref_df)

            print(df_demo.head())

            # Replace demographic DataFrame in memory (so merge below uses updated one)
            dfs = [df_demo if g.lower() == "demographic" else df for df, g in zip(dfs, groups.keys())]
        else:
            print("[WARN] No imaging reference found to compute AGE_AT_VISIT; demographic remains static.")

    if not dfs:
        raise ValueError("No expert CSVs provided.")

    # Merge all experts on PTID + SCANDATE (outer to preserve union of visits)
    base = dfs[0]
    for nxt in dfs[1:]:
        base = pd.merge(base, nxt, on=["PTID", "SCANDATE"], how="outer", suffixes=("", "_dup"))

        # Coalesce any duplicated DIAGNOSIS columns created by the merge
        diag_like = [c for c in base.columns if c.startswith("DIAGNOSIS")]
        if len(diag_like) > 1:
            # row-wise first non-null across any DIAGNOSIS* columns
            base["DIAGNOSIS"] = base[diag_like].bfill(axis=1).iloc[:, 0]
            # drop the dups (keep canonical 'DIAGNOSIS')
            to_drop = [c for c in diag_like if c != "DIAGNOSIS"]
            base = base.drop(columns=to_drop)

        # also drop any *_dup feature columns that may appear (should be rare)
        dup_feats = [c for c in base.columns if c.endswith("_dup")]
        if dup_feats:
            base = base.drop(columns=dup_feats)

    df = base.copy()

    # ---- Clean label and build y ----
    # Drop rows with missing DIAGNOSIS
    df = df.dropna(subset=["DIAGNOSIS"]).copy()

    col = df["DIAGNOSIS"]
    # Try numeric first (e.g., 1/2/3)
    if pd.api.types.is_numeric_dtype(col):
        y = pd.to_numeric(col, errors="coerce").map({1: 0, 2: 1, 3: 2})
    else:
        # Could be strings like "CN"/"MCI"/"AD" or numeric-as-string
        as_num = pd.to_numeric(col, errors="coerce")
        if as_num.notna().any() and as_num.isna().sum() < len(as_num):
            y = as_num.map({1: 0, 2: 1, 3: 2})
        else:
            s = col.astype(str).str.upper().str.strip()
            s = s.replace({"AD DEMENTIA": "AD", "DEMENTIA": "AD"})
            y = s.map({"CN": 0, "MCI": 1, "AD": 2})

    bad = y.isna().sum()
    if bad:
        print(f"[WARN] dropping {bad} rows with unrecognized DIAGNOSIS values")
    df = df.loc[y.notna()].copy()
    df["y"] = y.loc[y.notna()].astype(int)

    # ---- Availability flags per expert ----
    for name, feat in groups.items():
        if len(feat) == 0:
            df[f"has_{name}"] = 0
        else:
            present = [c for c in feat if c in df.columns]
            if not present:
                # no columns from this expert survived the merge
                df[f"has_{name}"] = 0
            else:
                df[f"has_{name}"] = df[present].notna().any(axis=1).astype(int)

    classes = ["CN", "MCI", "AD"]
    return df, groups, classes

def build_dataset(amy_path: Optional[str]=None,
                  tau_path: Optional[str]=None,
                  mri_path: Optional[str]=None):
    dfs, groups, label_cols = [], {}, []

    if amy_path:
        df_amy = read_csv(amy_path)
        amy_feats = pick_cols(df_amy, ("_SUVR",))
        df_amy["_has_amy"] = df_amy[amy_feats].notna().any(axis=1).astype(int)
        dfs.append(df_amy); groups["amy"] = amy_feats; label_cols.append(infer_diag_col(df_amy))

    if tau_path:
        df_tau = read_csv(tau_path)
        tau_feats = pick_cols(df_tau, ("_SUVR",))
        df_tau["_has_tau"] = df_tau[tau_feats].notna().any(axis=1).astype(int)
        dfs.append(df_tau); groups["tau"] = tau_feats; label_cols.append(infer_diag_col(df_tau))

    if mri_path:
        df_mri = read_csv(mri_path)
        mri_feats = pick_cols(df_mri, ("_VOLUME","_VOLUMN"))
        df_mri["_has_mri"] = df_mri[mri_feats].notna().any(axis=1).astype(int)
        dfs.append(df_mri); groups["mri"] = mri_feats; label_cols.append(infer_diag_col(df_mri))

    if not dfs: raise ValueError("Pass at least one of --amy/--tau/--mri.")

    from functools import reduce
    df = dfs[0].copy() if len(dfs)==1 else reduce(safe_merge, dfs)

    # coalesce labels
    label_cols = [c for c in label_cols if c in df.columns]
    if not label_cols: raise ValueError("No diagnosis/label column after merge.")
    df["LABEL"] = df[label_cols].bfill(axis=1).iloc[:,0]
    mapping = {"CN":0,"MCI":1,"AD":2,"Dementia":2,"AD Dementia":2, 1:0, 2:1, 3:2}
    df["y"] = df["LABEL"].map(mapping)
    df = df[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)

    # presence flags (default to 0 if missing)
    for col in ("_has_amy","_has_tau","_has_mri"):
        if col not in df.columns:
            df[col] = 0
    df["has_amy"] = df["_has_amy"].fillna(0).astype(int)
    df["has_tau"] = df["_has_tau"].fillna(0).astype(int)
    df["has_mri"] = df["_has_mri"].fillna(0).astype(int)

    # keep rows with at least one modality
    df = df[(df["has_amy"] + df["has_tau"] + df["has_mri"]) > 0].copy()

    return df, groups, ["CN","MCI","AD"]

def ordered_groups(groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    mods = [m for m in ("amy","tau","mri") if m in groups]
    return {m: groups[m] for m in mods}

# ----------------------------
# Demographic processing
# ----------------------------
def _compute_age_at_visit(df_demo: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AGE_AT_VISIT using PTDOB from demographic data
    and SCANDATE from imaging data.

    Returns a demographic dataframe that preserves all static demographic info
    (PTGENDER, PTEDUCAT, etc.) for every subject, including those without
    SCANDATE, and adds AGE_AT_VISIT where computable.
    """
    df_demo = df_demo.copy()
    if "PTDOB" not in df_demo.columns:
        print("[WARN] DEMOGRAPHIC expert missing PTDOB column; cannot compute age.")
        return df_demo

    # --- Parse birth dates ---
    df_demo["PTDOB_parsed"] = pd.to_datetime(df_demo["PTDOB"], errors="coerce")
    df_demo["PTID"] = df_demo["PTID"].astype(str).str.strip()

    # --- Prepare reference imaging SCANDATEs ---
    ref_df = ref_df.copy()
    ref_df["PTID"] = ref_df["PTID"].astype(str).str.strip()
    if "SCANDATE" in ref_df.columns:
        ref_df["SCANDATE"] = pd.to_datetime(ref_df["SCANDATE"], errors="coerce")
    else:
        ref_df["SCANDATE"] = np.nan
    if "VISCODE" not in ref_df.columns:
        ref_df["VISCODE"] = np.nan
    if "DIAGNOSIS" not in ref_df.columns:
        ref_df["DIAGNOSIS"] = np.nan

    # --- Compute AGE_AT_VISIT for visits with valid SCANDATE ---
    df_age = pd.merge(
        ref_df[["PTID", "SCANDATE", "VISCODE", "DIAGNOSIS"]],
        df_demo,
        on="PTID", how="left"
    )
    df_age["AGE_AT_VISIT"] = (df_age["SCANDATE"] - df_age["PTDOB_parsed"]).dt.days / 365.25

    valid = df_age["AGE_AT_VISIT"].notna().sum()
    print(f"[INFO] Computed AGE_AT_VISIT for {valid:,}/{len(df_age):,} visits ({valid/len(df_age)*100:.1f}% coverage).")

    # drop potential non-feature columns if present
    drop_cols = ["PTDOB", "PTDOB_parsed", "PTID.1"]
    df_age = df_age.drop(columns=[c for c in drop_cols if c in df_age.columns])

    # normalize SCANDATE back to string
    df_age["SCANDATE"] = df_age["SCANDATE"].dt.date.astype(str)

    # Convert categorical race/ethnicity safely to numeric (drop '4|5', etc.)
    for col in ["PTRACCAT", "PTETHCAT"]:
        if col in df_age.columns:
            df_age[col] = pd.to_numeric(df_age[col], errors="coerce")

    # # Handle missing values for MoE
    # # --- Continuous ---
    # df_age["PTEDUCAT"] = df_age["PTEDUCAT"].fillna(df_age["PTEDUCAT"].median())

    # # --- Categorical ---
    # for col in ["PTGENDER", "PTRACCAT", "PTETHCAT"]:
    #     df_age[col] = pd.to_numeric(df_age[col], errors="coerce")
    #     df_age[col] = df_age[col].fillna(-1)   # -1 = Missing category

    return df_age

# ----------------------------
# Metrics
# ----------------------------
def macro_auroc(y_true, proba, n_classes=3):
    """
    One vs rest macro AUROC for multi-class.
    """
    scores=[]
    for c in range(n_classes):
        yb = (y_true==c).astype(int)
        try: scores.append(roc_auc_score(yb, proba[:,c]))
        except ValueError: pass
    # If no valid per-class ROC AUCs could be computed (e.g., validation fold
    # contains only one class), return a sensible default (0.5 = random)
    # instead of NaN so downstream early-stopping and Optuna objectives
    # receive a numeric value.
    if not scores:
        return 0.5
    mean = np.mean(scores)
    if np.isnan(mean):
        return 0.5
    return float(mean)

def stratified_macro_auroc(df, val_idx, proba, y_true, groups):
    out = {}
    for name in groups.keys():
        mask = df.iloc[val_idx][f"has_{name}"] == 1
        idx = np.where(mask.values)[0]
        if len(idx) == 0:
            continue
        out[f"{name}_only"] = macro_auroc(y_true[idx], proba[idx], 3)
    return out

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def balanced_acc(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def regular_acc(y_true, y_pred):
    """Overall accuracy (proportion of correct predictions)."""
    return accuracy_score(y_true, y_pred)

def eval_multiclass_metrics(y_true, proba):
    # Defensive handling: ensure `proba` is a 2D (N, C) array. If not, try to
    # coerce or recover sensible defaults and log shapes for debugging.
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    try:
        print(f"[DEBUG] eval_multiclass_metrics: y_true.shape={y_true.shape}, proba.shape={proba.shape}")
    except Exception:
        pass

    auc = 0.5
    y_pred = None

    if proba.ndim == 2:
        # shapes should align: (N, C)
        if proba.shape[0] != y_true.shape[0]:
            # try a common transpose/reshape fix if plausible
            if proba.shape[1] == y_true.shape[0] and proba.shape[0] in (1, 3):
                proba = proba.T
                print(f"[WARN] transposed proba to match y_true: new proba.shape={proba.shape}")
            else:
                print(f"[WARN] proba.shape {proba.shape} does not match y_true.shape {y_true.shape}")
        try:
            y_pred = np.argmax(proba, axis=1)
            auc = macro_auroc(y_true, proba, proba.shape[1])
        except Exception as e:
            print(f"[WARN] could not compute argmax/auc on proba: {e}")
            # fallback: try argmax defensively
            try:
                y_pred = np.argmax(np.atleast_2d(proba), axis=1)
            except Exception:
                y_pred = np.zeros_like(y_true)

    elif proba.ndim == 1:
        # proba is 1D: this may be a flattened (N*C,) vector, or already
        # integer labels. Try to detect the case.
        if np.issubdtype(proba.dtype, np.integer):
            # treat as predicted labels
            print(f"[WARN] proba is 1D integer array; treating as predicted labels")
            y_pred = proba.astype(int)
            auc = 0.5
        else:
            # If size matches N*C (common case when a list of per-batch arrays
            # accidentally got flattened), attempt to reshape to (N,3).
            n = y_true.shape[0] if y_true.ndim >= 1 else None
            if n and proba.size == n * 3:
                try:
                    proba = proba.reshape(n, 3)
                    y_pred = np.argmax(proba, axis=1)
                    auc = macro_auroc(y_true, proba, 3)
                except Exception as e:
                    print(f"[WARN] failed to reshape flat proba into (N,3): {e}")
                    y_pred = np.zeros_like(y_true)
            else:
                print(f"[WARN] proba is 1D with shape {proba.shape}; cannot reshape to (N, C). Treating as labels")
                try:
                    y_pred = proba.astype(int)
                except Exception:
                    y_pred = np.zeros_like(y_true)

    else:
        print(f"[WARN] proba has unexpected ndim={getattr(proba, 'ndim', None)}; returning defaults")
        y_pred = np.zeros_like(y_true)
        auc = 0.5

    # compute metrics from y_pred (ensure arrays)
    try:
        y_pred = np.asarray(y_pred)
        bacc = float(balanced_acc(y_true, y_pred))
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro"))
    except Exception as e:
        print(f"[WARN] could not compute classification metrics: {e}")
        bacc, acc, f1 = 0.0, 0.0, 0.0

    try:
        auc = float(auc)
    except Exception:
        auc = 0.5

    return {"bacc": bacc, "auc": auc, "acc": acc, "f1": f1}


def confusion_matrix_from_proba(y_true: np.ndarray, proba: np.ndarray, labels=None):
    """Return (cm_counts, cm_row_norm, y_pred).

    cm_counts: integer confusion matrix with shape (C, C)
    cm_row_norm: float confusion matrix where rows sum to 1 (safe for empty rows)
    y_pred: argmax predictions
    """
    if labels is None:
        labels = [0, 1, 2]
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = np.argmax(proba, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype(int)

    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0.0] = 1.0
    cm_row = cm / row_sums

    return cm, cm_row, y_pred


def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray, labels=None):
    """Per-class precision/recall/F1 (no averaging)."""
    if labels is None:
        labels = [0, 1, 2]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1   = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "precision": prec.astype(float),
        "recall": rec.astype(float),
        "f1": f1.astype(float),
    }


def eval_confusion_report(y_true: np.ndarray, proba: np.ndarray, class_names=None):
    """Convenience wrapper for IEEE-style confusion matrix reporting.

    Returns a dict containing:
      - cm_counts: list[list[int]]
      - cm_row_norm: list[list[float]] rows sum to 1
      - per_class: {precision, recall, f1} as name->value maps
      - y_pred: list[int]

    `class_names` controls the printed ordering; defaults to CN/MCI/AD.
    """
    if class_names is None:
        class_names = ["CN", "MCI", "AD"]
    labels = list(range(len(class_names)))

    cm_counts, cm_row, y_pred = confusion_matrix_from_proba(y_true, proba, labels=labels)
    prf = per_class_prf(y_true, y_pred, labels=labels)

    per_class = {
        "precision": {class_names[i]: float(prf["precision"][i]) for i in range(len(class_names))},
        "recall": {class_names[i]: float(prf["recall"][i]) for i in range(len(class_names))},
        "f1": {class_names[i]: float(prf["f1"][i]) for i in range(len(class_names))},
    }

    return {
        "cm_counts": cm_counts.tolist(),
        "cm_row_norm": cm_row.tolist(),
        "per_class": per_class,
        "y_pred": y_pred.tolist(),
    }

# ----------------------------
# Split loaders
# ----------------------------
def load_splits(path: str, df: pd.DataFrame):
    """Supports PTID-based JSON (train_ptids/test_ptids) or position-based JSON."""
    with open(path, "r") as f:
        splits = json.load(f)

    df_ptid = df["PTID"].astype(str).str.strip()

    # PTID-based 3-way splits (preferred for clarity)
    if "train_ptids" in splits and "val_ptids" in splits and "test_ptids" in splits:
        tr_ptids = [str(p).strip() for p in splits["train_ptids"]]
        va_ptids = [str(p).strip() for p in splits["val_ptids"]]
        te_ptids = [str(p).strip() for p in splits["test_ptids"]]
        tr = np.where(df_ptid.isin(tr_ptids).to_numpy())[0]
        va = np.where(df_ptid.isin(va_ptids).to_numpy())[0]
        te = np.where(df_ptid.isin(te_ptids).to_numpy())[0]

        leak = set(df_ptid.iloc[tr]) & set(df_ptid.iloc[va])
        leak = leak | (set(df_ptid.iloc[tr]) & set(df_ptid.iloc[te]))
        leak = leak | (set(df_ptid.iloc[va]) & set(df_ptid.iloc[te]))
        if leak:
            raise ValueError(f"PTID leakage detected across train/val/test: {len(leak)}")
        return tr, va, te

    # PTID-based 2-way splits (backwards compatibility)
    if "train_ptids" in splits and "test_ptids" in splits:
        tr_ptids = [str(p).strip() for p in splits["train_ptids"]]
        te_ptids = [str(p).strip() for p in splits["test_ptids"]]
        tr = np.where(df_ptid.isin(tr_ptids).to_numpy())[0]
        va = np.where(df_ptid.isin(te_ptids).to_numpy())[0]

        leak = set(df_ptid.iloc[tr]) & set(df_ptid.iloc[va])
        if leak:
            raise ValueError(f"PTID leakage detected: {len(leak)}")
        return tr, va

    # fallback: indices (must match current df build)
    tr = np.asarray(splits.get("train_pool_indices", []), dtype=int)
    va = np.asarray(splits.get("test_indices", []), dtype=int)
    if tr.size == 0 or va.size == 0:
        raise ValueError("Splits JSON missing keys.")
    if tr.min()<0 or tr.max()>=len(df) or va.min()<0 or va.max()>=len(df):
        raise IndexError("Split indices out-of-bounds for this dataframe.")
    leak = set(df_ptid.iloc[tr]) & set(df_ptid.iloc[va])
    if leak:
        raise ValueError(f"PTID leakage detected: {len(leak)}")
    return tr, va

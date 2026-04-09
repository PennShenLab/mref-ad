#!/usr/bin/env python3
"""
Make fixed train/val/test splits by PTID (subject ID), stratified by label (`y`).
Writes one JSON per run with `train_ptids`, `val_ptids`, and `test_ptids`.

Default proportions are 80% / 10% / 10% (`--test_size 0.1 --val_size 0.1`). The script
appends `_seed_<seed>` to the output basename unless it already ends with `_seed_<digits>`.

From repository root:

  python data_preprocessing/make_splits.py \\
    --paths configs/paths.yaml \\
    --out configs/splits_by_ptid_80_10_10.json \\
    --test_size 0.1 \\
    --val_size 0.1 \\
    --seed 7

Repeat for each split RNG seed (e.g. 7, 13, 42, 1234, 2027, 99, 123, 555, 999, 1337).

Last visit per subject before splitting:

  python data_preprocessing/make_splits.py \\
    --paths configs/paths.yaml \\
    --out configs/splits_by_ptid_80_10_10.json \\
    --test_size 0.1 --val_size 0.1 \\
    --last_visit_only \\
    --seed 7
"""
import json
import os
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from utils import build_dataset

DEFAULT_SEED = 42


def _resolve_paths(paths_yaml: str):
    with open(paths_yaml, "r") as f:
        raw = yaml.safe_load(f) or {}

    def expand_env_directives(s: str) -> str:
        def repl(m):
            var = m.group(1)
            default = m.group(2)
            return os.environ.get(var, default if default is not None else "")

        return re.sub(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?:,\s*([^}]+))?\}", repl, s)

    def norm(p: str) -> str:
        p = os.path.expanduser(os.path.expandvars(p))
        p = p.replace("\\ ", " ")
        return os.path.normpath(p)

    data_dir = raw.get("DATA_DIR")
    if isinstance(data_dir, str):
        data_dir = expand_env_directives(data_dir)
        data_dir = norm(data_dir)

    out_dir = raw.get("OUT_DIR")
    if isinstance(out_dir, str):
        out_dir = expand_env_directives(out_dir)
        out_dir = norm(out_dir)

    data_dir = data_dir or os.environ.get("AD_DATA_DIR", "")
    out_dir = out_dir or os.environ.get("AD_OUT_DIR", os.path.join(os.getcwd(), "results"))

    def resolve_field(key):
        v = raw.get(key)
        if not isinstance(v, str):
            return None
        v = expand_env_directives(v)
        v = v.replace("${DATA_DIR}", data_dir).replace("${OUT_DIR}", out_dir)
        v = norm(v)
        return v

    resolved = {
        "amy": resolve_field("AMY_CSV"),
        "tau": resolve_field("TAU_CSV"),
        "mri": resolve_field("MRI_CSV"),
        "out_dir": out_dir if out_dir else os.path.join(os.getcwd(), "results"),
    }

    print("[INFO] resolved paths:")
    for k in ("amy", "tau", "mri", "out_dir"):
        print(f"  - {k}: {resolved[k]}")

    for k in ("amy", "tau", "mri"):
        p = resolved[k]
        if p and not os.path.exists(p):
            raise FileNotFoundError(f"{k} file not found: {p}")

    os.makedirs(resolved["out_dir"], exist_ok=True)
    return resolved


def make_holdout_splits_by_ptid(
    df: pd.DataFrame,
    *,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = DEFAULT_SEED,
):
    """Stratified by label, grouped by PTID: test holdout, then val from remainder."""
    df = df.copy()
    df["PTID"] = df["PTID"].astype(str).str.strip()
    df = df[df["PTID"].notna() & (df["PTID"] != "")]

    y = df["y"].values
    groups = df["PTID"].values

    if test_size <= 0 or test_size >= 0.5:
        raise ValueError("test_size must be in (0, 0.5) for a sensible holdout")
    if val_size < 0 or val_size >= 0.5:
        raise ValueError("val_size must be in [0, 0.5) for a sensible holdout")

    n_splits_test = max(2, int(round(1.0 / test_size)))
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf_test.split(np.zeros(len(df)), y, groups))

    if float(val_size) == 0.0:
        train_idx = trainval_idx
        train_ptids = sorted(pd.unique(df["PTID"].iloc[train_idx]).tolist())
        val_ptids = []
        test_ptids = sorted(pd.unique(df["PTID"].iloc[test_idx]).tolist())
    else:
        trainval_mask = np.zeros(len(df), dtype=bool)
        trainval_mask[trainval_idx] = True
        y_trainval = y[trainval_mask]
        groups_trainval = groups[trainval_mask]

        val_rel = float(val_size) / (1.0 - float(test_size))
        if not (0.0 < val_rel < 0.5):
            val_rel = 0.111111

        n_splits_val = max(2, int(round(1.0 / val_rel)))
        sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
        train_idx_rel, val_idx_rel = next(
            sgkf_val.split(np.zeros(len(y_trainval)), y_trainval, groups_trainval)
        )

        trainval_indices = np.nonzero(trainval_mask)[0]
        val_idx = trainval_indices[val_idx_rel]
        train_idx = trainval_indices[train_idx_rel]

        train_ptids = sorted(pd.unique(df["PTID"].iloc[train_idx]).tolist())
        val_ptids = sorted(pd.unique(df["PTID"].iloc[val_idx]).tolist())
        test_ptids = sorted(pd.unique(df["PTID"].iloc[test_idx]).tolist())

    return {
        "train_ptids": train_ptids,
        "val_ptids": val_ptids,
        "test_ptids": test_ptids,
        "seed": int(seed),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Write stratified PTID train/val/test splits JSON (e.g. 80/10/10 per RNG seed)."
    )
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--out", default="configs/splits_by_ptid.json")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="RNG seed for StratifiedGroupKFold; also stored in JSON and appended to output path",
    )
    ap.add_argument("--last_visit_only", action="store_true")
    args = ap.parse_args()

    paths = _resolve_paths(args.paths)

    df, _groups, _ = build_dataset(
        amy_path=paths["amy"],
        tau_path=None,
        mri_path=paths["mri"],
    )

    raw_amy = pd.read_csv(paths["amy"])
    raw_ptids = set(raw_amy["PTID"].astype(str).str.strip().unique())
    df_ptids = set(df["PTID"].astype(str).str.strip().unique())
    missing = sorted(list(raw_ptids - df_ptids))
    print(f"[DEBUG] Raw AMY rows: {len(raw_amy)}, after build_dataset: {len(df)}")
    if missing:
        print(f"[DEBUG] PTIDs missing after build_dataset: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    else:
        print("[DEBUG] No PTIDs missing after build_dataset.")

    if args.last_visit_only:
        if "SCANDATE" in df.columns and df["SCANDATE"].notna().all():
            df_sorted = df.sort_values(["PTID", "SCANDATE"])
            df_last = df_sorted.groupby("PTID").tail(1)
        elif "VISCODE" in df.columns and df["VISCODE"].notna().all():
            df_sorted = df.sort_values(["PTID", "VISCODE"])
            df_last = df_sorted.groupby("PTID").tail(1)
        else:
            df_last = df.groupby("PTID").tail(1)
        print(f"[INFO] filtered to last visit only: {len(df_last)} rows (from {len(df)})")
        df = df_last.reset_index(drop=True)

        base, ext = os.path.splitext(args.out)
        if "_lastvisit" not in base:
            args.out = base + "_lastvisit" + ext if ext == ".json" else args.out + "_lastvisit"

    splits = make_holdout_splits_by_ptid(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    all_ptids = set(df["PTID"].astype(str).str.strip().unique())
    holdout_union = set(splits["train_ptids"]) | set(splits["val_ptids"]) | set(splits["test_ptids"])
    print(f"[ASSERT] Holdout split covers {len(holdout_union)} / {len(all_ptids)} PTIDs")
    assert holdout_union == all_ptids, "Holdout train+val+test do not cover all PTIDs"

    base, ext = os.path.splitext(args.out)
    if not re.search(r"_seed_\d+$", base):
        args.out = f"{base}_seed_{int(args.seed)}{ext}"

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[INFO] wrote {args.out}")

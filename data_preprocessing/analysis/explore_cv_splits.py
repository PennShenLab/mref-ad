#!/usr/bin/env python3
"""
explore_cv_splits.py

Quick utility to check diagnosis group composition across train/val folds
for a given CV splits JSON (e.g. 5-fold or 10-fold) and dataset CSV.

Example:
python analysis/explore_cv_splits.py \
  --splits configs/splits_by_ptid_lastvisit_cv10.json \
  --data data/freesurfer_lastvisit_cv10/250826_DX_AMYLOID_last_visit.csv \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml
"""

import json
import argparse
import pandas as pd
import yaml
import os

def main():
    parser = argparse.ArgumentParser(description="Check diagnosis composition across CV splits.")
    parser.add_argument("--splits", required=True, help="Path to CV splits JSON file.")
    parser.add_argument("--data", required=True, help="Path to data CSV file containing PTID and DIAGNOSIS.")
    parser.add_argument("--experts_config", required=False, help="Path to YAML file with expert CSV paths for NaN diagnostics.")
    parser.add_argument("--out", required=False, default="explore_cv_splits_output.txt", help="Output file path for logs.")
    args = parser.parse_args()

    # Open output file in write mode
    with open(args.out, "w") as outfile:

        def log(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=outfile)

        # Load data and ensure correct column names
        df = pd.read_csv(args.data)
        if "PTID" not in df.columns:
            raise ValueError("Column 'PTID' not found in the provided data CSV.")
        if "DIAGNOSIS" not in df.columns:
            raise ValueError("Column 'DIAGNOSIS' not found in the provided data CSV.")
        df["PTID"] = df["PTID"].astype(str)

        # Load splits
        with open(args.splits, "r") as f:
            splits = json.load(f)

        if "cv_splits_ptid" not in splits:
            raise ValueError("Splits JSON must contain 'cv_splits_ptid'.")

        cv_splits = splits["cv_splits_ptid"]

        log(f"[INFO] Loaded {len(cv_splits)} folds from {args.splits}")
        log(f"[INFO] Dataset: {len(df)} rows, {df['PTID'].nunique()} unique PTIDs")
        log()

        # Function to summarize class composition
        def summarize(ids, label):
            subset = df[df["PTID"].isin(ids)]
            counts = subset["DIAGNOSIS"].value_counts().sort_index()
            total = len(subset)
            summary = ", ".join([f"{k}: {v} ({v/total*100:.1f}%)" for k, v in counts.items()])
            log(f"  {label}  n={total}  ->  {summary}")

        # Loop over folds
        for i, fold in enumerate(cv_splits):
            log(f"Fold {i+1}")
            train_keys = [k for k in fold.keys() if "train" in k.lower()]
            val_keys = [k for k in fold.keys() if "val" in k.lower()]
            if not train_keys or not val_keys:
                raise KeyError(f"Fold {i}: missing train/val keys ({list(fold.keys())})")
            train_ids = set(fold[train_keys[0]])
            val_ids = set(fold[val_keys[0]])
            summarize(train_ids, "Train")
            summarize(val_ids, "Val")
            log("-" * 60)

        # If experts_config is provided, do NaN diagnostics per expert
        if args.experts_config is not None:
            with open(args.experts_config, "r") as f:
                experts_cfg = yaml.safe_load(f)
            # Handle nested YAML configs with 'experts' key
            if isinstance(experts_cfg, dict) and "experts" in experts_cfg:
                experts_cfg = experts_cfg["experts"]
            if not isinstance(experts_cfg, dict):
                raise ValueError("Experts config YAML must be a dictionary of expert names to CSV paths.")

            log("\n[INFO] Starting all-NaN diagnostics per expert and fold...\n")

            for expert_name, expert_path in experts_cfg.items():
                if not os.path.isfile(expert_path):
                    log(f"[WARNING] Expert CSV file not found: {expert_path} (Skipping)")
                    continue
                expert_df = pd.read_csv(expert_path)
                if "PTID" not in expert_df.columns:
                    log(f"[WARNING] Expert CSV {expert_path} missing 'PTID' column (Skipping)")
                    continue
                expert_df["PTID"] = expert_df["PTID"].astype(str)

                # Columns to check (exclude PTID)
                cols = [c for c in expert_df.columns if c != "PTID"]

                # Overall all-NaN columns
                overall_all_nan = [col for col in cols if expert_df[col].isna().all()]
                log(f"[Expert: {expert_name}] Overall all-NaN columns ({len(overall_all_nan)}): {overall_all_nan}")

                # Per fold diagnostics
                for i, fold in enumerate(cv_splits):
                    train_keys = [k for k in fold.keys() if "train" in k.lower()]
                    val_keys = [k for k in fold.keys() if "val" in k.lower()]
                    if not train_keys or not val_keys:
                        continue
                    train_ids = set(fold[train_keys[0]])
                    val_ids = set(fold[val_keys[0]])

                    train_df = expert_df[expert_df["PTID"].isin(train_ids)]
                    val_df = expert_df[expert_df["PTID"].isin(val_ids)]

                    train_all_nan = [col for col in cols if train_df[col].isna().all()]
                    val_all_nan = [col for col in cols if val_df[col].isna().all()]

                    log(f"Fold {i+1} - Expert '{expert_name}':")
                    log(f"  Train all-NaN columns ({len(train_all_nan)}): {train_all_nan}")
                    log(f"  Val   all-NaN columns ({len(val_all_nan)}): {val_all_nan}")
                log("-" * 60)

if __name__ == "__main__":
    main()

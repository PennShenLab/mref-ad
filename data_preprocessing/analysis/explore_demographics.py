#!/usr/bin/env python3
"""
Explore ADNI demographic expert file and its overlap with imaging data.

Usage:
    python analysis/explore_demographics.py \
        --demo data/251020_DX_DEMOGRAPHIC_multi_visit.csv \
        --imaging data/250826_DX_AMYLOID_amy_ctx_multi_visit.csv
"""

import pandas as pd
import argparse
import numpy as np


def summarize_demographics(df):
    """Compute key descriptive statistics and distributions for ADNI demographics."""

    print("\n=== Basic Info ===")
    print(f"Rows: {len(df):,}")
    print(f"Unique participants: {df['PTID'].nunique():,}\n")

    # ---- Education ----
    if "PTEDUCAT" in df.columns:
        print("Education (years):")
        print(df["PTEDUCAT"].describe().to_string(), "\n")

    # ---- Gender distribution ----
    if "PTGENDER" in df.columns:
        print("Gender distribution:")
        print(df["PTGENDER"].value_counts(dropna=False).to_string(), "\n")

    # ---- Race ----
    if "PTRACCAT" in df.columns:
        print("Race distribution:")
        print(df["PTRACCAT"].value_counts(dropna=False).to_string(), "\n")

    # ---- Ethnicity ----
    if "PTETHCAT" in df.columns:
        print("Ethnicity distribution:")
        print(df["PTETHCAT"].value_counts(dropna=False).to_string(), "\n")

    # ---- Missingness summary ----
    print("\n=== Missing Data Summary (percent missing) ===")
    missing = df.isna().mean() * 100
    print(missing.sort_values(ascending=False).round(2).to_string())

    print("\n[INFO] Exploration complete. Use this summary to decide which demographic columns to include.")
    return df


def check_imaging_overlap(df_demo, imaging_path):
    """
    Check how many imaging visits have matching participant-level demographics (by PTID only).
    Also compute AGE_AT_VISIT dynamically from imaging SCANDATE and PTDOB.
    """
    df_img = pd.read_csv(imaging_path)
    print(f"\n[INFO] Loaded imaging file: {imaging_path}")
    print(f"Rows in imaging file: {len(df_img):,}")

    # Normalize key columns
    for d in [df_img, df_demo]:
        d.columns = d.columns.str.upper().str.strip()
        d["PTID"] = d["PTID"].astype(str).str.strip()

    # Determine overlap by PTID
    img_ptids = set(df_img["PTID"].unique())
    demo_ptids = set(df_demo["PTID"].unique())

    matched_ptids = img_ptids & demo_ptids
    missing_ptids = img_ptids - demo_ptids

    total_visits = len(df_img)
    matched_visits = df_img["PTID"].isin(matched_ptids).sum()
    missing_visits = df_img["PTID"].isin(missing_ptids).sum()

    print("\n=== Imaging–Demographic Overlap (by PTID only) ===")
    print(f"Total imaging visits: {total_visits:,}")
    print(f"Matched visits (PTID found in demographics): {matched_visits:,}")
    print(f"Missing visits (PTID not in demographics): {missing_visits:,}")
    print(f"Coverage: {matched_visits / total_visits * 100:.2f}%")
    print(f"Unique imaging PTIDs: {len(img_ptids):,}")
    print(f"Unique matched PTIDs: {len(matched_ptids):,}")
    print(f"Unique missing PTIDs: {len(missing_ptids):,}\n")

    if len(missing_ptids) > 0:
        print("Example missing PTIDs:")
        print(pd.Series(sorted(list(missing_ptids))).head(10).to_string(index=False))

    # --- Compute AGE_AT_VISIT dynamically if dates exist ---
    if "SCANDATE" in df_img.columns and "PTDOB" in df_demo.columns:
        df_demo["PTDOB_parsed"] = pd.to_datetime(df_demo["PTDOB"], errors="coerce")
        df_img["SCANDATE"] = pd.to_datetime(df_img["SCANDATE"], errors="coerce")

        df_img = df_img.merge(
            df_demo[["PTID", "PTDOB_parsed"]],
            on="PTID",
            how="left"
        )

        df_img["AGE_AT_VISIT"] = (df_img["SCANDATE"] - df_img["PTDOB_parsed"]).dt.days / 365.25
        df_img.loc[(df_img["AGE_AT_VISIT"] < 40) | (df_img["AGE_AT_VISIT"] > 110), "AGE_AT_VISIT"] = np.nan

        valid_age = df_img["AGE_AT_VISIT"].notna().sum()
        print(f"[INFO] Computed AGE_AT_VISIT for {valid_age:,}/{len(df_img):,} imaging visits "
              f"({valid_age / len(df_img) * 100:.1f}% coverage).")

    return df_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", required=True, help="Path to demographic expert CSV (e.g., 251020_DX_DEMOGRAPHIC_multi_visit.csv)")
    parser.add_argument("--imaging", help="Optional: Path to imaging file to check overlap (e.g., amy_ctx CSV)")
    args = parser.parse_args()

    df_demo = pd.read_csv(args.demo)
    print(f"[INFO] Loaded demographics from {args.demo}")
    df_demo.columns = df_demo.columns.str.strip()

    summarize_demographics(df_demo)

    if args.imaging:
        check_imaging_overlap(df_demo, args.imaging)


if __name__ == "__main__":
    main()
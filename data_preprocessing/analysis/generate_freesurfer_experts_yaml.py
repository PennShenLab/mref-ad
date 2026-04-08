#!/usr/bin/env python3
"""
generate_freesurfer_experts_yaml.py

This script automatically generates an experts configuration YAML file from the
Freesurfer–dataset column mapping produced by map_freesurfer_brain_regions_to_data.py.

It creates one expert group per anatomical region for both amyloid PET (SUVR) and MRI (VOLUME)
modalities, matching the style of the manually written experts.yaml.

Usage:
    python analysis/generate_freesurfer_experts_yaml.py \
        --mapping data/freesurfer_to_column_mapping.csv \
        --output configs/freesurfer_experts.yaml
"""

import argparse
import pandas as pd
import yaml
import re

def sanitize_region_name(region: str) -> str:
    region = region.lower()
    region = re.sub(r"[^a-z0-9]+", "_", region)
    region = re.sub(r"_+", "_", region).strip("_")
    return region

def main():
    parser = argparse.ArgumentParser(description="Generate freesurfer_experts.yaml from mapping CSV")
    parser.add_argument("--mapping", required=True, help="Path to Freesurfer-to-columns mapping CSV")
    parser.add_argument("--output", required=True, help="Path to output YAML file")
    args = parser.parse_args()

    df = pd.read_csv(args.mapping)
    if "Region" not in df.columns or "Columns" not in df.columns:
        raise ValueError("Mapping CSV must contain columns: Region, Columns")

    yaml_dict = {"version": 1, "groups": {}}

    # Group by anatomical region to gather all amyloid SUVR and MRI VOLUME columns across all measures
    for region in df["Region"].unique():
        region_clean = sanitize_region_name(region)
        amy_cols = []
        mri_cols = []

        # Select all rows belonging to this region
        subdf = df[df["Region"] == region]

        # Collect all columns for amyloid SUVR and MRI VOLUME modalities across measures
        for _, row in subdf.iterrows():
            if pd.isna(row["Columns"]):
                continue
            cols = [c.strip() for c in str(row["Columns"]).split(",") if c.strip()]
            for col in cols:
                if col.endswith("_SUVR"):
                    amy_cols.append(col)
                elif col.endswith("_VOLUME"):
                    mri_cols.append(col)

        if amy_cols:
            yaml_dict["groups"][f"amy_{region_clean}"] = {
                "include_regex": [f"^{c}$" for c in amy_cols],
                "exclude_regex": []
            }

        if mri_cols:
            yaml_dict["groups"][f"mri_{region_clean}"] = {
                "include_regex": [f"^{c}$" for c in mri_cols],
                "exclude_regex": []
            }

    # Add demographic expert group
    yaml_dict["groups"]["demographic"] = {
        "include_regex": [
            "PTID",
            "PTDOB",
            "PTGENDER",
            "PTEDUCAT",
            "PTRACCAT",
            "PTETHCAT"
        ],
        "exclude_regex": []
    }

    with open(args.output, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"Generated {len(yaml_dict['groups'])} expert groups → {args.output}")

if __name__ == "__main__":
    main()
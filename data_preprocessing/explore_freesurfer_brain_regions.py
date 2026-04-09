"""
explore_freesurfer_brain_regions.py
-----------------------------------
This script parses, cleans, and explores the "Freesurfer_sorted" sheet from the
FS-VBM-keys_SLRedit.xlsx file. It extracts both the *Volumes* and *Cortical
Measures (Thickness)* sections, merges them into a single standardized table,
and filters out noisy or excluded measures based on the "Exclude?" column.

The resulting dataset can be used to define regional groups (experts) for
downstream multimodal imaging analysis (e.g., MoE-AD).

Main steps performed:
1. Read and locate the 'Volumes' and 'Cortical Measures (Thickness)' sections.
2. Extract each section, normalize headers, and forward-fill missing region labels.
3. Combine the two sections into one DataFrame with consistent column names.
4. Filter out measures where "Exclude?" is not "no".
5. Save the cleaned table as `data/freesurfer_brain_regions.csv`.

Outputs:
- **data/freesurfer_brain_regions.csv** — cleaned and merged Freesurfer measures.

Example (from repository root):
    $ python data_preprocessing/explore_freesurfer_brain_regions.py

Example output:
    Raw sheet shape: (81, 5)
    Found 'Volumes' at row 0, 'Cortical Measures' at row 40
    Volumes table shape: (30, 6)
    Cortical table shape: (34, 6)
    Combined Freesurfer table shape: (64, 6)
    Kept 53 Freesurfer measures where Exclude? == 'no' (out of 64 total).
    Saved combined Freesurfer table → 'data/freesurfer_brain_regions.csv'

"""

import os
import sys

import pandas as pd
import re

# --------------------------
# 1. Load the full Freesurfer sheet
# --------------------------
fs_path = os.environ.get("MREF_FS_VBM_XLSX", "data/FS-VBM-keys_SLRedit.xlsx")
sheet_name = "Freesurfer_sorted"

if not os.path.isfile(fs_path):
    print(
        f"[ERROR] Missing FreeSurfer VBM keys spreadsheet: {fs_path}\n"
        "Set MREF_FS_VBM_XLSX or add FS-VBM-keys_SLRedit.xlsx under data/. "
        "If you already have data/freesurfer_brain_regions.csv, you can skip this script.",
        file=sys.stderr,
    )
    sys.exit(2)

raw = pd.read_excel(fs_path, sheet_name=sheet_name, header=None)
print("Raw sheet shape:", raw.shape)

# --------------------------
# 2. Find the start rows for the two tables
# --------------------------
volume_start = raw.index[raw.iloc[:,0].astype(str).str.contains("Volumes", case=False, na=False)][0]
cortical_start = raw.index[raw.iloc[:,0].astype(str).str.contains("Cortical", case=False, na=False)][0]

print(f"Found 'Volumes' at row {volume_start}, 'Cortical Measures' at row {cortical_start}")

# --------------------------
# 3. Extract each table and promote its header row
# --------------------------
vol_df = pd.read_excel(fs_path, sheet_name=sheet_name, header=volume_start + 1,
                       nrows=cortical_start - volume_start - 2)
vol_df = vol_df.dropna(how="all").reset_index(drop=True)

cor_df = pd.read_excel(fs_path, sheet_name=sheet_name, header=None)
# assign header similar to vol_df
header_row = cortical_start + 1
cort_header = raw.iloc[header_row].tolist()
cor_df.columns = cort_header
# get data rows after header (skip blank row if present)
cor_df = cor_df[header_row:].dropna(how="all").reset_index(drop=True)
cor_df.columns = vol_df.columns  # ensure both have consistent columns
cort_df = cor_df

print("vol_df:\n", vol_df.head())
print("cord_df:\n", cort_df.head())

# Add a column indicating the type
vol_df["MeasureType"] = "Volume"
cort_df["MeasureType"] = "Cortical"

print(f"Volumes table shape: {vol_df.shape}")
print(f"Cortical table shape: {cort_df.shape}")

# --------------------------
# 4. Standardize column names and combine
# --------------------------
for df in [vol_df, cort_df]:
    df.columns = df.columns.str.strip().str.replace(" ", "_")

fs_df = pd.concat([vol_df, cort_df], ignore_index=True)
# Drop completely empty columns and columns whose names start with 'Unnamed'
fs_df = fs_df.loc[:, ~fs_df.columns.astype(str).str.contains("^Unnamed", na=False)]
fs_df = fs_df.dropna(axis=1, how="all")
print(f"Combined Freesurfer table shape: {fs_df.shape}")

# --------------------------
# 5. Identify key columns
# --------------------------
fs_df.columns = fs_df.columns.fillna("").astype(str)  # ensure all headers are strings

measure_col = [c for c in fs_df.columns if "Measure" in c][0]
region_col = [c for c in fs_df.columns if "Region" in c][0]
exclude_col = [c for c in fs_df.columns if "Exclude" in c or "No" in c][0]

fs_df[region_col] = fs_df[region_col].ffill()

# --------------------------
# 6. Filter out excluded (anything not 'no')
# --------------------------

valid_fs = fs_df[fs_df[exclude_col].astype(str).str.lower() == "no"]
valid_fs = valid_fs[valid_fs[measure_col].notna()]
print(f"Kept {len(valid_fs)} Freesurfer measures where Exclude? == 'no' (out of {len(fs_df)} total).")

print("valid_fs:\n", valid_fs.tail())

excluded_fs = fs_df[
    (fs_df[exclude_col].astype(str).str.lower() != "no")
    & fs_df[measure_col].notna()
]
print("\nExcluded measures (non-empty):")
print(excluded_fs[[region_col, measure_col, exclude_col, "MeasureType"]])
# --------------------------
# 7. Quick sanity check of counts
# --------------------------
print("\nExclude? value counts:")
print(fs_df[exclude_col].value_counts(dropna=False))

# --------------------------
# 8. Save combined outputs
# --------------------------
valid_fs.to_csv("data/freesurfer_brain_regions.csv", index=False)

print("\nSaved combined Freesurfer table → 'freesurfer_brain_regions.csv'")

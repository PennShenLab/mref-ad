#!/usr/bin/env python3
"""
Summarize subject- and visit-level diagnosis distributions for ADNI multimodal datasets.

This script computes descriptive statistics for modality-specific datasets (e.g., Amyloid–MRI and Tau–MRI)
and identifies their intersection to determine the triply aligned subset (Amyloid + Tau + MRI). 
It outputs counts of unique participants, total imaging visits, and baseline-only samples per diagnosis.

Typical usage:
--------------
python analysis/explore_subjects.py 

Outputs:
--------
1. Summary for each dataset, including:
   - Number of unique participants (PTID)
   - Number of total imaging visits (PTID + VISCODE)
   - Baseline-only composition (VISCODE == 'bl')
   - Diagnosis distribution across CN, MCI, AD

2. Overlap statistics:
   - Number of overlapping visits (Amyloid ∩ Tau, matched by PTID + VISCODE)
   - Number of unique subjects with data in both datasets

These outputs correspond to:
   - Amyloid–MRI dataset  →  merged Amyloid PET + MRI data
   - Tau–MRI dataset      →  merged Tau PET + MRI data
   - Overlap subset       →  participants with all three modalities (Amyloid + Tau + MRI)

Purpose:
--------
This script supports data characterization in the IEEE ISBI MoE-AD paper,
providing the numbers reported in the Methods section (“Data”) and Table 1.

Example output:
---------------
=== Amyloid–MRI Summary ===
Unique participants per diagnosis:
AD     231
CN     655
MCI    644
Total unique participants: 1530

Visit-level samples per diagnosis:
AD      429
CN     1499
MCI    1214
Total visits: 3142

Baseline-only composition:
AD      61
CN     322
MCI    286
Total baseline subjects: 669

=== Tau–MRI Summary ===
Unique participants per diagnosis:
AD     100
CN     478
MCI    292
Total unique participants: 870
...
Overlapping visits (same subject + visit): 1130
Unique subjects with overlapping visits: 809

Author:
--------
Farica Zhuang (University of Pennsylvania)
Date: October 2025
"""

import pandas as pd

def summarize_diagnosis(file_path, label="Dataset"):
    """
    Summarize diagnosis counts (CN, MCI, AD) for a given ADNI dataset.
    Reports:
      - unique subjects (PTID)
      - total visits (samples)
      - baseline-only subjects (VISCODE == 'bl')
    """

    df = pd.read_csv(file_path)
    diagnosis_col = 'DIAGNOSIS'

    # Map diagnosis codes to labels (handles int or str)
    label_map = {1: 'CN', 2: 'MCI', 3: 'AD', '1': 'CN', '2': 'MCI', '3': 'AD'}
    df['DX_clean'] = df[diagnosis_col].replace(label_map)

    # Drop missing/invalid diagnoses
    df = df[df['DX_clean'].isin(['CN', 'MCI', 'AD'])].copy()

    # --- (A) Total visits (samples) ---
    visit_counts = df['DX_clean'].value_counts().sort_index()
    total_visits = visit_counts.sum()

    # --- (B) Unique subjects ---
    if 'PTID' in df.columns:
        subject_counts = df.drop_duplicates('PTID')['DX_clean'].value_counts().sort_index()
        total_subjects = subject_counts.sum()
    else:
        subject_counts = visit_counts
        total_subjects = total_visits

    # --- (C) Baseline-only subjects (VISCODE == 'bl') ---
    if 'VISCODE' in df.columns:
        baseline = df[df['VISCODE'].str.lower() == 'bl']
        baseline_counts = baseline.drop_duplicates('PTID')['DX_clean'].value_counts().sort_index()
        total_baseline = baseline_counts.sum()
    else:
        baseline_counts = pd.Series(dtype=int)
        total_baseline = 0

    # --- Print nicely ---
    print(f"\n=== {label} Summary ===")
    print("Unique participants per diagnosis:")
    print(subject_counts.to_string())
    print(f"Total unique participants: {total_subjects}\n")

    print("Visit-level samples per diagnosis:")
    print(visit_counts.to_string())
    print(f"Total visits: {total_visits}\n")

    if not baseline_counts.empty:
        print("Baseline-only composition:")
        print(baseline_counts.to_string())
        print(f"Total baseline subjects: {total_baseline}\n")

    # --- Return compact summary dict (useful for writing Methods) ---
    summary = {
        "unique_subjects": int(total_subjects),
        "visits": int(total_visits),
        "baseline_subjects": int(total_baseline),
        "subject_counts": subject_counts.to_dict(),
        "visit_counts": visit_counts.to_dict(),
        "baseline_counts": baseline_counts.to_dict()
    }
    return summary

# ---- Run summaries for each dataset ----
amy = summarize_diagnosis("data/250826_DX_AMYLOID_multi_visit.csv", label="Amyloid-MRI")
tau = summarize_diagnosis("data/250826_DX_TAU_multi_visit.csv", label="Tau-MRI")

# ---- Check overlap between the two ----
amy = pd.read_csv("data/250826_DX_AMYLOID_multi_visit.csv")
tau = pd.read_csv("data/250826_DX_TAU_multi_visit.csv")

# Normalize columns
amy.columns = amy.columns.str.upper()
tau.columns = tau.columns.str.upper()

# Clean PTID and VISCODE
amy['PTID'] = amy['PTID'].str.strip()
tau['PTID'] = tau['PTID'].str.strip()

print(amy.columns)
assert 'PTID' in amy.columns and 'PTID' in tau.columns, "PTID column missing in one of the datasets"
assert 'VISCODE' in amy.columns and 'VISCODE' in tau.columns, "VISCODE column missing in one of the datasets"
assert 'DIAGNOSIS' in amy.columns and 'DIAGNOSIS' in tau.columns, "DIAGNOSIS column missing in one of the datasets"

# Merge by both PTID and VISCODE (visit code)
merged = pd.merge(
    amy[['PTID', 'VISCODE', 'DIAGNOSIS']],
    tau[['PTID', 'VISCODE', 'DIAGNOSIS']],
    on=['PTID', 'VISCODE'],
    suffixes=('_AMY', '_TAU')
)

print(f"Total Amyloid visits: {len(amy)}")
print(f"Total Tau visits: {len(tau)}")
print(f"Overlapping visits (same subject + visit): {len(merged)}")

# If you want unique subjects among those overlapping visits
unique_overlap_subjects = merged['PTID'].nunique()
print(f"Unique subjects with overlapping visits: {unique_overlap_subjects}")

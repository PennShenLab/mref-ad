#!/usr/bin/env python3
"""
Summarize participant characteristics (Table 1) for the IEEE ICHI paper.

This script merges demographic and imaging data to compute:
- Number of participants per diagnosis (CN, MCI, AD)
- Age at visit (mean ± SD)
- Education (mean ± SD)
- Sex distribution (% female/male)

Usage (from repository root)::

    python data_preprocessing/summarize_participants.py

Defaults read ``data/PTDEMOG_30Sep2025.csv`` and
``data/freesurfer_lastvisit/250826_DX_AMYLOID_last_visit.csv``. Override paths with::

    export MREF_DEMO_CSV=/path/to/demographics.csv
    export MREF_IMG_CSV=/path/to/imaging_table.csv
    python data_preprocessing/summarize_participants.py

Writes under ``results/``: ``table1_participants_all_visits.csv`` / ``.tex`` and
``table1_participants_last_visit.csv`` / ``.tex`` (directory is created if missing).
"""

import os

import numpy as np
import pandas as pd

# ==== Load datasets (override with env or edit defaults for your tree) ====
_DEMO = os.environ.get("MREF_DEMO_CSV", "data/PTDEMOG_30Sep2025.csv")
_IMG = os.environ.get(
    "MREF_IMG_CSV",
    "data/freesurfer_lastvisit/250826_DX_AMYLOID_last_visit.csv",
)
os.makedirs("results", exist_ok=True)
demo = pd.read_csv(_DEMO)
img = pd.read_csv(_IMG)

# Normalize columns
for df in [demo, img]:
    df.columns = df.columns.str.upper().str.strip()
    df["PTID"] = df["PTID"].astype(str).str.strip()

# Merge by participant
# Avoid duplicate VISCODE / VISDATE columns from demographics (suffixes → imaging keeps plain names).
merged = pd.merge(img, demo, on="PTID", how="left", suffixes=("", "_DEMOG"))

# Map diagnosis codes to text
label_map = {1: "CN", 2: "MCI", 3: "AD", "1": "CN", "2": "MCI", "3": "AD"}
merged["DX_clean"] = merged["DIAGNOSIS"].replace(label_map)

# If age is missing but DOB and scan date exist, compute it
if "AGE_AT_VISIT" not in merged.columns and {"PTDOB", "SCANDATE"} <= set(merged.columns):
    merged["PTDOB_parsed"] = pd.to_datetime(merged["PTDOB"], errors="coerce")
    merged["SCANDATE"] = pd.to_datetime(merged["SCANDATE"], errors="coerce")
    merged["AGE_AT_VISIT"] = (merged["SCANDATE"] - merged["PTDOB_parsed"]).dt.days / 365.25

# Sort merged by PTID and SCANDATE ascending
merged = merged.sort_values(["PTID", "SCANDATE"], ascending=[True, True])

# Extract baseline diagnosis per participant (VISCODE == 'bl'), fallback to earliest visit if no 'bl' exists
baseline_rows = merged[merged["VISCODE"] == "bl"]
if baseline_rows.empty:
    # No baseline visits at all, fallback to earliest visit per participant
    baseline_dx = merged.groupby("PTID").first().reset_index()[["PTID", "DX_clean"]]
else:
    # Get baseline visits per participant
    baseline_dx = baseline_rows.drop_duplicates(subset="PTID", keep="first")[["PTID", "DX_clean"]]
    # Find participants without baseline visits
    baseline_ptids = set(baseline_dx["PTID"])
    all_ptids = set(merged["PTID"])
    missing_ptids = all_ptids - baseline_ptids
    if missing_ptids:
        # For participants without baseline visits, get earliest visit
        earliest_visits = merged[merged["PTID"].isin(missing_ptids)].groupby("PTID").first().reset_index()
        fallback_dx = earliest_visits[["PTID", "DX_clean"]]
        baseline_dx = pd.concat([baseline_dx, fallback_dx], ignore_index=True)

baseline_dx = baseline_dx.rename(columns={"DX_clean": "DX_group"})

# Verify number of unique PTIDs matches between baseline_dx and merged
unique_merged_ptids = merged["PTID"].nunique()
unique_baseline_ptids = baseline_dx["PTID"].nunique()

try:
    assert unique_merged_ptids == unique_baseline_ptids
    print(f"Assertion passed: Number of unique participants in baseline_dx ({unique_baseline_ptids}) matches merged data ({unique_merged_ptids}).")
except AssertionError:
    print(f"Assertion failed: baseline_dx has {unique_baseline_ptids} unique participants, merged data has {unique_merged_ptids}.")


# Merge baseline diagnosis back into full dataset
merged = pd.merge(merged, baseline_dx, on="PTID", how="left")

# --- Diagnostic progression analysis ---
# For each participant, extract the unique sequence of diagnoses across visits
dx_sequences = merged.sort_values(["PTID", "SCANDATE"]).groupby("PTID")["DX_clean"].apply(lambda x: list(pd.unique(x.dropna())))

# Define function to detect progression
def has_progression(seq):
    # Convert to set for fast lookup
    s = set(seq)
    # CN→MCI, CN→AD, or MCI→AD progression
    # (We require at least two unique states, and a later/advanced state present)
    progressed = False
    if "CN" in s and "MCI" in s:
        progressed = True
    if "CN" in s and "AD" in s:
        progressed = True
    if "MCI" in s and "AD" in s:
        progressed = True
    return progressed

progressed_mask = dx_sequences.apply(has_progression)
n_progressed = progressed_mask.sum()
total_subjects = dx_sequences.shape[0]
progression_rate = n_progressed / total_subjects * 100 if total_subjects > 0 else 0
print(f"Diagnostic progression: {n_progressed}/{total_subjects} participants ({progression_rate:.1f}%) showed CN→MCI→AD transitions")

# --- Detailed breakdown of progression patterns ---
def progression_type(seq):
    # Order-preserving unique (avoid pd.unique on plain list; pandas 3+ is strict)
    seq = list(dict.fromkeys(x for x in seq if pd.notna(x)))
    if "CN" in seq and "MCI" in seq and "AD" in seq:
        return "CN→MCI→AD"
    elif "CN" in seq and "MCI" in seq and "AD" not in seq:
        return "CN→MCI only"
    elif "MCI" in seq and "AD" in seq and "CN" not in seq:
        return "MCI→AD only"
    elif "CN" in seq and "AD" in seq and "MCI" not in seq:
        return "CN→AD direct"
    else:
        return "No progression"

progression_summary = dx_sequences.apply(progression_type).value_counts()
print("\n=== Diagnostic Progression Breakdown ===")
for pattern, count in progression_summary.items():
    pct = count / total_subjects * 100
    print(f"{pattern:<20}: {count:>5} participants ({pct:.1f}%)")


# --- Participant summary tables ---

# === TABLE 1A: All imaging visits ===
table_all_visits = (
    merged.groupby("DX_group")
    .agg(
        Participants=("PTID", "nunique"),
        Imaging_Visits=("PTID", "size"),
        AGE_MEAN=("AGE_AT_VISIT", "mean"),
        AGE_SD=("AGE_AT_VISIT", "std"),
        EDU_MEAN=("PTEDUCAT", "mean"),
        EDU_SD=("PTEDUCAT", "std"),
        N_MALE=("PTGENDER", lambda x: (x == 1).sum()),
        N_FEMALE=("PTGENDER", lambda x: (x == 2).sum()),
    )
    .round(2)
)
table_all_visits["Sex (M/F)"] = (
    table_all_visits["N_MALE"].astype(int).astype(str)
    + "/" + table_all_visits["N_FEMALE"].astype(int).astype(str)
)
table_all_visits["Age (years, mean±SD)"] = (
    table_all_visits["AGE_MEAN"].round(1).astype(str)
    + " ± " + table_all_visits["AGE_SD"].round(1).astype(str)
)
table_all_visits["Education (years, mean±SD)"] = (
    table_all_visits["EDU_MEAN"].round(1).astype(str)
    + " ± " + table_all_visits["EDU_SD"].round(1).astype(str)
)
table_all_visits = table_all_visits[["Participants", "Imaging_Visits", "Age (years, mean±SD)", "Sex (M/F)", "Education (years, mean±SD)"]]

#
# Transpose so that diagnoses are columns and info are rows
table_all_visits_T = table_all_visits.T
# Clear index and columns names for formatting consistency
table_all_visits_T.index.name = ""
table_all_visits_T.columns.name = ""

# === TABLE 1B: Last visit only ===
last_visits = (
    merged.sort_values(["PTID", "SCANDATE"])
    .groupby("PTID")
    .last()
    .reset_index()
)
table_last_visit = (
    last_visits.groupby("DX_group")
    .agg(
        Participants=("PTID", "nunique"),
        AGE_MEAN=("AGE_AT_VISIT", "mean"),
        AGE_SD=("AGE_AT_VISIT", "std"),
        EDU_MEAN=("PTEDUCAT", "mean"),
        EDU_SD=("PTEDUCAT", "std"),
        N_MALE=("PTGENDER", lambda x: (x == 1).sum()),
        N_FEMALE=("PTGENDER", lambda x: (x == 2).sum()),
    )
    .round(2)
)
table_last_visit["Sex (M/F)"] = (
    table_last_visit["N_MALE"].astype(int).astype(str)
    + "/" + table_last_visit["N_FEMALE"].astype(int).astype(str)
)
table_last_visit["Age (years, mean±SD)"] = (
    table_last_visit["AGE_MEAN"].round(1).astype(str)
    + " ± " + table_last_visit["AGE_SD"].round(1).astype(str)
)
table_last_visit["Education (years, mean±SD)"] = (
    table_last_visit["EDU_MEAN"].round(1).astype(str)
    + " ± " + table_last_visit["EDU_SD"].round(1).astype(str)
)
table_last_visit = table_last_visit.rename(columns={"Participants": "Participants (n)",
                                                    "Age (years, mean±SD)": "Age (years)",
                                                    "Education (years, mean±SD)": "Education (years)"})
table_last_visit = table_last_visit[["Participants (n)", "Age (years)", "Sex (M/F)", "Education (years)"]]

#
# === TABLE 1B: Last visit only (Transposed) ===
table_last_visit_T = table_last_visit.T
# Clear index and columns names for formatting consistency
table_last_visit_T.index.name = ""
table_last_visit_T.columns.name = ""

# --- Add Total column for both tables ---
def add_total_column(table_T):
    total_col = []
    for row in table_T.index:
        vals = table_T.loc[row]
        if "Participants" in row or "Imaging" in row:
            nums = pd.to_numeric(vals, errors="coerce")
            total_col.append(int(nums.sum()))
        elif "Sex" in row:
            male_total, female_total = 0, 0
            for v in vals:
                if isinstance(v, str) and "/" in v:
                    parts = v.split("/")
                    if len(parts) == 2:
                        try:
                            male_total += int(parts[0])
                            female_total += int(parts[1])
                        except ValueError:
                            continue
            total_col.append(f"{male_total}/{female_total}")
        elif "Age" in row or "Education" in row:
            means, sds, counts = [], [], []
            for col in table_T.columns:
                v = vals[col]
                if isinstance(v, str) and "±" in v:
                    mean_str, sd_str = v.split("±")
                    mean_val = float(mean_str.strip())
                    sd_val = float(sd_str.strip())
                    count = table_T.loc["Participants (n)", col] if "Participants (n)" in table_T.index else 1
                    means.append(mean_val)
                    sds.append(sd_val)
                    counts.append(count)
            if counts:
                mean_total = np.average(means, weights=counts)
                sd_total = np.sqrt(np.average(np.square(sds), weights=counts))
                total_col.append(f"{mean_total:.1f} ± {sd_total:.1f}")
            else:
                total_col.append("")
        else:
            total_col.append("")
    table_T["Total"] = total_col
    return table_T

# Apply to both tables
table_all_visits_T = add_total_column(table_all_visits_T)
table_last_visit_T = add_total_column(table_last_visit_T)

# Print and export both tables
# --- Table 1A ---
table_all_visits_T.index.name = ""
table_all_visits_T.columns.name = ""
table_all_visits_T.index = [
    "Participants (n)",
    "Imaging visits (n)",
    "Age (years)",
    "Sex (M/F)",
    "Education (years)",
]
print("\n=== Table 1A (CN, MCI, AD + Total): All Imaging Visits ===")
print(table_all_visits_T)
table_all_visits_T.to_csv("results/table1_participants_all_visits.csv")
table_all_visits_T.to_latex(
    "results/table1_participants_all_visits.tex",
    index=True,
    header=True,
    index_names=False,
    escape=False,
)

# --- Table 1B ---
table_last_visit_T.index.name = ""
table_last_visit_T.columns.name = ""
table_last_visit_T.index = [
    "Participants (n)",
    "Age (years)",
    "Sex (M/F)",
    "Education (years)",
]
print("\n=== Table 1B (CN, MCI, AD + Total): Last Visit Only ===")
print(table_last_visit_T)
table_last_visit_T.to_csv("results/table1_participants_last_visit.csv")
table_last_visit_T.to_latex(
    "results/table1_participants_last_visit.tex",
    index=True,
    header=True,
    index_names=False,
    escape=False,
)
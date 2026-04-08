#!/usr/bin/env python3
"""
Plot MoE interpretability from fold-based or seed-based results.

Generates:
    1. Bar plot: mean ± std gate weights across folds/seeds
    2. Heatmap: gate weights per fold/seed
    3. Subject-level heatmaps grouped by diagnosis
    4. Modality-region heatmaps
    5. Brain plotting CSV (R-compatible)

Usage:
    # Auto-detect seed-based results (default)
    python3 scripts/plot_moe_interpretability.py
  
    # Explicit seed-based with custom pattern
    python3 scripts/plot_moe_interpretability.py --input "results/moe_seed_*_full_final_per_subject.json" --input_type seed
  
    # Fold-based (CV folds)
    python3 scripts/plot_moe_interpretability.py --input results/moe_hierarchical_cv10_full.json --input_type fold
  
    # Custom output directory
    python3 scripts/plot_moe_interpretability.py --output_dir results/my_plots
"""

import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import argparse

# -----------------------------
# Parse arguments
# -----------------------------
ap = argparse.ArgumentParser(description="Plot MoE interpretability from fold-based or seed-based results")
ap.add_argument("--input", type=str, default=None, help="Input JSON file (fold-based) or glob pattern for seed-based per-subject JSONs (e.g., 'results/moe_seed_*_full_final_per_subject.json')")
ap.add_argument("--input_type", type=str, choices=["fold", "seed", "auto"], default="auto", help="Type of input: 'fold' for CV folds, 'seed' for multi-seed results, 'auto' to detect")
ap.add_argument("--output_dir", type=str, default="results/plots", help="Output directory for plots")
args = ap.parse_args()

# Determine input path and type
if args.input:
    IN_PATH = args.input
    INPUT_TYPE = args.input_type
else:
    # Default: try to detect seed-based results first, then fall back to fold-based
    seed_pattern = "results/moe_seed_*_full_final_per_subject.json"
    seed_files = glob.glob(seed_pattern)
    if seed_files and args.input_type in ["auto", "seed"]:
        IN_PATH = seed_pattern
        INPUT_TYPE = "seed"
        print(f"[INFO] Auto-detected seed-based results: {len(seed_files)} files")
    else:
        IN_PATH = "results/moe_hierarchical_cv10_full.json"
        INPUT_TYPE = "fold"
        print(f"[INFO] Using default fold-based results")

OUT_DIR = args.output_dir

# -----------------------------
# Load results
# -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)

def load_seed_based_results(pattern):
    """Load seed-based per-subject JSON files."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    seeds_data = []
    for file_path in files:
        # Extract seed number from filename
        import re
        match = re.search(r'seed[_-]?(\d+)', file_path)
        seed_id = int(match.group(1)) if match else len(seeds_data)
        
        with open(file_path, 'r') as f:
            per_subject = json.load(f)
        
        # Aggregate gate weights across subjects
        if not per_subject:
            continue
        
        # Extract gate weights from all subjects
        all_gate_weights = [s["gate_weights"] for s in per_subject if "gate_weights" in s]
        if not all_gate_weights:
            continue
        
        # Compute mean gate weights across subjects for this seed
        gate_keys = all_gate_weights[0].keys()
        gate_mean_weights = {key: np.mean([gw[key] for gw in all_gate_weights]) for key in gate_keys}
        
        # Convert to fold-like structure
        seed_record = {
            "fold": f"seed_{seed_id}",
            "seed": seed_id,
            "gate_mean_weights": gate_mean_weights,
            "gate_outputs_val": [
                {
                    "PTID": s["PTID"],
                    "y": s["y_true"],
                    "weights": s["gate_weights"]
                }
                for s in per_subject if "gate_weights" in s
            ]
        }
        seeds_data.append(seed_record)
    
    return seeds_data

def load_fold_based_results(path):
    """Load fold-based JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict) and "folds" in data:
            return data["folds"]
        else:
            return data

# Load data based on type
if INPUT_TYPE == "seed" or (INPUT_TYPE == "auto" and "*" in IN_PATH):
    folds = load_seed_based_results(IN_PATH)
    data_label = "Seed"
    xlabel_singular = "Seed"
    print(f"[INFO] Loaded {len(folds)} seeds from pattern: {IN_PATH}")
elif INPUT_TYPE == "fold":
    folds = load_fold_based_results(IN_PATH)
    data_label = "Fold"
    xlabel_singular = "Fold"
    print(f"[INFO] Loaded {len(folds)} folds from {IN_PATH}")
else:
    # Auto-detect
    if os.path.isfile(IN_PATH):
        folds = load_fold_based_results(IN_PATH)
        data_label = "Fold"
        xlabel_singular = "Fold"
        print(f"[INFO] Loaded {len(folds)} folds from {IN_PATH}")
    else:
        folds = load_seed_based_results(IN_PATH)
        data_label = "Seed"
        xlabel_singular = "Seed"
        print(f"[INFO] Loaded {len(folds)} seeds from pattern: {IN_PATH}")

# -----------------------------
# Aggregate gate weights
# -----------------------------
df = pd.DataFrame([
    {"fold": f["fold"], **f["gate_mean_weights"]}
    for f in folds
])

gate_cols = [c for c in df.columns if c != "fold"]
df = df[["fold"] + gate_cols]

# Compute mean and std across folds
mean_w = df[gate_cols].mean().sort_values(ascending=False)
std_w = df[gate_cols].std()[mean_w.index]

# -----------------------------
# Assign colors by modality
# -----------------------------
def modality_color(name):
    if name.startswith("amy_"):
        return "#5DADE2"   # blue for amyloid
    elif name.startswith("mri_"):
        return "#58D68D"   # green for MRI
    elif "demo" in name.lower():
        return "#E67E22"   # orange for demographic
    else:
        return "#ABB2B9"   # gray fallback

colors = [modality_color(x) for x in mean_w.index]

def clean_label(name):
    if name.startswith("mri_"):
        label = name[len("mri_"):]
    elif name.startswith("amy_"):
        label = name[len("amy_"):]
    elif "demo" in name.lower():
        return "Demographic"
    else:
        label = name
    label = label.replace("_", " ")
    return label.title()

# -----------------------------
# Figure 1: Bar plot with std
# -----------------------------
plt.figure(figsize=(8, 10))
mean_w_rev = mean_w[::-1]
std_w_rev = std_w[::-1]
colors_rev = colors[::-1]
plt.barh(range(len(mean_w_rev)), mean_w_rev, xerr=std_w_rev, capsize=3, color=colors_rev, alpha=0.9)
labels_clean = [clean_label(x) for x in mean_w_rev.index]
plt.yticks(range(len(mean_w_rev)), labels_clean, fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Mean Gate Weight ± STD", fontsize=14)
# plt.title(f"Average Expert Contribution Across {data_label}s", fontsize=16)
legend_patches = [
    mpatches.Patch(color="#58D68D", label="MRI"),
    mpatches.Patch(color="#5DADE2", label="Amyloid PET"),
    mpatches.Patch(color="#E67E22", label="Demographic")
]
plt.legend(handles=legend_patches, loc="lower right", frameon=False, fontsize=12)
plt.tight_layout(pad=1.0)

out_bar = os.path.join(OUT_DIR, "moe_gate_barplot.png")
plt.savefig(out_bar, dpi=600)
plt.close()
print(f"[INFO] Saved bar plot → {out_bar}")

# -----------------------------
# Figure 2: Heatmap across folds
# -----------------------------
plt.figure(figsize=(12, 8))
sorted_cols = mean_w.index.tolist()
sns.heatmap(
    df[sorted_cols].T,
    cmap="viridis_r",
    annot=False,
    cbar_kws={'label': 'Gate Weight'},
    linewidths=0.4
)
plt.yticks(rotation=0, fontsize=8)
plt.xlabel(xlabel_singular)
plt.ylabel("Expert (Amyloid / MRI / Demographic)")
plt.title(f"Consistency of Expert Importance Across {data_label}s")
plt.tight_layout()

out_heatmap = os.path.join(OUT_DIR, "moe_gate_heatmap.png")
plt.savefig(out_heatmap, dpi=600)
plt.close()
print(f"[INFO] Saved heatmap → {out_heatmap}")
# Save underlying data (fold/seed-level heatmap values) as TSV and CSV
df.to_csv(os.path.join(OUT_DIR, "moe_gate_heatmap_values.tsv"), sep="\t", index=False)
df.to_csv(os.path.join(OUT_DIR, "moe_gate_heatmap_values.csv"), index=False)
print(f"[INFO] Saved {data_label.lower()}-level heatmap data → {os.path.join(OUT_DIR, 'moe_gate_heatmap_values.tsv')}")
print(f"[INFO] Saved {data_label.lower()}-level heatmap data (CSV) → {os.path.join(OUT_DIR, 'moe_gate_heatmap_values.csv')}")

# -----------------------------
# Figure 3: Subject-level heatmap grouped by diagnosis
# -----------------------------
print("[INFO] Generating subject-level heatmap grouped by diagnosis...")

def clean_label(name):
    if name.startswith("mri_"):
        label = name[len("mri_"):]
    elif name.startswith("amy_"):
        label = name[len("amy_"):]
    elif "demo" in name.lower():
        return "Demographic"
    else:
        label = name
    label = label.replace("_", " ")
    return label.title()

def label_with_modality(name):
    if name.startswith("mri_"):
        return clean_label(name) + " (MRI)"
    elif name.startswith("amy_"):
        return clean_label(name) + " (Amyloid)"
    elif "demo" in name.lower():
        return "Demographic"
    else:
        return clean_label(name)

# Collect per-subject gate weights and diagnoses from all folds using gate_outputs_val
subject_data_list = []
diag_map = {0: "CN", 1: "MCI", 2: "AD"}

for fold in folds:
    gate_outputs_val = fold.get("gate_outputs_val", None)
    if gate_outputs_val is None:
        continue
    for entry in gate_outputs_val:
        weights = entry.get("weights", None)
        diagnosis = entry.get("y", None)
        if weights is None or diagnosis is None:
            continue
        # Optional: get subject ID if available
        subj_id = entry.get("PTID", None)
        data_entry = {"diagnosis": diag_map.get(diagnosis, str(diagnosis))}
        if subj_id is not None:
            data_entry["subject_id"] = subj_id
        data_entry.update(weights)
        subject_data_list.append(data_entry)

if len(subject_data_list) == 0:
    print("[WARNING] No subject-level gating weights or diagnoses found in folds. Skipping subject-level heatmap.")
else:
    subj_df = pd.DataFrame(subject_data_list)
    # Ensure gate columns present and reorder columns according to mean_w order
    gate_cols_subj = [col for col in mean_w.index if col in subj_df.columns]
    # Ensure demographic column is included if present in subj_df but missing in gate_cols_subj
    if "demographic" in subj_df.columns and "demographic" not in gate_cols_subj:
        gate_cols_subj.append("demographic")
    if len(gate_cols_subj) == 0:
        print("[WARNING] No matching gate columns found in subject-level data. Skipping subject-level heatmap.")
    else:
        # --- Branch 1: Subject-pooled version (existing logic, now labeled) ---
        diag_gate_means_subjectpooled = subj_df.groupby("diagnosis")[gate_cols_subj].mean()
        diag_order = ["CN", "MCI", "AD"]
        diag_gate_means_subjectpooled = diag_gate_means_subjectpooled.reindex(diag_order, axis=0)
        ordered_cols = [col for col in mean_w.index if col in diag_gate_means_subjectpooled.columns]
        diag_gate_means_subjectpooled = diag_gate_means_subjectpooled[ordered_cols]
        # Save subject-pooled TSV
        out_subjectpooled_tsv = os.path.join(OUT_DIR, "moe_gate_subject_heatmap_values_subjectpooled.tsv")
        diag_gate_means_subjectpooled.to_csv(out_subjectpooled_tsv, sep="\t")
        print(f"[INFO] Saved diagnosis-level subject heatmap data (subject-pooled) → {out_subjectpooled_tsv}")

        # --- Branch 2: Fold/Seed-averaged version ---
        # For each fold/seed, compute per-diagnosis mean, then average across folds/seeds
        fold_diag_means_list = []
        for fold in folds:
            gate_outputs_val = fold.get("gate_outputs_val", None)
            if gate_outputs_val is None:
                continue
            # Build DataFrame for this fold
            fold_entries = []
            for entry in gate_outputs_val:
                weights = entry.get("weights", None)
                diagnosis = entry.get("y", None)
                if weights is None or diagnosis is None:
                    continue
                fold_entry = {"diagnosis": diag_map.get(diagnosis, str(diagnosis))}
                fold_entry.update(weights)
                fold_entries.append(fold_entry)
            if not fold_entries:
                continue
            fold_df = pd.DataFrame(fold_entries)
            fold_gate_cols = [col for col in gate_cols_subj if col in fold_df.columns]
            # Only keep columns that exist in this fold
            fold_diag_means = fold_df.groupby("diagnosis")[fold_gate_cols].mean()
            # Reindex to ensure all diags in order
            fold_diag_means = fold_diag_means.reindex(diag_order, axis=0)
            # Reorder columns to match global mean order
            fold_diag_means = fold_diag_means[[col for col in ordered_cols if col in fold_diag_means.columns]]
            fold_diag_means_list.append(fold_diag_means)
        # Now, stack and average across folds/seeds (axis=0 is diagnosis, axis=1 is gate, axis=2 is fold/seed)
        if fold_diag_means_list:
            stacked = np.stack([df.values for df in fold_diag_means_list])
            # Average across folds/seeds (axis=0)
            foldavg_vals = stacked.mean(axis=0)
            diag_gate_means_foldavg = pd.DataFrame(foldavg_vals, index=diag_order, columns=ordered_cols)
            suffix = "foldavg" if INPUT_TYPE == "fold" else "seedavg"
            out_foldavg_tsv = os.path.join(OUT_DIR, f"moe_gate_subject_heatmap_values_{suffix}.tsv")
            diag_gate_means_foldavg.to_csv(out_foldavg_tsv, sep="\t")
            print(f"[INFO] Saved diagnosis-level subject heatmap data ({data_label.lower()}-averaged) → {out_foldavg_tsv}")
        else:
            diag_gate_means_foldavg = None
            print(f"[WARNING] No {data_label.lower()}-averaged diagnosis-level heatmap data could be computed.")

        # --- Existing plotting logic uses old diag_gate_means; keep for now ---
        diag_gate_means = diag_gate_means_subjectpooled

        # Clean y-axis labels with modality suffixes
        y_labels_clean = [label_with_modality(col) for col in diag_gate_means.columns]

        plt.figure(figsize=(8, 10))
        ax = sns.heatmap(
            diag_gate_means.T,
            cmap="viridis_r",
            annot=False,
            cbar_kws={'label': 'Mean Gate Weight'},
            linewidths=0.4
        )
        # Set y-tick labels (all black, no color)
        ax.set_yticks(np.arange(len(y_labels_clean)) + 0.5)
        ax.set_yticklabels(y_labels_clean, rotation=0, fontsize=13, color="black")
        plt.xticks(rotation=0, ha='center', fontsize=13)
        # Axis label paddings and font sizes (increased)
        ax.set_ylabel("Expert (Amyloid / MRI / Demographic)", labelpad=18, fontsize=14)
        ax.set_xlabel("Diagnosis", labelpad=15, fontsize=14)
        # Colorbar label padding and fontsize (increased)
        cbar = ax.collections[0].colorbar
        cbar.set_label("Mean Gate Weight", labelpad=18, fontsize=13)
        # Removed legend for modality colors as per instructions
        # Remove the title
        # Adjust layout for more space between heatmap and colorbar
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        out_subject_heatmap = os.path.join(OUT_DIR, "moe_gate_subject_heatmap.png")
        plt.savefig(out_subject_heatmap, dpi=600)
        plt.close()
        print(f"[INFO] Saved subject-level heatmap → {out_subject_heatmap}")
        # Save underlying data (diagnosis-level subject heatmap values) as TSV (original name for backward compatibility)
        diag_gate_means.to_csv(os.path.join(OUT_DIR, "moe_gate_subject_heatmap_values.tsv"), sep="\t")
        print(f"[INFO] Saved diagnosis-level heatmap data → {os.path.join(OUT_DIR, 'moe_gate_subject_heatmap_values.tsv')}")

        # -----------------------------
        # Additional Figure: Heatmap with global mean weights included
        # -----------------------------

        # Add global mean weights as an "All" row to diag_gate_means
        diag_gate_means_with_all = diag_gate_means.copy()
        # mean_w is a Series with index as columns, so convert to DataFrame with one row "All"
        all_row = mean_w[diag_gate_means_with_all.columns].to_frame().T
        all_row.index = ["All"]
        diag_gate_means_with_all = pd.concat([diag_gate_means_with_all, all_row], axis=0)

        # Reorder index to ["All", "CN", "MCI", "AD"]
        diag_gate_means_with_all = diag_gate_means_with_all.reindex(["All", "CN", "MCI", "AD"])

        # Mapping for row labels (experts), match ignoring case
        expert_label_map = {
            "subcortical_temporal": "Subcortical Temporal",
            "striatum_basal_ganglia": "Subcortical Striatum/Basal Ganglia",
            "subcortical_thalamus": "Subcortical Thalamus",
            "cerebral_cortex": "Cerebral (WM)",
            "cerebellum": "Cerebellum",
            "ventricles": "Ventricles",
            "corpus_callosum_wm": "Corpus Collosum (WM)",
            "corpus_collosum_wm": "Corpus Collosum (WM)",
            "brainstem": "Brainstem",
            "frontal_lobe": "Frontal Lobe",
            "cingulate": "Cingulate",
            "parietal_lobe": "Parietal Lobe",
            "temporal_lobe": "Temporal Lobe",
            "occipital_lobe": "Occipital Lobe",
            "sensory_motor_cortex": "Sensory-Motor Cortex",
            "demographic": "Demographic"
        }

        # Prepare row labels by mapping diag_gate_means_with_all columns (experts)
        # The heatmap is transposed, so rows correspond to columns of diag_gate_means_with_all
        new_row_labels = []
        for col in diag_gate_means_with_all.columns:
            # Remove modality prefix before mapping
            if col.startswith("mri_"):
                base_name = col[len("mri_"):]
                modality_suffix = " (MRI)"
            elif col.startswith("amy_"):
                base_name = col[len("amy_"):]
                modality_suffix = " (Amyloid)"
            elif "demo" in col.lower():
                base_name = "demographic"
                modality_suffix = ""
            else:
                base_name = col
                modality_suffix = ""

            col_lower = base_name.lower().replace(" ", "_")
            mapped_label = None
            for key in expert_label_map:
                if key == col_lower:
                    mapped_label = expert_label_map[key]
                    break
            if mapped_label is None:
                # If no exact match, try partial match ignoring case
                for key in expert_label_map:
                    if key in col_lower:
                        mapped_label = expert_label_map[key]
                        break
            if mapped_label is None:
                # Fallback: use original base_name with underscores replaced and title case
                mapped_label = base_name.replace("_", " ").title()

            if base_name == "demographic":
                mapped_label = "Demographic"
                modality_suffix = ""

            final_label = mapped_label + modality_suffix
            new_row_labels.append(final_label)

        plt.figure(figsize=(8, 10))
        cmap = sns.color_palette("mako_r", as_cmap=True)
        ax2 = sns.heatmap(
            diag_gate_means_with_all.T,
            cmap=cmap,
            annot=False,
            cbar_kws={'label': 'Average Expert Contribution'},
            linewidths=0.3,
            linecolor="white"
        )
        # Set y-tick labels with mapped expert labels, all black, no color
        ax2.set_yticks(np.arange(len(new_row_labels)) + 0.5)
        ax2.set_yticklabels(new_row_labels, rotation=0, fontsize=13, color="black")
        plt.xticks(rotation=0, ha='center', fontsize=13)
        # Axis label paddings and font sizes (increased)
        # Removed x-axis and y-axis labels
        # ax2.set_ylabel("Expert", labelpad=18, fontsize=14)
        # ax2.set_xlabel("Diagnosis", labelpad=15, fontsize=14)
        # Colorbar label padding and fontsize (increased)
        cbar2 = ax2.collections[0].colorbar
        cbar2.set_label("Average Expert Contribution", labelpad=18, fontsize=13)
        # Adjust layout for more space
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        out_subject_heatmap_all = os.path.join(OUT_DIR, "moe_gate_subject_heatmap_with_all_final.png")
        plt.savefig(out_subject_heatmap_all, dpi=600)
        plt.close()
        print(f"[INFO] Saved subject-level heatmap with global mean → {out_subject_heatmap_all}")
        # Save underlying data (heatmap with global mean) as TSV
        diag_gate_means_with_all.to_csv(os.path.join(OUT_DIR, "moe_gate_subject_heatmap_with_all_values.tsv"), sep="\t")
        print(f"[INFO] Saved heatmap data with global mean → {os.path.join(OUT_DIR, 'moe_gate_subject_heatmap_with_all_values.tsv')}")

        # -----------------------------
        # Additional Export: CSV for brain plotting in R
        # -----------------------------
        print("[INFO] Generating CSV for brain plotting...")
        
        # Extract mean gate weights for brain regions (from "All" row)
        all_row_data = diag_gate_means_with_all.loc["All"]
        
        # Create brain plot CSV
        brain_plot_rows = []
        for col in all_row_data.index:
            if col == "demographic" or "demo" in col.lower():
                continue  # Skip demographic
            
            # Parse modality and region
            if col.startswith("mri_"):
                modality = "mri"
                region = col[4:]
            elif col.startswith("amy_"):
                modality = "amy"
                region = col[4:]
            else:
                continue
            
            # Find or create matching row
            existing_row = next((r for r in brain_plot_rows if r["region"] == region), None)
            if existing_row is None:
                # Determine if cortical (DK) or subcortical (ASEG)
                is_subcortical = any(x in region.lower() for x in ["subcortical", "ventricle", "brainstem", "cerebellum", "corpus"])
                dk = 0 if is_subcortical else 1
                
                existing_row = {
                    "ggseg_dk": region.replace("_", " ").title(),
                    "region": region,
                    "mri": 0.0,
                    "amy": 0.0,
                    "dk": dk
                }
                brain_plot_rows.append(existing_row)
            
            # Set weight
            existing_row[modality] = float(all_row_data[col])
        
        # Create DataFrame
        brain_df = pd.DataFrame(brain_plot_rows)
        brain_df = brain_df[["ggseg_dk", "mri", "amy", "dk"]]  # Reorder columns
        
        # Save CSV
        brain_csv_path = os.path.join(OUT_DIR, "moe_gate_weights_for_brain_plot.csv")
        brain_df.to_csv(brain_csv_path, index=False)
        print(f"[INFO] Saved brain plot CSV → {brain_csv_path}")
        print(f"[INFO] This CSV can be used with brain_plot_clean_zixuan.R")
        print(f"       Update YOUR_ROI_DATA_ALL.csv path to: {brain_csv_path}")


        # -----------------------------
        # Additional Figure: Modality-specific heatmap by brain region
        # -----------------------------

        # Determine the ordered list of regions from expert_label_map values
        region_order = []
        for key in expert_label_map:
            label = expert_label_map[key]
            if label not in region_order:
                region_order.append(label)

        # Create mapping from region label to keys for lookup
        # Actually we want to sum weights per modality and region
        # We'll map each gate column to modality and region
        # For demographic, skip it since it's not a brain region

        # Prepare a dict mapping gate column to (modality, region)
        gate_mod_region = {}
        for gate_col in mean_w.index:
            if gate_col.startswith("mri_"):
                modality = "MRI"
                base_name = gate_col[len("mri_"):]
                base_name_key = base_name.lower().replace(" ", "_")
                # Map base_name_key to region label using expert_label_map keys (case insensitive)
                region_label = None
                for key in expert_label_map:
                    if key == base_name_key:
                        region_label = expert_label_map[key]
                        break
                if region_label is None:
                    # Try partial match
                    for key in expert_label_map:
                        if key in base_name_key:
                            region_label = expert_label_map[key]
                            break
                if region_label is None:
                    # fallback use title case base_name with spaces
                    region_label = base_name.replace("_", " ").title()
                gate_mod_region[gate_col] = (modality, region_label)
            elif gate_col.startswith("amy_"):
                modality = "Amyloid"
                base_name = gate_col[len("amy_"):]
                base_name_key = base_name.lower().replace(" ", "_")
                region_label = None
                for key in expert_label_map:
                    if key == base_name_key:
                        region_label = expert_label_map[key]
                        break
                if region_label is None:
                    for key in expert_label_map:
                        if key in base_name_key:
                            region_label = expert_label_map[key]
                            break
                if region_label is None:
                    region_label = base_name.replace("_", " ").title()
                gate_mod_region[gate_col] = (modality, region_label)
            elif "demo" in gate_col.lower():
                # Instead of skipping, duplicate demographic into both MRI and Amyloid modalities
                demo_val = mean_w.get("demographic", 0.0)
                # Assign the demographic value to both MRI and Amyloid under region "Demographic"
                gate_mod_region["demographic_MRI"] = ("MRI", "Demographic")
                gate_mod_region["demographic_Amyloid"] = ("Amyloid", "Demographic")
                # No need to process further for this key
                continue
            else:
                # Unknown modality, skip
                continue

        # Initialize data structure for sums
        modality_region_sums = {
            "MRI": {region: 0.0 for region in region_order},
            "Amyloid": {region: 0.0 for region in region_order}
        }
        modality_region_counts = {
            "MRI": {region: 0 for region in region_order},
            "Amyloid": {region: 0 for region in region_order}
        }

        # Sum the mean weights per modality and region
        for gate_col, (modality, region) in gate_mod_region.items():
            # For demographic, use mean_w["demographic"] as value
            if region == "Demographic":
                val = mean_w.get("demographic", 0.0)
            else:
                # If gate_col is a synthetic demographic key, skip (already handled above)
                if gate_col not in mean_w.index:
                    continue
                val = mean_w[gate_col]
            modality_region_sums[modality][region] += val
            modality_region_counts[modality][region] += 1

        # For each modality and region, compute average contribution by dividing sum by count if count > 0
        modality_region_avg = {
            mod: {
                region: (modality_region_sums[mod][region] / modality_region_counts[mod][region]
                         if modality_region_counts[mod][region] > 0 else 0.0)
                for region in region_order
            }
            for mod in modality_region_sums
        }

        # Create DataFrame for heatmap: rows=modalities, columns=regions
        modality_region_df = pd.DataFrame.from_dict(modality_region_avg, orient="index")
        modality_region_df = modality_region_df[region_order]  # ensure column order

        # Shorten region names for x-axis using standard neuroimaging abbreviations
        region_short_labels = {
            "Subcortical Temporal": "Subcort. Temp",
            "Subcortical Striatum/Basal Ganglia": "Striatum/BG",
            "Subcortical Thalamus": "Thalamus",
            "Cerebral (WM)": "Cerebral WM",
            "Cerebellum": "Cerebellum",
            "Ventricles": "Ventricles",
            "Corpus Collosum (WM)": "Corpus WM",
            "Brainstem": "Brainstem",
            "Frontal Lobe": "Frontal",
            "Cingulate": "Cingulate",
            "Parietal Lobe": "Parietal",
            "Temporal Lobe": "Temporal",
            "Occipital Lobe": "Occipital",
            "Sensory-Motor Cortex": "Sensorimotor",
            "Demographic": "Demographic"
        }
        modality_region_df.rename(columns=region_short_labels, inplace=True)

        # --- Compact, single-column–optimized modality-region heatmap ---
        plt.figure(figsize=(6, 2.5))  # more compact for single-column
        colorbar_position = "right"  # or "right"

        if colorbar_position == "bottom":
            # Reduce figure height to make boxes shorter
            plt.gcf().set_size_inches(6, 1.6)
            cbar_orientation = "horizontal"
            cbar_kws = {
                'orientation': cbar_orientation,
                # 'label': 'Average Expert Contribution',
                'pad': 0.25,
                'shrink': 0.8
            }
        else:
            # Make figure shorter vertically for single-column layout
            plt.gcf().set_size_inches(6, 1.8)
            cbar_orientation = "vertical"
            cbar_kws = {
                'orientation': cbar_orientation, 
                # 'label': 'Average Expert Contribution'
            }

        ax3 = sns.heatmap(
            modality_region_df,
            cmap="mako_r",
            annot=False,
            cbar_kws=cbar_kws,
            linewidths=0.4
        )

        # Relabel Amyloid → PET
        y_labels = [label.get_text().replace("Amyloid", "PET") for label in ax3.get_yticklabels()]
        ax3.set_yticklabels(y_labels, rotation=0, fontsize=12)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=10)

        if colorbar_position == "bottom":
            # Make heatmap boxes shorter by setting aspect ratio
            ax3.set_aspect(0.35)
            plt.tight_layout(rect=[0, 0.12, 1, 1])
            cbar = ax3.collections[0].colorbar
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Average Expert Contribution", labelpad=6, fontsize=11)
            # [left, bottom, width, height]
            cbar.ax.set_position([0.25, 0.2, 0.65, 0.04])  # tighter fit, centered
        else:
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.subplots_adjust(right=0.88)
            cbar = ax3.collections[0].colorbar
            cbar.ax.set_position([0.75, 0.4, 0.03, 0.5])
            # cbar.set_label("Average Expert Contribution", labelpad=8, fontsize=8)

        out_modality_heatmap = os.path.join(OUT_DIR, "moe_gate_modality_region_heatmap.png")
        plt.savefig(out_modality_heatmap, dpi=600)
        plt.close()
        print(f"[INFO] Saved modality-region heatmap → {out_modality_heatmap}")
        # Save underlying data as TSV
        modality_region_df.to_csv(os.path.join(OUT_DIR, "moe_gate_modality_region_heatmap.tsv"), sep="\t")
        print(f"[INFO] Saved modality-region heatmap data → {os.path.join(OUT_DIR, 'moe_gate_modality_region_heatmap.tsv')}")

        # -----------------------------
        # Additional Figure: Modality-region heatmap split by diagnosis
        # -----------------------------
        print("[INFO] Generating modality-region heatmap split by diagnosis...")

        diagnosis_order = ["All", "CN", "MCI", "AD"]

        # Helper to map a base region name (without modality prefix) to our canonical label
        def _to_region_label(base: str) -> str:
            base_key = base.lower().replace(" ", "_")
            for key in expert_label_map:
                if key == base_key or key in base_key:
                    return expert_label_map[key]
            return base.replace("_", " ").title()

        # Prepare sum and count dicts for averaging per (modality, diagnosis, region)
        sums = {mod: {diag: {r: 0.0 for r in region_order} for diag in diagnosis_order}
                for mod in ["MRI", "PET"]}
        counts = {mod: {diag: {r: 0 for r in region_order} for diag in diagnosis_order}
                  for mod in ["MRI", "PET"]}

        # Accumulate values per diagnosis
        for diag in diagnosis_order:
            if diag not in diag_gate_means_with_all.index:
                continue
            row = diag_gate_means_with_all.loc[diag]
            for col, val in row.items():
                if col.startswith("mri_"):
                    modality, base = "MRI", col[len("mri_"):]
                    region_label = _to_region_label(base)
                    if region_label in sums[modality][diag]:
                        sums[modality][diag][region_label] += float(val)
                        counts[modality][diag][region_label] += 1
                elif col.startswith("amy_"):
                    modality, base = "PET", col[len("amy_"):]
                    region_label = _to_region_label(base)
                    if region_label in sums[modality][diag]:
                        sums[modality][diag][region_label] += float(val)
                        counts[modality][diag][region_label] += 1
                elif "demo" in col.lower():
                    # Duplicate demographic into both modalities
                    region_label = "Demographic"
                    for modality in ["MRI", "PET"]:
                        if region_label in sums[modality][diag]:
                            sums[modality][diag][region_label] += float(val)
                            counts[modality][diag][region_label] += 1

        # Build averaged DataFrames for MRI and PET (rows=diagnosis, cols=regions in region_order)
        def _avg_df(modality: str) -> pd.DataFrame:
            data = []
            for diag in diagnosis_order:
                row_vals = []
                for region in region_order:
                    c = counts[modality][diag][region]
                    s = sums[modality][diag][region]
                    row_vals.append(s / c if c > 0 else 0.0)
                data.append(row_vals)
            df_mod = pd.DataFrame(data, index=diagnosis_order, columns=region_order)
            # Apply short labels
            df_mod.rename(columns=region_short_labels, inplace=True)
            return df_mod

        mri_df = _avg_df("MRI")
        pet_df = _avg_df("PET")

        # Save TSVs of the split heatmaps
        mri_tsv = os.path.join(OUT_DIR, "moe_gate_modality_region_heatmap_MRI_by_diagnosis.tsv")
        pet_tsv = os.path.join(OUT_DIR, "moe_gate_modality_region_heatmap_PET_by_diagnosis.tsv")
        mri_df.to_csv(mri_tsv, sep="\t")
        pet_df.to_csv(pet_tsv, sep="\t")

        # --- Plot two aligned stacked heatmaps: top=MRI, bottom=PET (shared colorbar) ---
        vmin = min(mri_df.min().min(), pet_df.min().min())
        vmax = max(mri_df.max().max(), pet_df.max().max())
        cmap = sns.color_palette("mako_r", as_cmap=True)

        fig, axes = plt.subplots(
            2, 1,
            figsize=(8, 4.0),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05}
        )

        # Define common color normalization so both are scaled identically
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Top: MRI
        ax_top = axes[0]
        sns.heatmap(
            mri_df,
            ax=ax_top,
            cmap=cmap,
            norm=norm,
            cbar=False,
            linewidths=0.4,
            linecolor="white"
        )
        ax_top.set_ylabel("MRI", rotation=0, labelpad=30, fontsize=12, va="center")
        ax_top.set_yticklabels(diagnosis_order, rotation=0, fontsize=10)
        ax_top.set_xticklabels([])
        ax_top.tick_params(axis='x', length=0)

        # Bottom: PET
        ax_bot = axes[1]
        hm = sns.heatmap(
            pet_df,
            ax=ax_bot,
            cmap=cmap,
            norm=norm,
            cbar=False,
            linewidths=0.4,
            linecolor="white"
        )
        ax_bot.set_ylabel("PET", rotation=0, labelpad=30, fontsize=12, va="center")
        ax_bot.set_yticklabels(diagnosis_order, rotation=0, fontsize=10)
        ax_bot.set_xticklabels(ax_bot.get_xticklabels(), rotation=45, ha='right', fontsize=8)

        # Shared colorbar spanning both heatmaps on the right
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation="vertical"
        )
        cbar.ax.tick_params(labelsize=9)
        # Optional label
        # cbar.set_label("Average Expert Contribution", fontsize=10, labelpad=8)

        # Adjust layout to ensure both heatmaps and colorbar are aligned
        fig.align_ylabels(axes)
        fig.tight_layout(rect=[0, 0.05, 0.95, 1])  # add extra bottom margin
        plt.subplots_adjust(bottom=0.25)           # ensure x-ticks visible

        out_path_diag = os.path.join(OUT_DIR, "moe_gate_modality_region_heatmap_diagnosis.png")
        plt.savefig(out_path_diag, dpi=600)
        plt.close()
        print(f"[INFO] Saved aligned modality-region diagnosis heatmap → {out_path_diag}")

    print(f"\n[DONE] Generated interpretability plots from {len(folds)} {data_label.lower()}s.")
    print(f"[INFO] Output directory: {OUT_DIR}")
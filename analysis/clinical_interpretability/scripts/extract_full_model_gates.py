#!/usr/bin/env python3
"""
Extract gate weights from full model with all seeds.
Loads gate weights from *_full_gates.npy files and aggregates across seeds.
Outputs CSV files compatible with brain plotting in R.

Usage:
    python3 scripts/extract_full_model_gates.py --results_dir results --output results/gate_weights_for_brain_plot.csv
"""

import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path


def load_expert_config(config_path="configs/experts.yaml"):
    """Load expert configuration to get region names."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except:
        return None


def parse_expert_name(expert_name):
    """Parse expert name to extract modality and region."""
    if expert_name.startswith("amy_"):
        modality = "amyloid"
        region = expert_name[4:]  # remove "amy_" prefix
    elif expert_name.startswith("mri_"):
        modality = "mri"
        region = expert_name[4:]  # remove "mri_" prefix
    elif "demographic" in expert_name.lower():
        modality = "demographic"
        region = "demographic"
    else:
        modality = "unknown"
        region = expert_name
    return modality, region


def find_gate_files(results_dir):
    """Find all *_full_gates.npy files in results directory."""
    # Look for seed-specific gate files
    patterns = [
        f"{results_dir}/*_seed*_full_gates.npy",
        f"{results_dir}/moe_seed*_full_gates.npy",
        f"{results_dir}/*_full_gates.npy"
    ]
    
    gate_files = []
    for pattern in patterns:
        gate_files.extend(glob.glob(pattern))
    
    return sorted(set(gate_files))


def load_gate_weights_from_files(gate_files):
    """Load gate weights from .npy files."""
    gates_by_seed = {}
    
    for gate_file in gate_files:
        seed_name = Path(gate_file).stem.replace("_full_gates", "")
        try:
            gates = np.load(gate_file)  # shape: (n_samples, n_experts)
            gates_by_seed[seed_name] = gates
            print(f"  Loaded {seed_name}: shape={gates.shape}")
        except Exception as e:
            print(f"  [WARN] Failed to load {gate_file}: {e}")
    
    return gates_by_seed


def load_gate_weights_from_json(results_dir):
    """Load gate weights from JSON results files."""
    # Try to find hierarchical CV results
    json_files = glob.glob(f"{results_dir}/*hierarchical*.json") + \
                 glob.glob(f"{results_dir}/*cv*.json") + \
                 glob.glob(f"{results_dir}/moe_*.json")
    
    if not json_files:
        return None
    
    # Load the first matching JSON file
    json_path = json_files[0]
    print(f"Loading gate weights from {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract gate_mean_weights from folds
    if isinstance(data, dict) and "folds" in data:
        folds = data["folds"]
    else:
        folds = data if isinstance(data, list) else []
    
    if not folds:
        return None
    
    # Aggregate gate weights across folds
    gate_weights = {}
    for fold in folds:
        gate_mean = fold.get("gate_mean_weights", {})
        for expert, weight in gate_mean.items():
            if expert not in gate_weights:
                gate_weights[expert] = []
            gate_weights[expert].append(weight)
    
    # Compute mean across folds
    mean_weights = {expert: np.mean(weights) for expert, weights in gate_weights.items()}
    
    return mean_weights


def aggregate_gate_weights(gates_by_seed):
    """Aggregate gate weights across all seeds and samples."""
    if not gates_by_seed:
        return None, None
    
    # Stack all gates from all seeds
    all_gates = np.vstack(list(gates_by_seed.values()))
    
    # Compute mean across all samples
    global_mean = all_gates.mean(axis=0)
    global_std = all_gates.std(axis=0)
    
    # Per-seed averages
    seed_means = {seed: gates.mean(axis=0) for seed, gates in gates_by_seed.items()}
    
    return global_mean, seed_means


def create_brain_plot_csv(gate_weights, expert_names, output_path):
    """
    Create CSV file compatible with brain_plot_clean_zixuan.R.
    
    Expected format:
    - Columns: ggseg_dk, mri, amy, dk
    - ggseg_dk: region name compatible with ggseg atlas
    - mri: MRI gate weight for this region
    - amy: Amyloid gate weight for this region
    - dk: 1 if cortical (DK atlas), 0 if subcortical (ASEG atlas)
    """
    rows = []
    
    # Group weights by region
    region_weights = {}
    
    for idx, expert_name in enumerate(expert_names):
        modality, region = parse_expert_name(expert_name)
        weight = gate_weights[idx] if isinstance(gate_weights, np.ndarray) else gate_weights.get(expert_name, 0.0)
        
        if region not in region_weights:
            region_weights[region] = {"mri": 0.0, "amy": 0.0}
        
        if modality == "mri":
            region_weights[region]["mri"] = float(weight)
        elif modality == "amyloid":
            region_weights[region]["amy"] = float(weight)
    
    # Create rows
    for region, weights in region_weights.items():
        if region == "demographic":
            continue  # Skip demographic for brain plots
        
        # Determine if cortical (DK) or subcortical (ASEG)
        # Subcortical regions typically include: subcortical_, ventricles, brainstem, cerebellum
        is_subcortical = any(x in region.lower() for x in ["subcortical", "ventricle", "brainstem", "cerebellum", "corpus"])
        dk = 0 if is_subcortical else 1
        
        # Clean region name for ggseg compatibility
        ggseg_name = region.replace("_", " ").title()
        
        rows.append({
            "ggseg_dk": ggseg_name,
            "mri": weights["mri"],
            "amy": weights["amy"],
            "dk": dk
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved brain plot CSV to {output_path}")
    
    return df


def create_summary_csv(gate_weights, expert_names, output_path):
    """Create summary CSV with all expert weights."""
    rows = []
    
    for idx, expert_name in enumerate(expert_names):
        modality, region = parse_expert_name(expert_name)
        weight = gate_weights[idx] if isinstance(gate_weights, np.ndarray) else gate_weights.get(expert_name, 0.0)
        
        rows.append({
            "expert": expert_name,
            "modality": modality,
            "region": region,
            "mean_weight": float(weight)
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_weight", ascending=False)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved summary CSV to {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract gate weights from full model")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--output", default="results/gate_weights_for_brain_plot.csv", help="Output CSV path")
    parser.add_argument("--summary_output", default="results/gate_weights_summary.csv", help="Summary CSV path")
    args = parser.parse_args()
    
    print(f"[INFO] Looking for gate weight files in {args.results_dir}...")
    
    # Try to load from .npy files first
    gate_files = find_gate_files(args.results_dir)
    
    if gate_files:
        print(f"[INFO] Found {len(gate_files)} gate weight files")
        gates_by_seed = load_gate_weights_from_files(gate_files)
        
        if gates_by_seed:
            global_mean, seed_means = aggregate_gate_weights(gates_by_seed)
            
            # Infer expert names from the first seed's gate file or use indices
            first_seed = list(gates_by_seed.keys())[0]
            n_experts = gates_by_seed[first_seed].shape[1]
            expert_names = [f"expert_{i}" for i in range(n_experts)]
            
            # Try to load expert names from JSON
            json_gate_weights = load_gate_weights_from_json(args.results_dir)
            if json_gate_weights:
                expert_names = list(json_gate_weights.keys())
            
            print(f"\n[INFO] Aggregated gate weights across {len(gates_by_seed)} seeds")
            print(f"[INFO] Total samples: {sum(g.shape[0] for g in gates_by_seed.values())}")
            print(f"[INFO] Number of experts: {n_experts}")
            
            # Create output CSVs
            create_brain_plot_csv(global_mean, expert_names, args.output)
            create_summary_csv(global_mean, expert_names, args.summary_output)
            
            print("\n[DONE] Gate weight extraction complete!")
            print(f"\nTo use with brain plotting in R:")
            print(f"  1. Open brain_plot_clean_zixuan.R")
            print(f"  2. Update YOUR_ROI_DATA_ALL.csv path to: {args.output}")
            print(f"  3. Run the R script to generate brain plots")
            
    else:
        print("[WARN] No .npy gate weight files found, trying JSON files...")
        json_gate_weights = load_gate_weights_from_json(args.results_dir)
        
        if json_gate_weights:
            expert_names = list(json_gate_weights.keys())
            weights = np.array(list(json_gate_weights.values()))
            
            print(f"\n[INFO] Loaded gate weights from JSON")
            print(f"[INFO] Number of experts: {len(expert_names)}")
            
            # Create output CSVs
            create_brain_plot_csv(weights, expert_names, args.output)
            create_summary_csv(weights, expert_names, args.summary_output)
            
            print("\n[DONE] Gate weight extraction complete!")
        else:
            print("[ERROR] No gate weight data found!")
            print("Make sure you have either:")
            print("  1. *_full_gates.npy files from retrain phase, or")
            print("  2. JSON results files with gate_mean_weights")


if __name__ == "__main__":
    main()

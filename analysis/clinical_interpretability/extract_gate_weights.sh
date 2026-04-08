#!/bin/bash
################################################################################
# Extract and aggregate gate weights from all seeds after retrain
# 
# Usage: bash extract_gate_weights.sh <ablation_name> <output_dir>
# Example: bash extract_gate_weights.sh gate_region_only results/gate_analysis
#
# This script:
# 1. Finds all *_full_gates.npy files from the ablation
# 2. Loads gate weights for each seed
# 3. Computes mean/std across seeds
# 4. Generates a summary CSV with expert selection statistics
################################################################################

set -e

ABLATION_NAME=${1:-"gate_region_only"}
OUTPUT_DIR=${2:-"results/gate_analysis"}

echo "[INFO] Extracting gate weights for ablation: $ABLATION_NAME"
echo "[INFO] Output directory: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Find all gate weight files
GATE_FILES=$(find results/ablations/$ABLATION_NAME -name "*_full_gates.npy" 2>/dev/null || true)

if [ -z "$GATE_FILES" ]; then
    echo "[ERROR] No gate weight files found in results/ablations/$ABLATION_NAME"
    echo "[INFO] Looking for files matching: *_full_gates.npy"
    ls results/ablations/$ABLATION_NAME/ | head -20
    exit 1
fi

echo "[INFO] Found $(echo "$GATE_FILES" | wc -l) gate weight files"
echo "$GATE_FILES" | head -5

# Create Python script to aggregate gate weights
python3 << 'PYTHON_SCRIPT'
import numpy as np
import os
import glob
import json
from pathlib import Path

ablation_name = os.environ.get('ABLATION_NAME', 'gate_region_only')
output_dir = os.environ.get('OUTPUT_DIR', 'results/gate_analysis')

# Find all gate weight files
gate_files = sorted(glob.glob(f"results/ablations/{ablation_name}/*_full_gates.npy"))

if not gate_files:
    print(f"[ERROR] No gate weight files found in results/ablations/{ablation_name}")
    exit(1)

print(f"[INFO] Found {len(gate_files)} gate weight files")

# Load and aggregate gate weights
all_gates = []
seed_stats = {}

for gate_file in gate_files:
    try:
        seed_name = Path(gate_file).stem.replace("_full_gates", "")
        gates = np.load(gate_file)  # shape: (n_samples, n_experts)
        
        print(f"[INFO] Loaded {seed_name}: shape={gates.shape}")
        
        # Compute statistics for this seed
        mean_gates = gates.mean(axis=0)
        std_gates = gates.std(axis=0)
        active_experts = (mean_gates > 0.01).sum()  # experts with >1% mean weight
        
        seed_stats[seed_name] = {
            "n_samples": gates.shape[0],
            "n_experts": gates.shape[1],
            "mean_weights": mean_gates.tolist(),
            "std_weights": std_gates.tolist(),
            "active_experts": int(active_experts),
            "expert_entropy": float(-np.sum(mean_gates * np.log(mean_gates + 1e-10)))
        }
        
        all_gates.append(gates)
    except Exception as e:
        print(f"[WARN] Failed to load {gate_file}: {e}")
        continue

if not all_gates:
    print("[ERROR] No gate weights could be loaded")
    exit(1)

# Aggregate across seeds
print(f"\n[INFO] Aggregating {len(all_gates)} seeds...")

# Concatenate all samples from all seeds
all_gates_concat = np.vstack(all_gates)
print(f"[INFO] Total shape (all samples across all seeds): {all_gates_concat.shape}")

# Compute global statistics
global_mean = all_gates_concat.mean(axis=0)
global_std = all_gates_concat.std(axis=0)
global_active = (global_mean > 0.01).sum()
global_entropy = float(-np.sum(global_mean * np.log(global_mean + 1e-10)))

print(f"\n[RESULTS] Global Statistics:")
print(f"  Mean expert weights: {global_mean}")
print(f"  Active experts (>1% mean): {global_active}")
print(f"  Expert entropy: {global_entropy:.4f}")

# Per-seed averages (average over samples within each seed, then average over seeds)
seed_means = []
for gates in all_gates:
    seed_mean = gates.mean(axis=0)
    seed_means.append(seed_mean)
seed_means = np.array(seed_means)

mean_across_seeds = seed_means.mean(axis=0)
std_across_seeds = seed_means.std(axis=0)

print(f"\n[RESULTS] Per-Seed Averages (then averaged over {len(seed_means)} seeds):")
print(f"  Mean expert weights: {mean_across_seeds}")
print(f"  Std of seed means: {std_across_seeds}")

# Save results
os.makedirs(output_dir, exist_ok=True)

# Save aggregated gate weights
np.save(f"{output_dir}/aggregated_gates_all_samples.npy", all_gates_concat)
np.save(f"{output_dir}/seed_means.npy", seed_means)

# Save seed statistics
with open(f"{output_dir}/seed_statistics.json", "w") as f:
    json.dump(seed_stats, f, indent=2)

# Save summary
summary = {
    "ablation_name": ablation_name,
    "n_seeds": len(seed_stats),
    "total_samples": all_gates_concat.shape[0],
    "n_experts": all_gates_concat.shape[1],
    "global_mean_weights": global_mean.tolist(),
    "global_std_weights": global_std.tolist(),
    "mean_across_seeds": mean_across_seeds.tolist(),
    "std_across_seeds": std_across_seeds.tolist(),
    "active_experts_global": int(global_active),
    "global_entropy": global_entropy,
    "seed_details": seed_stats
}

with open(f"{output_dir}/gate_weights_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n[INFO] Results saved to:")
print(f"  - {output_dir}/aggregated_gates_all_samples.npy")
print(f"  - {output_dir}/seed_means.npy")
print(f"  - {output_dir}/seed_statistics.json")
print(f"  - {output_dir}/gate_weights_summary.json")

# Print detailed summary
print(f"\n" + "="*80)
print(f"GATE WEIGHT ANALYSIS SUMMARY")
print(f"="*80)
print(f"Ablation: {ablation_name}")
print(f"Seeds analyzed: {len(seed_stats)}")
print(f"Total samples: {all_gates_concat.shape[0]}")
print(f"Number of experts: {all_gates_concat.shape[1]}")
print(f"\nMean expert weights (all samples across all seeds):")
for i, w in enumerate(global_mean):
    print(f"  Expert {i}: {w:.4f} ± {global_std[i]:.4f}")
print(f"\nActive experts (>1% mean weight): {global_active}")
print(f"Expert entropy (0=single expert, high=diverse): {global_entropy:.4f}")
print(f"="*80)

PYTHON_SCRIPT

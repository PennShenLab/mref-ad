#!/usr/bin/env python3
"""
Extract and visualize gate weights across seeds after retrain.

Usage:
    python3 scripts/analyze_gate_weights.py --ablation gate_region_only --output results/gate_analysis
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import glob


def load_gate_weights(ablation_name):
    """Load gate weights from all seeds."""
    gate_files = sorted(glob.glob(f"results/ablations/{ablation_name}/*_full_gates.npy"))
    
    if not gate_files:
        raise FileNotFoundError(f"No gate weight files found in results/ablations/{ablation_name}")
    
    gates_by_seed = {}
    for gate_file in gate_files:
        seed_name = Path(gate_file).stem.replace("_full_gates", "")
        gates = np.load(gate_file)
        gates_by_seed[seed_name] = gates
        print(f"  {seed_name}: {gates.shape}")
    
    return gates_by_seed


def compute_statistics(gates_by_seed):
    """Compute statistics across seeds."""
    # Per-seed statistics
    seed_stats = {}
    for seed_name, gates in gates_by_seed.items():
        mean_gates = gates.mean(axis=0)
        std_gates = gates.std(axis=0)
        active = (mean_gates > 0.01).sum()
        entropy = float(-np.sum(mean_gates * np.log(mean_gates + 1e-10)))
        
        seed_stats[seed_name] = {
            "n_samples": int(gates.shape[0]),
            "mean_weights": mean_gates.tolist(),
            "std_weights": std_gates.tolist(),
            "active_experts": int(active),
            "entropy": entropy
        }
    
    # Global statistics across all samples
    all_gates = np.vstack(list(gates_by_seed.values()))
    global_mean = all_gates.mean(axis=0)
    global_std = all_gates.std(axis=0)
    global_active = (global_mean > 0.01).sum()
    global_entropy = float(-np.sum(global_mean * np.log(global_mean + 1e-10)))
    
    # Per-seed average, then average over seeds
    seed_means = np.array([gates.mean(axis=0) for gates in gates_by_seed.values()])
    mean_over_seeds = seed_means.mean(axis=0)
    std_over_seeds = seed_means.std(axis=0)
    
    return {
        "seed_stats": seed_stats,
        "global_mean": global_mean.tolist(),
        "global_std": global_std.tolist(),
        "global_active_experts": int(global_active),
        "global_entropy": global_entropy,
        "mean_over_seeds": mean_over_seeds.tolist(),
        "std_over_seeds": std_over_seeds.tolist(),
        "all_gates": all_gates,
    }


def save_results(output_dir, ablation_name, stats):
    """Save analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy arrays
    np.save(f"{output_dir}/aggregated_gates.npy", stats["all_gates"])
    
    # Prepare JSON summary (exclude numpy arrays)
    summary = {
        "ablation_name": ablation_name,
        "n_seeds": len(stats["seed_stats"]),
        "total_samples": stats["all_gates"].shape[0],
        "n_experts": stats["all_gates"].shape[1],
        "global_mean_weights": stats["global_mean"],
        "global_std_weights": stats["global_std"],
        "global_active_experts": stats["global_active_experts"],
        "global_entropy": stats["global_entropy"],
        "mean_over_seeds": stats["mean_over_seeds"],
        "std_over_seeds": stats["std_over_seeds"],
        "seed_details": stats["seed_stats"]
    }
    
    with open(f"{output_dir}/gate_weights_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def print_summary(summary):
    """Print formatted summary."""
    print("\n" + "="*80)
    print("GATE WEIGHT ANALYSIS SUMMARY")
    print("="*80)
    print(f"Ablation: {summary['ablation_name']}")
    print(f"Seeds analyzed: {summary['n_seeds']}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Number of experts: {summary['n_experts']}")
    
    print(f"\nMean expert weights (all samples):")
    for i, (w, s) in enumerate(zip(summary['global_mean_weights'], summary['global_std_weights'])):
        active = "✓" if w > 0.01 else "✗"
        print(f"  Expert {i:2d}: {w:.4f} ± {s:.4f}  [{active}]")
    
    print(f"\nActive experts (>1% mean weight): {summary['global_active_experts']}")
    print(f"Expert entropy (0=single, high=diverse): {summary['global_entropy']:.4f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze gate weights from retrain phase")
    parser.add_argument("--ablation", default="gate_region_only", help="Ablation name")
    parser.add_argument("--output", default="results/gate_analysis", help="Output directory")
    args = parser.parse_args()
    
    print(f"[INFO] Loading gate weights from results/ablations/{args.ablation}...")
    gates_by_seed = load_gate_weights(args.ablation)
    
    print(f"[INFO] Computing statistics...")
    stats = compute_statistics(gates_by_seed)
    
    print(f"[INFO] Saving results to {args.output}...")
    summary = save_results(args.output, args.ablation, stats)
    
    print_summary(summary)
    
    print(f"[INFO] Results saved to:")
    print(f"  - {args.output}/aggregated_gates.npy")
    print(f"  - {args.output}/gate_weights_summary.json")


if __name__ == "__main__":
    main()

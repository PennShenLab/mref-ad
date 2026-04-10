#!/usr/bin/env python3
"""
Aggregate train/test/val results across multiple seeds.

Example:
  python scripts/aggregate_train_test_val_seeds.py \
    --seed_jsons results/missingness/pet_missing_train_test_val/p0p00/seed*.json \
    --out results/missingness/pet_missing_train_test_val/baseline_lr_all_p0p00_aggregated.json

  # Aggregate all fractions for a baseline
  python scripts/aggregate_train_test_val_seeds.py \
    --seed_jsons results/missingness/pet_missing_train_test_val/p0p00/seed*.json \
    --out results/missingness/pet_missing_train_test_val/baseline_aggregated_p0p00.json
"""

import argparse
import glob
import json
import numpy as np
from typing import Dict, List, Any


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def aggregate_seeds(seed_jsons: List[str], is_moe: bool = False) -> Dict[str, Any]:
    """
    Aggregate test_metrics across multiple seed JSONs.
    
    Input format (per seed):
      - mref-ad / train_moe (flat): {"test_auc": 0.79, "test_acc": 0.64, ...}
      - Baseline (nested): {"baseline_key": {"test_metrics": {"auc": 0.79, ...}}}
    
    Output format:
      {
        "baseline_key": {
          "test_metrics": {
            "auc": {"mean": 0.79, "std": 0.02},
            "acc": {"mean": 0.64, "std": 0.03},
            "f1": {"mean": 0.61, "std": 0.01}
          },
          "n_seeds": 5
        }
      }
    """
    # Collect all results
    all_data = []
    for path in seed_jsons:
        try:
            data = read_json(path)
            all_data.append(data)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
    
    if not all_data:
        raise ValueError("No valid seed JSONs found")
    
    # Handle mref-ad (train_moe) flat structure
    if is_moe:
        print(f"[INFO] Processing mref-ad flat JSON (train_moe output)")
        
        # Collect all test metrics
        metrics_per_seed = {}
        for i, seed_data in enumerate(all_data):
            for key, value in seed_data.items():
                # Only process test_* metrics that are numeric
                if not key.startswith("test_"):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                
                # Strip "test_" prefix for cleaner metric names
                metric_name = key.replace("test_", "")
                
                if metric_name not in metrics_per_seed:
                    metrics_per_seed[metric_name] = []
                metrics_per_seed[metric_name].append(float(value))
        
        # Print individual seed values
        if metrics_per_seed:
            print(f"[INFO] Individual seed values:")
            metric_names = sorted(metrics_per_seed.keys())
            for i, file_path in enumerate(seed_jsons[:len(all_data)]):
                seed_name = file_path.split('/')[-1].replace('.json', '')
                values_str = ", ".join([f"{m}={metrics_per_seed[m][i]:.4f}" for m in metric_names if i < len(metrics_per_seed[m])])
                print(f"  {seed_name}: {values_str}")
        
        # Compute mean/std for each metric
        aggregated_metrics = {}
        for metric_name, values in metrics_per_seed.items():
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            aggregated_metrics[metric_name] = {
                "mean": mean,
                "std": std
            }
        
        # Return aggregated results in nested format for consistency
        return {
            "moe": {
                "test_metrics": aggregated_metrics,
                "n_seeds": len(all_data)
            }
        }
    
    # Handle baseline nested structure
    # Get all baseline keys from first seed
    baseline_keys = list(all_data[0].keys())
    
    result = {}
    for baseline_key in baseline_keys:
        print(f"\n[INFO] Processing baseline: {baseline_key}")
        
        # Collect test_metrics from all seeds
        metrics_per_seed = {}
        seed_file_names = []
        for i, seed_data in enumerate(all_data):
            if baseline_key not in seed_data:
                continue
            if "test_metrics" not in seed_data[baseline_key]:
                continue
            
            # Extract seed name from file path for better output
            seed_file_names.append(seed_jsons[i])
            
            test_m = seed_data[baseline_key]["test_metrics"]
            for metric_name, metric_val in test_m.items():
                if metric_name not in metrics_per_seed:
                    metrics_per_seed[metric_name] = []
                try:
                    metrics_per_seed[metric_name].append(float(metric_val))
                except (ValueError, TypeError):
                    pass
        
        # Print individual seed values
        if metrics_per_seed:
            print(f"[INFO] Individual seed values:")
            metric_names = sorted(metrics_per_seed.keys())
            for i, file_path in enumerate(seed_file_names):
                seed_name = file_path.split('/')[-1].replace('.json', '')
                values_str = ", ".join([f"{m}={metrics_per_seed[m][i]:.4f}" for m in metric_names])
                print(f"  {seed_name}: {values_str}")
        
        # Compute mean/std for each metric
        aggregated_metrics = {}
        for metric_name, values in metrics_per_seed.items():
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            aggregated_metrics[metric_name] = {
                "mean": mean,
                "std": std
            }
        
        # Store aggregated results
        result[baseline_key] = {
            "test_metrics": aggregated_metrics,
            "n_seeds": len([d for d in all_data if baseline_key in d])
        }
        
        # Also include best_params from first seed (they should all be the same)
        if "best_params" in all_data[0][baseline_key]:
            result[baseline_key]["best_params"] = all_data[0][baseline_key]["best_params"]
    
    return result


def main():
    ap = argparse.ArgumentParser(description="Aggregate train/test/val results across seeds")
    ap.add_argument("--seed_jsons", nargs="+", required=True,
                    help="Paths to seed JSON files (can use glob patterns)")
    ap.add_argument("--out", type=str, required=True,
                    help="Output aggregated JSON path")
    ap.add_argument(
        "--is_moe",
        "--is_mref_ad",
        action="store_true",
        dest="is_moe",
        help="mref-ad (train_moe) flat per-seed JSON with top-level test_* keys (vs baseline nested structure).",
    )
    args = ap.parse_args()
    
    # Expand globs
    all_files = []
    for pattern in args.seed_jsons:
        matched = glob.glob(pattern)
        all_files.extend(matched if matched else [pattern])
    
    all_files = sorted(set(all_files))
    
    if not all_files:
        raise ValueError(f"No files matched: {args.seed_jsons}")
    
    print(f"[INFO] Aggregating {len(all_files)} seed files:")
    for f in all_files:
        print(f"  - {f}")
    
    result = aggregate_seeds(all_files, is_moe=args.is_moe)
    
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"[DONE] Wrote aggregated results to: {args.out}")
    
    # Print summary
    print("\n=== AGGREGATED RESULTS ===")
    for baseline_key, data in result.items():
        print(f"\n{baseline_key} (n_seeds={data['n_seeds']}):")
        for metric_name, stats in data["test_metrics"].items():
            print(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    main()

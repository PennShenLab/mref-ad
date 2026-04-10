

#!/usr/bin/env python3
"""Missing-modality robustness experiments (CV-by-PTID or fixed train/test/val splits).

This script supports two modes for missingness experiments:

1. CV-by-PTID mode (--split_mode cv_folds):
   - Uses cross-validation folds, masking test-fold PTIDs
   - Training PTIDs remain intact, evaluation is per-fold
   - Original approach from previous experiments

2. Train/Test/Val mode (--split_mode train_test_val):
   - Uses fixed train/test/val splits (e.g., 80/10/10)
   - Loads best hyperparameters from Optuna for each seed
   - Trains on TRAIN+VAL combined with best hyperparams
   - Evaluates on TEST with masked modalities
   - Aggregates results across seeds

Typical usage (examples):

CV-by-PTID approach (original):
  python scripts/missing_modality.py \
    --split_mode cv_folds \
    --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
    --splits configs/splits/splits_by_ptid_lastvisit.json \
    --split_type cv5 \
    --mode moe \
    --train_args "--epochs 40 --batch_size 128 --num_workers 16 --use_hierarchical_gate --lambda_sparse 0.05 --tau 0.5" \
    --drop_experts "amy" \
    --fractions "0,0.2,0.4,0.6,0.8,1.0" \
    --out_dir results/missingness \
    --tag pet_missing

Train/Test/Val approach (new):
  python scripts/missing_modality.py \
    --split_mode train_test_val \
    --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
    --splits configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json \
    --mode moe \
    --seeds "7,13,42,1234,2027" \
    --optuna_results_dir results \
    --drop_experts "amy" \
    --fractions "0,0.2,0.4,0.6,0.8,1.0" \
    --out_dir results/missingness \
    --tag pet_missing_fixed_splits

Train/Test/Val with FT-Transformer baseline:
  python scripts/missing_modality.py \
    --split_mode train_test_val \
    --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
    --splits configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json \
    --mode baseline \
    --baseline ftt \
    --seeds "7,13,42,1234,2027" \
    --optuna_results_dir results \
    --drop_experts "mri" \
    --fractions "0,1.0" \
    --out_dir results/missingness \
    --tag mri_missing_fixed_splits

Train/Test/Val with MLP baseline:
  python scripts/missing_modality.py \
    --split_mode train_test_val \
    --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
    --splits configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json \
    --mode baseline \
    --baseline mlp \
    --seeds "7,13,42,1234,2027" \
    --optuna_results_dir results \
    --drop_experts "amy" \
    --fractions "0,0.2,0.4,0.6,0.8,1.0" \
    --out_dir results/missingness \
    --tag pet_missing_mlp

Train/Test/Val with Logistic Regression baseline:
  python scripts/missing_modality.py \
    --split_mode train_test_val \
    --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
    --splits configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json \
    --mode baseline \
    --baseline lr \
    --seeds "7,13,42,1234,2027" \
    --optuna_results_dir results \
    --drop_experts "amy" \
    --fractions "0,1.0" \
    --out_dir results/missingness \
    --tag pet_missing_lr

"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd



# -----------------------------
# I/O helpers
# -----------------------------

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(obj: dict, path: str) -> None:
    import yaml
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _format_splits_template(template: str, seed_id: int) -> str:
    """Fill `{seed}` in a splits path template.

    Uses ``str.replace`` instead of ``str.format`` so a path is still valid if the shell
    ever mangles ``{seed}.json`` into ``{seed.json}`` (which would make ``.format`` try
    to read a ``.json`` attribute on the seed integer).
    """
    return template.replace("{seed}", str(int(seed_id)))


def _run(cmd: List[str], log_path: str) -> None:
    """Run a command, streaming stdout/stderr to both console and log file."""
    _ensure_dir(os.path.dirname(log_path) or ".")
    with open(log_path, "w") as lf:
        lf.write(" ".join(cmd) + "\n")
        lf.flush()
        
        # Stream output to both console and log file
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='')  # Print to console
            lf.write(line)  # Write to log file
            lf.flush()
        
        proc.wait()
    
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}). See log: {log_path}")


# -----------------------------
# Split parsing
# -----------------------------

@dataclass
class Fold:
    fold_idx: int
    train_ptids: List[str]
    test_ptids: List[str]


@dataclass
class Seed:
    seed_id: int
    train_ptids: List[str]
    val_ptids: List[str]
    test_ptids: List[str]


def _iter_cv_folds(splits_obj: dict) -> List[Fold]:
    """Support common CV fold split JSON formats.

    Expected formats (examples):

    1) {"folds": [{"train_ptids": [...], "test_ptids": [...]}, ...]}
    2) [{"train_ptids": [...], "test_ptids": [...]}, ...]
    3) {"cv_splits_ptid": [{"train_ptids": [...], "val_ptids": [...]}, ...]}

    Returns list of Fold.
    """
    if isinstance(splits_obj, dict) and "cv_splits_ptid" in splits_obj:
        # Format used in your repo: top-level contains a list of fold dicts under `cv_splits_ptid`.
        # Each fold has {train_ptids: [...], val_ptids: [...]}.
        folds = splits_obj["cv_splits_ptid"]
    elif isinstance(splits_obj, dict) and "folds" in splits_obj:
        folds = splits_obj["folds"]
    elif isinstance(splits_obj, list):
        folds = splits_obj
    elif isinstance(splits_obj, dict):
        # Support dict-of-folds formats, e.g. {"0": {train_ptids:..., test_ptids:...}, "1": {...}}
        # or {"fold0": {...}, "fold1": {...}}.
        fold_items = []
        for k, v in splits_obj.items():
            if isinstance(v, dict) and ("train_ptids" in v and ("test_ptids" in v or "val_ptids" in v)):
                fold_items.append((k, v))
        if fold_items:
            def _key_to_int(key: str) -> int:
                s = str(key)
                # extract trailing/embedded integer if present
                import re
                m = re.search(r"(\d+)", s)
                return int(m.group(1)) if m else 10**9

            fold_items.sort(key=lambda kv: _key_to_int(kv[0]))
            folds = [v for _, v in fold_items]
        else:
            raise ValueError(
                "Unsupported splits JSON format. Expected list, dict with 'folds'/'cv_splits_ptid', or dict-of-folds containing train_ptids and test_ptids/val_ptids."
            )
    else:
        raise ValueError(
            "Unsupported splits JSON format. Expected list, dict with 'folds'/'cv_splits_ptid', or dict-of-folds containing train_ptids and test_ptids/val_ptids."
        )

    out: List[Fold] = []
    for i, f in enumerate(folds):
        tr = [str(x).strip() for x in f.get("train_ptids", [])]
        te_list = f.get("test_ptids", None)
        if te_list is None:
            te_list = f.get("val_ptids", [])
        te = [str(x).strip() for x in te_list]
        if not tr or not te:
            raise ValueError(f"Fold {i} missing train_ptids/test_ptids")
        out.append(Fold(fold_idx=i, train_ptids=tr, test_ptids=te))
    return out


def _iter_train_val_test_splits(splits_obj: dict, seed_id: int = 0) -> List[Seed]:
    """Parse train/val/test splits format (e.g., from splits_by_ptid_80_10_10_seed_*.json).
    
    Expected formats:
    1) {"by_seed": [{"seed": 42, "train": [...], "val": [...], "test": [...]}, ...]}
    2) {"seed_42": {"train_ptids": [...], "val_ptids": [...], "test_ptids": [...]}, ...}
    3) Flat: {"train_ptids": [...], "val_ptids": [...], "test_ptids": [...]} (single seed)
    
    For format 3, seed_id parameter is required to identify which seed this is.
    """
    splits = []
    
    # Try by_seed format first
    if isinstance(splits_obj, dict) and "by_seed" in splits_obj:
        by_seed = splits_obj["by_seed"]
        if isinstance(by_seed, list):
            for item in by_seed:
                sid = item.get("seed", 0)
                train = [str(x).strip() for x in item.get("train", [])]
                val = [str(x).strip() for x in item.get("val", [])]
                test = [str(x).strip() for x in item.get("test", [])]
                if not train or not val or not test:
                    raise ValueError(f"Seed {sid}: missing train/val/test splits")
                splits.append(Seed(seed_id=sid, train_ptids=train, val_ptids=val, test_ptids=test))
            return splits
    
    # Try flat format (single seed file)
    if "train_ptids" in splits_obj and "val_ptids" in splits_obj and "test_ptids" in splits_obj:
        train = [str(x).strip() for x in splits_obj.get("train_ptids", [])]
        val = [str(x).strip() for x in splits_obj.get("val_ptids", [])]
        test = [str(x).strip() for x in splits_obj.get("test_ptids", [])]
        if train and val and test:
            return [Seed(seed_id=seed_id, train_ptids=train, val_ptids=val, test_ptids=test)]
    
    # Try dict-of-seeds format (seed_42, seed_13, etc.)
    seed_items = []
    for k, v in splits_obj.items():
        if isinstance(v, dict):
            # Extract seed number from key (e.g., "seed_42" -> 42)
            import re
            m = re.search(r"_?seed_?(\d+)", str(k), re.IGNORECASE)
            sid = int(m.group(1)) if m else 0
            
            # Look for train/val/test or train_ptids/val_ptids/test_ptids
            train = v.get("train", v.get("train_ptids", []))
            val = v.get("val", v.get("val_ptids", []))
            test = v.get("test", v.get("test_ptids", []))
            
            train = [str(x).strip() for x in train]
            val = [str(x).strip() for x in val]
            test = [str(x).strip() for x in test]
            
            if train and val and test:
                seed_items.append((sid, Seed(seed_id=sid, train_ptids=train, val_ptids=val, test_ptids=test)))
    
    if seed_items:
        seed_items.sort(key=lambda x: x[0])
        return [s for _, s in seed_items]
    
    raise ValueError(
        "Could not parse train/val/test splits. Expected format: "
        "{'by_seed': [...]} with each item having 'train', 'val', 'test' fields, "
        "or flat format with 'train_ptids', 'val_ptids', 'test_ptids', "
        "or dict of 'seed_*' keys with train/val/test or train_ptids/val_ptids/test_ptids."
    )


def _load_best_hyperparams(optuna_json_path: str) -> Dict:
    """Load best hyperparameters from Optuna result JSON.
    
    Expected formats:
    1) mref-ad (train_moe Optuna export): {"value": <metric>, "params": {...}}
    2) Flex-MoE export: {"best_params": {...}, "best_value": ...}
    3) Baselines: {"{baseline}_concat_all": {"best_params": {...}, ...}}
    """
    try:
        data = _read_json(optuna_json_path)
        
        # Try mref-ad / train_moe Optuna format first
        if "params" in data:
            return data["params"]
        # Flex-MoE / generic best_trial JSON (top-level best_params only)
        if isinstance(data.get("best_params"), dict) and "params" not in data:
            return dict(data["best_params"])
        
        # Try baseline format: {baseline}_concat_all -> best_params
        for key, value in data.items():
            if isinstance(value, dict) and "best_params" in value:
                return value["best_params"]
        
        raise ValueError(f"No 'params' or 'best_params' key found in {optuna_json_path}")
    except Exception as e:
        raise ValueError(f"Failed to load best hyperparams from {optuna_json_path}: {e}")


def _format_hyperparams_for_cmd(params: Dict) -> List[str]:
    """Convert hyperparameter dict to command-line arguments.
    
    Example:
        {"lr": 0.001, "weight_decay": 1e-5} -> ["--lr", "0.001", "--weight_decay", "1e-05"]
    """
    cmd_args = []
    for k, v in sorted(params.items()):
        cmd_args.append(f"--{k}")
        cmd_args.append(str(v))
    return cmd_args


def _fixed_hyperparams_json_path(mode: str, baseline: str, hyperparams_dir: str) -> str:
    d = os.path.expanduser(hyperparams_dir)
    if mode == "moe":
        return os.path.join(d, "mref_ad_best_trial.json")
    if mode == "flex_moe":
        return os.path.join(d, "flex_moe_best_trial.json")
    if mode == "baseline":
        bl = (baseline or "").strip().lower()
        if bl in ("ftt", "ftt_all"):
            return os.path.join(d, "ftt_best_trial.json")
        if "mlp" in bl:
            return os.path.join(d, "mlp_best_trial.json")
        if bl in ("lr_all", "lr", "logreg"):
            return os.path.join(d, "logreg_best_trial.json")
    raise ValueError(
        f"Cannot map mode={mode!r} baseline={baseline!r} to a JSON in hyperparams_dir={d!r}. "
        "Expected baselines: ftt_all, mlp_concat, lr_all."
    )


def _flatten_flex_moe_seed_json(path: str, seed_id: int) -> None:
    """Rewrite train_flex_moe.py output to flat test_* keys for aggregate_train_test_val_seeds --is_moe."""
    data = _read_json(path)
    sid = str(int(seed_id))
    block = (data.get("seeds") or {}).get(sid)
    if not block or "test_metrics" not in block:
        raise ValueError(f"Flex-MoE output {path} missing seeds[{sid}].test_metrics")
    tm = block["test_metrics"]
    flat = {
        "test_auc": float(tm["auc"]),
        "test_acc": float(tm["acc"]),
        "test_f1": float(tm["f1"]),
    }
    with open(path, "w") as f:
        json.dump(flat, f, indent=2)


def _build_flex_moe_cmd(
    args: argparse.Namespace,
    seed_yaml: str,
    seed_id: int,
    seed_json: str,
    best_params: Dict,
    save_dir: str,
) -> List[str]:
    cmd: List[str] = [
        "python",
        "-u",
        "scripts/train_flex_moe.py",
        "--experts_config",
        seed_yaml,
        "--splits_template",
        args.splits,
        "--seeds",
        str(int(seed_id)),
        "--out_json",
        seed_json,
        "--save_dir",
        save_dir,
        "--save",
        "false",
    ]
    key_int = {
        "train_epochs",
        "warm_up_epochs",
        "batch_size",
        "hidden_dim",
        "top_k",
        "num_patches",
        "num_experts",
        "num_layers_fus",
        "num_layers_pred",
        "num_heads",
        "num_routers",
        "num_layers_enc",
        "num_workers",
    }
    key_float = {"lr", "dropout", "gate_loss_weight"}
    allowed = key_int | key_float
    for k in sorted(k for k in best_params if k in allowed):
        v = best_params[k]
        cmd.append(f"--{k}")
        if k in key_int:
            cmd.append(str(int(v)))
        elif k in key_float or isinstance(v, float):
            cmd.append(f"{float(v):.12g}")
        else:
            cmd.append(str(v))
    if args.train_args.strip():
        cmd += args.train_args.strip().split()
    return cmd


def _train_test_val_model_prefix(mode: str, baseline: str) -> str:
    if mode == "baseline":
        return baseline
    if mode == "flex_moe":
        return "flex_moe"
    return "moe"


# -----------------------------
# Masking logic
# -----------------------------

KEY_COLS = {"PTID", "SCANDATE", "VISCODE"}
LABEL_COLS = {"DIAGNOSIS", "DX", "DX_bl", "DXCHANGE", "LABEL", "y"}


def _mask_expert_csv_for_ptids(
    in_csv: str,
    out_csv: str,
    ptids_to_mask: List[str],
) -> None:
    """Copy expert CSV and set feature columns to NaN for masked PTIDs."""
    df = pd.read_csv(in_csv)
    if "PTID" not in df.columns:
        # try common alternatives
        if "PTID.x" in df.columns:
            df = df.rename(columns={"PTID.x": "PTID"})
        elif "PTID.y" in df.columns:
            df = df.rename(columns={"PTID.y": "PTID"})
    if "PTID" not in df.columns:
        raise ValueError(f"{in_csv} missing PTID column")

    df["PTID"] = df["PTID"].astype(str).str.strip()
    mask = df["PTID"].isin([str(x).strip() for x in ptids_to_mask])

    # Feature columns = everything except keys/labels
    feat_cols = [c for c in df.columns if (c not in KEY_COLS and c not in LABEL_COLS)]

    if len(feat_cols) == 0:
        # still write copy
        df.to_csv(out_csv, index=False)
        return

    df.loc[mask, feat_cols] = np.nan
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)


def _select_mask_ptids(test_ptids: List[str], frac: float, seed: int) -> List[str]:
    test_ptids = [str(x).strip() for x in test_ptids]
    n = len(test_ptids)
    k = int(round(frac * n))
    rng = np.random.RandomState(seed)
    if k <= 0:
        return []
    if k >= n:
        return test_ptids
    idx = rng.choice(np.arange(n), size=k, replace=False)
    return [test_ptids[i] for i in idx]


# -----------------------------
# Experiment runner
# -----------------------------

def main():
    import sys
    ap = argparse.ArgumentParser()

    # Core split mode choice
    ap.add_argument(
        "--split_mode",
        type=str,
        default="cv_folds",
        choices=["cv_folds", "train_test_val"],
        help="Split mode: 'cv_folds' for CV-by-PTID (original), 'train_test_val' for fixed train/test/val with best hyperparams (new).",
    )

    ap.add_argument("--experts_config", type=str, required=True)
    ap.add_argument("--splits", type=str, required=True)
    ap.add_argument("--split_type", type=str, default="cv5", choices=["cv5"], help="Only cv5 supported here.")

    ap.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["moe", "baseline", "flex_moe"],
        help="Which trainer: 'moe' -> train_moe.py; 'flex_moe' -> train_flex_moe.py; 'baseline' -> train_baselines.py",
    )

    ap.add_argument(
        "--baseline",
        type=str,
        default="ftt",
        help="Baseline name for scripts/train_baselines.py (ignored for mode=moe). Use canonical 'ftt' (legacy 'ftt_all' also accepted).",
    )

    ap.add_argument(
        "--train_args",
        type=str,
        default="",
        help="Extra args passed verbatim to the underlying training script. Example: \"--epochs 40 --batch_size 128 ...\"",
    )
    ap.add_argument(
        "--default_epochs",
        type=int,
        default=None,
        help="Fallback epochs to use when Optuna params do not include 'epochs' (train_test_val, mode=moe).",
    )

    ap.add_argument(
        "--drop_experts",
        type=str,
        required=True,
        help="Comma-separated expert keys to mask in YAML (e.g., 'mri' or 'amy,tau').",
    )

    ap.add_argument(
        "--fractions",
        type=str,
        default="0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated missingness fractions applied to test-fold PTIDs.",
    )

    # For cv_folds mode
    ap.add_argument("--seed", type=int, default=42, help="Random seed for masking selection (cv_folds mode).")
    
    # For train_test_val mode
    ap.add_argument(
        "--seeds",
        type=str,
        default="7,13,42,1234,2027",
        help="Comma-separated seed IDs for train_test_val mode (e.g., '7,13,42,1234,2027').",
    )
    ap.add_argument(
        "--optuna_results_dir",
        type=str,
        default="results",
        help="Directory containing per-seed Optuna JSONs (train_test_val). Ignored if --hyperparams_dir is set.",
    )
    ap.add_argument(
        "--hyperparams_dir",
        type=str,
        default=None,
        help=(
            "If set, load fixed best hyperparameters from this directory (same for all seeds): "
            "mref_ad_best_trial.json (mref-ad), flex_moe_best_trial.json (flex_moe), "
            "ftt_best_trial.json (ftt), mlp_best_trial.json (mlp_concat), logreg_best_trial.json (lr_all)."
        ),
    )

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True, help="Name for this experiment condition (e.g., pet_missing).")
    ap.add_argument("--cleanup", action="store_true", help="Delete intermediate fold dirs after aggregation.")
    ap.add_argument("--save_fold_checkpoints", action="store_true", help="If set, save per-fold checkpoints (and preprocessing meta) when training folds. These will be written under each fold's data dir.")
    ap.add_argument(
        "--prep_only",
        action="store_true",
        help="Only prepare per-fold masked CSVs/YAMLs (for each fraction) and exit without training or aggregation.",
    )
    ap.add_argument(
        "--prep_clean_only",
        action="store_true",
        help="Only prepare per-fold CLEAN CSVs/YAMLs (copy originals) and exit. Use this to parallelize per-fold training separately.",
    )
    # Protocol for missing modality: train_and_eval (default) or eval_only (load pretrained checkpoints)
    ap.add_argument(
        "--protocol",
        type=str,
        default="train_and_eval",
        choices=["train_and_eval", "eval_only"],
        help="train_and_eval=old behavior; eval_only=load pretrained fold checkpoints and evaluate under missingness.",
    )
    ap.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory containing pretrained per-fold checkpoints. Expected files: fold{K}.pt or fold{K}.ckpt.",
    )
    ap.add_argument(
        "--ckpt_pattern",
        type=str,
        default="fold{fold}.pt",
        help="Filename pattern under --ckpt_dir for fold checkpoints.",
    )

    args = ap.parse_args()

    if args.split_mode == "cv_folds" and args.mode == "flex_moe":
        raise SystemExit("flex_moe is only supported with --split_mode train_test_val")

    # Route to appropriate implementation
    if args.split_mode == "cv_folds":
        _run_cv_folds_mode(args)
    else:  # train_test_val
        _run_train_test_val_mode(args)


def _run_cv_folds_mode(args):
    """Original CV-by-PTID implementation."""

    cfg = _load_yaml(args.experts_config)
    exp_map: Dict[str, str] = (cfg.get("experts", {}) or {})
    if not exp_map:
        raise ValueError("experts_config YAML has no 'experts' mapping")

    raw_drop = [x.strip() for x in args.drop_experts.split(",") if x.strip()]

    # Expand shorthand names (e.g., 'amy') to all experts with that prefix (e.g., 'amy_*').
    drop_experts: List[str] = []
    missing_tokens: List[str] = []
    for tok in raw_drop:
        if tok in exp_map:
            drop_experts.append(tok)
            continue

        # prefix match: tok_*
        pref = tok + "_"
        matches = [k for k in exp_map.keys() if k.startswith(pref)]
        if matches:
            drop_experts.extend(matches)
        else:
            missing_tokens.append(tok)

    # de-duplicate while preserving order
    seen = set()
    drop_experts = [x for x in drop_experts if not (x in seen or seen.add(x))]

    if missing_tokens:
        raise ValueError(
            f"drop_expert token(s) {missing_tokens} not found as exact keys or prefixes in experts_config. "
            f"Available: {sorted(exp_map.keys())}"
        )

    print(f"[INFO] drop_experts expanded: {raw_drop} -> {drop_experts}")

    fracs = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    for f in fracs:
        if f < 0 or f > 1:
            raise ValueError("fractions must be in [0,1]")

    folds = _iter_cv_folds(_read_json(args.splits))

    out_root = os.path.join(args.out_dir, args.tag)
    _ensure_dir(out_root)

    # Underlying script
    if args.mode == "moe":
        trainer = ["python", "-u", "scripts/mref-ad/train_moe.py"]
    else:
        trainer = ["python", "-u", "scripts/train_baselines.py"]

    # For each fraction, run all folds and aggregate
    for frac in fracs:
        frac_tag = f"p{frac:.2f}".replace(".", "p")
        run_dir = os.path.join(out_root, frac_tag)
        _ensure_dir(run_dir)

        fold_out_dir = os.path.join(run_dir, "folds")
        _ensure_dir(fold_out_dir)
        log_dir = os.path.join(run_dir, "logs")
        _ensure_dir(log_dir)

        # Run each fold sequentially here (parallelization handled by your outer sh script if desired)
        for fold in folds:
            # Choose which test PTIDs to mask in this fold
            masked_ptids = _select_mask_ptids(fold.test_ptids, frac=frac, seed=args.seed + fold.fold_idx)

            # Build fold data directory and file layout
            fold_data_dir = os.path.join(fold_out_dir, f"fold{fold.fold_idx}")
            _ensure_dir(fold_data_dir)

            # If requested, first create a CLEAN per-fold YAML (no masking) and
            # run the trainer to produce per-fold checkpoints trained on the
            # unmasked train+val data. This prevents leakage from masked test
            # rows into any retraining step.
            ckpt_path = os.path.join(fold_data_dir, f"fold{fold.fold_idx}.pt")
            fold_yaml_clean = os.path.join(fold_data_dir, "experts_clean.yaml")
            if args.save_fold_checkpoints:
                new_clean_map = {}
                for name, path in exp_map.items():
                    out_csv = os.path.join(fold_data_dir, f"{name}.csv")
                    # copy original CSV (unmasked)
                    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
                    shutil.copy2(path, out_csv)
                    new_clean_map[name] = out_csv
                new_clean_cfg = dict(cfg)
                new_clean_cfg["experts"] = new_clean_map
                _write_yaml(new_clean_cfg, fold_yaml_clean)

                # If caller only wanted to prepare clean per-fold YAMLs, skip
                # running the trainer here so an outer script can parallelize
                # per-fold training. This is used by the parallel-run helper.
                if args.prep_clean_only:
                    continue

                # Run trainer on the CLEAN per-fold YAML to save checkpoint.
                trainer_clean_cmd = trainer + [
                    "--experts_config",
                    fold_yaml_clean,
                    "--splits",
                    args.splits,
                    "--split_type", "cv5",
                    "--only_fold", str(fold.fold_idx),
                ]
                if args.mode == "baseline":
                    trainer_clean_cmd += ["--baseline", args.baseline]
                    # Save checkpoint and optionally retrain on full (train+val)
                    trainer_clean_cmd += ["--save_checkpoint", ckpt_path]
                    if args.retrain_on_full:
                        trainer_clean_cmd += ["--retrain_on_full"]
                else:
                    # For MOE, allow saving a checkpoint for eval-only protocol
                    # If caller requested --save_fold_checkpoints, pass --save_checkpoint to train_moe.py
                    if args.save_fold_checkpoints:
                        trainer_clean_cmd += ["--save_checkpoint", ckpt_path]
                    trainer_clean_cmd += ["--use_hierarchical_gate"]  # Use hierarchical modality->region gating
                    trainer_clean_cmd += ["--out_json", os.path.join(fold_data_dir, f"fold{fold.fold_idx}.moe.json")]

                clean_log = os.path.join(log_dir, f"fold{fold.fold_idx}.clean.log")
                print(f"[INFO] creating clean-fold checkpoint: {ckpt_path}")
                _run(trainer_clean_cmd, clean_log)

            # Build modified (masked) expert CSVs for this fold used for evaluation
            new_exp_map = {}
            for name, path in exp_map.items():
                out_csv = os.path.join(fold_data_dir, f"{name}.csv")
                if name in drop_experts:
                    _mask_expert_csv_for_ptids(path, out_csv, masked_ptids)
                else:
                    # just copy (if save_fold_checkpoints ran above this is idempotent)
                    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
                    if not os.path.exists(out_csv):
                        shutil.copy2(path, out_csv)
                new_exp_map[name] = out_csv

            # Write per-fold YAML pointing to masked CSVs
            fold_yaml = os.path.join(fold_data_dir, "experts_masked.yaml")
            new_cfg = dict(cfg)
            new_cfg["experts"] = new_exp_map
            _write_yaml(new_cfg, fold_yaml)

            # Output JSON for this fold
            fold_json = os.path.join(run_dir, f"fold{fold.fold_idx}.json")
            fold_log = os.path.join(log_dir, f"fold{fold.fold_idx}.log")
            if args.prep_only:
                # We already wrote the per-fold masked CSVs and experts_masked.yaml.
                # Do not launch training.
                continue

        # If caller requested only clean prep, exit after preparing clean YAMLs
        if args.prep_clean_only:
            print(f"[PREP_CLEAN_ONLY] frac={frac:.2f} prepared clean fold inputs under: {run_dir}")
            continue

            # Build command
            cmd = trainer + [
                "--experts_config",
                fold_yaml,
                "--splits",
                args.splits,
            ]

            # Protocol logic: if eval_only, add --eval_only and --ckpt
            if args.protocol == "eval_only":
                # We require --ckpt_dir and --ckpt_pattern
                if not args.ckpt_dir:
                    raise ValueError("--protocol=eval_only requires --ckpt_dir")
                ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_pattern.format(fold=fold.fold_idx))
                if not os.path.isfile(ckpt_path):
                    raise ValueError(f"Checkpoint file not found for fold {fold.fold_idx}: {ckpt_path}")
                if args.mode == "moe":
                    cmd += [
                        "--split_type", "cv5", "--only_fold", str(fold.fold_idx), "--out_json", fold_json,
                        "--eval_only", "--ckpt", ckpt_path,
                        "--use_hierarchical_gate"
                    ]
                else:
                    cmd += [
                        "--split_type", "cv5", "--only_fold", str(fold.fold_idx), "--baseline", args.baseline, "--out", fold_json,
                        "--eval_only", "--ckpt", ckpt_path
                    ]
                if args.train_args.strip():
                    cmd += args.train_args.strip().split()
            else:
                # train_and_eval (default) behavior
                if args.save_fold_checkpoints:
                    # If we've already produced clean per-fold checkpoints, do
                    # eval-only on the masked inputs using the saved checkpoint
                    # to avoid retraining on masked test rows (no leakage).
                    ckpt_path = os.path.join(fold_data_dir, f"fold{fold.fold_idx}.pt")
                    if args.mode == "moe":
                        cmd += [
                            "--split_type", "cv5", "--only_fold", str(fold.fold_idx), "--out_json", fold_json,
                            "--eval_only", "--ckpt", ckpt_path,
                            "--use_hierarchical_gate"
                        ]
                    else:
                        cmd += [
                            "--split_type", "cv5", "--only_fold", str(fold.fold_idx), "--baseline", args.baseline, "--out", fold_json,
                            "--eval_only", "--ckpt", ckpt_path,
                        ]
                else:
                    if args.mode == "moe":
                        cmd += ["--split_type", "cv5", "--only_fold", str(fold.fold_idx), "--out_json", fold_json, "--use_hierarchical_gate"]
                    else:
                        cmd += ["--split_type", "cv5", "--only_fold", str(fold.fold_idx), "--baseline", args.baseline, "--out", fold_json]
                        # when not saving fold checkpoints here, allow trainer to
                        # optionally save/retrain if caller requested via train_args
                if args.train_args.strip():
                    cmd += args.train_args.strip().split()

            _run(cmd, fold_log)

        if args.prep_only:
            print(f"[PREP_ONLY] frac={frac:.2f} prepared fold inputs under: {run_dir}")
            continue
        
        # Aggregate folds into one JSON
        agg_out = os.path.join(out_root, f"{args.mode}_{args.baseline if args.mode=='baseline' else 'moe'}_{frac_tag}.json")
        agg_log = os.path.join(run_dir, "aggregate.log")

        agg_cmd = [
            "python",
            "-u",
            "scripts/aggregate_cv.py",
            "--fold_dir",
            run_dir,
            "--pattern",
            "fold*.json",
            "--out_json",
            agg_out,
            "--print_overleaf",
        ]
        _run(agg_cmd, agg_log)

        # Optional cleanup
        if args.cleanup:
            # remove per-fold jsons and fold data
            for fn in os.listdir(run_dir):
                if fn.startswith("fold") and fn.endswith(".json"):
                    try:
                        os.remove(os.path.join(run_dir, fn))
                    except Exception:
                        pass
            # keep logs by default; remove big fold data
            try:
                shutil.rmtree(fold_out_dir)
            except Exception:
                pass

        print(f"[DONE] frac={frac:.2f} aggregated -> {agg_out}")


def _run_train_test_val_mode(args):
    """Train/Test/Val implementation with best hyperparameters from Optuna."""
    
    cfg = _load_yaml(args.experts_config)
    exp_map: Dict[str, str] = (cfg.get("experts", {}) or {})
    if not exp_map:
        raise ValueError("experts_config YAML has no 'experts' mapping")

    raw_drop = [x.strip() for x in args.drop_experts.split(",") if x.strip()]

    # Expand shorthand names
    drop_experts: List[str] = []
    missing_tokens: List[str] = []
    for tok in raw_drop:
        if tok in exp_map:
            drop_experts.append(tok)
            continue

        # prefix match: tok_*
        pref = tok + "_"
        matches = [k for k in exp_map.keys() if k.startswith(pref)]
        if matches:
            drop_experts.extend(matches)
        else:
            missing_tokens.append(tok)

    seen = set()
    drop_experts = [x for x in drop_experts if not (x in seen or seen.add(x))]

    if missing_tokens:
        raise ValueError(
            f"drop_expert token(s) {missing_tokens} not found as exact keys or prefixes in experts_config. "
            f"Available: {sorted(exp_map.keys())}"
        )

    print(f"[INFO] drop_experts expanded: {raw_drop} -> {drop_experts}")

    fracs = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    for f in fracs:
        if f < 0 or f > 1:
            raise ValueError("fractions must be in [0,1]")

    # Parse seeds from --seeds argument
    seed_ids = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    
    # Build seed list by loading per-seed splits files
    # --splits should be a pattern like "configs/splits/splits_by_ptid_80_10_10_seed_{seed}.json"
    seeds_list = []
    for seed_id in seed_ids:
        splits_path = _format_splits_template(args.splits, seed_id)
        if not os.path.isfile(splits_path):
            raise FileNotFoundError(f"Splits file not found for seed {seed_id}: {splits_path}")
        
        splits_obj = _read_json(splits_path)
        parsed_seeds = _iter_train_val_test_splits(splits_obj, seed_id=seed_id)
        # Should have exactly one seed per file
        if len(parsed_seeds) != 1:
            raise ValueError(f"Expected 1 seed in {splits_path}, got {len(parsed_seeds)}")
        seeds_list.append(parsed_seeds[0])
    
    print(f"[INFO] Using seeds: {[s.seed_id for s in seeds_list]}")

    out_root = os.path.join(args.out_dir, args.tag)
    _ensure_dir(out_root)

    # Underlying script (train_test_val)
    if args.mode == "moe":
        trainer = ["python", "-u", "scripts/mref-ad/train_moe.py"]
    elif args.mode == "baseline":
        trainer = ["python", "-u", "scripts/train_baselines.py"]
    else:
        trainer = None  # flex_moe: command built per seed

    best_params_global: Dict = {}
    if args.hyperparams_dir:
        hp_path = _fixed_hyperparams_json_path(args.mode, args.baseline, args.hyperparams_dir)
        if not os.path.isfile(hp_path):
            raise FileNotFoundError(f"Hyperparameters file not found: {hp_path}")
        best_params_global = _load_best_hyperparams(hp_path)
        print(f"[INFO] Using fixed hyperparameters from {hp_path}")

    # For each fraction, run all seeds and aggregate
    for frac in fracs:
        frac_tag = f"p{frac:.2f}".replace(".", "p")
        run_dir = os.path.join(out_root, frac_tag)
        _ensure_dir(run_dir)

        seed_out_dir = os.path.join(run_dir, "seeds")
        _ensure_dir(seed_out_dir)
        log_dir = os.path.join(run_dir, "logs")
        _ensure_dir(log_dir)

        # Run each seed
        for seed_obj in seeds_list:
            seed_id = seed_obj.seed_id
            print(f"\n[INFO] Processing seed={seed_id}, frac={frac:.2f}")
            
            # Load best hyperparameters for this seed (or fixed file for all seeds)
            if args.hyperparams_dir:
                best_params = dict(best_params_global)
            else:
                if args.mode == "moe":
                    optuna_file = os.path.join(
                        args.optuna_results_dir,
                        f"optuna_moe_seed_{seed_id}.json",
                    )
                elif args.mode == "flex_moe":
                    optuna_file = os.path.join(
                        args.optuna_results_dir,
                        f"optuna_flex_moe_seed_{seed_id}.json",
                    )
                else:
                    optuna_file = os.path.join(
                        args.optuna_results_dir,
                        f"optuna_{args.baseline}_seed_{seed_id}.json",
                    )
                if not os.path.isfile(optuna_file):
                    raise FileNotFoundError(f"Optuna file not found for seed {seed_id}: {optuna_file}")
                print(f"[INFO] Loading best hyperparameters from: {optuna_file}")
                best_params = _load_best_hyperparams(optuna_file)
                if args.mode == "moe" and "epochs" not in best_params:
                    best_trial_path = os.path.join(
                        args.optuna_results_dir,
                        f"optuna_moe_seed_{seed_id}_best_trial.json",
                    )
                    if os.path.isfile(best_trial_path):
                        try:
                            best_trial = _read_json(best_trial_path)
                            trial_params = best_trial.get("params", {})
                            if "epochs" in trial_params:
                                best_params["epochs"] = int(trial_params["epochs"])
                                print(f"[INFO] Using epochs from best_trial: {best_params['epochs']}")
                        except Exception as e:
                            print(f"[WARN] Failed to read epochs from {best_trial_path}: {e}")
            print(f"[INFO] Best hyperparameters for seed {seed_id}: {best_params}")
            
            # Select PTIDs to mask for this seed/frac
            masked_ptids = _select_mask_ptids(seed_obj.test_ptids, frac=frac, seed=seed_id)
            
            # Build seed data directory
            seed_data_dir = os.path.join(seed_out_dir, f"seed{seed_id}")
            _ensure_dir(seed_data_dir)
            
            # Create masked CSVs
            new_exp_map = {}
            for name, path in exp_map.items():
                out_csv = os.path.join(seed_data_dir, f"{name}.csv")
                if name in drop_experts:
                    _mask_expert_csv_for_ptids(path, out_csv, masked_ptids)
                else:
                    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
                    if not os.path.exists(out_csv):
                        shutil.copy2(path, out_csv)
                new_exp_map[name] = out_csv
            
            # Write seed YAML with masked CSVs
            seed_yaml = os.path.join(seed_data_dir, "experts_masked.yaml")
            new_cfg = dict(cfg)
            new_cfg["experts"] = new_exp_map
            _write_yaml(new_cfg, seed_yaml)
            
            # Construct the seed-specific splits file path
            seed_splits_path = _format_splits_template(args.splits, seed_id)
            
            # Output JSON for this seed
            model_prefix = _train_test_val_model_prefix(args.mode, args.baseline)
            seed_json = os.path.join(run_dir, f"{model_prefix}_seed{seed_id}.json")
            seed_log = os.path.join(log_dir, f"seed{seed_id}.log")
            
            # Build training command
            if args.mode == "flex_moe":
                flex_ckpt_dir = os.path.join(seed_data_dir, "flex_moe_ckpt")
                cmd = _build_flex_moe_cmd(
                    args, seed_yaml, seed_id, seed_json, best_params, flex_ckpt_dir
                )
            else:
                cmd = list(trainer) + [
                    "--experts_config",
                    seed_yaml,
                    "--splits",
                    seed_splits_path,
                    "--split_type",
                    "train_val_test",
                ]
                if args.mode == "baseline":
                    cmd.append("--skip_retrain")
                    cmd += ["--baseline", args.baseline]
                    cmd += ["--out", seed_json]
                    retrain_params_dict = {args.baseline: best_params}
                    cmd += ["--retrain_params", json.dumps(retrain_params_dict)]
                else:
                    cmd += ["--out_json", seed_json]
                    cmd.append("--retrain_only")
                    cmd.append("--retrain_on_full")
                    cmd.append("--no_early_stopping")
                    if "epochs" in best_params:
                        cmd += ["--epochs", str(int(best_params["epochs"]))]
                    elif args.default_epochs is not None:
                        cmd += ["--epochs", str(int(args.default_epochs))]
                    if "batch_size" in best_params:
                        cmd += ["--batch_size", str(int(best_params["batch_size"]))]
                    if "lr" in best_params:
                        cmd += ["--lr", f"{best_params['lr']:.6g}"]
                    if "wd" in best_params:
                        cmd += ["--wd", f"{best_params['wd']:.6g}"]
                    if "hidden_exp" in best_params:
                        cmd += ["--hidden_exp", str(int(best_params["hidden_exp"]))]
                    if "hidden_gate" in best_params:
                        cmd += ["--hidden_gate", str(int(best_params["hidden_gate"]))]
                    if "drop" in best_params:
                        cmd += ["--drop", f"{best_params['drop']:.4f}"]
                    if "lambda_sparse" in best_params:
                        cmd += ["--lambda_sparse", f"{best_params['lambda_sparse']:.4f}"]
                    if "lambda_diverse" in best_params:
                        cmd += ["--lambda_diverse", f"{best_params['lambda_diverse']:.5f}"]
                    if "tau" in best_params:
                        cmd += ["--tau", f"{best_params['tau']:.4f}"]
                    if best_params.get("tau_start") is not None:
                        cmd += ["--tau_start", f"{best_params['tau_start']:.4f}"]
                    if best_params.get("tau_decay") is not None:
                        cmd += ["--tau_decay", f"{best_params['tau_decay']:.4f}"]
                    if best_params.get("gate_noise") is not None:
                        cmd += ["--gate_noise", f"{best_params['gate_noise']:.4f}"]
                    if bool(best_params.get("gumbel_hard", False)):
                        cmd.append("--gumbel_hard")
                if args.train_args.strip():
                    cmd += args.train_args.strip().split()
            
            print(f"[INFO] Running: {' '.join(cmd)}")
            _run(cmd, seed_log)
            if args.mode == "flex_moe":
                _flatten_flex_moe_seed_json(seed_json, seed_id)
        
        # Aggregate seeds (mean/std) via aggregate_train_test_val_seeds.py
        print(f"\n[INFO] Aggregating seed results for frac={frac:.2f}")
        model_prefix = _train_test_val_model_prefix(args.mode, args.baseline)
        if args.mode == "baseline":
            agg_name = f"baseline_{args.baseline}_{frac_tag}_aggregated.json"
            agg_is_moe: List[str] = []
        elif args.mode == "flex_moe":
            agg_name = f"flex_moe_{frac_tag}_aggregated.json"
            agg_is_moe = ["--is_moe"]
        else:
            agg_name = f"moe_moe_{frac_tag}_aggregated.json"
            agg_is_moe = ["--is_moe"]
        agg_out = os.path.join(out_root, agg_name)

        seed_jsons = []
        for seed_obj in seeds_list:
            sj = os.path.join(run_dir, f"{model_prefix}_seed{seed_obj.seed_id}.json")
            if os.path.isfile(sj):
                seed_jsons.append(sj)

        if not seed_jsons:
            print(f"[WARN] No seed JSONs found for aggregation at frac={frac:.2f}")
            continue

        agg_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aggregate_train_test_val_seeds.py")
        cmd_agg = [sys.executable, "-u", agg_py, "--seed_jsons"] + seed_jsons + ["--out", agg_out] + agg_is_moe
        print(f"[INFO] {' '.join(cmd_agg)}")
        subprocess.run(cmd_agg, check=True)

        print(f"[DONE] frac={frac:.2f} aggregated -> {agg_out}")
        
        # Optional cleanup
        if args.cleanup:
            try:
                shutil.rmtree(seed_out_dir)
            except Exception:
                pass


if __name__ == "__main__":
    main()
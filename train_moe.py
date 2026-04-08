#!/usr/bin/env python3
"""
python scripts/train_moe.py \
  --experts_config configs/experts_files.yaml \
  --splits configs/splits_by_ptid.json \
  --epochs 50 \
  --batch_size 128

# With freesurfer experts:
python scripts/train_moe.py \
  --experts_config configs/freesurfer_experts_files.yaml \
  --splits configs/splits_by_ptid.json \
  --epochs 50 \
  --batch_size 128

# 5-fold CV:
python scripts/train_moe.py \
  --experts_config configs/freesurfer_experts_files.yaml \
  --splits configs/splits_by_ptid.json \
  --split_type cv5 \
  --epochs 40 \
  --batch_size 128 \
  --lr 1e-3

# 5-fold CV with last visit:
python scripts/train_moe.py \
  --experts_config configs/freesurfer_lastvisit_experts_files.yaml \
  --splits configs/splits_by_ptid_lastvisit.json \
  --split_type cv5 \
  --epochs 40 \
  --batch_size 128 \
  --out_json results/moe_lastvisit_cv_folds

# 10-fold CV with last visit:
python scripts/train_moe.py \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml \
  --splits configs/splits_by_ptid_lastvisit_cv10.json \
  --split_type cv5 \
  --epochs 40 \
  --batch_size 128 \
  --out_json results/moe_lastvisit_cv10_folds

python scripts/train_moe.py \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml \
  --splits configs/splits_by_ptid_lastvisit_cv10.json \
  --split_type cv5 \
  --epochs 40 \
  --batch_size 128 \
  --num_workers 16 \
  --use_hierarchical_gate \
  --lambda_sparse 0.05 \
  --tau 0.5 \
  --save_payloads \
  --redact_ptid \
  --out_json results/moe_hierarchical_cv10_isbi.json

  python scripts/train_moe.py \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml \
  --splits configs/splits_by_ptid_lastvisit_cv10.json \
  --split_type cv5 \
  --epochs 40 \
  --batch_size 128 \
  --num_workers 16 \
  --use_hierarchical_gate \
  --lambda_sparse 0.05 \
  --tau 0.5 \
  --save_payloads \
  --redact_ptid \
  --out_json results/moe_hierarchical_cv10_isbi.json

python scripts/train_moe.py \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml \
  --splits configs/splits_by_ptid_lastvisit_cv10.json \
  --split_type cv5 \
  --epochs 40 \
  --batch_size 128 \
  --num_workers 16 \
  --use_hierarchical_gate \
  --lambda_sparse 0.05 \
  --tau 0.5 \
  --save_payloads \
  --redact_ptid \
  --save_val_predictions \
  --out_json results/moe_hierarchical_cv10_ichi.json

# Top-k ablation
# Outputs: 
# - results/moe_hierarchical_cv10_top5.json
# - results/moe_hierarchical_cv10_top3.json
# - results/moe_hierarchical_cv10_top1.json
python scripts/train_moe.py \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml \
  --splits configs/splits_by_ptid_lastvisit_cv10.json \
  --split_type cv5 \
  --epochs 40 \
  --batch_size 128 \
  --use_hierarchical_gate \
  --topk \
  --out_json results/moe_hierarchical_cv10 \
  --lambda_sparse 0.1 \
  --tau 0.5

# Modality ablation with last visit:
python scripts/train_moe.py \
  --experts_config configs/freesurfer_lastvisit_no_amyloid.yaml \
  --splits configs/splits_by_ptid_lastvisit.json \
  --split_type cv5 \
  --epochs 40 \
  --batch_size 128 \
  --lambda_sparse 0.1 \
  --tau 0.5 \
  --use_hierarchical_gate \
  --out_json results/moe_lastvisit_cv5_folds_no_amyloid

# Running test evaluation based on best hyperparameters from optuna
python3 scripts/train_moe.py \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml \
  --splits configs/splits_by_ptid_80_10_10.json \
  --split_type train_val_test \
  --retrain_only \
  --retrain_on_full \
  --epochs 5 \
  --batch_size 16 \
  --lr 0.002673083048511336 \
  --wd 8.600884384364035e-06 \
  --hidden_exp 87 \
  --hidden_gate 217 \
  --drop 0.3175776178409976 \
  --lambda_sparse 0.12427911408817603 \
  --lambda_diverse 0.005089673911370669 \
  --tau 0.7304794259384962 \
  --tau_start 0.7131291558617883 \
  --tau_decay 0.9827623802068692 \
  --gate_noise 0.009590709637932598 \
  --gumbel_hard \
  --num_workers 16 \
  --save_checkpoint results/optuna_moe_trial0136_retrain_full.pt \
  --out_json results/optuna_moe_trial0136_retrain_full.json
"""
import argparse, os, re, json, math, random, hashlib, sys
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

import utils
from utils import (
    set_seed, load_experts_from_yaml, eval_multiclass_metrics, eval_confusion_report, load_splits
)


def _setup_device():
    """Return torch device and print info (small helper to keep run_once compact)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            name = torch.cuda.get_device_name(device)
        except Exception:
            name = "cuda"
        print(f"[INFO] device = {device} ({name})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] device = mps (Apple Silicon GPU backend)")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA/MPS not available, using CPU.")
    return device


def _collate(batch):
    """Reusable collate function for DataLoaders."""
    M = len(batch[0][0])
    xs = [torch.stack([b[0][m] for b in batch], dim=0) for m in range(M)]
    masks = torch.stack([b[1] for b in batch], dim=0)
    y = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return xs, masks, y


def _build_dataloaders(df, groups, train_idx, val_idx, params, device):
    """Create train/val datasets and DataLoaders. Returns (ds_train, ds_val, train_loader, val_loader, B, num_workers, pin_mem)."""
    if train_idx is None or val_idx is None:
        groupsplit = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=utils.SEED)
        ptids = df["PTID"].astype(str).values
        (train_idx, val_idx), = groupsplit.split(df, groups=ptids)
        print(f"[INFO] using random GroupShuffleSplit: train={len(train_idx)} | val={len(val_idx)}")
    else:
        print(f"[INFO] using fixed splits: train={len(train_idx)} | val={len(val_idx)}")

    # Build datasets depending on ablation
    if params.get("gate_ablation") == "modality_only":
        for prefix in ["mri", "amy", "demographic"]:
            region_cols = [c for c in df.columns if c.startswith(f"has_{prefix}_")]
            if region_cols:
                df[f"has_{prefix}"] = df[region_cols].max(axis=1)
        groups_modality = {}
        for prefix in ["mri", "amy", "demographic"]:
            matched = [cols for k, cols in groups.items() if k.startswith(prefix)]
            if matched:
                groups_modality[prefix] = sum(matched, [])
        ds_train = MoEDataset(df.iloc[train_idx], groups_modality)
        ds_val = MoEDataset(df.iloc[val_idx], groups_modality, scalers=ds_train.scalers)
        groups_local = groups_modality
    else:
        ds_train = MoEDataset(df.iloc[train_idx], groups)
        ds_val = MoEDataset(df.iloc[val_idx], groups, scalers=ds_train.scalers)
        groups_local = groups

    # reuse top-level collate
    collate = _collate

    B = params["batch_size"]
    num_workers = int(params.get("num_workers", 16))
    g = torch.Generator(device="cpu").manual_seed(utils.SEED)
    pin_mem = device.type == "cuda"

    train_loader = DataLoader(
        ds_train,
        batch_size=B,
        shuffle=True,
        collate_fn=collate,
        generator=g,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=B,
        shuffle=False,
        collate_fn=collate,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )

    return ds_train, ds_val, train_loader, val_loader, B, num_workers, pin_mem


def _init_optim_and_criterion(model, params, df, train_idx, device):
    """Initialize loss, optimizer and scheduler."""
    class_counts = np.bincount(df.iloc[train_idx]["y"].values, minlength=3) + 1
    weights = torch.tensor(class_counts.sum() / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, min_lr=1e-5)
    return criterion, opt, scheduler


def _to_np_safe(x):
    """Convert tensors or iterables of tensors to numpy, guarding failures.

    Returns None if conversion isn't possible.
    """
    try:
        # list/tuple of tensors
        if isinstance(x, (list, tuple)):
            out = []
            for e in x:
                try:
                    if hasattr(e, "detach"):
                        out.append(e.detach().cpu().numpy())
                    else:
                        out.append(np.asarray(e))
                except Exception:
                    out.append(None)
            return out
        # single tensor
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    except Exception:
        return None


def _save_debug_dir(base_dir, data: dict):
    """Save a dictionary of tensors/arrays to a directory as per-key .npy files

    Also write a manifest.json describing saved files. Values that are lists
    are saved as multiple numbered files: key_0.npy, key_1.npy, ...
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        manifest = {"files": {}}
        for k, v in data.items():
            if v is None:
                manifest["files"][k] = None
                continue
            # lists/tuples -> multiple files
            if isinstance(v, (list, tuple)):
                manifest["files"][k] = []
                for i, el in enumerate(v):
                    fn = f"{k}_{i}.npy"
                    path = os.path.join(base_dir, fn)
                    try:
                        arr = _to_np_safe(el)
                        if arr is None:
                            # write an empty placeholder
                            open(path + ".empty", "w").close()
                            manifest["files"][k].append({"file": fn, "shape": None})
                        else:
                            np.save(path, arr)
                            manifest["files"][k].append({"file": fn, "shape": getattr(arr, "shape", None)})
                    except Exception:
                        manifest["files"][k].append({"file": fn, "shape": None})
            else:
                fn = f"{k}.npy"
                path = os.path.join(base_dir, fn)
                try:
                    arr = _to_np_safe(v)
                    if arr is None:
                        open(path + ".empty", "w").close()
                        manifest["files"][k] = {"file": None, "shape": None}
                    else:
                        np.save(path, arr)
                        manifest["files"][k] = {"file": fn, "shape": getattr(arr, "shape", None)}
                except Exception:
                    manifest["files"][k] = {"file": None, "shape": None}
        # write manifest
        with open(os.path.join(base_dir, "manifest.json"), "w") as mf:
            json.dump(manifest, mf, indent=2)
        return True
    except Exception as e:
        try:
            print(f"[WARN] _save_debug_dir failed to write {base_dir}: {e}")
        except Exception:
            pass
        return False


def _prepare_batch_for_dump(xs=None, masks=None, y=None, logits=None, gate=None):
    """Prepare batch tensors for debug dumping by converting to numpy where possible."""
    return {
        "xs": _to_np_safe(xs),
        "masks": _to_np_safe(masks),
        "y": _to_np_safe(y),
        "logits": _to_np_safe(logits),
        "gate": _to_np_safe(gate),
    }


def _dump_debug_and_raise(base_dir: str, dump_dict: dict, message: str, raise_exc: bool = True):
    """Write debug artifacts to `base_dir` and optionally raise a RuntimeError.

    If writing fails, emit a warning but still raise if raise_exc is True.
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        try:
            _save_debug_dir(base_dir, dump_dict)
            if raise_exc:
                print(f"[FATAL] {message}; dumped to {base_dir}")
            else:
                print(f"[WARN] {message}; dumped to {base_dir}")
        except Exception as _e:
            print(f"[WARN] failed to write debug dir {base_dir}: {_e}")
    except Exception as _e:
        print(f"[WARN] could not create debug dir {base_dir}: {_e}")
    if raise_exc:
        raise RuntimeError(f"{message}; see {base_dir}")


def _build_checkpoint_meta(ds_train, groups, params):
    """Construct metadata dict to accompany a saved checkpoint.

    This mirrors the previous inline logic but centralizes it for reuse.
    """
    meta = {}
    try:
        meta["scalers"] = ds_train.scalers
    except Exception:
        meta["scalers"] = None
    try:
        meta["groups"] = groups
    except Exception:
        meta["groups"] = None
    meta["model_config"] = {
        "hidden_exp": params.get("hidden_exp", None),
        "hidden_gate": params.get("hidden_gate", None),
        "drop": params.get("drop", None),
        "gate_type": params.get("gate_type", None),
        "gumbel_hard": params.get("gumbel_hard", False),
        "gate_noise": params.get("gate_noise", None),
        "topk": params.get("topk", None),
        "use_hierarchical_gate": params.get("use_hierarchical_gate", False),
    }
    try:
        meta["train_params"] = {
            "tau": params.get("tau", None),
            "tau_start": params.get("tau_start", None),
            "tau_decay": params.get("tau_decay", None),
            "epochs": params.get("epochs", None),
        }
    except Exception:
        pass
    return meta


def _save_checkpoint_and_meta(ckpt_path, state_to_save, ds_train, groups, params):
    """Save torch checkpoint and write accompanying .meta.pkl file.

    Returns (ckpt_path, meta_path) on success, raises on failure.
    """
    import pickle

    # ensure directory exists
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)

    torch.save(state_to_save, ckpt_path)

    meta = _build_checkpoint_meta(ds_train, groups, params)

    meta_path = os.path.splitext(ckpt_path)[0] + ".meta.pkl"
    with open(meta_path, "wb") as mf:
        pickle.dump(meta, mf)

    print(f"[INFO] saved checkpoint to {ckpt_path} and meta to {meta_path}")
    return ckpt_path, meta_path


def _prepare_splits(args, df):
    """Normalize and return splits according to --split_type.

    Returns a dict containing at least a 'type' key and per-type entries:
      - for 'holdout': {'type':'holdout','train_pos', 'val_pos', 'test_pos'(opt)}
      - for 'train_val_test': {'type':'train_val_test','tr_idx','va_idx','te_idx'}
      - for 'cv5': {'type':'cv5','cv_splits'}

    This centralizes load_splits/json parsing and allows callers to keep
    `main` compact.
    """
    st = getattr(args, "split_type", "holdout")
    if st == "holdout":
        splits_ret = load_splits(args.splits, df)
        if isinstance(splits_ret, tuple) and len(splits_ret) == 3:
            train_pos, val_pos, test_pos = splits_ret
        elif isinstance(splits_ret, tuple) and len(splits_ret) == 2:
            train_pos, val_pos = splits_ret
            test_pos = None
        else:
            raise ValueError("Unexpected splits format returned from load_splits for holdout")
        return {"type": "holdout", "train_pos": train_pos, "val_pos": val_pos, "test_pos": test_pos}

    if st == "train_val_test":
        splits_ret = load_splits(args.splits, df)
        if isinstance(splits_ret, tuple) and len(splits_ret) == 3:
            tr_idx, va_idx, te_idx = splits_ret
            return {"type": "train_val_test", "tr_idx": tr_idx, "va_idx": va_idx, "te_idx": te_idx}
        elif isinstance(splits_ret, tuple) and len(splits_ret) == 2:
            raise ValueError("Splits JSON appears to be 2-way; train_val_test requires a 3-way split (train/val/test).")
        else:
            raise ValueError("Unexpected splits format returned from load_splits for train_val_test")

    if st == "cv5":
        with open(args.splits, "r") as f:
            splits_json = json.load(f)
        if "cv_splits_ptid" not in splits_json:
            raise ValueError("splits JSON must contain 'cv_splits_ptid' for --split_type=cv5")
        return {"type": "cv5", "cv_splits": splits_json["cv_splits_ptid"]}

    raise ValueError(f"Unknown split_type: {st}")


def _dispatch_mode(args, df, groups, classes):
    """Central dispatcher that implements the runtime branches currently
    expressed inline in `__main__`.

    This keeps `__main__` concise: we call this function and then exit so the
    original inline blocks remain in the file as a safe backup.
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/runs", exist_ok=True)

    # Eval-only: evaluate a provided checkpoint on a single CV fold
    if args.eval_only:
        if args.ckpt is None:
            raise ValueError("--eval_only requires --ckpt to be set to a checkpoint file path")
        if args.only_fold is None:
            raise ValueError("--eval_only requires --only_fold to specify which fold to evaluate")

        with open(args.splits, "r") as f:
            splits_json = json.load(f)
        if "cv_splits_ptid" not in splits_json:
            raise ValueError("splits JSON must contain 'cv_splits_ptid' for --split_type=cv5")
        cv_splits = splits_json["cv_splits_ptid"]
        fold_idx = args.only_fold
        split = cv_splits[fold_idx]
        train_keys = [kk for kk in split.keys() if "train" in kk.lower()]
        val_keys = [kk for kk in split.keys() if "val" in kk.lower()]
        if not val_keys:
            raise KeyError(f"Fold {fold_idx}: missing val keys in split: {list(split.keys())}")

        val_ptids = set(split[val_keys[0]])
        val_idx = df[df["PTID"].astype(str).isin(val_ptids)].index.tolist()

        # Load checkpoint meta if present
        meta_path = os.path.splitext(args.ckpt)[0] + ".meta.pkl"
        meta = {}
        if os.path.isfile(meta_path):
            try:
                import pickle

                with open(meta_path, "rb") as mf:
                    meta = pickle.load(mf)
            except Exception as e:
                print(f"[WARN] could not load meta file {meta_path}: {e}")

        scalers = meta.get("scalers", None)

        # choose tau (CLI override wins)
        try:
            cli_specified_tau = any([a == "--tau" or a.startswith("--tau=") for a in sys.argv])
        except Exception:
            cli_specified_tau = False
        tau_to_use = args.tau if cli_specified_tau else meta.get("train_params", {}).get("tau", args.tau)

        # Build dataset and evaluate
        ds_val = MoEDataset(df.iloc[val_idx], groups, scalers=scalers)
        B = args.batch_size
        num_workers = int(args.num_workers)
        pin_mem = torch.cuda.is_available()

        def _collate_local(batch):
            M = len(batch[0][0])
            xs = [torch.stack([b[0][m] for b in batch], dim=0) for m in range(M)]
            masks = torch.stack([b[1] for b in batch], dim=0)
            y = torch.tensor([b[2] for b in batch], dtype=torch.long)
            return xs, masks, y

        val_loader = DataLoader(ds_val, batch_size=B, shuffle=False, collate_fn=_collate_local, pin_memory=pin_mem, num_workers=num_workers)

        # Reconstruct model from meta if available
        model_cfg = meta.get("model_config", {})
        dims = [len(v) for v in groups.values()]
        use_hier = model_cfg.get("use_hierarchical_gate", False) or args.use_hierarchical_gate
        device = torch.device("cpu")
        if use_hier:
            model = HierarchicalMoE(
                groups,
                hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                n_classes=3,
                drop=model_cfg.get("drop", args.drop),
                gate_type=model_cfg.get("gate_type", args.gate_type),
                gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                topk=model_cfg.get("topk", args.topk),
            ).to(device)
            print(f"[DEBUG] Created HierarchicalMoE with topk={model.gate.topk}")
        else:
            model = MoE(
                dims,
                hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                n_classes=3,
                drop=model_cfg.get("drop", args.drop),
                gate_type=model_cfg.get("gate_type", args.gate_type),
                gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                topk=model_cfg.get("topk", args.topk),
            ).to(device)
            print(f"[DEBUG] Created MoE with topk={model.gate.topk}")

        if not os.path.isfile(args.ckpt):
            raise ValueError(f"Checkpoint not found: {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        all_p, all_y = [], []
        with torch.no_grad():
            for xs, masks, y in val_loader:
                xs = [x for x in xs]
                logits, gate_w = model([x for x in xs], masks, tau=tau_to_use)
                proba = F.softmax(logits, dim=1).cpu().numpy()
                all_p.append(proba)
                all_y.append(y.numpy())

        if all_p:
            all_p = np.vstack(all_p)
            all_y = np.concatenate(all_y)
        else:
            all_p = np.zeros((0, 3))
            all_y = np.array([])

        m = utils.eval_multiclass_metrics(all_y, all_p)
        out = {
            "val_auc": float(m.get("auc", float("nan"))),
            "val_acc": float(m.get("acc", float("nan"))),
            "val_bacc": float(m.get("bacc", float("nan"))),
            "val_f1": float(m.get("f1", float("nan"))),
            "proba": all_p.tolist(),
            "yva": all_y.tolist(),
        }

        out_path = args.out_json or f"results/moe_eval_fold{args.only_fold}.json"
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] wrote eval-only results to {out_path}")
        return

    # Evaluate a provided checkpoint on the test split (no training)
    if args.eval_ckpt:
        if args.ckpt is None:
            raise ValueError("--eval_ckpt requires --ckpt to be set to a checkpoint file path")
        if args.split_type != "train_val_test":
            raise ValueError("--eval_ckpt currently supports only --split_type train_val_test")

        splits = _prepare_splits(args, df)
        if splits.get("type") != "train_val_test":
            raise ValueError("--eval_ckpt requires --split_type=train_val_test and a 3-way splits JSON")
        tr_idx, va_idx, te_idx = splits["tr_idx"], splits["va_idx"], splits["te_idx"]

        # Load meta and evaluate on test
        meta = {}
        meta_path = os.path.splitext(args.ckpt)[0] + ".meta.pkl"
        if os.path.isfile(meta_path):
            try:
                import pickle

                with open(meta_path, "rb") as mf:
                    meta = pickle.load(mf)
            except Exception as e:
                print(f"[WARN] could not load meta file {meta_path}: {e}")

        scalers = meta.get("scalers", None)
        model_cfg = meta.get("model_config", {})

        ds_test = MoEDataset(df.iloc[te_idx], meta.get("groups", groups), scalers=scalers)
        B = args.batch_size
        num_workers = int(args.num_workers)
        pin_mem = torch.cuda.is_available()

        def _collate_local(batch):
            M = len(batch[0][0])
            xs = [torch.stack([b[0][m] for b in batch], dim=0) for m in range(M)]
            masks = torch.stack([b[1] for b in batch], dim=0)
            y = torch.tensor([b[2] for b in batch], dtype=torch.long)
            return xs, masks, y

        test_loader = DataLoader(ds_test, batch_size=B, shuffle=False, collate_fn=_collate_local, pin_memory=pin_mem, num_workers=num_workers)

        dims = [len(v) for v in meta.get("groups", groups).values()]
        use_hier = model_cfg.get("use_hierarchical_gate", False) or args.use_hierarchical_gate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_hier:
            model = HierarchicalMoE(
                meta.get("groups", groups),
                hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                n_classes=3,
                drop=model_cfg.get("drop", args.drop),
                gate_type=model_cfg.get("gate_type", args.gate_type),
                gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                topk=model_cfg.get("topk", args.topk),
            ).to(device)
        else:
            model = MoE(
                dims,
                hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                n_classes=3,
                drop=model_cfg.get("drop", args.drop),
                gate_type=model_cfg.get("gate_type", args.gate_type),
                gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                topk=model_cfg.get("topk", args.topk),
            ).to(device)

        if not os.path.isfile(args.ckpt):
            raise ValueError(f"Checkpoint not found: {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        try:
            cli_specified_tau = any([a == "--tau" or a.startswith("--tau=") for a in sys.argv])
        except Exception:
            cli_specified_tau = False
        tau_to_use = args.tau if cli_specified_tau else meta.get("train_params", {}).get("tau", args.tau)

        all_p, all_y = [], []
        with torch.no_grad():
            for xs, masks, y in test_loader:
                xs = [x.to(device) for x in xs]
                masks = masks.to(device)
                logits, gate_w = model(xs, masks, tau=tau_to_use)
                proba = F.softmax(logits, dim=1).cpu().numpy()
                all_p.append(proba)
                all_y.append(y.numpy())

        if all_p:
            all_p = np.vstack(all_p)
            all_y = np.concatenate(all_y)
        else:
            all_p = np.zeros((0, 3))
            all_y = np.array([])

        mtest = utils.eval_multiclass_metrics(all_y, all_p)
        out = {
            "test_auc": float(mtest.get("auc", float("nan"))),
            "test_acc": float(mtest.get("acc", float("nan"))),
            "test_bacc": float(mtest.get("bacc", float("nan"))),
            "test_f1": float(mtest.get("f1", float("nan"))),
            "proba": all_p.tolist(),
            "yte": all_y.tolist(),
        }

        out_path = args.out_json or f"results/moe_eval_ckpt.json"
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] wrote eval-ckpt results to {out_path}")
        return

    # Holdout / simple train/val run
    if args.split_type == "holdout":
        splits = _prepare_splits(args, df)
        if splits.get("type") != "holdout":
            raise ValueError("Expected holdout splits for split_type=holdout")
        train_pos, val_pos, _test_pos = splits["train_pos"], splits["val_pos"], splits.get("test_pos", None)

        if args.topk is not None:
            # If topk is specified, run with that value only
            k = args.topk
            print(f"\n[INFO] Running with topk={k}...")
            set_seed()
            params = dict(vars(args))
            params["topk"] = k
            out = run_once(
                df,
                groups,
                params,
                train_idx=train_pos,
                val_idx=val_pos,
                gating_fn=lambda gw, m, top_k=k: apply_topk_gating(gw, m, top_k=top_k),
            )
            print(f"[INFO] best val macro-AUROC (topk={k}):", out["val_auc"])
            out_path = args.out_json or f"results/moe_results_topk{k}.json"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[INFO] wrote results to {out_path}")
        else:
            # No topk: run with full gating
            params = dict(vars(args))
            params["topk"] = None
            out = run_once(df, groups, params, train_idx=train_pos, val_idx=val_pos)
            print("[INFO] best val macro-AUROC:", out["val_auc"])
            out_path = args.out_json or "results/moe_results_lastvisit.json"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[INFO] wrote results to {out_path}")
        return

    # train_val_test: tuner / retrain-on-full workflow
    if args.split_type == "train_val_test":
        splits = _prepare_splits(args, df)
        if splits.get("type") != "train_val_test":
            raise ValueError("train_val_test requires a 3-way splits JSON (train/val/test)")
        tr_idx, va_idx, te_idx = splits["tr_idx"], splits["va_idx"], splits["te_idx"]

        params = dict(vars(args))
        if getattr(args, "no_early_stopping", False):
            params["no_early_stopping"] = True
            # Keep patience effectively infinite when early stopping is disabled.
            params["patience"] = int(1e9)
        # params["topk"] = None  # <-- REMOVED: preserve topk from command-line args
        base_name = os.path.splitext(os.path.basename(args.out_json or "results/optuna_moe"))[0]
        ckpt_best = os.path.join("results", f"{base_name}_best.pt")
        if not params.get("save_checkpoint"):
            params["save_checkpoint"] = ckpt_best

        if getattr(args, "retrain_only", False):
            print("[INFO] --retrain_only set: skipping initial train->val (tune phase)")
            out_val = {}
        else:
            print(f"[INFO] Running train->val and saving best checkpoint to {ckpt_best}")
            print(f"[INFO] Tune phase config: topk={params.get('topk', 'None')}, epochs={params.get('epochs', 5)}, early_stopping=True")
            out_val = run_once(df, groups, params, train_idx=tr_idx, val_idx=va_idx)

        out_base = args.out_json or os.path.join("results", f"{base_name}_trainval.json")
        os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
        with open(out_base, "w") as f:
            json.dump({"train_val": out_val}, f, indent=2)
        print(f"[INFO] wrote train/val results to {out_base}")

        if args.retrain_on_full:
            print("[INFO] Retraining on train+val and evaluating on test...")
            print("[INFO] NOTE: Tune phase metrics on VAL set, retrain phase metrics on TEST set (different data)")
            print(f"[INFO] Tune phase split: train={len(tr_idx)} | val={len(va_idx)}")
            print(f"[INFO] Retrain phase split: train={len(tr_idx)+len(va_idx)} | test={len(te_idx)}")
            params_full = dict(vars(args))
            # Keep topk from tune phase (don't set to None)
            # params_full["topk"] = None  # <-- REMOVED: preserve topk setting
            ckpt_full = os.path.join("results", f"{base_name}_full.pt")
            params_full["save_checkpoint"] = ckpt_full
            params_full["__retrain_log_test_each_epoch"] = True
            
            # Use best_epoch from tune phase for retraining
            best_epoch_from_tune = out_val.get("best_epoch", None)
            if best_epoch_from_tune is not None and best_epoch_from_tune > 0:
                print(f"[INFO] Using best_epoch={best_epoch_from_tune} from tune phase for retraining")
                params_full["epochs"] = int(best_epoch_from_tune)
                params_full["no_early_stopping"] = True  # Disable early stopping for retrain
                print(f"[INFO] Retrain config: epochs={best_epoch_from_tune}, no_early_stopping=True, topk={params_full.get('topk', 'None')}")
            else:
                print("[WARN] best_epoch not found in tune phase, retraining with original settings")
            
            out_full = run_once(
                df,
                groups,
                params_full,
                train_idx=np.concatenate([tr_idx, va_idx]),
                val_idx=te_idx,
                no_validation=True,
            )
            test_metrics = {
                "test_auc": float(out_full.get("val_auc", float("nan"))),
                "test_acc": float(out_full.get("val_acc", float("nan"))),
                "test_bacc": float(out_full.get("val_bacc", float("nan"))),
                "test_f1": float(out_full.get("val_f1", float("nan"))),
            }
            final_out = {**out_val, **test_metrics}
            # mark default provenance for test metrics (assume produced by retrain run)
            final_out["_test_metrics_source"] = "retrain_run"
            # Ensure test metrics are valid; if retrain run returned NaNs or missing
            # values, try to compute metrics from any final_proba/_final_y dumps
            try:
                bad = False
                for k in ("test_auc", "test_acc", "test_bacc", "test_f1"):
                    v = final_out.get(k, None)
                    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        bad = True
                        break
                if bad:
                    # search for final_proba / final_y numpy dumps produced by run_once
                    import glob
                    base_path = None
                    # prefer explicit save_checkpoint or out_json to derive base name
                    if params_full.get("save_checkpoint"):
                        base_path = os.path.splitext(params_full.get("save_checkpoint"))[0]
                    elif params_full.get("out_json"):
                        base_path = os.path.splitext(params_full.get("out_json"))[0]
                    else:
                        base_path = os.path.join("results", base_name + "_full")

                    proba_pattern = f"{base_path}*final*proba*.npy"
                    proba_files = glob.glob(proba_pattern)
                    if proba_files:
                        pfile = proba_files[0]
                        yfile = pfile.replace("proba", "y")
                        if os.path.isfile(yfile):
                            try:
                                proba = np.load(pfile)
                                y = np.load(yfile)
                                m_eval = eval_multiclass_metrics(y, proba)
                                final_out["test_auc"] = float(m_eval.get("auc", float("nan")))
                                final_out["test_acc"] = float(m_eval.get("acc", float("nan")))
                                final_out["test_bacc"] = float(m_eval.get("bacc", float("nan")))
                                final_out["test_f1"] = float(m_eval.get("f1", float("nan")))
                                # provenance: metrics were computed from saved final_proba/_final_y dumps
                                final_out["_test_metrics_source"] = "final_proba_dump"
                                final_out["_test_metrics_provenance"] = {"file": pfile}
                                print(f"[INFO] computed retrain test metrics from dumps {pfile}")
                            except Exception as _e:
                                print(f"[WARN] failed to compute retrain metrics from dumps {pfile}: {_e}")
            except Exception:
                pass
            if "confusion_report" in out_full:
                final_out["confusion_report_test"] = out_full["confusion_report"]
            # Persist the retrain-on-full results (including test_* keys) to the
            # caller-provided --out_json path so external wrappers (e.g. the
            # optuna tuner) can read the retrain metrics at the expected
            # location. Previously we only wrote the train/val JSON (out_base)
            # and left retrain results only as proba/checkpoint files which
            # caused the wrapper to merge an incomplete JSON.
            out_path = args.out_json or os.path.join("results", f"{base_name}_trainval_test.json")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(final_out, f, indent=2)
            print(f"[INFO] wrote retrain-on-full results to {out_path}")

        else:
            print(f"[INFO] Evaluating saved best checkpoint {ckpt_best} on test set...")
            meta = {}
            meta_path = os.path.splitext(ckpt_best)[0] + ".meta.pkl"
            if os.path.isfile(meta_path):
                try:
                    import pickle

                    with open(meta_path, "rb") as mf:
                        meta = pickle.load(mf)
                except Exception as e:
                    print(f"[WARN] could not load meta file {meta_path}: {e}")

            scalers = meta.get("scalers", None)
            model_cfg = meta.get("model_config", {})
            ds_test = MoEDataset(df.iloc[te_idx], groups, scalers=scalers)
            B = args.batch_size
            num_workers = int(args.num_workers)
            pin_mem = torch.cuda.is_available()

            def _collate_local(batch):
                M = len(batch[0][0])
                xs = [torch.stack([b[0][m] for b in batch], dim=0) for m in range(M)]
                masks = torch.stack([b[1] for b in batch], dim=0)
                y = torch.tensor([b[2] for b in batch], dtype=torch.long)
                return xs, masks, y

            test_loader = DataLoader(ds_test, batch_size=B, shuffle=False, collate_fn=_collate_local, pin_memory=pin_mem, num_workers=num_workers)

            dims = [len(v) for v in groups.values()]
            use_hier = model_cfg.get("use_hierarchical_gate", False) or args.use_hierarchical_gate
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if use_hier:
                model = HierarchicalMoE(
                    groups,
                    hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                    hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                    n_classes=3,
                    drop=model_cfg.get("drop", args.drop),
                    gate_type=model_cfg.get("gate_type", args.gate_type),
                    gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                    gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                    topk=model_cfg.get("topk", args.topk),
                ).to(device)
            else:
                model = MoE(
                    dims,
                    hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                    hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                    n_classes=3,
                    drop=model_cfg.get("drop", args.drop),
                    gate_type=model_cfg.get("gate_type", args.gate_type),
                    gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                    gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                    topk=model_cfg.get("topk", args.topk),
                ).to(device)

            if os.path.isfile(ckpt_best):
                state = torch.load(ckpt_best, map_location=device)
                model.load_state_dict(state)
                model.eval()
            else:
                print(f"[WARN] checkpoint {ckpt_best} not found; skipping test evaluation")

            all_p, all_y = [], []
            try:
                cli_specified_tau = any([a == "--tau" or a.startswith("--tau=") for a in sys.argv])
            except Exception:
                cli_specified_tau = False
            tau_to_use = args.tau if cli_specified_tau else meta.get("train_params", {}).get("tau", args.tau)

            with torch.no_grad():
                for xs, masks, y in test_loader:
                    xs = [x.to(device) for x in xs]
                    masks = masks.to(device)
                    logits, gate_w = model(xs, masks, tau=tau_to_use)
                    proba = F.softmax(logits, dim=1).cpu().numpy()
                    all_p.append(proba)
                    all_y.append(y.numpy())

            if all_p:
                all_p = np.vstack(all_p)
                all_y = np.concatenate(all_y)
            else:
                all_p = np.zeros((0, 3))
                all_y = np.array([])

            mtest = utils.eval_multiclass_metrics(all_y, all_p)
            test_metrics = {
                "test_auc": float(mtest.get("auc", float("nan"))),
                "test_acc": float(mtest.get("acc", float("nan"))),
                "test_bacc": float(mtest.get("bacc", float("nan"))),
                "test_f1": float(mtest.get("f1", float("nan"))),
            }

            final_out = {**out_val, **test_metrics}
            out_path = args.out_json or os.path.join("results", f"{base_name}_trainval_test.json")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(final_out, f, indent=2)
            print(f"[INFO] wrote train/val/test results to {out_path}")
        return

    # CV branch
    if args.split_type == "cv5":
        with open(args.splits, "r") as f:
            splits_json = json.load(f)
        if "cv_splits_ptid" not in splits_json:
            raise ValueError("splits JSON must contain 'cv_splits_ptid' for --split_type=cv5")
        cv_splits = splits_json["cv_splits_ptid"]

        k_values = [5, 3, 1] if args.topk else [None]
        for k in k_values:
            label = f"top-{k}" if k is not None else "full"
            out_suffix = f"top-{k}" if k is not None else "full"
            if args.gate_ablation:
                out_suffix = f"{args.gate_ablation}_{out_suffix}"
            print(f"\n[INFO] ===== Running CV for {label} gating {'with top-' + str(k) + ' ' if k is not None else ''}=====")
            set_seed()
            results_folds = []
            all_payloads = []

            fold_indices = ([args.only_fold] if args.only_fold is not None else list(range(len(cv_splits))))
            for fold_idx in fold_indices:
                split = cv_splits[fold_idx]
                print(f"\n[INFO] Running fold {fold_idx + 1}/{len(cv_splits)} ({label})")
                train_keys = [kk for kk in split.keys() if "train" in kk.lower()]
                val_keys = [kk for kk in split.keys() if "val" in kk.lower()]
                if not train_keys or not val_keys:
                    raise KeyError(f"Fold {fold_idx}: missing train/val keys in {list(split.keys())}")

                train_ptids = set(split[train_keys[0]])
                val_ptids = set(split[val_keys[0]])

                train_idx = df[df["PTID"].astype(str).isin(train_ptids)].index.tolist()
                val_idx = df[df["PTID"].astype(str).isin(val_ptids)].index.tolist()

                tr_ptids = set(df.iloc[train_idx]["PTID"].astype(str))
                va_ptids = set(df.iloc[val_idx]["PTID"].astype(str))
                leak = tr_ptids.intersection(va_ptids)
                if leak:
                    print(f"[WARN] {len(leak)} PTIDs appear in both train and val in fold {fold_idx+1}")

                params = dict(vars(args))
                params["topk"] = k
                print(f"[DEBUG] Fold {fold_idx} running with topk={k}")

                if k is not None:
                    out = run_once(df, groups, params, train_idx=train_idx, val_idx=val_idx, gating_fn=lambda gw, m, top_k=k: apply_topk_gating(gw, m, top_k=top_k))
                else:
                    out = run_once(df, groups, params, train_idx=train_idx, val_idx=val_idx)

                out_fold = {"fold": fold_idx, "top_k": k, **out}
                results_folds.append(out_fold)

                if args.only_fold is not None:
                    if args.out_json:
                        fold_out_path = args.out_json
                    else:
                        fold_out_dir = os.path.join("results", "runs", "cv_folds")
                        os.makedirs(fold_out_dir, exist_ok=True)
                        fold_out_path = os.path.join(fold_out_dir, f"fold{fold_idx}.json")
                    os.makedirs(os.path.dirname(fold_out_path) or ".", exist_ok=True)
                    with open(fold_out_path, "w") as f:
                        json.dump(out_fold, f, indent=2)
                    print(f"[INFO] wrote fold result to {fold_out_path}")
                    return

                print(f"[INFO] Fold {fold_idx+1} ({label}) val macro-AUROC: {out['val_auc']:.4f}")

                if args.save_payloads and "payload" in out:
                    for rec in out["payload"]:
                        rec_with_fold = dict(rec)
                        rec_with_fold["fold"] = fold_idx
                        all_payloads.append(rec_with_fold)

            metric_names = ["val_auc", "val_acc", "val_bacc", "val_f1"]
            metrics = {m: [fold[m] for fold in results_folds] for m in metric_names}
            summary = {f"{m}_mean": float(np.mean(np.array(metrics[m]))) for m in metric_names}
            for m in metric_names:
                summary[f"{m}_std"] = float(np.std(np.array(metrics[m])))

            # Aggregate confusion matrices
            agg_confusion_report = None
            try:
                cms = []
                for fold in results_folds:
                    cr = fold.get("confusion_report", None)
                    if cr is None:
                        continue
                    cm_counts = cr.get("cm_counts", None)
                    if cm_counts is None:
                        continue
                    cms.append(np.array(cm_counts, dtype=int))
                if len(cms) > 0:
                    cm_total = np.sum(np.stack(cms, axis=0), axis=0).astype(int)
                    row_sums = cm_total.sum(axis=1, keepdims=True).astype(float)
                    row_sums[row_sums == 0.0] = 1.0
                    cm_row = cm_total / row_sums
                    tp = np.diag(cm_total).astype(float)
                    col_sums = cm_total.sum(axis=0).astype(float)
                    prec = np.divide(tp, col_sums, out=np.zeros_like(tp), where=(col_sums > 0))
                    rec = np.divide(tp, row_sums.squeeze(1), out=np.zeros_like(tp), where=(row_sums.squeeze(1) > 0))
                    f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(tp), where=((prec + rec) > 0))
                    class_names = ["CN", "MCI", "AD"]
                    agg_confusion_report = {
                        "cm_counts": cm_total.tolist(),
                        "cm_row_norm": cm_row.tolist(),
                        "per_class": {
                            "precision": {class_names[i]: float(prec[i]) for i in range(len(class_names))},
                            "recall": {class_names[i]: float(rec[i]) for i in range(len(class_names))},
                            "f1": {class_names[i]: float(f1[i]) for i in range(len(class_names))},
                        },
                    }
            except Exception as e:
                print(f"[WARN] could not aggregate confusion matrices across folds: {e}")

            if args.out_json:
                base, ext = os.path.splitext(args.out_json)
                out_path = f"{base}_{out_suffix}{ext or '.json'}"
            else:
                out_path = f"results/moe_hierarchical_cv10_{out_suffix}.json"

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"folds": results_folds, "summary": summary, "confusion_report_all_folds": agg_confusion_report}, f, indent=2)
            print(f"[INFO] wrote results to {out_path}")
        return


def _build_payloads(model, df, groups, scalers, device, B, num_workers, pin_mem, collate, params, train_idx, val_idx, fold_idx=None):
    """Construct per-sample payload records (extracted from run_once).

    Mirrors previous inline behavior but returns the list of payload records
    instead of writing side-effects here. Keeps errors local and returns an
    empty list on failure.
    """
    payload_records = []
    try:
        model.eval()
        names = list(groups.keys())
        run_name = os.path.splitext(os.path.basename(params.get("out_json") or "results/moe_results_lastvisit.json"))[0]

        # Build dataset over all rows using training scalers
        ds_all = MoEDataset(df, groups, scalers=scalers)
        all_loader = DataLoader(
            ds_all,
            batch_size=B,
            shuffle=False,
            collate_fn=collate,
            pin_memory=pin_mem,
            num_workers=num_workers,
        )

        train_idx_set = set(train_idx)
        val_idx_set = set(val_idx)
        global_idx = 0

        with torch.no_grad():
            for xs, masks, y in all_loader:
                xs = [x.to(device) for x in xs]
                masks = masks.to(device)
                y = y.to(device)

                logits, gate_w_val = model(xs, masks)

                # Gate ablation (same behavior as inline)
                if params.get("gate_ablation") is not None:
                    mode = params["gate_ablation"]
                    if mode == "region_only":
                        pass
                    elif mode == "modality_only":
                        names_local = names
                        prefix_to_idx = {}
                        for j, n in enumerate(names_local):
                            prefix = n.split("_")[0]
                            prefix_to_idx.setdefault(prefix, []).append(j)
                        new_gate = torch.zeros_like(gate_w_val)
                        for p, idxs in prefix_to_idx.items():
                            avg = gate_w_val[:, idxs].sum(dim=1, keepdim=True)
                            new_gate[:, idxs] = avg / (len(idxs) + 1e-8)
                        gate_w_val = new_gate * masks
                    elif mode == "random":
                        gate_w_val = torch.rand_like(gate_w_val) * masks
                    gate_w_val = gate_w_val / (gate_w_val.sum(dim=1, keepdim=True) + 1e-8)

                if params.get("gating_fn"):
                    gw = params.get("gating_fn")(gate_w_val, masks)
                else:
                    gw = gate_w_val * masks
                    gw = gw / (gw.sum(dim=1, keepdim=True) + 1e-8)

                probs = F.softmax(logits, dim=1)
                probs_np = probs.cpu().numpy()
                gw_np = gw.cpu().numpy()
                y_np = y.cpu().numpy()
                B_cur = y_np.shape[0]

                for i in range(B_cur):
                    idx = global_idx + i
                    raw_row = df.iloc[idx]

                    ptid_raw = str(raw_row["PTID"])
                    if params.get("redact_ptid", False):
                        ptid = hashlib.sha256(ptid_raw.encode("utf-8")).hexdigest()[:16]
                    else:
                        ptid = ptid_raw

                    prob_vec = probs_np[i]
                    pred_class = int(np.argmax(prob_vec))

                    # Raw feature values
                    if params.get("redact_values", False):
                        raw_values = None
                    else:
                        raw_values = {}
                        for g_name, cols in groups.items():
                            group_vals = {}
                            for col in cols:
                                if col not in df.columns:
                                    continue
                                v = raw_row[col]
                                if pd.isna(v):
                                    group_vals[col] = None
                                else:
                                    try:
                                        group_vals[col] = float(v)
                                    except Exception:
                                        group_vals[col] = None
                            raw_values[g_name] = group_vals

                        # Optionally add AGE_AT_VISIT
                        age_group_name = "demographic"
                        for age_col in ["AGE_AT_VISIT"]:
                            if age_col in df.columns:
                                age_val = raw_row.get(age_col, np.nan)
                                if age_group_name not in raw_values:
                                    raw_values[age_group_name] = {}
                                if age_col not in raw_values[age_group_name]:
                                    if pd.isna(age_val):
                                        raw_values[age_group_name][age_col] = None
                                    else:
                                        try:
                                            raw_values[age_group_name][age_col] = float(age_val)
                                        except Exception:
                                            raw_values[age_group_name][age_col] = None

                    if params.get("redact_values", False):
                        prob_list = [round(float(v), 3) for v in prob_vec]
                        confidence = float(round(float(prob_vec[pred_class]), 3))
                        token_gates = {name: round(float(gw_np[i, j]), 3) for j, name in enumerate(names)}
                    else:
                        prob_list = prob_vec.tolist()
                        confidence = float(prob_vec[pred_class])
                        token_gates = {name: float(gw_np[i, j]) for j, name in enumerate(names)}

                    if idx in train_idx_set:
                        split_label = "train"
                    elif idx in val_idx_set:
                        split_label = "val"
                    else:
                        split_label = "unused"

                    sample_record = {
                        "PTID": ptid,
                        "y_true": int(y_np[i]),
                        "y_pred": pred_class,
                        "probs": prob_list,
                        "confidence": confidence,
                        "token_gates": token_gates,
                        "model": run_name,
                        "fold": int(fold_idx) if fold_idx is not None else None,
                        "split": split_label,
                        "redacted": bool(params.get("redact_ptid", False) or params.get("redact_values", False)),
                        "raw_values": raw_values,
                    }
                    payload_records.append(sample_record)

                global_idx += B_cur
    except Exception as e:
        print(f"[WARN] _build_payloads failed: {e}")
        return []
    return payload_records


def _evaluate_loader(model, loader, device, criterion, params, groups, gating_fn, tau, eps, epoch, label="val"):
    """Evaluate `model` over `loader` and return a dict with loss, preds, labels and gate outputs.

    This captures the common validation/test loop including NaN/Inf guards and
    returns the raw arrays (may be empty arrays if loader has zero samples).
    """
    model.eval()
    all_y, all_p, all_gates = [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for xs, masks, y in tqdm(loader, desc=f"Epoch {epoch} [{label}]", leave=False):
            xs = [x.to(device) for x in xs]
            masks, y = masks.to(device), y.to(device)
            # forward (model may or may not accept tau)
            try:
                logits, gate_w = model(xs, masks, tau=tau)
            except TypeError:
                logits, gate_w = model(xs, masks)

            # Eval-time NaN/inf guard: dump offending batch but continue
            try:
                if not torch.isfinite(logits).all() or not torch.isfinite(gate_w).all():
                    base_dir = f"results/debug_nan_{label}_epoch{epoch}_batch{len(all_p)}"
                    prepared = _prepare_batch_for_dump(xs=xs, masks=masks, y=y, logits=logits, gate=gate_w)
                    _dump_debug_and_raise(base_dir, prepared, f"Non-finite values during {label} at epoch={epoch}", raise_exc=False)
            except Exception as _e:
                print(f"[WARN] {label} diagnostic check failed: {_e}")

            try:
                loss = criterion(logits, y)
                total_loss += float(loss.item()) * y.size(0)
            except Exception:
                pass

            proba = F.softmax(logits, dim=1).cpu().numpy()
            all_p.append(proba)
            all_y.append(y.cpu().numpy())

            if gating_fn:
                gw = gating_fn(gate_w, masks)
            else:
                gw = gate_w * masks
                gw = gw / (gw.sum(dim=1, keepdim=True) + eps)
            all_gates.append(gw.cpu().numpy())

    # Post-process stacked arrays and handle empty-case
    if params.get("gate_type", "softmax") == "gumbel":
        # caller should handle MC averaging if desired; keep placeholder
        return {"loss": float("nan"), "all_p": [], "all_y": [], "all_gates": []}
    else:
        try:
            if all_p:
                all_p = np.vstack(all_p)
                all_y = np.concatenate(all_y)
                total_loss = total_loss / (len(loader.dataset) if len(loader.dataset) > 0 else 1)
            else:
                all_p = np.zeros((0, params.get("n_classes", 3)))
                all_y = np.array([])
        except Exception:
            # if stacking fails, return safe empties
            all_p = np.zeros((0, params.get("n_classes", 3)))
            all_y = np.array([])
        return {"loss": total_loss, "all_p": all_p, "all_y": all_y, "all_gates": all_gates}


def _compute_metrics_from_preds(all_p, all_y, no_validation):
    """Compute multiclass metrics from predictions/labels with safe fallbacks."""
    if no_validation and (not all_p):
        return {"auc": 0.5, "acc": 0.0, "bacc": 0.0, "f1": 0.0}
    try:
        return utils.eval_multiclass_metrics(all_y, all_p)
    except Exception:
        return {"auc": 0.5, "acc": 0.0, "bacc": 0.0, "f1": 0.0}


def _finite(x):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return 0.0
        elif hasattr(x, "__float__"):
            x = float(x)
            if math.isnan(x) or math.isinf(x):
                return 0.0
            return x
        else:
            return 0.0
    except Exception:
        return 0.0


def _log_epoch_metrics(epoch, val_loss, val_auc, val_acc, val_bacc, val_f1, opt, params, groups, all_gates, printed_test_line=False):
    """Build `cur_metrics`, optionally mirror them to `test_*` keys for retrain-on-full, and print the epoch summary if needed.

    Returns the `cur_metrics` dict for upstream bookkeeping.
    """
    def _finite(x):
        try:
            if x is None or (
                isinstance(x, float) and (math.isnan(x) or math.isinf(x))
            ):
                return 0.0
            elif hasattr(x, "__float__"):
                x = float(x)
                if math.isnan(x) or math.isinf(x):
                    return 0.0
                return x
            else:
                return 0.0
        except Exception:
            return 0.0

    cur_lr = opt.param_groups[0]["lr"]

    cur_metrics = {
        "val_loss": _finite(val_loss),
        "val_auc": _finite(val_auc),
        "val_acc": _finite(val_acc),
        "val_bacc": _finite(val_bacc),
        "val_f1": _finite(val_f1),
    }

    # Mirror val_* -> test_* when retraining-on-full to avoid ambiguity in JSON logs
    if params.get("__retrain_log_test_each_epoch", False) and params.get("retrain_on_full", False):
        cur_metrics.update(
            {
                "test_loss": cur_metrics["val_loss"],
                "test_auc": cur_metrics["val_auc"],
                "test_acc": cur_metrics["val_acc"],
                "test_bacc": cur_metrics["val_bacc"],
                "test_f1": cur_metrics["val_f1"],
            }
        )

    # Print only if a per-epoch test print hasn't already been emitted
    if not printed_test_line:
        print(
            f"Epoch {epoch:03d} | val_loss={cur_metrics['val_loss']:.4f} | auc={cur_metrics['val_auc']:.4f} | "
            f"acc={cur_metrics['val_acc']:.4f} | bacc={cur_metrics['val_bacc']:.4f} | f1={cur_metrics['val_f1']:.4f} | lr={cur_lr:.2e}"
        )

    return cur_metrics


def gumbel_softmax_sample(logits, tau=1.0, hard=False, eps=1e-9):
    """Gumbel-Softmax sampling with optional straight-through estimator."""
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    y_soft = F.softmax((logits + gumbel) / tau, dim=-1)
    if not hard:
        return y_soft
    # Straight-through trick: forward one-hot, backward soft
    index = y_soft.max(dim=-1, keepdim=True)[1]
    y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
    return (y_hard - y_soft).detach() + y_soft


# ----------------------------
# Dataset
# ----------------------------
class MoEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, groups: Dict[str, List[str]], scalers=None):
        self.df = df.reset_index(drop=True)
        self.groups = groups
        self.scalers = scalers or {}
        # fit scalers per modality on available rows
        for m, cols in groups.items():
            if m not in self.scalers:
                sc = StandardScaler()
                X = self.df[cols].astype(float).to_numpy()
                col_meds = np.nanmedian(X, axis=0)
                X = np.where(np.isnan(X), col_meds, X)
                sc.fit(X)
                self.scalers[m] = sc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        xs, masks, sizes = [], [], []
        for m, cols in self.groups.items():
            x = row[cols].astype(float).values
            # Safe median imputation even when whole column is NaN
            subdf = self.df[cols].apply(pd.to_numeric, errors="coerce")
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'All-NaN slice encountered')
                col_meds = np.nanmedian(subdf.to_numpy(), axis=0)
            col_meds = np.where(np.isnan(col_meds), 0.0, col_meds)
            x = np.where(np.isnan(x), col_meds, x)
            sc = self.scalers[m]
            scale = np.where(sc.scale_ == 0, 1.0, sc.scale_)  # guard zero-variance cols
            x = (x - sc.mean_) / scale
            xs.append(torch.tensor(x, dtype=torch.float32))
            masks.append(torch.tensor(float(row[f"has_{m}"]), dtype=torch.float32))
            sizes.append(len(cols))
        y = int(row["y"])
        return xs, torch.stack(masks), y


# ----------------------------
# Model
# ----------------------------
class Expert(nn.Module):
    """3-layer MLP expert (matches baseline)."""

    def __init__(self, in_dim, hidden=256, n_classes=3, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class Gate(nn.Module):
    def __init__(
        self,
        dims: List[int],
        hidden=64,
        gate_type="softmax",
        gumbel_hard=False,
        gate_noise=0.02,
        topk=None,
    ):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, 32) for d in dims])
        self.fc = nn.Sequential(
            nn.Linear(32 * len(dims) + len(dims), hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(dims)),
        )
        self.gate_type = gate_type
        self.gumbel_hard = gumbel_hard
        self.gate_noise = gate_noise
        self.topk = topk

    def forward(self, xs: List[torch.Tensor], masks: torch.Tensor, tau: float = 0.9):
        emb = [torch.tanh(p(x)) for p, x in zip(self.proj, xs)]
        h = torch.cat(emb + [masks], dim=1)
        w = self.fc(h)
        if self.training and self.gate_noise > 0:
            w = w + self.gate_noise * torch.randn_like(w)
        w = w.masked_fill(masks <= 0, float("-inf"))
        if self.gate_type == "gumbel":
            gate = gumbel_softmax_sample(w, tau, hard=self.gumbel_hard)
        else:
            gate = F.softmax(w / tau, dim=1)
        gate = torch.nan_to_num(gate, nan=0.0)
        gate = gate * masks
        
        # Before topk: log natural sparsity
        if isinstance(self.topk, int) and self.topk > 0:
            pre_topk_active = (gate > 1e-6).sum(dim=1).float().mean().item()
            topk_vals, topk_idx = torch.topk(gate, k=self.topk, dim=1)
            mask = torch.zeros_like(gate)
            mask.scatter_(1, topk_idx, topk_vals)
            gate = mask
            post_topk_active = (gate > 0).sum(dim=1).float().mean().item()
            # Only log occasionally to avoid spam (1% of batches)
            if torch.rand(1).item() < 0.01:
                print(f"[GATE] topk={self.topk} | pre_topk_active={pre_topk_active:.2f} | post_topk_active={post_topk_active:.2f} | constrained={pre_topk_active > self.topk}")
        
        gate = gate / (gate.sum(dim=1, keepdim=True) + 1e-8)
        return gate


class MoE(nn.Module):
    def __init__(
        self,
        dims: List[int],
        hidden_exp=128,
        hidden_gate=64,
        n_classes=3,
        drop=0.2,
        gate_type="softmax",
        gumbel_hard=False,
        gate_noise=0.02,
        topk=None,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(d, hidden_exp, n_classes, drop) for d in dims]
        )
        self.gate = Gate(
            dims,
            hidden_gate,
            gate_type=gate_type,
            gumbel_hard=gumbel_hard,
            gate_noise=gate_noise,
            topk=topk,
        )
        self.n_classes = n_classes
        self.topk = topk

    def forward(self, xs: List[torch.Tensor], masks: torch.Tensor, tau: float = 0.9):
        logits_list = [exp(x) for exp, x in zip(self.experts, xs)]  # [B, M, C]
        logits = torch.stack(logits_list, dim=1)
        gate_w = self.gate(xs, masks, tau=tau)  # [B, M]
        mix = torch.sum(gate_w.unsqueeze(-1) * logits, dim=1)
        return mix, gate_w


###############################
# HierarchicalMoE and Gate
###############################
class HierarchicalGate(nn.Module):
    """
    Two-level hierarchical gating:
      - Modality-level gate weights (e.g., amyloid, mri, demographic)
      - Region-level gates distribute modality weights across their experts.
    The total output matches the number of experts.
    """

    def __init__(
        self,
        modality_groups: Dict[str, List[str]],
        hidden=64,
        gate_type="softmax",
        gumbel_hard=False,
        gate_noise=0.02,
        tau=0.9,
        topk=None,
    ):
        super().__init__()
        self.modalities = list(modality_groups.keys())
        self.gate_type = gate_type
        self.gumbel_hard = gumbel_hard
        self.gate_noise = gate_noise
        self.tau = tau
        self.topk = topk

        self.modality_to_experts = {
            mod: len([m for m in modality_groups.keys() if m.startswith(mod)])
            for mod in ["amy", "mri", "demographic"]
            if any(k.startswith(mod) for k in modality_groups.keys())
        }

        self.modality_fc = nn.Sequential(
            nn.Linear(len(self.modality_to_experts), hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(self.modality_to_experts)),
        )
        
        self.region_gates = nn.ModuleDict(
            {
                mod: nn.Sequential(
                    nn.Linear(self.modality_to_experts[mod], hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, self.modality_to_experts[mod]),
                )
                for mod in self.modality_to_experts.keys()
            }
        )

    def forward(self, xs: List[torch.Tensor], masks: torch.Tensor):
        # masks is [B, M] where M == number of experts (same order as self.modalities)
        B = masks.size(0)
        M = masks.size(1)

        # Map modality -> indices in expert list
        modality_regions: Dict[str, List[int]] = {}
        for mod in self.modality_to_experts.keys():
            modality_regions[mod] = [
                i for i, name in enumerate(self.modalities) if name.startswith(mod)
            ]

        # Build a simple modality embedding from available experts in that modality.
        # We keep the original “mean of means” idea, but make it mask-aware.
        modality_embs = []
        modality_has = []  # [B, n_modalities]
        modality_names = list(self.modality_to_experts.keys())

        for mod in modality_names:
            idxs = modality_regions.get(mod, [])
            if len(idxs) == 0:
                # Should not happen, but keep shapes sane
                modality_embs.append(torch.zeros((B, 1), device=masks.device))
                modality_has.append(torch.zeros((B, 1), device=masks.device))
                continue

            # Expert availability within this modality
            m_mod = masks[:, idxs]  # [B, R]
            has_mod = (m_mod.max(dim=1, keepdim=True).values > 0).float()  # [B,1]
            modality_has.append(has_mod)

            # xs[i] is [B, D_i]. Compute a scalar per expert, then mask and average.
            expert_scalars = []
            for i in idxs:
                # scalar summary per sample for this expert
                expert_scalars.append(xs[i].mean(dim=1, keepdim=True))
            expert_scalars = torch.cat(expert_scalars, dim=1)  # [B, R]

            # Mask-aware mean; if nothing present, leave as zeros
            denom = m_mod.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_emb = (expert_scalars * m_mod).sum(dim=1, keepdim=True) / denom
            # If the whole modality is missing, force embedding to 0
            mean_emb = mean_emb * has_mod
            modality_embs.append(mean_emb)

        h_mod = torch.cat(modality_embs, dim=1)  # [B, n_modalities]
        has_mod_mat = torch.cat(modality_has, dim=1)  # [B, n_modalities]

        # Modality-level logits -> mask missing modalities -> softmax
        w_mod_logits = self.modality_fc(h_mod)
        if self.training and self.gate_noise > 0:
            w_mod_logits = w_mod_logits + self.gate_noise * torch.randn_like(w_mod_logits)
        w_mod_logits = w_mod_logits.masked_fill(has_mod_mat <= 0, float("-inf"))
        w_mod = F.softmax(w_mod_logits / self.tau, dim=1)
        w_mod = torch.nan_to_num(w_mod, nan=0.0) * has_mod_mat
        w_mod = w_mod / (w_mod.sum(dim=1, keepdim=True).clamp_min(1e-8))

        # Build final expert gate weights in the SAME order as self.modalities
        gate_final = torch.zeros((B, M), device=masks.device, dtype=masks.dtype)

        for j, mod in enumerate(modality_names):
            idxs = modality_regions.get(mod, [])
            if len(idxs) == 0:
                continue

            # Region-level logits -> mask missing regions -> softmax
            # (Keep the learned region gate, but make it mask-aware.)
            n_regions = len(idxs)
            w_region_logits = self.region_gates[mod](
                torch.ones((B, n_regions), device=masks.device, dtype=masks.dtype)
            )
            if self.training and self.gate_noise > 0:
                w_region_logits = w_region_logits + self.gate_noise * torch.randn_like(w_region_logits)

            m_reg = masks[:, idxs]  # [B, R]
            w_region_logits = w_region_logits.masked_fill(m_reg <= 0, float("-inf"))
            w_region = F.softmax(w_region_logits / self.tau, dim=1)
            w_region = torch.nan_to_num(w_region, nan=0.0) * m_reg
            w_region = w_region / (w_region.sum(dim=1, keepdim=True).clamp_min(1e-8))

            # Combine modality weight with region weights
            w_expert = w_mod[:, j].unsqueeze(1) * w_region  # [B, R]

            # Place into final gate in expert order
            for k, idx in enumerate(idxs):
                gate_final[:, idx] = w_expert[:, k]

        # Ensure absent experts are zeroed, then optional top-k, then renormalize
        gate_final = gate_final * masks
        if isinstance(self.topk, int) and self.topk > 0:
            topk_vals, topk_idx = torch.topk(gate_final, k=min(self.topk, gate_final.size(1)), dim=1)
            mask_topk = torch.zeros_like(gate_final)
            mask_topk.scatter_(1, topk_idx, topk_vals)
            gate_final = mask_topk

        gate_final = gate_final / (gate_final.sum(dim=1, keepdim=True).clamp_min(1e-8))
        return gate_final


class HierarchicalMoE(nn.Module):
    def __init__(
        self,
        groups: Dict[str, List[int]],
        hidden_exp=128,
        hidden_gate=64,
        n_classes=3,
        drop=0.2,
        gate_type="softmax",
        gumbel_hard=False,
        gate_noise=0.02,
        topk=None,
    ):
        super().__init__()
        self.modalities = list(groups.keys())
        self.experts = nn.ModuleList(
            [Expert(len(groups[m]), hidden_exp, n_classes, drop) for m in self.modalities]
        )
        self.gate = HierarchicalGate(
            groups,
            hidden_gate,
            gate_type,
            gumbel_hard,
            gate_noise,
            topk=topk,
        )
        self.n_classes = n_classes
        self.topk = topk

    def forward(self, xs: List[torch.Tensor], masks: torch.Tensor, tau: float = 0.9):
        logits_list = [exp(x) for exp, x in zip(self.experts, xs)]
        logits = torch.stack(logits_list, dim=1)
        gate_w = self.gate(xs, masks)
        mix = torch.sum(gate_w.unsqueeze(-1) * logits, dim=1)
        return mix, gate_w


# ----------------------------
# Helper for MC inference
# ----------------------------
def predict_mc(model, val_loader, n_samples=10, device="cpu", tau=None):
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for xs, masks, y in val_loader:
            xs = [x.to(device) for x in xs]
            masks = masks.to(device)
            probs = []
            for _ in range(n_samples):
                try:
                    logits, _ = model(xs, masks, tau=tau)
                except TypeError:
                    logits, _ = model(xs, masks)
                probs.append(F.softmax(logits, dim=1))
            mean_probs = torch.stack(probs).mean(0)
            all_preds.append(mean_probs.cpu().numpy())
            all_y.append(y.numpy())
    return np.vstack(all_preds), np.concatenate(all_y)


# ----------------------------
# Top-k Gating
# ----------------------------
def apply_topk_gating(gate_w, masks, top_k=None):
    gate_masked = gate_w * masks
    if top_k is not None:
        topk_vals, topk_idx = torch.topk(gate_masked, k=top_k, dim=1)
        topk_mask = torch.zeros_like(gate_masked)
        topk_mask.scatter_(1, topk_idx, topk_vals)
        gate_masked = topk_mask
    gate_masked = gate_masked / (gate_masked.sum(dim=1, keepdim=True) + 1e-8)
    return gate_masked


# ----------------------------
# Training / Eval
# ----------------------------
def _train_epoch(model, train_loader, device, opt, criterion, params, gating_fn, epoch, debug_one=False, debug_stop=False, groups=None):
    """Run one training epoch (previously inlined inside `run_once`).

    Returns a dict with 'debug_one_batch': True if the debug_one path triggered
    and an early return is requested. Otherwise returns an empty dict.
    """
    # ensure model is in training mode
    model.train()
    eps = 1e-8
    for i, (xs, masks, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)):
        xs = [x.to(device) for x in xs]
        masks, y = masks.to(device), y.to(device)
        tau = max(params["tau"], params["tau_start"] * (params["tau_decay"] ** epoch))
        logits, gate_w = model(xs, masks, tau=tau)

        # Fail-fast NaN/inf diagnostic
        try:
            bad = False
            if not torch.isfinite(logits).all():
                bad = True
            if not torch.isfinite(gate_w).all():
                bad = True
            loss_check = criterion(logits, y)
            if not torch.isfinite(loss_check).all():
                bad = True
            if bad:
                dump_path = f"results/debug_nan_batch_epoch{epoch}_batch{i}"
                prepared = _prepare_batch_for_dump(xs=xs, masks=masks, y=y, logits=logits, gate=gate_w)
                _dump_debug_and_raise(dump_path, prepared, f"Non-finite values detected at epoch={epoch} batch={i}", raise_exc=True)
        except Exception as _e:
            if isinstance(_e, RuntimeError):
                raise
            else:
                print(f"[WARN] diagnostic check failed: {_e}")

        # Gate ablations (kept identical to previous behavior)
        if params.get("gate_ablation") is not None:
            mode = params["gate_ablation"]
            if mode == "region_only":
                pass
            elif mode == "modality_only":
                names = list(groups.keys())
                prefix_to_idx = {}
                for j, n in enumerate(names):
                    prefix = n.split("_")[0]
                    prefix_to_idx.setdefault(prefix, []).append(j)
                new_gate = torch.zeros_like(gate_w)
                for p, idxs in prefix_to_idx.items():
                    avg = gate_w[:, idxs].sum(dim=1, keepdim=True)
                    new_gate[:, idxs] = avg / (len(idxs) + 1e-8)
                gate_w = new_gate * masks
            elif mode == "random":
                gate_w = torch.rand_like(gate_w) * masks
            gate_w = gate_w / (gate_w.sum(dim=1, keepdim=True) + 1e-8)

        if gating_fn:
            gate_masked = gating_fn(gate_w, masks)
        else:
            gate_masked = gate_w * masks
            gate_masked = gate_masked / (gate_masked.sum(dim=1, keepdim=True) + eps)

        entropy = -(gate_masked * (gate_masked + eps).log()).sum(dim=1).mean()

        logits_list = [exp(x) for exp, x in zip(model.experts, xs)]
        logits_stack = torch.stack(logits_list, dim=1)
        try:
            # Guard covariance computation: torch.cov is unstable for <=1 observations
            logits_mean = logits_stack.mean(dim=2)  # [B, M]
            if logits_mean.size(0) < 2:
                # Not enough samples in this batch to compute a stable covariance
                diversity_loss = torch.tensor(0.0, device=device)
            else:
                cov = torch.cov(logits_mean.T)
                if torch.isnan(cov).any() or torch.isinf(cov).any():
                    cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
                diversity_loss = torch.mean(torch.tril(cov, diagonal=-1) ** 2)
                if not torch.isfinite(diversity_loss):
                    diversity_loss = torch.tensor(0.0, device=device)
        except Exception:
            diversity_loss = torch.tensor(0.0, device=device)

        loss = (
            criterion(logits, y)
            - params["lambda_sparse"] * entropy
            + params["lambda_diverse"] * diversity_loss
        )
        if epoch % 5 == 0 and i == 0:
            try:
                print(f"[DEBUG] tau={tau:.4f}, entropy={entropy.item():.4f}, diversity={diversity_loss.item():.4f}")
            except Exception:
                pass

        if debug_one:
            try:
                torch.autograd.set_detect_anomaly(True)
            except Exception:
                pass

        opt.zero_grad()
        try:
            loss.backward()
        except Exception as e:
            # Attempt to capture a rich debug dump on the first autograd failure
            base_dir = f"results/debug_autograd_fail_epoch{epoch}_batch{i}"
            try:
                prepared = _prepare_batch_for_dump(xs=xs, masks=masks, y=y, logits=logits, gate=gate_w)
                extra = {
                    "logits_stack": _to_np_safe(logits_stack),
                    "gate_w": _to_np_safe(gate_w),
                    "gate_masked": _to_np_safe(gate_masked) if 'gate_masked' in locals() else None,
                    "entropy": _to_np_safe(entropy),
                    "diversity_loss": _to_np_safe(diversity_loss),
                    "loss": _to_np_safe(loss),
                    "model_param_names": list(n for n, _ in model.named_parameters()),
                }
                dump_dict = {**prepared, **extra}
            except Exception:
                dump_dict = _prepare_batch_for_dump(xs=xs, masks=masks, y=y)
            # Dump and re-raise so the outer tooling / anomaly detection can capture
            _dump_debug_and_raise(base_dir, dump_dict, f"Exception during backward at epoch={epoch} batch={i}: {e}", raise_exc=True)

        # Gradient NaN/Inf check
        try:
            bad_grad = False
            bad_grads = {}
            for n, p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        bad_grad = True
                        try:
                            bad_grads[n] = p.grad.detach().cpu().numpy()
                        except Exception:
                            bad_grads[n] = None
            if bad_grad:
                base_dir = f"results/debug_nan_grad_epoch{epoch}_batch{i}"
                prepared = _prepare_batch_for_dump(xs=xs, masks=masks, y=y)
                param_meta = {k: v.shape for k, v in model.state_dict().items()}
                dump_dict = {**prepared, 'param_meta': param_meta, 'bad_grads': bad_grads}
                _dump_debug_and_raise(base_dir, dump_dict, f"Non-finite gradients detected at epoch={epoch} batch={i}", raise_exc=True)
        except Exception as _e:
            if isinstance(_e, RuntimeError):
                raise
            else:
                print(f"[WARN] gradient diagnostic check failed: {_e}")

        # Gradient clipping
        try:
            max_norm = float(params.get("grad_clip", 1.0))
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        except Exception:
            pass
        opt.step()

        if debug_one:
            print(f"[INFO] debug_one_batch: completed epoch={epoch} batch={i}; exiting early")
            try:
                os.makedirs('results', exist_ok=True)
                ckpt_path = 'results/debug_one_batch_checkpoint.pt'
                # save a CPU-cloned copy of the state dict and write meta via helper
                try:
                    # Prefer to save a CPU-cloned state dict to avoid CUDA tensors in file
                    try:
                        state_to_save = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    except Exception:
                        state_to_save = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                    # Always attempt to write the checkpoint first
                    torch.save(state_to_save, ckpt_path)

                    # Build meta using helper (works if ds_train is None)
                    try:
                        meta = _build_checkpoint_meta(ds_train if 'ds_train' in locals() else None, groups if 'groups' in locals() else None, params)
                        import pickle

                        meta_path = os.path.splitext(ckpt_path)[0] + ".meta.pkl"
                        with open(meta_path, "wb") as mf:
                            pickle.dump(meta, mf)
                        print(f"[INFO] debug_one_batch: saved model state and meta to {ckpt_path}")
                    except Exception as _e:
                        print(f"[WARN] debug_one_batch: checkpoint saved but meta write failed: {_e}")
                except Exception as _e:
                    print(f"[WARN] debug_one_batch: failed to save checkpoint: {_e}")
            except Exception as _e:
                print(f"[WARN] debug_one_batch: failed to save checkpoint: {_e}")
            return {'val_loss': None, 'val_auc': None, 'val_acc': None, 'val_bacc': None, 'val_f1': None, 'debug_one_batch': True}

        # Weight NaN/Inf check
        try:
            any_nonfinite = False
            for n, p in model.named_parameters():
                if not torch.isfinite(p).all():
                    any_nonfinite = True
                    bad_param_name = n
                    break
            if any_nonfinite:
                base_dir = f"results/debug_nan_weights_epoch{epoch}_batch{i}"
                try:
                    state_dict = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
                except Exception:
                    state_dict = {}
                prepared = _prepare_batch_for_dump(xs=xs, masks=masks, y=y)
                param_meta = {k: getattr(v, 'shape', None) for k, v in state_dict.items()}
                offending = {bad_param_name: state_dict.get(bad_param_name)} if 'bad_param_name' in locals() else {}
                dump_dict = {**prepared, 'param_meta': param_meta, 'offending': offending}
                _dump_debug_and_raise(base_dir, dump_dict, f"Non-finite model parameter detected at epoch={epoch} batch={i} param={bad_param_name}", raise_exc=True)
        except Exception as _e:
            if isinstance(_e, RuntimeError):
                raise
            else:
                print(f"[WARN] weight diagnostic check failed: {_e}")
    return {}


def _eval_epoch(model, val_loader, device, criterion, params, groups, gating_fn, epoch, tau, eps, no_validation=False):
    """Run validation/test evaluation for one epoch and optionally perform
    per-epoch test-eval when retraining-on-full.

    Returns: (val_loss, all_p, all_y, all_gates, printed_test_line, m_override)
    where m_override is a dict of metrics from per-epoch test eval when used
    (or None).
    """
    printed_test_line = False
    m_override = None
    if not no_validation:
        res = _evaluate_loader(model, val_loader, device, criterion, params, groups, gating_fn, tau, eps, epoch, label="val")
        if params.get("gate_type", "softmax") == "gumbel":
            # Monte Carlo averaging for Gumbel gate
            all_p, all_y = predict_mc(model, val_loader, n_samples=params.get("mc_samples", 10), device=device, tau=tau)
            val_loss = float("nan")
            all_gates = []
        else:
            val_loss = res.get("loss", float("nan"))
            all_p = res.get("all_p", [])
            all_y = res.get("all_y", [])
            all_gates = res.get("all_gates", [])
    else:
        # no_validation: placeholders (real eval happens after training)
        all_p = []
        all_y = []
        all_gates = []
        val_loss = float("nan")

    # When retraining-on-full we may want to run per-epoch test eval and
    # print test metrics instead of placeholder val metrics.
    if no_validation and params.get("__retrain_log_test_each_epoch", False):
        try:
            model.eval()
            all_p_test = []
            all_y_test = []
            all_gates_test = []
            all_ptid_test = []
            test_loss = 0.0
            batch_idx = 0
            with torch.no_grad():
                for xs_t, masks_t, y_t in val_loader:
                    xs_t = [x.to(device) for x in xs_t]
                    masks_t, y_t = masks_t.to(device), y_t.to(device)
                    try:
                        logits_t, gate_w_t = model(xs_t, masks_t, tau=tau)
                    except TypeError:
                        logits_t, gate_w_t = model(xs_t, masks_t)
                    # Eval-time NaN/inf guard: dump offending batch but continue
                    try:
                        if not torch.isfinite(logits_t).all() or not torch.isfinite(gate_w_t).all():
                            base_dir = f"results/debug_nan_eval_epoch{epoch}_batch{len(all_p_test)}"
                            prepared = _prepare_batch_for_dump(xs=xs_t, masks=masks_t, y=y_t, logits=logits_t, gate=gate_w_t)
                            _dump_debug_and_raise(base_dir, prepared, f"Non-finite values during eval at epoch={epoch}", raise_exc=False)
                    except Exception as _e:
                        print(f"[WARN] eval diagnostic check failed: {_e}")
                    try:
                        loss_t = criterion(logits_t, y_t)
                        test_loss += float(loss_t.item()) * y_t.size(0)
                    except Exception:
                        pass
                    all_p_test.append(F.softmax(logits_t, dim=1).cpu().numpy())
                    all_y_test.append(y_t.cpu().numpy())
                    # Collect gate weights: apply masking and normalization
                    if gating_fn:
                        gw = gating_fn(gate_w_t, masks_t)
                    else:
                        gw = gate_w_t * masks_t
                        gw = gw / (gw.sum(dim=1, keepdim=True) + eps)
                    all_gates_test.append(gw.cpu().numpy())
                    # Collect PTIDs for this batch
                    batch_size = y_t.size(0)
                    start_idx = batch_idx * val_loader.batch_size
                    end_idx = start_idx + batch_size
                    batch_ptids = df.iloc[val_idx[start_idx:end_idx]]["PTID"].astype(str).values
                    all_ptid_test.extend(batch_ptids)
                    batch_idx += 1

            if all_p_test:
                all_p_test = np.vstack(all_p_test)
                all_y_test = np.concatenate(all_y_test)
                if all_gates_test:
                    all_gates_test = np.vstack(all_gates_test)
                test_loss = test_loss / (len(val_loader.dataset) if len(val_loader.dataset) > 0 else 1)
            else:
                all_p_test = np.zeros((0, params.get("n_classes", 3)))
                all_y_test = np.array([])
                all_gates_test = np.zeros((0, len(groups) if groups else 0))
                all_ptid_test = []

            # Optionally dump per-epoch test proba/y/gates for diagnostics
            try:
                base_path = None
                if params.get("save_checkpoint"):
                    base_path = os.path.splitext(params.get("save_checkpoint"))[0]
                elif params.get("out_json"):
                    base_path = os.path.splitext(params.get("out_json"))[0]
                else:
                    base_path = os.path.join("results", "retrain_diag")

                np.save(f"{base_path}_epoch{epoch}_proba.npy", all_p_test)
                np.save(f"{base_path}_epoch{epoch}_y.npy", all_y_test)
                if all_gates_test is not None and len(all_gates_test) > 0:
                    np.save(f"{base_path}_epoch{epoch}_gates.npy", all_gates_test)
                    # Save per-subject gate weights with labels
                    if len(all_ptid_test) == len(all_y_test):
                        y_pred = np.argmax(all_p_test, axis=1)
                        per_subject_data = []
                        mod_names = list(groups.keys())
                        for i in range(len(all_ptid_test)):
                            subject_record = {
                                "PTID": str(all_ptid_test[i]),
                                "y_true": int(all_y_test[i]),
                                "y_pred": int(y_pred[i]),
                                "proba": [float(p) for p in all_p_test[i]],
                                "gate_weights": {mod_names[j]: float(all_gates_test[i][j]) for j in range(len(mod_names))}
                            }
                            per_subject_data.append(subject_record)
                        import json
                        with open(f"{base_path}_epoch{epoch}_per_subject.json", "w") as f:
                            json.dump(per_subject_data, f, indent=2)
                        print(f"[DEBUG] dumped per-epoch proba/y/gates/per_subject to {base_path}_epoch{epoch}_*.npy/*.json")
                    else:
                        print(f"[DEBUG] dumped per-epoch proba/y/gates to {base_path}_epoch{epoch}_*.npy")
                else:
                    print(f"[DEBUG] dumped per-epoch proba/y to {base_path}_epoch{epoch}_*.npy")
            except Exception:
                pass

            m_test = utils.eval_multiclass_metrics(all_y_test, all_p_test)
            # Use the test metrics in the epoch-level printout so users see
            # meaningful numbers when retraining on full.
            printed_test_line = True
            m_override = {
                "auc": m_test.get("auc", float("nan")),
                "acc": m_test.get("acc", float("nan")),
                "bacc": m_test.get("bacc", float("nan")),
                "f1": m_test.get("f1", float("nan")),
            }
        except Exception as e:
            print(f"[WARN] per-epoch test eval failed at epoch {epoch}: {e}")
            printed_test_line = True
    return val_loss, all_p, all_y, all_gates, printed_test_line, m_override

def run_once(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    params: dict,
    train_idx=None,
    val_idx=None,
    gating_fn=None,
    no_validation: bool = False,
):
    """Train and evaluate the MoE model once with fixed train/val indices."""
    # -------------------------
    # Device setup
    # -------------------------
    # Device/setup helpers keep run_once compact and make unit testing easier
    device = _setup_device()
    # from here on, all tensors/models are explicitly moved to `device`

    # -------------------------
    # Split setup
    # -------------------------
    if train_idx is None or val_idx is None:
        groupsplit = GroupShuffleSplit(
            n_splits=1, train_size=0.8, random_state=utils.SEED
        )
        ptids = df["PTID"].astype(str).values
        (train_idx, val_idx), = groupsplit.split(df, groups=ptids)
        print(
            f"[INFO] using random GroupShuffleSplit: train={len(train_idx)} | val={len(val_idx)}"
        )
    else:
        print(f"[INFO] using fixed splits: train={len(train_idx)} | val={len(val_idx)}")

    # --- Handle dataset setup based on ablation type ---
    if params.get("gate_ablation") == "modality_only":
        print("[INFO] Building modality-level dataset for 3-expert flat MoE")
        for prefix in ["mri", "amy", "demographic"]:
            region_cols = [c for c in df.columns if c.startswith(f"has_{prefix}_")]
            if region_cols:
                df[f"has_{prefix}"] = df[region_cols].max(axis=1)
        groups_modality = {}
        for prefix in ["mri", "amy", "demographic"]:
            matched = [cols for k, cols in groups.items() if k.startswith(prefix)]
            if matched:
                groups_modality[prefix] = sum(matched, [])
        ds_train = MoEDataset(df.iloc[train_idx], groups_modality)
        ds_val = MoEDataset(df.iloc[val_idx], groups_modality, scalers=ds_train.scalers)
        groups = groups_modality
    else:
        ds_train = MoEDataset(df.iloc[train_idx], groups)
        ds_val = MoEDataset(df.iloc[val_idx], groups, scalers=ds_train.scalers)

    # Cache the raw validation dataframe in the same order as ds_val
    df_val_raw = df.iloc[val_idx].reset_index(drop=True)

    dims = [len(v) for v in groups.values()]

    # --- Model initialization logic ---
    if params.get("gate_ablation") == "region_only":
        print("[INFO] Using flat region-level MoE (region_only ablation)")
        model = MoE(
            dims,
            hidden_exp=params["hidden_exp"],
            hidden_gate=params["hidden_gate"],
            n_classes=3,
            drop=params["drop"],
            gate_type=params.get("gate_type", "softmax"),
            gumbel_hard=params.get("gumbel_hard", False),
            gate_noise=params.get("gate_noise", 0.02),
            topk=params.get("topk", None),
        ).to(device)
    elif params.get("gate_ablation") == "modality_only":
        print("[INFO] Using flat 3-expert modality-level MoE (modality_only ablation)")
        groups_modality = {}
        for prefix in ["mri", "amy", "demographic"]:
            matched = [cols for k, cols in groups.items() if k.startswith(prefix)]
            if matched:
                groups_modality[prefix] = sum(matched, [])
        dims_modality = [len(v) for v in groups_modality.values()]
        model = MoE(
            dims_modality,
            hidden_exp=params["hidden_exp"],
            hidden_gate=params["hidden_gate"],
            n_classes=3,
            drop=params["drop"],
            gate_type=params.get("gate_type", "softmax"),
            gumbel_hard=params.get("gumbel_hard", False),
            gate_noise=params.get("gate_noise", 0.02),
            topk=params.get("topk", None),
        ).to(device)
        groups = groups_modality
    elif params.get("use_hierarchical_gate", False):
        print("[INFO] Using HierarchicalMoE model")
        model = HierarchicalMoE(
            groups,
            hidden_exp=params["hidden_exp"],
            hidden_gate=params["hidden_gate"],
            n_classes=3,
            drop=params["drop"],
            gate_type=params.get("gate_type", "softmax"),
            gumbel_hard=params.get("gumbel_hard", False),
            gate_noise=params.get("gate_noise", 0.02),
            topk=params.get("topk", None),
        ).to(device)
    else:
        model = MoE(
            dims,
            hidden_exp=params["hidden_exp"],
            hidden_gate=params["hidden_gate"],
            n_classes=3,
            drop=params["drop"],
            gate_type=params.get("gate_type", "softmax"),
            gumbel_hard=params.get("gumbel_hard", False),
            gate_noise=params.get("gate_noise", 0.02),
            topk=params.get("topk", None),
        ).to(device)

    # -------------------------
    # DataLoaders (factored)
    # -------------------------
    ds_train, ds_val, train_loader, val_loader, B, num_workers, pin_mem = _build_dataloaders(
        df, groups, train_idx, val_idx, params, device
    )
    # expose collate in this scope for later payload-building which expects
    # the name `collate` (keeps compatibility with legacy code paths)
    collate = _collate

    # -------------------------
    # Optimization setup (factored)
    # -------------------------
    criterion, opt, scheduler = _init_optim_and_criterion(model, params, df, train_idx, device)

    # -------------------------
    # Training loop
    # -------------------------
    best_metrics = {
        "val_loss": float("inf"),
        "val_auc": float("-inf"),
        "val_acc": float("-inf"),
        "val_bacc": float("-inf"),
        "val_f1": float("-inf"),
    }
    best_state = None
    best_gates_val = None
    best_val_proba = None
    best_val_y = None
    best_val_ptid = None
    patience, bad = params["patience"], 0
    epoch_gate_means = []
    gate_outputs_val = []
    payload_records = []
    eps = 1e-8

    early_stop_metric = params.get("early_stop_metric", "val_auc")
    valid_metrics = ["val_auc", "val_loss"]
    if early_stop_metric not in valid_metrics:
        print(
            f"[WARN] early_stop_metric '{early_stop_metric}' not recognized, defaulting to 'val_auc'"
        )
        early_stop_metric = "val_auc"
    print(f"[INFO] Early stopping metric: {early_stop_metric}")

    debug_one = bool(params.get("debug_one_batch", False))
    debug_stop = bool(params.get("stop_on_nan", False))
    if debug_one:
        print("[INFO] debug_one_batch enabled: will run one training batch with anomaly detection and exit")
    if debug_stop:
        print("[INFO] stop_on_nan enabled: anomaly detection will be active until first NaN/Inf is observed")
        # enable anomaly detection globally for the training run to capture
        # the first offending autograd op. We'll leave it enabled until exit.
        try:
            torch.autograd.set_detect_anomaly(True)
        except Exception:
            pass

    for epoch in range(1, params["epochs"] + 1):
        # ---- training ----
        # compute tau schedule for this epoch so eval can reuse the same value
        try:
            tau = max(params["tau"], params["tau_start"] * (params["tau_decay"] ** epoch))
        except Exception:
            tau = params.get("tau", 0.9)
        res = _train_epoch(model, train_loader, device, opt, criterion, params, gating_fn, epoch, debug_one=debug_one, debug_stop=debug_stop, groups=groups)
        if res.get("debug_one_batch", False):
            return res

        # ---- validation ----
        val_loss, all_p, all_y, all_gates, printed_test_line, m_override = _eval_epoch(
            model, val_loader, device, criterion, params, groups, gating_fn, epoch, tau, eps, no_validation=no_validation
        )

        try:
            # Log sizes/shapes to help debug empty or malformed prediction arrays
            ly = len(all_y) if hasattr(all_y, "__len__") else None
            lp = len(all_p) if hasattr(all_p, "__len__") else None
            print(f"[DEBUG] before eval_multiclass_metrics: len(all_y)={ly}, len(all_p)={lp}")
            if lp and lp > 0:
                try:
                    a0 = np.asarray(all_p[0])
                    print(f"[DEBUG] sample all_p[0].shape={a0.shape}")
                except Exception:
                    pass
        except Exception as _:
            pass
        # If we're in no_validation mode we expect the per-epoch placeholders
        # to be empty; avoid calling the metrics on empty lists which just
        # produces noisy warnings — return sensible defaults instead.
        if no_validation and (not all_p):
            print("[DEBUG] no_validation mode: skipping per-epoch metrics computation (placeholders empty)")
            m = {"bacc": 0.0, "auc": 0.5, "acc": 0.0, "f1": 0.0}
        else:
            m = utils.eval_multiclass_metrics(all_y, all_p)
        val_auc, val_acc, val_bacc, val_f1 = (
            m["auc"],
            m["acc"],
            m["bacc"],
            m["f1"],
        )

        # If _eval_epoch provided a per-epoch test metrics override (retrain-on-full), use it
        if m_override is not None:
            try:
                val_auc = float(m_override.get("auc", val_auc))
                val_acc = float(m_override.get("acc", val_acc))
                val_bacc = float(m_override.get("bacc", val_bacc))
                val_f1 = float(m_override.get("f1", val_f1))
            except Exception:
                pass

        cur_lr = opt.param_groups[0]["lr"]

        # If this run is a retrain-on-full invocation and the internal flag is
        # set, run the per-epoch test evaluation first and use those numbers in
        # the main epoch log line so we show the test metrics (not placeholder
        # val metrics) during retrain-on-full. This avoids printing misleading
        # NaNs when the validation set is empty.
        printed_test_line = False
        if no_validation and params.get("__retrain_log_test_each_epoch", False):
            try:
                model.eval()
                all_p_test = []
                all_y_test = []
                test_loss = 0.0
                with torch.no_grad():
                    for xs_t, masks_t, y_t in val_loader:
                        xs_t = [x.to(device) for x in xs_t]
                        masks_t, y_t = masks_t.to(device), y_t.to(device)
                        # use the same tau as this epoch's training forward pass
                        try:
                            logits_t, gate_w_t = model(xs_t, masks_t, tau=tau)
                        except TypeError:
                            # fallback if model signature doesn't accept tau
                            logits_t, gate_w_t = model(xs_t, masks_t)
                        # Eval-time NaN/inf guard: dump offending batch but continue
                        try:
                            if not torch.isfinite(logits_t).all() or not torch.isfinite(gate_w_t).all():
                                os.makedirs("results", exist_ok=True)
                                base_dir = f"results/debug_nan_eval_epoch{epoch}_batch{len(all_p_test)}"
                                prepared = _prepare_batch_for_dump(xs=xs_t, masks=masks_t, y=y_t, logits=logits_t, gate=gate_w_t)
                                try:
                                    _save_debug_dir(base_dir, prepared)
                                    print(f"[WARN] Non-finite values during eval at epoch={epoch}; dumped to {base_dir}")
                                except Exception as _e:
                                    print(f"[WARN] failed to write eval debug dir {base_dir}: {_e}")
                        except Exception as _e:
                            print(f"[WARN] eval diagnostic check failed: {_e}")
                        try:
                            loss_t = criterion(logits_t, y_t)
                            test_loss += float(loss_t.item()) * y_t.size(0)
                        except Exception:
                            pass
                        all_p_test.append(F.softmax(logits_t, dim=1).cpu().numpy())
                        all_y_test.append(y_t.cpu().numpy())

                if all_p_test:
                    all_p_test = np.vstack(all_p_test)
                    all_y_test = np.concatenate(all_y_test)
                    test_loss = test_loss / (len(val_loader.dataset) if len(val_loader.dataset) > 0 else 1)
                else:
                    all_p_test = np.zeros((0, params.get("n_classes", 3)))
                    all_y_test = np.array([])

                # Diagnostic dump: save per-epoch test probabilities/labels so
                # we can diff them against the final evaluation later.
                try:
                    if params.get("__retrain_log_test_each_epoch", False):
                        base_path = None
                        # prefer save_checkpoint or out_json to derive a base name
                        if params.get("save_checkpoint"):
                            base_path = os.path.splitext(params.get("save_checkpoint"))[0]
                        elif params.get("out_json"):
                            base_path = os.path.splitext(params.get("out_json"))[0]
                        else:
                            base_path = os.path.join("results", "retrain_diag")

                        np.save(f"{base_path}_epoch{epoch}_proba.npy", all_p_test)
                        np.save(f"{base_path}_epoch{epoch}_y.npy", all_y_test)
                        print(f"[DEBUG] dumped per-epoch proba/y to {base_path}_epoch{epoch}_*.npy")
                except Exception as _:
                    pass

                m_test = utils.eval_multiclass_metrics(all_y_test, all_p_test)
                # Use the test metrics in the epoch-level printout so users see
                # meaningful numbers when retraining on full.
                val_auc = m_test.get("auc", float("nan"))
                val_acc = m_test.get("acc", float("nan"))
                val_bacc = m_test.get("bacc", float("nan"))
                val_f1 = m_test.get("f1", float("nan"))
                print(
                    f"Epoch {epoch:03d} | test_loss={test_loss:.4f} | test_auc={val_auc:.4f} | "
                    f"test_acc={val_acc:.4f} | test_bacc={val_bacc:.4f} | test_f1={val_f1:.4f} | lr={cur_lr:.2e}"
                )
                printed_test_line = True
            except Exception as e:
                print(f"[WARN] per-epoch test eval failed at epoch {epoch}: {e}")
                print(
                    f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | auc={val_auc:.4f} | "
                    f"acc={val_acc:.4f} | bacc={val_bacc:.4f} | f1={val_f1:.4f} | lr={cur_lr:.2e}"
                )
                printed_test_line = True
        else:
            # defer printing to the centralized logger below (avoids duplication)
            pass
        

        if all_gates:
            G = np.vstack(all_gates)
            mean_w = G.mean(axis=0)
            mod_names = list(groups.keys())
            print("    gate mean weights (val):")
            for name, w in zip(mod_names, mean_w):
                print(f"      {name:25s}: {w:.3f}")

        if all_gates:
            G_epoch = np.vstack(all_gates)
            mean_w_epoch = G_epoch.mean(axis=0)
            epoch_gate_means.append(
                {
                    "epoch": epoch,
                    "weights": {
                        name: float(mean_w_epoch[i])
                        for i, name in enumerate(groups.keys())
                    },
                }
            )

        # Centralized construction and printing of epoch metrics
        cur_metrics = _log_epoch_metrics(
            epoch,
            val_loss,
            val_auc,
            val_acc,
            val_bacc,
            val_f1,
            opt,
            params,
            groups,
            all_gates,
            printed_test_line=printed_test_line,
        )

        update_best = False
        metric_value = cur_metrics.get(early_stop_metric, None)
        best_metric_value = _finite(
            best_metrics.get(
                early_stop_metric,
                0 if early_stop_metric != "val_loss" else float("inf"),
            )
        )

        if all(
            [not math.isnan(v) and not math.isinf(v) for v in cur_metrics.values()]
        ):
            if early_stop_metric == "val_loss":
                if metric_value < best_metric_value:
                    update_best = True
            else:
                if metric_value > best_metric_value:
                    update_best = True

        if best_state is None or update_best:
            bad = 0
            best_metrics = cur_metrics.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # record the epoch where the best validation snapshot was taken
            try:
                best_epoch = int(epoch)
            except Exception:
                best_epoch = None
            if all_gates:
                best_gates_val = np.vstack(all_gates)
            else:
                best_gates_val = None

            # Save per-subject validation predictions for this best epoch
            try:
                best_val_proba = np.array(all_p, copy=True)
                best_val_y = np.array(all_y, copy=True)
                best_val_ptid = df.iloc[val_idx]["PTID"].astype(str).values
            except Exception as e:
                print(f"[WARN] could not cache best val predictions: {e}")
                best_val_proba = None
                best_val_y = None
                best_val_ptid = None

            if best_gates_val is not None:
                PTIDs = df.iloc[val_idx]["PTID"].astype(str).values
                ys = df.iloc[val_idx]["y"].astype(int).values
                gate_outputs_val = []
                for i, ptid in enumerate(PTIDs):
                    weights = best_gates_val[i % len(best_gates_val)]
                    weights_dict = {
                        name: float(w) for name, w in zip(groups.keys(), weights)
                    }
                    gate_outputs_val.append(
                        {"PTID": ptid, "y": int(ys[i]), "weights": weights_dict}
                    )
        else:
            bad += 1
            # Only trigger early stopping if not explicitly disabled
            if (not params.get("no_early_stopping", False)) and bad >= patience:
                print("Early stopping.")
                break

        scheduler.step(val_loss)

    # ---- restore best weights ----
    # When running retrain-on-full (no_validation=True) we want the final
    # evaluation to reflect the model state after retraining on train+val.
    # Restoring `best_state` here would overwrite the retrained weights with
    # the earlier best-validation snapshot; only restore when not
    # retraining-on-full.
    if best_state is not None and (not no_validation):
        model.load_state_dict(best_state)

    # If we skipped validation during training, run a single evaluation now and
    # record final metrics and best_epoch as the full-training epoch count.
    if no_validation:
        # When retraining on full, best_epoch should already be set from the original trial.
        # If it wasn't set, default to the full epoch count.
        if 'best_epoch' not in locals() or best_epoch is None:
            try:
                best_epoch = int(params["epochs"])
            except Exception:
                best_epoch = None
        
        # run evaluation on val_loader (which in retrain-on-full usage is the test set)
        model.eval()
        all_y, all_p, all_gates = [], [], []
        all_ptid_final = []
        val_loss = 0.0
        # Debugging: report dataset size and will count processed batches
        try:
            ds_len = len(val_loader.dataset)
        except Exception:
            ds_len = None
        print(f"[DEBUG] final eval (no_validation): val_loader.dataset length={ds_len}")
        batch_count = 0
        final_batch_idx = 0
        with torch.no_grad():
            for xs, masks, y in tqdm(val_loader, desc="Final eval", leave=False):
                batch_count += 1
                try:
                    print(f"[DEBUG] final eval batch {batch_count}: y.shape={getattr(y,'shape',None)}")
                except Exception:
                    pass
                xs = [x.to(device) for x in xs]
                masks, y = masks.to(device), y.to(device)
                # Use the same tau schedule as training for final evaluation.
                try:
                    tau_final = max(
                        params.get("tau", 0.9),
                        params.get("tau_start", params.get("tau", 0.9))
                        * (params.get("tau_decay", 1.0) ** params.get("epochs", 1)),
                    )
                except Exception:
                    tau_final = params.get("tau", 0.9)
                logits, gate_w_val = model(xs, masks, tau=tau_final)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                proba = F.softmax(logits, dim=1).cpu().numpy()
                all_p.append(proba)
                all_y.append(y.cpu().numpy())
                if gating_fn:
                    gw = gating_fn(gate_w_val, masks)
                else:
                    gw = gate_w_val * masks
                    gw = gw / (gw.sum(dim=1, keepdim=True) + eps)
                all_gates.append(gw.cpu().numpy())
                # Collect PTIDs for final evaluation
                batch_size = y.size(0)
                start_idx = final_batch_idx * val_loader.batch_size
                end_idx = start_idx + batch_size
                batch_ptids = df.iloc[val_idx[start_idx:end_idx]]["PTID"].astype(str).values
                all_ptid_final.extend(batch_ptids)
                final_batch_idx += 1
        if params.get("gate_type", "softmax") == "gumbel":
            all_p, all_y = predict_mc(
                model,
                val_loader,
                n_samples=params.get("mc_samples", 10),
                device=device,
                tau=tau_final,
            )
            val_loss = float("nan")
        else:
            # Guard against empty dataset to avoid ZeroDivisionError
            try:
                ds_len_local = len(val_loader.dataset)
            except Exception:
                ds_len_local = 0
            if ds_len_local > 0:
                val_loss /= ds_len_local
                all_p = np.vstack(all_p) if all_p else np.zeros((ds_len_local, params.get("n_classes", 3)))
                all_y = np.concatenate(all_y) if all_y else np.array([])
            else:
                val_loss = float("nan")
                all_p = np.zeros((0, params.get("n_classes", 3)))
                all_y = np.array([])
        try:
            print(f"[DEBUG] final eval processed batches={batch_count}, all_p_list_len={len(all_p) if hasattr(all_p,'__len__') else None}, all_y_len={len(all_y) if hasattr(all_y,'__len__') else None}")
        except Exception:
            pass
        # If we performed a final evaluation (no_validation path), compute and
        # record the actual metrics from the evaluated predictions so callers
        # (e.g., the tuner) receive correct test metrics instead of placeholders.
        if no_validation:
            try:
                # all_p is an array of shape (N, C) and all_y is (N,)
                m_eval = utils.eval_multiclass_metrics(all_y, all_p)
            except Exception as e:
                print(f"[WARN] could not compute final eval metrics: {e}")
                m_eval = {"auc": 0.5, "acc": 0.0, "bacc": 0.0, "f1": 0.0}
            # Diagnostic dump: save final evaluation probabilities and labels
            try:
                base_path = None
                if params.get("save_checkpoint"):
                    base_path = os.path.splitext(params.get("save_checkpoint"))[0]
                elif params.get("out_json"):
                    base_path = os.path.splitext(params.get("out_json"))[0]
                else:
                    base_path = os.path.join("results", "retrain_diag_final")

                np.save(f"{base_path}_final_proba.npy", all_p)
                np.save(f"{base_path}_final_y.npy", all_y)
                
                # Save per-subject gate weights with predictions and labels
                if len(all_gates) > 0:
                    all_gates_stacked = np.vstack(all_gates)
                    np.save(f"{base_path}_final_gates.npy", all_gates_stacked)
                    
                    if len(all_ptid_final) == len(all_y):
                        y_pred = np.argmax(all_p, axis=1)
                        per_subject_data = []
                        mod_names = list(groups.keys())
                        for i in range(len(all_ptid_final)):
                            subject_record = {
                                "PTID": str(all_ptid_final[i]),
                                "y_true": int(all_y[i]),
                                "y_pred": int(y_pred[i]),
                                "proba": [float(p) for p in all_p[i]],
                                "gate_weights": {mod_names[j]: float(all_gates_stacked[i][j]) for j in range(len(mod_names))}
                            }
                            per_subject_data.append(subject_record)
                        import json
                        with open(f"{base_path}_final_per_subject.json", "w") as f:
                            json.dump(per_subject_data, f, indent=2)
                        print(f"[DEBUG] dumped final eval proba/y/gates/per_subject to {base_path}_final_*.npy/*.json")
                    else:
                        print(f"[DEBUG] dumped final eval proba/y/gates to {base_path}_final_*.npy")
                else:
                    print(f"[DEBUG] dumped final eval proba/y to {base_path}_final_*.npy")
            except Exception as e:
                print(f"[WARN] could not save final per-subject data: {e}")
            # Overwrite best_metrics with the actual evaluated metrics from the
            # final no-validation evaluation so these appear in the returned JSON.
            try:
                best_metrics["val_loss"] = float(val_loss) if (val_loss is not None) else float("nan")
                best_metrics["val_auc"] = float(m_eval.get("auc", 0.0))
                best_metrics["val_acc"] = float(m_eval.get("acc", 0.0))
                best_metrics["val_bacc"] = float(m_eval.get("bacc", 0.0))
                best_metrics["val_f1"] = float(m_eval.get("f1", 0.0))
            except Exception:
                pass
        # set best_epoch to the full-training epoch count if not already set
        if 'best_epoch' not in locals() or best_epoch is None:
            try:
                best_epoch = int(params["epochs"])
            except Exception:
                best_epoch = None
    confusion_report = None
    if best_val_y is not None and best_val_proba is not None:
        try:
            confusion_report = eval_confusion_report(best_val_y, best_val_proba, class_names=["CN", "MCI", "AD"])
        except Exception as e:
            print(f"[WARN] could not compute confusion_report: {e}")
            confusion_report = None

    # ---- build per-sample payload on best model (all patients, fixed global order) ----
    if params.get("save_payloads", False):
        payload_records = _build_payloads(
            model,
            df,
            groups,
            ds_train.scalers,
            device,
            B,
            num_workers,
            pin_mem,
            collate,
            params,
            train_idx,
            val_idx,
            params.get("fold_idx", None),
        )

    # ---- serialize interpretable gate mean weights for JSON ----
    def _json_safe(x):
        try:
            if x is None or (
                isinstance(x, float) and (math.isnan(x) or math.isinf(x))
            ):
                return 0.0
            elif hasattr(x, "__float__"):
                x = float(x)
                if math.isnan(x) or math.isinf(x):
                    return 0.0
                return x
            else:
                return 0.0
        except Exception:
            return 0.0

    gate_mean_weights = {}
    if best_gates_val is not None:
        mean_w = best_gates_val.mean(axis=0)
        mod_names = list(groups.keys())
        for name, w in zip(mod_names, mean_w):
            gate_mean_weights[name] = float(_json_safe(w))
        
        # Save gate weights to numpy file for later analysis
        try:
            gates_path = None
            if params.get("save_checkpoint"):
                gates_path = os.path.splitext(params.get("save_checkpoint"))[0] + "_gates.npy"
            elif params.get("out_json"):
                gates_path = os.path.splitext(params.get("out_json"))[0] + "_gates.npy"
            if gates_path:
                np.save(gates_path, best_gates_val)
                print(f"[INFO] saved gate weights to {gates_path}")
        except Exception as e:
            print(f"[WARN] could not save gate weights: {e}")

    safe_metrics = {k: _json_safe(v) for k, v in best_metrics.items()}

    out = {
        **safe_metrics,
        "gate_mean_weights": gate_mean_weights,
        "gate_epoch_traj": epoch_gate_means,
        "gate_outputs_val": gate_outputs_val,
        "confusion_report": confusion_report,
        "payload": payload_records,
        "best_epoch": locals().get("best_epoch", None),
    }

    # Optional: save per-subject best-epoch validation predictions for later plotting
    if params.get("save_val_predictions", False) and best_val_y is not None and best_val_proba is not None:
        try:
            y_pred = np.argmax(best_val_proba, axis=1)
            val_predictions = []
            ptids = best_val_ptid if best_val_ptid is not None else df.iloc[val_idx]["PTID"].astype(str).values
            for i, ptid in enumerate(ptids):
                val_predictions.append({
                    "PTID": str(ptid),
                    "y": int(best_val_y[i]),
                    "y_pred": int(y_pred[i]),
                    "proba": [float(x) for x in best_val_proba[i].tolist()],
                })
            out["val_predictions"] = val_predictions
        except Exception as e:
            print(f"[WARN] could not serialize val_predictions: {e}")

    # -------------------------
    # Optional: save checkpoint + meta for eval-only protocol
    # -------------------------
    if params.get("save_checkpoint", None) is not None:
        ckpt_path = params.get("save_checkpoint")
        try:
            # choose state to save: best_state (if available) else current model
            if best_state is not None:
                state_to_save = best_state
            else:
                state_to_save = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            _save_checkpoint_and_meta(ckpt_path, state_to_save, ds_train, groups, params)
        except Exception as e:
            print(f"[WARN] could not save checkpoint to {ckpt_path}: {e}")

    return out




# ----------------------------
# Entry
# ----------------------------
def main():
    set_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experts_config", help="YAML mapping expert_name -> CSV path"
    )
    ap.add_argument(
        "--splits",
        default=None,
        help="Path to fixed split JSON (train_pool_indices, test_indices)",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of DataLoader workers (0 = main process)",
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--hidden_exp", type=int, default=128)
    ap.add_argument("--hidden_gate", type=int, default=128)
    ap.add_argument(
        "--lambda_sparse",
        type=float,
        default=0.2,
        help="Weight for gate entropy penalty (smaller = gentler).",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Gate temperature (>1 = softer)",
    )
    ap.add_argument(
        "--gate_type",
        type=str,
        default="softmax",
        choices=["softmax", "gumbel"],
        help="Type of gating: softmax (default) or gumbel-softmax.",
    )
    ap.add_argument(
        "--mc_samples",
        type=int,
        default=10,
        help="Number of Monte Carlo samples for Gumbel inference averaging",
    )
    ap.add_argument(
        "--lambda_diverse",
        type=float,
        default=0.01,
        help="Weight for diversity penalty (larger -> more decorrelated experts)",
    )
    ap.add_argument(
        "--tau_start",
        type=float,
        default=1.0,
        help="Starting temperature before annealing",
    )
    ap.add_argument(
        "--tau_decay",
        type=float,
        default=0.995,
        help="Per-epoch multiplicative decay for temperature",
    )
    ap.add_argument(
        "--gate_noise",
        type=float,
        default=0.02,
        help="Std of Gaussian gate logits noise during training",
    )
    ap.add_argument(
        "--gumbel_hard",
        action="store_true",
        help="Use straight-through one-hot gating for Gumbel gate",
    )
    ap.add_argument(
        "--split_type",
        type=str,
        default="holdout",
        choices=["holdout", "cv5", "train_val_test"],
        help="Split type: 'holdout', 'cv5' or 'train_val_test' (default: holdout)",
    )
    # Insert --only_fold argument here
    ap.add_argument(
        "--only_fold",
        type=int,
        default=None,
        help="If set (0-indexed), run only this CV fold for --split_type=cv5 (for parallel CV).",
    )
    ap.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping (<=0 disables clipping).",
    )
    ap.add_argument(
        "--debug_one_batch",
        action="store_true",
        help="Run a single-batch forward/backward with anomaly detection and dump debug tensors (then exit).",
    )
    ap.add_argument(
        "--stop_on_nan",
        action="store_true",
        help="Keep anomaly detection on and continue training until the first NaN/Inf is observed, then dump debug artifacts and exit.",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Path to output JSON file for results (auto-named if not specified)",
    )
    ap.add_argument(
        "--save_checkpoint",
        type=str,
        default=None,
        help="Optional path to save best model state_dict (torch) for eval-only runs.",
    )
    ap.add_argument(
        "--retrain_on_full",
        action="store_true",
        help="If set, retrain best model on train+val for best_epoch before saving checkpoint (not implemented fully).",
    )
    ap.add_argument(
        "--retrain_only",
        action="store_true",
        help="If set, skip the initial train->val run and only perform retrain-on-full (train+val -> test). Intended for tuner-invoked retrain.",
    )
    ap.add_argument(
        "--eval_only",
        action="store_true",
        help="If set, do not train; load --ckpt and evaluate on the validation set for the specified --only_fold.",
    )
    ap.add_argument(
        "--eval_ckpt",
        action="store_true",
        help="If set, load --ckpt and evaluate on the test split for --split_type=train_val_test (no training).",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint file (torch .pt) to load when --eval_only is set.",
    )
    ap.add_argument(
        "--early_stop_metric",
        type=str,
        default="val_loss",
        choices=["val_auc", "val_loss"],
        help="Metric to use for early stopping: 'val_auc' (default) or 'val_loss'.",
    )
    ap.add_argument(
        "--no_early_stopping",
        action="store_true",
        help="If set, disable early stopping and train for exactly --epochs epochs. Useful for fixed-epoch evaluation in tuning or retrain phases.",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=None,
        nargs="?",
        const=2,
        help="If set without value, uses topk=2. If set with value (e.g., --topk 1), uses that value only. If omitted, uses full gating.",
    )
    ap.add_argument(
        "--use_hierarchical_gate",
        action="store_true",
        help="Use HierarchicalMoE model with hierarchical gating",
    )
    ap.add_argument(
        "--gate_ablation",
        type=str,
        default=None,
        choices=["region_only", "modality_only", "random"],
        help="Perform gate ablation: region_only, modality_only, or random",
    )
    ap.add_argument(
        "--save_payloads",
        action="store_true",
        help="If set, construct per-sample validation payloads for LLM explanation.",
    )
    ap.add_argument(
        "--redact_ptid",
        action="store_true",
        help="If set, hash PTID values in payloads to avoid exposing identifiers.",
    )
    ap.add_argument(
        "--redact_values",
        action="store_true",
        help="If set, round probabilities/gates in payloads.",
    )
    ap.add_argument(
        "--save_val_predictions",
        action="store_true",
        help="If set, include per-subject best-epoch validation probabilities/predictions (PTID, y, y_pred, proba) in the output JSON.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = main()
    print("[INFO] starting MoE training…")

    set_seed()

    df, groups, classes = load_experts_from_yaml(args.experts_config)
    mods_str = ", ".join([f"{m}={len(cols)}" for m, cols in groups.items()])
    print(f"[INFO] rows={len(df)} | feats: {mods_str}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/runs", exist_ok=True)
    # Call centralized dispatcher and exit. The original inline branches remain
    # in this file as a safe fallback but will not be executed when the
    # dispatcher returns.
    _dispatch_mode(args, df, groups, classes)
    sys.exit(0)

    # -------------------------
    # Eval-only path: load checkpoint + meta and evaluate on masked experts YAML
    # -------------------------
    if args.eval_only:
        if args.ckpt is None:
            raise ValueError("--eval_only requires --ckpt to be set to a checkpoint file path")
        if args.only_fold is None:
            raise ValueError("--eval_only requires --only_fold to specify which fold to evaluate")

        # Load splits to map fold -> val indices
        with open(args.splits, "r") as f:
            splits_json = json.load(f)
        if "cv_splits_ptid" not in splits_json:
            raise ValueError("splits JSON must contain 'cv_splits_ptid' for --split_type=cv5")
        cv_splits = splits_json["cv_splits_ptid"]
        fold_idx = args.only_fold
        split = cv_splits[fold_idx]
        train_keys = [kk for kk in split.keys() if "train" in kk.lower()]
        val_keys = [kk for kk in split.keys() if "val" in kk.lower()]
        if not val_keys:
            raise KeyError(f"Fold {fold_idx}: missing val keys in split: {list(split.keys())}")

        val_ptids = set(split[val_keys[0]])
        val_idx = df[df["PTID"].astype(str).isin(val_ptids)].index.tolist()

        # Load meta.pkl produced by training on clean fold
        meta_path = os.path.splitext(args.ckpt)[0] + ".meta.pkl"
        meta = {}
        if os.path.isfile(meta_path):
            try:
                import pickle
                with open(meta_path, "rb") as mf:
                    meta = pickle.load(mf)
            except Exception as e:
                print(f"[WARN] could not load meta file {meta_path}: {e}")

        # Use saved scalers (fit on clean train) if available
        scalers = meta.get("scalers", None)

        # Decide which tau to use: prefer explicit CLI override; otherwise use
        # the tau saved in the checkpoint meta if present.
        try:
            cli_specified_tau = any([a == "--tau" or a.startswith("--tau=") for a in sys.argv])
        except Exception:
            cli_specified_tau = False
        tau_to_use = args.tau if cli_specified_tau else meta.get("train_params", {}).get("tau", args.tau)

        # Reconstruct model using saved model_config if present
        model_cfg = meta.get("model_config", {})

        # Build validation dataset with saved scalers
        ds_val = MoEDataset(df.iloc[val_idx], groups, scalers=scalers)
        B = args.batch_size
        num_workers = int(args.num_workers)
        pin_mem = torch.cuda.is_available()
        val_loader = DataLoader(ds_val, batch_size=B, shuffle=False, collate_fn=lambda b: ( [torch.stack([x[0][m] for x in b],dim=0) for m in range(len(b[0][0]))], torch.stack([x[1] for x in b],dim=0), torch.tensor([x[2] for x in b],dtype=torch.long) ), pin_memory=pin_mem, num_workers=num_workers)

        # Build model
        dims = [len(v) for v in groups.values()]
        use_hier = model_cfg.get("use_hierarchical_gate", False) or args.use_hierarchical_gate
        if use_hier:
            model = HierarchicalMoE(
                groups,
                hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                n_classes=3,
                drop=model_cfg.get("drop", args.drop),
                gate_type=model_cfg.get("gate_type", args.gate_type),
                gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                topk=model_cfg.get("topk", args.topk),
            ).to(torch.device("cpu"))
        else:
            model = MoE(
                dims,
                hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                n_classes=3,
                drop=model_cfg.get("drop", args.drop),
                gate_type=model_cfg.get("gate_type", args.gate_type),
                gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                topk=model_cfg.get("topk", args.topk),
            ).to(torch.device("cpu"))

        # Load state
        if not os.path.isfile(args.ckpt):
            raise ValueError(f"Checkpoint not found: {args.ckpt}")
        state = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        # Run inference on validation set
        all_p = []
        all_y = []
        with torch.no_grad():
            for xs, masks, y in val_loader:
                xs = [x for x in xs]
                logits, gate_w = model([x for x in xs], masks, tau=tau_to_use)
                proba = F.softmax(logits, dim=1).cpu().numpy()
                all_p.append(proba)
                all_y.append(y.numpy())

        if all_p:
            all_p = np.vstack(all_p)
            all_y = np.concatenate(all_y)
        else:
            all_p = np.zeros((0, 3))
            all_y = np.array([])

        try:
            ly = len(all_y) if hasattr(all_y, "__len__") else None
            lp = len(all_p) if hasattr(all_p, "__len__") else None
            print(f"[DEBUG] eval_only: len(all_y)={ly}, len(all_p)={lp}")
            if lp and lp > 0:
                try:
                    a0 = np.asarray(all_p[0])
                    print(f"[DEBUG] eval_only sample all_p[0].shape={a0.shape}")
                except Exception:
                    pass
        except Exception:
            pass
        m = utils.eval_multiclass_metrics(all_y, all_p)
        out = {
            "val_auc": float(m.get("auc", float("nan"))),
            "val_acc": float(m.get("acc", float("nan"))),
            "val_bacc": float(m.get("bacc", float("nan"))),
            "val_f1": float(m.get("f1", float("nan"))),
            "proba": all_p.tolist(),
            "yva": all_y.tolist(),
        }

        out_path = args.out_json or f"results/moe_eval_fold{args.only_fold}.json"
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] wrote eval-only results to {out_path}")
        sys.exit(0)
    # ---- Evaluate a provided checkpoint on the test split (no training) ----
    if args.eval_ckpt:
        if args.ckpt is None:
            raise ValueError("--eval_ckpt requires --ckpt to be set to a checkpoint file path")
        # Expect train_val_test splits so we can evaluate on explicit test indices
        if args.split_type != "train_val_test":
            raise ValueError("--eval_ckpt currently supports only --split_type train_val_test")

        splits = _prepare_splits(args, df)
        if splits.get("type") != "train_val_test":
            raise ValueError("--eval_ckpt requires --split_type=train_val_test and a 3-way splits JSON")
        tr_idx, va_idx, te_idx = splits["tr_idx"], splits["va_idx"], splits["te_idx"]

        # Load meta (scalers, groups, model_config) from ckpt.meta.pkl
        meta = {}
        meta_path = os.path.splitext(args.ckpt)[0] + ".meta.pkl"
        if os.path.isfile(meta_path):
            try:
                import pickle

                with open(meta_path, "rb") as mf:
                    meta = pickle.load(mf)
            except Exception as e:
                print(f"[WARN] could not load meta file {meta_path}: {e}")

        scalers = meta.get("scalers", None)
        model_cfg = meta.get("model_config", {})

        # Build test dataset and loader
        ds_test = MoEDataset(df.iloc[te_idx], meta.get("groups", groups), scalers=scalers)
        B = args.batch_size
        num_workers = int(args.num_workers)
        pin_mem = torch.cuda.is_available()
        def _collate(batch):
            M = len(batch[0][0])
            xs = [torch.stack([b[0][m] for b in batch], dim=0) for m in range(M)]
            masks = torch.stack([b[1] for b in batch], dim=0)
            y = torch.tensor([b[2] for b in batch], dtype=torch.long)
            return xs, masks, y

        test_loader = DataLoader(ds_test, batch_size=B, shuffle=False, collate_fn=_collate, pin_memory=pin_mem, num_workers=num_workers)

        # Reconstruct model
        dims = [len(v) for v in meta.get("groups", groups).values()]
        use_hier = model_cfg.get("use_hierarchical_gate", False) or args.use_hierarchical_gate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_hier:
            model = HierarchicalMoE(
                meta.get("groups", groups),
                hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                n_classes=3,
                drop=model_cfg.get("drop", args.drop),
                gate_type=model_cfg.get("gate_type", args.gate_type),
                gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                topk=model_cfg.get("topk", args.topk),
            ).to(device)
        else:
            model = MoE(
                dims,
                hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                n_classes=3,
                drop=model_cfg.get("drop", args.drop),
                gate_type=model_cfg.get("gate_type", args.gate_type),
                gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                topk=model_cfg.get("topk", args.topk),
            ).to(device)

        # Load checkpoint
        if not os.path.isfile(args.ckpt):
            raise ValueError(f"Checkpoint not found: {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state)
        model.eval()

        # Run inference on test set
        all_p = []
        all_y = []
        # decide tau for eval: prefer CLI-specified, otherwise use meta train_params
        try:
            cli_specified_tau = any([a == "--tau" or a.startswith("--tau=") for a in sys.argv])
        except Exception:
            cli_specified_tau = False
        tau_to_use = args.tau if cli_specified_tau else meta.get("train_params", {}).get("tau", args.tau)

        with torch.no_grad():
            for xs, masks, y in test_loader:
                xs = [x.to(device) for x in xs]
                masks = masks.to(device)
                logits, gate_w = model(xs, masks, tau=tau_to_use)
                proba = F.softmax(logits, dim=1).cpu().numpy()
                all_p.append(proba)
                all_y.append(y.numpy())

        if all_p:
            all_p = np.vstack(all_p)
            all_y = np.concatenate(all_y)
        else:
            all_p = np.zeros((0, 3))
            all_y = np.array([])

        mtest = utils.eval_multiclass_metrics(all_y, all_p)
        out = {
            "test_auc": float(mtest.get("auc", float("nan"))),
            "test_acc": float(mtest.get("acc", float("nan"))),
            "test_bacc": float(mtest.get("bacc", float("nan"))),
            "test_f1": float(mtest.get("f1", float("nan"))),
            "proba": all_p.tolist(),
            "yte": all_y.tolist(),
        }

        out_path = args.out_json or f"results/moe_eval_ckpt.json"
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[INFO] wrote eval-ckpt results to {out_path}")
        sys.exit(0)
    if args.split_type == "holdout":
        splits = _prepare_splits(args, df)
        if splits.get("type") != "holdout":
            raise ValueError("Expected holdout splits for split_type=holdout")
        train_pos, val_pos, _test_pos = splits["train_pos"], splits["val_pos"], splits.get("test_pos", None)
        tr_ptids = set(df.iloc[train_pos]["PTID"].astype(str))
        va_ptids = set(df.iloc[val_pos]["PTID"].astype(str))
        leak = tr_ptids.intersection(va_ptids)
        if leak:
            print(
                f"[WARN] {len(leak)} PTIDs appear in both train and val; check your splits JSON."
            )

        if args.topk:
            for k in [5, 3, 1]:
                print(f"\n[INFO] Running top-{k} ablation...")
                set_seed()
                params = dict(vars(args))
                params["topk"] = k
                out = run_once(
                    df,
                    groups,
                    params,
                    train_idx=train_pos,
                    val_idx=val_pos,
                    gating_fn=lambda gw, m, top_k=k: apply_topk_gating(
                        gw, m, top_k=top_k
                    ),
                )

                print(f"[INFO] best val macro-AUROC (top-{k}):", out["val_auc"])
                out_path = f"results/runs/moe_results_top{k}.json"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(out, f, indent=2)

                if args.save_payloads and "payload" in out:
                    run_name = f"moe_holdout_top{k}"
                    payload_dir = os.path.join("results", "llm_payloads", run_name)
                    os.makedirs(payload_dir, exist_ok=True)
                    payload_path = os.path.join(payload_dir, "samples.jsonl")
                    with open(payload_path, "w") as pf:
                        for rec in out["payload"]:
                            pf.write(json.dumps(rec) + "\n")
        else:
            params = dict(vars(args))
            params["topk"] = None
            out = run_once(
                df, groups, params, train_idx=train_pos, val_idx=val_pos
            )

            print("[INFO] best val macro-AUROC:", out["val_auc"])
            out_path = args.out_json or "results/moe_results_lastvisit.json"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[INFO] wrote results to {out_path}")

            if args.save_payloads and "payload" in out:
                run_name = os.path.splitext(os.path.basename(out_path))[0]
                payload_dir = os.path.join("results", "llm_payloads", run_name)
                os.makedirs(payload_dir, exist_ok=True)
                payload_path = os.path.join(payload_dir, "samples.jsonl")
                with open(payload_path, "w") as pf:
                    for rec in out["payload"]:
                        pf.write(json.dumps(rec) + "\n")

    elif args.split_type == "train_val_test":
        # Expect a 3-way splits JSON (train_ptids / val_ptids / test_ptids) or index-based equivalent
        splits = _prepare_splits(args, df)
        if splits.get("type") != "train_val_test":
            raise ValueError("train_val_test requires a 3-way splits JSON (train/val/test)")
        tr_idx, va_idx, te_idx = splits["tr_idx"], splits["va_idx"], splits["te_idx"]

        print(f"[INFO] train={len(tr_idx)} | val={len(va_idx)} | test={len(te_idx)}")

        # 1) Tune / train on (train -> val)
        params = dict(vars(args))
        params["topk"] = None
        # Save best epoch/state for later evaluation or retrain-on-full
        base_name = os.path.splitext(os.path.basename(args.out_json or "results/optuna_moe"))[0]
        ckpt_best = os.path.join("results", f"{base_name}_best.pt")
        # Respect an explicit --save_checkpoint provided by the caller (e.g., the tuner).
        # Only set a default checkpoint path if none was specified on the command line.
        if not params.get("save_checkpoint"):
            params["save_checkpoint"] = ckpt_best

        if getattr(args, "retrain_only", False):
            # Tuner-invoked retrain: skip the initial train->val phase and only
            # perform retrain-on-full (train+val -> test). This avoids running
            # the duplicate initial training when the tuner already ran tuning.
            print("[INFO] --retrain_only set: skipping initial train->val (tune phase)")
            out_val = {}
        else:
            print(f"[INFO] Running train->val and saving best checkpoint to {ckpt_best}")
            out_val = run_once(df, groups, params, train_idx=tr_idx, val_idx=va_idx)

        # Write intermediate results (validation) if requested
        if args.out_json:
            out_base = args.out_json
        else:
            out_base = os.path.join("results", f"{base_name}_trainval.json")
        os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
        with open(out_base, "w") as f:
            json.dump({"train_val": out_val}, f, indent=2)
        print(f"[INFO] wrote train/val results to {out_base}")

        # 2) Evaluate on test: either retrain on train+val or evaluate saved best checkpoint
        if args.retrain_on_full:
            print("[INFO] Retraining on train+val and evaluating on test...")
            params_full = dict(vars(args))
            params_full["topk"] = None
            # save final full-model checkpoint
            ckpt_full = os.path.join("results", f"{base_name}_full.pt")
            params_full["save_checkpoint"] = ckpt_full
            # Internal flag: during retrain-on-full, log test metrics each epoch
            # for debugging. We do this automatically (no CLI arg required).
            params_full["__retrain_log_test_each_epoch"] = True
            out_full = run_once(
                df,
                groups,
                params_full,
                train_idx=np.concatenate([tr_idx, va_idx]),
                val_idx=te_idx,
                no_validation=True,
            )
            # map val->test keys
            test_metrics = {
                "test_auc": float(out_full.get("val_auc", float("nan"))),
                "test_acc": float(out_full.get("val_acc", float("nan"))),
                "test_bacc": float(out_full.get("val_bacc", float("nan"))),
                "test_f1": float(out_full.get("val_f1", float("nan"))),
            }
            final_out = {**out_val, **test_metrics}
            # include confusion_report from out_full if available
            if "confusion_report" in out_full:
                final_out["confusion_report_test"] = out_full["confusion_report"]
        else:
            print(f"[INFO] Evaluating saved best checkpoint {ckpt_best} on test set...")
            # load meta if present to reconstruct model and scalers
            meta = {}
            meta_path = os.path.splitext(ckpt_best)[0] + ".meta.pkl"
            if os.path.isfile(meta_path):
                try:
                    import pickle

                    with open(meta_path, "rb") as mf:
                        meta = pickle.load(mf)
                except Exception as e:
                    print(f"[WARN] could not load meta file {meta_path}: {e}")

            scalers = meta.get("scalers", None)
            model_cfg = meta.get("model_config", {})

            # Build test dataset and loader
            ds_test = MoEDataset(df.iloc[te_idx], groups, scalers=scalers)
            B = args.batch_size
            num_workers = int(args.num_workers)
            pin_mem = torch.cuda.is_available()
            # local collate (same semantics as run_once.collate)
            def _collate(batch):
                M = len(batch[0][0])
                xs = [torch.stack([b[0][m] for b in batch], dim=0) for m in range(M)]
                masks = torch.stack([b[1] for b in batch], dim=0)
                y = torch.tensor([b[2] for b in batch], dtype=torch.long)
                return xs, masks, y

            test_loader = DataLoader(ds_test, batch_size=B, shuffle=False, collate_fn=_collate, pin_memory=pin_mem, num_workers=num_workers)

            # Reconstruct model from meta
            dims = [len(v) for v in groups.values()]
            use_hier = model_cfg.get("use_hierarchical_gate", False) or args.use_hierarchical_gate
            if use_hier:
                model = HierarchicalMoE(
                    groups,
                    hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                    hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                    n_classes=3,
                    drop=model_cfg.get("drop", args.drop),
                    gate_type=model_cfg.get("gate_type", args.gate_type),
                    gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                    gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                    topk=model_cfg.get("topk", args.topk),
                ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            else:
                model = MoE(
                    dims,
                    hidden_exp=model_cfg.get("hidden_exp", args.hidden_exp),
                    hidden_gate=model_cfg.get("hidden_gate", args.hidden_gate),
                    n_classes=3,
                    drop=model_cfg.get("drop", args.drop),
                    gate_type=model_cfg.get("gate_type", args.gate_type),
                    gumbel_hard=model_cfg.get("gumbel_hard", args.gumbel_hard),
                    gate_noise=model_cfg.get("gate_noise", args.gate_noise),
                    topk=model_cfg.get("topk", args.topk),
                ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Load weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            if os.path.isfile(ckpt_best):
                state = torch.load(ckpt_best, map_location=device)
                model.load_state_dict(state)
                model.eval()
            else:
                print(f"[WARN] checkpoint {ckpt_best} not found; skipping test evaluation")

            # Run inference on test set
            all_p = []
            all_y = []
            with torch.no_grad():
                for xs, masks, y in test_loader:
                    xs = [x.to(device) for x in xs]
                    masks = masks.to(device)
                    # choose tau from CLI unless absent, then fall back to meta
                    try:
                        cli_specified_tau = any([a == "--tau" or a.startswith("--tau=") for a in sys.argv])
                    except Exception:
                        cli_specified_tau = False
                    tau_to_use = args.tau if cli_specified_tau else meta.get("train_params", {}).get("tau", args.tau)
                    logits, gate_w = model(xs, masks, tau=tau_to_use)
                    proba = F.softmax(logits, dim=1).cpu().numpy()
                    all_p.append(proba)
                    all_y.append(y.numpy())

            if all_p:
                all_p = np.vstack(all_p)
                all_y = np.concatenate(all_y)
            else:
                all_p = np.zeros((0, 3))
                all_y = np.array([])

            mtest = utils.eval_multiclass_metrics(all_y, all_p)
            test_metrics = {
                "test_auc": float(mtest.get("auc", float("nan"))),
                "test_acc": float(mtest.get("acc", float("nan"))),
                "test_bacc": float(mtest.get("bacc", float("nan"))),
                "test_f1": float(mtest.get("f1", float("nan"))),
            }

            final_out = {**out_val, **test_metrics}

        # write final JSON
        out_path = args.out_json or os.path.join("results", f"{base_name}_trainval_test.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(final_out, f, indent=2)
        print(f"[INFO] wrote train/val/test results to {out_path}")

    elif args.split_type == "cv5":
        with open(args.splits, "r") as f:
            splits_json = json.load(f)
        if "cv_splits_ptid" not in splits_json:
            raise ValueError(
                "splits JSON must contain 'cv_splits_ptid' for --split_type=cv5"
            )
        cv_splits = splits_json["cv_splits_ptid"]

        k_values = [5, 3, 1] if args.topk else [None]

        for k in k_values:
            label = f"top-{k}" if k is not None else "full"
            out_suffix = f"top-{k}" if k is not None else "full"
            if args.gate_ablation:
                out_suffix = f"{args.gate_ablation}_{out_suffix}"
            print(
                f"\n[INFO] ===== Running CV for {label} gating {'with top-' + str(k) + ' ' if k is not None else ''}====="
            )
            set_seed()
            results_folds = []
            all_payloads = []  # across folds for this k

            fold_indices = (
                [args.only_fold]
                if args.only_fold is not None
                else list(range(len(cv_splits)))
            )

            for fold_idx in fold_indices:
                # For cv5 we expect the cv_splits entries to contain train/val/test keys
                # (but we will use train/val here). If users want train_val_test behavior
                # they should use --split_type train_val_test which loads explicit 3-way splits.
                split = cv_splits[fold_idx]
                split = cv_splits[fold_idx]
                print(
                    f"\n[INFO] Running fold {fold_idx + 1}/{len(cv_splits)} ({label})"
                )

                train_keys = [kk for kk in split.keys() if "train" in kk.lower()]
                val_keys = [kk for kk in split.keys() if "val" in kk.lower()]
                if not train_keys or not val_keys:
                    raise KeyError(
                        f"Fold {fold_idx}: missing train/val keys in {list(split.keys())}"
                    )

                train_ptids = set(split[train_keys[0]])
                val_ptids = set(split[val_keys[0]])

                train_idx = df[df["PTID"].astype(str).isin(train_ptids)].index.tolist()
                val_idx = df[df["PTID"].astype(str).isin(val_ptids)].index.tolist()

                tr_ptids = set(df.iloc[train_idx]["PTID"].astype(str))
                va_ptids = set(df.iloc[val_idx]["PTID"].astype(str))
                leak = tr_ptids.intersection(va_ptids)
                if leak:
                    print(
                        f"[WARN] {len(leak)} PTIDs appear in both train and val in fold {fold_idx+1}"
                    )

                params = dict(vars(args))
                params["topk"] = k
                params["fold_idx"] = fold_idx

                if k is not None:
                    out = run_once(
                        df,
                        groups,
                        params,
                        train_idx=train_idx,
                        val_idx=val_idx,
                        gating_fn=lambda gw, m, top_k=k: apply_topk_gating(
                            gw, m, top_k=top_k
                        ),
                    )
                else:
                    out = run_once(
                        df,
                        groups,
                        params,
                        train_idx=train_idx,
                        val_idx=val_idx,
                    )

                out_fold = {"fold": fold_idx, "top_k": k, **out}
                results_folds.append(out_fold)

                # Single-fold mode (parallel CV): write per-fold JSON and exit
                if args.only_fold is not None:
                    # If user provided --out_json, use it; otherwise default to results/runs/cv_folds/fold{fold}.json
                    if args.out_json:
                        fold_out_path = args.out_json
                    else:
                        fold_out_dir = os.path.join("results", "runs", "cv_folds")
                        os.makedirs(fold_out_dir, exist_ok=True)
                        fold_out_path = os.path.join(fold_out_dir, f"fold{fold_idx}.json")

                    os.makedirs(os.path.dirname(fold_out_path) or ".", exist_ok=True)
                    with open(fold_out_path, "w") as f:
                        json.dump(out_fold, f, indent=2)
                    print(f"[INFO] wrote fold result to {fold_out_path}")

                    # Note: payload JSONL writing above is still performed if --save_payloads is set.
                    sys.exit(0)
                print(
                    f"[INFO] Fold {fold_idx+1} ({label}) val macro-AUROC: {out['val_auc']:.4f}"
                )

                if args.save_payloads and "payload" in out:
                    # Per-fold JSONL with fixed global order
                    if args.out_json:
                        base_name = os.path.splitext(os.path.basename(args.out_json))[0]
                        run_name = f"{base_name}_{out_suffix}"
                    else:
                        run_name = f"moe_hierarchical_cv10_{out_suffix}"
                    payload_dir = os.path.join("results", "llm_payloads", run_name)
                    os.makedirs(payload_dir, exist_ok=True)
                    payload_path = os.path.join(
                        payload_dir, f"samples_fold{fold_idx}.jsonl"
                    )
                    with open(payload_path, "w") as pf:
                        for rec in out["payload"]:
                            pf.write(json.dumps(rec) + "\n")

                if args.save_payloads and "payload" in out:
                    for rec in out["payload"]:
                        rec_with_fold = dict(rec)
                        rec_with_fold["fold"] = fold_idx
                        all_payloads.append(rec_with_fold)

            metric_names = ["val_auc", "val_acc", "val_bacc", "val_f1"]
            metrics = {m: [fold[m] for fold in results_folds] for m in metric_names}
            summary = {}
            for m in metric_names:
                arr = np.array(metrics[m])
                summary[f"{m}_mean"] = float(np.mean(arr))
                summary[f"{m}_std"] = float(np.std(arr))

            # ----------------------------
            # Aggregate confusion matrix across folds
            # ----------------------------
            agg_confusion_report = None
            try:
                cms = []
                for fold in results_folds:
                    cr = fold.get("confusion_report", None)
                    if cr is None:
                        continue
                    cm_counts = cr.get("cm_counts", None)
                    if cm_counts is None:
                        continue
                    cms.append(np.array(cm_counts, dtype=int))

                if len(cms) > 0:
                    cm_total = np.sum(np.stack(cms, axis=0), axis=0).astype(int)

                    # Row-normalized
                    row_sums = cm_total.sum(axis=1, keepdims=True).astype(float)
                    row_sums[row_sums == 0.0] = 1.0
                    cm_row = cm_total / row_sums

                    # Precision/recall/F1 from aggregated counts
                    tp = np.diag(cm_total).astype(float)
                    col_sums = cm_total.sum(axis=0).astype(float)
                    prec = np.divide(tp, col_sums, out=np.zeros_like(tp), where=(col_sums > 0))
                    rec = np.divide(tp, row_sums.squeeze(1), out=np.zeros_like(tp), where=(row_sums.squeeze(1) > 0))
                    f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(tp), where=((prec + rec) > 0))

                    class_names = ["CN", "MCI", "AD"]
                    agg_confusion_report = {
                        "cm_counts": cm_total.tolist(),
                        "cm_row_norm": cm_row.tolist(),
                        "per_class": {
                            "precision": {class_names[i]: float(prec[i]) for i in range(len(class_names))},
                            "recall": {class_names[i]: float(rec[i]) for i in range(len(class_names))},
                            "f1": {class_names[i]: float(f1[i]) for i in range(len(class_names))},
                        },
                    }
            except Exception as e:
                print(f"[WARN] could not aggregate confusion matrices across folds: {e}")
                agg_confusion_report = None

            print(
                f"\n[INFO] Cross-validation results for {label} gating (mean ± std):"
            )
            for m in metric_names:
                print(
                    f"    {m}: {summary[f'{m}_mean']:.4f} ± {summary[f'{m}_std']:.4f}"
                )


            if args.out_json:
                base, ext = os.path.splitext(args.out_json)
                out_path = f"{base}_{out_suffix}{ext or '.json'}"
            else:
                out_path = f"results/moe_hierarchical_cv10_{out_suffix}.json"

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({
                    "folds": results_folds,
                    "summary": summary,
                    "confusion_report_all_folds": agg_confusion_report,
                }, f, indent=2)
            print(f"[INFO] wrote results to {out_path}")

            if args.save_payloads and all_payloads:
                base_name = os.path.splitext(os.path.basename(out_path))[0]
                run_name = base_name
                payload_dir = os.path.join("results", "llm_payloads", run_name)
                os.makedirs(payload_dir, exist_ok=True)
                payload_path = os.path.join(payload_dir, "samples.jsonl")
                with open(payload_path, "w") as pf:
                    for rec in all_payloads:
                        pf.write(json.dumps(rec) + "\n")
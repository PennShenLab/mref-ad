#!/usr/bin/env python3
"""
Train/test Flex-MoE on this repository's experts YAML and PTID split JSONs.

Example (10 seeds with matching split files):
python scripts/train_flex_moe.py \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml \
  --splits_template configs/splits_by_ptid_80_10_10_seed_{seed}.json \
  --seeds 7 13 42 1234 2027 99 123 555 999 1337 \
  --out_json results/flex_moe_10seeds.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import utils


def parse_args():
    ap = argparse.ArgumentParser(description="Run Flex-MoE with repo-native splits/data")
    ap.add_argument("--experts_config", required=True, help="YAML mapping expert_name -> CSV path")
    ap.add_argument("--splits_template", required=True, help="Template path with {seed}, e.g. configs/splits_by_ptid_80_10_10_seed_{seed}.json")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="Seed list for train/test runs")
    ap.add_argument("--out_json", default="results/flex_moe_results.json")
    ap.add_argument("--save_dir", default="results/flex_moe_ckpts")
    ap.add_argument("--flex_moe_root", default="third_party/flex-moe")

    # Hyperparameters aligned with upstream Flex-MoE args.
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--modality", type=str, default="AMD", help="A=amyloid, M=mri, D=demographic")
    ap.add_argument("--preprocessed", action="store_true", help="Kept for compatibility; not used in repo-native loader")
    ap.add_argument("--initial_filling", type=str, default="mean", help="Kept for compatibility; not used in repo-native loader")
    ap.add_argument("--train_epochs", type=int, default=20)
    ap.add_argument("--warm_up_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--num_patches", type=int, default=16)
    ap.add_argument("--num_experts", type=int, default=8)
    ap.add_argument("--num_routers", type=int, default=1)
    ap.add_argument("--num_layers_enc", type=int, default=1)
    ap.add_argument("--num_layers_fus", type=int, default=1)
    ap.add_argument("--num_layers_pred", type=int, default=1)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--use_common_ids", type=lambda x: str(x).lower() == "true", default=False)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--gate_loss_weight", type=float, default=1e-2)
    ap.add_argument("--save", type=lambda x: str(x).lower() == "true", default=True)
    return ap.parse_args()


@dataclass
class SeedData:
    x_by_mod: Dict[str, np.ndarray]
    observed: np.ndarray
    mc_index: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class FlexDataset(Dataset):
    def __init__(self, x_by_mod, observed, mc_index, y, indices):
        self.mods = list(x_by_mod.keys())
        self.x_by_mod = {m: x_by_mod[m][indices] for m in self.mods}
        self.observed = observed[indices]
        self.mc_index = mc_index[indices]
        self.y = y[indices]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {m: self.x_by_mod[m][idx] for m in self.mods}
        return sample, int(self.y[idx]), int(self.mc_index[idx]), self.observed[idx]


def collate_fn(batch):
    samples, ys, mcs, obs = zip(*batch)
    mods = samples[0].keys()
    x = {m: torch.tensor(np.stack([s[m] for s in samples]), dtype=torch.float32) for m in mods}
    y = torch.tensor(ys, dtype=torch.long)
    mc = torch.tensor(mcs, dtype=torch.long)
    observed = torch.tensor(np.stack(obs), dtype=torch.bool)
    return x, y, mc, observed


def modality_groups(groups: Dict[str, List[str]], selected_letters: str) -> Dict[str, List[str]]:
    selected_letters = selected_letters.upper()
    mapping = {"A": "amy", "M": "mri", "D": "demographic"}
    want = {mapping[c] for c in selected_letters if c in mapping}
    out = {k: [] for k in ["amy", "mri", "demographic"] if k in want}
    for expert, cols in groups.items():
        name = expert.lower()
        if name.startswith("amy_") and "amy" in out:
            out["amy"].extend(cols)
        elif name.startswith("mri_") and "mri" in out:
            out["mri"].extend(cols)
        elif name == "demographic" and "demographic" in out:
            out["demographic"].extend(cols)
    out = {k: list(dict.fromkeys(v)) for k, v in out.items() if len(v) > 0}
    if len(out) == 0:
        raise ValueError("No modality columns found for requested --modality")
    return out


def get_modality_combinations(symbols: List[str]) -> Dict[str, int]:
    from itertools import combinations

    all_combinations = []
    for i in range(len(symbols), 0, -1):
        all_combinations.extend(combinations(symbols, i))
    return {"".join(sorted(c)): idx for idx, c in enumerate(all_combinations)}


def build_seed_data(args, df, mod_cols: Dict[str, List[str]], split_path: str) -> SeedData:
    with open(split_path, "r") as f:
        split = json.load(f)
    for key in ("train_ptids", "val_ptids", "test_ptids"):
        if key not in split:
            raise ValueError(f"{split_path} missing key: {key}")

    ptid = df["PTID"].astype(str).str.strip()
    tr = np.where(ptid.isin([str(x).strip() for x in split["train_ptids"]]).to_numpy())[0]
    va = np.where(ptid.isin([str(x).strip() for x in split["val_ptids"]]).to_numpy())[0]
    te = np.where(ptid.isin([str(x).strip() for x in split["test_ptids"]]).to_numpy())[0]

    y = df["y"].astype(int).to_numpy()
    x_by_mod = {}
    observed = np.zeros((len(df), len(mod_cols)), dtype=bool)

    mod_keys = list(mod_cols.keys())
    for i, mod in enumerate(mod_keys):
        cols = [c for c in mod_cols[mod] if c in df.columns]
        if not cols:
            raise ValueError(f"No columns found for modality '{mod}'")
        x_raw = df[cols].astype(float)
        observed[:, i] = x_raw.notna().any(axis=1).to_numpy()
        # Fit scaler on train only (after temporary fill), transform all.
        x_fill = x_raw.fillna(x_raw.mean())
        scaler = StandardScaler()
        scaler.fit(x_fill.iloc[tr].to_numpy())
        x_scaled = scaler.transform(x_fill.to_numpy()).astype(np.float32)
        x_by_mod[mod] = x_scaled

    sym = {"amy": "A", "mri": "M", "demographic": "D"}
    symbols = [sym[m] for m in mod_keys]
    combo_to_idx = get_modality_combinations(symbols)
    mc = []
    for r in range(observed.shape[0]):
        present = sorted([symbols[c] for c in range(len(symbols)) if observed[r, c]])
        key = "".join(present)
        mc.append(combo_to_idx.get(key, -1))
    mc = np.asarray(mc, dtype=np.int64)

    keep = np.where(mc >= 0)[0]
    x_by_mod = {m: arr[keep] for m, arr in x_by_mod.items()}
    observed = observed[keep]
    mc = mc[keep]
    y = y[keep]

    old_to_new = {old_i: new_i for new_i, old_i in enumerate(keep.tolist())}
    tr = np.asarray([old_to_new[i] for i in tr if i in old_to_new], dtype=np.int64)
    va = np.asarray([old_to_new[i] for i in va if i in old_to_new], dtype=np.int64)
    te = np.asarray([old_to_new[i] for i in te if i in old_to_new], dtype=np.int64)

    return SeedData(x_by_mod=x_by_mod, observed=observed, mc_index=mc, y=y, train_idx=tr, val_idx=va, test_idx=te)


def make_loaders(seed_data: SeedData, args):
    tr_ds = FlexDataset(seed_data.x_by_mod, seed_data.observed, seed_data.mc_index, seed_data.y, seed_data.train_idx)
    va_ds = FlexDataset(seed_data.x_by_mod, seed_data.observed, seed_data.mc_index, seed_data.y, seed_data.val_idx)
    te_ds = FlexDataset(seed_data.x_by_mod, seed_data.observed, seed_data.mc_index, seed_data.y, seed_data.test_idx)
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=collate_fn)
    tr_loader_sorted = DataLoader(tr_ds, shuffle=False, **kwargs)
    tr_loader = DataLoader(tr_ds, shuffle=True, **kwargs)
    va_loader = DataLoader(va_ds, shuffle=False, **kwargs)
    te_loader = DataLoader(te_ds, shuffle=False, **kwargs)
    return tr_loader_sorted, tr_loader, va_loader, te_loader


def run_epoch(args, loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=False, optimizer=None):
    all_preds, all_labels, all_probs = [], [], []
    task_losses, gate_losses = [], []
    if is_training:
        fusion_model.train()
        for enc in encoder_dict.values():
            enc.train()
    else:
        fusion_model.eval()
        for enc in encoder_dict.values():
            enc.eval()

    for batch_samples, batch_labels, batch_mcs, batch_observed in loader:
        batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_mcs = batch_mcs.to(device, non_blocking=True)
        batch_observed = batch_observed.to(device, non_blocking=True)

        fusion_input = []
        for mod, samples in batch_samples.items():
            midx = modality_dict[mod]
            mask = batch_observed[:, midx]
            encoded = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim), device=device)
            if mask.any():
                encoded[mask] = encoder_dict[mod](samples[mask])
            if (~mask).any():
                encoded[~mask] = missing_embeds[batch_mcs[~mask], midx]
            fusion_input.append(encoded)

        out = fusion_model(*fusion_input, expert_indices=batch_mcs)
        if is_training:
            optimizer.zero_grad(set_to_none=True)
            task_loss = criterion(out, batch_labels)
            gate_loss = fusion_model.gate_loss()
            loss = task_loss + args.gate_loss_weight * gate_loss
            loss.backward()
            optimizer.step()
            task_losses.append(float(task_loss.detach().cpu()))
            gate_losses.append(float(gate_loss.detach().cpu()))
        else:
            prob = torch.softmax(out, dim=1)
            pred = prob.argmax(dim=1)
            all_probs.extend(prob.detach().cpu().numpy())
            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(batch_labels.detach().cpu().numpy())

    if is_training:
        return task_losses, gate_losses
    return np.asarray(all_preds), np.asarray(all_labels), np.asarray(all_probs)


def metrics(y_true, probs):
    pred = np.argmax(probs, axis=1)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, average="macro")),
        "auc": float(roc_auc_score(y_true, probs, multi_class="ovr")),
    }


def main():
    args = parse_args()
    if not os.path.isdir(args.flex_moe_root):
        raise FileNotFoundError(f"--flex_moe_root not found: {args.flex_moe_root}")

    flex_abs = os.path.abspath(args.flex_moe_root)
    if flex_abs not in sys.path:
        sys.path.insert(0, flex_abs)

    from models import FlexMoE, PatchEmbeddings  # noqa: E402

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    df, groups, _ = utils.load_experts_from_yaml(args.experts_config)
    mod_cols = modality_groups(groups, args.modality)
    mod_keys = list(mod_cols.keys())
    num_modalities = len(mod_keys)
    modality_dict = {m: i for i, m in enumerate(mod_keys)}
    combo_to_idx = get_modality_combinations([{"amy": "A", "mri": "M", "demographic": "D"}[m] for m in mod_keys])
    full_modality_index = combo_to_idx["".join(sorted([{"amy": "A", "mri": "M", "demographic": "D"}[m] for m in mod_keys]))]

    all_results = {"args": vars(args), "seeds": {}, "summary": {}}
    val_accs, val_f1s, val_aucs, test_accs, test_f1s, test_aucs = [], [], [], [], [], []

    for seed in args.seeds:
        utils.set_seed(seed)
        split_path = args.splits_template.replace("{seed}", str(seed))
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing split file: {split_path}")

        seed_data = build_seed_data(args, df, mod_cols, split_path)
        tr_sorted, tr_loader, va_loader, te_loader = make_loaders(seed_data, args)

        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
        encoder_dict = nn.ModuleDict({
            m: PatchEmbeddings(seed_data.x_by_mod[m].shape[1], args.num_patches, args.hidden_dim).to(device)
            for m in mod_keys
        })
        fusion_model = FlexMoE(
            num_modalities=num_modalities,
            full_modality_index=full_modality_index,
            num_patches=args.num_patches,
            hidden_dim=args.hidden_dim,
            output_dim=3,
            num_layers=args.num_layers_fus,
            num_layers_pred=args.num_layers_pred,
            num_experts=args.num_experts,
            num_routers=args.num_routers,
            top_k=args.top_k,
            num_heads=args.num_heads,
            dropout=args.dropout,
        ).to(device)

        missing_embeds = torch.nn.Parameter(
            torch.randn((2**num_modalities) - 1, num_modalities, args.num_patches, args.hidden_dim, device=device)
        )
        params = list(fusion_model.parameters()) + [p for enc in encoder_dict.values() for p in enc.parameters()] + [missing_embeds]
        optimizer = torch.optim.Adam(params, lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()

        best_val = -1.0
        best_state = None
        for epoch in range(args.train_epochs):
            train_loader_epoch = tr_loader if epoch >= args.warm_up_epochs else tr_sorted
            run_epoch(args, train_loader_epoch, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=True, optimizer=optimizer)
            _, yv, pv = run_epoch(args, va_loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=False)
            vm = metrics(yv, pv)
            if vm["acc"] > best_val:
                best_val = vm["acc"]
                best_state = {
                    "fusion_model": {k: v.detach().cpu() for k, v in fusion_model.state_dict().items()},
                    "encoder_dict": {m: {k: v.detach().cpu() for k, v in enc.state_dict().items()} for m, enc in encoder_dict.items()},
                    "missing_embeds": missing_embeds.detach().cpu(),
                }

        fusion_model.load_state_dict(best_state["fusion_model"])
        for m in mod_keys:
            encoder_dict[m].load_state_dict(best_state["encoder_dict"][m])
        missing_embeds = nn.Parameter(best_state["missing_embeds"].to(device), requires_grad=False)

        _, yv, pv = run_epoch(args, va_loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=False)
        _, yt, pt = run_epoch(args, te_loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=False)
        vm = metrics(yv, pv)
        tm = metrics(yt, pt)

        if args.save:
            ckpt_path = os.path.join(args.save_dir, f"flex_moe_seed_{seed}.pt")
            torch.save(best_state, ckpt_path)
        else:
            ckpt_path = None

        all_results["seeds"][str(seed)] = {
            "split_file": split_path,
            "val_metrics": vm,
            "test_metrics": tm,
            "checkpoint": ckpt_path,
            "n_train": int(len(seed_data.train_idx)),
            "n_val": int(len(seed_data.val_idx)),
            "n_test": int(len(seed_data.test_idx)),
        }

        val_accs.append(vm["acc"])
        val_f1s.append(vm["f1"])
        val_aucs.append(vm["auc"])
        test_accs.append(tm["acc"])
        test_f1s.append(tm["f1"])
        test_aucs.append(tm["auc"])
        print(f"[seed={seed}] VAL acc={vm['acc']:.4f} f1={vm['f1']:.4f} auc={vm['auc']:.4f} | TEST acc={tm['acc']:.4f} f1={tm['f1']:.4f} auc={tm['auc']:.4f}")

    all_results["summary"] = {
        "val_acc_mean": float(np.mean(val_accs)),
        "val_acc_std": float(np.std(val_accs)),
        "val_f1_mean": float(np.mean(val_f1s)),
        "val_f1_std": float(np.std(val_f1s)),
        "val_auc_mean": float(np.mean(val_aucs)),
        "val_auc_std": float(np.std(val_aucs)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std": float(np.std(test_f1s)),
        "test_auc_mean": float(np.mean(test_aucs)),
        "test_auc_std": float(np.std(test_aucs)),
    }

    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[DONE] wrote {args.out_json}")


if __name__ == "__main__":
    main()

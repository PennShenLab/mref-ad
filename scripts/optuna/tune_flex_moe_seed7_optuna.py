#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState


def parse_args():
    ap = argparse.ArgumentParser(description="Optuna tuner for Flex-MoE seed-7 split")
    ap.add_argument("--experts_config", required=True)
    ap.add_argument("--splits_template", required=True, help="Template with {seed}")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--study_name", required=True)
    ap.add_argument("--storage", required=True, help="e.g., sqlite:///results/optuna_flex_moe_seed7.db")
    ap.add_argument("--n_trials", type=int, default=10, help="Trials attempted by this worker")
    ap.add_argument("--total_trials", type=int, default=200, help="Global study trial cap")
    ap.add_argument("--select_metric", choices=["val_acc", "val_f1", "val_auc"], default="val_f1")
    ap.add_argument("--flex_moe_root", default="third_party/flex-moe")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=None)
    return ap.parse_args()


def suggest_params(trial: optuna.Trial):
    train_epochs = trial.suggest_int("train_epochs", 10, 40)
    warm_up_epochs = trial.suggest_int("warm_up_epochs", 2, min(10, max(2, train_epochs - 1)))
    return {
        "train_epochs": train_epochs,
        "warm_up_epochs": warm_up_epochs,
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 96, 128, 192]),
        "top_k": trial.suggest_int("top_k", 1, 4),
        "num_patches": trial.suggest_categorical("num_patches", [8, 16, 32]),
        "num_experts": trial.suggest_categorical("num_experts", [4, 8, 12, 16]),
        "num_routers": 1,
        "num_layers_enc": 1,
        "num_layers_fus": trial.suggest_int("num_layers_fus", 1, 3),
        "num_layers_pred": trial.suggest_int("num_layers_pred", 1, 3),
        "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "gate_loss_weight": trial.suggest_float("gate_loss_weight", 1e-4, 1e-1, log=True),
    }


def objective(args, trial: optuna.Trial) -> float:
    p = suggest_params(trial)
    with tempfile.TemporaryDirectory(prefix="flex_moe_optuna_trial_") as td:
        out_json = Path(td) / "trial_result.json"
        cmd = [
            sys.executable,
            "scripts/train_flex_moe.py",
            "--experts_config",
            args.experts_config,
            "--splits_template",
            args.splits_template,
            "--seeds",
            str(args.seed),
            "--out_json",
            str(out_json),
            "--save_dir",
            str(Path(td) / "ckpt"),
            "--flex_moe_root",
            args.flex_moe_root,
            "--device",
            str(args.device),
            "--modality",
            "AMD",
            "--num_workers",
            str(args.num_workers),
            "--pin_memory",
            "true",
            "--use_common_ids",
            "false",
            "--save",
            "false",
            "--train_epochs",
            str(p["train_epochs"]),
            "--warm_up_epochs",
            str(p["warm_up_epochs"]),
            "--batch_size",
            str(p["batch_size"]),
            "--lr",
            str(p["lr"]),
            "--hidden_dim",
            str(p["hidden_dim"]),
            "--top_k",
            str(p["top_k"]),
            "--num_patches",
            str(p["num_patches"]),
            "--num_experts",
            str(p["num_experts"]),
            "--num_routers",
            str(p["num_routers"]),
            "--num_layers_enc",
            str(p["num_layers_enc"]),
            "--num_layers_fus",
            str(p["num_layers_fus"]),
            "--num_layers_pred",
            str(p["num_layers_pred"]),
            "--num_heads",
            str(p["num_heads"]),
            "--dropout",
            str(p["dropout"]),
            "--gate_loss_weight",
            str(p["gate_loss_weight"]),
        ]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if run.returncode != 0:
            raise RuntimeError(
                f"train_flex_moe failed (code={run.returncode})\nSTDOUT:\n{run.stdout[-2000:]}\nSTDERR:\n{run.stderr[-2000:]}"
            )
        with open(out_json, "r") as f:
            res = json.load(f)
        s = res["summary"]
        metric_map = {
            "val_acc": float(s["val_acc_mean"]),
            "val_f1": float(s["val_f1_mean"]),
            "val_auc": float(s["val_auc_mean"]),
        }
        score = metric_map[args.select_metric]
        trial.set_user_attr("out_json", str(out_json))
        trial.set_user_attr("test_auc_mean", float(s["test_auc_mean"]))
        return score


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    cb = MaxTrialsCallback(args.total_trials, states=(TrialState.COMPLETE,))
    study.optimize(lambda t: objective(args, t), n_trials=args.n_trials, timeout=args.timeout, callbacks=[cb])
    best = {
        "study_name": args.study_name,
        "storage": args.storage,
        "select_metric": args.select_metric,
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "best_trial_number": study.best_trial.number,
    }
    out = Path("results") / f"{args.study_name}_best_trial.json"
    with open(out, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[DONE] wrote {out}")


if __name__ == "__main__":
    main()

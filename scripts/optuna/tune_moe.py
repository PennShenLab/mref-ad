#!/usr/bin/env python3
"""
Optuna wrapper to tune hyperparameters for **mref-ad** (hierarchical / flat MoE), via
``scripts/mref-ad/train_moe.py``.

This script runs that trainer as a subprocess for each trial with sampled
hyperparameters and reads the produced JSON results to extract the validation metric
used as the Optuna objective.

Usage (example):
  python scripts/tune_moe.py \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_80_10_10.json \
  --tune_trials 200 \
  --tune_epochs 50 \
  --batch_size 64 \
  --num_workers 16 \
  --out_base results/optuna_moe_200 \
  --storage sqlite:////home/fzhuang/mref-ad/multimodal-imaging-agents/results/optuna_moe_200.db \
  --gpu_devices 0,1,2,3 \
  --trials_per_gpu 4 \
  --prune_intermediates

python3 scripts/tune_moe.py \
  --experts_config configs/freesurfer_lastvisit_cv10_experts_files.yaml \
  --splits configs/splits/splits_by_ptid_80_10_10.json \
  --tune_trials 2 \
  --tune_epochs 5 \
  --num_workers 16 \
  --trials_per_gpu 10 \
  --gpu_devices "0,1,2,3" \
  --out_base results/optuna_moe \
  --storage sqlite:////home/fzhuang/mref-ad/multimodal-imaging-agents/results/optuna_moe.db \
  --select_metric val_f1
"""
import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict

try:
    import optuna
except Exception as e:
    print("[ERROR] optuna is required for tuning. Please install optuna in your environment.")
    raise


def make_cmd(p: Dict[str, Any], args: argparse.Namespace, out_json: str, ckpt_path: str):
    # Base command calling the training script
    cmd = [sys.executable, "scripts/mref-ad/train_moe.py"]
    cmd += ["--experts_config", args.experts_config]
    cmd += ["--splits", args.splits]
    cmd += ["--split_type", "train_val_test"]
    cmd += ["--epochs", str(args.tune_epochs)]
    # batch size is set from sampled params where available (added later in cmd construction)
    cmd += ["--num_workers", str(args.num_workers)]
    # force hierarchical gate for this wrapper
    cmd += ["--use_hierarchical_gate"]
    # output / checkpoint
    cmd += ["--out_json", out_json]
    cmd += ["--save_checkpoint", ckpt_path]
    # early stopping metric for training (allow training to early-stop by loss while we
    # still select best trial by another metric)
    if getattr(args, "early_stop_metric", None):
        cmd += ["--early_stop_metric", args.early_stop_metric]

    # continuous hyperparams
    cmd += ["--lr", f"{p['lr']:.6g}"]
    cmd += ["--wd", f"{p['wd']:.6g}"]
    cmd += ["--hidden_exp", str(int(p['hidden_exp']))]
    cmd += ["--hidden_gate", str(int(p['hidden_gate']))]
    cmd += ["--drop", f"{p['drop']:.4f}"]
    cmd += ["--lambda_sparse", f"{p['lambda_sparse']:.4f}"]
    cmd += ["--lambda_diverse", f"{p['lambda_diverse']:.5f}"]
    cmd += ["--tau", f"{p['tau']:.4f}"]
    cmd += ["--tau_start", f"{p['tau_start']:.4f}"]
    cmd += ["--tau_decay", f"{p['tau_decay']:.6f}"]
    cmd += ["--gate_noise", f"{p['gate_noise']:.5f}"]

    # boolean flags
    if p.get("gumbel_hard", False):
        cmd.append("--gumbel_hard")

    # batch size: allow tune-time override coming from sampler (p)
    cmd += ["--batch_size", str(int(p.get("batch_size", args.batch_size)))]

    return cmd


def objective(trial: optuna.Trial, args: argparse.Namespace, trial_dir: str, metric: str):
    # sample hyperparameters
    p = {}
    # Use suggest_float/suggest_int (log where appropriate) to avoid deprecated APIs
    p["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    p["wd"] = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    p["hidden_exp"] = trial.suggest_int("hidden_exp", 64, 512, log=True)
    p["hidden_gate"] = trial.suggest_int("hidden_gate", 32, 256, log=True)
    p["drop"] = trial.suggest_float("drop", 0.0, 0.5)
    p["lambda_sparse"] = trial.suggest_float("lambda_sparse", 0.0, 0.2)
    p["lambda_diverse"] = trial.suggest_float("lambda_diverse", 0.0, 0.1)
    p["tau"] = trial.suggest_float("tau", 0.05, 1.0)
    p["tau_start"] = trial.suggest_float("tau_start", 0.5, 1.5)
    p["tau_decay"] = trial.suggest_float("tau_decay", 0.90, 0.999)
    p["gate_noise"] = trial.suggest_float("gate_noise", 0.0, 0.1)
    p["gumbel_hard"] = trial.suggest_categorical("gumbel_hard", [False, True])
    # sample batch size as a categorical choice (small set keeps search compact)
    p["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # prepare output paths for this trial
    out_json = os.path.join(trial_dir, f"trial_{trial.number:04d}.json")
    ckpt_path = os.path.join(trial_dir, f"trial_{trial.number:04d}_best.pt")

    cmd = make_cmd(p, args, out_json, ckpt_path)

    start = time.time()
    try:
        # Run the training script for this trial (sequential)
        print(f"[INFO] Trial {trial.number}: running command: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        dur = time.time() - start
        print(f"[INFO] Trial {trial.number} finished in {dur:.1f}s; returncode={res.returncode}")
        if res.returncode != 0:
            print(f"[WARN] training script returned non-zero exit code for trial {trial.number}")
            # persist subprocess output for offline debugging (helpful for parallel runs)
            try:
                log_base = os.path.join(trial_dir, f"trial_{trial.number:04d}")
                with open(log_base + ".stdout.txt", "w") as lf:
                    lf.write(res.stdout or "")
                with open(log_base + ".stderr.txt", "w") as lf:
                    lf.write(res.stderr or "")
            except Exception:
                pass
            print(res.stdout)
            print(res.stderr)
            # treat as failed trial
            raise optuna.exceptions.TrialPruned()

        # load produced JSON
        if not os.path.isfile(out_json):
            print(f"[WARN] expected output JSON not found: {out_json}")
            # persist any captured stderr/stdout (if subprocess didn't exit cleanly but left output)
            try:
                log_base = os.path.join(trial_dir, f"trial_{trial.number:04d}")
                # if res exists capture
                if 'res' in locals():
                    with open(log_base + ".stdout.txt", "w") as lf:
                        lf.write(res.stdout or "")
                    with open(log_base + ".stderr.txt", "w") as lf:
                        lf.write(res.stderr or "")
            except Exception:
                pass
            raise optuna.exceptions.TrialPruned()

        with open(out_json, "r") as f:
            j = json.load(f)

        # metric extraction: supports val_* or test_* keys
        if metric not in j:
            # sometimes train_val_test writes nested, try top-level keys
            # fallback: look for 'val_f1' or 'val_auc'
            if "val_f1" in j:
                val = float(j["val_f1"])
            elif "val_auc" in j:
                val = float(j["val_auc"])
            else:
                print(f"[WARN] metric {metric} not found in {out_json}; trial pruned")
                raise optuna.exceptions.TrialPruned()
        else:
            val = float(j[metric])

        # report intermediate result to optuna
        trial.report(val, 0)
        return val

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed: {e}")
        # write exception info to per-trial log for debugging
        try:
            import traceback

            log_base = os.path.join(trial_dir, f"trial_{trial.number:04d}")
            with open(log_base + ".exception.txt", "w") as lf:
                lf.write(traceback.format_exc())
        except Exception:
            pass
        raise optuna.exceptions.TrialPruned()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts_config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--tune_trials", type=int, default=50)
    ap.add_argument("--tune_epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_jobs", type=int, default=1, help="Parallel jobs (alias for --n_workers)")
    ap.add_argument("--n_workers", type=int, default=1, help="Number of parallel worker processes (requires --storage for RDB backend)")
    ap.add_argument("--gpu_devices", type=str, default=None, help="Comma-separated GPU device ids to assign to workers (e.g. '0,1,2,3')")
    ap.add_argument("--trials_per_gpu", type=int, default=None, help="If set, spawn this many worker processes per GPU (requires --gpu_devices). This allows running multiple concurrent trials per physical GPU (may oversubscribe memory).")
    ap.add_argument("--early_stop_metric", type=str, default="val_loss", help="Early-stopping metric to pass to training subprocess (e.g., val_loss)")
    ap.add_argument("--select_metric", type=str, default="val_f1", help="Metric to maximize (e.g., val_f1, val_auc)")
    ap.add_argument("--out_base", type=str, default="results/optuna_moe", help="Base path for trial outputs and summary (no extension)")
    ap.add_argument("--storage", type=str, default=None, help="Optuna storage (sqlite path) to persist study and resume")
    ap.add_argument("--study_name", type=str, default=None, help="Optional study name")
    # Auto-retrain should be the default behavior for tuning (tune->retrain->test).
    # Provide a --no_auto_retrain flag to explicitly disable it.
    ap.add_argument("--auto_retrain", dest="auto_retrain", action="store_true", help="Enable automatic retrain of best trial on train+val and evaluate on test (default: enabled)")
    ap.add_argument("--no_auto_retrain", dest="auto_retrain", action="store_false", help="Disable automatic retrain after tuning")
    ap.set_defaults(auto_retrain=True)
    ap.add_argument("--prune_intermediates", action="store_true", help="If set, remove per-trial intermediate files (trial JSONs, checkpoints, CSV, best_trial.json, and local sqlite storage) after producing the final summary JSON. Use with care.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_base) or ".", exist_ok=True)
    trial_dir = args.out_base + "_trials"
    os.makedirs(trial_dir, exist_ok=True)

    direction = "maximize"
    if args.select_metric.lower().endswith("loss"):
        direction = "minimize"

    print(f"[INFO] starting Optuna tuning: trials={args.tune_trials}, epochs={args.tune_epochs}, out={args.out_base}")

    # Parallel tuning: spawn worker processes each calling study.optimize against a
    # shared RDB-backed study. This requires --storage to be set (e.g., sqlite:///db.sqlite).
    if args.n_workers > 1 or args.num_jobs > 1 or (args.trials_per_gpu is not None):
        # Determine desired total worker count. Priority:
        # 1) If trials_per_gpu is set, require gpu_devices and compute workers = len(gpu_list) * trials_per_gpu
        # 2) Else use provided n_workers/num_jobs
        workers = None
        if args.trials_per_gpu is not None:
            if not args.gpu_devices:
                raise ValueError("--trials_per_gpu requires --gpu_devices to be set (e.g. '0,1')")
            gpu_list = [g.strip() for g in args.gpu_devices.split(",") if g.strip()]
            if not gpu_list:
                raise ValueError("no GPUs parsed from --gpu_devices")
            workers = len(gpu_list) * int(args.trials_per_gpu)
            print(f"[INFO] spawning {workers} workers ({args.trials_per_gpu} per GPU across {len(gpu_list)} GPUs)")
        else:
            workers = max(args.n_workers, args.num_jobs)
        if not args.storage:
            raise ValueError("Parallel tuning requires --storage (an RDB URL) to coordinate workers")

        # Normalize study name so all workers join the same study
        study_name_value = args.study_name or (os.path.basename(args.out_base) + "_study")

        # Create the study once in the main process to initialize DB schema before
        # spawning multiple processes (avoids concurrent table-creation races).
        optuna.create_study(direction=direction, study_name=study_name_value, storage=args.storage, load_if_exists=True)

        # split trials across workers
        n_workers = workers
        base = args.tune_trials // n_workers
        rem = args.tune_trials % n_workers
        trials_per_worker = [base + (1 if i < rem else 0) for i in range(n_workers)]

        from multiprocessing import Process

        def worker_main(worker_id: int, n_trials_worker: int):
            # Each worker loads the same study (RDB storage) and optimizes
            study = optuna.create_study(direction=direction, study_name=study_name_value, storage=args.storage, load_if_exists=True)
            # Optionally bind this worker to a specific GPU via CUDA_VISIBLE_DEVICES so
            # subprocess training calls inherit the assignment.
            if args.gpu_devices:
                gpu_list = [g.strip() for g in args.gpu_devices.split(",") if g.strip()]
                if gpu_list:
                    assigned = gpu_list[worker_id % len(gpu_list)]
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned)
                    # limit intra-op threads to reduce CPU oversubscription
                    os.environ.setdefault("OMP_NUM_THREADS", "1")

            print(f"[INFO] worker {worker_id} starting {n_trials_worker} trials (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
            study.optimize(lambda t: objective(t, args, trial_dir, args.select_metric), n_trials=n_trials_worker)
            print(f"[INFO] worker {worker_id} finished")

        procs = []
        for i, nt in enumerate(trials_per_worker):
            if nt <= 0:
                continue
            p = Process(target=worker_main, args=(i, nt))
            p.start()
            procs.append(p)

        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            print("[WARN] parallel tuning interrupted by user; terminating workers")
            for p in procs:
                p.terminate()
                p.join()

        # Load the study for reporting the best trial
        study = optuna.load_study(study_name=study_name_value, storage=args.storage)
    else:
        study = optuna.create_study(direction=direction, study_name=args.study_name, storage=args.storage, load_if_exists=True)
        try:
            study.optimize(lambda t: objective(t, args, trial_dir, args.select_metric), n_trials=args.tune_trials)
        except KeyboardInterrupt:
            print("[WARN] tuning interrupted by user")

    print("[INFO] tuning completed")
    # Guard against the case where all trials were pruned or no trials have a
    # recorded objective value in the RDB; Optuna will not populate best_trial
    # in that case. Select the best trial only from trials that reported a value
    # to avoid storage-related exceptions and to provide better debugging hints.
    completed_with_values = [t for t in study.trials if getattr(t, "value", None) is not None]

    if len(completed_with_values) == 0:
        print("[WARN] No trial reported a metric (all trials may have been pruned or failed).")
        print(f"[WARN] Check trial JSONs in {trial_dir} and the Optuna storage at {args.storage} for errors.")
        return

    # choose best according to study.direction
    try:
        # study.direction.name may be like 'MAXIMIZE' depending on Optuna versions - compare lowercase
        if getattr(study.direction, "name", "").lower() == "maximize":
            best_trial = max(completed_with_values, key=lambda t: t.value)
        else:
            best_trial = min(completed_with_values, key=lambda t: t.value)
        best_value = best_trial.value
        print(f"[INFO] best trial number={best_trial.number} value={best_value}")
        print(json.dumps(best_trial.params, indent=2))
    except Exception as e:
        print(f"[WARN] could not determine best trial from completed trials: {e}")
        print(f"[WARN] Check trial JSONs in {trial_dir} and the Optuna storage at {args.storage} for errors.")
        return

    # save best trial params to file (simple)
    best_path = args.out_base + "_best_trial.json"
    with open(best_path, "w") as f:
        json.dump({"value": best_value, "params": best_trial.params}, f, indent=2)
    print(f"[INFO] wrote best trial summary to {best_path}")
    # Ensure the saved best-trial JSON includes an `epochs` field inside params.
    # Prefer the trial's recorded best_epoch from its produced JSON if available,
    # otherwise fall back to the tuning epoch count (args.tune_epochs).
    try:
        # try to read trial JSON to obtain best_epoch
        trial_json_path = os.path.join(trial_dir, f"trial_{best_trial.number:04d}.json")
        be = None
        if os.path.isfile(trial_json_path):
            try:
                with open(trial_json_path, "r") as tf:
                    tj = json.load(tf)
                # top-level best_epoch or inside nested structures
                be = tj.get("best_epoch", None)
            except Exception:
                be = None
        epochs_to_record = int(be) if (be is not None) else int(args.tune_epochs)
        # update best_path with epochs
        try:
            with open(best_path, 'r') as bf:
                b = json.load(bf)
            b.setdefault('params', {})['epochs'] = epochs_to_record
            with open(best_path, 'w') as bf:
                json.dump(b, bf, indent=2)
            print(f"[INFO] updated {best_path} with epochs={epochs_to_record}")
        except Exception:
            pass
    except Exception:
        pass

    # save trials dataframe for inspection (this may be minimal depending on optuna version)
    try:
        df = study.trials_dataframe()
        df_path = args.out_base + "_trials.csv"
        df.to_csv(df_path, index=False)
        print(f"[INFO] wrote trials dataframe to {df_path}")
    except Exception:
        pass

    # Build a summary JSON matching baseline tuning outputs (one top-level key named
    # after the out_base basename, with best_params, best_val_* fields, test_metrics,
    # model/meta paths, and n_trials)
    base_key = os.path.splitext(os.path.basename(args.out_base))[0]
    summary = {
        base_key: {
            "best_params": best_trial.params,
            "best_val_metric_name": args.select_metric,
            "best_val_metric": None,
            "best_val_auc": None,
            "best_val_loss": float("nan"),
            "best_val_acc": None,
            "best_val_bacc": None,
            "best_val_f1": None,
            "test_metrics": {},
            "model_path": None,
            "meta_path": None,
            "n_trials": len(study.trials),
        }
    }

    # Try to load the per-trial JSON to extract val/test metrics and model/meta paths
    trial_json_path = os.path.join(trial_dir, f"trial_{best_trial.number:04d}.json")
    trial_ckpt_path = os.path.join(trial_dir, f"trial_{best_trial.number:04d}_best.pt")
    # Some runs write checkpoints into the top-level results/ directory instead of the trials subdir.
    # Use the selected best_trial number (avoid referencing study.best_trial which may be inconsistent across Optuna versions)
    alt_ckpt_path = os.path.join(os.path.dirname(args.out_base) or '.', f"trial_{best_trial.number:04d}_best.pt")
    if not os.path.isfile(trial_ckpt_path) and os.path.isfile(alt_ckpt_path):
        trial_ckpt_path = alt_ckpt_path
    trial_meta_path = os.path.splitext(trial_ckpt_path)[0] + ".meta.pkl"
    if os.path.isfile(trial_json_path):
        try:
            with open(trial_json_path, "r") as tf:
                tj = json.load(tf)
            # populate validation metrics if present
            for k in ["val_auc", "val_acc", "val_bacc", "val_f1", "val_loss"]:
                if k in tj:
                    vname = k.replace("val_", "best_val_")
                    summary[base_key][vname] = tj[k]
            # set best_val_metric explicitly if present
            sel = args.select_metric
            if sel in tj:
                summary[base_key]["best_val_metric"] = tj[sel]
            # populate test metrics
            test_metrics = {}
            for tk, outk in [("test_auc", "auc"), ("test_acc", "acc"), ("test_bacc", "bacc"), ("test_f1", "f1"), ("test_loss", "loss")]:
                if tk in tj:
                    test_metrics[outk] = tj[tk]
            if test_metrics:
                summary[base_key]["test_metrics"] = test_metrics
            # include optional val_predictions if present
            if "val_predictions" in tj:
                summary[base_key]["val_predictions"] = tj["val_predictions"]
        except Exception:
            pass

    if os.path.isfile(trial_ckpt_path):
        summary[base_key]["model_path"] = trial_ckpt_path
    if os.path.isfile(trial_meta_path):
        summary[base_key]["meta_path"] = trial_meta_path

    out_summary_path = args.out_base + ".json"
    with open(out_summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"[INFO] wrote detailed study summary to {out_summary_path}")

    # Prefer evaluating the saved checkpoint from the best trial (no retrain) to avoid
    # any possibility of test-set leakage from a retrain early-stopping run.
    best = best_trial.params
    trial_json_path = os.path.join(trial_dir, f"trial_{best_trial.number:04d}.json")
    trial_ckpt_path = os.path.join(trial_dir, f"trial_{best_trial.number:04d}_best.pt")
    eval_json = args.out_base + "_best_eval.json"

    if os.path.isfile(trial_ckpt_path) and os.path.isfile(trial_json_path):
        # Try to read best_epoch from the trial JSON. If present, perform a safe
        # retrain-on-full (train+val) for exactly that many epochs and then
        # evaluate on the test set. This mirrors the MLP pipeline and avoids
        # using test labels for early stopping.
        try:
            with open(trial_json_path, "r") as tf:
                tj = json.load(tf)
        except Exception:
            tj = {}

        be = tj.get("best_epoch", None)
        if be is not None:
            try:
                be_int = int(be)
            except Exception:
                be_int = None
        else:
            be_int = None

        if be_int and be_int > 0:
            print(f"[INFO] Best trial recorded best_epoch={be_int}; running safe retrain-on-full for {be_int} epochs")
            retrain_json = args.out_base + "_best_retrain.json"
            retrain_ckpt = os.path.splitext(retrain_json)[0] + ".pt"
            # Build retrain command: retrain on train+val for be_int epochs and save ckpt
            retrain_cmd = [
                sys.executable,
                "scripts/mref-ad/train_moe.py",
                "--experts_config",
                args.experts_config,
                "--splits",
                args.splits,
                "--split_type",
                "train_val_test",
                "--epochs",
                str(be_int),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--use_hierarchical_gate",
                "--out_json",
                retrain_json,
                "--retrain_on_full",
                "--save_checkpoint",
                retrain_ckpt,
            ]
            # append learned hyperparams from the elected best_trial
            best = best_trial.params
            retrain_cmd += ["--lr", f"{best['lr']:.6g}"]
            retrain_cmd += ["--wd", f"{best['wd']:.6g}"]
            retrain_cmd += ["--hidden_exp", str(int(best['hidden_exp']))]
            retrain_cmd += ["--hidden_gate", str(int(best['hidden_gate']))]
            retrain_cmd += ["--drop", f"{best['drop']:.4f}"]
            retrain_cmd += ["--lambda_sparse", f"{best['lambda_sparse']:.4f}"]
            retrain_cmd += ["--lambda_diverse", f"{best['lambda_diverse']:.5f}"]
            retrain_cmd += ["--tau", f"{best['tau']:.4f}"]
            if bool(best.get('gumbel_hard', False)):
                retrain_cmd.append("--gumbel_hard")
            # ensure retrain uses the batch size chosen by the tuner (if present)
            retrain_cmd += ["--batch_size", str(int(best.get('batch_size', args.batch_size)))]
            # indicate retrain-only mode so train_moe skips the redundant initial train->val
            retrain_cmd.append("--retrain_only")

            try:
                res = subprocess.run(retrain_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(res.stdout)
                if res.returncode != 0:
                    print(f"[WARN] retrain subprocess returned non-zero exit code: {res.returncode}")
                    print(res.stderr)
                # merge retrain results
                if os.path.isfile(retrain_json):
                    try:
                        with open(retrain_json, "r") as rf:
                            rj = json.load(rf)
                        if os.path.isfile(out_summary_path):
                            with open(out_summary_path, "r") as sf:
                                summ = json.load(sf)
                        else:
                            summ = {}
                        key = list(summ.keys())[0] if summ else base_key
                        for tk, outk in [("test_auc", "auc"), ("test_acc", "acc"), ("test_bacc", "bacc"), ("test_f1", "f1"), ("test_loss", "loss")]:
                            if tk in rj:
                                summ[key].setdefault("test_metrics", {})[outk] = rj[tk]
                        if os.path.isfile(retrain_ckpt):
                            summ[key]["model_path"] = retrain_ckpt
                        possible_meta = os.path.splitext(retrain_ckpt)[0] + ".meta.pkl"
                        if os.path.isfile(possible_meta):
                            summ[key]["meta_path"] = possible_meta
                        # record retrain epoch used (store as 'epochs' in best_params)
                        summ[key].setdefault("best_params", {})["epochs"] = be_int

                        # Log the final test metrics that will be added to the summary
                        final_test_metrics = summ[key].get("test_metrics", {})
                        print(f"[INFO] final test metrics to merge (retrain): {final_test_metrics}")
                        print(f"[INFO] retrain checkpoint path: {retrain_ckpt}, meta: {possible_meta if os.path.isfile(possible_meta) else 'N/A'}")

                        with open(out_summary_path, "w") as sf:
                            json.dump(summ, sf, indent=2)
                        print(f"[INFO] merged retrain results into {out_summary_path}")
                    except Exception as e:
                        print(f"[WARN] could not merge retrain results: {e}")
            except Exception as e:
                print(f"[ERROR] retrain failed: {e}")
        else:
            # fallback: evaluate the saved checkpoint (no retrain)
            print(f"[INFO] best_epoch not found in trial JSON; falling back to evaluating saved checkpoint {trial_ckpt_path}")
            eval_cmd = [
                sys.executable,
                "scripts/mref-ad/train_moe.py",
                "--experts_config",
                args.experts_config,
                "--splits",
                args.splits,
                "--split_type",
                "train_val_test",
                "--ckpt",
                trial_ckpt_path,
                "--eval_ckpt",
                "--out_json",
                eval_json,
            ]
            try:
                res = subprocess.run(eval_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(res.stdout)
                if res.returncode != 0:
                    print(f"[WARN] eval subprocess returned non-zero exit code: {res.returncode}")
                    print(res.stderr)
                if os.path.isfile(eval_json):
                    try:
                        with open(eval_json, "r") as rf:
                            rj = json.load(rf)
                        if os.path.isfile(out_summary_path):
                            with open(out_summary_path, "r") as sf:
                                summ = json.load(sf)
                        else:
                            summ = {}
                        key = list(summ.keys())[0] if summ else base_key
                        for tk, outk in [("test_auc", "auc"), ("test_acc", "acc"), ("test_bacc", "bacc"), ("test_f1", "f1"), ("test_loss", "loss")]:
                            if tk in rj:
                                summ[key].setdefault("test_metrics", {})[outk] = rj[tk]
                        possible_meta = os.path.splitext(trial_ckpt_path)[0] + ".meta.pkl"
                        if os.path.isfile(trial_ckpt_path):
                            summ[key]["model_path"] = trial_ckpt_path
                        if os.path.isfile(possible_meta):
                            summ[key]["meta_path"] = possible_meta
                        # note: no retrain was performed in this branch (we evaluated saved checkpoint),
                        # but record that epochs is 0 to indicate no retrain-on-full occurred
                        summ[key].setdefault("best_params", {})["epochs"] = 0

                        # Log the final test metrics that will be added to the summary
                        final_test_metrics = summ[key].get("test_metrics", {})
                        print(f"[INFO] final test metrics to merge (eval saved ckpt): {final_test_metrics}")
                        print(f"[INFO] evaluated checkpoint path: {trial_ckpt_path}, meta: {possible_meta if os.path.isfile(possible_meta) else 'N/A'}")

                        with open(out_summary_path, "w") as sf:
                            json.dump(summ, sf, indent=2)
                        print(f"[INFO] merged eval results into {out_summary_path}")
                    except Exception as e:
                        print(f"[WARN] could not merge eval results: {e}")
            except Exception as e:
                print(f"[ERROR] eval failed: {e}")
    else:
        # fallback: suggest retrain command (unchanged behavior)
        cmd = [
            sys.executable,
            "scripts/mref-ad/train_moe.py",
            "--experts_config",
            args.experts_config,
            "--splits",
            args.splits,
            "--split_type",
            "train_val_test",
            "--epochs",
            str(args.tune_epochs * 4),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--use_hierarchical_gate",
            "--out_json",
            args.out_base + "_best_retrain.json",
            "--retrain_on_full",
        ]
        # append best_trial params consistently
        best = best_trial.params
        cmd += ["--lr", f"{best['lr']:.6g}"]
        cmd += ["--wd", f"{best['wd']:.6g}"]
        cmd += ["--hidden_exp", str(int(best['hidden_exp']))]
        cmd += ["--hidden_gate", str(int(best['hidden_gate']))]
        cmd += ["--drop", f"{best['drop']:.4f}"]
        cmd += ["--lambda_sparse", f"{best['lambda_sparse']:.4f}"]
        cmd += ["--lambda_diverse", f"{best['lambda_diverse']:.5f}"]
        cmd += ["--tau", f"{best['tau']:.4f}"]
        if bool(best.get('gumbel_hard', False)):
            cmd.append("--gumbel_hard")
        # ensure auto-retrain uses the batch size chosen by the tuner (if present)
        cmd += ["--batch_size", str(int(best.get('batch_size', args.batch_size)))]
        # when performing fallback auto-retrain, request retrain-only behavior
        cmd.append("--retrain_only")

        print("[INFO] Suggested retrain command for best params:")
        print(" ".join(cmd))
        if getattr(args, "auto_retrain", False):
            print("[INFO] auto_retrain enabled — running retrain with best params now...")
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print(res.stdout)
                if res.returncode != 0:
                    print(f"[WARN] retrain subprocess returned non-zero exit code: {res.returncode}")
                    print(res.stderr)
                retrain_json = args.out_base + "_best_retrain.json"
                if os.path.isfile(retrain_json):
                    try:
                        with open(retrain_json, "r") as rf:
                            rj = json.load(rf)
                        if os.path.isfile(out_summary_path):
                            with open(out_summary_path, "r") as sf:
                                summ = json.load(sf)
                        else:
                            summ = {}
                        key = list(summ.keys())[0] if summ else base_key
                        for tk, outk in [("test_auc", "auc"), ("test_acc", "acc"), ("test_bacc", "bacc"), ("test_f1", "f1"), ("test_loss", "loss")]:
                            if tk in rj:
                                summ[key].setdefault("test_metrics", {})[outk] = rj[tk]
                        possible_ckpt = os.path.splitext(retrain_json)[0] + ".pt"
                        possible_meta = os.path.splitext(retrain_json)[0] + ".meta.pkl"
                        if os.path.isfile(possible_ckpt):
                            summ[key]["model_path"] = possible_ckpt
                        if os.path.isfile(possible_meta):
                            summ[key]["meta_path"] = possible_meta
                        # we ran an automatic retrain in this fallback path; record epochs used
                        summ[key].setdefault("best_params", {})["epochs"] = int(args.tune_epochs * 4)

                        # Log the final test metrics that will be added to the summary
                        final_test_metrics = summ[key].get("test_metrics", {})
                        print(f"[INFO] final test metrics to merge (auto-retrain fallback): {final_test_metrics}")
                        print(f"[INFO] auto-retrain checkpoint path: {possible_ckpt if os.path.isfile(possible_ckpt) else 'N/A'}, meta: {possible_meta if os.path.isfile(possible_meta) else 'N/A'}")

                        with open(out_summary_path, "w") as sf:
                            json.dump(summ, sf, indent=2)
                        print(f"[INFO] merged retrain results into {out_summary_path}")
                    except Exception as e:
                        print(f"[WARN] could not merge retrain results: {e}")
            except Exception as e:
                print(f"[ERROR] retrain failed: {e}")

    # Optionally remove intermediate artifacts so only the summary JSON remains
    if getattr(args, "prune_intermediates", False):
        print("[INFO] prune_intermediates enabled — removing intermediate trial artifacts...")
        # remove trial directory
        try:
            if os.path.isdir(trial_dir):
                import shutil

                shutil.rmtree(trial_dir)
                print(f"[INFO] removed trial directory {trial_dir}")
        except Exception as e:
            print(f"[WARN] could not remove trial directory {trial_dir}: {e}")

        # remove best trial JSON
        try:
            if os.path.isfile(best_path):
                os.remove(best_path)
                print(f"[INFO] removed best trial summary {best_path}")
        except Exception as e:
            print(f"[WARN] could not remove best trial summary {best_path}: {e}")

        # remove trials CSV
        try:
            df_path = args.out_base + "_trials.csv"
            if os.path.isfile(df_path):
                os.remove(df_path)
                print(f"[INFO] removed trials dataframe {df_path}")
        except Exception as e:
            print(f"[WARN] could not remove trials dataframe {df_path}: {e}")

        # remove retrain json and associated checkpoints if present
        try:
            retrain_json = args.out_base + "_best_retrain.json"
            if os.path.isfile(retrain_json):
                # also try to remove retrain checkpoint/meta
                retrain_ckpt = os.path.splitext(retrain_json)[0] + ".pt"
                retrain_meta = os.path.splitext(retrain_json)[0] + ".meta.pkl"
                try:
                    if os.path.isfile(retrain_ckpt):
                        os.remove(retrain_ckpt)
                        print(f"[INFO] removed retrain checkpoint {retrain_ckpt}")
                except Exception:
                    pass
                try:
                    if os.path.isfile(retrain_meta):
                        os.remove(retrain_meta)
                        print(f"[INFO] removed retrain meta {retrain_meta}")
                except Exception:
                    pass
                os.remove(retrain_json)
                print(f"[INFO] removed retrain json {retrain_json}")
        except Exception as e:
            print(f"[WARN] could not remove retrain artifacts: {e}")

        # optionally remove local sqlite storage if provided as sqlite:///path
        try:
            if args.storage and args.storage.startswith("sqlite://"):
                # support sqlite:///absolute/path or sqlite://relative/path
                sqlite_path = args.storage.split("sqlite://", 1)[1]
                if sqlite_path.startswith("/"):
                    db_path = sqlite_path
                else:
                    db_path = os.path.join(os.getcwd(), sqlite_path)
                if os.path.isfile(db_path):
                    os.remove(db_path)
                    print(f"[INFO] removed sqlite storage {db_path}")
        except Exception as e:
            print(f"[WARN] could not remove sqlite storage: {e}")

    # Update the best_trial summary file to include retrain_epochs if we recorded it into the summary
    try:
        if os.path.isfile(out_summary_path) and os.path.isfile(best_path):
            with open(out_summary_path, 'r') as sf:
                summ = json.load(sf)
            key = list(summ.keys())[0] if summ else None
            if key and 'best_params' in summ[key] and 'epochs' in summ[key]['best_params']:
                retrain_epochs = summ[key]['best_params']['epochs']
                # update best_path to include epochs inside params
                try:
                    with open(best_path, 'r') as bf:
                        b = json.load(bf)
                    b.setdefault('params', {})['epochs'] = retrain_epochs
                    with open(best_path, 'w') as bf:
                        json.dump(b, bf, indent=2)
                    print(f"[INFO] updated {best_path} with epochs={retrain_epochs}")
                except Exception:
                    pass
    except Exception:
        pass


if __name__ == "__main__":
    main()

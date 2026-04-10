#!/usr/bin/env python3
"""
Paired statistical comparison of test metrics across the same 10 data-split seeds.

Hyperparameters are documented under ``configs/best_hyperparameters/``; this script
consumes already-produced per-seed JSON under ``results/`` (same seeds for every model).

Tests (per baseline, paired on seeds):
  - Wilcoxon signed-rank: one-sided (primary > baseline) and two-sided (distributions differ).
  - Paired t-test: same one- and two-sided variants (differences roughly normal).

Multiple comparisons: Holm adjustment separately for each test type × direction (one-sided Wilcoxon,
two-sided Wilcoxon, one-sided t, two-sided t), within each metric family across baselines.

Paper setup (proposed model vs baselines):
  Default is ``--primary moe`` (**mref-ad**, mixture-of-experts) vs ``--baselines lr,rf,mlp,xgb,ftt,flex_moe``
  — classical baselines plus Flex-MoE on the same 10 split seeds. You can use ``--primary mref_ad``
  (same loader). One-sided tests support the claim "mref-ad > baseline"; two-sided tests only ask
  whether paired scores differ.

Example:
  python analysis/evaluation/compare_models_paired_stats.py
  # same as: --primary moe --baselines lr,rf,mlp,xgb,ftt,flex_moe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats


SEEDS = [7, 13, 42, 1234, 2027, 99, 123, 555, 999, 1337]


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down; return adjusted p-values in original order (matches R p.adjust(..., 'holm'))."""
    m = len(p_values)
    if m == 0:
        return []
    arr = np.array(p_values, dtype=float)
    order = np.argsort(arr)
    sorted_p = arr[order]
    q = np.minimum(1.0, (m - np.arange(m)) * sorted_p)
    adjusted_sorted = np.maximum.accumulate(q)
    out = np.empty(m, dtype=float)
    out[order] = adjusted_sorted
    return out.tolist()


def load_mref_ad(repo: Path) -> dict[int, dict[str, float]]:
    """Per-seed test metrics for mref-ad (legacy filenames ``moe_seed_*.json``)."""
    out: dict[int, dict[str, float]] = {}
    for s in SEEDS:
        p = repo / f"results/seed7_bestparams_all10/moe_seed_{s}.json"
        with open(p) as f:
            j = json.load(f)
        out[s] = {"acc": float(j["test_acc"]), "f1": float(j["test_f1"])}
    return out


def load_summary_per_seed(repo: Path, rel: str) -> dict[int, dict[str, float]]:
    with open(repo / rel) as f:
        j = json.load(f)
    out: dict[int, dict[str, float]] = {}
    for k, v in j["per_seed"].items():
        sk = int(k)
        out[sk] = {
            "acc": float(v["acc"]),
            "f1": float(v["f1"]),
        }
    return out


def load_ftt(repo: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for s in SEEDS:
        p = repo / f"results/ftt_seed7_bestparams_all10/ftt_seed_{s}.json"
        with open(p) as f:
            j = json.load(f)
        tm = j["ftt_concat_all"]["test_metrics"]
        out[s] = {"acc": float(tm["acc"]), "f1": float(tm["f1"])}
    return out


def load_flex_moe(repo: Path, rel: str = "results/flex_moe_10seeds_optuna_seed7_bestparams.json") -> dict[int, dict[str, float]]:
    """10-seed Flex-MoE test metrics (Optuna seed-7 best hyperparameters)."""
    with open(repo / rel) as f:
        j = json.load(f)
    block = j.get("seeds") or j
    out: dict[int, dict[str, float]] = {}
    for s in SEEDS:
        tm = block[str(s)]["test_metrics"]
        out[s] = {"acc": float(tm["acc"]), "f1": float(tm["f1"])}
    return out


def get_loaders(repo: Path, flex_moe_json: str) -> dict[str, Callable[[Path], dict[int, dict[str, float]]]]:
    return {
        "moe": load_mref_ad,
        "mref_ad": load_mref_ad,
        "flex_moe": lambda r: load_flex_moe(r, flex_moe_json),
        "lr": lambda r: load_summary_per_seed(r, "results/eval_lr/summary.json"),
        "rf": lambda r: load_summary_per_seed(r, "results/eval_rf/summary.json"),
        "mlp": lambda r: load_summary_per_seed(r, "results/eval_mlp/summary.json"),
        "xgb": lambda r: load_summary_per_seed(r, "results/eval_xgb/summary.json"),
        "ftt": load_ftt,
    }


def vec(model: dict[int, dict[str, float]], metric: str) -> np.ndarray:
    return np.array([model[s][metric] for s in SEEDS], dtype=float)


def compare_one(
    primary_v: np.ndarray, base_v: np.ndarray
) -> dict[str, Any]:
    d = primary_v - base_v
    mean_diff = float(np.mean(d))
    std_diff = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    n_pos = int(np.sum(d > 0))
    n_zero = int(np.sum(d == 0))
    n_neg = int(np.sum(d < 0))

    wilcox_stat = None
    wilcox_note = None
    wilcox_p_one = float("nan")
    wilcox_p_two = float("nan")
    if np.all(d == 0):
        wilcox_note = "all_differences_zero"
        wilcox_p_one = 1.0
        wilcox_p_two = 1.0
    else:
        try:
            w1 = stats.wilcoxon(
                primary_v,
                base_v,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            )
            wilcox_stat = float(w1.statistic) if w1.statistic is not None else None
            wilcox_p_one = float(w1.pvalue)
        except ValueError as e:
            wilcox_note = str(e)
        try:
            w2 = stats.wilcoxon(
                primary_v,
                base_v,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            if wilcox_stat is None and w2.statistic is not None:
                wilcox_stat = float(w2.statistic)
            wilcox_p_two = float(w2.pvalue)
        except ValueError as e:
            if wilcox_note is None:
                wilcox_note = str(e)
            wilcox_p_two = float("nan")

    t_stat = None
    t_note = None
    t_p_one = float("nan")
    t_p_two = float("nan")
    try:
        t1 = stats.ttest_rel(primary_v, base_v, alternative="greater")
        t_stat = float(t1.statistic)
        t_p_one = float(t1.pvalue)
    except Exception as e:
        t_note = str(e)
    try:
        t2 = stats.ttest_rel(primary_v, base_v, alternative="two-sided")
        if t_stat is None:
            t_stat = float(t2.statistic)
        t_p_two = float(t2.pvalue)
    except Exception as e:
        if t_note is None:
            t_note = str(e)

    return {
        "mean_diff_primary_minus_baseline": mean_diff,
        "std_diff": std_diff,
        "n_seeds_primary_wins_ties_losses": [n_pos, n_zero, n_neg],
        "wilcoxon_statistic": wilcox_stat,
        "wilcoxon_p_one_sided_greater": wilcox_p_one,
        "wilcoxon_p_two_sided": wilcox_p_two,
        "wilcoxon_note": wilcox_note,
        "paired_t_statistic": t_stat,
        "paired_t_p_one_sided_greater": t_p_one,
        "paired_t_p_two_sided": t_p_two,
        "paired_t_note": t_note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root containing results/",
    )
    parser.add_argument(
        "--primary",
        type=str,
        default="moe",
        choices=["moe", "mref_ad", "flex_moe", "lr", "rf", "mlp", "xgb", "ftt"],
        help="Model under test: mref-ad MoE (keys moe or mref_ad; default moe).",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="lr,rf,mlp,xgb,ftt,flex_moe",
        help="Comma-separated baselines vs primary (default: LR, RF, MLP, XGB, FTT, Flex-MoE).",
    )
    parser.add_argument(
        "--flex-moe-json",
        type=str,
        default="results/flex_moe_10seeds_optuna_seed7_bestparams.json",
        help="JSON with seeds.<id>.test_metrics from train_flex_moe 10-seed run.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write full results JSON (default: results/paired_stats_<primary>_vs_baselines.json).",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    loaders = get_loaders(repo, args.flex_moe_json)
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    baselines = [b for b in baselines if b != args.primary]
    if not baselines:
        raise SystemExit("No baselines left after excluding primary.")

    primary = loaders[args.primary](repo)
    loaded: dict[str, dict[int, dict[str, float]]] = {args.primary: primary}
    for b in baselines:
        loaded[b] = loaders[b](repo)

    for name, data in loaded.items():
        missing = [s for s in SEEDS if s not in data]
        if missing:
            raise SystemExit(f"Model {name!r} missing seeds {missing}")

    report: dict[str, Any] = {
        "seeds": SEEDS,
        "primary": args.primary,
        "baselines": baselines,
        "comparison_design": (
            f"Paired tests on the same split seeds: primary ({args.primary}) vs each baseline. "
            "One-sided p-values test primary > baseline (paper claim). Two-sided p-values test any "
            "difference. Holm adjusts across all listed baselines separately per metric per block."
        ),
        "hyperparameter_note": (
            "Tuned hyperparameters are recorded under configs/best_hyperparameters/ "
            "(flex_moe_best_trial.json, mref_ad_best_trial.json, etc.). "
            "This repo's results/ hold per-seed test metrics from fixed seed-7 Optuna configs "
            "evaluated on each split seed."
        ),
        "metrics": {},
    }

    for metric in ("acc", "f1"):
        pv = vec(loaded[args.primary], metric)
        rows = []
        wilcox_ps_1 = []
        wilcox_ps_2 = []
        t_ps_1 = []
        t_ps_2 = []
        for b in baselines:
            bv = vec(loaded[b], metric)
            row = {"baseline": b, **compare_one(pv, bv)}
            rows.append(row)
            w1 = row["wilcoxon_p_one_sided_greater"]
            wilcox_ps_1.append(w1 if not np.isnan(w1) else 1.0)
            w2 = row["wilcoxon_p_two_sided"]
            wilcox_ps_2.append(w2 if not np.isnan(w2) else 1.0)
            t1 = row["paired_t_p_one_sided_greater"]
            t_ps_1.append(t1 if not np.isnan(t1) else 1.0)
            t2 = row["paired_t_p_two_sided"]
            t_ps_2.append(t2 if not np.isnan(t2) else 1.0)

        holm_w1 = holm_adjust(wilcox_ps_1)
        holm_w2 = holm_adjust(wilcox_ps_2)
        holm_t1 = holm_adjust(t_ps_1)
        holm_t2 = holm_adjust(t_ps_2)
        for i, row in enumerate(rows):
            row["wilcoxon_p_one_sided_holm"] = holm_w1[i]
            row["wilcoxon_p_two_sided_holm"] = holm_w2[i]
            row["paired_t_p_one_sided_holm"] = holm_t1[i]
            row["paired_t_p_two_sided_holm"] = holm_t2[i]
        report["metrics"][metric] = {"comparisons": rows}

    out_path = args.out_json or (repo / f"results/paired_stats_{args.primary}_vs_baselines.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {out_path}\n")
    alpha = 0.05
    for metric in ("acc", "f1"):
        print(f"=== Test {metric.upper()} (paired on n={len(SEEDS)} seeds) ===")
        print(
            f"{'Baseline':<8} {'meanΔ':>8} {'W1 p':>8} {'W1 Holm':>9} {'*':>3} "
            f"{'t1 p':>8} {'t1 Holm':>8}  wins/ties/loss"
        )
        print("  (one-sided: primary > baseline; Holm within this block.)")
        for row in report["metrics"][metric]["comparisons"]:
            sig = "*" if row["wilcoxon_p_one_sided_holm"] < alpha else ""
            wts = row["n_seeds_primary_wins_ties_losses"]
            wtl = f"{wts[0]}/{wts[1]}/{wts[2]}"
            print(
                f"{row['baseline']:<8} {row['mean_diff_primary_minus_baseline']:8.4f} "
                f"{row['wilcoxon_p_one_sided_greater']:8.4f} {row['wilcoxon_p_one_sided_holm']:9.4f} "
                f"{sig:>3} {row['paired_t_p_one_sided_greater']:8.4f} {row['paired_t_p_one_sided_holm']:8.4f}  {wtl}"
            )
        print(
            f"\n{'Baseline':<8} {'W2 p':>8} {'W2 Holm':>9} {'*':>3} "
            f"{'t2 p':>8} {'t2 Holm':>8}"
        )
        print("  (two-sided: scores differ; no direction in H1; Holm within this block.)")
        for row in report["metrics"][metric]["comparisons"]:
            sig2 = "*" if row["wilcoxon_p_two_sided_holm"] < alpha else ""
            print(
                f"{row['baseline']:<8} "
                f"{row['wilcoxon_p_two_sided']:8.4f} {row['wilcoxon_p_two_sided_holm']:9.4f} "
                f"{sig2:>3} {row['paired_t_p_two_sided']:8.4f} {row['paired_t_p_two_sided_holm']:8.4f}"
            )
        print(
            f"\nHolm adjusts across the {len(baselines)} baselines separately for each block above "
            f"(familywise α={alpha}).\n"
        )


if __name__ == "__main__":
    main()

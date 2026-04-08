#!/usr/bin/env python3
"""
Plot missing-modality robustness curves from aggregated CV JSON outputs.

Example:
  python scripts/plot_missingness.py \
        --inputs results/missingness/pet_missing/baseline_ftt_p*.json \
    --label "FT-Transformer" \
    --metric val_auc \
    --out results/missingness/pet_missing/plots/ftt_all_pet_missing_val_auc

To compare multiple models, pass multiple --inputs/--label pairs:
  python scripts/plot_missingness.py \
        --inputs results/missingness/pet_missing/baseline_ftt_p*.json \
    --label "FT-Transformer" \
    --inputs results/missingness/pet_missing/moe_moe_p*.json \
    --label "MREF-AD (MoE)" \
    --metric val_auc \
    --out results/missingness/pet_missing/plots/compare_val_auc

# Compare FT-Transformer vs. MLP:
python scripts/plot_missingness.py \
        --inputs results/missingness/pet_missing/baseline_ftt_p*.json \
    --label "FT-Transformer" \
    --inputs results/missingness/pet_missing/baseline_mlp_concat_p*.json \
    --label "MLP" \
    --metric val_auc \
    --out results/missingness/pet_missing/plots/compare_val_auc

# FT-Transformer vs. MLP vs MREF-AD F1 score:
python scripts/plot_missingness.py \
        --inputs results/missingness/pet_missing/baseline_ftt_p*.json \
    --label "FT-Transformer" \
    --inputs results/missingness/pet_missing/baseline_mlp_concat_p*.json \
    --label "MLP" \
    --inputs results/missingness/pet_missing/aggregated/moe_pet_missing_p*.json \
    --label "MREF-AD" \
    --metric val_f1 \
    --out results/missingness/pet_missing/plots/compare_val_f1

# FT-Transformer vs. MLP Accuracy:
python scripts/plot_missingness.py \
    --inputs results/missingness/pet_missing/baseline_ftt_all_p*.json \
    --label "FT-Transformer" \
    --inputs results/missingness/pet_missing/baseline_mlp_concat_p*.json \
    --label "MLP" \
    --metric val_acc \
    --out results/missingness/pet_missing/plots/compare_val_acc

# Plot grouped bar plots comparing missing modality with 10 fold CV:
python scripts/plot_missingness.py \
  --plot_type bar \
  --group_labels "Full,PET missing,MRI missing" \
  --inputs results/missingness/pet_missing/aggregated/moe_pet_missing_p0p00.json results/missingness/pet_missing/aggregated/moe_pet_missing_p1p00.json results/missingness/mri_missing/aggregated/moe_mri_missing_p1p00.json \
  --label "MREF-AD" \
    --inputs results/missingness/pet_missing/baseline_ftt_p0p00.json results/missingness/pet_missing/baseline_ftt_p1p00.json results/missingness/mri_missing/baseline_ftt_p1p00.json \
  --label "FT-Transformer" \
  --inputs results/missingness/pet_missing/baseline_mlp_concat_p0p00.json results/missingness/pet_missing/baseline_mlp_concat_p1p00.json results/missingness/mri_missing/baseline_mlp_concat_p1p00.json \
  --label "MLP" \
  --metric val_f1 \
  --title "" \
  --out results/missingness/plots/grouped_bar_val_f1 

# With missinng PET, MRI, demographic
python scripts/plot_missingness.py \
  --plot_type bar \
  --group_labels "Full,PET missing,MRI missing,Demo missing" \
  --inputs \
    results/missingness/pet_missing/aggregated/moe_pet_missing_p0p00.json \
    results/missingness/pet_missing/aggregated/moe_pet_missing_p1p00.json \
    results/missingness/mri_missing/aggregated/moe_mri_missing_p1p00.json \
    results/missingness/demo_missing/aggregated/moe_demo_missing_p1p00.json \
  --label "MREF-AD" \
  --inputs \
    results/missingness/pet_missing/baseline_ftt_p0p00.json \
    results/missingness/pet_missing/baseline_ftt_p1p00.json \
    results/missingness/mri_missing/baseline_ftt_p1p00.json \
    results/missingness/demo_missing/baseline_ftt_p1p00.json \
  --label "FT-Transformer" \
  --inputs \
    results/missingness/pet_missing/baseline_mlp_concat_p0p00.json \
    results/missingness/pet_missing/baseline_mlp_concat_p1p00.json \
    results/missingness/mri_missing/baseline_mlp_concat_p1p00.json \
    results/missingness/demo_missing/baseline_mlp_concat_p1p00.json \
  --label "MLP" \
  --metric val_f1 \
  --title "" \
  --out results/missingness/plots/grouped_bar_val_f1_with_demo

# Plot grouped bar plot for missing Amy and MRI with 10 seeds of train/val/test:
python scripts/plot_missingness.py \
  --plot_type bar \
  --group_labels "Full,PET missing,MRI missing" \
  --inputs results/missingness/pet_missing_train_test_val/moe_moe_p0p00_aggregated.json results/missingness/pet_missing_train_test_val/moe_moe_p1p00_aggregated.json results/missingness/mri_missing_train_test_val/moe_moe_p1p00_aggregated.json \
  --label "MREF-AD" \
  --inputs results/missingness/pet_missing_train_test_val/baseline_ftt_all_p0p00_aggregated.json results/missingness/pet_missing_train_test_val/baseline_ftt_all_p1p00_aggregated.json results/missingness/mri_missing_train_test_val/baseline_ftt_all_p1p00_aggregated.json \
  --label "FT-Transformer" \
  --inputs results/missingness/pet_missing_train_test_val/baseline_mlp_concat_p0p00_aggregated.json results/missingness/pet_missing_train_test_val/baseline_mlp_concat_p1p00_aggregated.json results/missingness/mri_missing_train_test_val/baseline_mlp_concat_p1p00_aggregated.json \
  --label "MLP" \
  --inputs results/missingness/pet_missing_train_test_val/baseline_lr_all_p0p00_aggregated.json results/missingness/pet_missing_train_test_val/baseline_lr_all_p1p00_aggregated.json results/missingness/mri_missing_train_test_val/baseline_lr_all_p1p00_aggregated.json \
  --label "Logistic Regression" \
  --metric test_f1 \
  --title "" \
  --out results/missingness/plots/grouped_bar_test_f1_train_val_test_split_10_seeds
"""
# -----------------------------
# Additional helpers for grouped bar plots and model ordering
# -----------------------------

import string
from typing import Tuple, List

def _canonical_label(label: str) -> str:
    """Return a canonicalized version of the label for ordering."""
    s = label.lower()
    s = s.replace("-", "").replace("_", "")
    s = "".join(ch for ch in s if ch not in string.punctuation and not ch.isspace())
    return s

def _order_models(labels: List[str], inputs: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Reorder models so that:
      (1) label containing 'mref' or 'moe' or 'ours' first,
      (2) label containing 'ft' and 'transformer' (or 'ftt') second,
      (3) label containing 'mlp' third,
      (4) rest in original order.
    """
    def rank(lbl: str) -> int:
        cl = _canonical_label(lbl)
        if any(k in cl for k in ["mref", "moe", "ours"]):
            return 0
        if (("ft" in cl or "ftt" in cl) and "transformer" in cl) or "fttransformer" in cl or "ftt" in cl:
            return 1
        if "mlp" in cl:
            return 2
        return 3
    idxs = list(range(len(labels)))
    idxs_sorted = sorted(idxs, key=lambda i: (rank(labels[i]), i))
    new_labels = [labels[i] for i in idxs_sorted]
    new_inputs = [inputs[i] for i in idxs_sorted]
    return new_labels, new_inputs


# -----------------------------
# Plotting
# -----------------------------

import argparse

import glob
import json
import os
import re
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_frac_from_path(path: str) -> float:
    """
    Extract fraction from filenames like:
      ..._p0p20.json -> 0.20
      .../p0p60/...  -> 0.60
    """
    base = os.path.basename(path)
    m = re.search(r"_p(\d+)p(\d+)\.json$", base)
    if m:
        return float(f"{int(m.group(1))}.{m.group(2)}")
    # fallback: directory names like .../p0p20/...
    m2 = re.search(r"/p(\d+)p(\d+)(/|$)", path)
    if m2:
        return float(f"{int(m2.group(1))}.{m2.group(2)}")
    raise ValueError(f"Could not parse fraction from path: {path}")


def _normalize_metric_name(metric: str) -> str:
    """Map CLI metric names to keys used in fold_results."""
    m = metric.strip()
    if m.startswith("val_"):
        m = m[len("val_"):]
    if m.startswith("test_"):
        m = m[len("test_"):]
    # common aliases
    aliases = {
        "auc": "auc",
        "acc": "acc",
        "bacc": "bacc",
        "balanced_acc": "bacc",
        "balanced_accuracy": "bacc",
        "f1": "f1",
        "macro_f1": "f1",
    }
    return aliases.get(m, m)


def _find_cv_mean_std(obj: Dict[str, Any], metric: str) -> Optional[Tuple[float, float]]:
    """
    Tries to locate "mean ± std" for a metric in common aggregate_cv.py formats.

    We support a few likely schemas:
      - obj["cv_summary"][metric] == {"mean":..., "std":...}
      - obj["cv_summary"]["mean"][metric] and obj["cv_summary"]["std"][metric]
      - obj["cv"][metric] == {"mean":..., "std":...}
      - obj["summary"][metric] == {"mean":..., "std":...}
    """
    # case 1: cv_summary[metric] = {mean,std}
    for top in ["cv_summary", "summary", "cv", "metrics", "fold_summary"]:
        if top in obj and isinstance(obj[top], dict):
            d = obj[top]
            if metric in d and isinstance(d[metric], dict):
                mm = d[metric]
                if "mean" in mm and "std" in mm:
                    return float(mm["mean"]), float(mm["std"])
            # case 2: cv_summary has mean/std dicts
            if "mean" in d and "std" in d and isinstance(d["mean"], dict) and isinstance(d["std"], dict):
                if metric in d["mean"] and metric in d["std"]:
                    return float(d["mean"][metric]), float(d["std"][metric])
    return None


def _mean_std_from_folds(obj: Dict[str, Any], metric: str) -> Optional[Tuple[float, float]]:
    """
    Compute mean/std across folds from:
      obj["folds"][i]["fold_results"][fold_key][model_key][metric]
    where fold_key is typically the string form of only_fold (e.g., "0").
    """
    if "folds" not in obj or not isinstance(obj["folds"], list) or len(obj["folds"]) == 0:
        return None

    metric_k = _normalize_metric_name(metric)  # val_auc -> auc, etc.

    # Pick model_key by inspecting the first fold that has content
    model_key = None
    for fold in obj["folds"]:
        fr_outer = fold.get("fold_results")
        if not isinstance(fr_outer, dict) or len(fr_outer) == 0:
            continue

        # fold_results: {"0": {"ftt_concat_all": {...}}}
        if len(fr_outer) == 1:
            inner = next(iter(fr_outer.values()))
        else:
            k = fold.get("only_fold")
            inner = fr_outer.get(str(k), next(iter(fr_outer.values())))

        if isinstance(inner, dict) and len(inner) > 0:
            model_key = next(iter(inner.keys()))  # e.g., "ftt_concat_all"
            break

    if model_key is None:
        return None

    vals = []
    for fold in obj["folds"]:
        fr_outer = fold.get("fold_results")
        if not isinstance(fr_outer, dict) or len(fr_outer) == 0:
            continue

        if len(fr_outer) == 1:
            inner = next(iter(fr_outer.values()))
        else:
            k = fold.get("only_fold")
            inner = fr_outer.get(str(k), next(iter(fr_outer.values())))

        if not isinstance(inner, dict) or model_key not in inner:
            continue

        m = inner[model_key]
        if isinstance(m, dict) and metric_k in m:
            try:
                vals.append(float(m[metric_k]))
            except Exception:
                pass

    if len(vals) == 0:
        return None

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def _find_pooled_metric(obj: Dict[str, Any], metric: str) -> Optional[float]:
    """Try to find pooled metric value if present in common aggregate outputs."""
    metric_k = _normalize_metric_name(metric)

    for top in ["pooled", "pooled_metrics", "all_fold_pooled", "pooled_oof"]:
        if top in obj and isinstance(obj[top], dict):
            d = obj[top]
            # direct: d[metric]
            if metric in d:
                try:
                    return float(d[metric])
                except Exception:
                    pass
            if metric_k in d:
                try:
                    return float(d[metric_k])
                except Exception:
                    pass

    # schema where pooled metrics are per-model
    for top in ["pooled", "pooled_metrics"]:
        if top in obj and isinstance(obj[top], dict):
            for _, md in obj[top].items():
                if isinstance(md, dict):
                    if metric_k in md:
                        try:
                            return float(md[metric_k])
                        except Exception:
                            pass
    return None


def _fallback_from_printed_strings(obj: Dict[str, Any], metric: str) -> Optional[Tuple[float, float]]:
    """
    Last resort: if your JSON stores a string like:
      "val_auc: 0.7910 ± 0.0314"
    """
    s = json.dumps(obj)
    pat = re.compile(rf"{re.escape(metric)}\s*[:=]\s*([0-9]*\.?[0-9]+)\s*[±\+/-]\s*([0-9]*\.?[0-9]+)")
    m = pat.search(s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def _aggregated_seed_test_metrics(obj: Dict[str, Any], metric: str) -> Optional[Tuple[float, float]]:
    """
    Handle aggregated seed results from aggregate_train_test_val_seeds.py:
      {"baseline_key": {"test_metrics": {"auc": {"mean": 0.79, "std": 0.02}}}}
    """
    metric_k = _normalize_metric_name(metric)  # test_f1 -> f1, test_auc -> auc, etc.
    
    for key, val in obj.items():
        if isinstance(val, dict) and "test_metrics" in val:
            test_m = val["test_metrics"]
            if isinstance(test_m, dict) and metric_k in test_m:
                metric_data = test_m[metric_k]
                # Check if it's the aggregated format with mean/std
                if isinstance(metric_data, dict) and "mean" in metric_data and "std" in metric_data:
                    try:
                        return float(metric_data["mean"]), float(metric_data["std"])
                    except Exception:
                        pass
    return None


def _single_seed_test_metrics(obj: Dict[str, Any], metric: str) -> Optional[Tuple[float, float]]:
    """
    Handle single-seed train_test_val results where we have a structure like:
      {"baseline_key": {"test_metrics": {"auc": 0.79, "acc": 0.64, "f1": 0.61}}}
    
    For single seeds, return (value, 0.0) since there's no variance.
    """
    metric_k = _normalize_metric_name(metric)  # test_f1 -> f1, test_auc -> auc, etc.
    
    # Look for a top-level key that has test_metrics
    for key, val in obj.items():
        if isinstance(val, dict) and "test_metrics" in val:
            test_m = val["test_metrics"]
            if isinstance(test_m, dict) and metric_k in test_m:
                # Check if it's raw value (single seed) not the aggregated format
                if not isinstance(test_m[metric_k], dict):
                    try:
                        return float(test_m[metric_k]), 0.0
                    except Exception:
                        pass
    return None


def extract_point(path: str, metric: str) -> Tuple[float, float, Optional[float]]:
    """
    Return (mean, std, pooled_optional).
    """
    obj = _read_json(path)

    ms = _find_cv_mean_std(obj, metric)
    if ms is None:
        ms = _mean_std_from_folds(obj, metric)
    if ms is None:
        ms = _aggregated_seed_test_metrics(obj, metric)
    if ms is None:
        ms = _single_seed_test_metrics(obj, metric)
    if ms is None:
        ms = _fallback_from_printed_strings(obj, metric)
    if ms is None:
        raise KeyError(
            f"Could not find mean/std for metric '{metric}' in {path}. "
            f"Open the JSON and tell me the top-level keys; I’ll match the schema exactly."
        )

    pooled = _find_pooled_metric(obj, metric)
    return ms[0], ms[1], pooled


# -----------------------------
# Plotting
# -----------------------------


def plot_curve(fracs: np.ndarray, means: np.ndarray, stds: np.ndarray,
               label: str, show_band: bool = True):
    plt.plot(fracs, means, marker="o", linewidth=2, label=label)
    if show_band:
        lo = means - stds
        hi = means + stds
        plt.fill_between(fracs, lo, hi, alpha=0.2)


def plot_grouped_bars(group_labels: List[str],
                      model_labels: List[str],
                      means: np.ndarray,
                      stds: np.ndarray,
                      colors: List[str],
                      ylabel: str,
                      title: str,
                      show_grid: bool = False,
                      font_scale: float = 1.0):
    # Plot grouped bars: x=groups (scenarios), bars=models.
    n_models = len(model_labels)
    n_groups = len(group_labels)

    x = np.arange(n_groups)
    # bar width chosen for IEEE 2-col figures; adjust automatically
    width = min(0.22, 0.8 / max(1, n_models))
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    for i, (lab, c) in enumerate(zip(model_labels, colors)):
        plt.bar(x + offsets[i], means[i, :], width=width, label=lab,
                yerr=stds[i, :], capsize=2, color=c, error_kw=dict(elinewidth=1.2, capsize=3, capthick=1.2, ecolor="#333333"))

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    fs_tick = 11.0 * font_scale
    fs_xtick = 12.0 * font_scale
    fs_label = 13.0 * font_scale
    fs_title = 13.0 * font_scale
    plt.xticks(x, group_labels, rotation=0, fontsize=fs_xtick)
    plt.yticks(fontsize=fs_tick)
    plt.ylabel(ylabel, labelpad=13, fontsize=fs_label)
    plt.title(title, fontsize=fs_title)
    if show_grid:
        plt.grid(True, axis='y', linewidth=0.5, alpha=0.5)
    leg = plt.legend(
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(1.42, 1.0),
        fontsize=11.0 * font_scale,
    )
    leg.get_frame().set_edgecolor("0.5")
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_facecolor("white")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", action="append", nargs="+", required=True,
                    help="Glob pattern(s) for JSONs. Shell-expanded globs are accepted. "
                         "Provide multiple --inputs blocks to compare multiple models.")
    ap.add_argument("--label", action="append", required=True,
                    help="Legend label for each --inputs (same count).")
    ap.add_argument("--metric", type=str, default="val_auc",
                    help="Metric key, e.g., val_auc/auc, val_acc/acc, val_bacc/bacc, val_f1/f1.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output path prefix (no extension). Writes .png and .pdf")
    ap.add_argument("--title", type=str, default="Missing-modality robustness",
                    help="Figure title.")
    ap.add_argument("--ylabel", type=str, default=None,
                    help="Y label. If omitted, uses a pretty label for common metrics (e.g., AUC, Accuracy).")
    ap.add_argument("--show_pooled", action="store_true",
                    help="If pooled metrics are present, overlay them as 'x' markers.")
    ap.add_argument("--no_band", action="store_true",
                    help="Disable mean±std shaded band.")
    ap.add_argument("--plot_type", type=str, default="curve", choices=["curve", "bar"],
                    help="Plot type: curve (missingness sweep) or bar (grouped scenario bars).")
    ap.add_argument("--group_labels", type=str, default=None,
                    help="Comma-separated group labels for bar plots (e.g., 'Full,PET missing,MRI missing'). Required when --plot_type=bar.")
    ap.add_argument("--colors", type=str, default=None,
                    help="Comma-separated colors for model bars. If omitted, uses a publication-friendly default with MREF-AD highlighted.")
    ap.add_argument("--legend_loc", type=str, default="best", help="Matplotlib legend location (e.g., 'upper right', 'upper left', 'best').")
    ap.add_argument("--font_scale", type=float, default=1.0,
                    help="Global text scale factor (e.g., 1.1, 1.2).")
    ap.add_argument("--dpi", type=int, default=300,
                    help="Export DPI for both PNG and PDF saves.")
    args = ap.parse_args()

    if len(args.inputs) != len(args.label):
        raise ValueError("You must provide the same number of --inputs and --label.")

    out_dir = os.path.dirname(args.out) or "."
    _ensure_dir(out_dir)

    # Pretty default y-labels for paper figures
    pretty = {
        "val_auc": "AUC",
        "auc": "AUC",
        "val_acc": "Accuracy",
        "acc": "Accuracy",
        "val_bacc": "Balanced Accuracy",
        "bacc": "Balanced Accuracy",
        "balanced_acc": "Balanced Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "val_f1": "F1-score",
        "f1": "F1-score",
        "macro_f1": "F1-score",
    }
    ylabel = args.ylabel if args.ylabel else pretty.get(args.metric, pretty.get(_normalize_metric_name(args.metric), args.metric))

    # Reorder models for consistent presentation (MREF-AD first, then FT-Transformer, then MLP)
    args.label, args.inputs = _order_models(args.label, args.inputs)

    if args.plot_type == "bar":
        if args.group_labels is None:
            raise ValueError("--group_labels is required for --plot_type=bar")
        group_labels = [s.strip() for s in args.group_labels.split(",") if s.strip()]
        n_groups = len(group_labels)
        n_models = len(args.label)
        # Build files for each model, preserving order and deduplicating
        all_files = []
        for patterns in args.inputs:
            files = []
            seen = set()
            for p in patterns:
                matched = glob.glob(p)
                for f in (matched if matched else [p]):
                    if f not in seen:
                        files.append(f)
                        seen.add(f)
            # Do NOT sort; preserve user order
            if len(files) != n_groups:
                raise ValueError(f"Model {patterns} got {len(files)} files, expected {n_groups} (group_labels).")
            all_files.append(files)
        # For each model i, for each group j, extract (mean, std, pooled)
        means = np.zeros((n_models, n_groups), dtype=float)
        stds = np.zeros((n_models, n_groups), dtype=float)
        for i in range(n_models):
            for j in range(n_groups):
                mean, std, _ = extract_point(all_files[i][j], args.metric)
                means[i, j] = mean
                stds[i, j] = std
        # Colors
        if args.colors:
            colors = [s.strip() for s in args.colors.split(",") if s.strip()]
            if len(colors) != n_models:
                raise ValueError(f"Provided --colors has {len(colors)} colors, expected {n_models}.")
        else:
            # Default: MREF-AD (pink), FT-Transformer (dark gray), MLP (light gray), LR (blue)
            # base = ["#EC6B83", "#555555", "#9F9E9E", "#91B0CE"]
            base = ["#EC6B83", '#4A4A4A', '#9F9E9E', "#7FA7C9"]
            extras = ["#999999", "#CCCCCC", "#9E9E9E"]
            colors = []
            for i in range(n_models):
                if i < len(base):
                    colors.append(base[i])
                else:
                    colors.append(extras[(i - len(base)) % len(extras)])
        plt.figure(figsize=(6.5, 3.4))
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plot_grouped_bars(group_labels, args.label, means, stds, colors, ylabel, args.title, show_grid=False, font_scale=args.font_scale)
        plt.tight_layout()
        png_path = args.out + ".png"
        pdf_path = args.out + ".pdf"
        plt.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        plt.savefig(pdf_path, dpi=args.dpi, bbox_inches="tight")
        print(f"[OK] Wrote: {png_path}")
        print(f"[OK] Wrote: {pdf_path}")

        # Also write a version without the "Full" group (missingness-only view)
        full_idxs = [i for i, g in enumerate(group_labels) if g.strip().lower() == "full"]
        if len(full_idxs) == 1 and n_groups > 1:
            full_idx = full_idxs[0]
            missing_labels = [g for i, g in enumerate(group_labels) if i != full_idx]
            missing_means = np.delete(means, full_idx, axis=1)
            missing_stds = np.delete(stds, full_idx, axis=1)

            plt.figure(figsize=(6.5, 3.4))
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plot_grouped_bars(missing_labels, args.label, missing_means, missing_stds, colors, ylabel, args.title, show_grid=False, font_scale=args.font_scale)
            plt.tight_layout()

            missing_png = args.out + "_missing_only.png"
            missing_pdf = args.out + "_missing_only.pdf"
            plt.savefig(missing_png, dpi=args.dpi, bbox_inches="tight")
            plt.savefig(missing_pdf, dpi=args.dpi, bbox_inches="tight")
            print(f"[OK] Wrote: {missing_png}")
            print(f"[OK] Wrote: {missing_pdf}")
        return

    # Curve mode (default)
    plt.figure(figsize=(6.2, 3.6))  # IEEE-ish
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    all_fracs_seen: List[float] = []

    for patterns, lab in zip(args.inputs, args.label):
        files = []
        for p in patterns:
            matched = glob.glob(p)
            files.extend(matched if matched else [p])
        files = sorted(set(files))
        if not files:
            raise FileNotFoundError(f"No files matched patterns: {patterns}")

        pts = []
        for fp in files:
            frac = _parse_frac_from_path(fp)
            mean, std, pooled = extract_point(fp, args.metric)
            pts.append((frac, mean, std, pooled))

        pts.sort(key=lambda x: x[0])
        fracs = np.array([p[0] for p in pts], dtype=float)
        means = np.array([p[1] for p in pts], dtype=float)
        stds  = np.array([p[2] for p in pts], dtype=float)
        all_fracs_seen.extend(list(fracs))

        plot_curve(fracs, means, stds, label=lab, show_band=(not args.no_band))

        if args.show_pooled:
            pooled_vals = [p[3] for p in pts]
            if all(v is not None for v in pooled_vals):
                plt.scatter(fracs, np.array(pooled_vals, dtype=float), marker="x", s=45)

    # Ensure the x-axis includes 1.0 when applicable (common for missingness sweeps)
    if len(all_fracs_seen) > 0:
        fmax = float(np.max(all_fracs_seen))
        fmin = float(np.min(all_fracs_seen))
        # If we have a full-missingness point, make sure it's visible and labeled.
        if fmax >= 0.99:
            plt.xlim(min(0.0, fmin), 1.0)
            plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=11 * args.font_scale)
        else:
            plt.xticks(fontsize=11 * args.font_scale)
    else:
        plt.xticks(fontsize=11 * args.font_scale)
    plt.yticks(fontsize=11 * args.font_scale)
    plt.xlabel("Missingness fraction", fontsize=12 * args.font_scale)
    plt.ylabel(ylabel, labelpad=12, fontsize=12 * args.font_scale)
    plt.title(args.title, fontsize=13 * args.font_scale)
    if not args.no_grid:
        plt.grid(True, linewidth=0.5, alpha=0.5)
    leg = plt.legend(
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=11.0 * args.font_scale,
    )
    leg.get_frame().set_edgecolor("0.5")   # medium gray border
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_facecolor("white")
    plt.tight_layout()

    png_path = args.out + ".png"
    pdf_path = args.out + ".pdf"
    plt.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] Wrote: {png_path}")
    print(f"[OK] Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
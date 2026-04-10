"""Baselines registry and utilities.

This module provides a small, non-invasive registry used by the
`scripts/train_baselines.py` CLI to resolve which baseline runs should be
executed for cross-validation. It intentionally does not import training
functions to avoid circular imports; instead it returns lightweight
descriptors (key, type, mod) which the CLI maps to the actual functions
already defined in `scripts/train_baselines.py`.

The registry is a single source of truth for CLI baseline aliases.
"""
from typing import List, Tuple

# All allowed CLI baseline choices (kept in sync with scripts/train_baselines.py)
AVAILABLE_BASELINES = [
    "all", "single", "concat", "latefusion",
    "mlp_single", "mlp_concat", "mlp_latefusion", "mlp_all",
    "rf_all", "xgb_all", "lr_all",
    # Only the official FT-Transformer is supported now (key: 'ftt')
    "ftt",
]


def build_run_baselines(baseline_arg: str, mods: List[str]) -> List[Tuple[str, str, str | None]]:
    """Resolve a CLI `--baseline` argument into a list of runs.

    Returns a list of tuples (key, btype, mod) where:
      - key: stable result key used in JSON outputs
      - btype: canonical baseline type used by the runner switch in the CLI
      - mod: optional modality name for per-modality runs

    This mirrors the logic previously embedded in `scripts/train_baselines.py`.
    """
    run_baselines = []
    # single-modality LR
    if baseline_arg in ("single", "all"):
        for m in mods:
            run_baselines.append(("single_" + m, "single", m))

    # concat / latefusion (non-MLP)
    if baseline_arg in ("concat", "all") and len(mods) >= 2:
        run_baselines.append(("concat_all", "concat", None))
    if baseline_arg in ("latefusion", "all") and len(mods) >= 2:
        run_baselines.append(("latefusion_avg", "latefusion", None))

    # MLP variants
    if baseline_arg in ("mlp_single", "mlp_all", "all"):
        for m in mods:
            run_baselines.append((f"mlp_single_{m}", "mlp_single", m))
    if baseline_arg in ("mlp_concat", "mlp_all", "all") and len(mods) >= 2:
        run_baselines.append(("mlp_concat_all", "mlp_concat", None))
    if baseline_arg in ("mlp_latefusion", "mlp_all", "all") and len(mods) >= 2:
        run_baselines.append(("mlp_latefusion_avg", "mlp_latefusion", None))

    # classical ML algorithms (all features)
    if baseline_arg == "rf_all":
        run_baselines.append(("rf_concat_all", "rf", None))
    if baseline_arg == "xgb_all":
        run_baselines.append(("xgb_concat_all", "xgb", None))
    if baseline_arg == "lr_all":
        run_baselines.append(("lr_concat_all", "lr", None))

    # FT-Transformer (official only). Accept legacy 'ftt_all' as alias.
    if baseline_arg in ("ftt_all", "ftt"):
        run_baselines.append(("ftt_concat_all", "ftt", None))

    return run_baselines
    
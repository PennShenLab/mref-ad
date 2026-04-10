"""Runtime registry mapping baseline types to runner callables.

Each runner returned here is a callable with signature:

    runner(df, groups, tr_idx, va_idx, mod=None, **kwargs) -> dict

The registry adapts the various underlying functions (LR, MLP, FT-Transformer,
and sklearn adapters) into a uniform callable so the CLI can dispatch
baselines generically.
"""
from typing import Callable, Dict, Optional

from baselines import runners


def _concat_cols(groups, df=None, include_has_flags: bool = False):
    cols = [c for _, feat in groups.items() for c in feat]
    if include_has_flags:
        for name in groups.keys():
            h = f"has_{name}"
            if df is None or (hasattr(df, "columns") and h in df.columns):
                cols.append(h)
    # de-duplicate while preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _sklearn_adapter(baseline_name: str) -> Callable:
    def _run(df, groups, tr_idx, va_idx, mod=None, **kwargs):
        cols = _concat_cols(groups, df=df, include_has_flags=False)
        # sklearn runner only accepts a small set of kwargs (seed, n_jobs).
        # The CLI may pass additional args (e.g. early_stop_metric) which
        # are irrelevant for sklearn baselines; filter them out to avoid
        # unexpected keyword argument errors.
        seed = kwargs.get("seed", 42)
        n_jobs = kwargs.get("n_jobs", 1)
        return runners.train_eval_sklearn_baselines(baseline_name, df, cols, tr_idx, va_idx, seed=seed, n_jobs=n_jobs)
    return _run


_REGISTRY: Dict[str, Callable] = {
    # classical ML
    "rf": _sklearn_adapter("rf_all"),
    "xgb": _sklearn_adapter("xgb_all"),
    "lr": _sklearn_adapter("lr_all"),

    # logistic/regression single/concat/latefusion
    "single": lambda df, groups, tr_idx, va_idx, mod=None, **kw: runners.run_single_modality_LR(df, groups, mod, tr_idx, va_idx),
    "concat": lambda df, groups, tr_idx, va_idx, mod=None, **kw: runners.run_concat_LR(df, groups, tr_idx, va_idx),
    "latefusion": lambda df, groups, tr_idx, va_idx, mod=None, **kw: runners.run_latefusion(df, groups, tr_idx, va_idx),

    # MLP variants
    "mlp_single": lambda df, groups, tr_idx, va_idx, mod=None, **kw: runners.run_single_modality_MLP(df, groups, mod, tr_idx, va_idx, **kw),
    "mlp_concat": lambda df, groups, tr_idx, va_idx, mod=None, **kw: runners.run_concat_MLP(df, groups, tr_idx, va_idx, **kw),
    "mlp_latefusion": lambda df, groups, tr_idx, va_idx, mod=None, **kw: runners.run_latefusion_MLP(df, groups, tr_idx, va_idx, **kw),

    # FT-Transformer: use the official implementation only (no custom/mask variants)
    # We compute concatenated columns and delegate to the official trainer in runners.
    # FT-Transformer: the runners.train_eval_ftt function does not accept
    # scheduler/global kwargs like `seed` or `n_jobs` that the CLI may pass;
    # filter them out to avoid unexpected-kwarg errors.
    "ftt": lambda df, groups, tr_idx, va_idx, mod=None, **kw: (
        runners.train_eval_ftt(
            df,
            _concat_cols(groups, df=df, include_has_flags=False),
            tr_idx,
            va_idx,
            **{k: v for k, v in kw.items() if k not in {"seed", "n_jobs"}},
        )
    ),
}


def get_runner(btype: str) -> Optional[Callable]:
    """Return a runner callable for the given baseline type or None if unknown."""
    return _REGISTRY.get(btype)


def list_baselines() -> list:
    return list(_REGISTRY.keys())

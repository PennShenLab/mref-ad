"""Utility helpers for baselines package.

Right now this provides a single device detection helper used by multiple
baseline modules. Centralizing it avoids duplicated logic and makes it
easier to change later (for example, preferring CUDA over MPS, etc.).

Named device_util (not utils) so the top-level shared ``utils`` module is not
shadowed when running ``python scripts/baselines/train_baselines.py``.
"""
from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    """Return the preferred torch.device for training/inference.

    Order of preference:
      1. Apple MPS (if available)
      2. CUDA (if available)
      3. CPU
    """
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        # Be conservative: if querying MPS raises for this environment, ignore it.
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

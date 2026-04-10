#!/usr/bin/env python3
"""Thin launcher for analysis/model_complexity/compute_model_params.py (same CLI)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

import runpy

runpy.run_path(
    str(ROOT / "analysis" / "model_complexity" / "compute_model_params.py"),
    run_name="__main__",
)

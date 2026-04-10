"""Put repository root and ``scripts/`` on ``sys.path`` for imports (``utils``, ``baselines``)."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_imports() -> Path:
    """Return repo root; ensure it and ``<repo>/scripts`` are importable."""
    repo = Path(__file__).resolve().parent.parent.parent
    scripts = repo / "scripts"
    for p in (repo, scripts):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return repo

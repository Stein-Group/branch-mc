"""Shared pytest configuration for the branch_mc suite.

Puts the repo root and ``tests/scripts`` on ``sys.path`` so ``from gcmc import
...`` resolves without an editable install and the engine tests can ``import
_harness`` after the restructure moved the harness under ``tests/scripts``.

Tiering is marker-based (``slow``, ``characterisation`` in pyproject); there is
deliberately no ``--runslow`` flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # tests/
_ROOT = _HERE.parent  # repo root (holds gcmc/, orig/, examples/)
_SCRIPTS = _HERE / "scripts"  # helpers + check_*.py live here after M7

for _p in (_ROOT, _SCRIPTS):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

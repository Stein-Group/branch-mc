"""Characterisation tier: run every ``tests/scripts/check_*.py`` as a
subprocess and require exit code 0.

These scripts drive the current ``gcmc`` package against the frozen original in
``orig/`` and assert bit- or value-level parity. They are oracles, not unit
tests: slow, process-level, and opt-in via ``pytest -m characterisation``. A
script that needs generated data absent from a bare checkout (e.g. ``refs/``)
is skipped with a reason; a script that currently diverges from the original
for a real, unresolved reason is xfailed so the finding stays visible without
reddening the tier.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_ROOT = _SCRIPTS.parent.parent  # repo root (holds gcmc/, data/)

# R1: the frozen original lives in tag v0-original, not in the tree. Skipping
# HERE -- once, at module level -- is the only correct place to do it: the check
# scripts run as SUBPROCESSES, where a skip raised inside one is just a non-zero
# exit that this harness would report as a failure. The message carries the two
# commands that make this module run.
sys.path.insert(0, str(_SCRIPTS))
from _frozen import INSTRUCTIONS, orig_root

if orig_root() is None:
    pytest.skip(INSTRUCTIONS, allow_module_level=True)

_ARGV_RE = re.compile(r"ARGV\s*=\s*\{(.*?)\n\}", re.S)

# Scripts that currently fail against the modernized engine for a real reason,
# pending sign-off. xfail keeps them visible and runnable without reddening the
# tier; strict=False so a future fix reports XPASS instead of breaking the run.
# Each reason is a standing finding recorded in modernize_plan.md.
_KNOWN_FAIL = {
    # check_parallel's 2D-only divergence was resolved by decision, not by
    # restoring the guard: the parallel engine now supports 3D deliberately
    # (cell_unify_plan.md U0, Dipesh GO 2026-07-14). The script and
    # test_item29_support_matrix now encode the new matrix, so it is no longer
    # a known failure.
}


def _modes(text: str) -> list[str]:
    """The ARGV dispatch keys a script accepts as its single CLI argument, or []
    for a script that takes no mode."""
    m = _ARGV_RE.search(text)
    return re.findall(r'"([^"]+)"\s*:', m.group(1)) if m else []


def _ident(script: str, mode: str | None) -> str:
    return f"{script}:{mode}" if mode else script


def _discover() -> list[object]:
    params: list[object] = []
    for f in sorted(_SCRIPTS.glob("check_*.py")):
        modes = _modes(f.read_text()) or [None]
        for mode in modes:
            marks = ()
            if f.name in _KNOWN_FAIL:
                marks = (pytest.mark.xfail(reason=_KNOWN_FAIL[f.name], strict=False),)
            params.append(
                pytest.param(f.name, mode, marks=marks, id=_ident(f.name, mode))
            )
    return params


_PARAMS = _discover()


def _skip_reason(script: str) -> str | None:
    """Why this script cannot run in a bare checkout, or None to run it."""
    src = (_SCRIPTS / script).read_text()
    if '"refs"' in src and not (_ROOT / "refs").is_dir():
        return "needs a generated refs/ directory (run tests/scripts/refgen.py first)"
    return None


@pytest.mark.characterisation
@pytest.mark.timeout(900)
@pytest.mark.parametrize("script,mode", _PARAMS)
def test_check_script(script: str, mode: str | None) -> None:
    reason = _skip_reason(script)
    if reason is not None:
        pytest.skip(reason)
    args = [sys.executable, str(_SCRIPTS / script)] + ([mode] if mode else [])
    env = dict(os.environ, PYTHONPATH=str(_ROOT))
    proc = subprocess.run(args, cwd=str(_ROOT), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"{script} {mode or ''} exited {proc.returncode}\n"
        f"--- stdout tail ---\n{proc.stdout[-1500:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-1500:]}"
    )

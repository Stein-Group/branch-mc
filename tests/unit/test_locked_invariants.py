"""Locked bodies: count_and_sample and ljbranch_weight_sum in samplers/branch.py
are never to be modified (their weights are combinatorially exact and every
downstream identity test trusts them verbatim). This freezes each function's
AST -- structure and names, ignoring formatting and line numbers -- against the
delivered tree. A hash mismatch means someone edited a body that must not change;
if the change is truly intended it needs a deliberate re-baseline here, not a
silent edit.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

_BRANCH = Path(__file__).resolve().parents[2] / "gcmc" / "samplers" / "branch.py"

# SHA-256 of ast.dump(node) for each frozen function, baselined 2026-07-12.
_FROZEN = {
    "count_and_sample": "e3fc4c51a508427cf1d33e8765b6e6c5fee4a43f3ef0ab09228ab1a6dd5a997c",
    "ljbranch_weight_sum": "84f3d9613a3226393bb7a39b034a91396f35890dfea1c8ba821e2dd4611681c3",
}


def _ast_hash(name: str) -> str:
    tree = ast.parse(_BRANCH.read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    dump = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()


def test_count_and_sample_body_is_unchanged():
    assert _ast_hash("count_and_sample") == _FROZEN["count_and_sample"]


def test_ljbranch_weight_sum_body_is_unchanged():
    assert _ast_hash("ljbranch_weight_sum") == _FROZEN["ljbranch_weight_sum"]

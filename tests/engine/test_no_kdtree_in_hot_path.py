"""There is exactly one neighbour structure in the hot path.

`_advance_metropolis` keeps its kd-tree forever: it is the independent reference
engine that the converted paths are validated against, and it would stop being
independent if it shared their index. Everything else in `MC_engine` must go
through `gcmc.neighbors.CellList`.

The allowed set shrinks one stage at a time. Every shrink is recorded in the
plan's status ledger; it never grows back.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# Shrinks one stage at a time; each shrink is recorded in the ledger.
#   A1 (now):  {_advance_metropolis, advance, _lj_set_energy}
#   A3:        {_advance_metropolis, _lj_set_energy}
#   A4 onward: {_advance_metropolis}
ALLOWED = {"_advance_metropolis"}

_TREE_NAMES = {"cKDTree", "KDTree"}


def _enclosing_functions_calling_trees(path):
    tree = ast.parse(Path(path).read_text())
    hits = set()

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def _fn(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = _fn
        visit_AsyncFunctionDef = _fn

        def visit_Call(self, node):
            f = node.func
            name = getattr(f, "id", None) or getattr(f, "attr", None)
            if name in _TREE_NAMES:
                hits.add(self.stack[0] if self.stack else "<module>")
            self.generic_visit(node)

    V().visit(tree)
    return hits


def test_engine_builds_no_kdtree_outside_metropolis():
    hits = _enclosing_functions_calling_trees(
        os.path.join(ROOT, "gcmc", "MC_engine.py")
    )
    extra = hits - ALLOWED
    assert not extra, (
        f"kd-tree construction in MC_engine functions {sorted(extra)}; the hot "
        f"path has one neighbour structure, gcmc.neighbors.CellList"
    )


def test_neighbors_module_does_not_import_scipy():
    src = Path(ROOT, "gcmc", "neighbors.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for n in names:
            assert not n.startswith("scipy"), f"neighbors.py imports {n}"

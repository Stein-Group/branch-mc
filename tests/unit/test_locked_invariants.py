"""Locked body: ljbranch_weight_sum in samplers/branch.py is never to be modified (their weights are combinatorially exact and every
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
    "ljbranch_weight_sum": "84f3d9613a3226393bb7a39b034a91396f35890dfea1c8ba821e2dd4611681c3",
}

# count_and_sample was UNPINNED on purpose. Its clique COUNT was always exact,
# but the example clique it returned was the lowest one in degeneracy order,
# which is not a uniform draw -- at k = 3 over a third of the cliques were
# unreachable at any vertex labelling. The acceptance ratio's log Z term assumes
# a uniform draw, so the witness was a detailed-balance violation, and there is
# no point freezing a body that has to change. The count is still protected
# behaviourally by tests/scripts/check_branch.py (parity against orig/) and the
# draw by tests/unit/test_clique_draw_uniform.py.


def _ast_hash(name: str) -> str:
    tree = ast.parse(_BRANCH.read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    dump = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dump.encode()).hexdigest()


def test_ljbranch_weight_sum_body_is_unchanged():
    assert _ast_hash("ljbranch_weight_sum") == _FROZEN["ljbranch_weight_sum"]

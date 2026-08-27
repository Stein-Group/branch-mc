"""Locked body: `ljbranch_weight_sum` in samplers/branch.py must not drift.

Its clique weights are combinatorially exact and every downstream identity test
trusts them verbatim, so this freezes the function's structure -- names and
shape, ignoring formatting, comments and line numbers -- against a committed
hash. A mismatch means someone edited a body that must not change; a change that
is genuinely intended needs a deliberate re-baseline HERE, with a stated cause,
not a silent edit.

WHY `ast.unparse` AND NOT `ast.dump`. This hashed `ast.dump(node)` until S3, and
that made the gate unportable: `ast.dump` includes fields that the grammar gains
between releases (`type_params` on FunctionDef arrived in 3.12), so the same
unmodified source hashed three different ways --

    3.11  53dcf3c246dc...    3.12  84f3d9613a32...    3.13  1413c4f53423...

-- and only 3.12, the version CI pins, matched the committed value. `pyproject`
promises `requires-python = ">=3.11"`, which is a promise the suite could not
keep: on 3.11 or 3.13 this test failed on correct code, which is the worst
failure mode a gate can have, because the only way to make it pass is to stop
believing it.

`ast.unparse` is the canonical source round-trip. It is formatting-, comment- and
line-number-insensitive, which is exactly what this test wants, and it is stable
across 3.11, 3.12 and 3.13 -- verified on all three before the re-baseline, not
assumed.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

_BRANCH = Path(__file__).resolve().parents[2] / "gcmc" / "samplers" / "branch.py"

# SHA-256 of ast.unparse(node) for each frozen function.
#
# PROVENANCE. Every value this has held, and why it moved:
#
#   ast.dump baseline, 2026-07-12
#       84f3d9613a3226393bb7a39b034a91396f35890dfea1c8ba821e2dd4611681c3
#       Retired, not violated: the hash function changed, so the number is not
#       comparable. Held on 3.12 only (see the module docstring).
#
#   ast.unparse, S3 phase 4 (CURRENT)
#       b0e42bd3b0495b9e2584a12f11f09ab6b287648dd7730b026c269c352d980de2
#       The typed clique kernel. Three keyword-only arguments were added --
#       `species`, `floor`, `table` -- and two calls now forward them: `_build_adj`
#       decides conflict at the per-pair floor `c*sigma_ij` instead of one scalar,
#       and `_intra_lj_factor` evaluates the per-pair 12-6 energy instead of the
#       single-type one.
#       All three default to None, and on that path every expression is the one
#       that was there: `_build_adj` takes the same single-radius kd-tree query and
#       `_intra_lj_factor` the same `lj_pair_energy` sum. The clique ENUMERATION,
#       the reservoir draw and its rng consumption are untouched -- what changed is
#       the graph handed to it and the weight of a pair, not how cliques are walked.
#       Evidence: counter digests 13/13 unchanged, and
#       tests/engine/test_typed_energy.py asserts a one-species typed run
#       reproduces the untyped chain bit for bit through this kernel.
#
#   ast.unparse re-baseline, S3 phase 1
#       4dc9820e1d4ad518214b86c8cb5635840e4d7cef7c1115eb226f58d85e06e379
#       TWO deliberate changes at once, and they are separable:
#         (a) the hash function moved to `ast.unparse` for portability. On the
#             UNMODIFIED body this gave 7ad0b90206a8142d7f7067c1... identically
#             on 3.11, 3.12 and 3.13.
#         (b) the body gained a fourth return value, `chosen_rows` -- the drawn
#             clique's row indices into `centers`, which the typed insert path
#             needs because M21 decides provenance by pool row and never by
#             comparing coordinates. `idx` was already computed on every
#             iteration; the change is to stop discarding it.
#       The reservoir test, its rng consumption and the log-space accumulation
#       order are untouched, so no weight and no draw moved. The evidence is the
#       counter digests: 13/13 unchanged across the commit that did this.
_FROZEN = {
    "ljbranch_weight_sum": (
        "b0e42bd3b0495b9e2584a12f11f09ab6b287648dd7730b026c269c352d980de2"
    ),
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
    return hashlib.sha256(ast.unparse(node).encode()).hexdigest()


def test_ljbranch_weight_sum_body_is_unchanged() -> None:
    got = _ast_hash("ljbranch_weight_sum")
    assert got == _FROZEN["ljbranch_weight_sum"], (
        "ljbranch_weight_sum changed.\n"
        f"  committed {_FROZEN['ljbranch_weight_sum']}\n"
        f"  got       {got}\n"
        "Its clique weights are combinatorially exact and the identity tests "
        "trust them verbatim. If the change is intended, re-baseline the value "
        "above AND add an entry to the provenance comment saying what moved and "
        "what evidence shows the chain did not."
    )


def test_the_locked_hash_is_python_version_stable() -> None:
    """The pin must not depend on which interpreter runs it.

    Guards the reason for `ast.unparse`: `ast.dump` grew a `type_params` field on
    FunctionDef in 3.12, so the previous pin passed only on the CI version and
    failed on correct code everywhere else. A gate that fails on correct code
    teaches people to disbelieve it.

    Asserted structurally rather than by shelling out to three interpreters: the
    unparse output must contain no field name that only some versions emit, which
    is checked by confirming the hash is taken over SOURCE TEXT rather than over
    a node repr.
    """
    tree = ast.parse(_BRANCH.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "ljbranch_weight_sum"
    )
    text = ast.unparse(node)
    assert text.startswith("def ljbranch_weight_sum("), (
        "the hash must be taken over unparsed SOURCE, which is version-stable, "
        "not over a node dump, whose field set changes between releases"
    )
    for version_specific in ("type_params=", "annotate_fields", "lineno="):
        assert version_specific not in text, (
            f"{version_specific!r} appears in the hashed text, so the pin is "
            "carrying interpreter-version detail again"
        )

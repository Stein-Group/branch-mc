"""The overlap convention exists once, in geometry.py, and stays there.

This is a standing invariant rather than a preference. Four coexisting copies of
`d <= floor - 1e-12` were live in this package at once, and the cost of that has
already been paid: the copies happened to agree, so nothing failed, and the
disagreement when it came was measurable but not attributable. The multi-species
work makes it worse, because the number being compared against is now per pair
and a copy that reads the wrong element of `core_floor[si, sj]` gets one whole
species wrong in one direction only.

So the rule is mechanical: the tolerance literal appears in `geometry.py` and
nowhere else in `gcmc/`, and every overlap decision routes through
`geometry.overlaps`, `geometry.pair_overlaps` or `overlap.overlap_radius`.
Modules that need the constant import it (`samplers/channel.py` and
`mof_channel.py` both do).

A new site that genuinely cannot use the helpers -- one working in squared
distances, say -- takes the threshold from `overlap_radius` and squares it, which
is what `mof_field.overlaps_many` does. That still leaves one subtraction in one
place.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

from gcmc import geometry
from gcmc.potentials import overlap
from gcmc.potentials.pair_table import PairTable, overlaps_table

_PKG = Path(geometry.__file__).parent
_GEOMETRY = _PKG / "geometry.py"

# The literal, written so this file does not itself contain a second copy that
# the scan would then have to special-case.
_TOL_RE = re.compile(r"1e-12|1\.0e-12|1E-12")


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Source lines with comments and docstrings removed, so prose about the
    convention is not mistaken for an implementation of it."""
    src = path.read_text()
    tree = ast.parse(src)
    doc_spans: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            doc_spans.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    out = []
    for i, line in enumerate(src.splitlines(), start=1):
        if i in doc_spans:
            continue
        code = line.split("#", 1)[0]
        if code.strip():
            out.append((i, code))
    return out


def test_the_tolerance_literal_lives_only_in_geometry():
    """Every other module imports OVERLAP_TOL or calls a helper.

    Scoped to the overlap tolerance specifically: 1e-12 is also a perfectly
    reasonable number for an unrelated closeness assertion, so a hit is reported
    with its line for a human to read rather than being assumed guilty. What the
    test forbids is a hit that is subtracted from a floor or a sigma.
    """
    offenders = []
    for path in sorted(_PKG.rglob("*.py")):
        if path == _GEOMETRY:
            continue
        for lineno, code in _code_lines(path):
            if not _TOL_RE.search(code):
                continue
            # An overlap comparison subtracts the tolerance from a floor. An
            # unrelated tolerance (an allclose atol, a sum check) does not.
            if re.search(r"(floor|sigma|diameter|cutoff)\w*\s*-\s*1", code):
                offenders.append(f"{path.relative_to(_PKG)}:{lineno}: {code.strip()}")
    assert not offenders, (
        "the overlap comparison is reimplemented outside geometry.py:\n  "
        + "\n  ".join(offenders)
        + "\nRoute these through overlap.overlaps / pair_overlaps / "
        "overlap_radius instead."
    )


def test_pair_overlaps_is_the_scalar_helper_at_one_species():
    """At one type the per-pair form must be the global form exactly, or the
    single-species reduction stops being a special case of the mixture."""
    d = np.array([0.5, 0.9 - 1e-13, 0.9, 0.9 + 1e-13, 1.5])
    floor = np.array([[0.9]])
    got = overlap.pair_overlaps(d, 0, 0, floor)
    want = overlap.overlaps(d, 0.9)
    assert got.tolist() == want.tolist()


def test_a_pair_exactly_at_its_floor_is_admissible():
    """The convention is strictly-below, per pair as well as globally. This is the
    boundary the four copies were most likely to disagree on."""
    floor = np.array([[0.8, 0.95], [0.95, 1.1]])
    for i in (0, 1):
        for j in (0, 1):
            f = floor[i, j]
            assert not overlap.pair_overlaps(f, i, j, floor)
            assert overlap.pair_overlaps(f - 1e-9, i, j, floor)


def test_the_pair_floor_is_read_the_right_way_round():
    """An asymmetric read would be invisible in a symmetric table, so this uses a
    deliberately asymmetric one. The table the engine gets is symmetric by
    construction; the point is that the INDEXING is checked independently of that
    guarantee."""
    floor = np.array([[0.5, 0.6], [9.0, 0.7]])  # not symmetric, on purpose
    # 1.0 sits ABOVE floor[0, 1] = 0.6 and BELOW floor[1, 0] = 9.0, so the two
    # index orders must disagree. Picking a distance below both would have made
    # the test pass whichever way round the read went.
    assert overlap.pair_overlaps(1.0, 0, 1, floor).item() is False
    assert overlap.pair_overlaps(1.0, 1, 0, floor).item() is True


def test_pair_overlaps_broadcasts_a_neighbour_column():
    """The hot-path shape: one trial species against an array of neighbour
    species, one distance each."""
    floor = np.array([[0.8, 0.95], [0.95, 1.1]])
    d = np.array([0.85, 0.85, 1.05])
    sj = np.array([0, 1, 1])
    got = overlap.pair_overlaps(d, 1, sj, floor)
    assert got.tolist() == [True, True, True]
    got = overlap.pair_overlaps(d, 0, sj, floor)
    assert got.tolist() == [False, True, False]


def test_max_core_floor_is_the_query_reach():
    floor = np.array([[0.8, 0.95], [0.95, 1.1]])
    assert overlap.max_core_floor(floor) == 1.1
    # a query at this radius cannot miss an overlap for any pair
    assert overlap.overlap_radius(overlap.max_core_floor(floor)) < 1.1


def test_overlaps_table_delegates_rather_than_reimplementing(monkeypatch):
    """Behavioural proof that the wrapper is a wrapper: break the central helper
    and the table form must break with it."""
    table = PairTable.from_reduced(("A",), 1, np.ones((1, 1)), np.ones((1, 1)), 0.9)
    assert overlaps_table(0.5, 0, 0, table)
    monkeypatch.setattr(overlap, "pair_overlaps", lambda *a, **k: "sentinel")
    assert overlaps_table(0.5, 0, 0, table) == "sentinel"


@pytest.mark.parametrize("mod", ["gcmc.mof_field", "gcmc.mof_metropolis"])
def test_the_mof_layer_imports_the_overlap_convention(mod):
    """Both MOF overlap sites were independent copies until P3. If one loses the
    import it has almost certainly grown a copy back.

    Renamed at S2: the convention moved from `geometry` (layer 2) down to
    `potentials.overlap` (layer 1), because every part of it is a function of
    sigma and none of it knows about a box. The guard is unchanged in intent --
    only the name of the module that must be imported has moved."""
    import importlib

    assert hasattr(importlib.import_module(mod), "overlap")

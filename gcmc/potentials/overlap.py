"""The overlap convention: what counts as too close, and how close is too close.

WHY THIS IS LAYER 1 AND NOT LAYER 2 (S2, correcting restructure_plan.md §5).
These four functions lived in `geometry.py` until S2, and three layer-1 modules
-- `species`, `pair_table`, `mof_field` -- had to reach UP into layer 2 to use
them. That was three of the four upward imports in the whole package, and the
layer-ladder test is what surfaced it.

The resolution is not to allow the exception. It is that these were never
geometry: every one of them is a function of SIGMA. `overlap_radius` is
`sigma - tol`; `pair_overlaps` compares against `c * sigma_ij` out of the pair
table; `max_core_floor` reduces that same table. Nothing here knows about a box,
a cell list, a minimum image or a periodic wrap -- which is precisely what
layer 2 is for.

So the split is now clean:

    L1  this module      owns the FLOOR: what separation counts as an overlap
    L2  geometry         owns the DISTANCE: how far apart two things are
    L3  system           applies one to the other

and no module reaches up.

WHY ONE MODULE AND NOT FOUR LITERALS. The convention is `d <= floor - tol`, with
a pair at exactly its floor admissible. That boundary is a measure-zero set, so
four copies that disagree about `<` versus `<=` would agree on every test anyone
thought to write and diverge on the one configuration that matters. This
codebase has already paid that cost once. There is one comparison, and it is
here.

UNITS: reduced throughout. `sigma`, `d` and every entry of `core_floor` are
lengths in the same reduced scale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._types import BoolArray, FloatArray, IntArray

__all__ = [
    "OVERLAP_TOL",
    "max_core_floor",
    "overlap_radius",
    "overlaps",
    "pair_overlaps",
]

#: Slack subtracted before every overlap comparison, so a pair sitting exactly
#: at its floor is admissible rather than rejected by floating-point noise.
#: One constant, read by all four functions below; never re-typed at a call site.
OVERLAP_TOL = 1e-12


def overlap_radius(sigma: FloatArray | float) -> Any:
    """The largest separation that still counts as overlapping, i.e. the radius
    to hand a `query_ball_point` / `query_pairs` overlap search so that a pair at
    exactly sigma is excluded from the overlap set.

    Accepts an array of floors as well as one diameter, because a caller working
    in SQUARED distances needs the threshold rather than the verdict and would
    otherwise write the subtraction itself. Returns whatever it was given, minus
    the tolerance, so a scalar in gives a scalar out.
    """
    return sigma - OVERLAP_TOL


def overlaps(d: FloatArray | float, sigma: float) -> BoolArray:
    """Boolean overlap indicator for separations `d` against diameter `sigma`."""
    return np.asarray(d, dtype=float) <= sigma - OVERLAP_TOL


def pair_overlaps(
    d: FloatArray | float,
    si: IntArray | int,
    sj: IntArray | int,
    core_floor: FloatArray,
) -> BoolArray:
    """Boolean overlap indicator against the PER-PAIR floor `core_floor[si, sj]`.

    The multi-species form of `overlaps`, and the only thing that differs is
    which number the comparison is against: `c * sigma_ij` instead of one global
    diameter. The convention `d <= floor - OVERLAP_TOL` is unchanged, so a pair
    at exactly its floor is still admissible.

    This exists so the comparison keeps living in one place. A scalar floor gets
    a whole species wrong in one direction -- a separation admissible for the
    small-small pair can be an overlap for the large-large pair -- and the cost of
    four coexisting copies of a convention has already been paid once in this
    codebase. `si` and `sj` may be scalars or arrays; numpy's fancy indexing
    broadcasts them against `d` the same way either way.
    """
    return np.asarray(d, dtype=float) <= np.asarray(core_floor)[si, sj] - OVERLAP_TOL


def max_core_floor(core_floor: FloatArray) -> float:
    """The largest per-pair floor in a table.

    Two uses, both of which need the same number and neither of which should
    compute it inline. It is the scalar radius a neighbour query must reach to be
    certain of catching every possible overlap, via `overlap_radius`; and it is
    the term that sets the cell-list cutoff, `max(rc, max_ij core_floor_ij)`, so
    the grid pitch is keyed to the largest radius any query will use.
    """
    return float(np.max(core_floor))

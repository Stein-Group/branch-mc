"""Partition the Cartesian product of the axes into ACCEPTED and REJECTED.

The validator is the only authority. This module builds every point, hands each
to `sim_setup.from_params`, and sorts it by what came back:

    accepted   from_params returned a Config     -> test_smoke runs it
    rejected   from_params raised ValueError     -> test_coverage reads the
                                                    reason and checks it names
                                                    the offending key

Two properties follow, and they are the reason for doing it this way rather
than writing the list out:

1. **A new feature cannot be added silently.** Put a sampler in the schema enum
   and it appears in the product; if `from_params` accepts it, it gets run.
   There is no second list to update and therefore no list to forget.

2. **The rejections are tested too.** A validator that stops rejecting an
   illegal combination is as much a regression as one that starts rejecting a
   legal one, and the second kind is invisible to a suite that only runs the
   things that work. The rejected set is committed (`expected.json`) so a
   change in EITHER direction shows up as a diff.

Cost note. Building the product is pure validation -- no chain, no allocation
beyond a Config -- so enumerating a few thousand points costs well under a
second. Only the accepted set is ever advanced, and that is what the tier-2
budget pays for.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, NamedTuple

from gcmc.run import sim_setup

from . import _axes
from ._axes import AXES, Point, build_params

_EXPECTED = Path(__file__).resolve().parent / "expected.json"


class Verdict(NamedTuple):
    """What the validator said about one point.

    `reason` is the exception message verbatim for a rejection, and empty for an
    acceptance. It is kept in full rather than summarised because the committed
    expectation file is meant to be READ in a diff: "this combination started
    being rejected, and here is the sentence explaining why" is the review this
    file exists to make possible.
    """

    point: Point
    accepted: bool
    reason: str


def _all_points() -> list[Point]:
    """Every point of the Cartesian product of `AXES`, in a stable order.

    Ordering is by axis declaration order so the enumeration is reproducible
    across runs and machines; an unstable order would make the committed
    expectation file churn on every regeneration and destroy its reviewability.
    """
    names = [a.name for a in AXES]
    fields = list(Point._fields)
    if names != fields:
        raise AssertionError(
            f"AXES and Point have drifted apart: axes={names} point={fields}. "
            "They are two spellings of one thing and must be edited together."
        )
    return [Point(*combo) for combo in itertools.product(*(a.values for a in AXES))]


def _redundant(point: Point) -> bool:
    """Whether this point is a DUPLICATE of one already in the product.

    Some axes are only consulted by some samplers -- `channel_length` means
    nothing to metropolis -- so the raw product contains points that build
    byte-identical params. Running both wastes tier-2 budget and reports one
    feature twice. This collapses them by keeping only the first value of an
    axis the point does not consult.

    This is NOT a legality filter. A collapsed point is identical to one that
    remains, so nothing is lost; a filter that dropped a DISTINCT point would be
    hiding a feature and is exactly what this file refuses to do.
    """
    axis = {a.name: a for a in AXES}

    # `channel_length` is read only by the fixed-length channel. The cyclic rod
    # sizes itself per move, so it must NOT multiply points across this axis.
    if (
        point.sampler != "channel"
        and point.channel_length != axis["channel_length"].values[0]
    ):
        return True

    # metropolis writes no depth key at all (it is depth-0 by definition), so
    # every depth value builds the same params.
    if point.sampler == "metropolis" and point.depth != axis["depth"].values[0]:
        return True

    # A species file states which Hamiltonian it describes; pairing it with the
    # other one is a contradiction the validator rejects, and enumerating that
    # contradiction across every other axis buries the real rejections under
    # more than a thousand copies of one message.
    want = _axes.SPECIES_FILE_POTENTIAL.get(point.species_file or "")
    return want is not None and point.potential != want


def enumerate_points() -> list[Verdict]:
    """Build every point and ask the validator about it.

    Returns
    -------
    list of Verdict
        One entry per non-redundant point, in stable order.
    """
    out: list[Verdict] = []
    for point in _all_points():
        if _redundant(point):
            continue
        try:
            sim_setup.from_params(build_params(point))
        except Exception as exc:  # any rejection is a rejection
            out.append(Verdict(point, False, f"{type(exc).__name__}: {exc}"))
        else:
            out.append(Verdict(point, True, ""))
    return out


def accepted(verdicts: list[Verdict] | None = None) -> list[Point]:
    """Just the points the validator accepts: the modes tier 2 actually runs."""
    return [v.point for v in (verdicts or enumerate_points()) if v.accepted]


def pairs_of(point: Point) -> set[tuple[str, Any, str, Any]]:
    """Every unordered pair of (axis, value) this point exhibits.

    A pair is the unit of coverage rather than a whole point because almost all
    real defects are either single-value ("reweight crashes") or two-value
    ("the channel breaks in 2D"). Three-way interactions exist but are rare
    enough that paying 10x the wall clock for them on every push is the wrong
    trade; tier 3 runs the full product nightly and catches them there.
    """
    fields = list(Point._fields)
    vals = [getattr(point, f) for f in fields]
    out = set()
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            out.add((fields[i], vals[i], fields[j], vals[j]))
    return out


def pairwise_cover(points: list[Point]) -> list[Point]:
    """A subset of `points` exhibiting every legal (axis, value) PAIR at least once.

    WHY NOT RUN EVERYTHING. The validator accepts 642 points. At roughly a third
    of a second each that is four minutes, which does not fit a per-push budget
    once parallel points (process spawn) are counted -- and a tier that people
    start skipping because it is slow protects nothing.

    WHY NOT HAND-PICK. A curated subset is the failure mode this whole module
    exists to avoid (R13). It goes stale invisibly.

    So the subset is COMPUTED: greedy set cover over the uncovered pairs, taking
    at each step the point that covers the most. Greedy is within a log factor of
    optimal for set cover, which is far better than needed here -- the point is
    the guarantee ("every legal pair is exercised"), not minimality.

    Determinism matters as much as size: the same input must give the same
    subset on every machine, or the committed digests are worthless. Ties are
    therefore broken on `point.id`, never on iteration order.

    Parameters
    ----------
    points
        The accepted points, in stable order.

    Returns
    -------
    list of Point
        A covering subset, ordered as selected (most-covering first), which also
        makes a truncated run the best possible partial cover.
    """
    remaining: set[tuple[str, Any, str, Any]] = set()
    catalogue = {p.id: pairs_of(p) for p in points}
    for ps in catalogue.values():
        remaining |= ps

    by_id = {p.id: p for p in points}
    chosen: list[Point] = []
    while remaining:
        best_id = max(
            sorted(by_id),  # sorted() makes the tie-break deterministic
            key=lambda pid: (len(catalogue[pid] & remaining), pid),
        )
        gained = catalogue[best_id] & remaining
        if not gained:
            break  # nothing left is reachable; the rest of `remaining` is unpaired
        chosen.append(by_id[best_id])
        remaining -= gained
    return chosen


def load_expected() -> dict[str, Any]:
    if not _EXPECTED.exists():
        return {}
    with _EXPECTED.open() as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def dump_expected(verdicts: list[Verdict]) -> dict[str, Any]:
    """Serialise a partition for committing.

    Reasons are stored truncated to their first line. The full message is
    valuable when a test fails, but a multi-line message in a committed JSON
    turns every reword of an error string into a large diff, which trains people
    to regenerate the file without reading it.
    """
    return {
        v.point.id: {
            "accepted": v.accepted,
            "reason": v.reason.splitlines()[0] if v.reason else "",
        }
        for v in verdicts
    }

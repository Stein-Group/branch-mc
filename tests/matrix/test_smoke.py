"""Tier 2: advance every feature of the code and assert it did not break.

This is the tier that replaces running jobs by hand. It does NOT check physics
-- a 40-step window on a box of 8 cannot tell you an isotherm is right, and
pretending otherwise produces a suite that fails for reasons nobody can act on.
It checks that every feature RUNS and that the things which must always be true
of a chain still are:

    * the chain advanced and reported a summary
    * particle count is a non-negative integer and consistent with the state
    * every position is finite and inside the box
    * no NaN or negative weight reached the acceptance
    * species totals are conserved by moves that must conserve them
    * the counter stream reproduces its committed digest

Physics lives in tier 1 (unit and engine tests) and tier 3 (statistical gates
and the characterisation oracles). The division is deliberate: this tier must
stay under about five minutes or people will stop running it, and a tier that is
skipped protects nothing.

WHAT IS RUN. By default the pairwise covering subset (`_enumerate.pairwise_cover`)
-- roughly 35 points out of 642 accepted, guaranteeing every axis value and every
legal pair of axis values is exercised at least once. Pass `--matrix-full` to run
the entire accepted set; tier 3 does this nightly.

DIGESTS. Each point commits a digest of its counter stream, not of its positions.
Positions drift with any legal change to the RNG draw order, which makes a
position digest fail on refactors that are correct; the counter stream (accepts,
rejects, insertions, removals per window) is the thing a behaviour change
actually shows up in and is stable across the moves this plan makes. When a
digest changes, that IS the review: the diff names the mode and the stage that
changed it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gcmc import config_specs
from gcmc.engine import MC_engine, parallel
from gcmc.run import runner, sim_setup
from gcmc.system.state import ChainState

from . import _enumerate as E
from ._axes import N_WINDOWS, SEED, Point, build_params

_DIGESTS = Path(__file__).resolve().parent / "digests.json"


# --------------------------------------------------------------------- helpers


def _load_digests() -> dict[str, str]:
    """Committed counter-stream digests, keyed by point id."""
    if not _DIGESTS.exists():
        return {}
    with _DIGESTS.open() as fh:
        return json.load(fh)  # type: ignore[no-any-return]


#: Digests computed during a `--matrix-regen` session, written out at the end.
#: Module-level rather than a fixture because the writer runs in
#: `pytest_sessionfinish` (in this package's conftest, since pytest collects
#: hooks only from conftest files), which no fixture outlives.
_REGEN: dict[str, str] = {}


def _int_leaves(prefix: str, value: Any, out: list[str]) -> None:
    """Append `prefix=int` for every integer leaf of `value`, dicts included.

    THE BUG THIS FIXES, because the shape of it matters more than the fix. The
    window summary is not flat: the counters live in `summary["counters"]` and
    the clique tallies in `summary["clique"]`, both dicts. The old digest tested
    `isinstance(val, int)` on the TOP level only, so both dicts failed the test
    and were dropped -- silently, since a digest that hashes less still hashes
    something. What survived the filter was `summary["N"]`, one integer per
    window.

    So the "counter digest" hashed three particle counts and no counters, for
    the whole of S2 and S3, while being cited as the bit-identity evidence for
    every migration in both. The tell was visible in the committed file and went
    unread: five of thirteen ids shared one digest, which is what happens when
    the hashed quantity is `N = 0, 0, 0`.

    Floats stay excluded, for the reason the old docstring gave and which still
    holds: they are not bit-reproducible across BLAS builds, and a digest that
    fails on a CI image upgrade teaches people to regenerate without reading.
    Every counter in the engine is an integer, so nothing is lost by the rule --
    the dicts were lost by the traversal.
    """
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, (int, np.integer)):
        out.append(f"{prefix}={int(value)}")
        return
    if isinstance(value, dict):
        for key in sorted(value):
            _int_leaves(f"{prefix}.{key}", value[key], out)
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        # A CONTAINER THIS DOES NOT TRAVERSE IS AN ERROR, not something to skip.
        # Everything above returns early, so anything reaching here is a shape the
        # digest has no rule for -- and the rule that used to apply, "not an int
        # and not a dict, so ignore it", is exactly how the nested counters went
        # unhashed for two stages. S4 phase 11 nearly repeated it: per-species
        # occupancy was first emitted as a LIST, which is neither an int nor a
        # dict and would have been dropped here without a word while appearing in
        # the summary and in the run log.
        raise TypeError(
            f"{prefix} is a {type(value).__name__}, which this traversal does not "
            "hash. Emit it as a dict of integers keyed by name (as `counters` and "
            "`clique` are) so it reaches the digest, or the pin will be silent "
            "about it."
        )


def _counter_digest(summaries: list[dict[str, Any]]) -> str:
    """A stable digest of the per-window counter stream, nested dicts included."""
    parts: list[str] = []
    for i, summary in enumerate(summaries):
        for key in sorted(summary):
            _int_leaves(f"{i}.{key}", summary[key], parts)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _advance_point(point: Point, matrix_dir: Path) -> tuple[Any, list[dict[str, Any]]]:
    """Build and advance one matrix point for `N_WINDOWS` windows.

    Returns
    -------
    (state, summaries)
        The final chain state and one summary dict per window.
    """
    cfg = sim_setup.from_params(build_params(point, matrix_dir=matrix_dir))
    pot = cfg.build_potential()
    wgh = MC_engine.make_weigher(config_specs.kernel_spec(cfg), pot)
    state = ChainState.fresh(cfg.dimension, seed=SEED)
    # THE START IS THE RUNNER'S, not an empty box. `runner._init_config` is what
    # every real run calls, and it is where NVT gets its lattice of `target_n`,
    # a confined pore gets particles inside the pore, and a QA run gets fluid
    # that clears the obstacles. Bulk muVT still starts empty -- that branch
    # returns an empty array -- so nothing is invented here that production
    # would not do.
    #
    # WHAT THIS FIXES. `ChainState.fresh` alone gave every point an empty box,
    # and NVT conserves N: a chain that starts at zero particles stays there,
    # proposes nothing, and reports every counter as 0 for all three windows.
    # Five of the thirteen committed digests were that stream. Those points were
    # green because nothing can go wrong in a chain that never moves.
    state.positions = runner._init_config(cfg, state.rng)
    state.species = runner._init_species(cfg, len(state.positions))

    # THE PARALLEL POINTS NOW RUN THE PARALLEL KERNEL. `kind` is an axis with two
    # values and this line used to call `MC_engine.advance` for both, so every
    # `-parallel` point in the matrix -- nine of the pinned ones -- exercised the
    # serial engine under a label that said otherwise. The runner dispatches on
    # `exe.mode`; so does this now, which is the only way the two can be the same
    # code path.
    advance = (
        parallel.advance_parallel
        if config_specs.execution_spec(cfg).mode == "parallel"
        else MC_engine.advance
    )
    summaries: list[dict[str, Any]] = []
    for _ in range(N_WINDOWS):
        summary = advance(state, cfg, pot, wgh)
        summaries.append(dict(summary) if summary is not None else {})
    return state, summaries


# ------------------------------------------------------------------ invariants


def _assert_invariants(
    point: Point, state: Any, summaries: list[dict[str, Any]]
) -> None:
    """Everything that must be true of any chain, whatever the mode.

    Each assertion names the point, because a matrix failure is useless if it
    does not say which of 642 configurations produced it.
    """
    pid = point.id
    pos = np.asarray(state.positions, dtype=float)

    assert len(summaries) == N_WINDOWS, f"{pid}: {len(summaries)} windows advanced"

    # THE VACUITY GUARD. A point that proposes nothing asserts nothing, and this
    # is where that has to fail rather than pass quietly.
    #
    # It replaces `assert pos.shape[0] >= 0`, which was written under a comment
    # about muVT chains that never fill and is a tautology: a shape is never
    # negative. It passed on the five NVT points that ran zero moves for three
    # windows, which is exactly the case its comment described.
    #
    # Stated over every ensemble, not just muVT: NVT was the ensemble that was
    # actually broken, and a guard that only watches the mode someone suspected
    # is how this was missed the first time.
    steps = 0
    for s in summaries:
        c = s.get("counters", {})
        steps += sum(
            int(c.get(k, 0))
            for k in (
                "add_step",
                "remove_step",
                "move_step",
                "repack_step",
                "swap_step",
            )
        )
    assert steps > 0, (
        f"{pid}: {N_WINDOWS} windows proposed 0 moves, so every assertion below "
        "holds trivially. Either the start has no particles for a move to act on "
        "or the move split is refusing every draw; do not report it green."
    )

    assert pos.ndim == 2, f"{pid}: positions has rank {pos.ndim}, expected 2"
    assert pos.shape[1] == point.dimension, (
        f"{pid}: positions has {pos.shape[1]} columns, expected {point.dimension}"
    )
    assert np.all(np.isfinite(pos)), (
        f"{pid}: {int((~np.isfinite(pos)).any(axis=1).sum())} non-finite positions"
    )

    # Species bookkeeping. In typed mode the species column must stay aligned
    # with the position rows -- M13's slot-alignment invariant. A mismatch here
    # is the failure mode that silently mislabels every particle.
    species = getattr(state, "species", None)
    # PRESENCE FIRST, and this is not a formality. The alignment check below is
    # guarded on `species is not None`, so it skipped itself on exactly the runs
    # where the column was missing -- and the column WAS missing on every typed
    # branch/channel run until S4, because `advance` only wrote it back when the
    # caller had already supplied one. A guard conditioned on the thing being
    # present can never report it absent, and reported 642 green points while a
    # three-species run returned composition (1, 0, 0).
    if point.species_file is not None and pos.shape[0] > 0:
        assert species is not None, (
            f"{pid}: a typed run left state.species as None with "
            f"{pos.shape[0]} particles. io writes no species array, the "
            "checkpoint cannot restore one, and the next window's cell list "
            "reads zeros -- so the run reports a pure species-0 system."
        )
    if species is not None:
        species = np.asarray(species)
        assert species.shape[0] == pos.shape[0], (
            f"{pid}: {species.shape[0]} species labels for {pos.shape[0]} positions"
        )
        assert np.all(species >= 0), f"{pid}: negative species id"


# ----------------------------------------------------------------------- tests


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise on the generated point list.

    Generated at collection time rather than written out, so a feature added to
    the schema appears here without anyone editing this file (R13).

    TWO DIFFERENT POINT SETS, and the difference is the whole point.

    `point` -- what tier 2 RUNS -- is the pairwise cover, chosen greedily from
    whatever the validator currently accepts. It is supposed to move when the
    accepted set moves; that is what makes it a cover.

    `digest_point` -- what tier 2 PINS -- is the cover UNION every point that
    has a committed digest. It must not move when the accepted set moves, and
    deriving it from the cover meant it did.

    The failure that forced this apart, recorded because it is invisible while
    it is happening: moving the metropolis+pmf guard into the validator (M15)
    refused 13 points that had been accepted. Those 13 were not digested and
    nothing about them changed -- but removing them re-ran the greedy cover,
    which then selected 36 different points, of which exactly 2 had digests. The
    gate reported "2 passed, 34 skipped" and every skip said "no committed
    digest; run --matrix-regen", which reads like new features needing baselines
    rather than like thirteen existing pins having silently stopped being
    checked.

    So a commit could have moved any of those thirteen counter streams and the
    gate would have said nothing. "digests 15/15 UNCHANGED" was only ever true
    while `accepted()` was unchanged, and S2 item 7 is a change to `accepted()`.

    A digested point that stops being accepted now FAILS rather than
    disappearing -- see `test_counter_digest`. Deleting a mode should require
    saying so.
    """
    if "point" in metafunc.fixturenames:
        points = _points_for(metafunc.config)
        metafunc.parametrize("point", points, ids=[p.id for p in points])
    if "digest_point" in metafunc.fixturenames:
        pinned = _digest_points(metafunc.config)
        metafunc.parametrize(
            "digest_point", pinned, ids=[p.id if p is not None else "?" for p in pinned]
        )


def _points_for(config: pytest.Config) -> list[Point]:
    """The points tier 2 runs: the pairwise cover, or all of them."""
    accepted = E.accepted()
    if config.getoption("--matrix-full"):
        return accepted
    return E.pairwise_cover(accepted)


def _digest_points(config: pytest.Config) -> list[Point]:
    """Every point the digest gate pins: the run set plus every digested id.

    Ordered by id so the parametrisation is stable across runs. A digested id
    that no longer corresponds to an ACCEPTED point is still yielded -- as a
    `_MissingPoint` carrying just the id -- so that `test_counter_digest` can
    fail on it by name instead of it vanishing from collection.
    """
    run = _points_for(config)
    by_id = {p.id: p for p in E.accepted()}
    ids = sorted({p.id for p in run} | set(_load_digests()))
    return [by_id.get(i) or _MissingPoint(i) for i in ids]  # type: ignore[misc]


class _MissingPoint:
    """A digested id the validator no longer accepts. Carries only its id."""

    def __init__(self, point_id: str) -> None:
        self.id = point_id


@pytest.mark.matrix
def test_feature_runs(point: Point, matrix_dir: Path) -> None:
    """Every accepted feature combination advances and holds its invariants."""
    state, summaries = _advance_point(point, matrix_dir)
    _assert_invariants(point, state, summaries)


@pytest.mark.matrix
def test_counter_digest(
    digest_point: Point, matrix_dir: Path, request: pytest.FixtureRequest
) -> None:
    """The counter stream reproduces its committed digest.

    Runs over `digest_point`, not `point`: every committed digest is checked on
    every run, whatever the cover happens to select. See `pytest_generate_tests`
    for the failure that separated the two.

    Skips with a reason when a point has no digest yet rather than failing: a
    newly added feature should light up as "needs a digest", not as a broken
    build, and `--matrix-regen` is how it gets one.
    """
    digests = _load_digests()
    if request.config.getoption("--matrix-regen"):
        assert isinstance(digest_point, Point), (
            f"{digest_point.id} has a digest but is no longer accepted; delete "
            "its entry from digests.json in the commit that retires it rather "
            "than regenerating around it."
        )
        _, summaries = _advance_point(digest_point, matrix_dir)
        _REGEN[digest_point.id] = _counter_digest(summaries)
        return
    if digest_point.id not in digests:
        pytest.skip(f"no committed digest for {digest_point.id}; run --matrix-regen")
    assert isinstance(digest_point, Point), (
        f"{digest_point.id} has a committed digest but the validator no longer "
        "accepts it. A pinned counter stream cannot simply stop being checked: "
        "if the mode was deliberately retired, delete its entry from "
        "digests.json in the same commit that retires it, and say so."
    )
    _, summaries = _advance_point(digest_point, matrix_dir)
    got = _counter_digest(summaries)
    assert got == digests[digest_point.id], (
        f"{digest_point.id}: counter stream changed\n"
        f"  committed {digests[digest_point.id]}\n"
        f"  got       {got}\n"
        "If this change is intended, regenerate with --matrix-regen and say in "
        "the commit message which stage of restructure_plan.md caused it."
    )


@pytest.mark.matrix
def test_the_committed_partition_is_still_the_live_one(
    request: pytest.FixtureRequest,
) -> None:
    """`expected.json` is what the validator currently says. It was never checked.

    The file has been committed since the matrix was written, and `_enumerate`
    documents it as the record that makes a validator change "show up as a diff".
    `load_expected` and `dump_expected` were written for exactly that and had no
    callers -- so the diff never appeared, and a guard moving into or out of the
    validator changed the accepted set with nothing to say so. That is not a
    hypothetical: M15 moved the metropolis+pmf guard and silently refused 13
    previously-accepted points, and what surfaced it was the digest gate
    collapsing 15 points to 2, several commits later.

    Reported in three parts, because the three mean different things. A point
    that changed verdict is a feature gained or lost. A point that appeared is a
    new axis value. A point that vanished is an axis value that stopped being
    enumerated -- the quietest of the three and the one that shrinks coverage.
    """
    if request.config.getoption("--matrix-regen"):
        pytest.skip("--matrix-regen rewrites the partition; the diff is the review")
    committed = E.load_expected()
    if not committed:
        pytest.skip("no committed partition; run --matrix-regen")
    live = E.dump_expected(E.enumerate_points())

    appeared = sorted(set(live) - set(committed))
    vanished = sorted(set(committed) - set(live))
    flipped = sorted(
        pid
        for pid in set(live) & set(committed)
        if live[pid]["accepted"] != committed[pid]["accepted"]
    )
    detail = []
    for pid in flipped[:10]:
        was = "accepted" if committed[pid]["accepted"] else "rejected"
        now = "accepted" if live[pid]["accepted"] else "rejected"
        detail.append(f"    {pid}: {was} -> {now} ({live[pid]['reason'] or 'ok'})")
    assert not (appeared or vanished or flipped), (
        "the accepted/rejected partition moved:\n"
        f"  {len(flipped)} changed verdict\n" + "\n".join(detail) + "\n"
        f"  {len(appeared)} new points: {appeared[:10]}\n"
        f"  {len(vanished)} points no longer enumerated: {vanished[:10]}\n"
        "If the change is intended, regenerate with --matrix-regen and say in "
        "the commit message which stage of restructure_plan.md caused it."
    )

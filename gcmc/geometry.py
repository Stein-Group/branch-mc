"""Geometry: the pure spatial work, with no physics in it.

This layer answers "where are the particles" questions and nothing about
energy. The weight of a trial is applied afterwards by the potential (or, for
reweighting, by that sampler), so the search here is identical across every
method and the method-specific code lives in one place elsewhere.

Serial pieces only in this module:
  wrap                  periodic wrap of coordinates into [0, side)
  indices_in_cavity     indices of particles inside the periodic box of
                        half-width `radius` about a centre (per-axis, wrapped)
  find_neighbors        kd-tree query plus the non-overlap filter, returning
                        the raw arrays (coords, dists, idxs, keep_mask)

`find_neighbors` returns None when every trial overlaps the system, matching
the original's (None, None) short-circuit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import numpy.typing as npt

from ._types import BoolArray, FloatArray, IntArray
from .config_specs import EnsembleSpec, SamplerSpec

# S2: the overlap CONVENTION moved down to layer 1 (potentials.overlap): it is a
# function of sigma, not of geometry. This module still asks the question "do
# these two overlap?", so it imports the answer downward, which is legal and is
# the direction the ladder wants.
from .potentials.overlap import overlap_radius

if TYPE_CHECKING:
    from .neighbors import CellList

# (trial coords, padded distances, padded neighbour ids, keep mask or None):
# the raw neighbour result every weighting mode consumes. dists/idxs are
# whatever the container returned (scipy gives float64/intp), so they are
# typed as the arrays they are used as, not re-cast at runtime.
#: (coords, dists, idxs, keep_mask, species) for the trials that survived the
#: overlap filter. The last two are PER-TRIAL COLUMNS carried through the same
#: `[good]` filter as the coordinates: `keep_mask` marks the removal shadow, and
#: `species` the drawn type of each trial (M19). Both are None when the caller
#: passed none. They are threaded here rather than reconstructed by the caller
#: because only this function knows which trials survived, and matching survivors
#: back up by coordinate is exactly the identification M21 forbids.
RawNeighbors: TypeAlias = tuple[
    FloatArray, FloatArray, IntArray, "BoolArray | None", "IntArray | None"
]

# The overlap convention itself now lives one layer down, in
# `potentials/overlap.py`, and is imported above -- see that module for why it is
# layer 1. This module owns DISTANCES; it does not own the floor those distances
# are compared against.


def query_saturating(
    kdtree: Any, points: npt.ArrayLike, k: int, upper_cutoff: float
) -> tuple[FloatArray, IntArray]:
    """kd-tree query for the k nearest neighbours within `upper_cutoff`, doubling
    k until the query is unsaturated (its farthest returned slot is infinite for
    every point) or k reaches the tree size. A fixed-k query that comes back full
    (every slot finite) may have dropped further in-range neighbours, which
    silently truncates any energy summed over that list; doubling until a padded
    slot appears guarantees the returned list holds every neighbour within the
    cutoff. Returns (dists, idxs) as 2-D arrays (one row per point), with the
    kd-tree's inf / N padding kept.

    When the query is not saturated at the requested k (the common case at
    ordinary density) the first k columns are exactly what a plain fixed-k query
    returns, so callers see no change until saturation actually occurs.
    """
    n = kdtree.n
    k = min(int(k), n) if n else int(k)
    while True:
        dist, idx = kdtree.query(points, k=k, distance_upper_bound=upper_cutoff)
        if k == 1:
            dist = np.asarray(dist)[:, None]
            idx = np.asarray(idx)[:, None]
        if k >= n or not np.isfinite(dist[:, -1]).any():
            return dist, idx
        k = min(2 * k, n)


def assert_no_overlap(
    coords: FloatArray,
    sigma: float,
    side: float | FloatArray,
    *,
    what: str = "configuration",
) -> None:
    """Raise unless every pair in `coords` is at least `sigma` apart under the
    per-axis minimum image.

    A hard-core chain can NEVER repair an overlap it starts with: displacement
    moves reject any move that would overlap, and an exchange move only removes
    whole particles, so an overlapping pair stays overlapping for the entire
    run while still contributing to N. The result is a silently inflated density
    (measured: a warm start with ~12 overlapping pairs in a 10 x 10 box pushed
    eta from 0.309 to 0.357 and made every sampler disagree by ~0.02). Nothing
    downstream can detect it, so it is checked once, here, at the point a
    configuration enters the chain.

    Uses the same `sigma - OVERLAP_TOL` convention as the rest of this module,
    so a pair at exactly sigma (a lattice at contact) is legal.
    """
    # local import: this is a one-shot start-up check, and geometry.py must not
    # put a tree in the hot path (tests/engine/test_no_kdtree_in_hot_path.py)
    from scipy.spatial import KDTree

    pts = np.asarray(coords, dtype=float)
    if len(pts) < 2:
        return
    sides = np.broadcast_to(np.asarray(side, dtype=float), (pts.shape[1],))
    tree = KDTree(np.mod(pts, sides), boxsize=sides)
    pairs = tree.query_pairs(r=overlap_radius(sigma), output_type="ndarray")
    if len(pairs) == 0:
        return
    i, j = int(pairs[0, 0]), int(pairs[0, 1])
    d = pts[i] - pts[j]
    d -= np.round(d / sides) * sides
    raise ValueError(
        f"{what}: {len(pairs)} overlapping pair(s) at a hard-core diameter of "
        f"{sigma:g}; the closest is particles {i} and {j} at "
        f"{float(np.sqrt((d**2).sum())):.6g}. A hard-core chain cannot separate "
        f"an overlap it starts with, so this run would report an inflated "
        f"density for its entire length. Re-equilibrate the warm start, or "
        f"check that its box matches ensemble.box ({sides})."
    )


def inside_pore(
    coords: FloatArray, ens: EnsembleSpec, samp: SamplerSpec
) -> BoolArray | bool | None:
    """Which centres satisfy a confining hard wall. None when there is no pore.

    `coords` is (M, dim) or a single (dim,) point; the return follows -- an
    array for the first, a plain bool for the second. A sphere of radius
    `samp.radius` must keep its CENTRE off the wall by that radius, which is why
    the clearance is the configured particle radius and not a hardcoded 0.5.

    THREE COPIES BECAME ONE (S2.11). This lived as `MC_engine._inside_pore`
    (L6), was duplicated verbatim for single points as
    `runner._inside_pore_seed` (L7), and was stated a third time as bounds by a
    `MC_engine._pore_bounds` that nothing called. Two live copies of a hard-wall
    test is the kind of duplication that does not announce itself: they agree
    until one is edited, and the symptom is seeded configurations seated inside
    a wall that the engine then refuses to move, which reads as a sampling
    problem rather than a geometry one.

    It belongs HERE because that is the layer both callers can see -- the
    ladder's own instruction when a shared thing forces an upward import. It is
    also simply true: a wall test asks where a particle is, which is what this
    module is for, and it reads no energy.
    """
    if ens.pore_geometry is None:
        return None
    a = np.atleast_2d(np.asarray(coords, float))
    cl = samp.radius
    if ens.pore_geometry == "slit":
        assert ens.pore_H is not None  # config validation pairs geometry+size
        z0 = (ens.box - ens.pore_H) / 2.0
        z = a[:, 2]
        ok = (z >= z0 + cl) & (z <= z0 + ens.pore_H - cl)
    else:
        assert ens.pore_R is not None  # config validation pairs geometry+size
        c = ens.box / 2.0
        ok = np.hypot(a[:, 0] - c, a[:, 1] - c) <= (ens.pore_R - cl)
    return ok if np.ndim(coords) > 1 else bool(ok[0])


def wrap(coords: FloatArray, side: float | FloatArray) -> FloatArray:
    """Map coordinates into the primary periodic image [0, side)."""
    return np.mod(coords, side)


def _axis_in_cavity(col: FloatArray, c: float, radius: float, side: float) -> BoolArray:
    """Per-axis membership in the periodic interval [c - radius, c + radius]
    on a ring of circumference `side`: when the interval pokes past either
    face it is tested as the union of its two in-box pieces, which is the
    wrap-aware equivalent of |x - c| <= radius. Shared by the O(N) scan and
    the cell-gather shortlist, so both use the SAME inclusion convention
    (closed interval at both ends)."""
    lo = c - radius
    hi = c + radius
    if lo < 0:
        return ((col >= 0) & (col <= hi)) | ((col >= lo + side) & (col <= side))
    if hi > side:
        return ((col >= lo) & (col <= side)) | ((col >= 0) & (col <= hi - side))
    return (col >= lo) & (col <= hi)


def indices_in_cavity(
    coords: FloatArray, center: FloatArray, radius: float, side: float, two_dim: bool
) -> IntArray:
    """Row indices of all particles inside the axis-aligned CUBE cavity of
    half-width `radius` around `center` (periodic, per-axis test). The cavity
    of the depth-0 local exchange move is this cube, not a sphere -- the move's
    acceptance algebra only needs the same body for removal membership and
    trial generation, not isotropy. O(N) full scan; the cell-list shortlist
    version is `indices_in_cavity_cl`."""
    mask = _axis_in_cavity(coords[:, 0], center[0], radius, side)
    mask &= _axis_in_cavity(coords[:, 1], center[1], radius, side)
    if not two_dim:
        mask &= _axis_in_cavity(coords[:, 2], center[2], radius, side)
    return np.nonzero(mask)[0]


def _admissible(
    distances: FloatArray,
    indices: IntArray,
    lower_cutoff: float,
    species: IntArray | None,
    core_floor: FloatArray | None,
    container: Any,
) -> BoolArray:
    """Which trials clear the overlap floor: scalar, or per pair type (A1.4).

    The scalar branch is `np.all(distances >= lower_cutoff, axis=1)` unchanged --
    same expression, same order -- so an untyped run is bit-identical.

    The typed branch compares each pair against `core_floor[t_trial, t_neigh]`.
    `container.species` supplies the neighbour types by SLOT id, which is what
    `indices` holds, so no coordinate is consulted. Pad columns carry an infinite
    distance and clear any finite floor, so padding stays neutral exactly as it
    does in the scalar test.
    """
    if species is None or core_floor is None:
        return np.all(distances >= lower_cutoff, axis=1)
    nt = container.species_at(indices)
    floors = np.asarray(core_floor)[np.asarray(species)[:, None], nt]
    return np.all(distances >= floors, axis=1)


def find_neighbors_pair(
    cl: CellList | None,
    trials: FloatArray,
    members: FloatArray,
    lower_cutoff: float,
    upper_cutoff: float,
    no_neigh: int,
    keep_mask: BoolArray | None = None,
    valid: BoolArray | None = None,
    dists_needed: bool = True,
    trial_species: IntArray | None = None,
    member_species: IntArray | None = None,
    core_floor: FloatArray | None = None,
) -> tuple[RawNeighbors | None, RawNeighbors | None]:
    """`find_neighbors` for the trial cloud and the removed members in ONE query.

    An exchange asks the same question twice: neighbours of the trials, then
    neighbours of the members it is about to remove. Both bodies sit in the same
    cuboid, and a cell-list query costs far more in numpy dispatch than in
    arithmetic -- about 40 calls whatever the point count -- so asking once for
    the concatenation and splitting the rows is close to half the neighbour cost
    of an exchange step.

    The rows come back one column wider than either query alone would give (the
    width is the largest in-range count over BOTH bodies). Every live weigher is
    insensitive to that: padding is neutral in a product and masked out of a
    count. See `test_neighbors_parity.py::test_weights_are_identical_from_either_container`.

    `dists_needed=False` asks for the nearest neighbour only. The hard-core
    weight is `1` for every admissible trial and reads no distance at all, so on
    that path the full padded rows are computed and thrown away; asking for the
    minimum skips a scatter and a per-row sort. Anything that reads `dists` or
    `idxs` (LJ, aux, reweight) must leave it True.

    Returns (raw_trials, raw_members), each in `find_neighbors` form or None.
    """
    if cl is None or cl.n_active == 0:
        return (
            find_neighbors(
                None,
                trials,
                lower_cutoff,
                upper_cutoff,
                no_neigh,
                keep_mask=keep_mask,
                valid=valid,
                species=trial_species,
                core_floor=core_floor,
            ),
            find_neighbors(
                None,
                members,
                lower_cutoff,
                upper_cutoff,
                no_neigh,
                species=member_species,
                core_floor=core_floor,
            ),
        )
    nt = len(trials)
    both = np.vstack((trials, members)) if len(members) else trials
    distances, indices = cl.query(both, upper_cutoff, min_only=not dists_needed)
    # trials and members are queried together, so the floor test is applied to
    # the two blocks with their own type columns and then split.
    _tspc = trial_species
    _mspc = member_species
    if _tspc is not None and core_floor is not None:
        both_spc = (
            np.concatenate([_tspc, _mspc])
            if _mspc is not None and len(members)
            else _tspc
        )
    else:
        both_spc = None
    ok = _admissible(distances, indices, lower_cutoff, both_spc, core_floor, cl)

    good = ok[:nt]
    if valid is not None:
        good = good & np.asarray(valid, bool)
    if np.any(good):
        km = keep_mask[good] if keep_mask is not None else None
        spc = trial_species[good] if trial_species is not None else None
        raw_t = trials[good], distances[:nt][good], indices[:nt][good], km, spc
    else:
        raw_t = None

    if len(members) == 0:
        return raw_t, None
    gm = ok[nt:]
    if not np.any(gm):
        return raw_t, None
    m_spc = member_species[gm] if member_species is not None else None
    return raw_t, (members[gm], distances[nt:][gm], indices[nt:][gm], None, m_spc)


def find_neighbors(
    kdtree: Any,
    trial_mols: FloatArray,
    lower_cutoff: float,
    upper_cutoff: float,
    no_neigh: int,
    keep_mask: BoolArray | None = None,
    valid: BoolArray | None = None,
    saturation_requery: bool = False,
    species: IntArray | None = None,
    core_floor: FloatArray | None = None,
) -> RawNeighbors | None:
    """Query each trial's neighbours within `upper_cutoff`, drop any trial that
    overlaps the system (a neighbour closer than `lower_cutoff`), and return the
    surviving raw arrays. Weighting is the caller's job.

    Returns (coords, dists, idxs, keep_mask, species) or None if nothing
    survives; `keep_mask` and `species` are per-trial columns filtered by the
    same survivor mask as the coordinates (see `RawNeighbors`).
    `dists` and `idxs` are (N, k) with the kd-tree's inf / N padding kept,
    exactly as the weighting code downstream expects. Overlap detection is safe
    at any k (the k nearest already include any overlap), but an energy summed
    over the returned neighbour list is only complete when the list is not
    truncated. `saturation_requery=True` (used by the energy-bearing modes)
    doubles k until the query is unsaturated so no in-range neighbour is dropped
    from that sum; the hard-core path leaves it off, since its weight ignores the
    distances entirely. When the fixed-k query is already unsaturated the columns
    are identical either way, so the flag is a no-op below ordinary density.

    THE OVERLAP FLOOR IS PER PAIR WHEN `core_floor` IS GIVEN (A1.4). A trial is
    admissible when no neighbour sits closer than the floor for THAT pair of
    types, `core_floor[trial_type, neighbour_type]`, rather than closer than one
    scalar. Both arguments are needed together: the trial types come from
    `species` and the neighbour types from the container's own species column, so
    this is the only place both are in hand at once. Without the table the test
    is the scalar one, bit for bit -- and with a single type whose floor is that
    same scalar, the two agree by construction.

    `kdtree` may instead be a `neighbors.CellList`. It answers a radius query, so
    its neighbour list is complete by construction and `saturation_requery` has
    nothing to do; its rows are as wide as the largest in-range count rather than
    `no_neigh`, which every live weigher is insensitive to (padding is neutral in
    a product or a masked count). An empty cell list takes the same
    all-admissible path an absent tree does, so the two agree on an empty system.
    """
    if kdtree is not None and hasattr(kdtree, "n_active"):
        if kdtree.n_active == 0:
            kdtree = None  # empty system: the None path, exactly
        else:
            distances, indices = kdtree.query(trial_mols, upper_cutoff)
            good = _admissible(
                distances, indices, lower_cutoff, species, core_floor, kdtree
            )
            if valid is not None:
                good = good & np.asarray(valid, bool)
            if not np.any(good):
                return None
            km = keep_mask[good] if keep_mask is not None else None
            spc = species[good] if species is not None else None
            return trial_mols[good], distances[good], indices[good], km, spc
    if kdtree is None:
        # empty system: nothing to overlap and no neighbours. Every trial is
        # admissible with padded (infinite-distance) neighbour rows, so the
        # downstream member-vs-system weight is 1.
        n = len(trial_mols)
        good = np.ones(n, dtype=bool)
        if valid is not None:
            good = good & np.asarray(valid, bool)
        if not np.any(good):
            return None
        dists = np.full((int(good.sum()), no_neigh), np.inf)
        idxs = np.zeros((int(good.sum()), no_neigh), dtype=int)
        km = keep_mask[good] if keep_mask is not None else None
        spc = species[good] if species is not None else None
        return trial_mols[good], dists, idxs, km, spc
    if saturation_requery:
        distances, indices = query_saturating(
            kdtree, trial_mols, no_neigh, upper_cutoff
        )
    else:
        distances, indices = kdtree.query(
            trial_mols, k=no_neigh, distance_upper_bound=upper_cutoff
        )
    good = np.all(distances >= lower_cutoff, axis=1)
    if valid is not None:  # one-body pore wall: out-of-pore
        good = good & np.asarray(valid, bool)  # trials are rejected like overlaps
    if not np.any(good):
        return None

    coords = trial_mols[good]
    dists = distances[good]
    idxs = indices[good]
    km = keep_mask[good] if keep_mask is not None else None
    spc = species[good] if species is not None else None
    return coords, dists, idxs, km, spc


def indices_in_cavity_cl(
    cl: CellList,
    center: FloatArray,
    radius: float,
    side: float | FloatArray,
    two_dim: bool,
) -> IntArray:
    """`indices_in_cavity` restricted to the cells the cube cavity touches.

    Returns SLOT IDS, sorted ascending. The gather is a superset; the per-axis
    wrap-aware test below is the same one the full scan uses, so the cavity is
    exactly the cavity it always was. `side` may be a scalar (cubic) or a
    per-axis length array (orthorhombic); the wrap on each axis uses its own
    length.
    """
    center = np.asarray(center, float)[: cl.dim]
    sides = np.broadcast_to(np.asarray(side, float), (cl.dim,))
    cand = cl.gather_aabb(center - radius, center + radius, sort=False)
    if cand.size and cl.n_matrix:
        cand = cand[~cl.is_matrix[cand]]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64)
    coords = cl.coords[cand]
    mask = _axis_in_cavity(coords[:, 0], center[0], radius, float(sides[0]))
    mask &= _axis_in_cavity(coords[:, 1], center[1], radius, float(sides[1]))
    if not two_dim:
        mask &= _axis_in_cavity(coords[:, 2], center[2], radius, float(sides[2]))
    return np.sort(cand[mask])

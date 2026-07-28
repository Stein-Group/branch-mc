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

if TYPE_CHECKING:
    from .neighbors import CellList

# (trial coords, padded distances, padded neighbour ids, keep mask or None):
# the raw neighbour result every weighting mode consumes. dists/idxs are
# whatever the container returned (scipy gives float64/intp), so they are
# typed as the arrays they are used as, not re-cast at runtime.
RawNeighbors: TypeAlias = tuple[FloatArray, FloatArray, IntArray, "BoolArray | None"]

# The single overlap convention for the whole package. Two centres overlap iff
# their separation is at most `sigma - OVERLAP_TOL`; a pair at exactly sigma is a
# valid (non-overlapping) edge. Every overlap / adjacency test routes through the
# two helpers below so there is one implementation, not four literals that differ
# on a measure-zero set.
OVERLAP_TOL = 1e-12


def overlap_radius(sigma: float) -> float:
    """The largest separation that still counts as overlapping, i.e. the radius
    to hand a `query_ball_point` / `query_pairs` overlap search so that a pair at
    exactly sigma is excluded from the overlap set."""
    return sigma - OVERLAP_TOL


def overlaps(d: FloatArray | float, sigma: float) -> BoolArray:
    """Boolean overlap indicator for separations `d` against diameter `sigma`."""
    return np.asarray(d, dtype=float) <= sigma - OVERLAP_TOL


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
            ),
            find_neighbors(None, members, lower_cutoff, upper_cutoff, no_neigh),
        )
    nt = len(trials)
    both = np.vstack((trials, members)) if len(members) else trials
    distances, indices = cl.query(both, upper_cutoff, min_only=not dists_needed)
    ok = np.all(distances >= lower_cutoff, axis=1)

    good = ok[:nt]
    if valid is not None:
        good = good & np.asarray(valid, bool)
    if np.any(good):
        km = keep_mask[good] if keep_mask is not None else None
        raw_t = trials[good], distances[:nt][good], indices[:nt][good], km
    else:
        raw_t = None

    if len(members) == 0:
        return raw_t, None
    gm = ok[nt:]
    if not np.any(gm):
        return raw_t, None
    return raw_t, (members[gm], distances[nt:][gm], indices[nt:][gm], None)


def find_neighbors(
    kdtree: Any,
    trial_mols: FloatArray,
    lower_cutoff: float,
    upper_cutoff: float,
    no_neigh: int,
    keep_mask: BoolArray | None = None,
    valid: BoolArray | None = None,
    saturation_requery: bool = False,
) -> RawNeighbors | None:
    """Query each trial's neighbours within `upper_cutoff`, drop any trial that
    overlaps the system (a neighbour closer than `lower_cutoff`), and return the
    surviving raw arrays. Weighting is the caller's job.

    Returns (coords, dists, idxs, keep_mask) or None if nothing survives.
    `dists` and `idxs` are (N, k) with the kd-tree's inf / N padding kept,
    exactly as the weighting code downstream expects. Overlap detection is safe
    at any k (the k nearest already include any overlap), but an energy summed
    over the returned neighbour list is only complete when the list is not
    truncated. `saturation_requery=True` (used by the energy-bearing modes)
    doubles k until the query is unsaturated so no in-range neighbour is dropped
    from that sum; the hard-core path leaves it off, since its weight ignores the
    distances entirely. When the fixed-k query is already unsaturated the columns
    are identical either way, so the flag is a no-op below ordinary density.

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
            good = np.all(distances >= lower_cutoff, axis=1)
            if valid is not None:
                good = good & np.asarray(valid, bool)
            if not np.any(good):
                return None
            km = keep_mask[good] if keep_mask is not None else None
            return trial_mols[good], distances[good], indices[good], km
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
        return trial_mols[good], dists, idxs, km
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
    return coords, dists, idxs, km


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

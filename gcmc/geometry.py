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
import numpy as np

# The single overlap convention for the whole package. Two centres overlap iff
# their separation is at most `sigma - OVERLAP_TOL`; a pair at exactly sigma is a
# valid (non-overlapping) edge. Every overlap / adjacency test routes through the
# two helpers below so there is one implementation, not four literals that differ
# on a measure-zero set.
OVERLAP_TOL = 1e-12


def overlap_radius(sigma):
    """The largest separation that still counts as overlapping, i.e. the radius
    to hand a `query_ball_point` / `query_pairs` overlap search so that a pair at
    exactly sigma is excluded from the overlap set."""
    return sigma - OVERLAP_TOL


def overlaps(d, sigma):
    """Boolean overlap indicator for separations `d` against diameter `sigma`."""
    return np.asarray(d, dtype=float) <= sigma - OVERLAP_TOL


def query_saturating(kdtree, points, k, upper_cutoff):
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


def wrap(coords, side):
    return np.mod(coords, side)


def _axis_in_cavity(col, c, radius, side):
    lo = c - radius
    hi = c + radius
    if lo < 0:
        return ((col >= 0) & (col <= hi)) | ((col >= lo + side) & (col <= side))
    if hi > side:
        return ((col >= lo) & (col <= side)) | ((col >= 0) & (col <= hi - side))
    return (col >= lo) & (col <= hi)


def indices_in_cavity(coords, center, radius, side, two_dim):
    mask = _axis_in_cavity(coords[:, 0], center[0], radius, side)
    mask &= _axis_in_cavity(coords[:, 1], center[1], radius, side)
    if not two_dim:
        mask &= _axis_in_cavity(coords[:, 2], center[2], radius, side)
    return np.nonzero(mask)[0]


def find_neighbors(kdtree, trial_mols, lower_cutoff, upper_cutoff, no_neigh,
                   keep_mask=None, valid=None, saturation_requery=False):
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
    """
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
        distances, indices = query_saturating(kdtree, trial_mols, no_neigh,
                                              upper_cutoff)
    else:
        distances, indices = kdtree.query(trial_mols, k=no_neigh,
                                          distance_upper_bound=upper_cutoff)
    good = np.all(distances >= lower_cutoff, axis=1)
    if valid is not None:                 # one-body pore wall: out-of-pore
        good = good & np.asarray(valid, bool)   # trials are rejected like overlaps
    if not np.any(good):
        return None

    coords = trial_mols[good]
    dists = distances[good]
    idxs = indices[good]
    km = keep_mask[good] if keep_mask is not None else None
    return coords, dists, idxs, km

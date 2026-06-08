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
                   keep_mask=None):
    """Query each trial's neighbours within `upper_cutoff`, drop any trial that
    overlaps the system (a neighbour closer than `lower_cutoff`), and return the
    surviving raw arrays. Weighting is the caller's job.

    Returns (coords, dists, idxs, keep_mask) or None if nothing survives.
    `dists` and `idxs` are (N, no_neigh) with the kd-tree's inf / N padding kept,
    exactly as the weighting code downstream expects.
    """
    distances, indices = kdtree.query(trial_mols, k=no_neigh,
                                      distance_upper_bound=upper_cutoff)
    good = np.all(distances >= lower_cutoff, axis=1)
    if not np.any(good):
        return None

    coords = trial_mols[good]
    dists = distances[good]
    idxs = indices[good]
    km = keep_mask[good] if keep_mask is not None else None
    return coords, dists, idxs, km

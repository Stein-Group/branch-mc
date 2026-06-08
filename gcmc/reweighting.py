"""Order parameters for reweighting GCMC.

Reweighting runs the hard-core system but biases insertion by a local order
parameter, so the chain spends more time in configurations of a chosen
structural character and the free-energy difference can be reweighted out
afterward. In 2D the parameter is the hexatic psi6; the trial carries the raw
6-fold phasor sums (cos, sin) and the neighbour count, and the selection weight
and the per-molecule term are the phasor magnitude psi6 = |<exp(6 i theta)>|,
with an isotropic-liquid floor of 0.3 where a molecule has no neighbours.

calculate_psi_6_2d and calculate_q6_3d are ported verbatim from the original.
The 3D Steinhardt Q6 is included for completeness but is not wired into the
engine: in the original its Q6 weight never reaches the selection step (which
reads the default phasors), so the 3D path was never a coherent estimator. The
method is a 2D-hard-disk hexatic technique; the engine raises on 3D reweight.
"""

from __future__ import annotations
import numpy as np


def calculate_psi_6_2d(deltas2d, finite_mask, gamma):
    """6-fold bond-orientational phasor sums for each point. Returns
    (cos_sum, sin_sum, counts); the magnitude is formed later by psi6_magnitude.
    gamma is unused here, kept for signature parity with the original."""
    thetas = np.arctan2(deltas2d[..., 1], deltas2d[..., 0])
    cos_6 = np.cos(6 * thetas) * finite_mask
    sin_6 = np.sin(6 * thetas) * finite_mask
    cos_sum = np.sum(cos_6, axis=1)
    sin_sum = np.sum(sin_6, axis=1)
    counts = finite_mask.sum(axis=1)
    return cos_sum, sin_sum, counts


def psi6_magnitude(cos_sum, sin_sum, counts):
    """|<exp(6 i theta)>| per molecule, with the original's liquid floor: a zero
    or undefined value (no neighbours) is set to 0.3, the psi6 of a random
    arrangement."""
    with np.errstate(divide="ignore", invalid="ignore"):
        psi6 = np.sqrt((cos_sum / counts) ** 2 + (sin_sum / counts) ** 2)
    psi6 = np.asarray(psi6, float)
    psi6[psi6 == 0] = 0.3
    psi6[np.isnan(psi6)] = 0.3
    return psi6


def _sph_harm(m, l, phi, theta):
    """Y_l^m, tolerant of the scipy rename (sph_harm -> sph_harm_y with swapped
    argument order)."""
    from scipy import special
    if hasattr(special, "sph_harm"):
        return special.sph_harm(m, l, phi, theta)
    return special.sph_harm_y(l, m, theta, phi)


def calculate_q6_3d(deltas3d, finite_mask, gamma):
    """Local Steinhardt Q6 (dormant; the 3D reweight path is not wired)."""
    x, y, z = deltas3d[..., 0], deltas3d[..., 1], deltas3d[..., 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(np.clip(z / np.where(r > 0, r, 1.0), -1.0, 1.0))
    phi = np.arctan2(y, x)
    N = deltas3d.shape[0]
    q6m = np.zeros((N, 13), dtype=complex)
    counts = finite_mask.sum(axis=1)
    has = counts > 0
    for i, m in enumerate(range(-6, 7)):
        Y6m = _sph_harm(m, 6, phi, theta) * finite_mask
        s = np.sum(Y6m, axis=1)
        q6m[has, i] = s[has] / counts[has]
    Q6 = np.ones(N) * 0.3
    Q6[has] = np.sqrt(4 * np.pi / 13 * np.sum(np.abs(q6m) ** 2, axis=1)[has])
    return Q6 ** gamma


def calculate_order_parameter(coords, dists, idxs, kdtree, side_of_box, gamma):
    """Bridge: gather neighbour displacements under PBC and dispatch by
    dimensionality. 2D returns (cos_sum, sin_sum, counts); 3D returns Q6**gamma.
    """
    n_data = kdtree.data.shape[0]
    ndim = coords.shape[1]
    box = np.array([side_of_box] * ndim) if np.isscalar(side_of_box) else np.asarray(side_of_box)
    safe = np.minimum(idxs, n_data - 1)
    neigh = kdtree.data[safe]
    finite_mask = np.isfinite(dists)
    if ndim == 2:
        deltas = neigh[..., :2] - coords[:, None, :2]
        deltas -= box[:2] * np.round(deltas / box[:2])
        return calculate_psi_6_2d(deltas, finite_mask, gamma)
    if ndim == 3:
        deltas = neigh - coords[:, None, :]
        deltas -= box * np.round(deltas / box)
        return calculate_q6_3d(deltas, finite_mask, gamma)
    raise ValueError(f"unsupported dimensionality {ndim}")

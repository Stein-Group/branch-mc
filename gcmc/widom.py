"""Widom test-particle insertion into a frozen matrix (quenched-annealed Tier-0).

For hard spheres the insertion energy is 0 (no overlap) or +infinity (overlap),
so the Widom average collapses to the free-volume fraction:

    phi = < exp(-beta dU) > = fraction of random points not overlapping the matrix

and the excess chemical potential of the fluid at infinite dilution is

    beta * mu_excess_inf = -ln(phi).

This is the cleanest check that the matrix-exclusion insertion/deletion obeys
detailed balance: a correct GCMC sampler must reproduce rho_fluid / z -> phi as
the activity z = exp(beta*mu) -> 0. It also fixes the chemical-potential
convention: this package's `mu` is the TOTAL chemical potential (z = exp(beta*mu)),
so to compare against an excess-convention reference (e.g. Fernaud-Lomba-Lee),
use  beta*mu_excess = beta*mu_code - ln(rho_fluid),  which -> -ln(phi) as rho->0.
"""

import numpy as np
from scipy.spatial import cKDTree


def free_volume_fraction(matrix, box, diameter, n_test=400_000, seed=0):
    """phi = fraction of uniformly random points whose nearest matrix sphere is
    at least `diameter` away (i.e. a hard-sphere test particle fits there)."""
    matrix = np.asarray(matrix, float)
    dim = matrix.shape[1]
    if len(matrix) == 0:
        return 1.0
    tree = cKDTree(matrix % box, boxsize=[box] * dim)
    pts = np.random.default_rng(seed).uniform(0, box, (n_test, dim))
    return float(np.mean(tree.query(pts % box, k=1)[0] >= diameter))


def excess_mu_infinite_dilution(matrix, box, diameter, **kw):
    """beta * mu_excess at infinite dilution = -ln(phi)."""
    phi = free_volume_fraction(matrix, box, diameter, **kw)
    if phi <= 0.0:
        return np.inf
    return -np.log(phi)

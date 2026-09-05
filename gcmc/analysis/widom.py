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

`widom_henry` is the
same estimator with a soft host instead of a hard one: uniform test insertions
into a rigid framework, where an overlap contributes exp(-inf) = 0 and everything
else contributes its Boltzmann factor. It sat in the force-field module because
that is where the field it samples lives, which put an ESTIMATOR at L1 -- and an
estimator is the one thing the engine must never be able to reach. It belongs
here, beside the hard-sphere case it generalises.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .._types import FloatArray
from ..forcefield.lj_table import FrameworkField


def free_volume_fraction(
    matrix: FloatArray,
    box: float,
    diameter: float,
    n_test: int = 400_000,
    seed: int = 0,
) -> float:
    """phi = fraction of uniformly random points whose nearest matrix sphere is
    at least `diameter` away (i.e. a hard-sphere test particle fits there)."""
    matrix = np.asarray(matrix, float)
    dim = matrix.shape[1]
    if len(matrix) == 0:
        return 1.0
    tree = cKDTree(matrix % box, boxsize=[box] * dim)
    pts = np.random.default_rng(seed).uniform(0, box, (n_test, dim))
    return float(np.mean(tree.query(pts % box, k=1)[0] >= diameter))


def excess_mu_infinite_dilution(
    matrix: FloatArray, box: float, diameter: float, **kw: Any
) -> float:
    """beta * mu_excess at infinite dilution = -ln(phi)."""
    phi = free_volume_fraction(matrix, box, diameter, **kw)
    if phi <= 0.0:
        return np.inf
    return -np.log(phi)


def widom_henry(
    field: FrameworkField,
    guest_type: int,
    beta: float,
    rng: np.random.Generator,
    n_samples: int = 200_000,
) -> dict[str, float]:
    """A3.4: Widom test-particle insertion of one guest type into the framework.

    At infinite dilution the guest sees only the framework (no other guests), so
    the Widom average is a pure one-body integral over the accessible cell:

        <exp(-beta u)> = (1/V) integral exp(-beta u_gf(r)) dr

    estimated by uniform random insertion. A trial that overlaps a framework atom
    contributes exp(-inf) = 0 and is simply counted as zero, which is how the
    excluded volume enters. The dimensionless Henry constant in reduced units is

        K* = <exp(-beta u)>,

    the ratio of loading to activity as z* -> 0 (Henry's law). The caller turns
    K* into the paper's mol/kg/Pa with the framework mass and sigma_ref^3; that
    conversion is unit bookkeeping and lives in the calling script, not here, so this
    stays a pure reduced-unit measurement.

    Returns the Boltzmann average, its standard error, and the accessible
    fraction (trials that were not an overlap), which is a useful diagnostic:
    for a hot guest most of the average comes from the few deep-well insertions.
    """
    box = field.box
    dim = box.shape[0]
    acc = 0.0
    acc_sq = 0.0
    n_accessible = 0
    # chunk the sampling so a large n_samples does not allocate one giant array
    chunk = 20_000
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        pts = rng.uniform(0.0, 1.0, size=(m, dim)) * box
        # batched: overlaps contribute 0; the rest get exp(-beta u) from the
        # vectorised framework energy. Points are drawn exactly as the per-point
        # loop did, so this agrees with it to float noise.
        ov = field.overlaps_many(pts, guest_type)
        free = pts[~ov]
        if len(free):
            u = field.energy_many(free, guest_type)
            w = np.where(beta * u < 700, np.exp(-beta * u), 0.0)
            acc += float(w.sum())
            acc_sq += float((w * w).sum())
            n_accessible += int(len(free))
        done += m
    mean = acc / n_samples
    var = max(acc_sq / n_samples - mean * mean, 0.0)
    return {
        "henry_star": mean,
        "stderr": math.sqrt(var / n_samples),
        "accessible_fraction": n_accessible / n_samples,
    }

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


def widom_bulk(
    positions: FloatArray,
    box: float | FloatArray,
    beta: float,
    rc: float,
    *,
    n_test: int = 20_000,
    lower: float = 0.0,
    tail_pref: float | None = None,
    seed: int = 0,
) -> float:
    """beta * mu_excess of a BULK fluid by test-particle insertion.

    beta mu_ex = -ln < exp(-beta dU) > over uniformly placed test particles,
    with dU the truncated pair energy against the configuration (plus the
    mean-field tail's insertion increment when one is configured). The bulk case
    of the same estimator as `free_volume_fraction`, which is the hard-sphere
    limit of this: there dU is 0 or +inf and the average collapses to the free
    volume.

    WHY THIS IS THE CHECK WORTH HAVING. It closes the chemical potential against
    itself:

        beta mu  ==  ln rho  +  beta mu_ex

    in this package's reduced convention (Lambda = 1, so z = exp(beta mu) and
    the ideal-gas term is ln rho). It needs NO literature value and no second
    sampler, and it catches weight errors that a single-point comparison against
    an equation of state cannot -- a wrong activity, a missing volume factor or
    a double-counted tail all move <N> to a self-consistent-looking place, and
    all three break this identity.

    THE TAIL INCREMENT is `tail_pref * (2N + 1)`, exactly the term the insertion
    acceptance uses, because a test particle IS an insertion. Omitting it while
    the chain includes it is a systematic offset of about that size in
    beta mu_ex, which at rho* ~ 0.6 is not small.

    `lower` is the hard-core floor: a test point inside it contributes exactly
    zero rather than a huge Boltzmann factor, which is what the engine's own
    overlap test does and what keeps the two conventions the same.
    """
    P = np.asarray(positions, float)
    n, dim = P.shape if P.ndim == 2 else (0, len(np.atleast_1d(box)))
    b = np.broadcast_to(np.asarray(box, float), (dim,))
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n_test, dim)) * b
    if n == 0:
        return 0.0
    tree = cKDTree(P % b, boxsize=b)
    acc = np.zeros(n_test)
    nb = tree.query_ball_point(pts % b, rc)
    from ..forcefield.lennard_jones import lj_pair_energy_sq

    tail = 0.0
    if tail_pref is not None:
        tail = float(tail_pref) * (2.0 * n + 1.0)
    for i, js in enumerate(nb):
        if not js:
            acc[i] = math.exp(-beta * tail)
            continue
        d = P[js] - pts[i]
        d -= b * np.round(d / b)
        d2 = np.einsum("ij,ij->i", d, d)
        if lower > 0.0 and bool((d2 <= (lower - 1e-12) ** 2).any()):
            acc[i] = 0.0
            continue
        u = float(np.sum(lj_pair_energy_sq(d2))) + tail
        acc[i] = math.exp(-beta * u) if -beta * u < 700.0 else math.inf
    mean = float(acc.mean())
    if mean <= 0.0:
        return math.inf
    return -math.log(mean)

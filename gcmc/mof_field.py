"""A3.2/A3.3: a guest's energy against fluid AND framework, by type, in one place.

The frozen engine's `_lj_energy` sums guest-guest pairs from a distance list and
knows nothing about type or framework: quenched-annealed obstacles are dropped
from that sum (they exclude by overlap only). The MOF inverts that -- the
framework is the whole point, and every guest-framework pair carries a real,
type-dependent energy.

This module is the energy core the conventional MOF metropolis path (A3.5) will
call. It is deliberately small and standalone so it can be gated against
ref_energy.py (the V0-confirmed oracle) BEFORE it is wired into the hot loop:
that is A3.5's "no wiring before Gate 1" made concrete.

Everything is in the engine's reduced units: positions over sigma_ref, energies
over eps_ref, box per axis. The single scale that connects these to ref_energy.py
(which works in angstrom/kelvin) is applied by the caller, and the gate test
checks that the two agree once it is.

Overlap uses the per-pair floor (A3.3): a guest cannot sit inside a framework
atom, and a close approach that RASPA would admit is admissible here too, because
the floor is the same c*sigma_ij the P0.6 measurement fixed.
"""

from __future__ import annotations

import math

import numpy as np

from ._types import FloatArray, IntArray
from .potentials.lj_table import LennardJonesTable
from .potentials.pair_table import PairTable


def _mic_dists(point: FloatArray, others: FloatArray, box: FloatArray) -> FloatArray:
    """Minimum-image distances from one point to many, per axis. Reduced units."""
    if len(others) == 0:
        return np.empty(0)
    dr = np.asarray(point, float)[None, :] - np.asarray(others, float)
    box = np.asarray(box, float)
    dr -= box * np.round(dr / box)
    return np.sqrt(np.einsum("ij,ij->i", dr, dr))


class FrameworkField:
    """The static framework, ready to answer "what does a guest feel here?".

    Holds the framework positions and their PairTable type ids once. Every query
    is a guest of a given type at a given point; the answer is the truncated LJ
    energy against the framework, by pair type, in reduced units.

    Guest-guest energy is handled by the same `LennardJonesTable.total_energy`
    the conventional path already uses; this class is only the host half, kept
    separate so the framework positions live in one place and are never confused
    with the mutable fluid.
    """

    def __init__(
        self,
        frame_pos: FloatArray,
        frame_types: IntArray,
        table: PairTable,
        box: FloatArray,
        upper_cutoff: float,
    ) -> None:
        self.pos = np.ascontiguousarray(frame_pos, dtype=float)
        self.types = np.ascontiguousarray(frame_types, dtype=np.int64)
        self.table = table
        self.box = np.asarray(box, float)
        self.rc = float(upper_cutoff)
        if len(self.pos) != len(self.types):
            raise ValueError("framework positions and type ids differ in length")

    def energy(self, point: FloatArray, guest_type: int) -> float:
        """Guest-framework energy of one guest at `point`, reduced units.

        Truncated at rc, evaluated per pair type. A guest exactly on a framework
        atom (distance 0) would be an overlap the caller must have rejected first;
        here a 0 distance is simply skipped, as in ref_energy.py.
        """
        d = _mic_dists(point, self.pos, self.box)
        inside = (d < self.rc) & (d > 0.0)
        if not inside.any():
            return 0.0
        pot = LennardJonesTable(
            beta=1.0, table=self.table, trial_type=int(guest_type), upper_cutoff=self.rc
        )
        return pot.total_energy(d[inside], neighbour_types=self.types[inside])

    def overlaps(self, point: FloatArray, guest_type: int) -> bool:
        """Does a guest of this type at `point` sit inside any framework atom?

        Per-pair floor (A3.3): the offending separation is c*sigma_ij, one value
        per framework type. Uses the same strictly-below convention as
        geometry.overlaps, so a guest exactly at the floor is admissible.
        """
        d = _mic_dists(point, self.pos, self.box)
        if len(d) == 0:
            return False
        floor = self.table.core_floor[int(guest_type)][self.types]
        # geometry.OVERLAP_TOL = 1e-12; excludes strictly below the floor only
        return bool((d <= floor - 1e-12).any())


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
    conversion is unit bookkeeping and lives in the V1 script, not here, so this
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
        for p in pts:
            if field.overlaps(p, guest_type):
                # exp(-beta * +inf) = 0; contributes nothing but counts in V
                continue
            u = field.energy(p, guest_type)
            w = math.exp(-beta * u) if beta * u < 700 else 0.0
            acc += w
            acc_sq += w * w
            n_accessible += 1
        done += m
    mean = acc / n_samples
    var = max(acc_sq / n_samples - mean * mean, 0.0)
    return {
        "henry_star": mean,
        "stderr": math.sqrt(var / n_samples),
        "accessible_fraction": n_accessible / n_samples,
    }

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
from .potentials import overlap
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
        # One reduced-units potential per table type, built once. energy() used to
        # construct a fresh LennardJonesTable on every call (millions per run); the
        # object is a few references, so caching it is a pure hot-path win with no
        # behaviour change (beta is folded in by the caller, so beta=1.0 here).
        self._pots = {
            t: LennardJonesTable(
                beta=1.0, table=table, trial_type=t, upper_cutoff=self.rc
            )
            for t in range(len(table.types))
        }

    def energy_many(
        self, points: FloatArray, guest_type: int, chunk: int = 4096
    ) -> FloatArray:
        """Guest-framework energy of many guests of one type, reduced units, as a
        vectorised batch (chunked to bound the P x N_frame distance matrix). Same
        result as calling `energy` point by point; this is what makes batched
        Widom (Q5) and the grid build fast."""
        pts = np.ascontiguousarray(points, float)
        pot = self._pots[int(guest_type)]
        out = np.zeros(len(pts))
        box = self.box
        for c0 in range(0, len(pts), chunk):
            cp = pts[c0 : c0 + chunk]
            dr = cp[:, None, :] - self.pos[None, :, :]
            dr -= box * np.round(dr / box)
            d = np.sqrt(np.einsum("cij,cij->ci", dr, dr))
            inside = (d < self.rc) & (d > 0.0)
            rows, cols = np.nonzero(inside)
            if rows.size:
                uu = pot._pair_u(d[rows, cols], self.types[cols])
                local = np.zeros(len(cp))
                np.add.at(local, rows, uu)
                out[c0 : c0 + len(cp)] = local
        return out

    def overlaps_many(
        self, points: FloatArray, guest_type: int, chunk: int = 4096
    ) -> np.ndarray:
        """Per-point framework overlap of many guests of one type (batched). A
        point overlaps if any framework atom is within the per-pair floor
        c*sigma. Same decision as `overlaps`, vectorised for Widom (Q5)."""
        pts = np.ascontiguousarray(points, float)
        floor = self.table.core_floor
        gt = int(guest_type)
        out = np.zeros(len(pts), dtype=bool)
        box = self.box
        for c0 in range(0, len(pts), chunk):
            cp = pts[c0 : c0 + chunk]
            dr = cp[:, None, :] - self.pos[None, :, :]
            dr -= box * np.round(dr / box)
            d2 = np.einsum("cij,cij->ci", dr, dr)
            # squared distances, so this needs the THRESHOLD rather than the
            # verdict; geometry.overlap_radius is that threshold, applied per
            # framework type. Same number pair_overlaps compares against.
            thr = overlap.overlap_radius(floor[gt, self.types])
            out[c0 : c0 + len(cp)] = (d2 <= (thr * thr)[None, :]).any(axis=1)
        return out

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
        return self._pots[int(guest_type)].total_energy(
            d[inside], neighbour_types=self.types[inside]
        )

    def overlaps(self, point: FloatArray, guest_type: int) -> bool:
        """Does a guest of this type at `point` sit inside any framework atom?

        Per-pair floor (A3.3): the offending separation is c*sigma_ij, one value
        per framework type. Uses the same strictly-below convention as
        geometry.overlaps, so a guest exactly at the floor is admissible.
        """
        d = _mic_dists(point, self.pos, self.box)
        if len(d) == 0:
            return False
        # one comparison, in geometry: excludes strictly below the floor only
        return bool(
            overlap.pair_overlaps(
                d, int(guest_type), self.types, self.table.core_floor
            ).any()
        )


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
        # batched (Q5): overlaps contribute 0; the rest get exp(-beta u) from the
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

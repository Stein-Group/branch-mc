"""A1.5: the conventional LJ path made type-aware, reducing to the frozen path.

The frozen `LennardJones` (registered "lj") answers the sampler's two questions
from scalar distances alone: one world, sigma = eps = 1. The MOF needs the same
two answers with per-pair sigma and eps, which means the potential has to know
the TYPE of the trial and the types of its neighbours. This class carries a
`PairTable` and the neighbour type ids, and computes the same Boltzmann factors
per pair type.

The design constraint from A1.2 carries through here: with a single guest type at
sigma* = eps* = 1, every method must return exactly what `LennardJones` returns,
so a single-species Xe run reuses the frozen "lj" trajectories bit for bit. That
is asserted in test_lj_table_potential.py against the real `LennardJones` object,
not merely against a formula.

Scope fence: this is the CONVENTIONAL move path only. The sweep
(`ljbranch`) is untouched this phase; its AST-locked kernels
(`count_and_sample`, `ljbranch_weight_sum`) stay exactly as they are, and the
table reaches them only in a later phase behind their own gate.

Tail correction: there is none. In a pore the mean-field tail is meaningless, so
`weight_bulk` refuses the `n_particles`/`choice` tail arguments rather than
silently dropping a term the caller thought was being applied.

THE FRAMEWORK HALF.
`LennardJonesTable` answers "what does this guest feel from the other guests?".
`FrameworkField` answers the same question about the STATIC host, out of the same
`PairTable` and at the same per-pair floor, and it does so by holding one
`LennardJonesTable` per type -- it was always a consumer of this class rather
than a peer of it, and the separate file made that look like a layering rather
than a dependency.

The ladder said this half merges into `pair_table`. It cannot: `lj_table` imports
`pair_table`, so a `FrameworkField` there would be an import cycle. It lands here
instead, one layer up from the tables it reads and still inside L1, which is what
the merge was actually for. The Widom estimator that shipped alongside it is not
force field at all and went to `widom` (L3) with the other insertion estimator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._types import FloatArray, IntArray
from . import overlap
from .base import Potential, register
from .pair_table import PairTable, table_pair_energy, table_pair_energy_vec


@register("lj_table")
class LennardJonesTable(Potential):
    """Truncated 12-6 LJ with per-pair sigma and eps from a PairTable.

    `trial_type` is the type id of the particle being inserted or moved; its
    neighbours' type ids are supplied per call (they are known to the caller from
    the same array the neighbour search indexes). With one type and the unit
    table, this is `LennardJones` exactly.
    """

    def __init__(
        self,
        beta: float,
        table: PairTable,
        trial_type: int = 0,
        upper_cutoff: float = 2.5,
    ) -> None:
        self.beta = float(beta)
        self.table = table
        self.trial_type = int(trial_type)
        self.upper_cutoff = float(upper_cutoff)
        # the per-pair floor for THIS trial type against every type, so the
        # overlap test is one array lookup in the hot path
        self.diameter = float(table.core_floor[self.trial_type, self.trial_type])

    @classmethod
    def build(  # type: ignore[override]
        cls,
        beta: float,
        table: PairTable,
        trial_type: int = 0,
        upper_cutoff: float = 2.5,
        **_ignored: Any,
    ) -> LennardJonesTable:
        """Registry factory. `**_ignored` absorbs the constants other potentials
        take (volume for the bulk tail, the g3 table) so `make_potential` can
        forward one keyword set to any registered mode; this one has no tail, by
        design (see the module docstring)."""
        return cls(
            beta=beta, table=table, trial_type=trial_type, upper_cutoff=upper_cutoff
        )

    def _pair_u(self, r: FloatArray, neighbour_types: IntArray | None) -> FloatArray:
        """Per-pair energy of the trial against neighbours at distances `r`.

        `neighbour_types` is one type id per distance; None means "all the same
        type as the trial", which is the single-species reduction and lets a
        caller that has no type array (the frozen path) get the frozen answer.
        """
        r = np.asarray(r, float)
        if neighbour_types is None:
            return table_pair_energy(r, self.trial_type, self.trial_type, self.table)
        nt = np.asarray(neighbour_types, int)
        out = np.empty_like(r)
        # group by neighbour type so the table lookup is per unique type, not per
        # pair; the number of types is tiny (a handful), so this is a short loop
        for t in np.unique(nt):
            m = nt == t
            out[m] = table_pair_energy(r[m], self.trial_type, int(t), self.table)
        return out

    def pair_u_grid(
        self,
        r: FloatArray,
        trial_types: IntArray,
        neighbour_types: IntArray,
    ) -> FloatArray:
        """Per-pair energy for a whole trial CLOUD of mixed types.

        `r` is (N, k), `trial_types` is (N,) and `neighbour_types` is (N, k).
        Returns (N, k).

        WHY THIS EXISTS ALONGSIDE `_pair_u`. That method takes one
        `self.trial_type` for every row, which is right for the MOF path -- one
        guest is inserted at a time. The exchange move in `metropolis_dynamics.advance`
        weights an entire trial cloud in one call, and after P3 each trial in
        that cloud carries its OWN drawn species. A per-instance `trial_type`
        cannot express that, and building one potential object per type would
        mean N table lookups and a Python loop over trials in the hot path.

        One gather over the whole block, not a loop over type combinations:
        `table_pair_energy_vec` fancy-indexes the pair tables, so the pair
        parameters come out as arrays and the arithmetic is identical to the
        scalar form. That matters more than it looks -- the same grouping loop in
        `branch._intra_lj_factor` runs once per CLIQUE, and it cost 53 s on one
        matrix point before being replaced by this gather.
        """
        r = np.asarray(r, float)
        tt = np.asarray(trial_types, int)
        nt = np.asarray(neighbour_types, int)
        # ONE gather over the whole (N, k) block: `sigma[si, sj]` fancy-indexes
        # the pair tables, so this is the same arithmetic `table_pair_energy` does
        # with no loop over type combinations. The per-row trial type broadcasts
        # across the k columns.
        return table_pair_energy_vec(
            r, np.broadcast_to(tt[:, None], r.shape), nt, self.table
        )

    def weight_cloud(
        self,
        dists: FloatArray,
        trial_types: IntArray,
        neighbour_types: IntArray,
    ) -> FloatArray:
        """Per-trial Boltzmann factor for a cloud of mixed types.

        The row-product analogue of `weight_pair` on a (N, k) input, with each
        row weighted by its own trial type. Pads (inf distance) give u = 0 and a
        factor of 1, exactly as in `weight_pair`, so a saturating query's padding
        stays neutral.
        """
        d = np.asarray(dists, float)
        g = np.ones_like(d)
        inside = d < self.upper_cutoff
        if np.any(inside):
            u = self.pair_u_grid(d, trial_types, neighbour_types)
            with np.errstate(over="ignore"):
                g[inside] = np.exp(-self.beta * u[inside])
        return np.prod(g, axis=1) if g.ndim == 2 else g

    def total_energy_typed(
        self,
        dists: FloatArray,
        trial_type: int,
        neighbour_types: IntArray,
    ) -> float:
        """`total_energy` for one particle of a stated type.

        The instance's own `trial_type` is deliberately NOT used: the engine's
        translate and swap moves ask about a particle already in the box, whose
        type comes from the cell list rather than from how this object was built.
        Passing it explicitly is what stops those two paths silently reading the
        type of whatever was inserted last.
        """
        d = np.asarray(dists, float)
        keep = (d < self.upper_cutoff) & (d > 0)
        if not keep.any():
            return 0.0
        nt = np.asarray(neighbour_types, int)[keep]
        tt = np.full(nt.shape, int(trial_type), dtype=int)
        return float(np.sum(self.pair_u_grid(d[keep][None, :], tt[:1], nt[None, :])))

    def weight_pair(
        self,
        r: FloatArray | float,
        gamma: float = 1.0,
        *,
        neighbour_types: IntArray | None = None,
        **_ignored: Any,
    ) -> FloatArray:
        """Boltzmann factor exp(-beta u) per pair, 1 beyond rc; a 2-D input is
        reduced by row product, exactly as `LennardJones.weight_pair`.

        The inf pads that a saturating neighbour query leaves in a row give
        u = 0 and a factor of 1, so padding stays neutral. Overflow of exp on a
        deep overlap is left to produce inf and a downstream 0 weight, matching
        the frozen path's deliberate behaviour.
        """
        r = np.asarray(r, dtype=float)
        rc = self.upper_cutoff
        g_r = np.ones_like(r)
        inside = r < rc
        if np.any(inside):
            nt = (
                None if neighbour_types is None else np.asarray(neighbour_types)[inside]
            )
            u = self._pair_u(r[inside], nt)
            with np.errstate(over="ignore"):
                g_r[inside] = np.exp(-self.beta * u)
        return np.prod(g_r, axis=1) if g_r.ndim == 2 else g_r

    def weight_bulk(
        self,
        dists: FloatArray,
        *,
        neighbour_types: IntArray | None = None,
        n_particles: int | None = None,
        choice: str | None = None,
        gamma: float = 1.0,
        **_ignored: Any,
    ) -> FloatArray:
        """Per-trial pair Boltzmann factors. No tail term exists in a pore, so a
        caller that passes the tail arguments is asking for something this mode
        cannot honour, and gets an error rather than a silently missing term."""
        if n_particles is not None or choice is not None:
            raise ValueError(
                "lj_table has no long-range tail (pore convention is no-tail); "
                "do not pass n_particles/choice to weight_bulk"
            )
        return self.weight_pair(dists, neighbour_types=neighbour_types)

    def total_energy(
        self, dists: FloatArray, neighbour_types: IntArray | None = None
    ) -> float:
        """Total truncated-LJ energy of the trial against a distance list, the
        table analogue of the engine's `_lj_energy`: sum of u(r) over 0 < r < rc,
        self-pairs (r = 0) and beyond-cutoff pairs dropping out.

        This is the function the conventional delta-U paths call; at one type it
        returns exactly what `_lj_energy` returns.
        """
        d = np.asarray(dists, float)
        keep = (d < self.upper_cutoff) & (d > 0)
        if not keep.any():
            return 0.0
        nt = None if neighbour_types is None else np.asarray(neighbour_types)[keep]
        return float(np.sum(self._pair_u(d[keep], nt)))


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
        Widom and the grid build fast."""
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
        c*sigma. Same decision as `overlaps`, vectorised for Widom."""
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

        Per-pair floor: the offending separation is c*sigma_ij, one value
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

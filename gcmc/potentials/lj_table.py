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

Scope fence (A1.5): this is the CONVENTIONAL move path only. The sweep
(`ljbranch`) is untouched this phase; its AST-locked kernels
(`count_and_sample`, `ljbranch_weight_sum`) stay exactly as they are, and the
table reaches them only in a later phase behind their own gate.

Tail correction: there is none. In a pore the mean-field tail is meaningless, so
`weight_bulk` refuses the `n_particles`/`choice` tail arguments rather than
silently dropping a term the caller thought was being applied.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._types import FloatArray, IntArray
from .base import Potential, register
from .pair_table import PairTable, table_pair_energy


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

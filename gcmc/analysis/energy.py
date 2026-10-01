"""Total potential energy of one configuration, recomputed from scratch.

The engines never carry U: every acceptance needs only a difference, so a
running total would be bookkeeping the hot loop pays for and never reads. The
log wants the number anyway, so it is rebuilt here once per logged row from
the materialised positions, off the clock (`wall_s` does not include it).

What goes into it is exactly what the engine's energy differences contain,
no more:

    U = sum_{fluid pairs, r < rc} u(r)
      + sum_{fluid-matrix pairs, r < rc} u(r)
      + tail_pref * N^2

`u` is the truncated 12-6 shape from `lennard_jones` (untyped) or the
per-pair table (typed); the tail is the mean-field term whose increments the
exchange acceptance applies as `beta * tail * (n_new^2 - n_old^2)`. Matrix
particles are type 0, as the engine's cell list builds them.

Units are reduced (epsilon of the reference pair). Cost is O(N * neighbours)
through one throwaway cell list, which is noise next to a window of moves but
is not free on a very large box logged every cycle.
"""

from __future__ import annotations

import numpy as np

from .._types import FloatArray, IntArray
from ..forcefield.lennard_jones import lj_pair_energy
from ..forcefield.lj_table import LennardJonesTable
from ..space import neighbors


def configuration_energy(
    positions: FloatArray,
    box: float | FloatArray,
    dim: int,
    upper_cutoff: float,
    *,
    species: IntArray | None = None,
    matrix: FloatArray | None = None,
    table: LennardJonesTable | None = None,
    tail_pref: float | None = None,
) -> float:
    """Reduced total energy U of the fluid configuration `positions`.

    `table` switches to the typed pair energy and then `species` must be the
    fluid's type column; without it every pair uses the single-species LJ.
    `matrix` is the frozen obstacle set, counted once per fluid-matrix pair.
    `tail_pref` of None or 0 means no tail.
    """
    pos = np.asarray(positions, float).reshape(-1, dim)
    n = len(pos)
    tail = 0.0 if not tail_pref else float(tail_pref) * n * n
    if n == 0:
        return tail
    has_matrix = matrix is not None and len(matrix) > 0
    cl = neighbors.CellList.from_positions(
        pos,
        box,
        dim,
        upper_cutoff,
        matrix=matrix if has_matrix else None,
        species=species,
    )
    fluid = [int(s) for s in cl.active_slots() if not cl.is_matrix[s]]
    u_ff = 0.0  # every fluid pair is seen from both ends; halved below
    u_fm = 0.0  # a matrix site is never a query centre, so seen once
    for s in fluid:
        d, nb = cl.query_point(cl.coords[s], upper_cutoff, sort=False, inclusive=False)
        other = nb != s
        d, nb = d[other], nb[other]
        if d.size == 0:
            continue
        mat = cl.matrix_at(nb)
        if table is None:
            e = lj_pair_energy(d)
            u_ff += float(np.sum(e[~mat]))
            u_fm += float(np.sum(e[mat]))
        else:
            ti = int(cl.species[s])
            if (~mat).any():
                u_ff += table.total_energy_typed(d[~mat], ti, cl.species_at(nb[~mat]))
            if mat.any():
                u_fm += table.total_energy_typed(d[mat], ti, cl.species_at(nb[mat]))
    return 0.5 * u_ff + u_fm + tail

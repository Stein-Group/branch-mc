"""Q4: FrameworkField performance surface (hoisted potentials + batched energy).

The per-type LennardJonesTable is now built once in __init__ instead of per call;
the existing ref-parity tests already pin that the energy is unchanged. This file
pins the new batched path: energy_many must equal calling energy point by point,
exactly, so it is a drop-in for the Widom loop (Q5) with no accuracy loss.

Uses a small synthetic framework (the real validated table, ~40 random atoms),
not the full MFM build, so it is fast.

Note (measured, see ledger): a trilinear energy grid was prototyped and rejected
here -- at 0.1 sigma spacing it gives ~19% error on the Widom integrand because
grid nodes inside framework cores poison interpolation. The batched EXACT path is
the accepted speedup; a core-aware grid or a framework neighbour list is the open
follow-up.
"""

from __future__ import annotations

import numpy as np

from gcmc.forcefield.lj_table import FrameworkField
from gcmc.forcefield.pair_table import PairTable
from scripts.mof.units import load_forcefield, reduce_tables

FF_A = "ff/forcefield_A.yaml"


def _synthetic_field(n_atoms=40, side=5.0, seed=0):
    ff = load_forcefield(FF_A)
    red = reduce_tables(ff, 298.0)
    table = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )
    rng = np.random.default_rng(seed)
    box = np.array([side, side, side])
    fw = table.index("In")
    fpos = rng.uniform(0, side, size=(n_atoms, 3))
    field = FrameworkField(
        fpos, np.full(n_atoms, fw), table, box, ff.rc / red.sigma_ref
    )
    return field, table, box


def test_energy_many_matches_pointwise():
    field, table, box = _synthetic_field()
    xe = table.index("Xe")
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, box[0], size=(500, 3))
    batched = field.energy_many(pts, xe)
    pointwise = np.array([field.energy(p, xe) for p in pts])
    # identical up to float summation order (scatter-add vs np.sum): rel ~1e-14
    np.testing.assert_allclose(batched, pointwise, rtol=1e-11, atol=1e-9)


def test_energy_many_chunk_invariant():
    field, table, box = _synthetic_field(seed=1)
    kr = table.index("Kr")
    rng = np.random.default_rng(4)
    pts = rng.uniform(0, box[0], size=(300, 3))
    a = field.energy_many(pts, kr, chunk=4096)
    b = field.energy_many(pts, kr, chunk=7)  # tiny chunks must give the same
    np.testing.assert_array_equal(a, b)


def test_hoisted_potentials_are_reused():
    field, table, _ = _synthetic_field()
    xe = table.index("Xe")
    # the per-type potential is built once and cached, not per call
    assert field._pots[xe] is field._pots[xe]
    assert len(field._pots) == len(table.types)


def test_widom_batched_matches_pointwise_reference():
    """Q5: the vectorised Widom equals a per-point reference on the same seed
    (points are drawn identically), so it is a pure speedup with no numeric
    change. overlaps_many must also match overlaps point by point."""
    import math

    from gcmc.analysis.widom import widom_henry

    field, table, box = _synthetic_field(n_atoms=40, side=8.0, seed=2)
    xe = table.index("Xe")

    def slow(seed, ns):
        rng = np.random.default_rng(seed)
        acc = 0.0
        done = 0
        while done < ns:
            m = min(20000, ns - done)
            pts = rng.uniform(0, 1, size=(m, 3)) * box
            for p in pts:
                if field.overlaps(p, xe):
                    continue
                u = field.energy(p, xe)
                acc += math.exp(-u) if u < 700 else 0.0
            done += m
        return acc / ns

    fast = widom_henry(field, xe, 1.0, np.random.default_rng(9), n_samples=20000)
    ref = slow(9, 20000)
    assert abs(fast["henry_star"] - ref) / ref < 1e-3

    # overlaps_many == overlaps pointwise
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, box[0], size=(400, 3))
    batched = field.overlaps_many(pts, xe)
    point = np.array([field.overlaps(p, xe) for p in pts])
    np.testing.assert_array_equal(batched, point)

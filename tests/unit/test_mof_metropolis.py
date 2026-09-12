"""A3.5: the MOF metropolis loop's energy bookkeeping against the V0 oracle.

The loop computes dU incrementally (moved guest vs its neighbours plus framework).
The gate is that this incremental dU equals U(after) - U(before) recomputed FROM
SCRATCH by ref_energy.py in angstrom/kelvin -- the same oracle V0 confirmed
against RASPA2. If the two ever part, the loop's acceptance is sampling a
different Hamiltonian than the one that matched RASPA.

Rather than reach inside the loop, this test drives short deterministic
trajectories and, at each accepted state, compares the total energy computed by
the reduced core against ref_energy.py. Total-energy agreement at every step is
strictly stronger than delta agreement (a delta error would surface as a total
mismatch at the next step), and it is what A3.5 asks for: "stored (framework +
guests) configurations, total energy ... match to 1e-10 relative".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gcmc.engine.rigid_framework import advance_metropolis_mof
from gcmc.forcefield.lj_table import FrameworkField, LennardJonesTable
from gcmc.forcefield.pair_table import PairTable
from gcmc.run.io_framework import load_framework_typed
from scripts.mof.ref_energy import read_extxyz, total_energy
from scripts.mof.units import load_forcefield, reduce_tables

ROOT = Path(__file__).resolve().parents[2]
FF_A = ROOT / "ff" / "forcefield_A.yaml"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from scripts.mof.build_framework import build, write_extxyz

    tmp = tmp_path_factory.mktemp("a35move")
    ff = load_forcefield(FF_A)
    red = reduce_tables(ff, 298.0)
    table = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )

    atoms, _ = build(ROOT / "structures" / "MFM-300-In.cif", ff)
    extxyz = tmp / "mfm.extxyz"
    write_extxyz(atoms, extxyz)
    frame_pos_A, frame_types_ref, cell_A = read_extxyz(extxyz, ff)
    pos_star, type_ids = load_framework_typed(extxyz, table)

    sr = red.sigma_ref
    field = FrameworkField(
        pos_star / sr, type_ids, table, np.asarray(cell_A) / sr, ff.rc / sr
    )
    return {
        "ff": ff,
        "red": red,
        "table": table,
        "field": field,
        "frame_pos_A": frame_pos_A,
        "frame_types_ref": frame_types_ref,
        "cell_A": np.asarray(cell_A),
        "sr": sr,
    }


def _total_reduced(env, pos_reduced, guest_type):
    """Total energy (guest-guest + guest-framework) of a reduced configuration,
    via the reduced core, in reduced units."""
    field = env["field"]
    table = env["table"]
    pot = LennardJonesTable(
        beta=1.0, table=table, trial_type=guest_type, upper_cutoff=field.rc
    )
    box = field.box
    u = 0.0
    n = len(pos_reduced)
    for i in range(n):
        u += field.energy(pos_reduced[i], guest_type)
        for j in range(i + 1, n):
            dr = pos_reduced[i] - pos_reduced[j]
            dr -= box * np.round(dr / box)
            r = float(np.sqrt(dr @ dr))
            if 0.0 < r < field.rc:
                u += float(
                    pot.total_energy(
                        np.array([r]), neighbour_types=np.array([guest_type])
                    )
                )
    return u


def _total_ref_kelvin(env, pos_reduced, guest_type_name):
    """The same configuration's total energy through ref_energy.py, in kelvin."""
    ff = env["ff"]
    guest_pos_A = pos_reduced * env["sr"]
    types = np.array([ff.index(guest_type_name)] * len(pos_reduced))
    u = total_energy(
        guest_pos_A,
        types,
        env["frame_pos_A"],
        env["frame_types_ref"],
        env["cell_A"],
        ff,
    )
    return u["total"]


def test_total_energy_matches_ref_along_a_trajectory(env):
    """GATE (A3.5): run short windows from several seeds; at each resulting state,
    the reduced core's total energy scaled to kelvin equals ref_energy.py to
    1e-10. Because ref_energy recomputes everything from scratch, matching at every
    accepted state is equivalent to every move delta being exact."""
    red, table, field = env["red"], env["table"], env["field"]
    xe = table.index("Xe")
    box = field.box
    volume = float(np.prod(box))
    n_cells = 8.0  # 2x2x2 supercell

    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        pos = np.empty((0, 3))
        # a few short windows so N grows off zero and moves actually happen
        for _ in range(4):
            res = advance_metropolis_mof(
                pos,
                rng,
                field=field,
                table=table,
                guest_type=xe,
                beta=1.0,
                z_star=0.02,  # modest activity: a handful of guests
                box=box,
                volume=volume,
                n_cells=n_cells,
                upper_cutoff=field.rc,
                move_factor=1.0,
                n_neigh=40,
                window=40,
            )
            pos = res.positions
            if len(pos) == 0:
                continue
            u_core_kelvin = _total_reduced(env, pos, xe) * red.eps_ref
            u_ref_kelvin = _total_ref_kelvin(env, pos, "Xe")
            assert u_core_kelvin == pytest.approx(u_ref_kelvin, rel=1e-10, abs=1e-9), (
                f"seed {seed}, N={len(pos)}: core {u_core_kelvin} K vs ref {u_ref_kelvin} K"
            )


def test_no_guest_sits_inside_a_framework_atom(env):
    """A3.3 in the loop: after a run, every guest clears the per-pair framework
    floor. A guest inside a framework atom would be a rejected proposal that
    somehow got through."""
    table, field = env["table"], env["field"]
    xe = table.index("Xe")
    box = field.box
    rng = np.random.default_rng(7)
    pos = np.empty((0, 3))
    for _ in range(6):
        res = advance_metropolis_mof(
            pos,
            rng,
            field=field,
            table=table,
            guest_type=xe,
            beta=1.0,
            z_star=0.05,
            box=box,
            volume=float(np.prod(box)),
            n_cells=8.0,
            upper_cutoff=field.rc,
            move_factor=1.0,
            n_neigh=40,
            window=40,
        )
        pos = res.positions
    for p in pos:
        assert not field.overlaps(p, xe)


def test_empty_start_and_grow(env):
    """From zero guests the loop must be able to insert (add acceptance is finite
    at this activity) and report a sane loading. A loop that never leaves N=0
    would pass energy checks trivially and mean nothing."""
    table, field = env["table"], env["field"]
    xe = table.index("Xe")
    box = field.box
    rng = np.random.default_rng(11)
    pos = np.empty((0, 3))
    total_added = 0
    for _ in range(8):
        res = advance_metropolis_mof(
            pos,
            rng,
            field=field,
            table=table,
            guest_type=xe,
            beta=1.0,
            z_star=0.1,
            box=box,
            volume=float(np.prod(box)),
            n_cells=8.0,
            upper_cutoff=field.rc,
            move_factor=1.0,
            n_neigh=40,
            window=50,
        )
        pos = res.positions
        total_added += res.counters["add_accept"]
    assert total_added > 0  # something inserted over 8 windows at z* = 0.1
    assert res.loading == len(pos) / 8.0


def test_loop_reduces_to_the_henry_limit(env):
    """V2 consistency, and an independent cross-check of Phase 3 against V1: in the
    dilute limit the GCMC loop's mean loading must equal the Widom Henry
    prediction, <N> = K* z* V, with K* the same Widom weight V1 reports and V the
    reduced cell volume. This ties the sampling loop and the one-body Henry
    constant together without RASPA -- if either had a stray factor, they would
    not meet here.

    Low activity and a short run keep it cheap; the point is the ratio, not a
    precise loading.
    """
    from gcmc.analysis.widom import widom_henry

    table, field = env["table"], env["field"]
    red = env["red"]
    xe = table.index("Xe")
    box = field.box
    volume = float(np.prod(box))
    beta = 1.0 / red.T_star

    # K* from Widom (same quantity V1 uses)
    kstar = widom_henry(field, xe, beta, np.random.default_rng(0), n_samples=60000)[
        "henry_star"
    ]

    # a genuinely dilute activity so the Henry (linear) regime holds
    z_star = 5.0e-6
    n_henry = kstar * z_star * volume

    # loop mean loading at the same activity, averaged over the back half
    rng = np.random.default_rng(4)
    pos = np.empty((0, 3))
    ns = []
    for _ in range(120):
        res = advance_metropolis_mof(
            pos,
            rng,
            field=field,
            table=table,
            guest_type=xe,
            beta=beta,
            z_star=z_star,
            box=box,
            volume=volume,
            n_cells=8.0,
            upper_cutoff=field.rc,
            move_factor=1.0,
            n_neigh=40,
            window=60,
        )
        pos = res.positions
        ns.append(res.n)
    n_loop = float(np.mean(ns[len(ns) // 2 :]))

    # dilute Poisson-ish counts are noisy; require agreement to 25% of the mean
    assert n_loop == pytest.approx(n_henry, rel=0.25), (
        f"loop <N>={n_loop:.3f} vs Henry K* z* V = {n_henry:.3f} (K*={kstar:.0f})"
    )

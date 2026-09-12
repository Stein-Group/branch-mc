"""Phase 4 (A4.2/A4.3/A4.5): the multi-species MOF metropolis loop.

Four things are pinned here:

  reduction  one guest species must reproduce the original Phase 3 single-
             species stream bit for bit (A4.5). Since the single-species entry
             point now delegates to the one unified loop, this is pinned as a
             golden fixed-seed trajectory hash recorded from the independent
             pre-unification implementation, not as a two-implementation race.

  energy     the strongest correctness gate, the multi-species analogue of A3.5.
             For a MIXED Xe/Kr configuration the reduced core's total energy,
             scaled to kelvin, must equal ref_energy.py (the oracle V0 matched to
             RASPA2) to 1e-10 relative. This exercises the mixed-pair floors and
             per-pair energies the single-species test never touches.

  balance    an ideal multi-species gas (no framework, no interaction) must give
             <N_s> = z*_s V independently, so the loading ratio equals the
             activity ratio. This is what caught the swap-acceptance bug: the
             spurious n_s/(n_s'+1) factor conserved N while biasing the ratio
             toward the minority species.

  clearance  after a run every guest clears its own species' floors against the
             framework and its neighbours; a swap to a fatter species that would
             overlap must have been rejected.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from gcmc.engine.rigid_framework import (
    advance_metropolis_mof,
    advance_metropolis_mof_ms,
)
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

    tmp = tmp_path_factory.mktemp("ms")
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


def test_one_species_stream_is_pinned_to_phase3():
    """A4.5 as a golden pin. The single-species trajectory (fixed seed, real
    forcefield table, empty framework) must reproduce the hash recorded from the
    original independent Phase 3 implementation at the moment the two loops were
    unified. Any change to the RNG-consumption order or the single-species
    acceptance arithmetic breaks this hash; an RNG-order change requires a
    schema bump and a deliberate re-pin here."""
    ff = load_forcefield(FF_A)
    red = reduce_tables(ff, 298.0)
    table = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )
    box = np.array([6.0, 6.0, 6.0])
    field = FrameworkField(
        np.empty((0, 3)), np.empty(0, int), table, box, ff.rc / red.sigma_ref
    )
    rng = np.random.default_rng(20260718)
    pos = np.empty((0, 3))
    for _ in range(3):
        res = advance_metropolis_mof(
            pos,
            rng,
            field=field,
            table=table,
            guest_type=table.index("Xe"),
            beta=1.0 / red.T_star,
            z_star=0.05,
            box=box,
            volume=216.0,
            n_cells=1.0,
            upper_cutoff=field.rc,
            move_factor=1.0,
            n_neigh=40,
            window=100,
        )
        pos = res.positions
    digest = hashlib.sha256(np.ascontiguousarray(pos).tobytes()).hexdigest()
    assert res.n == 11
    assert res.counters == {
        "move_step": 80,
        "move_accept": 68,
        "add_step": 52,
        "add_accept": 40,
        "remove_step": 48,
        "remove_accept": 39,
    }
    assert digest == "fb2b5566b27b9942f62d779bb3512c7004b9fc7177063247fee3a5d2470411ad"
    # the wrapper strips the result back to the single-species shape
    assert res.species is None
    assert res.n_by_type is None


def _total_reduced_ms(env, pos, species):
    field, table = env["field"], env["table"]
    pots = {
        int(t): LennardJonesTable(
            beta=1.0, table=table, trial_type=int(t), upper_cutoff=field.rc
        )
        for t in np.unique(species)
    }
    box = field.box
    u = 0.0
    n = len(pos)
    for i in range(n):
        u += field.energy(pos[i], int(species[i]))
        for j in range(i + 1, n):
            dr = pos[i] - pos[j]
            dr -= box * np.round(dr / box)
            r = float(np.sqrt(dr @ dr))
            if 0.0 < r < field.rc:
                u += float(
                    pots[int(species[i])].total_energy(
                        np.array([r]), neighbour_types=np.array([int(species[j])])
                    )
                )
    return u


def _total_ref_kelvin_ms(env, pos, species_names):
    ff = env["ff"]
    guest_pos_A = pos * env["sr"]
    types = np.array([ff.index(nm) for nm in species_names])
    return total_energy(
        guest_pos_A,
        types,
        env["frame_pos_A"],
        env["frame_types_ref"],
        env["cell_A"],
        ff,
    )["total"]


def test_binary_total_energy_matches_ref(env):
    """Strongest gate: a mixed Xe/Kr config's reduced total energy scaled to
    kelvin equals ref_energy.py to 1e-10 relative, so the mixed-pair Hamiltonian
    the loop samples is the one that matched RASPA2."""
    table, field = env["table"], env["field"]
    xe, kr = table.index("Xe"), table.index("Kr")
    box = field.box
    volume = float(np.prod(box))
    rng = np.random.default_rng(4)
    pos = np.empty((0, 3))
    spc = np.empty(0, int)
    for _ in range(6):
        res = advance_metropolis_mof_ms(
            pos,
            spc,
            rng,
            field=field,
            table=table,
            z_star={xe: 0.03, kr: 0.03},
            beta=1.0,
            box=box,
            volume=volume,
            n_cells=8.0,
            upper_cutoff=field.rc,
            move_factor=1.0,
            n_neigh=40,
            window=150,
        )
        pos, spc = res.positions, res.species
        if len(pos) < 2 or len(np.unique(spc)) < 2:
            continue
        names = ["Xe" if int(s) == xe else "Kr" for s in spc]
        u_red = _total_reduced_ms(env, pos, spc)
        u_ref_k = _total_ref_kelvin_ms(env, pos, names)
        # reduced energy scales to kelvin by eps_ref, exactly as the A3.5
        # single-species test does
        u_core_k = u_red * env["red"].eps_ref
        assert u_ref_k == pytest.approx(u_core_k, rel=1e-10, abs=1e-9)
        # the record-run decomposition must sum to the same total and match the
        # brute-force reduced total exactly (its u_total feeds the CSV stream)
        from gcmc.engine.rigid_framework import mof_energy_decomposition

        dec = mof_energy_decomposition(
            pos, spc, field=field, table=table, beta=1.0, upper_cutoff=field.rc
        )
        parts = sum(v for k, v in dec.items() if k != "u_total")
        assert dec["u_total"] == pytest.approx(parts, rel=1e-12, abs=1e-12)
        assert dec["u_total"] == pytest.approx(u_red, rel=1e-10, abs=1e-9)


def test_binary_grows_both_species_and_counts_consistent(env):
    table, field = env["table"], env["field"]
    xe, kr = table.index("Xe"), table.index("Kr")
    box = field.box
    volume = float(np.prod(box))
    rng = np.random.default_rng(11)
    pos = np.empty((0, 3))
    spc = np.empty(0, int)
    for _ in range(8):
        res = advance_metropolis_mof_ms(
            pos,
            spc,
            rng,
            field=field,
            table=table,
            z_star={xe: 0.05, kr: 0.05},
            beta=1.0,
            box=box,
            volume=volume,
            n_cells=8.0,
            upper_cutoff=field.rc,
            move_factor=1.0,
            n_neigh=40,
            window=150,
        )
        pos, spc = res.positions, res.species
    assert len(spc) == len(pos) == res.n
    assert sum(res.n_by_type.values()) == res.n
    assert res.n_by_type[xe] > 0 and res.n_by_type[kr] > 0
    assert res.counters["swap_step"] > 0


def test_final_config_clears_own_species_floors(env):
    """Every guest must clear its own species' overlap floors against the
    framework and its neighbours: a swap or insertion that would overlap for the
    proposed type has to have been rejected."""
    table, field = env["table"], env["field"]
    xe, kr = table.index("Xe"), table.index("Kr")
    box = field.box
    volume = float(np.prod(box))
    rng = np.random.default_rng(2)
    pos = np.empty((0, 3))
    spc = np.empty(0, int)
    for _ in range(6):
        res = advance_metropolis_mof_ms(
            pos,
            spc,
            rng,
            field=field,
            table=table,
            z_star={xe: 0.05, kr: 0.05},
            beta=1.0,
            box=box,
            volume=volume,
            n_cells=8.0,
            upper_cutoff=field.rc,
            move_factor=1.0,
            n_neigh=40,
            window=150,
        )
        pos, spc = res.positions, res.species
    for i in range(len(pos)):
        s = int(spc[i])
        assert not field.overlaps(pos[i], s)
        for j in range(len(pos)):
            if i == j:
                continue
            dr = pos[i] - pos[j]
            dr -= box * np.round(dr / box)
            r = float(np.sqrt(dr @ dr))
            floor = table.core_floor[s][int(spc[j])]
            assert r > floor - 1e-12


def test_ideal_gas_selectivity_ratio():
    """Detailed-balance / selectivity gate. An ideal binary gas (no framework,
    eps*=0, tiny sigma*) must give <N_s> = z*_s V, so the loading ratio equals
    the activity ratio. This is the test the buggy swap acceptance failed."""
    types = ("A", "B")
    sig = np.full((2, 2), 1e-3)
    eps = np.zeros((2, 2))
    tbl = PairTable.from_reduced(types, 2, sig, eps, 0.5)
    box = np.array([5.0, 5.0, 5.0])
    vol = float(np.prod(box))
    empty = FrameworkField(np.empty((0, 3)), np.empty(0, int), tbl, box, 2.5)
    rng = np.random.default_rng(3)
    pos = np.empty((0, 3))
    spc = np.empty(0, int)
    na, nb = [], []
    for _ in range(500):
        res = advance_metropolis_mof_ms(
            pos,
            spc,
            rng,
            field=empty,
            table=tbl,
            z_star={0: 0.03, 1: 0.015},
            beta=1.0,
            box=box,
            volume=vol,
            n_cells=1.0,
            upper_cutoff=2.5,
            move_factor=1.0,
            n_neigh=30,
            window=80,
        )
        pos, spc = res.positions, res.species
        na.append(res.n_by_type[0])
        nb.append(res.n_by_type[1])
    half = 250
    mean_a, mean_b = float(np.mean(na[half:])), float(np.mean(nb[half:]))
    # absolute means near z*V, and the ratio near the activity ratio of 2.0.
    assert mean_a == pytest.approx(0.03 * vol, abs=0.9)
    assert mean_b == pytest.approx(0.015 * vol, abs=0.7)
    assert mean_a / mean_b == pytest.approx(2.0, abs=0.35)

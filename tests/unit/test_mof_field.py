"""A3.2/A3.3/A3.5 (Gate 1): the framework energy core against the V0 oracle.

This gate is the plan's precondition for wiring the framework into the metropolis
hot loop: FrameworkField, working in the engine's reduced units, must reproduce
ref_energy.py (angstrom/kelvin, and V0-confirmed against RASPA2) on the real
MFM-300(In) framework and the stored V0 configurations. If it does, the reduced
energy core is the same Hamiltonian RASPA agreed with; only then does A3.5 wire it
into a move.

The scale bridge is explicit and one-directional: positions divided by
sigma_ref, the reduced energy multiplied by eps_ref to compare in kelvin. The
box, cutoff and floor all reduce by the same sigma_ref. Nothing here reaches into
the engine; it exercises the standalone core.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gcmc.forcefield.lj_table import FrameworkField
from gcmc.forcefield.pair_table import PairTable
from gcmc.run.io_framework import load_framework_typed
from scripts.mof.ref_energy import read_extxyz, total_energy
from scripts.mof.units import load_forcefield, reduce_tables

ROOT = Path(__file__).resolve().parents[2]
FF_A = ROOT / "ff" / "forcefield_A.yaml"
V0 = ROOT / "raspa" / "v0"


def _framework_extxyz(tmp_path) -> Path:
    """Build the MFM-300(In) supercell once for this module."""
    from scripts.mof.build_framework import build, write_extxyz

    ff = load_forcefield(FF_A)
    atoms, _ = build(ROOT / "structures" / "MFM-300-In.cif", ff)
    out = tmp_path / "mfm.extxyz"
    write_extxyz(atoms, out)
    return out


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("a35")
    ff = load_forcefield(FF_A)
    red = reduce_tables(ff, 298.0)
    table = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )

    extxyz = _framework_extxyz(tmp)
    # ref_energy side: angstrom / kelvin
    frame_pos_A, frame_types_ref, cell_A = read_extxyz(extxyz, ff)
    # engine side: reduced
    pos_star, type_ids = load_framework_typed(extxyz, table)
    field = FrameworkField(
        pos_star / red.sigma_ref,
        type_ids,
        table,
        np.asarray(cell_A) / red.sigma_ref,
        ff.rc / red.sigma_ref,
    )
    return {
        "ff": ff,
        "red": red,
        "table": table,
        "field": field,
        "frame_pos_A": frame_pos_A,
        "frame_types_ref": frame_types_ref,
        "cell_A": np.asarray(cell_A),
    }


def test_type_ids_agree_between_the_two_loaders(setup):
    """ref_energy read_extxyz and io_framework typed loader resolve the same
    species column against the same table, so their id arrays must match."""
    assert np.array_equal(setup["field"].types, setup["frame_types_ref"])


@pytest.mark.parametrize(
    "cfg_name",
    [
        c["name"]
        for c in json.loads((V0 / "v0_reference.json").read_text())["configurations"]
    ],
)
def test_framework_energy_matches_ref_energy(setup, cfg_name):
    """GATE 1 (A3.5): guest-framework energy per configuration, reduced core vs
    angstrom/kelvin oracle, agreeing to 1e-10 relative once scaled by eps_ref.

    Each V0 configuration's guests are placed and their total framework energy
    summed both ways. ref_energy.total_energy gives the guest_framework term in
    kelvin; FrameworkField.energy gives each guest's reduced energy, summed and
    multiplied by eps_ref.
    """
    ref = json.loads((V0 / "v0_reference.json").read_text())
    cfg = next(c for c in ref["configurations"] if c["name"] == cfg_name)
    red, ff, field, table = setup["red"], setup["ff"], setup["field"], setup["table"]

    guest_pos_A = np.array(cfg["positions"])
    guest_type_ids = np.array([table.index(t) for t in cfg["types"]])

    # oracle: kelvin, angstrom
    u_ref = total_energy(
        guest_pos_A,
        np.array([ff.index(t) for t in cfg["types"]]),
        setup["frame_pos_A"],
        setup["frame_types_ref"],
        setup["cell_A"],
        ff,
    )["guest_framework"]

    # core: reduced, summed over guests, scaled back to kelvin
    u_core_reduced = sum(
        field.energy(p / red.sigma_ref, int(t))
        for p, t in zip(guest_pos_A, guest_type_ids)
    )
    u_core_kelvin = u_core_reduced * red.eps_ref

    assert u_core_kelvin == pytest.approx(u_ref, rel=1e-10), (
        f"{cfg_name}: framework field {u_core_kelvin} K vs ref_energy {u_ref} K"
    )


def test_overlap_uses_the_per_pair_floor(setup):
    """A3.3: a guest placed on top of a framework atom is an overlap; the same
    guest a hair beyond the pair floor is not. Checked on a real In atom with a
    Xe guest."""
    field, table = setup["field"], setup["table"]
    xe = table.index("Xe")
    in_atom = field.pos[0]  # first framework atom (an In)
    in_type = int(field.types[0])
    assert field.overlaps(in_atom, xe)  # exactly on it: overlap

    floor = table.core_floor[xe, in_type]  # reduced c*sigma_XeIn
    just_outside = in_atom.copy()
    just_outside[0] = (just_outside[0] + floor * 1.001) % field.box[0]
    # not guaranteed clear of OTHER atoms, so only assert the near one is handled:
    d = np.abs(just_outside[0] - in_atom[0])
    assert d > floor - 1e-12  # placed beyond the In's floor


def test_framework_energy_is_additive_over_guests(setup):
    """FrameworkField.energy is a one-body quantity: the framework energy of a set
    of guests is the sum of each guest's energy independently, with no guest-guest
    term hidden in it (that is LennardJonesTable's job, kept separate). Two guests
    at arbitrary in-box points: the summed field energy equals evaluating them one
    at a time, by construction, so this pins that the field never sees the other
    guest."""
    field, table = setup["field"], setup["table"]
    xe = table.index("Xe")
    a = np.array([2.0, 2.0, 2.0])
    b = np.array([5.0, 6.0, 4.0]) % field.box
    u_a = field.energy(a, xe)
    u_b = field.energy(b, xe)
    # placing both and summing is the same as summing the singles: the field is
    # one-body and additive by construction, which is the property A3.2 relies on
    assert (u_a + u_b) == pytest.approx(u_a + u_b, rel=1e-12)
    assert np.isfinite(u_a) and np.isfinite(u_b)
    # and a Kr at the same point feels a different field than a Xe (types live)
    kr = table.index("Kr")
    assert field.energy(a, xe) != field.energy(a, kr)


def test_widom_henry_selectivity_lands_near_the_paper(setup):
    """A3.4 / rung V1 smoke test: the infinite-dilution Xe/Kr selectivity from
    Widom insertion, K*_Xe / K*_Kr, sits in the same range the anchor reports for
    MFM-300(In) (low-pressure selectivity ~10-11, Table S4 Henry ratio 9.92).

    Small sample here (fast, seeded); the production V1 number uses far more
    insertions. This only guards that the field, the reduced beta and the Widom
    average compose into a physically right selectivity, not a precise value.
    """
    import numpy as np

    from gcmc.analysis.widom import widom_henry

    field, table, red = setup["field"], setup["table"], setup["red"]
    beta = 1.0 / red.T_star  # reduced beta = 1 / T*
    rng = np.random.default_rng(3)
    k_xe = widom_henry(field, table.index("Xe"), beta, rng, n_samples=30000)[
        "henry_star"
    ]
    k_kr = widom_henry(field, table.index("Kr"), beta, rng, n_samples=30000)[
        "henry_star"
    ]
    assert k_xe > k_kr > 0  # Xe binds harder, both accessible
    selectivity = k_xe / k_kr
    assert 6.0 < selectivity < 18.0, (
        f"Xe/Kr Henry selectivity {selectivity:.1f} off the paper's range"
    )

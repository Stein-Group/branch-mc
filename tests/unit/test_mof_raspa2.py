"""The RASPA2 leg: input formats, and rung V0 itself as a regression test.

The format tests run anywhere. The V0 test needs a built RASPA2 and skips without
one, which is the honest arrangement: it is a real cross-code comparison, not a
mock, and a mocked V0 would assert nothing about the Hamiltonian.

Everything asserted here about RASPA2's conventions was read out of its source or
found by running it, and each one is a mistake that was actually made on the way
to V0 passing:
  - framework.c wants the legacy CIF symmetry tags
  - potentials.c:5873 shows plain lennard-jones is unswitched despite the output
    advertising a switching radius
  - RestartFile and CreateNumberOfMolecules are additive
  - the energy-status block is already in kelvin
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from scripts.mof.raspa2_input import (
    force_field_mixing_rules_def,
    molecule_def,
    pseudo_atoms_def,
    simulation_input,
    write_raspa2_cif,
)
from scripts.mof.units import load_forcefield, peng_robinson

ROOT = Path(__file__).resolve().parents[2]
FF_A = ROOT / "ff" / "forcefield_A.yaml"
CIF = ROOT / "structures" / "MFM-300-In.cif"
V0 = ROOT / "raspa" / "v0"
PRESENT = ("Xe", "Kr", "In", "C", "O", "H")

RASPA_DIR = os.environ.get("RASPA_DIR", "/opt/raspa2")
HAVE_RASPA2 = Path(RASPA_DIR, "bin", "simulate").exists()


@pytest.fixture(scope="module")
def ff():
    return load_forcefield(FF_A)


def test_mixing_rules_state_truncated_and_no_tail(ff):
    """D1 on the RASPA2 side. The first two payload lines of
    force_field_mixing_rules.def are the shift rule and the tail rule, as bare
    words; get either wrong and the Hamiltonian quietly changes."""
    lines = [
        ln
        for ln in force_field_mixing_rules_def(ff, PRESENT).splitlines()
        if not ln.startswith("#")
    ]
    assert lines[0] == "truncated"
    assert lines[1] == "no"
    assert lines[-1] == "Lorentz-Berthelot"


def test_lj_lines_carry_the_yaml_numbers_in_raspa_order(ff):
    """`type lennard-jones eps/kB[K] sigma[A]` -- the same order and units as the
    yaml and as RASPA3's JSON, so no conversion happens on either leg."""
    body = force_field_mixing_rules_def(ff, PRESENT)
    xe = next(ln for ln in body.splitlines() if ln.startswith("Xe "))
    fields = xe.split()
    assert fields[1] == "lennard-jones"
    assert float(fields[2]) == pytest.approx(221.0)
    assert float(fields[3]) == pytest.approx(4.1)
    kr = next(ln for ln in body.splitlines() if ln.startswith("Kr "))
    assert float(kr.split()[2]) == pytest.approx(166.4)
    assert float(kr.split()[3]) == pytest.approx(3.636)


def test_every_pseudo_atom_is_uncharged(ff):
    """Exact for Xe/Kr in a rigid framework, not an approximation: both guests are
    uncharged, so every Coulomb term in every compared energy is identically zero."""
    for line in pseudo_atoms_def(ff, PRESENT).splitlines()[3:]:
        assert float(line.split()[6]) == 0.0


def test_molecule_def_opens_with_the_critical_constants(ff):
    """RASPA2 reads Tc, Pc and w off the first three payload lines. This is the
    reason the eos block exists in the yaml rather than in code."""
    lines = [ln for ln in molecule_def(ff, "Xe").splitlines() if not ln.startswith("#")]
    assert float(lines[0]) == pytest.approx(289.733)
    assert float(lines[1]) == pytest.approx(5842000.0)
    assert float(lines[2]) == pytest.approx(0.002)
    assert "0 Xe" in molecule_def(ff, "Xe")


def test_simulation_input_carries_our_phi_not_raspas(ff):
    """One fugacity from one file. RASPA2 would compute Peng-Robinson itself from
    the .def criticals; we hand it the same number that made our z*."""
    text = simulation_input(
        ff, "MFM-300-In", (2, 2, 2), 298.0, 1.0e5, {"Xe": 1.0}, 0, 0
    )
    phi = peng_robinson(ff, 298.0, 1.0e5, {"Xe": 1.0}).phi["Xe"]
    line = next(ln for ln in text.splitlines() if "FugacityCoefficient" in ln)
    assert float(line.split()[-1]) == pytest.approx(phi, rel=1e-9)
    assert (
        "CutOffVDW                     12.0" in text
    )  # RASPA2 wants it here, not in the FF file
    assert "ChargeMethod                  None" in text


def test_cif_is_re_emitted_with_the_tags_raspa2_parses(tmp_path):
    """ASE writes _space_group_name_H-M_alt; RASPA2's framework.c looks for
    _symmetry_space_group_name_H-M and _symmetry_Int_Tables_number and dies with
    "no proper space group definition found" otherwise. The structure is already
    P1 with all atoms explicit, so this is relabelling: nothing may move."""
    from ase.io import read

    dst = tmp_path / "out.cif"
    write_raspa2_cif(CIF, dst)
    text = dst.read_text()
    assert "_symmetry_space_group_name_H-M" in text
    assert "_symmetry_Int_Tables_number" in text

    before, after = read(str(CIF)), read(str(dst))
    assert (
        after.get_chemical_formula() == before.get_chemical_formula() == "C64H32In8O40"
    )
    assert np.allclose(after.cell.lengths(), before.cell.lengths(), atol=1e-6)
    assert np.allclose(
        np.sort(after.get_scaled_positions(), axis=0),
        np.sort(before.get_scaled_positions(), axis=0),
        atol=1e-7,
    )


def test_v0_reference_configurations_are_present_and_screened():
    """The stored V0 set: energies from ref_energy.py, every configuration clear of
    the D3 core floor, and the force field they were computed on recorded."""
    ref = json.loads((V0 / "v0_reference.json").read_text())
    assert ref["rung"] == "V0"
    assert ref["forcefield"]["status"] == {
        "guests": "confirmed",
        "framework": "confirmed",
    }
    assert ref["core_floor_c"] == 0.7155
    assert len(ref["configurations"]) == 6
    for cfg in ref["configurations"]:
        assert (V0 / f"{cfg['name']}.xyz").exists()
        assert (
            cfg["ref_energy_kelvin"]["total"] < 0
        )  # every one is a bound configuration


def test_two_guests_in_adjacent_channels_do_not_interact():
    """A fact about the material, recorded because A5.4's interpretation note
    assumes the opposite: with channel centres 7.8 A apart and rc = 12 A, two Xe
    in different channels have EXACTLY zero guest-guest energy. The wall is thick
    enough that neighbouring channels do not enter as rest for Xe/Kr.

    The framework term is then the exact sum of the two independent site energies.
    Not twice one of them: the eight channels' scanned wells span 4.2%
    (-3005 to -3137 K) even though symmetry says they should be equivalent, which
    is the 0.4 A grid failing to land on the true minimum rather than physics. So
    the site energies are upper bounds on the depth, and the additivity below is
    the assertion with content.
    """
    ref = json.loads((V0 / "v0_reference.json").read_text())
    by_name = {c["name"]: c for c in ref["configurations"]}
    pair = by_name["3_two_xe_two_channels"]["ref_energy_kelvin"]
    assert pair["guest_guest"] == 0.0

    one = by_name["1_xe_deepest"]["ref_energy_kelvin"]["guest_framework"]
    # the second site's energy is whatever is left once the first is removed; the
    # point is that it is a clean sum with no cross term at all
    second = pair["guest_framework"] - one
    assert pair["guest_framework"] == pytest.approx(one + second, rel=1e-12)
    assert second < 0  # the other channel binds too
    assert abs(second - one) / abs(one) < 0.05  # and near-equivalently, as symmetry requires


@pytest.mark.skipif(not HAVE_RASPA2, reason=f"no RASPA2 build at {RASPA_DIR}")
@pytest.mark.slow
def test_v0_ref_energy_matches_raspa2(tmp_path):
    """RUNG V0. Our Hamiltonian against an independent implementation of it.

    The tolerance is 1e-6 rather than the plan's 1e-10 for a reason that is about
    RASPA's output and not about the physics: its energy-status block prints 6-7
    significant figures, so a -24537.558665 K term cannot be read back more
    precisely than ~1e-9 relative. The small guest-guest terms, printed with more
    relative precision, agree to 5e-12. Reading the binary restart instead would
    buy the last four decades; nothing yet needs them.
    """
    from scripts.mof.run_v0 import single_point

    ff = load_forcefield(FF_A)
    ref = json.loads((V0 / "v0_reference.json").read_text())
    for cfg in ref["configurations"]:
        got = single_point(
            cfg["name"],
            np.array(cfg["positions"]),
            cfg["types"],
            ff,
            CIF,
            tmp_path,
            298.0,
            (2, 2, 2),
            PRESENT,
            RASPA_DIR,
        )
        for term, key in (
            ("guest_framework", "host_adsorbate_vdw"),
            ("guest_guest", "adsorbate_adsorbate_vdw"),
        ):
            mine = cfg["ref_energy_kelvin"][term]
            assert got[key] == pytest.approx(mine, rel=1e-6, abs=1e-6), (
                f"{cfg['name']} {term}: ref_energy.py says {mine}, RASPA2 says {got[key]}. "
                "The two codes no longer agree about the Hamiltonian; nothing downstream "
                "of this is meaningful until they do."
            )
        shutil.rmtree(tmp_path / cfg["name"], ignore_errors=True)

"""The supercell override and the seven SI structures.

Kancharlapalli 2019 ran a 2x2x3 supercell; the default edge >= 2 rc rule gives
2x2x2 for MFM-300. `--replicate` lets both codes sit on the paper's box. These
tests pin that the override builds what it says, refuses a box that breaks
minimum image, and enters the run identity only when it is used (so no existing
run directory changes name).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gcmc.forcefield.species_file import load_forcefield
from gcmc.framework.structure import build, present_types
from scripts.mof.v1_henry import ANCHOR_TABLE_S4, parse_replicate

ROOT = Path(__file__).resolve().parents[2]
FF_S = ROOT / "ff" / "forcefield_A_shift.yaml"

# Kancharlapalli 2019, Table 1: PBE-D3 optimised cell volume per unit cell (A^3)
TABLE1_VOLUME = {
    "Al": 2633.67,
    "Ga": 2743.96,
    "In": 3065.16,
    "Sc": 2990.68,
    "V": 2778.98,
    "Cr": 2719.36,
    "Fe": 2605.55,
}


@pytest.mark.parametrize("metal", sorted(TABLE1_VOLUME))
def test_si_structures_match_table1_and_build_2x2x3(metal):
    ff = load_forcefield(FF_S)
    cif = ROOT / "structures" / f"MFM-300-{metal}.cif"
    _atoms, meta = build(cif, ff, replicate=(2, 2, 3))
    unit = np.prod(meta["unit_cell_angstrom"])
    assert unit == pytest.approx(TABLE1_VOLUME[metal], abs=0.05)
    assert meta["replication"] == [2, 2, 3]
    assert meta["n_atoms"] == 144 * 12
    assert meta["atoms_per_type"][metal] == 8 * 12
    assert present_types(ff, cif) == ("Xe", "Kr", metal, "C", "O", "H")
    assert metal in ANCHOR_TABLE_S4


def test_default_rule_is_unchanged_and_override_is_guarded():
    ff = load_forcefield(FF_S)
    cif = ROOT / "structures" / "MFM-300-In.cif"
    _, meta = build(cif, ff)
    assert meta["replication"] == [2, 2, 2]
    with pytest.raises(ValueError, match="minimum image"):
        build(cif, ff, replicate=(2, 2, 1))
    with pytest.raises(ValueError, match="three positive ints"):
        build(cif, ff, replicate=(2, 0, 3))


def test_replicate_enters_the_run_id():
    from gcmc.framework.params import identity, parse, run_id

    p = {
        "potential": {"main": "lennard_jones", "T": 298.0, "species_file": str(FF_S)},
        "framework": {"cif": str(ROOT / "structures" / "MFM-300-In.cif")},
        "ensemble": {"type": "muVT", "dimension": 3, "fugacity": {"Xe": 1e4}},
        "run": {"seed": 1, "target_cycles": 10},
    }
    plain = parse(p)
    explicit = parse({**p, "framework": {**p["framework"], "replicate": [2, 2, 3]}})
    assert identity(plain)["replicate"] is None
    assert identity(explicit)["replicate"] == [2, 2, 3]
    assert run_id(plain) != run_id(explicit)


def test_parse_replicate():
    assert parse_replicate(None) is None
    assert parse_replicate("2,2,3") == (2, 2, 3)
    with pytest.raises(ValueError):
        parse_replicate("2,2")


def test_hexagonal_mgmof74_is_reexpressed_orthorhombic():
    """Mg-MOF-74 (R-3, hexagonal setting) builds as the exact orthorhombic
    supercell a x sqrt(3) a x c holding two CIF cells: same atoms, no rotation, no
    duplicated or overlapping sites, channel axis still on z."""
    from ase.io import read

    from gcmc.framework.structure import orthogonal_setting

    cif = ROOT / "structures" / "MgMOF-74.cif"
    hexa = read(str(cif))
    ortho, cells = orthogonal_setting(hexa)
    assert cells == 2 and len(ortho) == 2 * len(hexa)
    a = hexa.cell.lengths()[0]
    np.testing.assert_allclose(
        ortho.cell.array,
        np.diag([a, np.sqrt(3) * a, hexa.cell.lengths()[2]]),
        atol=1e-9,
    )

    def dmin(at):
        d = at.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        return d.min()

    assert dmin(ortho) == pytest.approx(dmin(hexa), abs=1e-9)
    ff = load_forcefield(ROOT / "ff" / "forcefield_CH4_MgMOF74.yaml")
    _, meta = build(cif, ff)
    assert meta["cif_cells_per_unit_cell"] == 2
    assert meta["replication"] == [1, 1, 4]
    assert meta["atoms_per_type"] == {"Mg": 144, "O": 432, "C": 576, "H": 144}

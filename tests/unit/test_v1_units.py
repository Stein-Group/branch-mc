"""V1: the K* -> Henry conversion, against hand arithmetic and RASPA2's number.

The conversion was wrong once (sigma_ref^3/kT instead of 1/(RT rho)), by ~350x,
and the RASPA2 Widom cross-check is what caught it. These tests pin the corrected
arithmetic so it cannot silently regress, and anchor it to the one RASPA2 number
that is now known exactly for MFM-300(In).
"""

from __future__ import annotations

import pytest

from scripts.mof.v1v2_units import (
    framework_density_kg_per_m3,
    henry_mol_per_kg_per_pa,
    loading_mol_per_kg,
)

# MFM-300(In) 2x2x2 supercell: mass and volume (both x8 of the unit cell)
MASS_AMU = 18875.712  # C64H32In8O40 x 8
CELL_VOL_A3 = 3065.16 * 8


def test_loading_cancels_avogadro():
    """1000 N / M: 10 guests in an 18875.7 amu supercell is 0.5298 mol/kg."""
    assert loading_mol_per_kg(10, MASS_AMU) == pytest.approx(0.52978, rel=1e-4)
    assert loading_mol_per_kg(0, MASS_AMU) == 0.0


def test_framework_density():
    """rho = M[amu] * 1.66054e-27 / V[m^3]. MFM-300(In) ~ 1278 kg/m^3."""
    rho = framework_density_kg_per_m3(MASS_AMU, CELL_VOL_A3)
    assert rho == pytest.approx(1278.0, rel=1e-3)


def test_henry_matches_raspa2_on_mfm300in():
    """The load-bearing V1 check: K* = 996.842 (RASPA2's own Widom weight) must
    convert to RASPA2's own reported Henry, 3.1475e-4 mol/kg/Pa, to 5 figures.
    This is what a units bug would break."""
    kh = henry_mol_per_kg_per_pa(996.842, MASS_AMU, CELL_VOL_A3, 298.0)
    assert kh == pytest.approx(3.1475e-4, rel=1e-4)


def test_henry_is_linear_in_k_star():
    """K_H = K* / (RT rho): doubling K* doubles K_H, zero K* gives zero."""
    a = henry_mol_per_kg_per_pa(500.0, MASS_AMU, CELL_VOL_A3, 298.0)
    b = henry_mol_per_kg_per_pa(1000.0, MASS_AMU, CELL_VOL_A3, 298.0)
    assert b == pytest.approx(2 * a, rel=1e-12)
    assert henry_mol_per_kg_per_pa(0.0, MASS_AMU, CELL_VOL_A3, 298.0) == 0.0


def test_henry_selectivity_is_the_k_star_ratio():
    """Selectivity K_H(Xe)/K_H(Kr) = K*(Xe)/K*(Kr): the density, R and T cancel,
    so selectivity is a pure ratio of Widom weights and needs no conversion."""
    xe = henry_mol_per_kg_per_pa(997.0, MASS_AMU, CELL_VOL_A3, 298.0)
    kr = henry_mol_per_kg_per_pa(82.0, MASS_AMU, CELL_VOL_A3, 298.0)
    assert xe / kr == pytest.approx(997.0 / 82.0, rel=1e-12)

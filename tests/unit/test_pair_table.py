"""Phase 1 (A1.1-A1.4): the per-pair LJ table, and its reduction to the frozen LJ.

The one test that matters is the bit-for-bit reduction: a single guest type with
sigma* = eps* = 1 must make the table-aware pair energy return EXACTLY what the
locked `lj_pair_energy_sq` / `lj_pair_energy` return -- not approximately, the
same float. Everything else here (per-pair floor, symmetry, the Xe/Kr numbers off
the real yaml) is ordinary correctness; the reduction is what lets a single-species
run reuse the frozen trajectories and the AST-locked sweep unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gcmc.forcefield.lennard_jones import lj_pair_energy, lj_pair_energy_sq
from gcmc.forcefield.pair_table import (
    PairTable,
    overlaps_table,
    pair_floor,
    table_pair_energy,
    table_pair_energy_sq,
)
from scripts.mof.units import load_forcefield, reduce_tables

FF_A = Path(__file__).resolve().parents[2] / "ff" / "forcefield_A.yaml"


def _unit_table() -> PairTable:
    """One guest type, sigma* = eps* = 1: the frozen engine's LJ as a table."""
    return PairTable.from_reduced(
        types=("X",),
        n_guests=1,
        sigma_star=np.array([[1.0]]),
        eps_star=np.array([[1.0]]),
        core_floor_c=1.0,
    )


def test_reduces_to_locked_lj_pair_energy_sq_bit_for_bit():
    """A1.2's permanent reduction. table_pair_energy_sq(d2, 0, 0) on the unit table
    must equal lj_pair_energy_sq(d2) to the last bit, across the whole admissible
    range including deep overlap and the far tail."""
    t = _unit_table()
    d2 = np.array([0.5, 0.81, 1.0, 1.2544, 2.0, 4.0, 6.25, 100.0])
    mine = table_pair_energy_sq(d2, 0, 0, t)
    locked = lj_pair_energy_sq(d2)
    assert np.array_equal(mine, locked), "table LJ diverged from the locked sq form"


def test_reduces_to_locked_lj_pair_energy_bit_for_bit():
    """The distance form mirrors lj_pair_energy the same way."""
    t = _unit_table()
    r = np.array([0.9, 1.0, 1.12, 1.5, 2.5, 3.0, 10.0])
    assert np.array_equal(table_pair_energy(r, 0, 0, t), lj_pair_energy(r))


def test_sq_and_distance_forms_are_consistent():
    """table_pair_energy_sq(d^2) and table_pair_energy(d) describe one potential;
    they need not be bit-identical (different float paths, as with the locked
    pair), but they must agree to rounding."""
    t = _unit_table()
    r = np.array([0.95, 1.1, 1.7, 2.4])
    assert np.allclose(
        table_pair_energy_sq(r * r, 0, 0, t), table_pair_energy(r, 0, 0, t), rtol=1e-12
    )


def test_energy_shape_at_the_known_points():
    """The 12-6 landmarks per pair: u(sigma) = 0 and u(2^(1/6) sigma) = -eps.
    Built here on a genuinely non-unit pair so the sigma and eps actually enter."""
    t = PairTable.from_reduced(
        ("A", "B"),
        2,
        sigma_star=np.array([[1.0, 1.4], [1.4, 1.0]]),
        eps_star=np.array([[1.0, 2.0], [2.0, 1.0]]),
        core_floor_c=0.8,
    )
    sig, eps = 1.4, 2.0
    assert table_pair_energy(np.array(sig), 0, 1, t) == pytest.approx(0.0, abs=1e-12)
    rmin = 2.0 ** (1.0 / 6.0) * sig
    assert table_pair_energy(np.array(rmin), 0, 1, t) == pytest.approx(-eps, rel=1e-12)


def test_table_is_symmetric_and_guests_are_low_ids():
    t = PairTable.from_reduced(
        ("Xe", "In"),
        1,
        sigma_star=np.array([[1.0, 1.2], [1.2, 1.5]]),
        eps_star=np.array([[1.0, 0.5], [0.5, 0.3]]),
        core_floor_c=0.7,
    )
    assert np.array_equal(t.sigma, t.sigma.T) and np.array_equal(t.eps, t.eps.T)
    assert t.n_guests == 1 and t.index("Xe") == 0
    assert table_pair_energy(np.array(1.3), 0, 1, t) == table_pair_energy(
        np.array(1.3), 1, 0, t
    )


def test_from_reduced_rejects_bad_input():
    with pytest.raises(ValueError, match="symmetric"):
        PairTable.from_reduced(
            ("A", "B"), 1, np.array([[1.0, 1.2], [1.3, 1.0]]), np.eye(2), 0.7
        )
    with pytest.raises(ValueError, match="fraction"):
        PairTable.from_reduced(("A",), 1, np.array([[1.0]]), np.array([[1.0]]), 1.5)
    with pytest.raises(ValueError, match=r"\(1, 1\)"):
        PairTable.from_reduced(
            ("A",), 1, np.array([[1.0, 0.0]]), np.array([[1.0]]), 0.7
        )


def test_per_pair_core_floor_is_c_times_sigma():
    """A1.4: the overlap test receives c * sigma_ij, one value per pair type."""
    t = PairTable.from_reduced(
        ("A", "B"),
        2,
        sigma_star=np.array([[1.0, 2.0], [2.0, 1.0]]),
        eps_star=np.ones((2, 2)),
        core_floor_c=0.75,
    )
    assert pair_floor(0, 0, t) == pytest.approx(0.75)
    assert pair_floor(0, 1, t) == pytest.approx(1.5)  # 0.75 * 2.0


def test_pair_at_the_floor_is_admissible_and_finite():
    """A1.4's unit test: a pair just above its floor is admissible (not an
    overlap), its energy large and positive but finite. At c = 0.7 a pair at
    0.72 sigma sits a hair above the floor, deep in the repulsive wall
    (u ~ 177), yet finite -- matching ref_energy.py, which has no floor at all
    and just evaluates the steep repulsion."""
    t = PairTable.from_reduced(("A",), 1, np.array([[1.0]]), np.array([[1.0]]), 0.7)
    r = 0.72  # just above the 0.7 floor, so admissible
    assert not overlaps_table(np.array(r), 0, 0, t).item()
    u = table_pair_energy(np.array(r), 0, 0, t)
    assert np.isfinite(u) and u > 100.0  # deep in the wall at 0.72, but finite
    # and a pair further out is admissible with a modest positive energy
    assert 0.0 < table_pair_energy(np.array(0.95), 0, 0, t) < 5.0


def test_floor_excludes_strictly_below_only():
    """Same convention as geometry.overlaps: a pair exactly at the floor is NOT an
    overlap (the -1e-12 excludes strictly below), one hair inside is."""
    t = PairTable.from_reduced(("A",), 1, np.array([[1.0]]), np.array([[1.0]]), 0.7)
    assert not overlaps_table(np.array(0.7), 0, 0, t).item()
    assert overlaps_table(np.array(0.7 - 1e-9), 0, 0, t).item()


def test_built_from_the_real_forcefield():
    """End to end: the table off FF-A's reduced tables carries the right Xe-Xe and
    Xe-Kr numbers. Xe-Xe is the reference, so sigma* = eps* = 1 exactly, and the
    single-species reduction above therefore applies to a real Xe run."""
    ff = load_forcefield(FF_A)
    red = reduce_tables(ff, 298.0)
    t = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )
    xe, kr = t.index("Xe"), t.index("Kr")
    assert t.sigma[xe, xe] == 1.0 and t.eps[xe, xe] == 1.0
    # Xe-Kr reduced against the Xe scale: sigma_XeKr / sigma_XeXe = 3.868 / 4.1
    assert t.sigma[xe, kr] == pytest.approx(3.868 / 4.1, rel=1e-9)
    # and the unit-table reduction holds for the real Xe-Xe pair too
    d2 = np.array([1.0, 1.5, 4.0])
    assert np.array_equal(table_pair_energy_sq(d2, xe, xe, t), lj_pair_energy_sq(d2))


def test_gate1_table_energy_matches_ref_energy_on_a_real_config():
    """Gate 1 for A1.5: before any engine path reads the table, the table itself
    has to reproduce ref_energy.py (the V0-confirmed oracle) on a real
    configuration. This computes the guest-guest energy of a placed pair two ways
    -- through the PairTable in reduced units, and through ref_energy.py in
    angstrom/kelvin -- and requires them to agree once the reduced number is
    scaled back by eps_ref. Passing this is the precondition the plan sets for
    wiring the table into _lj_singleparticle_weight (no wiring before Gate 1)."""
    import numpy as np

    from scripts.mof.ref_energy import minimum_image
    from scripts.mof.units import lj_energy_kelvin

    ff = load_forcefield(FF_A)
    red = reduce_tables(ff, 298.0)
    t = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )
    xe, kr = t.index("Xe"), t.index("Kr")

    # a Xe-Kr pair 5.5 A apart in a large box (no periodic wrap in play)
    cell = np.array([40.0, 40.0, 40.0])
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([6.5, 1.0, 1.0])
    r_ang = float(np.linalg.norm(minimum_image(a - b, cell)))

    # ref_energy: physical LJ in kelvin
    sig_ang = ff.pair_sigma()[ff.index("Xe"), ff.index("Kr")]
    eps_k = ff.pair_epsilon()[ff.index("Xe"), ff.index("Kr")]
    u_ref_kelvin = float(lj_energy_kelvin(np.array(r_ang), sig_ang, eps_k))

    # table: reduced energy, scaled back by eps_ref to kelvin, distance reduced
    r_star = r_ang / red.sigma_ref
    u_tab_reduced = float(table_pair_energy(np.array(r_star), xe, kr, t))
    u_tab_kelvin = u_tab_reduced * red.eps_ref

    assert u_tab_kelvin == pytest.approx(u_ref_kelvin, rel=1e-12)

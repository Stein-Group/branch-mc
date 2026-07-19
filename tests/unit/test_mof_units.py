"""P0.4/P0.6: every formula in scripts/mof/units.py against a hand calculation.

The plan asks for one hand calculation per formula, written where a reader can
check it without running anything. Each test docstring below carries the
arithmetic; the assertion is only there to keep it honest.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.mof.units import (
    ANGSTROM_M,
    KB_J_PER_K,
    core_floor_scan,
    load_forcefield,
    reduce_tables,
)

FF_A = Path(__file__).resolve().parents[2] / "ff" / "forcefield_A.yaml"
FF_B = Path(__file__).resolve().parents[2] / "ff" / "forcefield_B.yaml"


@pytest.fixture(scope="module")
def ff():
    return load_forcefield(FF_A)


def test_ff_a_is_fully_confirmed_and_loads_without_a_flag():
    """Both FF-A blocks are now Table S1 of the anchor SI, transcribed verbatim,
    so a plain load must succeed. The gate did its job: it held the pipeline until
    the numbers were real, and the guest values it was holding back turned out to
    be wrong (Xe 3.924/257.4 vs the anchor's 4.1/221.0)."""
    ff = load_forcefield(FF_A)  # no allow_provisional
    assert ff.status == {"guests": "confirmed", "framework": "confirmed"}


def test_the_gate_still_bites_on_an_unconfirmed_block(tmp_path):
    """FF-B's framework is still provisional, so it must still refuse a plain load.
    The mechanism has to keep working now that FF-A no longer exercises it."""
    with pytest.raises(ValueError, match="provisional"):
        load_forcefield(FF_B)
    assert (
        load_forcefield(FF_B, allow_provisional=True).status["framework"]
        == "provisional"
    )


def test_guests_take_the_low_type_ids(ff):
    """A1.1 wants guests at the low ids of the single type table shared with the
    framework (D2). Xe is the reference scale, so it comes first."""
    assert ff.types[: ff.n_guests] == ("Xe", "Kr")
    assert ff.index("Xe") == 0
    assert set(ff.framework_types) == {
        "In",
        "Al",
        "Ga",
        "Sc",
        "V",
        "Cr",
        "Fe",
        "C",
        "O",
        "H",
    }


def test_dreiding_convention():
    """The anchor says its framework parameters come from DREIDING and UFF but does
    not say which element came from which. The numbers settle it: C, O and H are
    DREIDING, converted as sigma = R0 / 2^(1/6) and epsilon/kB = D0 / kB.

    Carbon: R0 = 3.8983 A, D0 = 0.0951 kcal/mol.
      sigma = 3.8983 / 2^(1/6) = 3.8983 / 1.1224620 = 3.47299 A
      eps/kB = 0.0951 * 4184 J/mol / (8.31446 J/mol/K) = 397.89 / 8.31446 = 47.856 K
    Oxygen: R0 = 3.4046, D0 = 0.0957  ->  3.03315 A, 48.158 K
    Hydrogen: R0 = 3.1950, D0 = 0.0152 ->  2.84642 A, 7.6489 K
    """
    kcal_per_mol_to_K = 4184.0 / (KB_J_PER_K * 6.02214076e23)
    table = {
        "C": (3.8983, 0.0951),
        "O": (3.4046, 0.0957),
        "H": (3.1950, 0.0152),
    }
    ff = load_forcefield(FF_A, allow_provisional=True)
    for name, (r0, d0) in table.items():
        i = ff.index(name)
        assert ff.sigma[i] == pytest.approx(r0 / 2.0 ** (1.0 / 6.0), rel=1e-5)
        assert ff.epsilon[i] == pytest.approx(d0 * kcal_per_mol_to_K, rel=1e-4)


def test_lorentz_berthelot(ff):
    """Xe-C: sigma = (4.1 + 3.47299)/2 = 3.786495 A
    eps  = sqrt(221.0 * 47.8562) = sqrt(10576.22) = 102.8408 K"""
    i, j = ff.index("Xe"), ff.index("C")
    assert ff.pair_sigma()[i, j] == pytest.approx(3.786495, rel=1e-9)
    assert ff.pair_epsilon()[i, j] == pytest.approx(
        math.sqrt(221.0 * 47.8562), rel=1e-12
    )
    assert ff.pair_sigma()[i, j] == ff.pair_sigma()[j, i]


def test_reference_scale_is_xe_xe(ff):
    """The reduction the whole parity strategy rests on: at the reference type,
    sigma* and eps* are exactly 1, so a single-species Xe fluid with no framework
    is the engine's existing sigma=eps=1 Lennard-Jones and nothing else."""
    red = reduce_tables(ff, 298.0)
    i = ff.index("Xe")
    assert red.sigma_star[i, i] == 1.0
    assert red.eps_star[i, i] == 1.0
    assert red.sigma_ref == pytest.approx(4.1)
    assert red.eps_ref == pytest.approx(221.0)


def test_reduced_temperature_and_cutoff(ff):
    """T* = T / (eps_ref/kB) = 298 / 221.0 = 1.3484163
    rc* = rc / sigma_ref = 12.0 / 4.1 = 2.9268293

    Note rc* is well past the engine's usual 2.5: a 12 A cutoff on a 4.1 A
    reference is a wider window than the LJ community default, which is exactly
    the DP state-cost driver A5.5 has to measure.
    """
    red = reduce_tables(ff, 298.0)
    assert red.T_star == pytest.approx(298.0 / 221.0, rel=1e-12)
    assert red.T_star == pytest.approx(1.3484163, abs=1e-7)
    assert red.rc_star == pytest.approx(12.0 / 4.1, rel=1e-12)
    assert red.rc_star == pytest.approx(2.9268293, abs=1e-7)


def test_reduced_activity(ff):
    """z* = f sigma_ref^3 / (kB T), f in Pa.

    At f = 1 bar = 1e5 Pa and T = 298 K:
      sigma_ref^3 = (4.1e-10 m)^3 = 6.892100e-29 m^3
      kB T        = 1.380649e-23 * 298 = 4.1143340e-21 J
      z*          = 1e5 * 6.892100e-29 / 4.1143340e-21 = 1.675144e-3
    """
    red = reduce_tables(ff, 298.0)
    assert red.z_star(1.0e5) == pytest.approx(1.675144e-3, rel=1e-5)
    sigma3 = (4.1 * ANGSTROM_M) ** 3
    assert red.z_star(1.0e5) == pytest.approx(
        1.0e5 * sigma3 / (KB_J_PER_K * 298.0), rel=1e-12
    )
    # linear in fugacity, and an ideal gas at zero pressure has zero activity
    assert red.z_star(2.0e5) == pytest.approx(2.0 * red.z_star(1.0e5), rel=1e-12)
    assert red.z_star(0.0) == 0.0


def test_core_floor_is_defined_by_the_weakest_live_pair(ff):
    """P0.6: at the floor c, the worst pair's Boltzmann factor sits just under the
    1e-10 target, and the pair that binds it is the weakest well among pairs that
    can actually approach.

    For the MFM-300(In) type set the weakest live well is Kr-H:
      eps_KrH/kB = sqrt(166.4 * 7.64893) = 35.676 K
    and the floor solves exp(-4 (eps/T) (c^-12 - c^-6)) = 1e-10 at T = 298 K.
    """
    present = ("Xe", "Kr", "In", "C", "O", "H")
    scan = core_floor_scan(ff, 298.0, present=present)
    assert scan["binding_pair"] == "Kr-H"
    assert scan["eps_min_kelvin"] == pytest.approx(math.sqrt(166.4 * 7.64893), rel=1e-9)
    assert scan["pairs"][0]["boltzmann_at_floor"] < scan["target_boltzmann"]
    # and it is the largest such c on the grid: a hair looser already fails
    c = scan["c"]
    looser = c + 1.0e-4
    eps = scan["eps_min_kelvin"]
    assert (
        math.exp(-4.0 * (eps / 298.0) * (looser**-12 - looser**-6))
        > scan["target_boltzmann"]
    )


def test_core_floor_ignores_host_host_pairs(ff):
    """The framework is rigid: an Fe-Fe pair never approaches anything, so it must
    not drag the floor down. Fe is the weakest well in FF-A's framework by a wide
    margin (eps/kB = 6.54 K), so a scan that counted host-host pairs would report a
    visibly smaller c. Every scanned pair must have a guest on one side."""
    full = core_floor_scan(ff, 298.0)
    for row in full["pairs"]:
        assert set(row["pair"].split("-")) & {"Xe", "Kr"}, row["pair"]
    all_pairs = {p["pair"] for p in full["pairs"]}
    assert "Fe-Fe" not in all_pairs and "C-C" not in all_pairs


def test_core_floor_is_configuration_independent(ff):
    """The floor depends on the force field and T only. Same inputs, same number,
    every time: A5.4's dead-trial pruning leans on this."""
    a = core_floor_scan(ff, 298.0, present=("Xe", "Kr", "In", "C", "O", "H"))
    b = core_floor_scan(ff, 298.0, present=("Xe", "Kr", "In", "C", "O", "H"))
    assert a["c"] == b["c"]
    hotter = core_floor_scan(ff, 400.0, present=("Xe", "Kr", "In", "C", "O", "H"))
    assert hotter["c"] < a["c"]  # a hotter run tolerates a closer approach


def test_ff_b_loads_and_shares_the_guest_scale():
    """FF-B is the same guests on a UFF framework, so the reference scale, T* and
    rc* are identical and only the framework columns move. That is what makes it
    a force-field sensitivity test rather than a different experiment."""
    a = reduce_tables(load_forcefield(FF_A), 298.0)
    b = reduce_tables(load_forcefield(FF_B, allow_provisional=True), 298.0)
    assert b.sigma_ref == a.sigma_ref and b.eps_ref == a.eps_ref
    assert b.T_star == a.T_star and b.rc_star == a.rc_star
    ia, ib = a.types.index("C"), b.types.index("C")
    assert not np.isclose(a.eps_star[0, ia], b.eps_star[0, ib])

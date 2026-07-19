"""P0.1: the generated RASPA case and the branch_mc tables must be the same physics.

The whole anti-divergence claim is that one yaml feeds both codes. That claim is
only worth something if something checks it, so these tests read the emitted JSON
back and compare it against the force field and the reduced tables, rather than
against a copy of the numbers.

The RASPA3 conventions asserted here were read out of the RASPA3 sources
(forcefield.cpp), not remembered:
  - parameters are [epsilon/kB in K, sigma in A]
  - shifting happens only when TruncationMethod == "shifted"
  - CutOff lives in force_field.json and defaults to 12.0 A
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write as ase_write

from scripts.mof.raspa_input import (
    force_field_json,
    guest_json,
    simulation_json,
    write_case,
)
from scripts.mof.units import load_forcefield, peng_robinson, reduce_tables

FF_A = Path(__file__).resolve().parents[2] / "ff" / "forcefield_A.yaml"


@pytest.fixture(scope="module")
def ff():
    return load_forcefield(FF_A)


def test_self_interactions_are_the_yaml_verbatim(ff):
    """No conversion anywhere: the yaml stores epsilon/kB in K and sigma in A, and
    that is exactly RASPA3's parameter order and units. Xe must come out as
    [221.0, 4.1] and not, say, [4.1, 221.0]."""
    doc = force_field_json(ff)
    by_name = {e["name"]: e["parameters"] for e in doc["SelfInteractions"]}
    assert by_name["Xe"] == [221.0, 4.1]
    assert by_name["Kr"] == [166.4, 3.636]
    assert by_name["C"] == [47.8562, 3.47299]
    for name, (eps, sigma) in by_name.items():
        i = ff.index(name)
        assert eps == pytest.approx(ff.epsilon[i]) and sigma == pytest.approx(
            ff.sigma[i]
        )
        assert eps < sigma * 1000  # epsilon in K is the big one; order sanity


def test_truncation_and_tail_conventions(ff):
    """D1: 12 A, truncated, unshifted, no tail. RASPA3 shifts only on the exact
    string "shifted" (forcefield.cpp:207), so anything else is unshifted; the
    cutoff must be stated rather than left to the 12.0 default it happens to
    match."""
    doc = force_field_json(ff)
    assert doc["TruncationMethod"] != "shifted"
    assert doc["TailCorrections"] is False
    assert doc["CutOff"] == 12.0 == ff.rc
    assert doc["MixingRule"] == "Lorentz-Berthelot"


def test_guests_and_framework_are_labelled_correctly(ff):
    """`framework: true` is how RASPA knows what not to move. Guests are the low
    type ids on our side; the flag must follow the same split."""
    doc = force_field_json(ff)
    flags = {e["name"]: e["framework"] for e in doc["PseudoAtoms"]}
    assert flags["Xe"] is False and flags["Kr"] is False
    for t in ff.framework_types:
        assert flags[t] is True


def test_every_charge_is_zero(ff):
    """Not an approximation for Xe/Kr in a rigid framework: both guests are
    uncharged, so every Coulomb term in every compared energy is identically
    zero, and ChargeMethod None is exact."""
    doc = force_field_json(ff)
    assert all(e["charge"] == 0.0 for e in doc["PseudoAtoms"])


def test_present_filter_drops_types_the_framework_lacks(ff):
    doc = force_field_json(ff, present=("Xe", "Kr", "In", "C", "O", "H"))
    names = {e["name"] for e in doc["PseudoAtoms"]}
    assert names == {"Xe", "Kr", "In", "C", "O", "H"}
    assert not names & {"Al", "Ga", "Sc", "V", "Cr", "Fe"}  # other MFM-300 metals


def test_sources_travel_with_the_numbers(ff):
    """A parameter without its provenance is a number someone will have to
    re-derive. RASPA carries a source field; use it."""
    doc = force_field_json(ff)
    for entry in doc["SelfInteractions"] + doc["PseudoAtoms"]:
        assert entry["source"].strip()
    xe = next(e for e in doc["SelfInteractions"] if e["name"] == "Xe")
    assert "Table S1" in xe["source"]  # the anchor table, named in the input file


def test_guest_definition_is_a_single_site_at_the_origin(ff):
    doc = guest_json(ff, "Xe")
    assert doc["Type"] == "rigid"
    assert doc["pseudoAtoms"] == [["Xe", [0.0, 0.0, 0.0]]]
    assert doc["CriticalTemperature"] == pytest.approx(289.733)


def test_simulation_json_pins_charges_and_the_pr_fugacity(ff):
    sim = simulation_json(
        ff, "MFM-300-In", [2, 2, 2], 298.0, 1.0e5, {"Xe": 0.2, "Kr": 0.8}, 1000, 100
    )
    sys0 = sim["Systems"][0]
    assert sys0["ChargeMethod"] == "None"
    assert sys0["ExternalTemperature"] == 298.0
    assert sys0["ExternalPressure"] == 1.0e5
    comps = {c["Name"]: c for c in sim["Components"]}
    assert set(comps) == {"Xe", "Kr"}
    # The anchor uses Peng-Robinson, so pressure is NOT fugacity: phi must be the
    # PR value, and it must be stated rather than left for RASPA3 to recompute
    # (component.cpp:237 fills it in only when absent).
    fug = peng_robinson(ff, 298.0, 1.0e5, {"Xe": 0.2, "Kr": 0.8})
    assert comps["Xe"]["FugacityCoefficient"] == pytest.approx(fug.phi["Xe"])
    assert comps["Kr"]["FugacityCoefficient"] == pytest.approx(fug.phi["Kr"])
    assert 0.9 < comps["Xe"]["FugacityCoefficient"] < 1.0  # real gas, below ideal
    assert comps["Xe"]["MoleFraction"] == 0.2
    assert all(c["SwapProbability"] > 0 for c in comps.values())


def test_mol_fractions_must_sum_to_one(ff):
    with pytest.raises(ValueError, match="sum to 1"):
        simulation_json(ff, "X", [1, 1, 1], 298.0, 1e5, {"Xe": 0.3, "Kr": 0.3}, 10, 1)


def test_case_provenance_ties_both_sides_to_one_state_point(tmp_path, ff):
    """The generated case and the branch_mc reduced tables have to describe the
    same thermodynamic state, computed from the same yaml. z* per species is the
    number that has to agree, so the sidecar carries it next to the fugacity that
    produced it."""
    cif = tmp_path / "toy.cif"
    ase_write(str(cif), Atoms("C", positions=[[0, 0, 0]], cell=[6, 6, 6], pbc=True))
    meta = write_case(
        cif,
        ff,
        tmp_path / "case",
        298.0,
        1.0e5,
        {"Xe": 0.2, "Kr": 0.8},
        [2, 2, 2],
        1000,
        100,
    )

    red = reduce_tables(ff, 298.0)
    fug = peng_robinson(ff, 298.0, 1.0e5, {"Xe": 0.2, "Kr": 0.8})
    # z* comes from the PR fugacity, never from the partial pressure
    assert meta["state_point"]["z_star"]["Xe"] == pytest.approx(
        red.z_star(fug.fugacity_pa["Xe"])
    )
    assert meta["state_point"]["z_star"]["Kr"] == pytest.approx(
        red.z_star(fug.fugacity_pa["Kr"])
    )
    assert meta["state_point"]["z_star"]["Xe"] != pytest.approx(red.z_star(0.2e5))
    assert meta["state_point"]["T_star"] == pytest.approx(298.0 / 221.0)
    assert meta["forcefield"]["sha256"] == ff.sha256
    assert meta["status"]["guests"] == "confirmed"  # travels into the case

    case = tmp_path / "case"
    for f in (
        "force_field.json",
        "simulation.json",
        "Xe.json",
        "Kr.json",
        "toy.cif",
        "provenance.json",
    ):
        assert (case / f).exists(), f
    # and the emitted force field really is this yaml's, read back off disk
    doc = json.loads((case / "force_field.json").read_text())
    xe = next(e for e in doc["SelfInteractions"] if e["name"] == "Xe")
    assert xe["parameters"] == [
        float(ff.epsilon[ff.index("Xe")]),
        float(ff.sigma[ff.index("Xe")]),
    ]


def test_lorentz_berthelot_is_what_raspa_will_reconstruct(ff):
    """We hand RASPA self-interactions and a mixing rule, and it builds the cross
    terms itself. Our tables build them in units.py. Same rule, same inputs, so
    the cross terms must agree -- checked here on the pair that matters most, the
    guest-guest cross term of a binary run."""
    doc = force_field_json(ff)
    p = {e["name"]: e["parameters"] for e in doc["SelfInteractions"]}
    eps_raspa = math.sqrt(p["Xe"][0] * p["Kr"][0])
    sig_raspa = 0.5 * (p["Xe"][1] + p["Kr"][1])
    i, j = ff.index("Xe"), ff.index("Kr")
    assert eps_raspa == pytest.approx(ff.pair_epsilon()[i, j], rel=1e-12)
    assert sig_raspa == pytest.approx(ff.pair_sigma()[i, j], rel=1e-12)


# --------------------------------------------------- G1: Peng-Robinson fugacity


def test_peng_robinson_matches_raspa3s_own_algebra(ff):
    """Transcribed from RASPA3 (src/raspakit/utilities/equation_of_states.cpp), so
    check the intermediate quantities it defines rather than only the answer.

    Pure Xe at 298 K, w = 0.002:
      kappa = 0.37464 + 1.54226*0.002 - 0.26992*0.002^2 = 0.3777244
      Tr    = 298 / 289.733 = 1.0285332
      alpha = (1 + kappa (1 - sqrt(Tr)))^2 = 0.9893268
      a     = 0.45724 * alpha * (R*Tc)^2 / Pc
      b     = 0.07780 * R*Tc / Pc
    and at 1 bar the gas is nearly ideal, so Z and phi both sit just under 1.
    """
    R = 8.31446261815324
    Tc, Pc, w = 289.733, 5842000.0, 0.002
    kappa = 0.37464 + 1.54226 * w - 0.26992 * w * w
    assert kappa == pytest.approx(0.3777244, abs=1e-6)
    alpha = (1.0 + kappa * (1.0 - math.sqrt(298.0 / Tc))) ** 2
    assert alpha == pytest.approx(0.9893268, abs=1e-6)
    a = 0.45724 * alpha * (R * Tc) ** 2 / Pc
    b = 0.07780 * (R * Tc) / Pc

    got = peng_robinson(ff, 298.0, 1.0e5, {"Xe": 1.0})
    A = a * 1.0e5 / (R * 298.0) ** 2
    B = b * 1.0e5 / (R * 298.0)
    Z = got.compressibility
    # the cubic RASPA3 solves, evaluated at the root we took: must vanish
    resid = Z**3 + (B - 1.0) * Z**2 + (A - 3 * B**2 - 2 * B) * Z - (A * B - B**2 - B**3)
    assert resid == pytest.approx(0.0, abs=1e-12)
    assert 0.99 < Z < 1.0
    assert got.phi["Xe"] == pytest.approx(0.993986, abs=1e-5)


def test_fugacity_is_not_the_pressure(ff):
    """The whole reason G1 exists. At 10 bar phi_Xe = 0.9408, so treating pressure
    as fugacity misplaces the activity by 6% exactly where the anchor's isotherms
    are read. At 0.1 bar the gas is ideal to four decimals and it would not have
    mattered."""
    assert peng_robinson(ff, 298.0, 1.0e6, {"Xe": 1.0}).phi["Xe"] == pytest.approx(
        0.9408, abs=1e-3
    )
    assert peng_robinson(ff, 298.0, 1.0e4, {"Xe": 1.0}).phi["Xe"] == pytest.approx(
        0.9994, abs=1e-3
    )


def test_fugacity_coefficient_goes_to_one_at_zero_pressure(ff):
    """Every equation of state has to reproduce the ideal gas eventually."""
    assert peng_robinson(ff, 298.0, 1.0, {"Xe": 1.0}).phi["Xe"] == pytest.approx(
        1.0, abs=1e-6
    )
    assert peng_robinson(ff, 298.0, 1.0, {"Xe": 1.0}).compressibility == pytest.approx(
        1.0, abs=1e-6
    )


def test_mixture_fugacity_differs_from_the_pure_components(ff):
    """van der Waals mixing with kij = 0 still couples the species through Amix and
    Bmix, so a 20:80 Xe/Kr mixture is not two independent pure gases. Kr, being the
    bulk of the mixture, drags phi_Xe up toward its own."""
    mix = peng_robinson(ff, 298.0, 1.0e6, {"Xe": 0.2, "Kr": 0.8})
    pure_xe = peng_robinson(ff, 298.0, 1.0e6, {"Xe": 1.0})
    assert mix.phi["Xe"] != pytest.approx(pure_xe.phi["Xe"], abs=1e-4)
    assert pure_xe.phi["Xe"] < mix.phi["Xe"] < mix.phi["Kr"]
    # partial fugacities still sum sensibly below the total pressure
    assert sum(mix.fugacity_pa.values()) < 1.0e6


def test_fugacity_is_insensitive_to_the_acentric_factor(ff, tmp_path):
    """w is the one PR input the anchor never reported, and published values for
    noble gases scatter in the third decimal (RASPA's argon.def just uses 0.0).

    Measured rather than argued: across w from -0.002 to 0.008, phi_Xe at 10 bar
    moves by 3e-5, against a 6% departure from ideality. The unsourced number
    cannot reach any result we report, so the gap is closed.
    """
    import copy

    import yaml

    base = yaml.safe_load(FF_A.read_text())
    phis = []
    for w in (-0.002, 0.0, 0.002, 0.00363, 0.008):
        d = copy.deepcopy(base)
        d["eos"]["Xe"]["acentric_factor"] = w
        path = tmp_path / f"w{w}.yaml"
        path.write_text(yaml.safe_dump(d))
        phis.append(
            peng_robinson(load_forcefield(path), 298.0, 1.0e6, {"Xe": 1.0}).phi["Xe"]
        )
    assert max(phis) - min(phis) < 1e-4
    assert all(p == pytest.approx(0.9408, abs=1e-3) for p in phis)


def test_mol_fractions_must_sum_to_one_for_the_eos(ff):
    with pytest.raises(ValueError, match="sum to 1"):
        peng_robinson(ff, 298.0, 1.0e5, {"Xe": 0.5, "Kr": 0.4})

"""C6: the RASPA2 contenders of the CFCMC benchmark, generated from one yaml.

What is pinned:
  - no arm leaves RASPA2's default 10-trial CBMC swap in place by accident: the
    conventional arm writes k = 1 explicitly, CBMC writes its k, and identity
    changes are pinned to one first-bead trial in every arm;
  - CFCMC writes the lambda move, its histogram, and a nonzero equilibration
    phase (where Wang-Landau calibrates), and no plain swap;
  - mixtures get identity changes over every component;
  - the legacy writer (arm=None) is unchanged, because V0 and V1 ran on it.

Keyword spellings were read from RASPA2's own parser (src/input.c, iRASPA/RASPA2
master): SwapProbability, CFSwapLambdaProbability, CFLambdaHistogramSize,
CFWangLandauScalingFactor, NumberOfEquilibrationCycles, OptimizeTranslation,
NumberOfTrialPositionsForTheFirstBeadSwap (used by SwapAdd/RemoveAdsorbateMove,
mc_moves.c), NumberOfTrialPositionsForTheFirstBeadIdentityChange,
IdentityChangeProbability, NumberOfIdentityChanges, IdentityChangesList.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gcmc.forcefield.species_file import load_forcefield
from gcmc.framework.raspa2 import Raspa2Arm, simulation_input

FF_S = Path("ff/forcefield_A_shift.yaml")


def _text(arm, fracs=None):
    ff = load_forcefield(FF_S)
    return simulation_input(
        ff,
        "MFM-300-In",
        (2, 2, 3),
        298.0,
        1.0e6,
        fracs or {"Xe": 1.0},
        1000,
        100,
        arm=arm,
    )


def _value(text: str, key: str) -> list[str]:
    return [ln.split()[-1] for ln in text.splitlines() if ln.split()[:1] == [key]]


def test_legacy_writer_is_unchanged():
    t = _text(None)
    assert "NumberOfTrialPositionsForTheFirstBeadSwap" not in t
    assert _value(t, "TranslationProbability") == ["0.5"]
    assert _value(t, "SwapProbability") == ["1.0"]
    assert "NumberOfEquilibrationCycles" not in t


def test_conventional_arm_is_single_trial():
    t = _text(Raspa2Arm("conventional"))
    assert _value(t, "NumberOfTrialPositionsForTheFirstBeadSwap") == ["1"]
    assert _value(t, "NumberOfTrialPositionsForTheFirstBeadIdentityChange") == ["1"]
    assert _value(t, "SwapProbability") == ["1.0"]
    assert "CFSwapLambdaProbability" not in t
    with pytest.raises(ValueError, match="k_trials = 1"):
        Raspa2Arm("conventional", k_trials=10)


def test_cbmc_arm_writes_its_k():
    t = _text(Raspa2Arm("cbmc", k_trials=8))
    assert _value(t, "NumberOfTrialPositionsForTheFirstBeadSwap") == ["8"]
    with pytest.raises(ValueError, match="k_trials >= 2"):
        Raspa2Arm("cbmc", k_trials=1)


def test_cfcmc_arm_writes_lambda_move_and_calibration_phase():
    t = _text(Raspa2Arm("cfcmc", equilibration_cycles=5000, cf_wl_scaling=0.01))
    assert _value(t, "CFSwapLambdaProbability") == ["1.0"]
    assert _value(t, "CFLambdaHistogramSize") == ["20"]
    assert _value(t, "CFWangLandauScalingFactor") == ["0.01"]
    assert _value(t, "NumberOfEquilibrationCycles") == ["5000"]
    assert "SwapProbability " not in t.replace("CFSwapLambdaProbability", "")
    with pytest.raises(ValueError, match="Wang-Landau"):
        Raspa2Arm("cfcmc")


def test_mixture_gets_identity_changes_over_every_component():
    t = _text(
        Raspa2Arm("conventional", p_translation=2.0, p_exchange=2.0, p_identity=1.0),
        {"Xe": 0.2, "Kr": 0.8},
    )
    assert _value(t, "IdentityChangeProbability") == ["1.0", "1.0"]
    assert _value(t, "NumberOfIdentityChanges") == ["2", "2"]
    lists = [ln for ln in t.splitlines() if "IdentityChangesList" in ln]
    assert all(ln.split()[1:] == ["0", "1"] for ln in lists) and len(lists) == 2
    # single component: no identity block at all
    assert "IdentityChangeProbability" not in _text(Raspa2Arm("conventional"))


def test_unknown_arm_and_negative_weights_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        Raspa2Arm("gibbs")
    with pytest.raises(ValueError, match="non-negative"):
        Raspa2Arm("conventional", p_translation=-1.0)


def test_fugacity_route_sets_phi_to_one():
    """Torres-Knoop plot against fugacity: the pressure field carries f itself and
    no equation of state may enter (phi = 1); the default route keeps PR's phi."""
    ff = load_forcefield(FF_S)
    args = (ff, "MFM-300-In", (2, 2, 3), 298.0, 1.0e6, {"Xe": 1.0}, 1000, 100)
    t = simulation_input(*args, arm=Raspa2Arm("conventional"), phi={"Xe": 1.0})
    assert _value(t, "FugacityCoefficient") == ["1.0000000000"]
    assert _value(t, "ExternalPressure") == ["1000000.0"]
    pr = simulation_input(*args, arm=Raspa2Arm("conventional"))
    assert float(_value(pr, "FugacityCoefficient")[0]) < 0.99

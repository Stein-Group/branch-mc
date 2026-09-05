"""A legacy trajectory's only metadata was its filename, so migration mines the
run parameters back out of it. These pin the token grammar against two real
names (the second exercises the 2D + aux-potential + gamma/eta tail).
"""

from __future__ import annotations

from gcmc.run.migrate import _parse_name


def test_parse_minimal_3d_name():
    got = _parse_name("mu_3.0_r_1.5_rhok_0.5_k_30_2d_False_.extxyz")
    assert got["chem_pot"] == 3.0
    assert got["cavity_radius"] == 1.5
    assert got["trial_density"] == 0.5
    assert got["no_trials"] == 30
    assert got["two_dim"] is False
    assert got["aux_pot"] is False  # defaulted when the token is absent


def test_parse_full_2d_aux_name():
    got = _parse_name(
        "mu_2.0_r_2.5_rhok_0.4_k_20_2d_True_aux_pot_True_gammamax_5.0_deltaeta_0.05_.extxyz"
    )
    assert got["chem_pot"] == 2.0
    assert got["two_dim"] is True
    assert got["aux_pot"] is True
    assert got["gamma_max"] == 5.0
    assert got["delta_eta"] == 0.05

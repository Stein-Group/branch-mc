"""Item 17: potential_type is derived from three orthogonal axes.

The flat engine tag used to be assigned by a chain of elif branches and stored as
a field. It is now derived from (potential, bias, sampler) by one function, and
exposed as a property so run fingerprints keep reading potential_type unchanged.
This pins the full mapping so the split cannot silently change a dispatch key.
"""

from gcmc.run.sim_setup import _derive_potential_type, from_params

# (potential, bias, sampler) -> expected flat tag
CASES = {
    ("hard_sphere", None, "metropolis"): "hard_sphere",
    ("hard_sphere", None, "rosenbluth"): "hard_sphere",
    ("hard_sphere", None, "branch"): "hard_sphere",
    ("hard_sphere", None, "channel"): "hard_sphere",
    ("hard_sphere", "pmf", "rosenbluth"): "aux",
    ("hard_sphere", "psi6", "reweight"): "reweight",
    ("lennard_jones", None, "metropolis"): "lj",
    ("lennard_jones", None, "rosenbluth"): "lj",
    ("lennard_jones", None, "branch"): "ljbranch",
    ("lennard_jones", None, "channel"): "ljbranch",
    ("lennard_jones", "pmf", "rosenbluth"): "ljaux",
    # bias=pmf wins over the branch sampler, matching the old elif order
    ("lennard_jones", "pmf", "branch"): "ljaux",
}


def test_derivation_table():
    for (pot, bias, samp), expected in CASES.items():
        got = _derive_potential_type(pot, bias, samp)
        assert got == expected, f"{(pot, bias, samp)} -> {got}, expected {expected}"


def test_config_property_matches_and_fields_are_orthogonal():
    cfg = from_params(
        {
            "potential": {"main": "lennard_jones", "T": 1.0},
            "ensemble": {"mu": -3.0, "dimension": 3, "box": 10.0},
            "sampler": {
                "type": "branch",
                "trials": 20,
                "trial_box_side": 3.0,
            },
            "engine": {"depth": 1},
            "run": {},
        }
    )
    assert cfg.potential == "lennard_jones"
    assert cfg.bias is None
    assert cfg.sampler == "branch"
    assert cfg.potential_type == "ljbranch"


if __name__ == "__main__":
    test_derivation_table()
    test_config_property_matches_and_fields_are_orthogonal()
    print("PASS test_item17_potential_type")

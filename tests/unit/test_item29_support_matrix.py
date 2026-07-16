"""Original item 29: the documented support matrix is enforced.

Each disallowed sampler/potential/ensemble/parallel mix must raise ValueError,
and a few clearly supported mixes must build. This guards the matrix in the
module docstring against drift.
"""

from gcmc.sim_setup import from_params


def _raises(p):
    try:
        from_params(p)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {p}")


def test_reweight_constraints():
    # reweight is hard-core, 2D, serial only
    _raises(
        {
            "potential": {"main": "lennard_jones", "T": 1.0},
            "ensemble": {"mu": -2.0, "dimension": 2, "box": 30.0},
            "sampler": {"type": "reweight"},
            "run": {},
        }
    )
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": -2.0, "dimension": 3, "box": 30.0},
            "sampler": {"type": "reweight"},
            "run": {},
        }
    )
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": -2.0, "dimension": 2, "box": 30.0},
            "sampler": {"type": "reweight"},
            "run": {"kind": "parallel"},
        }
    )


def test_channel_constraints():
    base_smp = {
        "type": "channel",
        "trial_density": 2.0,
        "depth": 1,
        "channel_length": 6.0,
        "channel_width": 1.5,
    }
    # 2D
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 2.0, "dimension": 2, "box": 12.0},
            "sampler": base_smp,
            "run": {},
        }
    )
    # parallel
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
            "sampler": base_smp,
            "run": {"kind": "parallel"},
        }
    )
    # depth 0
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
            "sampler": {**base_smp, "depth": 0},
            "run": {},
        }
    )


def test_qa_constraints():
    # QA rejects metropolis
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"type": "QA", "mu": -2.0, "dimension": 3, "box": 12.0},
            "sampler": {"type": "metropolis"},
            "run": {"matrix_config": "/tmp/m.xyz"},
        }
    )
    # QA rejects plain LJ (needs LJ+branch)
    _raises(
        {
            "potential": {"main": "lennard_jones", "T": 1.0},
            "ensemble": {"type": "QA", "mu": -2.0, "dimension": 3, "box": 12.0},
            "sampler": {
                "type": "rosenbluth",
                "trials": 20,
                "trial_box_side": 3.0,
                "depth": 0,
            },
            "run": {"matrix_config": "/tmp/m.xyz"},
        }
    )


def test_confined_constraints():
    # confined is 3D only
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {
                "type": "confined",
                "mu": -2.0,
                "dimension": 2,
                "box": 12.0,
                "pore": {"geometry": "slit", "H": 4.0},
            },
            "sampler": {"type": "metropolis"},
            "run": {},
        }
    )


def test_supported_combinations_build():
    ok = [
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": -2.0, "dimension": 3, "box": 12.0},
            "sampler": {"type": "metropolis"},
            "run": {},
        },
        {
            "potential": {"main": "lennard_jones", "T": 1.2},
            "ensemble": {"mu": -3.0, "dimension": 3, "box": 12.0},
            "sampler": {
                "type": "branch",
                "trials": 20,
                "trial_box_side": 3.0,
                "depth": 1,
            },
            "run": {},
        },
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
            "sampler": {
                "type": "channel",
                "trial_density": 2.0,
                "depth": 1,
                "channel_length": 6.0,
                "channel_width": 1.5,
            },
            "run": {},
        },
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": -2.0, "dimension": 2, "box": 30.0},
            "sampler": {"type": "reweight"},
            "run": {},
        },
    ]
    for p in ok:
        from_params(p)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("PASS test_item29_support_matrix")

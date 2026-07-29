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
            "run": {"kind": "parallel", "n_cell": 8},
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
            "run": {"kind": "parallel", "n_cell": 8},
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


def _parallel_run(dim):
    # cell_length + buffer must fit the box; 8 + 2 <= 30 grids into 27/9 cells.
    return {
        "kind": "parallel",
                "sampling_window": 50,
        "cell_length": 8.0,
        "buffer": 2.0,
    }


def test_parallel_support_matrix():
    # The 2D-only parallel constraint was dropped by decision (cell_unify_plan.md
    # U0, GO 2026-07-14). metropolis, rosenbluth, and branch now build under
    # parallel in both 2D and 3D; from_params must accept every one of those
    # six. This is the case the old test_item29 never probed directly, so the
    # relaxation slipped past the unit tier and only check_parallel caught it.
    samplers = {
        "metropolis": {"type": "metropolis"},
        "rosenbluth": {
            "type": "rosenbluth",
            "trials": 20,
            "trial_box_side": 3.0,
            "depth": 0,
        },
        "branch": {"type": "branch", "trials": 20, "trial_box_side": 3.0, "depth": 1},
    }
    for dim in (2, 3):
        for smp in samplers.values():
            p = {
                "potential": {"main": "hard_core"},
                "ensemble": {"mu": 1.0, "dimension": dim, "box": 30.0},
                "sampler": smp,
                "run": _parallel_run(dim),
            }
            from_params(p)  # must not raise

    # Still serial-only under parallel: reweight (no psi6 field in the
    # decomposition), channel (serial only), confined (walls dropped by the
    # decomposition).
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 1.0, "dimension": 2, "box": 30.0},
            "sampler": {"type": "reweight"},
            "run": _parallel_run(2),
        }
    )
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 30.0},
            "sampler": {
                "type": "channel",
                "trial_density": 2.0,
                "depth": 1,
                "channel_length": 6.0,
                "channel_width": 1.5,
            },
            "run": _parallel_run(3),
        }
    )
    _raises(
        {
            "potential": {"main": "hard_core"},
            "ensemble": {
                "type": "confined",
                "mu": 1.0,
                "dimension": 3,
                "box": 30.0,
                "pore": {"geometry": "slit", "H": 6.0},
            },
            "sampler": {"type": "metropolis"},
            "run": _parallel_run(3),
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

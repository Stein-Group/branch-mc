"""expand_sweep turns list-valued axes into the Cartesian product of concrete
runs with readable id suffixes; cores_per_run reports the core cost of one run.
Together they set the sweep size and the core budget the launcher enforces.
"""

from __future__ import annotations

import pytest

from gcmc.launcher import cores_per_run, expand_sweep

_BASE = {
    "potential": {"main": "hard_core"},
    "sampler": {"type": "branch", "depth": 1, "trials": 10, "trial_box_side": 3.0},
    "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
    "run": {"seed": 1},
}


def _parallel_base(**run):
    """A parallel base. One cell is one core, so cores-per-run is the cell
    count and is never stated separately."""
    return {
        **_BASE,
        "ensemble": {"mu": 2.0, "dimension": 3, "box": 30.0},
        "run": {"kind": "parallel", "seed": 1, **run},
    }


def test_expand_sweep_two_by_two_gives_four_labelled_combos():
    sweep = {
        **_BASE,
        "ensemble": {"mu": [2.0, 3.0], "dimension": 3, "box": 12.0},
        "sampler": {
            "type": "branch",
            "depth": [1, 2],
            "trials": 10,
            "trial_box_side": 3.0,
        },
    }
    combos, labels = expand_sweep(sweep)
    assert len(combos) == 4
    assert labels == ["mu2.0_depth1", "mu2.0_depth2", "mu3.0_depth1", "mu3.0_depth2"]
    # every combo is scalar-valued on the swept axes
    assert {c["ensemble"]["mu"] for c in combos} == {2.0, 3.0}
    assert {c["sampler"]["depth"] for c in combos} == {1, 2}


def test_expand_sweep_no_list_axis_is_a_single_run():
    combos, labels = expand_sweep(_BASE)
    assert len(combos) == 1 and labels == ["single"]


def test_cores_per_run_serial_vs_parallel():
    assert cores_per_run(_BASE) == 1
    assert cores_per_run(_parallel_base(n_cell=4)) == 4
    # a per-axis submission costs the product
    assert cores_per_run(_parallel_base(n_cell=[2, 2, 2])) == 8
    # manual geometry: the count falls out of it, and still equals the cores
    assert cores_per_run(_parallel_base(cell_length=8.0, buffer=2.0)) == 27


def test_cores_per_run_refuses_n_cores_on_a_parallel_base():
    """One cell is one core, so a per-run core count cannot be stated (D4)."""
    with pytest.raises(ValueError, match="n_cores does not apply"):
        cores_per_run(_parallel_base(n_cell=4, n_cores=64))

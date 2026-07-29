"""expand_sweep turns list-valued axes into the Cartesian product of concrete
runs with readable id suffixes; cores_per_run reports the core cost of one run.
Together they set the sweep size and the core budget the launcher enforces.
"""

from __future__ import annotations

import pytest

from gcmc.launcher import cores_per_run, expand_sweep, plan

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


def test_cores_per_run_works_on_a_swept_base():
    """REGRESSION: cores_per_run asks the config layer for the cell count, and a
    swept base still holds lists (run.seed: [1, 2, ...]) which from_params
    rightly refuses as an unexpanded sweep. It must ask an expanded combo, or
    every swept parallel launch dies at the budget check."""
    base = _parallel_base(n_cell=8)
    base["run"]["seed"] = [1, 2, 3]
    assert cores_per_run(base) == 8


def test_a_per_axis_n_cell_is_not_mistaken_for_a_sweep_axis():
    """n_cell accepts a per-axis list, and expand_sweep expands list-valued axes.
    n_cell is deliberately not among them, so [2, 2, 2] stays one run of 8 cells
    rather than three runs of 2."""
    base = _parallel_base(n_cell=[2, 2, 2])
    combos, labels = expand_sweep(base)
    assert len(combos) == 1 and labels == ["single"]
    assert combos[0]["run"]["n_cell"] == [2, 2, 2]
    assert cores_per_run(base) == 8


def test_n_cores_is_the_sweep_budget_and_is_allowed_on_a_parallel_base():
    """REGRESSION: the two checks used to contradict each other. Omitting the
    budget made main() say "pass --cores N or set run.n_cores", and supplying
    run.n_cores made cores_per_run refuse it -- impossible advice for a parallel
    base. run.n_cores is the sweep budget (combinations x cores-per-run), a
    different quantity from cores-per-run, which is the cell count."""
    base = _parallel_base(n_cell=8, n_cores=64)
    assert cores_per_run(base) == 8
    _, _, per, conc = plan(base, 64)
    assert per == 8 and conc == 8


def test_the_budget_error_names_the_cell_count():
    base = _parallel_base(n_cell=27)
    base["run"]["seed"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="One cell is one core"):
        plan(base, 8)

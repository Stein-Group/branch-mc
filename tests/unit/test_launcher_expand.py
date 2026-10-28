"""expand_sweep turns list-valued axes into the Cartesian product of concrete
runs with readable id suffixes; cores_per_run reports the core cost of one run.
Together they set the sweep size and the core budget the launcher enforces.
"""

from __future__ import annotations

import pytest

from gcmc.run.launcher import cores_per_run, expand_sweep

_BASE = {
    "potential": {"main": "hard_core"},
    "sampler": {"type": "branch", "trials": 10, "trial_box_side": 3.0},
    "engine": {"depth": 1},
    "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
    "run": {"seed": 1},
}


def test_expand_sweep_two_by_two_gives_four_labelled_combos():
    sweep = {
        **_BASE,
        "ensemble": {"mu": [2.0, 3.0], "dimension": 3, "box": 12.0},
        "sampler": {
            "type": "branch",
            "trials": 10,
            "trial_box_side": 3.0,
        },
        "engine": {"depth": [1, 2]},
    }
    combos, labels = expand_sweep(sweep)
    assert len(combos) == 4
    assert labels == ["mu2.0_depth1", "mu2.0_depth2", "mu3.0_depth1", "mu3.0_depth2"]
    # every combo is scalar-valued on the swept axes
    assert {c["ensemble"]["mu"] for c in combos} == {2.0, 3.0}
    assert {c["engine"]["depth"] for c in combos} == {1, 2}


def test_expand_sweep_no_list_axis_is_a_single_run():
    combos, labels = expand_sweep(_BASE)
    assert len(combos) == 1 and labels == ["single"]


def _parallel(run):
    # box 30, not _BASE's 12: at box 12 an 8-cell plan gives a 3.0 cell edge,
    # which does not exceed twice the cavity radius and is refused by cell_plan.
    # The core-count question is about the planner arithmetic, not the geometry,
    # so use a box where the plan is feasible.
    return {
        **_BASE,
        "ensemble": {**_BASE["ensemble"], "box": 30.0},
        "run": {"kind": "parallel", **run},
    }


def test_cores_per_run_serial_is_one():
    assert cores_per_run(_BASE) == 1


def test_cores_per_run_parallel_is_the_cell_count():
    """One cell is one core (parallel_input_plan.md D4/D6), so cores-per-run is
    the cell count and there is no separate key for it. cores_per_run asks
    from_params rather than reconstructing the number, so the launcher and the
    engine cannot disagree -- which also means it needs a whole valid params
    file, not a bare run block."""
    assert cores_per_run(_parallel({"n_cell": 8})) == 8
    assert cores_per_run(_parallel({"n_cell": [3, 2, 1]})) == 6


def test_cores_per_run_of_a_swept_base_asks_the_first_combo():
    """A base with a list axis is not a single run; from_params rightly refuses
    an unexpanded sweep, so cores_per_run expands first."""
    assert cores_per_run(_parallel({"n_cell": 8, "seed": [1, 2, 3]})) == 8


def test_the_model_axis_sweeps_species_files_and_labels_them_by_stem():
    """The k_ij demixing series is a sweep over FILES, and its run directories
    have to be readable.

    Two things would go wrong without the file-axis handling: a list-valued
    `potential.species_file` would reach `from_params` unexpanded (the same
    failure the `trials` axis had), and a label built from the raw value would
    put a path -- slashes, dots and all -- into a directory name.
    """
    sweep = {
        **_BASE,
        "potential": {
            "main": "lennard_jones",
            "T": 1.0,
            "species_file": [
                "ff/species_binary_k0.yaml",
                "ff/species_binary_k0.5.yaml",
            ],
        },
    }
    combos, labels = expand_sweep(sweep)
    assert labels == ["modelbinary_k0", "modelbinary_k0.5"]
    assert [c["potential"]["species_file"] for c in combos] == [
        "ff/species_binary_k0.yaml",
        "ff/species_binary_k0.5.yaml",
    ]


def test_a_single_species_file_is_not_a_sweep():
    """One model is one run: an axis with a single value is applied and kept out
    of the label, exactly as every other axis is."""
    base = {
        **_BASE,
        "potential": {
            "main": "lennard_jones",
            "T": 1.0,
            "species_file": "ff/species_binary_k0.yaml",
        },
    }
    combos, labels = expand_sweep(base)
    assert len(combos) == 1 and labels == ["single"]


def test_the_start_axis_sweeps_warm_starts_and_composes_with_the_model_axis():
    """Six models x three starts is one launch, and the labels stay readable.

    The start is a swept axis because a composition at fixed mu can be trapped:
    the only evidence that a reported composition is the stationary one is that
    arms begun from opposite ends met. That makes "which start" part of the
    experiment's design rather than a fixed input, and it has to compose with
    the model axis so one base file states the whole thing.
    """
    sweep = {
        **_BASE,
        "potential": {
            "main": "lennard_jones",
            "T": 1.0,
            "species_file": ["ff/a.yaml", "ff/b.yaml"],
        },
        "run": {
            "seed": 1,
            "initial_config": ["run-files/x0.9.extxyz", "run-files/x0.1.extxyz"],
        },
    }
    combos, labels = expand_sweep(sweep)
    assert len(combos) == 4
    assert labels == [
        "modela_startx0.9",
        "modela_startx0.1",
        "modelb_startx0.9",
        "modelb_startx0.1",
    ]


def test_the_grid_token_on_the_start_axis_is_not_a_sweep():
    """`initial_config` takes the literal token "grid" as well as a path, and one
    value on an axis is one run -- the token must not be expanded into
    characters or land in a label."""
    base = {**_BASE, "run": {"seed": 1, "initial_config": "grid"}}
    combos, labels = expand_sweep(base)
    assert len(combos) == 1 and labels == ["single"]
    assert combos[0]["run"]["initial_config"] == "grid"


def test_the_engine_axis_sweeps_and_labels_the_two_dynamics():
    """The engine is a swept axis, so one launch produces both arms of a
    tau_N comparison at one state point rather than two launches that have to
    be kept apart by hand."""
    sweep = {
        **_BASE,
        "sampler": {
            "type": "channel",
            "trial_density": 1.0,
            "channel_length": 5.0,
            "channel_width": 1.5,
        },
        "engine": {"type": ["metropolis_dynamics", "glauber_dynamics"], "depth": 1},
    }
    combos, labels = expand_sweep(sweep)
    assert len(combos) == 2
    assert labels == ["enginemetropolis_dynamics", "engineglauber_dynamics"]
    assert [c["engine"]["type"] for c in combos] == [
        "metropolis_dynamics",
        "glauber_dynamics",
    ]


def test_a_single_engine_is_applied_but_stays_out_of_the_label():
    """Adding the axis must not rename any existing run directory: an axis with
    one value is applied and contributes no id fragment."""
    base = {
        **_BASE,
        "sampler": {**_BASE["sampler"], "engine": "metropolis_dynamics"},
    }
    combos, labels = expand_sweep(base)
    assert len(combos) == 1 and labels == ["single"]
    assert combos[0]["sampler"]["engine"] == "metropolis_dynamics"


def test_starts_that_share_a_file_name_still_get_separate_directories():
    """The start sweep a production run actually writes is a list of per-run
    output folders, and those folders all hold a file with the SAME name.

    `run.initial_config` is not identity-bearing on purpose, so the run id is
    the same across the arms and the sweep label is the only thing that puts
    them in different directories. A fragment built from the stem alone would
    collapse the whole sweep onto one directory, where the arms would take
    turns overwriting one checkpoint and one log -- a sweep that looks like it
    ran and is one chain. So the fragment reaches back into the path until the
    arms separate.
    """
    sweep = {
        **_BASE,
        "run": {
            "seed": 1,
            "initial_config": [
                "equilibrated/rep01/final.xyz",
                "equilibrated/rep02/final.xyz",
                "equilibrated/rep03/final.xyz",
            ],
        },
    }
    combos, labels = expand_sweep(sweep)
    assert len(combos) == 3
    assert labels == ["startrep01_final", "startrep02_final", "startrep03_final"]
    assert len(set(labels)) == 3
    assert [c["run"]["sweep_label"] for c in combos] == labels


def test_the_start_fragment_stays_the_stem_when_the_stems_already_differ():
    """The fragment grows only as far as it has to, so every sweep whose files
    are already distinguishable keeps the directory names it had."""
    sweep = {
        **_BASE,
        "run": {
            "seed": 1,
            "initial_config": ["deep/a/b/x0.9.extxyz", "other/c/x0.1.extxyz"],
        },
    }
    _combos, labels = expand_sweep(sweep)
    assert labels == ["startx0.9", "startx0.1"]


def test_a_start_list_reaches_as_far_back_as_the_paths_differ():
    """Two levels of shared tail is not a special case -- the depth is whatever
    separates the axis, and it is the same depth for every arm so the names
    stay comparable."""
    sweep = {
        **_BASE,
        "run": {
            "seed": 1,
            "initial_config": [
                "/scratch/runA/traj/final.xyz",
                "/scratch/runB/traj/final.xyz",
            ],
        },
    }
    _combos, labels = expand_sweep(sweep)
    assert labels == ["startrunA_traj_final", "startrunB_traj_final"]


def test_the_grid_token_composes_with_paths_on_the_start_axis():
    """ "grid" is a token rather than a path and has one component, so it must
    survive a sweep whose other arms are deep paths."""
    sweep = {
        **_BASE,
        "run": {
            "seed": 1,
            "initial_config": ["grid", "eq/rep01/final.xyz", "eq/rep02/final.xyz"],
        },
    }
    _combos, labels = expand_sweep(sweep)
    assert labels == ["startgrid", "startrep01_final", "startrep02_final"]


def test_two_combinations_may_never_share_a_label():
    """A repeated value on an axis is not a two-arm sweep, it is one arm
    launched twice into one directory. Silence there is the worst outcome: the
    launcher reports the runs it started and the second quietly writes over the
    first."""
    sweep = {
        **_BASE,
        "run": {"seed": 1, "initial_config": ["eq/a.xyz", "eq/a.xyz"]},
    }
    with pytest.raises(ValueError, match=r"shared by more than one combination"):
        expand_sweep(sweep)


def test_a_single_run_may_not_carry_the_whole_start_list():
    """An unexpanded list must be refused by name. Passed through, it reaches
    the start resolver as a list where a path belongs and fails there with an
    error that names neither the key nor the launcher."""
    from gcmc.run import sim_setup

    sweep = {
        **_BASE,
        "run": {"seed": 1, "initial_config": ["eq/a.xyz", "eq/b.xyz"]},
    }
    with pytest.raises(ValueError, match=r"run\.initial_config is a list"):
        sim_setup.from_params(sweep)

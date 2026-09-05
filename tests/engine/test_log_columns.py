"""The run log carries the columns the run can fill, and the user can narrow it.

THE DEFECT THIS CLOSES, measured on a cluster job rather than reasoned about. An
semi-grand multi-species run wrote 27 columns of which twelve -- add, remove,
translate and the entire A/R clique block -- were structurally zero, because a
fixed-N ensemble forces the
repack move and never proposes any of them. The counters that carry what the run
actually did (`repack_*`, and the `P_*` clique triple the engine has always
accumulated) had no column at all. The log read as a chain doing nothing for 810
windows while its composition went from 50:50 to 4:96.

So the rule is: a column is written when the run CAN put a nonzero value in it.
The tests below pin both directions of that -- the columns that appear and the
ones that must NOT -- because a rule enforced in one direction only is how the
zero columns got there in the first place.
"""

from __future__ import annotations

import json

import pytest

from gcmc.run import runner
from gcmc.run.sim_setup import from_params

BOX = 8.0
MODEL = "ff/species_binary_k0.5.yaml"

_MUVT = {
    "potential": {"main": "lennard_jones", "T": 1.5, "species_file": MODEL},
    "sampler": {
        "type": "channel",
        "depth": 1,
        "trial_density": 1.0,
        "channel_width": 1.5,
        "channel_length": 3.0,
    },
    "ensemble": {
        "type": "muVT",
        "dimension": 3,
        "box": BOX,
        "mu": {"A": -3.4, "B": -2.55},
    },
    "run": {
        "kind": "single",
        "seed": 5,
        "traj_every": 0,
        "target_cycles": 2400,
        "sampling_window": 800,
        "checkpoint_every": 5,
        "initial_config": "grid",
    },
}


def _nvt(**run_over):
    # The fixed-N arm of the log is the SEMI-GRAND ensemble: it states a state
    # point (the activity ratio moves composition) and its only move is the
    # repack. Its type used to be spelled "NVT"; canonical NVT is a different
    # column set and is covered in test_canonical_ensemble.py.
    p = json.loads(json.dumps(_MUVT))
    p["ensemble"] = {
        "type": "semigrand",
        "dimension": 3,
        "box": BOX,
        "N": 300,
        "mu": {"A": 0.0, "B": 0.0},
    }
    p["run"].update(run_over)
    return p


def _header(params, tmp_path, name):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(params))
    _state, paths = runner.run(str(path), root=str(tmp_path / name))
    lines = [ln for ln in paths.log.read_text().splitlines() if ln.strip()]
    names = lines[0].lstrip("#").split()
    rows = [ln.split() for ln in lines[1:]]
    assert rows, "the run logged no rows, so nothing below is being tested"
    assert all(len(r) == len(names) for r in rows)
    return names, rows


def test_an_nvt_run_logs_repack_and_not_the_moves_it_never_proposes(tmp_path):
    names, rows = _header(_nvt(), tmp_path, "nvt")

    assert {"rpk_acc", "rpk_att", "rpk_stp"} <= set(names), (
        f"a semigrand log has no repack columns: {names}. The repack is the ONLY move "
        "that ensemble proposes"
    )
    assert {"clqP_n", "clqP_pos", "clqP_new"} <= set(names), (
        "the engine accumulates a P_* clique triple for the repack and the log "
        "does not write it"
    )
    for absent in (
        "add_acc",
        "add_att",
        "add_stp",
        "rm_acc",
        "rm_stp",
        "mv_acc",
        "mv_stp",
        "swp_acc",
        "clqA_n",
        "clqR_n",
    ):
        assert absent not in names, (
            f"a semigrand log declares {absent!r}, which that ensemble can never make "
            "nonzero -- twelve such columns is what made a real run unreadable"
        )

    # ...and the columns it DOES declare are not themselves a wall of zeros,
    # which is the reachable-value half: a rule that only removed columns could
    # satisfy everything above by removing the wrong ones.
    stp = [int(r[names.index("rpk_stp")]) for r in rows]
    assert min(stp) > 0, f"repack proposals per window were {stp}"


def test_a_single_species_muvt_header_is_the_historical_layout(tmp_path):
    """The default narrows only where a column could not vary, so the layout
    every existing log and reader knows is untouched."""
    p = {
        "potential": {"main": "hard_core"},
        "sampler": {"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 3.0},
        "ensemble": {"mu": 3.0, "dimension": 3, "box": 12.0, "type": "muVT"},
        "run": {
            "kind": "single",
            "radius": 0.5,
            "seed": 1,
            "traj_every": 0,
            "target_cycles": 2000,
            "sampling_window": 1000,
            "checkpoint_every": 5,
        },
    }
    names, _rows = _header(p, tmp_path, "hs")
    assert "#" + " ".join(names) == "#" + " ".join(
        runner.LOG_HEADER.lstrip("#").split()
    )
    assert "swp_acc" not in names and "rpk_acc" not in names


def test_a_typed_muvt_run_logs_the_swap(tmp_path):
    names, rows = _header(_MUVT, tmp_path, "typed")
    assert {"swp_acc", "swp_stp"} <= set(names), (
        f"a typed muVT log has no swap columns: {names}. The identity swap is a "
        "fifth of the move budget there"
    )
    assert max(int(r[names.index("swp_stp")]) for r in rows) > 0


@pytest.mark.parametrize(
    ("key", "value", "gone", "kept"),
    [
        ("log_drop", ["clique", "clique_acc"], "clqP_n", "rpk_stp"),
        ("log_drop", ["state"], "density", "N"),
        ("log_columns", ["moves"], "density", "rpk_stp"),
        ("log_columns", [], "rpk_stp", "N"),
    ],
)
def test_the_selection_narrows_the_log(key, value, gone, kept, tmp_path):
    names, _rows = _header(
        _nvt(**{key: value}), tmp_path, f"sel_{key}_{len(value)}_{gone}"
    )
    assert gone not in names, f"{key}={value} left {gone!r} in {names}"
    assert kept in names, f"{key}={value} removed {kept!r}, which it should keep"


def test_the_forced_groups_survive_any_selection(tmp_path):
    """`core` places a row in time and `species` is the entire output of a
    mixture run. Neither can be dropped, and asking is an error rather than a
    silently ignored request."""
    names, _rows = _header(_nvt(log_columns=[]), tmp_path, "forced")
    assert {"MC_cycle(M)", "N", "wall_s", "N_A", "N_B"} <= set(names)

    with pytest.raises(ValueError, match="cannot be dropped"):
        from_params(_nvt(log_drop=["species"]))


def test_a_typo_and_a_double_spelling_are_refused(tmp_path):
    """Both at parse time, before the run directory exists (M15).

    A typo that silently kept the default would be a setting that appears to
    work and does nothing -- the failure mode this codebase refuses -- and two
    ways to say one thing is how a params file ends up meaning something its
    author did not intend.
    """
    with pytest.raises(ValueError, match="unknown column group"):
        from_params(_nvt(log_drop=["cliques"]))
    with pytest.raises(ValueError, match="two ways to say one thing"):
        from_params(_nvt(log_columns=["state"], log_drop=["moves"]))
    with pytest.raises(ValueError, match="a LIST of column-group names"):
        from_params(_nvt(log_drop="clique"))

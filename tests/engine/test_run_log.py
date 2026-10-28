"""The run log: raw per-window columns, the running energy, and the file format.

Every column is a raw count or sum that cannot be rebuilt from the others or
from the params file, and a column appears only where the run can change it.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from gcmc.analysis.energy import configuration_energy
from gcmc.run import io, runner
from gcmc.run.sim_setup import from_params

BOX = 8.0
TYPED = "ff/species_binary_k0.5.yaml"
CHANNEL = {
    "type": "channel",
    "trial_density": 1.0,
    "channel_width": 1.5,
    "channel_length": 3.0,
}
RUN = {
    "kind": "single",
    "seed": 5,
    "sampling_window": 60,
    "target_cycles": 240,
    "checkpoint_every": 2,
    "initial_config": "grid",
}

TRANSLATE = ["translate_tried", "translate_accepted", "translate_sum_dr2"]
EXCHANGE = [
    "insert_tried",
    "insert_evaluated",
    "insert_accepted",
    "delete_tried",
    "delete_evaluated",
    "delete_accepted",
]
LADDER = [
    "ladder_unchanged",
    "ladder_sum_abs_dN",
    "ladder_sum_dN2",
    "ladder_sum_k_in",
    "ladder_max_k",
    "ladder_sum_pool",
    "ladder_infeasible",
    "ladder_cut_kmax",
    "ladder_cut_pool",
]


def _bulk(engine: dict, sampler: dict, ensemble: dict, **over) -> dict:
    p = {
        "potential": {"main": "lennard_jones", "T": 1.5},
        "engine": engine,
        "sampler": sampler,
        "ensemble": {"dimension": 3, "box": BOX, **ensemble},
        "run": {**RUN, "upper_cutoff": 2.5},
    }
    for key, value in over.items():
        p[key] = value
    return p


def _columns(params: dict, tmp_path) -> tuple[list[str], dict[str, np.ndarray]]:
    _state, paths = runner.run(copy.deepcopy(params), root=str(tmp_path / "runs"))
    return io.log_columns(paths.log), io.read_log(paths.log)


def test_single_particle_metropolis(tmp_path):
    p = _bulk({}, {"type": "metropolis"}, {"mu": -2.0})
    cols, _ = _columns(p, tmp_path)
    assert cols == [
        "MC_cycle",
        "N",
        "U",
        *TRANSLATE,
        "insert_tried",
        "insert_accepted",
        "delete_tried",
        "delete_accepted",
        "wall_s",
    ]


def test_cavity_exchange_logs_evaluated_counts_and_the_body(tmp_path):
    p = _bulk({"depth": 1}, CHANNEL, {"mu": -2.0})
    cols, data = _columns(p, tmp_path)
    assert cols == [
        "MC_cycle",
        "N",
        "U",
        *TRANSLATE,
        *EXCHANGE,
        "cavity_sum_n",
        "cavity_draws",
        "wall_s",
    ]
    assert (data["insert_evaluated"] <= data["insert_tried"]).all()
    assert (data["delete_evaluated"] <= data["delete_tried"]).all()


def test_fixed_n_repack_logs_the_clique_sums(tmp_path):
    p = _bulk({"depth": 1}, CHANNEL, {"type": "NVT", "N": 150})
    p["potential"] = {"main": "lennard_jones", "T": 1.5}
    cols, data = _columns(p, tmp_path)
    assert cols == [
        "MC_cycle",
        "N",
        "U",
        *TRANSLATE,
        "repack_tried",
        "repack_evaluated",
        "repack_accepted",
        "repack_sum_size",
        "repack_sum_new",
        "repack_sum_size_accepted",
        "repack_sum_new_accepted",
        "cavity_sum_n",
        "cavity_draws",
        "wall_s",
    ]
    assert (data["N"] == 150).all()


@pytest.mark.parametrize("engine", ["glauber_dynamics", "suwa_todo_dynamics"])
def test_ladder_columns(engine, tmp_path):
    p = _bulk({"type": engine}, CHANNEL, {"mu": -2.0})
    cols, _ = _columns(p, tmp_path)
    assert cols == ["MC_cycle", "N", "U", *TRANSLATE, *LADDER, "wall_s"]


def test_full_span_ladder_logs_the_body_length(tmp_path):
    sampler = {**CHANNEL, "channel_length": "full"}
    p = _bulk({"type": "suwa_todo_dynamics"}, sampler, {"mu": -2.0})
    cols, _ = _columns(p, tmp_path)
    assert "ladder_sum_length" in cols


def test_a_typed_run_logs_species_counts_and_the_swap(tmp_path):
    p = _bulk({"depth": 1}, CHANNEL, {"mu": {"A": -3.4, "B": -2.55}})
    p["potential"] = {"main": "lennard_jones", "T": 1.5, "species_file": TYPED}
    del p["run"]["upper_cutoff"]
    cols, _ = _columns(p, tmp_path)
    assert cols[:4] == ["MC_cycle", "N_A", "N_B", "U"]
    assert "N" not in cols
    assert {"swap_tried", "swap_accepted"} <= set(cols)


def test_semigrand_ladder_has_no_number_change_sums(tmp_path):
    p = _bulk(
        {"type": "glauber_dynamics"},
        CHANNEL,
        {"type": "semigrand", "N": 150, "mu": {"A": 0.0, "B": 0.0}},
    )
    p["potential"] = {"main": "lennard_jones", "T": 1.5, "species_file": TYPED}
    del p["run"]["upper_cutoff"]
    cols, _ = _columns(p, tmp_path)
    assert "ladder_sum_abs_dN" not in cols and "ladder_sum_dN2" not in cols


def test_a_hard_core_run_has_no_energy_column(tmp_path):
    p = _bulk({}, {"type": "metropolis"}, {"mu": 1.0})
    p["potential"] = {"main": "hard_core"}
    del p["run"]["upper_cutoff"]
    cols, _ = _columns(p, tmp_path)
    assert "U" not in cols


def test_the_target_controller_logs_mu(tmp_path):
    p = _bulk({}, {"type": "metropolis"}, {"target_N": 200})
    cols, _ = _columns(p, tmp_path)
    assert cols[:4] == ["MC_cycle", "N", "U", "mu"]


@pytest.mark.parametrize(
    "engine,sampler",
    [
        ({}, {"type": "metropolis"}),
        ({"depth": 1}, CHANNEL),
        ({"full_repacking": True}, {"type": "branch", "trial_box_side": 1.5}),
        ({"type": "suwa_todo_dynamics"}, CHANNEL),
    ],
)
def test_the_running_energy_matches_a_recompute(engine, sampler, tmp_path):
    """U carried from window to window equals a from-scratch sum at the end,
    with no checkpoint in between to resynchronise it."""
    p = _bulk(engine, sampler, {"mu": -1.0})
    p["run"]["checkpoint_every"] = 1000
    state, paths = runner.run(copy.deepcopy(p), root=str(tmp_path / "runs"))
    data = io.read_log(paths.log)
    cfg = from_params(p)
    full = configuration_energy(
        state.positions,
        cfg.box_dims,
        3,
        cfg.upper_cutoff,
        tail_pref=cfg.lj_tail_pref,
    )
    assert data["U"][-1] == pytest.approx(full, rel=1e-8, abs=1e-8)


def test_retired_keys_are_refused_by_name():
    p = _bulk({}, {"type": "metropolis"}, {"mu": -2.0})
    p["run"]["esjd"] = True
    with pytest.raises(ValueError, match=r"retired run key.*esjd"):
        from_params(p)


def test_the_log_format(tmp_path):
    path = tmp_path / "run.log"
    log = io.RunLog(path)
    log.write({"MC_cycle": 0.001, "N": 3, "a_very_long_column_name": -566.95664912})
    log.write({"MC_cycle": 0.002, "N": 4, "a_very_long_column_name": -0.978})
    log.close()
    lines = path.read_text().splitlines()
    assert lines[0].startswith("#")
    assert io.log_columns(path) == ["MC_cycle", "N", "a_very_long_column_name"]
    data = io.read_log(path)
    assert list(data["N"]) == [3, 4]
    assert data["a_very_long_column_name"][0] == pytest.approx(-566.95664912)
    # a resumed run appends under the same header
    log = io.RunLog(path)
    log.write({"MC_cycle": 0.003, "N": 5, "a_very_long_column_name": 1.0})
    log.close()
    assert list(io.read_log(path)["N"]) == [3, 4, 5]
    # and a row with different columns is refused
    log = io.RunLog(path)
    with pytest.raises(ValueError, match="columns"):
        log.write({"MC_cycle": 0.004, "N": 6})
    log.close()


def test_a_log_in_the_old_format_is_refused(tmp_path):
    path = tmp_path / "run.log"
    path.write_text("# MC_cycle(M)  pckng_frctn  N\n 0.001 0.3 10\n")
    with pytest.raises(ValueError, match="older format"):
        io.RunLog(path)


def test_trim_log_drops_rows_past_the_checkpoint(tmp_path):
    path = tmp_path / "run.log"
    log = io.RunLog(path)
    for k in range(1, 6):
        log.write({"MC_cycle": k * 0.0001, "N": k})
    log.close()
    io.trim_log(path, 3 * 0.0001)
    assert list(io.read_log(path)["N"]) == [1, 2, 3]


def test_a_quenched_annealed_run_gets_an_id_and_keeps_its_energy(tmp_path):
    """The frozen matrix is hashed as a digest, not as raw coordinates, and the
    running U (fluid-fluid plus fluid-matrix) survives the checkpoint check."""
    from conftest import _poisson_disk

    mfile = tmp_path / "matrix.txt"
    np.savetxt(mfile, _poisson_disk(30, 3, 10.0, min_sep=1.6, seed=3))
    p = _bulk(
        {"depth": 1},
        {"type": "branch", "trial_box_side": 1.5, "trial_density": 4},
        {"type": "QA", "mu": -1.0},
    )
    p["ensemble"]["box"] = 10.0
    p["run"]["matrix_config"] = str(mfile)
    _state, paths = runner.run(copy.deepcopy(p), root=str(tmp_path / "runs"))
    assert len(io.read_log(paths.log)["U"]) == RUN["target_cycles"] // 60


def test_parallel_lennard_jones_branch_is_refused():
    p = _bulk(
        {"depth": 1},
        {"type": "branch", "trial_box_side": 1.5, "trial_density": 4},
        {"mu": -1.0},
    )
    p["run"].update(kind="parallel", cell_length=10.0, buffer=3.0)
    p["ensemble"]["box"] = 24.0
    with pytest.raises(ValueError, match="parallel does not wire Lennard-Jones"):
        from_params(p)


def test_flush_puts_every_written_row_on_disk(tmp_path):
    """Rows are batched between flushes; a checkpoint's flush must leave the
    file holding every row written so far."""
    path = tmp_path / "run.log"
    log = io.RunLog(path)
    for k in range(1, 4):
        log.write({"MC_cycle": k * 0.001, "N": k})
    log.flush()
    assert list(io.read_log(path)["N"]) == [1, 2, 3]
    log.close()

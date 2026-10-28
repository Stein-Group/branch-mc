"""C8: the language-neutral cost model and the IAT report.

Synthetic inputs only, no simulation: an AR(1) series with a known integrated
autocorrelation time, written as a BRANCH_MC run.log and as a parsed RASPA2
stream, so the cost bookkeeping and the tau estimate can each be checked against
a number derived by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gcmc.forcefield.species_file import load_forcefield
from gcmc.framework.raspa2 import Raspa2Arm, simulation_input
from gcmc.run import io
from scripts.mof.benchmark_report import (
    _main,
    analyse_branch,
    analyse_raspa2,
    branch_pee_per_row,
    compare,
    parse_raspa2_moves,
    raspa2_expected_pee_per_step,
)


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def _branch_dir(tmp: Path, x: np.ndarray, glauber: bool = False) -> Path:
    d = tmp / "branch"
    d.mkdir()
    if glauber:
        (d / "state_point.json").write_text(json.dumps({"ladder": {"pool_M": 12}}))
    log = io.RunLog(d / "run.log")
    for i, v in enumerate(x):
        n = 100 + int(round(10 * v))
        row = {"MC_cycle": (i + 1) * 1e-4, "N_Xe": n // 2, "N_Kr": n - n // 2}
        row.update(translate_tried=100, insert_tried=50, delete_tried=50)
        row["swap_tried"] = 25
        if glauber:
            row["ladder_sum_pool"] = 120
        log.write(row)
    log.close()
    return d


def test_branch_row_cost_is_exact():
    one = np.ones(1)
    cols = {
        "MC_cycle": one,
        "translate_tried": 100 * one,
        "insert_tried": 50 * one,
        "delete_tried": 50 * one,
        "swap_tried": 25 * one,
    }
    assert branch_pee_per_row(cols, None, None)[0] == 2 * 100 + 50 + 50 + 2 * 25
    cols["ladder_sum_pool"] = 120 * one
    # pairs = 120 (12 - 1) / 2 = 660, priced at 330 pairs per PEE
    assert branch_pee_per_row(cols, 12, 330.0)[0] == 350 + 120 + 2.0
    with pytest.raises(ValueError, match="pairs-per-pee"):
        branch_pee_per_row(cols, 12, None)


def test_a_ladder_run_reads_its_pool_size(tmp_path):
    d = _branch_dir(tmp_path, _ar1(200, 0.5, 6), glauber=True)
    r = analyse_branch("g", d, "N_total", 0.0, 330.0)
    assert r.pee_per_sample == pytest.approx(350 + 120 + 2.0)


def test_branch_tau_matches_the_ar1_value(tmp_path):
    phi = 0.8
    x = _ar1(40000, phi, 1)
    r = analyse_branch("b", _branch_dir(tmp_path, x), "N_total", 0.0, None)
    # AR(1): tau_int = (1 + phi) / (2 (1 - phi)) = 4.5 samples; rounding to
    # integer N adds a little white noise, which can only lower it
    assert r.reliable
    assert 3.8 < r.tau_samples < 4.8
    assert r.pee_per_sample == pytest.approx(350.0)
    assert r.tau_pee == pytest.approx(r.tau_samples * 350.0)


def test_raspa_expected_cost_from_the_generated_input(tmp_path):
    ff = load_forcefield("ff/forcefield_A_shift.yaml")
    sim = tmp_path / "simulation.input"
    arm = Raspa2Arm(
        "cbmc", p_translation=2.0, p_exchange=2.0, p_identity=1.0, k_trials=8
    )
    sim.write_text(
        simulation_input(
            ff,
            "MFM-300-In",
            (2, 2, 3),
            298.0,
            1e6,
            {"Xe": 0.2, "Kr": 0.8},
            10,
            0,
            arm=arm,
        )
    )
    moves = parse_raspa2_moves(sim)
    assert moves["k_swap"] == 8.0 and len(moves["components"]) == 2
    pee, kinds = raspa2_expected_pee_per_step(moves)
    # per component: translation 2/5 x 2, swap 2/5 x 8, identity 1/5 x 2
    assert pee == pytest.approx(0.4 * 2 + 0.4 * 8 + 0.2 * 2)
    assert sum(kinds.values()) == pytest.approx(1.0)


def _raspa_log(tmp: Path, x: np.ndarray, every: int = 5) -> Path:
    p = tmp / "parsed.log"
    log = io.RunLog(p)
    for i, v in enumerate(x):
        n = 100 + int(round(10 * v))
        log.write(
            {
                "MC_cycle": i * every / 1e6,
                "N_Xe": n // 2,
                "N_Kr": n - n // 2,
                "frac_Xe": 1,
                "frac_Kr": 1,
            }
        )
    log.close()
    return p


def test_raspa_cost_per_sample_and_the_control(tmp_path):
    ff = load_forcefield("ff/forcefield_A_shift.yaml")
    sim = tmp_path / "simulation.input"
    sim.write_text(
        simulation_input(
            ff,
            "MFM-300-In",
            (2, 2, 3),
            298.0,
            1e6,
            {"Xe": 1.0},
            10,
            0,
            arm=Raspa2Arm("conventional"),
        )
    )
    x = _ar1(40000, 0.8, 2)
    r = analyse_raspa2("r", _raspa_log(tmp_path, x), sim, "N_total", 0.0)
    # one component: translation 1/2 x 2 + swap 1/2 x 1 = 1.5 PEE per step;
    # steps per cycle = max(20, N + 1 fractional) x 1, 5 cycles per sample
    nads = np.array([100 + int(round(10 * v)) for v in x]) + 2
    assert r.pee_per_sample == pytest.approx(
        float(np.mean(np.maximum(20, nads))) * 5 * 1.5
    )
    b = analyse_branch(
        "b", _branch_dir(tmp_path, _ar1(40000, 0.8, 3)), "N_total", 0.0, None
    )
    c = compare(b, r)
    assert c["ratio"] == pytest.approx(r.tau_pee / b.tau_pee)
    assert abs(c["mean_z"]) < 4.0


def test_ratio_is_refused_when_a_tau_is_unreliable(tmp_path):
    short = _ar1(60, 0.98, 4)  # far shorter than 50 tau
    a = analyse_branch("short", _branch_dir(tmp_path, short), "N_total", 0.0, None)
    assert not a.reliable
    assert compare(a, a)["ratio"] is None


def test_cli_writes_a_report(tmp_path):
    d = _branch_dir(tmp_path, _ar1(5000, 0.5, 5))
    out = tmp_path / "rep.json"
    assert (
        _main(["--branch", f"conv={d}", "--observable", "x_Xe", "--json", str(out)])
        == 0
    )
    rep = json.loads(out.read_text())
    assert rep["arms"]["conv"]["observable"] == "x_Xe"

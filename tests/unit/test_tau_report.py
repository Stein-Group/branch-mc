"""`scripts.tau_report` reads a finished run directory and reports tau_N per
wall-second, and it REFUSES the engine ratio when either arm's tau is not
resolved.

The refusal is the part worth a test. The rest of this file is parsing, and
parsing that goes wrong announces itself; a ratio computed from two
under-resolved taus does not -- it prints a plausible number, and the number is
wrong in a direction that flatters whichever engine happens to be measured.
That is not hypothetical: the same comparison read 1.5x in the Glauber engine's
favour at 40 measured windows and parity at 400.

Synthetic logs throughout, built from an AR(1) series whose tau is known in
closed form, so nothing here runs a chain and the test is a few milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gcmc.run import io
from scripts.tau_report import analyse_run, ratios, read_log

_HEADER = "# MC_cycle      N   wall_s\n"


def _ar1(n: int, phi: float, seed: int, mean: float = 100.0) -> np.ndarray:
    """An AR(1) series, whose integrated autocorrelation time is known:
    tau = (1 + phi) / (2 (1 - phi)). At phi = 0 it is 0.5, the uncorrelated
    floor; at phi = 0.9 it is 9.5."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.0
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return mean + 10.0 * x


def _write_run(
    d: Path,
    N: np.ndarray,
    *,
    engine: str = "glauber_dynamics",
    sampler: str = "channel",
    mu: float = -2.5,
    wall: float = 0.5,
) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rows = [f"{(i + 1) * 2e-4:14.6f} {n:6.0f} {wall:8.4f}" for i, n in enumerate(N)]
    (d / "run.log").write_text(_HEADER + "\n".join(rows) + "\n")
    (d / "params.json").write_text(
        json.dumps(
            {
                "potential": {"main": "lennard_jones", "T": 1.5},
                "sampler": {"type": sampler},
                "engine": {"type": engine},
                "ensemble": {"type": "muVT", "box": 8.0, "mu": mu},
                "run": {"seed": 1},
            }
        )
    )
    return d


def test_the_log_is_parsed_by_column_name(tmp_path):
    d = _write_run(tmp_path / "r", np.arange(10.0) + 100.0)
    cols = read_log(d / "run.log")
    assert set(cols) == {"MC_cycle", "N", "wall_s"}
    assert cols["N"][0] == 100.0 and cols["N"][-1] == 109.0


def test_a_log_whose_header_does_not_match_its_rows_raises(tmp_path):
    """A log written by a different version of the column table must not be
    read as if the names still lined up -- that would silently report the wrong
    column as N."""
    d = tmp_path / "r"
    d.mkdir()
    (d / "run.log").write_text("# a  b  c\n1.0 2.0\n")
    with pytest.raises(ValueError, match="columns"):
        read_log(d / "run.log")


def test_a_file_with_no_header_is_refused(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "run.log").write_text("1.0 2.0 3.0\n")
    with pytest.raises(ValueError, match="header"):
        read_log(d / "run.log")


def test_tau_and_the_cost_per_independent_N_come_out_of_the_log(tmp_path):
    """A long AR(1) series at phi = 0.9 has tau = 9.5 exactly; the estimator
    should land near it and the cost should be tau times the per-window wall."""
    d = _write_run(tmp_path / "r", _ar1(20_000, 0.9, seed=3), wall=0.5)
    got = analyse_run(d)
    assert got["reliable"], "20k samples at tau = 9.5 must be resolved"
    assert 8.0 < got["tau_windows"] < 11.0, got["tau_windows"]
    assert got["wall_s_per_window"] == pytest.approx(0.5)
    assert got["wall_s_per_independent_N"] == pytest.approx(got["tau_windows"] * 0.5)
    assert got["engine"] == "glauber_dynamics" and got["sampler"] == "channel"
    # no window group in this synthetic log, so none is reported
    assert "ladder" not in got


def test_half_the_series_is_discarded(tmp_path):
    d = _write_run(tmp_path / "r", np.arange(1000.0))
    got = analyse_run(d)
    assert got["windows"] == 1000 and got["windows_measured"] == 500
    assert got["N_mean"] == pytest.approx(np.arange(500, 1000).mean())


def test_the_ratio_is_refused_when_either_arm_is_unresolved(tmp_path):
    """The whole point. One arm long, one arm short, and no ratio is printed --
    the refusal names which arm was short and what to do about it."""
    long_arm = analyse_run(
        _write_run(
            tmp_path / "long", _ar1(20_000, 0.9, seed=1), engine="glauber_dynamics"
        )
    )
    short_arm = analyse_run(
        _write_run(
            tmp_path / "short", _ar1(60, 0.9, seed=2), engine="metropolis_dynamics"
        )
    )
    assert long_arm["reliable"] and not short_arm["reliable"]
    out = ratios([long_arm, short_arm])
    (verdict,) = out.values()
    assert "refused" in verdict
    assert "metropolis_dynamics" in verdict["refused"]
    assert "wall_s_per_independent_N" not in verdict


def test_the_ratio_is_computed_when_both_arms_are_resolved(tmp_path):
    """And it must fail if the ratio is inverted: the Glauber arm here is
    twice as expensive per window at the same tau, so the speedup over
    Metropolis is a half and not a two."""
    series = _ar1(20_000, 0.9, seed=7)
    hb = analyse_run(
        _write_run(tmp_path / "hb", series, engine="glauber_dynamics", wall=1.0)
    )
    mc = analyse_run(
        _write_run(tmp_path / "mc", series, engine="metropolis_dynamics", wall=0.5)
    )
    (verdict,) = ratios([hb, mc]).values()
    assert "refused" not in verdict
    sp = verdict["speedup_over_metropolis_dynamics"]
    assert sp["metropolis_dynamics"] == pytest.approx(1.0)
    assert sp["glauber_dynamics"] == pytest.approx(0.5, rel=1e-6)


def test_two_samplers_at_one_mu_are_not_compared_with_each_other(tmp_path):
    """`engine` is `metropolis_dynamics` for the channel arm AND for the
    single-particle reference arm, so grouping on mu alone would compare the
    reference against itself and drop the Glauber engine."""
    series = _ar1(20_000, 0.9, seed=11)
    rows = [
        analyse_run(
            _write_run(
                tmp_path / "ch_mc", series, engine="metropolis_dynamics", wall=0.5
            )
        ),
        analyse_run(
            _write_run(tmp_path / "ch_hb", series, engine="glauber_dynamics", wall=1.0)
        ),
        analyse_run(
            _write_run(
                tmp_path / "br",
                series,
                engine="metropolis_dynamics",
                sampler="branch",
                wall=0.25,
            )
        ),
    ]
    out = ratios(rows)
    # the branch arm has only one engine at this mu, so it forms no pair
    assert list(out) == ["mu-2.5/channel"]
    assert set(out["mu-2.5/channel"]["wall_s_per_independent_N"]) == {
        "metropolis_dynamics",
        "glauber_dynamics",
    }


def test_the_ladder_counts_are_reported_when_the_run_wrote_them(tmp_path):
    """A nonzero `ladder_cut_pool` count is a scope statement rather than an
    exactness failure, so it is surfaced rather than summarised away."""
    d = tmp_path / "r"
    d.mkdir()
    log = io.RunLog(d / "run.log")
    for i in range(10):
        log.write(
            {
                "MC_cycle": (i + 1) * 2e-4,
                "N": 100 + i,
                "ladder_unchanged": 40,
                "ladder_sum_abs_dN": 300,
                "ladder_max_k": 9,
                "ladder_infeasible": 0,
                "ladder_cut_kmax": 0,
                "ladder_cut_pool": 3,
                "wall_s": 0.5,
            }
        )
    log.close()
    (d / "params.json").write_text(
        json.dumps(
            {
                "sampler": {"type": "channel"},
                "engine": {"type": "glauber_dynamics"},
                "ensemble": {"mu": -2.5, "box": 8.0},
                "run": {"seed": 1, "sampling_window": 200},
            }
        )
    )
    w = analyse_run(d)["ladder"]
    assert w["cut_pool"] == 30.0  # 3 per window over ten windows
    assert w["unchanged_fraction"] == pytest.approx(40.0 / 200.0)
    assert w["dN_per_move"] == pytest.approx(300.0 / 200.0)
    assert w["max_k"] == 9.0

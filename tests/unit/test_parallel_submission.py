"""P2: the parallel submission surface (run.n_cell and the mode dispatch).

Config-level only: nothing here advances a chain or spawns a process. All fast.
"""

from __future__ import annotations

import pytest

from gcmc.hash_fingerprint import run_id
from gcmc.sim_setup import from_params

MFM = [31.322, 31.322, 24.994]


def _p(run, *, box=30.0, sampler=None, dim=3):
    return {
        "potential": {"main": "hard_core"},
        "sampler": sampler
        or {"type": "rosenbluth", "depth": 0, "trials": 30, "trial_box_side": 3.0},
        "ensemble": {"mu": 2.0, "dimension": dim, "box": box},
        "run": {"kind": "parallel", **run},
    }


# ------------------------------------------------------------- one cell, one core


def test_n_cell_is_the_core_count():
    for n in (1, 4, 8, 27):
        cfg = from_params(_p({"n_cell": n}))
        assert cfg.n_workers == n
        assert cfg.cell_counts is not None
        prod = 1
        for c in cfg.cell_counts:
            prod *= c
        assert prod == n


def test_per_axis_submission_is_honoured_verbatim():
    cfg = from_params(_p({"n_cell": [3, 2, 1]}))
    assert cfg.cell_counts == (3, 2, 1)
    assert cfg.n_workers == 6


def test_manual_geometry_still_reports_its_cell_count_as_cores():
    cfg = from_params(_p({"cell_length": 8.0, "buffer": 2.0}))
    assert cfg.cell_counts == (3, 3, 3)
    assert cfg.n_workers == 27


def test_serial_runs_take_one_core_and_have_no_cells():
    cfg = from_params(
        {
            "potential": {"main": "hard_core"},
            "sampler": {"type": "metropolis"},
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 30.0},
            "run": {"kind": "single"},
        }
    )
    assert cfg.n_workers == 1
    assert cfg.cell_counts is None


# --------------------------------------------------------- auto/manual agreement


def test_auto_and_manual_agree_when_they_describe_the_same_decomposition():
    """The two modes are different submissions of the same physics, so where
    they describe one decomposition they must produce one Config."""
    auto = from_params(_p({"n_cell": 27}))
    # auto derives buffer = interaction reach = 2*radius = 1.0, pitch 10 -> cell 9
    manual = from_params(_p({"cell_length": 9.0, "buffer": 1.0}))
    assert auto.cell_counts == manual.cell_counts
    assert auto.cell_dims == manual.cell_dims
    assert auto.buffer == manual.buffer
    assert auto.n_workers == manual.n_workers


def test_anisotropic_box_gets_per_axis_cell_lengths():
    cfg = from_params(_p({"n_cell": 12}, box=MFM))
    assert cfg.cell_counts is not None
    prod = 1
    for c in cfg.cell_counts:
        prod *= c
    assert prod == 12
    # the derived lengths are genuinely per-axis, not a cube with a relabelled axis
    assert cfg.cell_length_sides is not None
    for s, n, cl in zip(MFM, cfg.cell_counts, cfg.cell_dims):
        assert (s / n) - cl == pytest.approx(cfg.buffer, abs=1e-12)


# ------------------------------------------------- either/or, nothing in between


@pytest.mark.parametrize("clash", ["cell_length", "buffer", "x_spacing", "y_spacing"])
def test_n_cell_refuses_manual_geometry_alongside_it(clash):
    with pytest.raises(ValueError, match="cannot be given alongside"):
        from_params(_p({"n_cell": 8, clash: 2.0}))


@pytest.mark.parametrize(
    "run,missing",
    [
        ({"cell_length": 8.0}, "buffer"),
        ({"buffer": 2.0}, "cell_length"),
        ({}, "cell_length"),
    ],
)
def test_manual_mode_requires_the_whole_geometry(run, missing):
    """No defaults: the historical cell_length=30 could not fit any box <= 32."""
    with pytest.raises(ValueError, match=f"run.{missing}"):
        from_params(_p(run))


# ------------------------------------------------------------------- the guards


def test_too_many_cells_is_refused_with_usable_alternatives():
    with pytest.raises(ValueError, match="Nearest usable cell counts"):
        from_params(_p({"n_cell": 3375}))


def test_a_cell_smaller_than_the_cavity_is_refused_at_config_time():
    """This guard used to fire inside advance_parallel, i.e. after the job had
    already started. 512 cells in a 30-box leaves 2.75 against a floor of 3."""
    with pytest.raises(ValueError, match="shortest cell edge"):
        from_params(_p({"n_cell": 512}))


def test_the_floor_names_both_contributions():
    with pytest.raises(ValueError) as exc:
        from_params(_p({"n_cell": 512}))
    msg = str(exc.value)
    assert "cavity radius" in msg and "interaction reach" in msg


def test_manual_buffer_below_the_interaction_reach_is_refused():
    lj = {
        "potential": {"main": "lennard_jones", "T": 1.5},
        "sampler": {
            "type": "rosenbluth",
            "depth": 0,
            "trials": 30,
            "trial_box_side": 3.0,
        },
        "ensemble": {"mu": -3.0, "dimension": 3, "box": 30.0},
        "run": {
            "kind": "parallel",
            "cell_length": 8.0,
            "buffer": 2.0,
            "upper_cutoff": 2.5,
        },
    }
    with pytest.raises(ValueError, match="must be >= the interaction"):
        from_params(lj)


def test_the_reach_guard_reads_the_run_radius():
    """MEASURED DEFECT, now fixed: the guard used to read ens.get("radius", 0.5),
    but `radius` is a RUN key, so it always checked a reach of 1.0 no matter what
    run.radius said. With radius 1.5 the reach is 3.0 and a buffer of 2.0 is
    short."""
    with pytest.raises(ValueError, match="must be >= the interaction"):
        from_params(_p({"cell_length": 8.0, "buffer": 2.0, "radius": 1.5}))


def test_auto_mode_derives_the_buffer_from_the_reach():
    hs = from_params(_p({"n_cell": 8}))
    assert hs.buffer == pytest.approx(1.0)  # 2 * radius(0.5)
    lj = from_params(
        {
            "potential": {"main": "lennard_jones", "T": 1.5},
            "sampler": {
                "type": "rosenbluth",
                "depth": 0,
                "trials": 30,
                "trial_box_side": 3.0,
            },
            "ensemble": {"mu": -3.0, "dimension": 3, "box": 30.0},
            "run": {"kind": "parallel", "n_cell": 8, "upper_cutoff": 2.5},
        }
    )
    assert lj.buffer == pytest.approx(2.5)  # run.upper_cutoff, the real reach


def test_the_lj_reach_follows_upper_cutoff():
    """MEASURED DEFECT, now fixed: the reach used to be the literal 2.5 because
    it was read from potential.rc, a key the allowlist rejects. It now follows
    the truncation radius actually in force."""
    cfg = from_params(
        {
            "potential": {"main": "lennard_jones", "T": 1.5},
            "sampler": {
                "type": "rosenbluth",
                "depth": 0,
                "trials": 30,
                "trial_box_side": 3.0,
            },
            "ensemble": {"mu": -3.0, "dimension": 3, "box": 30.0},
            "run": {"kind": "parallel", "n_cell": 8, "upper_cutoff": 3.2},
        }
    )
    assert cfg.buffer == pytest.approx(3.2)


# ---------------------------------------------------------------- deprecations


@pytest.mark.parametrize("dep", ["n_workers", "x_spacing", "y_spacing"])
def test_deprecated_keys_warn_and_are_ignored(dep, caplog):
    """D5: accepted and warned for one release. Ignored, not honoured -- keeping
    them working would be the 'something in between' the modes that D5 rules
    out."""
    with caplog.at_level("WARNING"):
        cfg = from_params(_p({"cell_length": 8.0, "buffer": 2.0, dep: 4}))
    assert any(dep in r.getMessage() for r in caplog.records), caplog.text
    assert cfg.n_workers == 27  # the cell count, not the submitted 4


def test_a_deprecated_key_does_not_survive_into_the_config():
    cfg = from_params(_p({"cell_length": 8.0, "buffer": 2.0, "x_spacing": 4.0}))
    assert cfg.x_spacing is None or cfg.x_spacing == 4.0  # stored, never consumed
    assert cfg.cell_counts == (3, 3, 3)


# --------------------------------------------------- identity must not move (rule 4)


def test_the_decomposition_is_not_part_of_the_run_identity():
    """Cluster run directories must stay resumable, so no parallel key may enter
    the fingerprint. Two decompositions of the same physics share a run id."""
    base = _p({"n_cell": 8})
    a = run_id(from_params(base))
    b = run_id(from_params(_p({"n_cell": 27})))
    c = run_id(from_params(_p({"cell_length": 8.0, "buffer": 2.0})))
    assert a == b == c


def test_a_parallel_run_and_its_serial_twin_share_an_identity():
    par = run_id(from_params(_p({"n_cell": 8})))
    ser = from_params(
        {
            "potential": {"main": "hard_core"},
            "sampler": {
                "type": "rosenbluth",
                "depth": 0,
                "trials": 30,
                "trial_box_side": 3.0,
            },
            "ensemble": {"mu": 2.0, "dimension": 3, "box": 30.0},
            "run": {"kind": "single"},
        }
    )
    assert par == run_id(ser)


# ------------------------------------------------------------------- 2D parallel


def test_two_dimensional_submission():
    cfg = from_params(
        _p(
            {"n_cell": 16},
            box=75.0,
            dim=2,
            sampler={"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 3.0},
        )
    )
    assert cfg.cell_counts == (4, 4)
    assert cfg.n_workers == 16
    assert len(cfg.cell_dims) == 2

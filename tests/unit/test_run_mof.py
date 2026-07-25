"""Q3: the checkpointed MOF run driver must resume bit-identically.

A run killed by walltime and resumed from its checkpoint has to reproduce the
trajectory of an unkilled run to the last bit; otherwise the two legs are a
different Markov chain and the streamed statistics are silently wrong. The RNG
state round-trips through the schema-3 checkpoint (species column included), so
this is a property test on the whole wrapper: run 4 windows straight vs run 2
then resume to 4, compare final positions and species exactly.

Uses a tiny bulk box (no framework build) so it is fast and needs no CIF.
"""

from __future__ import annotations

import numpy as np

from scripts.mof.run_mof import MofRunSpec, run_mof

FF = "ff/forcefield_A.yaml"


def _spec(root, windows):
    return MofRunSpec(
        forcefield=FF,
        T=298.0,
        z_star_by_name={"Xe": 0.03, "Kr": 0.03},
        seed=12345,
        windows=windows,
        window_cycles=40,
        bulk_box=6.0,
        checkpoint_every=2,
        root=root,
    )


def test_resume_is_bit_identical(tmp_path):
    # uninterrupted: 4 windows straight
    a = run_mof(_spec(tmp_path / "straight", windows=4))
    assert a.resumed_from == 0
    assert a.windows_done == 4

    # interrupted: 2 windows, then resume the SAME run to 4
    root_b = tmp_path / "chained"
    b1 = run_mof(_spec(root_b, windows=2))
    assert b1.windows_done == 2
    b2 = run_mof(_spec(root_b, windows=4))
    assert b2.resumed_from == 2  # picked up the checkpoint, did not restart
    assert b2.windows_done == 4

    # same physics -> same run id/directory across the two legs
    assert b1.run_id == b2.run_id == a.run_id

    np.testing.assert_array_equal(a.positions, b2.positions)
    np.testing.assert_array_equal(a.species, b2.species)
    assert a.n_by_type == b2.n_by_type


def test_run_id_tracks_physics_not_windows(tmp_path):
    base = _spec(tmp_path / "a", windows=4)
    same_more_windows = _spec(tmp_path / "b", windows=99)
    from scripts.mof.run_mof import mof_run_id

    assert mof_run_id(base) == mof_run_id(same_more_windows)

    diff_seed = _spec(tmp_path / "c", windows=4)
    diff_seed.seed = 999
    assert mof_run_id(diff_seed) != mof_run_id(base)

    # sweep-move settings are part of the identity (Blocker 2)
    diff_sweep = _spec(tmp_path / "d", windows=4)
    diff_sweep.sweep_moves_per_window = 3
    assert mof_run_id(diff_sweep) != mof_run_id(base)


def test_csv_has_one_line_per_window(tmp_path):
    r = run_mof(_spec(tmp_path / "log", windows=3))
    lines = r.paths.log.read_text().strip().splitlines()
    # header + 3 window rows
    assert lines[0].startswith("window,")
    assert len(lines) == 1 + 3

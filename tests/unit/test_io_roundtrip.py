"""io round-trips: a checkpoint restores positions and the RNG stream exactly,
the previous checkpoint rotates to .bak, and the CSV Log writes its header once
across a reopen. These are the disk-boundary guarantees a resume relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc import io
from gcmc.hash_fingerprint import RunPaths


def _paths(d) -> RunPaths:
    return RunPaths(
        directory=d,
        checkpoint=d / "checkpoint.npz",
        checkpoint_bak=d / "checkpoint.npz.bak",
        trajectory=d / "trajectory.extxyz",
        log=d / "log.csv",
        params=d / "params.json",
        lock=d / "lock",
    )


def test_checkpoint_roundtrip_restores_positions_and_rng(tmp_path):
    p = _paths(tmp_path)
    pos = np.random.default_rng(0).uniform(0, 10, (7, 3))
    rng = np.random.default_rng(42)
    rng.uniform(size=5)  # advance off the seed point
    io.save_checkpoint(
        p,
        positions=pos,
        step=1000,
        rng=io.rng_state(rng),
        counters={"acc": 3},
        params={"run": {"seed": 42}},
        mu=2.0,
    )
    want_next = rng.uniform(size=4)  # the stream continues from the saved state

    ck = io.load_checkpoint(p)
    assert ck is not None
    assert np.array_equal(ck["positions"], pos)
    assert ck["step"] == 1000 and ck["counters"] == {"acc": 3}

    restored = io.set_rng_state(np.random.default_rng(), ck["rng"])
    assert np.array_equal(restored.uniform(size=4), want_next)


def test_second_save_rotates_previous_to_bak(tmp_path):
    p = _paths(tmp_path)
    common = {
        "rng": io.rng_state(np.random.default_rng(1)),
        "counters": {},
        "params": {},
    }
    io.save_checkpoint(p, positions=np.zeros((1, 3)), step=1, **common)
    assert not p.checkpoint_bak.exists()
    io.save_checkpoint(p, positions=np.zeros((2, 3)), step=2, **common)
    assert p.checkpoint_bak.exists()
    assert io.load_checkpoint(p)["step"] == 2  # primary is the newest


def test_log_writes_header_once_across_reopen(tmp_path):
    path = tmp_path / "run.csv"
    log = io.Log(path, "# a,b,c", flush_every=1)
    log.write("1,2,3")
    log.close()
    reopened = io.Log(path, "# a,b,c", flush_every=1)
    reopened.write("4,5,6")
    reopened.close()
    lines = path.read_text().splitlines()
    assert lines == ["# a,b,c", "1,2,3", "4,5,6"]


def test_trim_log_drops_rows_past_the_cutoff(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("# cycleM x\n0.001 a\n0.002 b\n0.003 c\n")
    io.trim_log(path, 0.002)
    kept = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    assert kept == ["0.001 a", "0.002 b"]


def _comment(paths) -> str:
    """The second line of an extxyz frame: the Lattice/Properties/pbc header."""
    return paths.trajectory.read_text().splitlines()[1]


def test_a_2d_frame_declares_a_degenerate_open_z_axis(tmp_path):
    """The padded third axis is not a physical direction: it has no length and no
    periodicity. Writing a real side there described a cubic 3D box that was never
    simulated, so a viewer would wrap along z and count spurious images."""
    paths = _paths(tmp_path)
    io.write_frame(
        paths,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        two_dim=True,
        box=75.0,
        info={"step": 1000},
        append=False,
    )
    line = _comment(paths)
    assert 'Lattice="75.0 0.0 0.0 0.0 75.0 0.0 0.0 0.0 0.0"' in line
    assert 'pbc="T T F"' in line


def test_a_2d_frame_with_a_per_axis_box_keeps_both_real_sides(tmp_path):
    """The old code reached the scalar branch first, so a per-axis 2D box was
    given a bogus z side too. Both 2D paths are covered now."""
    paths = _paths(tmp_path)
    io.write_frame(
        paths,
        np.array([[1.0, 2.0]]),
        two_dim=True,
        box=[75.0, 60.0],
        info={"step": 1},
        append=False,
    )
    line = _comment(paths)
    assert 'Lattice="75.0 0.0 0.0 0.0 60.0 0.0 0.0 0.0 0.0"' in line
    assert 'pbc="T T F"' in line


@pytest.mark.parametrize(
    "box,want",
    [
        (30.0, '"30.0 0.0 0.0 0.0 30.0 0.0 0.0 0.0 30.0"'),
        ([31.322, 31.322, 24.994], '"31.322 0.0 0.0 0.0 31.322 0.0 0.0 0.0 24.994"'),
    ],
    ids=["cubic", "orthorhombic"],
)
def test_3d_frames_are_unchanged(tmp_path, box, want):
    paths = _paths(tmp_path)
    io.write_frame(
        paths,
        np.array([[1.0, 2.0, 3.0]]),
        two_dim=False,
        box=box,
        info={"step": 1},
        append=False,
    )
    line = _comment(paths)
    assert f"Lattice={want}" in line
    assert 'pbc="T T T"' in line


def test_a_2d_frame_still_loads_back(tmp_path):
    """A zero cell vector must not break the equilibrated warm-start library: the
    loaders take positions only and never consult the cell or pbc."""
    paths = _paths(tmp_path)
    pos = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    io.write_frame(paths, pos, two_dim=True, box=75.0, info={"step": 1}, append=False)
    assert np.allclose(io.load_config_file(paths.trajectory, 2), pos)

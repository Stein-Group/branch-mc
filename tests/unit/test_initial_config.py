"""Initial configurations from the equilibrated library.

The library stores strict extended-XYZ files (atom-count line, a Lattice/Properties
comment, then `species x y z` rows), nested by system, temperature (LJ) and box:
    lj/T_0.85/box_20/mu_0.0_lj_3d_box20.xyz
    hard_disk_2d/box_200/mu_1.00_hard_2d_box_200.xyz
Filename decimal and box_<L> vs box<L> styles differ between families, so a run
picks its file by matching mu, box, dimension and (for LJ) temperature rather than
by rebuilding an exact name. The reader must handle extended-XYZ and a bare float
table, and give a clear error when nothing or too much matches.
"""

import os
import tempfile

import numpy as np

from gcmc.run import io, runner, sim_setup


def _write_extxyz(path, pts, L):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{len(pts)}\n")
        f.write(
            f'Lattice="{L} 0.0 0.0 0.0 {L} 0.0 0.0 0.0 {L}" '
            f'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        )
        for p in pts:
            xyz = list(p) + [0.0] * (3 - len(p))
            f.write(f"H  {xyz[0]:.8f}  {xyz[1]:.8f}  {xyz[2]:.8f}\n")


def _library(root, rng):
    _write_extxyz(
        f"{root}/lj/T_0.85/box_20/mu_-3.0_lj_3d_box20.xyz",
        rng.uniform(0, 20, (50, 3)),
        20.0,
    )
    _write_extxyz(
        f"{root}/lj/T_1.0/box_20/mu_-3.0_lj_3d_box20.xyz",
        rng.uniform(0, 20, (70, 3)),
        20.0,
    )
    _write_extxyz(
        f"{root}/hard_disk_2d/box_200/mu_2.00_hard_2d_box_200.xyz",
        rng.uniform(0, 200, (40, 2)),
        200.0,
    )
    _write_extxyz(
        f"{root}/hard_disk_3d/box_20/mu_2.00_hard_3d_box20.xyz",
        rng.uniform(0, 20, (60, 3)),
        20.0,
    )


def _cfg(pot, T, mu, dim, box, root):
    p = {
        "potential": {"main": pot},
        "sampler": {"type": "metropolis"},
        "ensemble": {"mu": mu, "dimension": dim, "box": box},
        "run": {"initial_config": root},
    }
    if pot == "lennard_jones":
        p["potential"]["T"] = T
    return sim_setup.from_params(p)


def test_extxyz_is_parsed():
    with tempfile.TemporaryDirectory() as d:
        f = f"{d}/c.xyz"
        _write_extxyz(f, [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], 10.0)
        pos = io.load_config_file(f, 3)
        assert pos.shape == (2, 3)
        assert np.allclose(pos[0], [1.0, 2.0, 3.0])
        # a 2D run keeps two columns
        assert io.load_config_file(f, 2).shape == (2, 2)


def test_bare_table_still_works():
    with tempfile.TemporaryDirectory() as d:
        f = f"{d}/t.dat"
        np.savetxt(f, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        pos = io.load_config_file(f, 3)
        assert pos.shape == (2, 3) and np.allclose(pos[1], [4.0, 5.0, 6.0])


def test_resolver_matches_lj_by_temperature():
    with tempfile.TemporaryDirectory() as d:
        _library(d, np.random.default_rng(0))
        a = runner._init_config(
            _cfg("lennard_jones", 0.85, -3.0, 3, 20.0, d), np.random.default_rng(1)
        )
        b = runner._init_config(
            _cfg("lennard_jones", 1.0, -3.0, 3, 20.0, d), np.random.default_rng(1)
        )
        assert len(a) == 50 and len(b) == 70  # T disambiguates the two files


def test_resolver_matches_hard_both_box_styles():
    with tempfile.TemporaryDirectory() as d:
        _library(d, np.random.default_rng(0))
        two = runner._init_config(
            _cfg("hard_core", None, 2.0, 2, 200.0, d), np.random.default_rng(1)
        )
        three = runner._init_config(
            _cfg("hard_core", None, 2.0, 3, 20.0, d), np.random.default_rng(1)
        )
        assert len(two) == 40 and len(three) == 60


def test_no_match_lists_available():
    with tempfile.TemporaryDirectory() as d:
        _library(d, np.random.default_rng(0))
        try:
            runner._init_config(
                _cfg("hard_core", None, 99.0, 3, 20.0, d), np.random.default_rng(1)
            )
        except FileNotFoundError as e:
            assert "matches this run" in str(e) and "mu=99" in str(e)
            return
        raise AssertionError("a non-existent mu must raise")


def test_ambiguous_match_lists_candidates():
    with tempfile.TemporaryDirectory() as d:
        # two files with identical mu/box/dim and no T to separate them
        _write_extxyz(
            f"{d}/a/box_20/mu_1.0_hard_3d_box20.xyz",
            np.random.default_rng(0).uniform(0, 20, (10, 3)),
            20.0,
        )
        _write_extxyz(
            f"{d}/b/box_20/mu_1.0_hard_3d_box20.xyz",
            np.random.default_rng(1).uniform(0, 20, (11, 3)),
            20.0,
        )
        try:
            runner._init_config(
                _cfg("hard_core", None, 1.0, 3, 20.0, d), np.random.default_rng(1)
            )
        except ValueError as e:
            assert "ambiguous" in str(e)
            return
        raise AssertionError("two equally-matching files must raise")


def test_bad_table_columns_error():
    with tempfile.TemporaryDirectory() as d:
        f = f"{d}/bad.xyz"
        with open(f, "w") as fh:
            fh.write("1.0 2.0\n3.0 4.0\n")  # first line not an int -> table
        try:
            io.load_config_file(f, 3)
        except ValueError as e:
            assert "columns" in str(e)
            return
        raise AssertionError("2-column table for a 3D run must raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("PASS test_initial_config")

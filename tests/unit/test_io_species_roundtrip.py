"""Phase 4 (A4.1/A4.4): the optional species column in the checkpoint format.

The bump 2 -> 3 is a pure format extension, so two things must both hold: a
schema-3 checkpoint carrying a species column round-trips exactly, and a
schema-2 checkpoint written before the column existed still loads, reporting
species as None (single-species). If the second broke, every pre-Phase-4
checkpoint on disk would become unreadable.
"""

from __future__ import annotations

import numpy as np

from gcmc.run import io
from gcmc.system.state import ChainState


def _paths(tmp_path):
    class P:
        checkpoint = tmp_path / "chk.npz"
        checkpoint_bak = tmp_path / "chk.npz.bak"

    return P()


def test_schema_is_three():
    assert io.SCHEMA == 3


def test_species_column_round_trips(tmp_path):
    paths = _paths(tmp_path)
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 5, size=(7, 3))
    species = np.array([0, 1, 1, 0, 2, 1, 0], dtype=np.int64)
    io.save_checkpoint(
        paths,
        positions=pos,
        step=10,
        rng=rng.bit_generator.state,
        params={"mode": "mof-ms"},
        species=species,
    )
    got = io.load_checkpoint(paths)
    assert got["schema"] == 3
    np.testing.assert_array_equal(got["species"], species)
    np.testing.assert_allclose(got["positions"], pos)


def test_single_species_writes_no_species_and_loads_none(tmp_path):
    """No species argument means no species array on disk, and load reports None,
    which is how a single-species schema-3 file is byte-for-byte the same content
    as the old schema-2 file."""
    paths = _paths(tmp_path)
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 5, size=(4, 3))
    io.save_checkpoint(
        paths,
        positions=pos,
        step=3,
        rng=rng.bit_generator.state,
        params={},
    )
    with np.load(paths.checkpoint, allow_pickle=False) as data:
        assert "species" not in data.files
    got = io.load_checkpoint(paths)
    assert got["species"] is None


def test_schema_two_checkpoint_still_loads(tmp_path):
    """A checkpoint written under the old schema (2, no species key) loads under
    schema-3 code and reports species None."""
    paths = _paths(tmp_path)
    rng = np.random.default_rng(2)
    pos = rng.uniform(0, 5, size=(3, 3))
    io.save_checkpoint(
        paths,
        positions=pos,
        step=1,
        rng=rng.bit_generator.state,
        params={},
        schema=2,
    )
    got = io.load_checkpoint(paths)
    assert got["schema"] == 2
    assert got["species"] is None
    np.testing.assert_allclose(got["positions"], pos)


def test_chainstate_species_of_defaults_to_zeros():
    st = ChainState(positions=np.zeros((5, 3)), step=0, rng=np.random.default_rng(0))
    assert st.species is None
    np.testing.assert_array_equal(st.species_of, np.zeros(5, dtype=np.int64))

    typed = ChainState(
        positions=np.zeros((3, 3)),
        step=0,
        rng=np.random.default_rng(0),
        species=np.array([0, 2, 1]),
    )
    np.testing.assert_array_equal(typed.species_of, np.array([0, 2, 1]))

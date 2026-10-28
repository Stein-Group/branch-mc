"""Gate P6: a typed configuration survives going out to a file and back.

BOTH DIRECTIONS USED TO LOSE SPECIES, and neither failed while doing it.
`io.write_frame` labelled every particle `"H"` whatever its type was, and every
reader called `get_positions()` and discarded `get_chemical_symbols()`. So a
trajectory frame of a three-species mixture was a valid-looking warm start that
restored the coordinates and none of the composition -- and a run started from it
reported a pure species-0 system at the density of a mixture.

The checkpoint had the same shape one layer up, and it is the sharper example
because the code was already there: `io.save_checkpoint` grew a `species`
parameter at D2, `_read_npz` reads the array back, `load_checkpoint` returns it in
its dict -- and `runner._save` never passed it and the resume path never read it.
A parameter with no caller and a return value with no reader, facing each other
across a gap.

WHAT THE GATE ASSERTS, in the order the plan states it:

  * positions come back bit-identically and species exactly, through a
    trajectory frame and through a checkpoint;
  * a schema-2 checkpoint (no species array) still loads;
  * a symbol that is not a species of the model raises and NAMES it;
  * a bare coordinate table raises in typed mode rather than defaulting.

The last two are the ones worth having. Mapping an unknown symbol to type 0, or
defaulting a bare table to all-type-0, produces a run that starts, finishes, and
reports a composition it never sampled.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gcmc.run import io

_SYMBOLS = ("Ne", "Ar", "Kr")


def _paths(tmp_path: Path):
    """A minimal RunPaths pointing into tmp_path."""
    from gcmc.run.hash_fingerprint import RunPaths

    root = tmp_path / "run"
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        directory=root,
        checkpoint=root / "ckpt.npz",
        checkpoint_bak=root / "ckpt.npz.bak",
        trajectory=root / "traj.extxyz",
        log=root / "run.log",
        params=root / "params.json",
        lock=root / "run.lock",
    )


def _sample(n: int = 17, box: float = 8.0):
    rng = np.random.default_rng(4242)
    pos = rng.uniform(0.0, box, (n, 3))
    species = rng.integers(0, len(_SYMBOLS), n).astype(np.int64)
    # every type must appear, or the round trip could pass while dropping one
    species[: len(_SYMBOLS)] = np.arange(len(_SYMBOLS))
    return pos, species, box


# ------------------------------------------------------------ trajectory frame


def test_a_trajectory_frame_round_trips_positions_and_species(tmp_path) -> None:
    """Write a mixture, read it back, get the same coordinates and the same types."""
    paths = _paths(tmp_path)
    pos, species, box = _sample()
    io.write_frame(
        paths,
        pos,
        two_dim=False,
        box=box,
        info={"step": 0},
        append=False,
        species=species,
        symbols=_SYMBOLS,
    )

    back_pos = io.load_config_file(paths.trajectory, 3)
    back_species = io.load_config_species(
        paths.trajectory, symbols=_SYMBOLS, n_expected=len(pos)
    )

    assert back_species.tolist() == species.tolist(), (
        "species did not survive the round trip; the frame's symbol column and "
        "the species file's symbol table disagree about the mapping"
    )
    # extxyz writes fixed-precision decimals, so positions come back CLOSE rather
    # than bit-identical. Stated as a tolerance on the FORMAT, not on the physics:
    # ASE's extxyz writer emits 8 decimals, so 1e-7 is the format's own floor and
    # anything larger than that is a real discrepancy.
    assert np.allclose(back_pos, pos, atol=1e-7, rtol=0.0), (
        f"positions moved by up to {np.abs(back_pos - pos).max():.3e}, which is "
        "more than extxyz's own decimal precision"
    )


def test_a_single_species_frame_still_writes_hydrogen(tmp_path) -> None:
    """The untyped path is untouched, and that is load-bearing.

    `tests/engine/test_stageA_traj_identity.py` compares trajectory files against
    frozen references. Writing real symbols on a one-species run would move every
    one of them for no gain -- there is nothing to disambiguate -- so `species`
    of None keeps the old "H".
    """
    paths = _paths(tmp_path)
    pos, _, box = _sample()
    io.write_frame(paths, pos, two_dim=False, box=box, info={}, append=False)
    text = paths.trajectory.read_text().splitlines()
    labels = {line.split()[0] for line in text[2:] if line.strip()}
    assert labels == {"H"}, f"single-species frame used symbols {labels}"


def test_a_mislabelled_symbol_raises_and_names_it(tmp_path) -> None:
    """A frame from another force field is not a warm start for this one."""
    paths = _paths(tmp_path)
    pos, species, box = _sample()
    io.write_frame(
        paths,
        pos,
        two_dim=False,
        box=box,
        info={},
        append=False,
        species=species,
        symbols=_SYMBOLS,
    )
    with pytest.raises(ValueError, match=r"Kr|not species of this model"):
        io.load_config_species(
            paths.trajectory, symbols=("Ne", "Ar"), n_expected=len(pos)
        )


def test_a_bare_coordinate_table_raises_in_typed_mode(tmp_path) -> None:
    """Composition is not recoverable from coordinates, so it is not guessed."""
    table = tmp_path / "bare.txt"
    pos, _, _ = _sample()
    np.savetxt(table, pos)
    # the positions still load -- the untyped path is unaffected
    assert io.load_config_file(table, 3).shape == pos.shape
    with pytest.raises(ValueError, match=r"bare coordinate table|species column"):
        io.load_config_species(table, symbols=_SYMBOLS, n_expected=len(pos))


def test_a_count_mismatch_raises(tmp_path) -> None:
    """Labels that do not belong to the coordinates are refused, not truncated."""
    paths = _paths(tmp_path)
    pos, species, box = _sample()
    io.write_frame(
        paths,
        pos,
        two_dim=False,
        box=box,
        info={},
        append=False,
        species=species,
        symbols=_SYMBOLS,
    )
    with pytest.raises(ValueError, match="symbols for"):
        io.load_config_species(
            paths.trajectory, symbols=_SYMBOLS, n_expected=len(pos) + 1
        )


def test_write_frame_refuses_a_species_column_with_no_symbols(tmp_path) -> None:
    """Type ids alone cannot be written as element symbols."""
    paths = _paths(tmp_path)
    pos, species, box = _sample()
    with pytest.raises(ValueError, match="no symbol table"):
        io.write_frame(
            paths,
            pos,
            two_dim=False,
            box=box,
            info={},
            append=False,
            species=species,
        )


def test_write_frame_refuses_a_misaligned_species_column(tmp_path) -> None:
    """One label per position, or the frame mislabels every particle."""
    paths = _paths(tmp_path)
    pos, species, box = _sample()
    with pytest.raises(ValueError, match="species labels for"):
        io.write_frame(
            paths,
            pos,
            two_dim=False,
            box=box,
            info={},
            append=False,
            species=species[:-1],
            symbols=_SYMBOLS,
        )


# ----------------------------------------------------------------- checkpoint


def test_a_checkpoint_round_trips_the_species_column(tmp_path) -> None:
    """positions bit-identical (npz is binary) and species exact."""
    paths = _paths(tmp_path)
    pos, species, _ = _sample()
    io.save_checkpoint(
        paths,
        positions=pos,
        step=7,
        rng={"a": 1},
        params={},
        species=species,
    )
    ckpt = io.load_checkpoint(paths)
    assert ckpt is not None
    assert np.asarray(ckpt["positions"]).tobytes() == pos.tobytes(), (
        "npz is a binary format; positions must come back bit-identical"
    )
    assert ckpt["species"] is not None, (
        "the checkpoint lost the species column, so a resumed multi-species run "
        "would restart with every particle relabelled species 0"
    )
    assert np.asarray(ckpt["species"]).tolist() == species.tolist()


def test_a_schema_two_checkpoint_still_loads(tmp_path) -> None:
    """No species array at all reads as None -- single species, not an error.

    Written by hand rather than by `save_checkpoint(species=None)` so the test
    does not depend on the writer to describe the old format: the point is that a
    file produced before the column existed still loads.
    """
    paths = _paths(tmp_path)
    pos, _, _ = _sample()
    meta = json.dumps(
        {
            "schema": 2,
            "step": 4,
            "n": len(pos),
            "counters": {},
            "rng": {},
            "params": {},
        }
    )
    np.savez(paths.checkpoint, positions=pos, meta=np.array(meta))
    ckpt = io.load_checkpoint(paths)
    assert ckpt is not None and ckpt["schema"] == 2
    assert ckpt["species"] is None
    assert np.asarray(ckpt["positions"]).tobytes() == pos.tobytes()

"""A3.1: load a framework as positions PLUS force-field type ids.

`io.load_matrix` returns positions only, which is all a quenched-annealed run
needs: its obstacles interact by overlap exclusion, so their type never matters.
The MOF needs the type of every framework atom, because a guest-framework pair
energy depends on it. This module reads the P0.3 extxyz -- which carries a
species column of force-field type NAMES -- and returns those names resolved to
the integer ids of a PairTable, alongside the positions.

It is a separate function rather than an edit to load_matrix for the same reason
everything in this project is separate: the position-only path is frozen and
QA runs depend on it byte-for-byte. Nothing here touches it.

    positions, type_ids = load_framework_typed("framework.extxyz", table)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .._types import FloatArray, IntArray
from ..forcefield.pair_table import PairTable


def load_framework_typed(
    path: str | Path, table: PairTable, dimension: int = 3
) -> tuple[FloatArray, IntArray]:
    """Framework positions and per-atom PairTable type ids from a P0.3 extxyz.

    The species column holds force-field type names (In, C, O, H, ...); each is
    resolved through `table.index`, so a name the table has no parameters for is
    a hard error here rather than a silent zero later. Only the last frame is
    read, matching load_matrix: a framework file is one static structure, but if
    it were a trajectory the final frame is the built supercell.
    """
    path = Path(path)
    if path.suffix.lower() not in (".extxyz", ".xyz"):
        raise ValueError(
            f"{path}: framework must be extxyz with a species column; got {path.suffix!r}"
        )
    from ase.io import read as ase_read

    atoms = ase_read(str(path), index="-1", format="extxyz")
    assert not isinstance(atoms, list)  # index="-1" yields a single frame

    pos = np.ascontiguousarray(
        np.asarray(atoms.get_positions(), dtype=np.float64)[:, :dimension]
    )
    symbols = atoms.get_chemical_symbols()
    unknown = sorted({s for s in symbols if s not in table.types})
    if unknown:
        raise ValueError(
            f"{path}: framework contains {unknown}, which the PairTable has no parameters for"
        )
    type_ids = np.array([table.index(s) for s in symbols], dtype=np.int64)
    return pos, type_ids

"""Shared system setup for the MOF drivers (Q2).

v1_henry, v2_isotherm and v3_mixture all turn a forcefield yaml plus either a
framework CIF or a bulk box side into the same objects: the reduced pair table,
the FrameworkField (real or empty), the reduction constants, and the per-run
scalars (mass, cell count, reduced box, beta). This is that one path, so the
drivers differ only in their move set and reporting, not in how they build the
system. Behaviour is identical to the inline blocks it replaces.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gcmc.forcefield.lj_table import FrameworkField
from gcmc.forcefield.pair_table import PairTable
from scripts.mof.units import load_forcefield, reduce_tables


@dataclass
class SystemSetup:
    ff: object  # ForceField
    red: object  # ReducedTables
    table: PairTable
    field: FrameworkField
    box: np.ndarray  # reduced per-axis box
    volume: float  # reduced volume
    beta: float  # 1 / T_star
    sigma_ref: float
    mass_amu: float  # framework mass (0.0 for a bulk box)
    n_cells: float  # unit cells in the supercell (1.0 for a bulk box)
    label: str  # human-readable system name


def setup_system(
    forcefield: Path,
    T: float,
    *,
    cif: Path | None = None,
    bulk_box: float | None = None,
    reference: str = "Xe",
) -> SystemSetup:
    """Build the reduced table and field for one system. Exactly one of `cif`
    (a framework) or `bulk_box` (a cubic bulk cell side in angstrom) is given."""
    if (cif is None) == (bulk_box is None):
        raise ValueError("give exactly one of cif or bulk_box")

    ff = load_forcefield(forcefield)
    red = reduce_tables(ff, T, reference=reference)
    if ff.core_floor_c is None:
        raise ValueError("core floor not measured; run P0.6")
    table = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )
    sr = red.sigma_ref

    if cif is not None:
        from gcmc.run.io_framework import load_framework_typed
        from scripts.mof.build_framework import build, write_extxyz

        atoms, meta = build(cif, ff)
        mass_amu = float(atoms.get_masses().sum())
        with tempfile.TemporaryDirectory() as td:
            ex = Path(td) / "framework.extxyz"
            write_extxyz(atoms, ex)
            fpos, tids = load_framework_typed(ex, table)
            box = np.asarray(meta["supercell_angstrom"]) / sr
            field = FrameworkField(fpos / sr, tids, table, box, ff.rc / sr)
        n_cells = float(np.prod(meta["replication"]))
        label = cif.stem
    else:
        assert bulk_box is not None
        box = np.full(3, bulk_box / sr)
        field = FrameworkField(
            np.empty((0, 3)), np.empty(0, int), table, box, ff.rc / sr
        )
        mass_amu = 0.0
        n_cells = 1.0
        label = f"bulk {bulk_box:g} A box"

    return SystemSetup(
        ff=ff,
        red=red,
        table=table,
        field=field,
        box=box,
        volume=float(np.prod(box)),
        beta=1.0 / red.T_star,
        sigma_ref=sr,
        mass_amu=mass_amu,
        n_cells=n_cells,
        label=label,
    )

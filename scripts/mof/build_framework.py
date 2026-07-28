"""P0.3: CIF in, framework extxyz plus a provenance sidecar out.

Reads a CIF through ASE (which applies the symmetry operations, so a
symmetry-reduced deposition comes out in P1), replicates it until every cell edge
reaches 2*rc, and writes the result as extxyz with a types column that names
force-field types rather than elements. The sidecar records what went in
(hashes of the CIF and the yaml), what came out (cell, per-type counts) and the
replication that got there, so a later disagreement with RASPA can be traced to a
file rather than argued about.

Axes stay orthogonal by fence: the script refuses a cell with non-90 angles
rather than quietly rotating it, because a rotation would put the channel axis
somewhere the sweep's axis-aligned cuboid (A5.4) cannot follow.

    python -m scripts.mof.build_framework MFM-300-In.cif \\
        --forcefield ff/forcefield_A.yaml --out framework.extxyz --provisional
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
from ase import Atoms
from ase.io import read

from scripts.mof.units import ForceField, load_forcefield

# Elements whose force-field type name differs from the chemical symbol would go
# here. Empty for Xe/Kr in MFM-300: Dreiding/UFF assign one parameter set per
# element, so the element symbol IS the type name and there is no atom typing
# step to get wrong. The moment a force field needs C_aromatic vs C_sp3, this
# mapping stops being the identity and P0.3 grows a real typing pass.
TYPE_ALIASES: dict[str, str] = {}


def _type_name(symbol: str) -> str:
    return TYPE_ALIASES.get(symbol, symbol)


def orthogonal_lengths(atoms: Atoms, tol: float = 1e-6) -> np.ndarray:
    """The three edge lengths, after refusing anything that is not orthogonal."""
    cell = atoms.cell.array
    angles = atoms.cell.angles()
    if np.any(np.abs(angles - 90.0) > 1e-4):
        raise ValueError(
            f"cell angles {angles} are not all 90 degrees. Triclinic and monoclinic "
            "cells are out of scope (mof_raspa_plan.md, SCOPE FENCES)."
        )
    off = np.abs(cell - np.diag(np.diag(cell)))
    if off.max() > tol * max(1.0, np.abs(cell).max()):
        raise ValueError(
            "cell vectors are orthogonal but not axis-aligned; rotate the CIF first"
        )
    return np.abs(np.diag(cell))


def replication_for(lengths: np.ndarray, rc: float) -> np.ndarray:
    """Smallest per-axis replication with every edge >= 2 rc.

    Orthogonal axes, so the perpendicular width of a slab equals its edge length
    and the usual minimum-image condition is exactly this.
    """
    return np.maximum(1, np.ceil(2.0 * rc / lengths - 1e-12).astype(int))


def build(cif: Path, ff: ForceField) -> tuple[Atoms, dict]:
    atoms = cast(
        Atoms, read(str(cif), index=-1)
    )  # one frame: a CIF with several would be ambiguous here
    lengths = orthogonal_lengths(atoms)
    reps = replication_for(lengths, ff.rc)
    super_atoms = atoms.repeat(tuple(int(r) for r in reps))
    super_lengths = orthogonal_lengths(super_atoms)

    symbols = super_atoms.get_chemical_symbols()
    unknown = sorted({s for s in symbols if _type_name(s) not in ff.types})
    if unknown:
        raise ValueError(
            f"{cif.name} contains {unknown}, which {ff.name} has no parameters for. "
            "Add them to the yaml with a source, or pick a different framework."
        )
    counts: dict[str, int] = {}
    for s in symbols:
        counts[_type_name(s)] = counts.get(_type_name(s), 0) + 1

    meta = {
        "cif": {
            "path": str(cif),
            "sha256": hashlib.sha256(cif.read_bytes()).hexdigest(),
        },
        "forcefield": {
            "name": ff.name,
            "path": str(ff.path),
            "sha256": ff.sha256,
            "status": ff.status,
        },
        "unit_cell_angstrom": lengths.tolist(),
        "replication": reps.tolist(),
        "supercell_angstrom": super_lengths.tolist(),
        "rc_angstrom": ff.rc,
        "min_edge_over_2rc": float(super_lengths.min() / (2.0 * ff.rc)),
        "n_atoms": len(super_atoms),
        "atoms_per_type": counts,
    }
    return super_atoms, meta


def write_extxyz(atoms: Atoms, path: Path) -> None:
    """extxyz with a types column, wrapped into the cell.

    Written directly rather than via ase.io.write so the column layout is fixed
    here and ref_energy.py can parse it without ASE.
    """
    atoms = atoms.copy()
    atoms.wrap()
    L = orthogonal_lengths(atoms)
    lattice = f"{L[0]:.10f} 0.0 0.0 0.0 {L[1]:.10f} 0.0 0.0 0.0 {L[2]:.10f}"
    lines = [
        str(len(atoms)),
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"',
    ]
    for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f"{_type_name(s):<4} {p[0]:>18.10f} {p[1]:>18.10f} {p[2]:>18.10f}")
    path.write_text("\n".join(lines) + "\n")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cif", type=Path)
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="framework extxyz to write")
    p.add_argument("--provisional", action="store_true")
    a = p.parse_args(argv)

    ff = load_forcefield(a.forcefield, allow_provisional=a.provisional)
    atoms, meta = build(a.cif, ff)
    write_extxyz(atoms, a.out)
    sidecar = a.out.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2))

    L = meta["supercell_angstrom"]
    print(f"{a.cif.name}: unit cell {meta['unit_cell_angstrom']}")
    print(
        f"  replication {meta['replication']} -> {L[0]:.3f} x {L[1]:.3f} x {L[2]:.3f} A"
    )
    print(
        f"  {meta['n_atoms']} atoms: "
        + ", ".join(f"{k}x{v}" for k, v in meta["atoms_per_type"].items())
    )
    print(f"  min edge / 2rc = {meta['min_edge_over_2rc']:.3f}  (must be >= 1)")
    print(f"  wrote {a.out} and {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

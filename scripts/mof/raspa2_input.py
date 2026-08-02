"""The RASPA2 side of the one-yaml rule (D4's tie-breaker, and what the anchor ran).

D4 pins RASPA3 as primary and names RASPA2 (2.0.4x) as the cross-check when a
rung disagrees. RASPA3 needs a C++23 toolchain; RASPA2 is plain C and builds
here, and it is also the code the anchor itself used (their ref 75 = Dubbeldam
2016). So the V0 single points can be run now rather than waited on, against the
very version the anchor's numbers came from.

Same rule as raspa_input.py: every parameter comes from ff/*.yaml. The formats
differ (RASPA2 wants .def files, RASPA3 wants JSON) but the numbers must not.

RASPA2 conventions that matter here, read from its own example inputs and source:
  - force_field_mixing_rules.def opens with the shift rule then the tail rule,
    both as bare words: we write "truncated" and "no", which is D1.
  - the LJ line is `type lennard-jones eps/kB[K] sigma[A]`, the same order and
    units as the yaml and as RASPA3's JSON.
  - CutOffVDW belongs in simulation.input, not in the force field file.
  - molecule .def files open with the three critical constants, which is why the
    eos block exists in the yaml.

Placing guests at exact positions goes through RASPA2's restart machinery: it
writes a restart file, and reads one back from RestartInitial/. Rather than
reverse-engineer that format we let RASPA2 write it and patch the coordinates,
which is the same measure-don't-guess move as everywhere else.

    python -m scripts.mof.raspa2_input --forcefield ff/forcefield_A.yaml \\
        --cif structures/MFM-300-In.cif --out /tmp/r2case --T 298 --pressure 1e5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

from ase import Atoms
from ase.io import read

from scripts.mof.raspa_input import MASSES, guest_critical
from scripts.mof.units import ForceField, load_forcefield, peng_robinson


def write_raspa2_cif(src: Path, dst: Path) -> None:
    """Re-emit a P1 CIF with the symmetry tags RASPA2 actually parses.

    ASE writes `_space_group_name_H-M_alt`; RASPA2's framework.c looks for the
    older `_symmetry_space_group_name_H-M` / `_symmetry_Int_Tables_number` and
    dies with "no proper space group definition found" otherwise. The structure is
    already P1 with all 144 atoms explicit, so this is a relabelling, not a
    symmetry operation: nothing moves.
    """
    atoms = cast(Atoms, read(str(src), index=-1))
    a, b, c = atoms.cell.lengths()
    al, be, ga = atoms.cell.angles()
    lines = [
        f"data_{src.stem}",
        f"_cell_length_a     {a:.10f}",
        f"_cell_length_b     {b:.10f}",
        f"_cell_length_c     {c:.10f}",
        f"_cell_angle_alpha  {al:.6f}",
        f"_cell_angle_beta   {be:.6f}",
        f"_cell_angle_gamma  {ga:.6f}",
        "_symmetry_cell_setting             triclinic",
        "_symmetry_space_group_name_Hall    'P 1'",
        "_symmetry_space_group_name_H-M     'P 1'",
        "_symmetry_Int_Tables_number        1",
        "",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "  'x,y,z'",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_charge",
    ]
    frac = atoms.get_scaled_positions()
    for sym, f in zip(atoms.get_chemical_symbols(), frac):
        # the label carries no number: RemoveAtomNumberCodeFromLabel is on, but a
        # bare symbol leaves nothing to strip and nothing to misparse
        lines.append(f"{sym:<4} {sym:<4} {f[0]:12.8f} {f[1]:12.8f} {f[2]:12.8f}  0.0")
    dst.write_text("\n".join(lines) + "\n")


def pseudo_atoms_def(ff: ForceField, present: tuple[str, ...]) -> str:
    lines = [
        "#number of pseudo atoms",
        str(len(present)),
        "#type      print   as    chem  oxidation   mass        charge   polarization "
        "B-factor radii  connectivity anisotropic anisotropic-type   tinker-type",
    ]
    for t in present:
        if t not in MASSES:
            raise KeyError(f"no mass on file for {t!r}")
        # charge is 0 for every type: uncharged guests in a rigid framework, so
        # ChargeMethod None is exact rather than an approximation
        lines.append(
            f"{t:<10} yes     {t:<5} {t:<5} 0           {MASSES[t]:<11.6f} 0.0      0.0"
            f"          1.0      1.0    0            0           relative           0"
        )
    return "\n".join(lines) + "\n"


def force_field_mixing_rules_def(ff: ForceField, present: tuple[str, ...]) -> str:
    lines = [
        "# general rule for shifted vs truncated",
        "truncated",  # D1: truncated, unshifted
        "# general rule tailcorrections",
        "no",  # D1: no tail correction
        "# number of defined interactions",
        str(len(present)),
        "# type interaction, parameters.    IMPORTANT: define shortest matches first",
    ]
    for t in present:
        i = ff.index(t)
        src = str(ff.raw["guests" if i < ff.n_guests else "framework"][t]["source"])
        lines.append(
            f"{t:<14} lennard-jones    {ff.epsilon[i]:<9.5f} {ff.sigma[i]:<12.5f} // {src.strip()[:70]}"
        )
    lines += ["# general mixing rule for Lennard-Jones", "Lorentz-Berthelot"]
    return "\n".join(lines) + "\n"


def molecule_def(ff: ForceField, guest: str) -> str:
    crit = guest_critical(ff, guest)
    return (
        "\n".join(
            [
                "# critical constants: Temperature [T], Pressure [Pa], and Acentric factor [-]",
                f"{crit['CriticalTemperature']}",
                f"{crit['CriticalPressure']}",
                f"{crit['AcentricFactor']}",
                "# Number Of Atoms",
                "1",
                "# Number Of Groups",
                "1",
                "# Group-name",
                "rigid",
                "# number of atoms",
                "1",
                "# atomic positions",
                f"0 {guest}",
                "# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  Torsion Imp. Torsion Bond/Bond "
                "Stretch/Bend Bend/Bend Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb",
                "               0 0    0           0     0           0        0       0            0         "
                "0            0         0               0            0        0",
                "# Number of config moves",
                "0",
            ]
        )
        + "\n"
    )


def simulation_input(
    ff: ForceField,
    framework_name: str,
    unit_cells: tuple[int, int, int],
    T: float,
    pressure: float,
    mol_fractions: dict[str, float],
    cycles: int,
    init_cycles: int,
    counts: dict[str, int] | None = None,
    restart: bool = False,
    print_every: int | None = None,
) -> str:
    fug = peng_robinson(ff, T, pressure, mol_fractions)
    lines = [
        "SimulationType                MonteCarlo",
        f"NumberOfCycles                {cycles}",
        f"NumberOfInitializationCycles  {init_cycles}",
        # R1.8: this stream is the primary data source, not a progress log.
        f"PrintEvery                    "
        f"{print_every if print_every else (max(1, cycles // 100) if cycles else 1)}",
        f"RestartFile                   {'yes' if restart else 'no'}",
        "",
        "Forcefield                    Local",
        # RASPA2 wants the cutoff here, not in the force field file
        f"CutOffVDW                     {ff.rc}",
        "ChargeMethod                  None",
        "",
        "Framework                     0",
        f"FrameworkName                 {framework_name}",
        "UseChargesFromCIFFile         no",
        "RemoveAtomNumberCodeFromLabel yes",
        f"UnitCells                     {unit_cells[0]} {unit_cells[1]} {unit_cells[2]}",
        f"ExternalTemperature           {T}",
        f"ExternalPressure              {pressure}",
        "",
    ]
    for k, (name, x) in enumerate(mol_fractions.items()):
        n = 0 if counts is None else counts.get(name, 0)
        lines += [
            f"Component {k} MoleculeName              {name}",
            "            MoleculeDefinition        Local",
            f"            MolFraction               {x}",
            "            IdealGasRosenbluthWeight  1.0",
            # our phi, not RASPA's: the same number that made our z*
            f"            FugacityCoefficient       {fug.phi[name]:.10f}",
            "            TranslationProbability    0.5",
            "            SwapProbability           1.0",
            f"            CreateNumberOfMolecules   {n}",
            "",
        ]
    return "\n".join(lines)


def write_case(
    out: Path,
    ff: ForceField,
    cif: Path,
    T: float,
    pressure: float,
    mol_fractions: dict[str, float],
    unit_cells: tuple[int, int, int],
    cycles: int,
    init_cycles: int,
    present: tuple[str, ...],
    counts: dict[str, int] | None = None,
    restart: bool = False,
    print_every: int | None = None,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    write_raspa2_cif(cif, out / cif.name)
    (out / "pseudo_atoms.def").write_text(pseudo_atoms_def(ff, present))
    (out / "force_field_mixing_rules.def").write_text(
        force_field_mixing_rules_def(ff, present)
    )
    (out / "force_field.def").write_text(
        "# rules to overwrite\n0\n# number of defined interactions\n0\n# mixing rules to overwrite\n0\n"
    )
    for g in mol_fractions:
        (out / f"{g}.def").write_text(molecule_def(ff, g))
    (out / "simulation.input").write_text(
        simulation_input(
            ff,
            cif.stem,
            unit_cells,
            T,
            pressure,
            mol_fractions,
            cycles,
            init_cycles,
            counts,
            restart,
            print_every,
        )
    )
    return out


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument("--cif", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--pressure", type=float, required=True)
    p.add_argument("--guests", default="Xe")
    p.add_argument("--unit-cells", default="2,2,2")
    p.add_argument("--cycles", type=int, default=0)
    p.add_argument("--init-cycles", type=int, default=0)
    p.add_argument("--present", default="Xe,Kr,In,C,O,H")
    p.add_argument(
        "--y",
        default=None,
        help="gas mole fractions per guest, comma-separated in --guests order "
        "(e.g. 0.2,0.8). Default: uniform. REQUIRED for the 20:80 binary record "
        "runs -- the uniform default silently gives 50:50.",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=None,
        help="PrintEvery in cycles. The instantaneous stream is the PRIMARY data "
        "source (R1.8) and block-length doubling needs thousands of rows, so the "
        "cycles//100 default (100 rows per leg) is too coarse for a record run.",
    )
    a = p.parse_args(argv)

    ff = load_forcefield(a.forcefield)
    names = [n.strip() for n in a.guests.split(",") if n.strip()]
    if a.y is None:
        fracs = {n: 1.0 / len(names) for n in names}
    else:
        vals = [float(x) for x in a.y.split(",")]
        if len(vals) != len(names):
            p.error("--y must list one mole fraction per guest, in --guests order")
        if abs(sum(vals) - 1.0) > 1e-9:
            p.error("--y must sum to 1")
        fracs = dict(zip(names, vals, strict=True))
    out = write_case(
        a.out,
        ff,
        a.cif,
        a.T,
        a.pressure,
        fracs,
        tuple(int(x) for x in a.unit_cells.split(",")),  # type: ignore[arg-type]
        a.cycles,
        a.init_cycles,
        tuple(a.present.split(",")),
        print_every=a.print_every,
    )
    print(f"wrote RASPA2 case in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

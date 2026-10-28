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

A run of the RASPA2 arms is a params file with engine.type raspa2_* (see
`gcmc.framework.params`); `write_case` is what it calls, and v1_henry calls it
for the Widom cross-check.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from ase import Atoms
from ase.io import read

from ..forcefield.species_file import ForceField, peng_robinson
from .raspa2_parse import write_parsed
from .structure import build, orthogonal_setting, present_types

if TYPE_CHECKING:
    from ..run.hash_fingerprint import RunPaths
    from .driver import FrameworkResult
    from .params import FrameworkRun

# Atomic masses for every type the shipped force fields carry. RASPA reports
# masses and uses them for nothing compared here (no MD, no mass-weighted moves),
# but it demands them, and a wrong number in an input file is wrong even when
# inert.
MASSES = {
    "Xe": 131.293,
    "Kr": 83.798,
    "In": 114.818,
    "Al": 26.9815386,
    "Ga": 69.723,
    "Sc": 44.955912,
    "V": 50.9415,
    "Cr": 51.9961,
    "Fe": 55.845,
    "C": 12.0107,
    "O": 15.9994,
    "H": 1.00794,
    "Mg": 24.305,
    "CH4": 16.04246,
}


def guest_critical(ff: ForceField, guest: str) -> dict[str, float]:
    """The critical constants RASPA wants, from the yaml's eos block (one source
    for every number). RASPA would compute Peng-Robinson from them if
    FugacityCoefficient were left out; we supply the coefficient, and these
    travel along so a case is self-describing."""
    eos = ff.raw.get("eos")
    if eos is None or guest not in eos:
        raise KeyError(f"{ff.path}: no eos entry for {guest!r}")
    return {
        "CriticalTemperature": float(eos[guest]["critical_temperature"]),
        "CriticalPressure": float(eos[guest]["critical_pressure"]),
        "AcentricFactor": float(eos[guest]["acentric_factor"]),
    }


@dataclass(frozen=True)
class Raspa2Arm:
    """One contender of the CFCMC benchmark, as RASPA2 move settings.

    `kind`:
      conventional  single-trial swap: NumberOfTrialPositionsForTheFirstBeadSwap 1.
                    RASPA2's DEFAULT swap is CBMC with 10 first-bead trials
                    (input.c sets every *FirstBead* count to 10), so "SwapProbability"
                    alone is NOT conventional GCMC; this arm pins k = 1.
      cbmc          the same swap with `k_trials` first-bead positions (for a
                    single-site guest CBMC is exactly k-trial Rosenbluth insertion)
      cfcmc         continuous fractional component swap (CFSwapLambdaProbability),
                    Wang-Landau biasing calibrated during NumberOfEquilibrationCycles
                    and frozen in production, as in Torres-Knoop et al. JCTC 2014

    Move weights are RASPA2's per-component relative probabilities. The defaults
    mirror the BRANCH_MC conventional loop: translation : exchange = 1 : 1 at one
    species (0.5 / 0.25 + 0.25) and translation : exchange : identity = 2 : 2 : 1
    for a mixture (0.4 / 0.2 + 0.2 / 0.2, moves.MS_PROBS).

    Identity changes regrow at the same position (CBMC_PARTIAL_INSERTION), and
    their first-bead trial count is pinned to 1 too, so no arm hides a multi-trial
    move outside its exchange move.
    """

    kind: str
    p_translation: float = 1.0
    p_exchange: float = 1.0
    p_identity: float = 0.5
    k_trials: int = 1
    cf_lambda_bins: int = 20
    cf_wl_scaling: float | None = None  # None: RASPA2's own default
    equilibration_cycles: int = 0
    optimize_translation: bool = True

    def __post_init__(self) -> None:
        if self.kind not in ("conventional", "cbmc", "cfcmc"):
            raise ValueError(f"unknown RASPA2 arm {self.kind!r}")
        if self.kind == "conventional" and self.k_trials != 1:
            raise ValueError("the conventional arm is k_trials = 1 by definition")
        if self.kind == "cbmc" and self.k_trials < 2:
            raise ValueError("a CBMC arm needs k_trials >= 2 (k = 1 is conventional)")
        if self.kind == "cfcmc" and self.equilibration_cycles <= 0:
            raise ValueError(
                "CFCMC needs NumberOfEquilibrationCycles > 0: that phase is where "
                "Wang-Landau calibrates the lambda biasing factors"
            )
        if min(self.p_translation, self.p_exchange, self.p_identity) < 0:
            raise ValueError("move weights must be non-negative")


def arm_global_lines(arm: Raspa2Arm) -> list[str]:
    """simulation.input lines that are global (not per component) for an arm."""
    return [
        f"NumberOfEquilibrationCycles   {arm.equilibration_cycles}",
        f"OptimizeTranslation           {'yes' if arm.optimize_translation else 'no'}",
        f"NumberOfTrialPositionsForTheFirstBeadSwap            {arm.k_trials}",
        "NumberOfTrialPositionsForTheFirstBeadIdentityChange  1",
    ]


def arm_component_lines(arm: Raspa2Arm, n_components: int) -> list[str]:
    """Per-component move lines for an arm (indented like the rest)."""
    out = [f"            TranslationProbability    {arm.p_translation}"]
    if arm.kind == "cfcmc":
        out += [
            f"            CFSwapLambdaProbability   {arm.p_exchange}",
            f"            CFLambdaHistogramSize     {arm.cf_lambda_bins}",
        ]
        if arm.cf_wl_scaling is not None:
            out.append(f"            CFWangLandauScalingFactor {arm.cf_wl_scaling}")
    else:
        out.append(f"            SwapProbability           {arm.p_exchange}")
    if n_components > 1 and arm.p_identity > 0:
        out += [
            f"            IdentityChangeProbability {arm.p_identity}",
            f"            NumberOfIdentityChanges   {n_components}",
            "            IdentityChangesList       "
            + " ".join(str(i) for i in range(n_components)),
        ]
    return out


def write_raspa2_cif(src: Path, dst: Path) -> None:
    """Re-emit a P1 CIF with the symmetry tags RASPA2 actually parses.

    ASE writes `_space_group_name_H-M_alt`; RASPA2's framework.c looks for the
    older `_symmetry_space_group_name_H-M` / `_symmetry_Int_Tables_number` and
    dies with "no proper space group definition found" otherwise. The structure is
    already P1 with all 144 atoms explicit, so this is a relabelling, not a
    symmetry operation: nothing moves. A hexagonal cell is written in the same
    orthorhombic setting build_framework gives BRANCH_MC, so both codes simulate
    the same periodic box.
    """
    atoms, _ = orthogonal_setting(cast(Atoms, read(str(src), index=-1)))
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
    """pseudo_atoms.def for the types in play: masses from MASSES, every charge
    zero (uncharged guests in a rigid host, so ChargeMethod None is exact)."""
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
    """force_field_mixing_rules.def from the yaml: shifted or truncated as the
    file says, no tail correction, Lorentz-Berthelot, one LJ line per type."""
    lines = [
        "# general rule for shifted vs truncated",
        # D1 is truncated-unshifted; a force field with interaction.shifted: true
        # (FF-A-shift, the Kancharlapalli convention) writes RASPA2's "shifted",
        # which potentials.c matches by prefix (strncasecmp "shifted")
        "shifted" if ff.shifted else "truncated",
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
    """<guest>.def: a rigid single-site molecule carrying the yaml's critical
    constants, which RASPA2 requires even when the coefficient is supplied."""
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
    arm: Raspa2Arm | None = None,
    phi: dict[str, float] | None = None,
    seed: int | None = None,
) -> str:
    """The simulation.input text. `arm=None` writes exactly what this function
    always wrote (V0/V1 depend on it); an arm writes the benchmark move set.

    `phi`: fugacity coefficients per component. None = our Peng-Robinson at
    (T, pressure, mol_fractions); a run stated as fugacities passes phi = 1 with
    pressure = the summed fugacity, so RASPA2's f_s = phi y_s P is f_s exactly.
    `seed` = RandomSeed (None or 0: RASPA2 seeds from the clock)."""
    if phi is None:
        phi = peng_robinson(ff, T, pressure, mol_fractions).phi
    lines = [
        "SimulationType                MonteCarlo",
        f"NumberOfCycles                {cycles}",
        f"NumberOfInitializationCycles  {init_cycles}",
        # R1.8: this stream is the primary data source, not a progress log.
        f"PrintEvery                    "
        f"{print_every if print_every else (max(1, cycles // 100) if cycles else 1)}",
        f"RestartFile                   {'yes' if restart else 'no'}",
        *([f"RandomSeed                    {seed}"] if seed else []),
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
    if arm is not None:
        lines[3:3] = arm_global_lines(arm)
    for k, (name, x) in enumerate(mol_fractions.items()):
        n = 0 if counts is None else counts.get(name, 0)
        moves = (
            [
                "            TranslationProbability    0.5",
                "            SwapProbability           1.0",
            ]
            if arm is None
            else arm_component_lines(arm, len(mol_fractions))
        )
        lines += [
            f"Component {k} MoleculeName              {name}",
            "            MoleculeDefinition        Local",
            f"            MolFraction               {x}",
            "            IdealGasRosenbluthWeight  1.0",
            # our phi, not RASPA's: the same number that made our z*
            f"            FugacityCoefficient       {phi[name]:.10f}",
            *moves,
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
    arm: Raspa2Arm | None = None,
    phi: dict[str, float] | None = None,
    seed: int | None = None,
) -> Path:
    """Write a complete RASPA2 case into `out`: the CIF in the orthorhombic
    setting BRANCH_MC uses, the .def files from the yaml and simulation.input.
    Returns `out`; running it is `run_simulate`."""
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
            arm,
            phi,
            seed,
        )
    )
    return out


FINISHED = "Simulation finished on"


def arm_of(fr: FrameworkRun) -> Raspa2Arm:
    """The RASPA2 move set of a raspa2_* params file."""
    t, e, s = fr.move_probs
    return Raspa2Arm(
        kind=fr.engine.removeprefix("raspa2_"),
        p_translation=t,
        p_exchange=e,
        p_identity=s,
        k_trials=fr.trials,
        cf_lambda_bins=fr.lambda_bins,
        equilibration_cycles=fr.wang_landau_cycles,
    )


def run_raspa2(fr: FrameworkRun, paths: RunPaths) -> FrameworkResult:
    """Write the case into <run>/raspa2, run it unless it already finished, and
    parse its production stream into <run>/run.log (+ moves.json: move
    statistics and the frozen Wang-Landau weights)."""
    from .driver import FrameworkResult

    assert fr.cif is not None  # params refuses raspa2_* with an empty_box
    case = paths.directory / "raspa2"
    data = sorted(case.glob("Output/System_0/*.data"))
    done = bool(data) and FINISHED in data[0].read_text(errors="replace")[-4000:]
    atoms, meta = build(fr.cif, fr.forcefield, replicate=fr.replicate)
    if not done:
        if case.exists():
            shutil.rmtree(case)  # a killed case cannot be resumed; start it again
        write_case(
            case,
            fr.forcefield,
            fr.cif,
            fr.T,
            fr.pressure_pa,
            fr.y,
            tuple(meta["replication"]),
            int(fr.target_cycles),
            fr.burn_in,
            present_types(fr.forcefield, fr.cif),
            counts=fr.preload,
            print_every=fr.sampling_window,
            arm=arm_of(fr),
            phi=fr.phi,
            seed=fr.seed,
        )
        run_simulate(case)
        data = sorted(case.glob("Output/System_0/*.data"))
        if not data or FINISHED not in data[0].read_text(errors="replace")[-4000:]:
            raise RuntimeError(f"RASPA2 did not finish; see {case / 'raspa2.log'}")
    write_parsed(data, paths.log, paths.directory / "moves.json")
    (paths.directory / "state_point.json").write_text(
        json.dumps(
            {
                "fugacity_pa": fr.fugacity_pa,
                "pressure_pa": fr.pressure_pa,
                "y": fr.y,
                "fugacity_coefficient": fr.phi,
                "unit_cells": list(meta["replication"]),
                "framework_mass_amu": float(atoms.get_masses().sum()),
                "cif_cells": int(np.prod(meta["replication"]))
                * meta["cif_cells_per_unit_cell"],
                "move_probabilities": dict(
                    zip(
                        ("translation", "exchange", "identity_swap"),
                        fr.move_probs,
                        strict=True,
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with paths.log.open() as fh:
        last = list(csv.DictReader(fh))[-1]
    n = {g: int(last[f"N_{g}"]) for g in fr.guests}
    return FrameworkResult(
        directory=paths.directory, step=int(fr.target_cycles), n_by_species=n
    )


def raspa_executable() -> Path:
    """$RASPA_DIR/bin/simulate, or a clear error naming what to set."""
    root = os.environ.get("RASPA_DIR")
    if not root:
        raise FileNotFoundError(
            "engine.type raspa2_* needs RASPA2: export RASPA_DIR=<install prefix>"
        )
    exe = Path(root) / "bin" / "simulate"
    if not exe.exists() or shutil.which(str(exe)) is None:
        raise FileNotFoundError(f"no RASPA2 executable at {exe}")
    return exe


def run_simulate(case: Path) -> None:
    """Run RASPA2 in `case`; its own log goes to case/raspa2.log."""
    exe = raspa_executable()
    with (case / "raspa2.log").open("w") as log:
        rc = subprocess.run(
            [str(exe), "simulation.input"],
            cwd=case,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ),
            check=False,
        ).returncode
    if rc != 0:
        raise RuntimeError(f"RASPA2 exited with {rc}; see {case / 'raspa2.log'}")

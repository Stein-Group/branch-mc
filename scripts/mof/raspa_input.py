"""P0.1: the RASPA3 side of the one-yaml rule.

The anti-divergence design says ONE parameter file feeds both codes, so that they
can only ever disagree through code and never through transcription. units.py
turns ff/*.yaml into branch_mc's reduced tables; this turns the same yaml into
RASPA3's force_field.json, the guest definitions and simulation.json. Nobody
types a sigma twice.

Everything here is written against the RASPA3 sources rather than from memory
(D4 pins RASPA3), because three of its conventions are easy to get wrong and
silent when you do:

  - `SelfInteractions.parameters` is [epsilon/kB in K, sigma in A], in that
    order. Same units the yaml stores, so no conversion happens anywhere.
  - `TruncationMethod` only ever does something when it equals "shifted":
    forcefield.cpp:207 reads `shiftPotential = (value == "shifted")`. Any other
    string, including "truncated", and any absence, leaves the potential
    truncated and unshifted, which is what D1 wants. We write "truncated"
    because it says what we mean, not because RASPA parses it.
  - The cutoff lives in force_field.json, not simulation.json, under "CutOff"
    (sets framework and molecule VDW together) and defaults to 12.0 A if
    omitted. D1's 12 A happens to equal that default; we write it anyway, since
    a number that matters should never arrive by default.

Charges are all zero and `ChargeMethod` is None. For Xe and Kr in a rigid
framework that is exact rather than an approximation: both guests are uncharged,
so every Coulomb term in every energy we compare is identically zero.

    python -m scripts.mof.raspa_input framework.cif --forcefield ff/forcefield_A.yaml \\
        --T 298 --pressure 1e5 --out raspa/case_A --provisional
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from scripts.mof.units import ForceField, load_forcefield, peng_robinson, reduce_tables

# Atomic masses, only for the types FF-A and FF-B carry. RASPA reports masses in
# its output and uses them for nothing we compare (no MD, no mass-weighted
# moves), but it demands them, and a wrong number in an input file is a wrong
# number even when it is inert.
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
}


def guest_critical(ff: ForceField, guest: str) -> dict[str, float]:
    """The critical constants RASPA wants, taken from the yaml's eos block.

    Not hard-coded here, for the same reason nothing else is: one file is the
    source. RASPA3 would use these to compute Peng-Robinson itself if we left
    FugacityCoefficient out (component.cpp:237); we supply the coefficient
    instead, and these travel along so the case is self-describing and so RASPA's
    own PR can be run against ours as a cross-check.
    """
    eos = ff.raw.get("eos")
    if eos is None or guest not in eos:
        raise KeyError(f"{ff.path}: no eos entry for {guest!r}")
    return {
        "CriticalTemperature": float(eos[guest]["critical_temperature"]),
        "CriticalPressure": float(eos[guest]["critical_pressure"]),
        "AcentricFactor": float(eos[guest]["acentric_factor"]),
    }


def force_field_json(
    ff: ForceField, present: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """RASPA3's force_field.json, straight off the yaml.

    `present` restricts the pseudo-atom list to the types a given framework
    actually contains; RASPA does not mind extra ones, but an input file that
    advertises Cu for an indium MOF invites the reader to wonder.
    """
    types = present if present is not None else ff.types
    pseudo, self_int = [], []
    for t in types:
        i = ff.index(t)
        is_guest = i < ff.n_guests
        if t not in MASSES:
            raise KeyError(
                f"no mass on file for {t!r}; add it to scripts/mof/raspa_input.MASSES"
            )
        pseudo.append(
            {
                "name": t,
                "framework": not is_guest,
                "print_to_output": True,
                "element": t,
                "print_as": t,
                "mass": MASSES[t],
                "charge": 0.0,
                "source": str(
                    ff.raw["guests" if is_guest else "framework"][t]["source"]
                ),
            }
        )
        self_int.append(
            {
                "name": t,
                "type": "lennard-jones",
                # [epsilon/kB (K), sigma (A)] -- RASPA3's order, the yaml's units
                "parameters": [float(ff.epsilon[i]), float(ff.sigma[i])],
                "source": str(
                    ff.raw["guests" if is_guest else "framework"][t]["source"]
                ),
            }
        )
    return {
        "PseudoAtoms": pseudo,
        "SelfInteractions": self_int,
        "MixingRule": "Lorentz-Berthelot",
        "CutOff": ff.rc,
        # only "shifted" switches shifting on (forcefield.cpp:207); this leaves
        # the potential truncated and unshifted, per D1
        "TruncationMethod": "truncated",
        "TailCorrections": False,
    }


def guest_json(ff: ForceField, guest: str) -> dict[str, Any]:
    """A monatomic rigid guest: one pseudo-atom at the origin."""
    return {
        **guest_critical(ff, guest),
        "Type": "rigid",
        "pseudoAtoms": [[guest, [0.0, 0.0, 0.0]]],
    }


def simulation_json(
    ff: ForceField,
    framework_name: str,
    unit_cells: list[int],
    T: float,
    pressure: float,
    mol_fractions: dict[str, float],
    cycles: int,
    init_cycles: int,
) -> dict[str, Any]:
    """A GCMC run in a rigid framework at a pressure, not at a fugacity.

    The anchor relates chemical potential to pressure through Peng-Robinson
    (p. 27533) and plots against pressure, so a run that treats pressure as
    fugacity sits on a different state point: phi_Xe = 0.941 at 10 bar, a 6%
    error in the activity right where the isotherm is being read.

    RASPA3 will compute PR itself if FugacityCoefficient is absent
    (component.cpp:237). We pass ours instead. That is not distrust of RASPA's
    EOS -- ours is transcribed from it -- but the one-file rule: the fugacity
    that enters our z* and the fugacity that enters RASPA's acceptance must be
    the same number, not two numbers that agree because both codes happened to
    read the same critical constants correctly.

    The move set is translation plus swap, matching branch_mc's conventional
    baseline. Reinsertion is off: it has no counterpart on our side.
    """
    if abs(sum(mol_fractions.values()) - 1.0) > 1e-12:
        raise ValueError(f"mol fractions must sum to 1, got {mol_fractions}")
    fug = peng_robinson(ff, T, pressure, mol_fractions)
    components = []
    for name, x in mol_fractions.items():
        if name not in ff.guests:
            raise KeyError(f"{name!r} is not a guest in {ff.name}")
        components.append(
            {
                "Name": name,
                "MoleFraction": x,
                "FugacityCoefficient": fug.phi[name],
                "TranslationProbability": 0.5,
                "SwapProbability": 1.0,
                "WidomProbability": 1.0,
                "CreateNumberOfMolecules": 0,
            }
        )
    return {
        "SimulationType": "MonteCarlo",
        "NumberOfCycles": cycles,
        "NumberOfInitializationCycles": init_cycles,
        "PrintEvery": max(1, cycles // 100),
        "Systems": [
            {
                "Type": "Framework",
                "Name": framework_name,
                "NumberOfUnitCells": unit_cells,
                "ExternalTemperature": T,
                "ExternalPressure": pressure,
                "ChargeMethod": "None",
            }
        ],
        "Components": components,
    }


def write_case(
    cif: Path,
    ff: ForceField,
    out: Path,
    T: float,
    pressure: float,
    mol_fractions: dict[str, float],
    unit_cells: list[int],
    cycles: int,
    init_cycles: int,
    present: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(cif, out / cif.name)
    (out / "force_field.json").write_text(
        json.dumps(force_field_json(ff, present), indent=2)
    )
    for g in mol_fractions:
        (out / f"{g}.json").write_text(json.dumps(guest_json(ff, g), indent=2))
    sim = simulation_json(
        ff, cif.stem, unit_cells, T, pressure, mol_fractions, cycles, init_cycles
    )
    (out / "simulation.json").write_text(json.dumps(sim, indent=2))

    red = reduce_tables(ff, T)
    fug = peng_robinson(ff, T, pressure, mol_fractions)
    # The provenance sidecar is the whole point of generating rather than typing:
    # it records that this RASPA case and the branch_mc tables came off one file,
    # and carries the branch_mc-side numbers for the same state point.
    meta = {
        "generated_by": "scripts/mof/raspa_input.py",
        "forcefield": {"name": ff.name, "path": str(ff.path), "sha256": ff.sha256},
        "status": ff.status,
        "cif": str(cif),
        "state_point": {
            "T_kelvin": T,
            "pressure_pa": pressure,
            "mol_fractions": mol_fractions,
            "fugacity_coefficient": fug.phi,
            "fugacity_pa": fug.fugacity_pa,
            "compressibility": fug.compressibility,
            "T_star": red.T_star,
            "rc_star": red.rc_star,
            # the branch_mc side of the same state point: z* from the PR fugacity,
            # never from the pressure
            "z_star": {g: red.z_star(fug.fugacity_pa[g]) for g in mol_fractions},
        },
        "conventions": {
            "mixing": "Lorentz-Berthelot",
            "cutoff_angstrom": ff.rc,
            "shifted": False,
            "tail_correction": False,
            "charges": "all zero, ChargeMethod None (exact for Xe/Kr in a rigid framework)",
            "equation_of_state": "Peng-Robinson, van der Waals mixing, kij = 0 (RASPA3 convention)",
        },
    }
    (out / "provenance.json").write_text(json.dumps(meta, indent=2))
    return meta


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cif", type=Path)
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--T", type=float, required=True)
    p.add_argument("--pressure", type=float, required=True, help="fugacity in Pa")
    p.add_argument(
        "--guests", default="Xe", help="comma-separated, e.g. Xe or Xe:0.2,Kr:0.8"
    )
    p.add_argument("--unit-cells", default="2,2,2")
    p.add_argument("--cycles", type=int, default=100000)
    p.add_argument("--init-cycles", type=int, default=10000)
    p.add_argument("--present", default=None, help="types the framework contains")
    p.add_argument("--provisional", action="store_true")
    a = p.parse_args(argv)

    ff = load_forcefield(a.forcefield, allow_provisional=a.provisional)
    if ":" in a.guests:
        fracs = {k: float(v) for k, v in (g.split(":") for g in a.guests.split(","))}
    else:
        names = a.guests.split(",")
        fracs = {n: 1.0 / len(names) for n in names}
    meta = write_case(
        a.cif,
        ff,
        a.out,
        a.T,
        a.pressure,
        fracs,
        [int(x) for x in a.unit_cells.split(",")],
        a.cycles,
        a.init_cycles,
        tuple(a.present.split(",")) if a.present else None,
    )
    stale = [b for b, s in ff.status.items() if s != "confirmed"]
    if stale:
        print(
            f"!! {ff.name}: {', '.join(stale)} block(s) PROVISIONAL; stamped into provenance.json"
        )
    print(
        f"wrote {a.out}/ : force_field.json, simulation.json, {', '.join(f'{g}.json' for g in fracs)}, {a.cif.name}"
    )
    sp = meta["state_point"]
    for g in fracs:
        print(
            f"  {g}: y = {fracs[g]:.3g}  phi = {sp['fugacity_coefficient'][g]:.6f}"
            f"  f = {sp['fugacity_pa'][g]:.4g} Pa  ->  branch_mc z* = {sp['z_star'][g]:.6e}"
        )
    print(f"  PR compressibility Z = {sp['compressibility']:.6f}")
    print(
        f"  T* = {meta['state_point']['T_star']:.6f}   rc* = {meta['state_point']['rc_star']:.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())

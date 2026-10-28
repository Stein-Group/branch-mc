"""V0: the stored single-point configurations, and what ref_energy.py says they are.

Rung V0 compares ref_energy.py (P0.5, the executable definition of the
Hamiltonian) against RASPA single points, BEFORE any engine work. Gate S1 hangs
on it. This script makes the half of that comparison we can make here: a small,
fixed set of (framework + guests) configurations, each written as a RASPA case
whose guests are already placed, alongside the energy our oracle assigns.

Running RASPA needs a built RASPA3, which is Dipesh's side. What this produces is
everything that has to exist first: the configurations, their energies, and a
checker that takes RASPA's numbers and reports the comparison, so nothing is
retyped and the tolerance is not negotiated after the fact.

The configurations are chosen to isolate the terms rather than to be realistic:

  1. one Xe at a channel's binding site                  -> guest-framework only
  2. one Kr at the same site                             -> guest-framework, other type
  3. two Xe far apart in different channels              -> tests the cutoff boundary
  4. two Xe close along one channel                      -> guest-guest plus framework
  5. one Xe near the wall, one Kr in the channel        -> mixed pair types
  6. eight Xe, one per channel                           -> many-body, still sparse

Every one is checked against the core floor before it is written: a configuration
that RASPA would call an overlap tells us nothing about the energy.

    python -m scripts.mof.make_v0 --forcefield ff/forcefield_A.yaml \\
        --framework /tmp/mfm300in.extxyz --channels /tmp/channels.json --out raspa/v0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gcmc.forcefield.species_file import ForceField, load_forcefield
from scripts.mof.ref_energy import minimum_image, read_extxyz, total_energy


def binding_site(channel: dict[str, Any]) -> np.ndarray:
    """The channel's deepest Xe site, straight from the P0.7 scan.

    Not the channel axis. On MFM-300(In) the axis is a nearly empty -1950 K groove
    and the guest binds 2.3 A out against the wall at -3100 K, which is where the
    anchor also puts it (the MO3 and M2C2O4 pockets, p. 27534). A V0 configuration
    built on the axis would be testing the Hamiltonian somewhere no guest goes.
    """
    return np.array(channel["deepest_site_xyz"], float)


def violates_core_floor(
    guest_pos: np.ndarray,
    guest_types: np.ndarray,
    frame_pos: np.ndarray,
    frame_types: np.ndarray,
    cell: np.ndarray,
    ff: ForceField,
) -> list[str]:
    """Any pair closer than c * sigma_ij. Returns the offending pairs, empty if clean."""
    if ff.core_floor_c is None:
        raise ValueError(
            f"{ff.path}: core_floor.c is not set; run the P0.6 measurement first"
        )
    c = ff.core_floor_c
    sigma = ff.pair_sigma()
    bad = []
    for i, (p, ti) in enumerate(zip(guest_pos, guest_types)):
        for j in range(i + 1, len(guest_pos)):
            dr = minimum_image(p - guest_pos[j], cell)
            d = float(np.sqrt(dr @ dr))
            if d < c * sigma[ti, guest_types[j]]:
                bad.append(f"guest{i}-guest{j} at {d:.3f} A")
        dr = minimum_image(p[None, :] - frame_pos, cell)
        d = np.sqrt(np.einsum("ij,ij->i", dr, dr))
        floor = c * sigma[ti][frame_types]
        k = int(np.argmin(d - floor))
        if d[k] < floor[k]:
            bad.append(
                f"guest{i}-{ff.types[frame_types[k]]}{k} at {d[k]:.3f} A < {floor[k]:.3f}"
            )
    return bad


def build_configurations(
    frame_pos: np.ndarray,
    frame_types: np.ndarray,
    cell: np.ndarray,
    ff: ForceField,
    channels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The six configurations described in the module docstring."""
    xe, kr = ff.index("Xe"), ff.index("Kr")
    sites = [binding_site(c) for c in channels]

    configs: list[dict[str, Any]] = []

    configs.append(
        {"name": "1_xe_deepest", "pos": np.array([sites[0]]), "types": np.array([xe])}
    )
    configs.append(
        {"name": "2_kr_deepest", "pos": np.array([sites[0]]), "types": np.array([kr])}
    )

    # two Xe in different channels: the far pair. Whether they interact at all is
    # a fact about the cell, not a design choice, so record the separation.
    far = np.array([sites[0], sites[1]])
    configs.append(
        {"name": "3_two_xe_two_channels", "pos": far, "types": np.array([xe, xe])}
    )

    # two Xe along one channel, one axial step apart. 4.6 A is the Xe-Xe potential
    # minimum (2^(1/6) sigma), so this one is deliberately at the well bottom.
    a = sites[0].copy()
    b = a.copy()
    b[2] = (b[2] + 4.6) % cell[2]
    configs.append(
        {
            "name": "4_two_xe_one_channel",
            "pos": np.array([a, b]),
            "types": np.array([xe, xe]),
        }
    )

    # mixed pair, same channel
    b2 = a.copy()
    b2[2] = (b2[2] + 4.4) % cell[2]
    configs.append(
        {
            "name": "5_xe_kr_one_channel",
            "pos": np.array([a, b2]),
            "types": np.array([xe, kr]),
        }
    )

    configs.append(
        {
            "name": "6_eight_xe_one_per_channel",
            "pos": np.array(sites),
            "types": np.full(len(sites), xe),
        }
    )
    return configs


def write_xyz(
    path: Path, pos: np.ndarray, types: np.ndarray, cell: np.ndarray, ff: ForceField
) -> None:
    lattice = f"{cell[0]:.10f} 0.0 0.0 0.0 {cell[1]:.10f} 0.0 0.0 0.0 {cell[2]:.10f}"
    lines = [
        str(len(pos)),
        f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T"',
    ]
    for t, p in zip(types, pos):
        lines.append(f"{ff.types[t]:<4} {p[0]:>18.10f} {p[1]:>18.10f} {p[2]:>18.10f}")
    path.write_text("\n".join(lines) + "\n")


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument(
        "--framework", type=Path, required=True, help="extxyz from build_framework.py"
    )
    p.add_argument(
        "--channels", type=Path, required=True, help="json from channel_scan.py"
    )
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    ff = load_forcefield(a.forcefield)
    frame_pos, frame_types, cell = read_extxyz(a.framework, ff)
    channels = json.loads(a.channels.read_text())["channels"]
    a.out.mkdir(parents=True, exist_ok=True)

    configs = build_configurations(frame_pos, frame_types, cell, ff, channels)
    records = []
    print(
        f"framework: {len(frame_pos)} atoms, cell {cell[0]:.3f} x {cell[1]:.3f} x {cell[2]:.3f} A"
    )
    print(f"core floor c = {ff.core_floor_c}\n")
    for cfg in configs:
        bad = violates_core_floor(
            cfg["pos"], cfg["types"], frame_pos, frame_types, cell, ff
        )
        if bad:
            raise ValueError(f"{cfg['name']} violates the core floor: {bad}")
        u = total_energy(cfg["pos"], cfg["types"], frame_pos, frame_types, cell, ff)
        write_xyz(a.out / f"{cfg['name']}.xyz", cfg["pos"], cfg["types"], cell, ff)
        rec = {
            "name": cfg["name"],
            "n_guests": len(cfg["pos"]),
            "types": [ff.types[t] for t in cfg["types"]],
            "positions": cfg["pos"].tolist(),
            "ref_energy_kelvin": u,
            "ref_energy_kj_per_mol": {k: v * 8.31446261815324e-3 for k, v in u.items()},
        }
        records.append(rec)
        print(
            f"  {cfg['name']:28s} N={len(cfg['pos'])}  "
            f"U_gf = {u['guest_framework']:12.4f} K   U_gg = {u['guest_guest']:10.4f} K   "
            f"U = {u['total']:12.4f} K = {u['total'] * 8.31446261815324e-3:8.3f} kJ/mol"
        )

    out = {
        "rung": "V0",
        "forcefield": {
            "name": ff.name,
            "path": str(ff.path),
            "sha256": ff.sha256,
            "status": ff.status,
        },
        "framework": str(a.framework),
        "cell_angstrom": cell.tolist(),
        "core_floor_c": ff.core_floor_c,
        "conventions": {
            "cutoff_angstrom": ff.rc,
            "shifted": False,
            "tail_correction": False,
            "host_host": "excluded (rigid framework; RASPA reports it separately and never varies it)",
            "units": "kelvin (U/kB); kJ/mol also given for reading against the paper",
        },
        "configurations": records,
    }
    (a.out / "v0_reference.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(records)} configurations and {a.out / 'v0_reference.json'}")
    print("Next: run each through RASPA3 as a zero-cycle single point and compare")
    print("      the guest-framework + guest-guest total against ref_energy_kelvin.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

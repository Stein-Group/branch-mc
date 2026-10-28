"""P0.5: the Hamiltonian, written out slowly and obviously.

This script is the executable definition of what we mean by "the energy of this
configuration". Rung V0 compares it to RASPA single points before any engine work
starts, and from Phase 3 on the engine has to match it to 1e-10 relative on
stored configurations. So it deliberately does the dumb thing: explicit loops,
real units (angstrom and kelvin), no cell list, no core floor, no shortcuts that
could accidentally agree with the engine for the wrong reason.

It imports nothing from `gcmc`. An oracle that shares code with the thing it
checks is not an oracle.

The Hamiltonian, in full:

  - 12-6 Lennard-Jones, per-pair sigma_ij and eps_ij by Lorentz-Berthelot from
    the force-field yaml (P0.2, P0.4);
  - truncated at rc: pairs with r >= rc contribute exactly zero. Unshifted by
    default (pairs inside carry no offset); a force field with
    `interaction.shifted: true` subtracts u_ij(rc) from every pair inside rc,
    which is RASPA's "shifted" and the convention of the Kancharlapalli anchor;
  - no tail correction;
  - per-axis minimum image in an orthogonal cell;
  - guest-guest and guest-framework terms only. The framework is rigid, so
    host-host is a configuration-independent constant and is dropped on both
    sides (RASPA reports it as a framework self-energy and never varies it).

Energies come back in kelvin (that is, U/kB), which is the unit the force field
is stored in and the unit RASPA prints alongside its kJ/mol column.

    python -m scripts.mof.ref_energy framework.extxyz --guests guests.xyz \\
        --forcefield ff/forcefield_A.yaml --provisional
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gcmc.forcefield.species_file import ForceField, lj_energy_kelvin, load_forcefield


def minimum_image(dr: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Per-axis minimum image in an orthogonal cell. `cell` is (Lx, Ly, Lz)."""
    dr = np.asarray(dr, float)
    cell = np.asarray(cell, float)
    return dr - cell * np.round(dr / cell)


def check_cell(cell: np.ndarray, rc: float) -> None:
    """rc must fit inside half the shortest edge, or minimum image is a lie."""
    cell = np.asarray(cell, float)
    if cell.shape != (3,):
        raise ValueError("only orthogonal cells (Lx, Ly, Lz) are in scope")
    if np.any(cell <= 0):
        raise ValueError(f"cell edges must be positive, got {cell}")
    if rc > 0.5 * cell.min() + 1e-12:
        raise ValueError(
            f"cutoff {rc} exceeds half the shortest cell edge ({0.5 * cell.min():.4f}); "
            "build a larger supercell (P0.3 enforces edge >= 2 rc)"
        )


def pair_energy(
    r: float,
    ti: int,
    tj: int,
    sigma_ij: np.ndarray,
    eps_ij: np.ndarray,
    rc: float,
    shifted: bool = False,
) -> float:
    """One truncated 12-6 pair in kelvin. Beyond rc: exactly zero. Inside rc it
    is u(r), or u(r) - u(rc) when `shifted`."""
    if r >= rc:
        return 0.0
    u = float(lj_energy_kelvin(np.array(r), sigma_ij[ti, tj], eps_ij[ti, tj]))
    if shifted:
        u -= float(lj_energy_kelvin(np.array(rc), sigma_ij[ti, tj], eps_ij[ti, tj]))
    return u


def total_energy(
    guest_pos: np.ndarray,
    guest_types: np.ndarray,
    frame_pos: np.ndarray,
    frame_types: np.ndarray,
    cell: np.ndarray,
    ff: ForceField,
) -> dict[str, float]:
    """Total energy in kelvin, split into its two live terms.

    Type ids index the force field's type table (`ff.types`), for guests and
    framework atoms alike: one table, as D2 requires.
    """
    check_cell(cell, ff.rc)
    sigma_ij, eps_ij = ff.pair_sigma(), ff.pair_epsilon()
    rc = ff.rc

    u_gg = 0.0
    n = len(guest_pos)
    for i in range(n):
        for j in range(i + 1, n):
            dr = minimum_image(guest_pos[i] - guest_pos[j], cell)
            u_gg += pair_energy(
                float(np.sqrt(dr @ dr)),
                int(guest_types[i]),
                int(guest_types[j]),
                sigma_ij,
                eps_ij,
                rc,
                ff.shifted,
            )

    u_gf = 0.0
    for i in range(n):
        for k in range(len(frame_pos)):
            dr = minimum_image(guest_pos[i] - frame_pos[k], cell)
            u_gf += pair_energy(
                float(np.sqrt(dr @ dr)),
                int(guest_types[i]),
                int(frame_types[k]),
                sigma_ij,
                eps_ij,
                rc,
                ff.shifted,
            )

    return {"guest_guest": u_gg, "guest_framework": u_gf, "total": u_gg + u_gf}


def one_body_field(
    points: np.ndarray,
    guest_type: int,
    frame_pos: np.ndarray,
    frame_types: np.ndarray,
    cell: np.ndarray,
    ff: ForceField,
) -> np.ndarray:
    """The framework field a guest of one type feels at each point, in kelvin.

    This is the quantity the sweep multiplies into a trial's energy weight (A5.1)
    and the quantity P0.7 scans to find the channels. Vectorised over framework
    atoms because a grid scan is otherwise unusable, but the arithmetic is the
    same truncated (and, when the force field says so, shifted), minimum-image
    sum as `total_energy`.
    """
    check_cell(cell, ff.rc)
    sigma_ij, eps_ij = ff.pair_sigma(), ff.pair_epsilon()
    points = np.atleast_2d(np.asarray(points, float))
    cell = np.asarray(cell, float)

    sig = sigma_ij[guest_type][frame_types]  # (n_frame,)
    eps = eps_ij[guest_type][frame_types]
    out = np.empty(len(points))
    for i, p in enumerate(points):
        dr = minimum_image(p[None, :] - frame_pos, cell)
        r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
        inside = r < ff.rc
        if np.any(r[inside] == 0.0):
            # a grid point landing exactly on a framework atom: the field there is
            # infinite, which is a real answer, not a divide-by-zero to warn about
            out[i] = np.inf
            continue
        ir6 = (sig[inside] / r[inside]) ** 6
        u = 4.0 * eps[inside] * ir6 * (ir6 - 1.0)
        if ff.shifted:
            ir6c = (sig[inside] / ff.rc) ** 6
            u = u - 4.0 * eps[inside] * ir6c * (ir6c - 1.0)
        out[i] = float(np.sum(u))
    return out


def read_extxyz(
    path: str | Path, ff: ForceField
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a P0.3 framework file: positions, force-field type ids, cell.

    Written by hand rather than through ASE for the same reason the energy is:
    this side of the comparison should not depend on the library the other side
    used to build the file.
    """
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    comment = lines[1]
    lattice = comment.split('Lattice="')[1].split('"')[0].split()
    L = np.array([float(lattice[0]), float(lattice[4]), float(lattice[8])])
    if any(abs(float(lattice[i])) > 1e-8 for i in (1, 2, 3, 5, 6, 7)):
        raise ValueError(
            f"{path}: off-diagonal lattice terms; only orthogonal cells are in scope"
        )
    pos = np.empty((n, 3))
    typ = np.empty(n, int)
    for i, line in enumerate(lines[2 : 2 + n]):
        f = line.split()
        typ[i] = ff.index(f[0])
        pos[i] = [float(f[1]), float(f[2]), float(f[3])]
    return pos, typ, L


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("framework", type=Path, help="extxyz from build_framework.py")
    p.add_argument(
        "--guests", type=Path, default=None, help="extxyz of guest atoms (optional)"
    )
    p.add_argument("--forcefield", type=Path, required=True)
    p.add_argument("--provisional", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)

    ff = load_forcefield(a.forcefield, allow_provisional=a.provisional)
    frame_pos, frame_types, cell = read_extxyz(a.framework, ff)
    if a.guests is not None:
        guest_pos, guest_types, gcell = read_extxyz(a.guests, ff)
        if not np.allclose(cell, gcell):
            raise ValueError("guest and framework files disagree about the cell")
    else:
        guest_pos = np.empty((0, 3))
        guest_types = np.empty(0, int)

    u = total_energy(guest_pos, guest_types, frame_pos, frame_types, cell, ff)
    print(f"cell        : {cell[0]:.4f} x {cell[1]:.4f} x {cell[2]:.4f} A")
    print(f"framework   : {len(frame_pos)} atoms")
    print(f"guests      : {len(guest_pos)} atoms")
    print(f"U(guest-guest)     = {u['guest_guest']:.10e} K")
    print(f"U(guest-framework) = {u['guest_framework']:.10e} K")
    print(f"U(total)           = {u['total']:.10e} K")
    if a.json:
        out: dict[str, Any] = {
            "energy_kelvin": u,
            "cell": cell.tolist(),
            "n_framework": len(frame_pos),
            "n_guests": len(guest_pos),
            "forcefield": {"name": ff.name, "sha256": ff.sha256, "status": ff.status},
        }
        a.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

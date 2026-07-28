"""P0.4/P0.6: the force field in SI-ish units, and its reduction to the engine's
reduced units.

One yaml under ff/ is the single source of truth (P0.2). This module reads it,
builds the per-pair tables by Lorentz-Berthelot, and divides through by the Xe-Xe
scale so that a single-species Xe fluid with no framework lands exactly on the
engine's existing sigma=eps=1 convention. Nothing else in the pipeline is allowed
to hard-code a force-field number; ref_energy.py, channel_scan.py and the RASPA
input writer all come through here.

Conventions, fixed once:

  sigma_ref = sigma_XeXe, eps_ref = eps_XeXe   (reference scale, P0.4)
  sigma*_ij = sigma_ij / sigma_ref             (dimensionless)
  eps*_ij   = eps_ij / eps_ref
  T*        = kB T / eps_ref = T[K] / eps_ref[K]     (eps stored as eps/kB)
  rc*       = rc / sigma_ref
  z*_s      = f_s sigma_ref^3 / (kB T)         (f_s a fugacity in Pa)

`epsilon` in the yaml always means eps/kB in kelvin, so T* is a plain ratio of
kelvin and needs no Boltzmann constant at all; kB only appears in z*, where the
fugacity is a real pressure.

Run it for a look at the tables, or to measure the core floor:

    python -m scripts.mof.units ff/forcefield_A.yaml --T 298 --provisional
    python -m scripts.mof.units ff/forcefield_A.yaml --T 298 --core-floor --provisional
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# CODATA 2018; the only physical constant in the pipeline.
KB_J_PER_K = 1.380649e-23
ANGSTROM_M = 1.0e-10


@dataclass(frozen=True)
class ForceField:
    """One force-field yaml, parsed and type-ordered.

    `types` lists atom type names with the guests first (low ids, as A1.1 wants),
    each guest in yaml order, then framework types in yaml order. `sigma` and
    `epsilon` are per-type arrays in angstrom and kelvin, aligned to `types`.
    `n_guests` is the guest/framework split point: type ids < n_guests are guests.
    """

    name: str
    path: Path
    types: tuple[str, ...]
    sigma: np.ndarray  # angstrom, per type
    epsilon: np.ndarray  # eps/kB in kelvin, per type
    n_guests: int
    rc: float  # angstrom
    core_floor_c: float | None
    status: dict[str, str]  # block name -> provisional | confirmed
    sha256: str
    raw: dict[str, Any]

    @property
    def guests(self) -> tuple[str, ...]:
        return self.types[: self.n_guests]

    @property
    def framework_types(self) -> tuple[str, ...]:
        return self.types[self.n_guests :]

    def index(self, type_name: str) -> int:
        try:
            return self.types.index(type_name)
        except ValueError:
            raise KeyError(
                f"{type_name!r} is not in {self.name}; known types: {', '.join(self.types)}"
            ) from None

    def pair_sigma(self) -> np.ndarray:
        """Lorentz: sigma_ij = (sigma_i + sigma_j) / 2, angstrom."""
        return 0.5 * (self.sigma[:, None] + self.sigma[None, :])

    def pair_epsilon(self) -> np.ndarray:
        """Berthelot: eps_ij = sqrt(eps_i eps_j), kelvin."""
        return np.sqrt(self.epsilon[:, None] * self.epsilon[None, :])


def _blocks_status(doc: dict[str, Any]) -> dict[str, str]:
    return {
        block: str(doc.get(block, {}).get("status", "provisional"))
        for block in ("guests", "framework")
    }


def load_forcefield(path: str | Path, *, allow_provisional: bool = False) -> ForceField:
    """Parse a force-field yaml. Guests take the low type ids.

    A block still marked `provisional` is a transcription nobody has checked
    against the primary source; loading one without `allow_provisional` raises,
    so a provisional number cannot reach a V-rung by accident.
    """
    path = Path(path)
    text = path.read_bytes()
    doc = yaml.safe_load(text)

    if doc.get("mixing") != "lorentz-berthelot":
        raise ValueError(f"{path}: only lorentz-berthelot mixing is implemented")
    form = doc["interaction"]["form"]
    if form != "lennard-jones-12-6":
        raise ValueError(f"{path}: unsupported interaction form {form!r}")
    if doc["interaction"]["shifted"]:
        raise ValueError(
            f"{path}: shifted potentials are out of scope (truncated, unshifted)"
        )
    if doc["interaction"]["tail_correction"]:
        raise ValueError(
            f"{path}: tail corrections are out of scope (no-tail on both sides)"
        )

    status = _blocks_status(doc)
    stale = [b for b, s in status.items() if s != "confirmed"]
    if stale and not allow_provisional:
        raise ValueError(
            f"{path}: block(s) {', '.join(stale)} are still marked provisional. "
            "Confirm the numbers against the primary source and flip the status, "
            "or pass allow_provisional=True (--provisional on the CLI) to proceed "
            "with the status stamped into the output."
        )

    guest_names = [k for k in doc["guests"] if k != "status"]
    frame_names = [k for k in doc["framework"] if k != "status"]
    clash = set(guest_names) & set(frame_names)
    if clash:
        raise ValueError(f"{path}: type name(s) {sorted(clash)} appear in both blocks")

    types = tuple(guest_names + frame_names)
    sigma = np.array(
        [doc["guests"][n]["sigma"] for n in guest_names]
        + [doc["framework"][n]["sigma"] for n in frame_names],
        float,
    )
    epsilon = np.array(
        [doc["guests"][n]["epsilon"] for n in guest_names]
        + [doc["framework"][n]["epsilon"] for n in frame_names],
        float,
    )
    if np.any(sigma <= 0) or np.any(epsilon <= 0):
        raise ValueError(f"{path}: sigma and epsilon must be positive")

    c = doc.get("core_floor", {}).get("c")
    return ForceField(
        name=str(doc["name"]),
        path=path,
        types=types,
        sigma=sigma,
        epsilon=epsilon,
        n_guests=len(guest_names),
        rc=float(doc["interaction"]["rc"]),
        core_floor_c=None if c is None else float(c),
        status=status,
        sha256=hashlib.sha256(text).hexdigest(),
        raw=doc,
    )


@dataclass(frozen=True)
class ReducedTables:
    """The force field as the engine sees it: everything over the Xe-Xe scale."""

    types: tuple[str, ...]
    n_guests: int
    sigma_star: np.ndarray  # (n_types, n_types)
    eps_star: np.ndarray  # (n_types, n_types)
    T_star: float
    rc_star: float
    sigma_ref: float  # angstrom
    eps_ref: float  # kelvin
    T_kelvin: float
    core_floor_c: float | None

    def z_star(self, fugacity_pa: float) -> float:
        """Reduced activity from a fugacity in Pa: z* = f sigma_ref^3 / (kB T)."""
        return (
            fugacity_pa
            * (self.sigma_ref * ANGSTROM_M) ** 3
            / (KB_J_PER_K * self.T_kelvin)
        )

    def as_dict(self, ff: ForceField) -> dict[str, Any]:
        return {
            "forcefield": {"name": ff.name, "path": str(ff.path), "sha256": ff.sha256},
            "status": ff.status,
            "types": list(self.types),
            "n_guests": self.n_guests,
            "reference_scale": {
                "sigma_ref_angstrom": self.sigma_ref,
                "eps_ref_kelvin": self.eps_ref,
            },
            "T_kelvin": self.T_kelvin,
            "T_star": self.T_star,
            "rc_star": self.rc_star,
            "core_floor_c": self.core_floor_c,
            "sigma_star": self.sigma_star.tolist(),
            "eps_star": self.eps_star.tolist(),
        }


def reduce_tables(
    ff: ForceField, T_kelvin: float, *, reference: str = "Xe"
) -> ReducedTables:
    """Divide the force field through by the reference (Xe-Xe) scale.

    The reference type's self-pair lands on sigma* = eps* = 1 exactly, which is
    what makes the single-species reduction test against the frozen engine
    possible at all.
    """
    if T_kelvin <= 0:
        raise ValueError("temperature must be positive")
    r = ff.index(reference)
    sigma_ij = ff.pair_sigma()
    eps_ij = ff.pair_epsilon()
    sigma_ref = float(sigma_ij[r, r])
    eps_ref = float(eps_ij[r, r])
    return ReducedTables(
        types=ff.types,
        n_guests=ff.n_guests,
        sigma_star=sigma_ij / sigma_ref,
        eps_star=eps_ij / eps_ref,
        T_star=T_kelvin / eps_ref,
        rc_star=ff.rc / sigma_ref,
        sigma_ref=sigma_ref,
        eps_ref=eps_ref,
        T_kelvin=T_kelvin,
        core_floor_c=ff.core_floor_c,
    )


def lj_energy_kelvin(r_angstrom: np.ndarray, sigma: float, eps: float) -> np.ndarray:
    """4 eps ((sigma/r)^12 - (sigma/r)^6), eps and the result in kelvin.

    Same 12-6 shape as gcmc.potentials.lennard_jones._lj_from_ir6, written out
    here because P0.5 must not import the engine it is supposed to check.
    """
    ir6 = (sigma / np.asarray(r_angstrom, float)) ** 6
    return 4.0 * eps * ir6 * (ir6 - 1.0)


def core_floor_scan(
    ff: ForceField,
    T_kelvin: float,
    *,
    present: tuple[str, ...] | None = None,
    target: float = 1e-10,
    grid: int = 20001,
) -> dict[str, Any]:
    """P0.6: the largest floor fraction c whose worst pair is still colder than
    `target` in Boltzmann weight at c*sigma_ij.

    For every pair type, exp(-u_ij(c sigma_ij) / T) depends on c alone once
    scaled: u_ij(c sigma_ij) = 4 eps_ij (c^-12 - c^-6), so the pair's Boltzmann
    factor at the floor is exp(-4 (eps_ij/T) (c^-12 - c^-6)). The pair that binds
    the choice is therefore the one with the smallest eps_ij/T, and c is a
    property of the force field and temperature only, never of a configuration.
    """
    eps_ij = ff.pair_epsilon()
    iu = np.triu_indices(len(ff.types))
    # Only pairs with at least one guest can ever approach: the framework is
    # rigid, so host-host pairs enter no energy and face no floor. Letting one of
    # them bind c would drag the floor down to serve a pair that cannot move.
    live = (iu[0] < ff.n_guests) | (iu[1] < ff.n_guests)
    if present is not None:
        # A type the framework does not contain cannot bind the floor. The type
        # list comes from the P0.3 sidecar, not from the yaml, which carries
        # parameters for more elements than any one MOF uses.
        keep = np.zeros(len(ff.types), bool)
        for t in present:
            keep[ff.index(t)] = True
        live &= keep[iu[0]] & keep[iu[1]]
    iu = (iu[0][live], iu[1][live])
    if not len(iu[0]):
        raise ValueError("no live pairs to scan: check the `present` type list")
    eps_flat = eps_ij[iu]
    weakest = int(np.argmin(eps_flat))
    eps_min = float(eps_flat[weakest])

    cs = np.linspace(0.05, 1.0, grid)
    # worst case over pairs = the weakest well, since c^-12 - c^-6 > 0 below sigma
    boltz = np.exp(-4.0 * (eps_min / T_kelvin) * (cs**-12 - cs**-6))
    ok = boltz < target
    if not ok.any():
        raise RuntimeError("no admissible floor fraction on the scanned grid")
    c = float(cs[ok][-1])

    pairs: list[dict[str, Any]] = []
    for a, b in zip(*iu):
        e = float(eps_ij[a, b])
        pairs.append(
            {
                "pair": f"{ff.types[a]}-{ff.types[b]}",
                "sigma_ij_angstrom": float(ff.pair_sigma()[a, b]),
                "eps_ij_kelvin": e,
                "boltzmann_at_floor": float(
                    np.exp(-4.0 * (e / T_kelvin) * (c**-12 - c**-6))
                ),
                "u_at_floor_kelvin": float(4.0 * e * (c**-12 - c**-6)),
            }
        )
    pairs.sort(key=lambda p: -float(p["boltzmann_at_floor"]))
    return {
        "T_kelvin": T_kelvin,
        "target_boltzmann": target,
        "c": c,
        "binding_pair": pairs[0]["pair"],
        "eps_min_kelvin": eps_min,
        "pairs": pairs,
    }


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("forcefield", type=Path)
    p.add_argument("--T", type=float, required=True, help="temperature in kelvin")
    p.add_argument("--reference", default="Xe")
    p.add_argument(
        "--fugacity", type=float, default=None, help="a fugacity in Pa, to report z*"
    )
    p.add_argument("--core-floor", action="store_true", help="run the P0.6 measurement")
    p.add_argument(
        "--present",
        default=None,
        help="comma-separated types the system actually contains (e.g. Xe,Kr,In,C,O,H); "
        "defaults to every type in the yaml",
    )
    p.add_argument(
        "--provisional", action="store_true", help="proceed on unconfirmed numbers"
    )
    p.add_argument("--json", type=Path, default=None, help="write the tables here")
    a = p.parse_args(argv)

    ff = load_forcefield(a.forcefield, allow_provisional=a.provisional)
    red = reduce_tables(ff, a.T, reference=a.reference)

    stale = [b for b, s in ff.status.items() if s != "confirmed"]
    if stale:
        print(
            f"!! {ff.name}: {', '.join(stale)} block(s) PROVISIONAL, numbers unconfirmed"
        )
    print(f"{ff.name} ({ff.path}) sha256 {ff.sha256[:12]}")
    print(f"  types      : {', '.join(ff.types)}  (guests: {', '.join(ff.guests)})")
    print(
        f"  reference  : {a.reference}-{a.reference}  sigma_ref = {red.sigma_ref:.5f} A"
        f"  eps_ref/kB = {red.eps_ref:.4f} K"
    )
    print(f"  T          : {a.T:g} K   ->  T* = {red.T_star:.6f}")
    print(f"  rc         : {ff.rc:g} A  ->  rc* = {red.rc_star:.6f}")
    if a.fugacity is not None:
        print(f"  f = {a.fugacity:g} Pa  ->  z* = {red.z_star(a.fugacity):.6e}")
    w = max(len(t) for t in ff.types) + 1
    print("  sigma*_ij / eps*_ij:")
    print("    " + " " * w + "".join(f"{t:>18}" for t in ff.types))
    for i, ti in enumerate(ff.types):
        cells = "".join(
            f"{red.sigma_star[i, j]:>9.4f}/{red.eps_star[i, j]:<8.4f}"
            for j in range(len(ff.types))
        )
        print(f"    {ti:<{w}}" + cells)

    out: dict[str, Any] = red.as_dict(ff)
    if a.core_floor:
        present = tuple(a.present.split(",")) if a.present else None
        scan = core_floor_scan(ff, a.T, present=present)
        out["core_floor_scan"] = scan
        print(
            f"\n  P0.6 core floor at T = {a.T:g} K, target Boltzmann < {scan['target_boltzmann']:g}"
        )
        print(
            f"    c = {scan['c']:.4f}   binding pair {scan['binding_pair']}"
            f" (eps_ij/kB = {scan['eps_min_kelvin']:.4f} K)"
        )
        for row in scan["pairs"][:5]:
            print(
                f"      {row['pair']:<10} sigma_ij = {row['sigma_ij_angstrom']:6.3f} A"
                f"  u(c sigma) = {row['u_at_floor_kelvin']:.4e} K"
                f"  exp(-u/T) = {row['boltzmann_at_floor']:.3e}"
            )

    if a.json:
        a.json.write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())


# --------------------------------------------------------------- Peng-Robinson


@dataclass(frozen=True)
class Fugacity:
    """What a pressure means, once the gas is not ideal."""

    phi: dict[str, float]  # fugacity coefficient per species
    fugacity_pa: dict[str, float]  # f_s = y_s phi_s P
    compressibility: float  # the root PR was evaluated at
    pressure_pa: float
    T_kelvin: float
    mol_fractions: dict[str, float]


def peng_robinson(
    ff: ForceField, T_kelvin: float, pressure_pa: float, mol_fractions: dict[str, float]
) -> Fugacity:
    """Peng-Robinson fugacity coefficients, van der Waals mixing, kij = 0.

    Transcribed from RASPA3's own implementation
    (src/raspakit/utilities/equation_of_states.cpp), not from a textbook, because
    the point is for both codes to land on the same fugacity:

        kappa = 0.37464 + 1.54226 w - 0.26992 w^2
        alpha = (1 + kappa (1 - sqrt(Tr)))^2
        a_i   = 0.45724 alpha (R Tc)^2 / Pc,     b_i = 0.07780 R Tc / Pc
        A_i   = a_i P / (R T)^2,                 B_i = b_i P / (R T)
        a_ij  = (1 - kij) sqrt(a_i a_j),         A_ij likewise
        Amix  = sum_i sum_j y_i y_j a_ij  * P/(RT)^2
        Bmix  = sum_i y_i b_i * P/(RT)
        Z^3 + (Bmix - 1) Z^2 + (Amix - 3 Bmix^2 - 2 Bmix) Z
              - (Amix Bmix - Bmix^2 - Bmix^3) = 0
        ln phi_i = (B_i/Bmix)(Z - 1) - ln(Z - Bmix)
                   - Amix/(2 sqrt2 Bmix) (sum_k 2 y_k A_ik / Amix - B_i/Bmix)
                     ln((Z + (1+sqrt2) Bmix) / (Z + (1-sqrt2) Bmix))

    RASPA3 applies this only when the user leaves FugacityCoefficient out
    (component.cpp:237). We compute it here and hand RASPA the answer explicitly,
    so the fugacity is one number from one file rather than two codes agreeing by
    luck about critical constants.

    The gas root (largest real Z) is taken: at 298 K these guests are supercritical
    or well above their dew points over the whole 0.1-10 bar range of interest.
    """
    if abs(sum(mol_fractions.values()) - 1.0) > 1e-12:
        raise ValueError(f"mol fractions must sum to 1, got {mol_fractions}")
    eos = ff.raw.get("eos")
    if eos is None:
        raise ValueError(
            f"{ff.path}: no `eos` block; cannot convert a pressure to a fugacity"
        )
    if eos["form"] != "peng-robinson":
        raise ValueError(f"{ff.path}: unsupported eos form {eos['form']!r}")
    kij = float(eos.get("binary_interaction_parameter", 0.0))

    names = list(mol_fractions)
    R = 8.31446261815324  # J/mol/K, CODATA 2018
    RT = R * T_kelvin
    a, b = {}, {}
    for s in names:
        if s not in eos:
            raise KeyError(
                f"{ff.path}: no critical constants for {s!r} in the eos block"
            )
        Tc = float(eos[s]["critical_temperature"])
        Pc = float(eos[s]["critical_pressure"])
        w = float(eos[s]["acentric_factor"])
        kappa = 0.37464 + 1.54226 * w - 0.26992 * w * w
        alpha = (1.0 + kappa * (1.0 - math.sqrt(T_kelvin / Tc))) ** 2
        a[s] = 0.45724 * alpha * (R * Tc) ** 2 / Pc
        b[s] = 0.07780 * (R * Tc) / Pc

    A = {s: a[s] * pressure_pa / RT**2 for s in names}
    B = {s: b[s] * pressure_pa / RT for s in names}
    Aij = {(i, j): (1.0 - kij) * math.sqrt(A[i] * A[j]) for i in names for j in names}
    Amix = (
        sum(
            mol_fractions[i] * mol_fractions[j] * (1.0 - kij) * math.sqrt(a[i] * a[j])
            for i in names
            for j in names
        )
        * pressure_pa
        / RT**2
    )
    Bmix = sum(mol_fractions[i] * b[i] for i in names) * pressure_pa / RT

    roots = np.roots(
        [
            1.0,
            Bmix - 1.0,
            Amix - 3.0 * Bmix**2 - 2.0 * Bmix,
            -(Amix * Bmix - Bmix**2 - Bmix**3),
        ]
    )
    real = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-10 and r.real > Bmix)
    if not real:
        raise RuntimeError(f"no physical PR root at {T_kelvin} K, {pressure_pa} Pa")
    Z = real[-1]  # gas root

    phi = {}
    for i in names:
        temp = sum(2.0 * mol_fractions[k] * Aij[(i, k)] for k in names)
        phi[i] = math.exp(
            (B[i] / Bmix) * (Z - 1.0)
            - math.log(Z - Bmix)
            - (Amix / (2.0 * math.sqrt(2.0) * Bmix))
            * (temp / Amix - B[i] / Bmix)
            * math.log(
                (Z + (1.0 + math.sqrt(2.0)) * Bmix)
                / (Z + (1.0 - math.sqrt(2.0)) * Bmix)
            )
        )
    return Fugacity(
        phi=phi,
        fugacity_pa={s: mol_fractions[s] * phi[s] * pressure_pa for s in names},
        compressibility=Z,
        pressure_pa=pressure_pa,
        T_kelvin=T_kelvin,
        mol_fractions=dict(mol_fractions),
    )

"""P2 of multispecies_plan.md: everything species-indexed for one typed run.

`sim_setup.from_params` builds exactly one of these when `potential.species_file`
is present, and hangs it off `Config.species`. Every array is `(T,)` or `(T, T)`
and sized from the species file; there is no `n_species == 2` anywhere and no `A`
or `B` in a signature (M13). A run without a species file leaves
`Config.species` at None and takes the existing scalar path unchanged, which is
what the single-species bit-identity gates pin.

The split the plan insists on (M9): the species file is the MODEL -- sigma,
epsilon, lambda, p_trial, symbol, k_ij, rc, the core floor, the auxiliary
multiplier -- and `params.json` is the EXPERIMENT. The only species-indexed thing
in the run config is the state point, because that is the axis you sweep.

UNITS. Everything here is engine-REDUCED: sigma in units of the reference sigma,
epsilon in units of the reference epsilon, lambda as a ratio, temperature and mu
in reduced units. The species FILE may be written in Angstroms and Kelvin;
`species_file` converts at load, and that is the single boundary (R10). Nothing
below this line has seen a physical unit, which is why a bare number here can be
compared against another bare number without asking where either came from.

Three mutually exclusive state-point routes, exactly one present (M10):

    ensemble.mu:       {A: ..., ...}   ->  z_s = exp(beta mu_s) / lambda_s^d
    ensemble.fugacity: {A: ..., ...}   ->  z_s = beta f_s
    ensemble.z:        {A: ..., ...}   ->  z_s given directly

`lambda_s` is consulted only on the `mu` route. Supplying it under the other two
is not an error, since the file is the model and may be used with either route,
but `route` and `lam_consulted` are recorded so a later reader cannot
misinterpret a stored mu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._types import FloatArray
from .potentials import overlap
from .potentials.pair_table import PairTable
from .species_file import ForceField, load_forcefield, reduce_tables

# The three routes, in the order their errors list them.
STATE_POINT_ROUTES = ("mu", "fugacity", "z")


@dataclass(frozen=True)
class SpeciesSpec:
    """The species-indexed half of a typed run's identity.

    Frozen for the same reason `Config` is: nothing may mutate the physics
    mid-run, and the run fingerprint hashes a canonical rendering of the fields
    below (`Config.species_identity`).
    """

    names: tuple[str, ...]
    symbol: tuple[str, ...]
    dimension: int

    # model, from the species file
    sigma_self: FloatArray  # (T,) per-species sigma
    eps_self: FloatArray  # (T,) per-species epsilon
    sigma: FloatArray  # (T, T) pair sigma, Lorentz
    eps: FloatArray  # (T, T) pair epsilon, Berthelot + k_ij
    k_ij: FloatArray  # (T, T) symmetric, zero diagonal
    lam: FloatArray  # (T,) thermal wavelength
    p_trial: FloatArray  # (T,) trial-cloud species probabilities
    core_floor_c: float
    core_floor: FloatArray  # (T, T) = c * sigma_ij
    rc: float
    aux_cutoff: FloatArray | None  # (T, T) = a * sigma_ij, or None when off
    aux_control: bool
    table: PairTable  # what the hot path and the sweeps read
    volume_of_molecule: FloatArray  # (T,) hard-sphere volume at sigma_s

    # experiment, from params.json
    route: str  # "mu" | "fugacity" | "z"
    lam_consulted: bool
    mu: FloatArray  # (T,) chemical potential
    z: FloatArray  # (T,) activity, the authoritative state point

    # provenance
    species_file: str
    sha256: str
    name: str

    @property
    def n(self) -> int:
        """Number of species, guests and framework together."""
        return len(self.names)

    def index(self, name: str) -> int:
        """Type id of a species by name, raising KeyError with the known names.

        Named lookup rather than positional, because a params file states species
        by name and the id is an implementation detail of the tables.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not a species of {self.name}; known: "
                f"{', '.join(self.names)}"
            ) from None

    @property
    def max_core_floor(self) -> float:
        """Largest per-pair overlap floor over the whole table.

        A CONSERVATIVE STAND-IN for the legacy scalar geometric guards (pore
        gaps, cell pitch floors, buffer minima): a guard that passes on the
        maximum passes on every pair. Nothing physical reads it -- the hot path
        reads `core_floor` per pair.
        """
        return overlap.max_core_floor(self.core_floor)

    @property
    def max_aux_cutoff(self) -> float | None:
        """Largest per-pair auxiliary radius, or None when the auxiliary is off.

        The same upper-bound convention as `max_core_floor`, for the guards that
        need one number: the periodic guard needs the widest distance the sweep
        can read, and taking the maximum is what makes it safe for every pair.
        """
        return None if self.aux_cutoff is None else float(self.aux_cutoff.max())

    @property
    def list_cutoff(self) -> float:
        """Cell-list cutoff: the largest distance any query can care about.

        `max(rc, max_ij core_floor_ij)` per P3. With a floor fraction below 1 the
        rc term always wins for a truncated LJ, but stating the max rather than
        assuming it is what keeps the 27-cell stencil argument true if a floor is
        ever raised above the cutoff.
        """
        return max(self.rc, self.max_core_floor)

    @property
    def sweep_reach(self) -> float:
        """The largest distance the channel sweep will read, which sets the
        periodic-guard margin (P5). The auxiliary, when on, is strictly shorter
        than rc, so it is the binding margin exactly then."""
        if self.aux_cutoff is None:
            return max(self.rc, self.max_core_floor)
        return max(float(self.aux_cutoff.max()), self.max_core_floor)

    def packing_fraction(self, counts: Any, total_volume: float) -> float:
        """sum_s n_s v_s / V (P2). `counts` is a per-species particle count."""
        n = np.asarray(counts, float)
        if n.shape != (self.n,):
            raise ValueError(f"counts must have shape ({self.n},), got {n.shape}")
        return float((n * self.volume_of_molecule).sum() / total_volume)


def _molecule_volume(sigma: FloatArray, dimension: int) -> FloatArray:
    """Hard-sphere volume (area in 2D) of a sphere of diameter sigma.

    Same arithmetic as the scalar path's `volume_of_molecule`, which is written
    in terms of `radius = sigma/2`, so at one species with sigma = 1 the two
    agree to the last bit.
    """
    r = 0.5 * np.asarray(sigma, float)
    if dimension == 2:
        return math.pi * r * r
    return (4.0 / 3.0) * math.pi * r**3


def one_route(ens: dict[str, Any]) -> str:
    """The single state-point route an ensemble block states: mu, fugacity or z.

    Exactly one is required (M10). Zero is under-specified and more than one is
    ambiguous -- and silently preferring one would make the other a setting that
    appears to work and does nothing -- so both cases raise and name what was
    found.
    """
    present = [k for k in STATE_POINT_ROUTES if k in ens]
    if not present:
        raise ValueError(
            "a typed (multi-species) run needs exactly one state-point route: "
            f"ensemble.{', ensemble.'.join(STATE_POINT_ROUTES)}"
        )
    if len(present) > 1:
        raise ValueError(
            f"ensemble states {len(present)} state-point routes "
            f"({', '.join(present)}); they are mutually exclusive, give one"
        )
    return present[0]


def _route_values(
    ens: dict[str, Any], route: str, names: tuple[str, ...]
) -> FloatArray:
    """Read one route's map into a per-species array, one entry per species."""
    raw = ens[route]
    if not isinstance(raw, dict):
        raise ValueError(
            f"ensemble.{route} must be a map with exactly one entry per species "
            f"({', '.join(names)}) in a typed run, got {type(raw).__name__}. A "
            "scalar mu means a single-species run and needs no species file."
        )
    unknown = sorted(set(raw) - set(names))
    if unknown:
        raise ValueError(
            f"ensemble.{route} names unknown species {unknown}; the species file "
            f"declares {', '.join(names)}"
        )
    missing = [s for s in names if s not in raw]
    if missing:
        raise ValueError(
            f"ensemble.{route} has no entry for {missing}; every species needs a "
            "state point, there is no default"
        )
    out = np.empty(len(names), float)
    for i, s in enumerate(names):
        v = raw[s]
        if isinstance(v, list):
            raise ValueError(
                f"ensemble.{route}[{s}] is a list. A sweep is a list OF maps, not "
                "a map of lists: the state point is one point per run and the "
                "launcher expands the outer list into one run per point."
            )
        out[i] = float(v)
    if not np.all(np.isfinite(out)):
        bad = [s for s, v in zip(names, out) if not math.isfinite(v)]
        raise ValueError(f"ensemble.{route} is not finite for {bad}")
    if route in ("fugacity", "z") and np.any(out <= 0.0):
        bad = [s for s, v in zip(names, out) if v <= 0.0]
        raise ValueError(
            f"ensemble.{route} must be positive (it is an activity, not a log) "
            f"but is <= 0 for {bad}"
        )
    return out


def resolve_state_point(
    ens: dict[str, Any],
    names: tuple[str, ...],
    *,
    beta: float,
    dimension: int,
    lam: FloatArray,
) -> tuple[str, FloatArray, FloatArray, bool]:
    """(route, mu, z, lam_consulted) from the ensemble block (M10).

    On the `mu` route mu is stored as given and z derived from it; on the other
    two z is stored as given and mu derived, so whichever number the user typed
    survives round-tripping exactly and only the derived one carries the
    exp/log rounding.
    """
    route = one_route(ens)
    vals = _route_values(ens, route, names)
    lam = np.asarray(lam, float)
    if route == "mu":
        mu = vals
        z = np.exp(beta * mu) / lam**dimension
        return route, mu, z, True
    z = beta * vals if route == "fugacity" else vals
    mu = np.log(z * lam**dimension) / beta
    return route, mu, z, False


def build_species_spec(
    species_file: str,
    ens: dict[str, Any],
    *,
    beta: float,
    dimension: int,
    allow_provisional: bool = False,
) -> SpeciesSpec:
    """Load the species file, apply the state-point route, and freeze the result.

    Every check that is a property of the FILE already fired inside
    `load_forcefield` (M15). Everything checked here is a property of the
    SUBMISSION: which route, one entry per species, no unknown species.
    """
    ff: ForceField = load_forcefield(species_file, allow_provisional=allow_provisional)
    if ff.schema < 2:
        raise ValueError(
            f"{species_file}: schema {ff.schema} is a MOF force field "
            f"(`guests:` + `framework:`), not a species file. potential."
            "species_file wants a schema-2 file declaring `species:` and `units:`."
        )
    if ff.framework_types:
        raise ValueError(
            f"{species_file}: a species file for the bulk engine carries no "
            f"framework, but declares {list(ff.framework_types)}. The framework "
            "path is the MOF driver, which takes a schema-1 force field."
        )
    red = reduce_tables(ff, 1.0 if ff.units == "reduced" else _needs_T(species_file))
    assert red.core_floor_c is not None  # schema 2 requires it, checked in the loader

    names = ff.types
    route, mu, z, lam_consulted = resolve_state_point(
        ens, names, beta=beta, dimension=dimension, lam=ff.lam
    )
    table = PairTable.from_reduced(
        names, ff.n_guests, red.sigma_star, red.eps_star, red.core_floor_c
    )
    return SpeciesSpec(
        names=names,
        symbol=ff.symbol,
        dimension=dimension,
        sigma_self=np.diag(red.sigma_star).copy(),
        eps_self=np.diag(red.eps_star).copy(),
        sigma=red.sigma_star,
        eps=red.eps_star,
        k_ij=ff.k_ij,
        lam=ff.lam,
        p_trial=ff.p_trial,
        core_floor_c=red.core_floor_c,
        core_floor=table.core_floor,
        rc=red.rc_star,
        aux_cutoff=red.aux_cutoff_star,
        aux_control=ff.aux_control,
        table=table,
        volume_of_molecule=_molecule_volume(np.diag(red.sigma_star), dimension),
        route=route,
        lam_consulted=lam_consulted,
        mu=mu,
        z=z,
        species_file=str(species_file),
        sha256=ff.sha256,
        name=ff.name,
    )


def _needs_T(species_file: str) -> float:
    """An Angstrom-Kelvin species file cannot be reduced without its temperature.

    The engine's `potential.T` is already reduced (T*), so it is not the kelvin
    number `reduce_tables` would need, and silently reusing it would rescale
    every epsilon by the wrong factor. Rather than invent a second temperature
    key, the bulk typed path takes reduced species files only; the conversion
    belongs upstream, where the reduced file is written.
    """
    raise ValueError(
        f"{species_file}: units are angstrom-kelvin, but the bulk typed engine "
        "works in reduced units and its potential.T is already T*. Write the "
        "species file with `units: reduced` (see ff/species_xekr_reduced.yaml, "
        "which is ff/forcefield_A.yaml divided through by Xe-Xe) so there is one "
        "temperature in the submission and no hidden conversion."
    )

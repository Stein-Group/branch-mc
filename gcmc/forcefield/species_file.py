"""P0.4/P0.6: the force field in SI-ish units, and its reduction to the engine's
reduced units. Also (P1 of multispecies_plan.md) the species file.

One yaml under ff/ is the single source of truth (P0.2). This module reads it,
builds the per-pair tables by Lorentz-Berthelot with optional `k_ij` cross terms,
and -- for an Angstrom-Kelvin file -- divides through by a reference self-pair so
that a single-species fluid with no framework lands exactly on the engine's
existing sigma=eps=1 convention. Nothing else in the pipeline is allowed to
hard-code a force-field number; ref_energy.py, channel_scan.py and the RASPA
input writer all come through here, and so does the multi-species channel work.

There is one loader and one validator for both file kinds. The two kinds are
told apart by `schema`, not by guesswork:

  schema: 1   the MOF force fields, `guests:` + `framework:`, Angstrom-Kelvin,
              reference supplied by the caller. ff/forcefield_A.yaml and
              ff/forcefield_B.yaml. Untouched by the multispecies plan.
  schema: 2   a species file, `species:`, `units:` mandatory, per-species
              `lambda` / `p_trial` / `symbol`, per-pair `k_ij`, global `aux`.
              ff/species_*.yaml.

Conventions, fixed once:

  sigma_ij  = (sigma_i + sigma_j) / 2                        (Lorentz, M1)
  eps_ij    = (1 - k_ij) sqrt(eps_i eps_j),  eps_ii = eps_i  (Berthelot + M1)
  sigma*_ij = sigma_ij / sigma_ref                           (dimensionless)
  eps*_ij   = eps_ij / eps_ref
  T*        = kB T / eps_ref = T[K] / eps_ref[K]     (eps stored as eps/kB)
  rc*       = rc / sigma_ref
  z*_s      = f_s sigma_ref^3 / (kB T)         (f_s a fugacity in Pa)

`epsilon` in an Angstrom-Kelvin yaml always means eps/kB in kelvin, so T* is a
plain ratio of kelvin and needs no Boltzmann constant at all; kB only appears in
z*, where the fugacity is a real pressure.

A `units: reduced` file is already in engine units: sigma_ref = eps_ref = 1 by
definition, it carries no `reference`, and `reduce_tables` is a passthrough.
Every sigma in such a file is free; none is pinned to 1. In reduced mode the
`T_kelvin` field of ReducedTables carries T* itself, and `z_star` refuses to run
because there is no kB scale to convert a pressure with (use the `z` or
`fugacity` route of M10 instead).

Lives under gcmc/ rather than scripts/ because gcmc.run.sim_setup validates a
species file at from_params time and the engine package must not depend on
scripts/, which is not installed. scripts/mof/units.py re-exports every name
here, so the MOF pipeline and its CLI are unchanged.

Run it for a look at the tables, or to measure the core floor:

    python -m gcmc.forcefield.species_file ff/forcefield_A.yaml --T 298 --provisional
    python -m scripts.mof.units ff/forcefield_A.yaml --T 298 --core-floor --provisional
    python -m scripts.mof.units ff/species_three_reduced.yaml --T 1.2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase.data import chemical_symbols

# CODATA 2018; the only physical constant in the pipeline.
KB_J_PER_K = 1.380649e-23
ANGSTROM_M = 1.0e-10

# The accepted unit systems. Absent means angstrom-kelvin, which is what
# every schema-1 file is; a schema-2 file must say which one it is.
UNIT_SYSTEMS = ("reduced", "angstrom-kelvin")

# Below this multiple of sigma_ij the auxiliary proposal radius sits inside the
# LJ first shell, sd(Delta_corr) explodes and acceptance collapses while the run
# still completes at a plausible density. Same number as the warning in
# scripts/mof/submit_mof.py, kept as one constant so the two cannot drift.
AUX_FIRST_SHELL = 1.4

# Sum tolerance for p_trial. Absolute, because the quantity is a probability.
_P_TRIAL_TOL = 1e-9

_SPECIES_KEYS = frozenset({"sigma", "epsilon", "lambda", "p_trial", "symbol", "source"})
_FRAMEWORK_KEYS = frozenset({"sigma", "epsilon", "symbol", "source"})
_AUX_KEYS = frozenset({"multiplier", "control"})
_ELEMENTS = frozenset(chemical_symbols[1:])  # index 0 is the "X" placeholder


@dataclass(frozen=True)
class ForceField:
    """One force-field or species yaml, parsed and type-ordered.

    `types` lists atom type names with the guests first (low ids, as A1.1 wants),
    each guest in yaml order, then framework types in yaml order. `sigma` and
    `epsilon` are per-type arrays, in angstrom and kelvin for an Angstrom-Kelvin
    file and already dimensionless for a reduced one. `n_guests` is the
    guest/framework split point: type ids < n_guests are guests. A species file
    is all guests, so `n_guests == len(types)`.

    The multispecies fields are per type and always present, so nothing
    downstream needs to know which kind of file it came from:

      lam       thermal wavelength, default 1.0, framework entries 1.0
      p_trial   trial-cloud species probabilities, summing to 1 over the
                guests, 0.0 on framework entries
      symbol    element symbol for trajectory I/O, defaulting to the type name
      k_ij      (n, n) symmetric cross-term matrix, zero diagonal, zero by
                default, so a file without a k_ij block is exactly
                Lorentz-Berthelot
    """

    name: str
    path: Path
    schema: int
    units: str
    reference: str | None
    types: tuple[str, ...]
    sigma: np.ndarray  # per type
    epsilon: np.ndarray  # eps/kB in kelvin (angstrom-kelvin) or reduced
    lam: np.ndarray  # thermal wavelength per type
    p_trial: np.ndarray  # per type, guests sum to 1, framework 0
    symbol: tuple[str, ...]  # per type
    k_ij: np.ndarray  # (n, n) symmetric, zero diagonal
    n_guests: int
    rc: float
    core_floor_c: float | None
    aux_multiplier: float | None  # None = auxiliary off
    aux_control: bool  # the exact no-op control arm (M12)
    status: dict[str, str]  # block name -> provisional | confirmed
    sha256: str
    raw: dict[str, Any]

    @property
    def guests(self) -> tuple[str, ...]:
        """The mobile species, which occupy the LOW type ids by convention.

        That ordering is load-bearing: `p_trial` is normalised over this block
        alone, and the insertion draw indexes it directly.
        """
        return self.types[: self.n_guests]

    @property
    def framework_types(self) -> tuple[str, ...]:
        """The frozen species, which take the type ids above the guests.

        Empty for a bulk fluid. These are never inserted, removed, moved or
        counted; they interact by their table entries alone.
        """
        return self.types[self.n_guests :]

    def index(self, type_name: str) -> int:
        """Type id by name over guests AND framework, raising with the known names.

        One namespace for both, so a k_ij entry between a guest and a framework
        atom is stated the same way as one between two guests.
        """
        try:
            return self.types.index(type_name)
        except ValueError:
            raise KeyError(
                f"{type_name!r} is not in {self.name}; known types: {', '.join(self.types)}"
            ) from None

    def pair_sigma(self) -> np.ndarray:
        """Lorentz: sigma_ij = (sigma_i + sigma_j) / 2."""
        return 0.5 * (self.sigma[:, None] + self.sigma[None, :])

    def pair_epsilon(self) -> np.ndarray:
        """Berthelot with cross terms: eps_ij = (1 - k_ij) sqrt(eps_i eps_j).

        The diagonal is copied from the input, never computed. At k = 0 the
        computed diagonal sqrt(e*e) already equals e for every double in the
        normal range, but relying on that is relying on a rounding theorem: the
        copy makes the single-species reduction bit-identical by construction,
        which is what the frozen gates pin against.
        """
        eps = np.sqrt(self.epsilon[:, None] * self.epsilon[None, :])
        if self.k_ij.any():
            eps = (1.0 - self.k_ij) * eps
        np.fill_diagonal(eps, self.epsilon)
        return eps

    def aux_cutoff(self) -> np.ndarray | None:
        """Per-pair auxiliary cutoff a * sigma_ij, or None when off.

        One dimensionless multiplier, so the proposal radius tracks the pair it
        proposes into and is automatically shorter for the smaller pairs.

        THE CONTROL ARM RETURNS `rc` FOR EVERY PAIR, and this is the correction
        of a real defect rather than a refinement. It used to return None -- the
        same value as `aux: none` -- so `aux: {control: true}` produced a run with
        no auxiliary machinery in it at all. A control arm that IS the thing it
        controls against cannot fail, and the docstring calling it "the exact
        no-op arm the tuning sweep uses as its own reference" is what made it look
        like a gate.

        Returning `rc` instead turns the machinery ON at a radius where the
        correction's value is identically zero: the two shell energies are the
        same sum over the same pairs, so the difference is 0.0 and not 1e-17, and
        the chain must reproduce the auxiliary-off chain BIT-IDENTICALLY. That is
        It is only a test if the code under test runs.
        See `tests/engine/test_aux_noop_control.py` for the untyped twin of this
        argument, where the same vacuity appears one layer down as
        `aux_on = ljbranch and rc_sel < uc`.
        """
        if self.aux_control:
            return np.full_like(self.pair_sigma(), self.rc)
        if self.aux_multiplier is None:
            return None
        return self.aux_multiplier * self.pair_sigma()


def _blocks_status(doc: dict[str, Any]) -> dict[str, str]:
    """Status of every parameter block the file actually carries.

    Only blocks that are present are reported, so a species file with no
    framework does not fail the provisional gate on a block it does not have.
    """
    return {
        block: str(doc[block].get("status", "provisional"))
        for block in ("guests", "species", "framework")
        if isinstance(doc.get(block), dict)
    }


def _entry_keys(
    path: Path, block: str, name: str, entry: Any, allowed: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(
            f"{path}: {block}.{name} must be a map, got {type(entry).__name__}"
        )
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise ValueError(
            f"{path}: {block}.{name} has unknown key(s) {unknown}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )
    return entry


def _read_units(path: Path, doc: dict[str, Any], schema: int) -> tuple[str, str | None]:
    """`units` and `reference`, with the M3 iff-rule enforced at the loader."""
    if schema >= 2 and "units" not in doc:
        raise ValueError(
            f"{path}: a schema-2 species file must declare `units`, one of "
            f"{', '.join(UNIT_SYSTEMS)}"
        )
    units = str(doc.get("units", "angstrom-kelvin"))
    if units not in UNIT_SYSTEMS:
        raise ValueError(
            f"{path}: units {units!r} is not one of {', '.join(UNIT_SYSTEMS)}"
        )
    reference = doc.get("reference")
    if units == "reduced" and reference is not None:
        raise ValueError(
            f"{path}: a reduced file carries no `reference` (it declares "
            f"sigma_ref = eps_ref = 1 by definition), but names {reference!r}"
        )
    if units == "angstrom-kelvin" and schema >= 2 and reference is None:
        raise ValueError(
            f"{path}: units angstrom-kelvin needs a `reference` naming the "
            "species whose self-pair sets the reduction scale"
        )
    return units, None if reference is None else str(reference)


def _read_k_ij(
    path: Path, doc: dict[str, Any], mixing: str, types: tuple[str, ...]
) -> np.ndarray:
    """Parse the `k_ij` block into a symmetric matrix with a zero diagonal.

    Stated as a nested map, partner under species, in either order:

        k_ij:
          A:
            B: 0.2
            C: -0.1
          B:
            C: 0.0

    Nested rather than an "A-B" string key so a species name may contain any
    character without inventing a delimiter escape.
    """
    n = len(types)
    k = np.zeros((n, n), float)
    block = doc.get("k_ij") or {}
    if not block:
        return k
    if mixing != "lorentz-berthelot-kij":
        raise ValueError(
            f"{path}: a k_ij block requires `mixing: lorentz-berthelot-kij`, "
            f"but the file says {mixing!r}"
        )
    if not isinstance(block, dict):
        raise ValueError(f"{path}: k_ij must be a map of species -> partner -> value")
    idx = {t: i for i, t in enumerate(types)}
    seen: dict[frozenset[str], float] = {}
    for a_name, row in block.items():
        if a_name not in idx:
            raise ValueError(
                f"{path}: k_ij names unknown species {a_name!r}; "
                f"known: {', '.join(types)}"
            )
        if not isinstance(row, dict):
            raise ValueError(
                f"{path}: k_ij.{a_name} must be a map of partner -> value, "
                f"got {type(row).__name__}"
            )
        for b_name, value in row.items():
            if b_name not in idx:
                raise ValueError(
                    f"{path}: k_ij.{a_name} names unknown species {b_name!r}; "
                    f"known: {', '.join(types)}"
                )
            if a_name == b_name:
                raise ValueError(
                    f"{path}: k_ij has a diagonal entry {a_name}-{a_name}; the "
                    "diagonal is zero by definition and eps_ii is copied, not mixed"
                )
            v = float(value)
            if not math.isfinite(v):
                raise ValueError(f"{path}: k_ij {a_name}-{b_name} is not finite")
            if v >= 1.0:
                raise ValueError(
                    f"{path}: k_ij {a_name}-{b_name} = {v} would make eps_ij "
                    "zero or negative; k must be < 1"
                )
            pair = frozenset((a_name, b_name))
            if pair in seen and seen[pair] != v:
                raise ValueError(
                    f"{path}: pair {a_name}-{b_name} is stated twice in both "
                    f"orders with different values ({seen[pair]} and {v})"
                )
            seen[pair] = v
            k[idx[a_name], idx[b_name]] = v
            k[idx[b_name], idx[a_name]] = v
    return k


def _read_aux(
    path: Path,
    doc: dict[str, Any],
    sigma_ij: np.ndarray,
    rc: float,
    types: tuple[str, ...],
) -> tuple[float | None, bool]:
    """Parse the `aux` block into (multiplier, control) per M12.

    `aux: none`, `aux: false` and an absent block all mean off.
    `aux: {control: true}` is the exact no-op arm the tuning sweep uses as its
    own reference. Anything else needs a multiplier, and the resulting cutoff
    must be strictly shorter than rc for *every* pair.
    """
    block = doc.get("aux")
    if block is None or block is False or block == "none":
        return None, False
    if not isinstance(block, dict):
        raise ValueError(
            f"{path}: aux must be `none` or a map with `multiplier` or `control`, "
            f"got {block!r}"
        )
    unknown = sorted(set(block) - _AUX_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: aux has unknown key(s) {unknown}; "
            f"allowed: {', '.join(sorted(_AUX_KEYS))}"
        )
    control = bool(block.get("control", False))
    multiplier = block.get("multiplier")
    if control:
        if multiplier is not None:
            raise ValueError(
                f"{path}: aux.control is the exact no-op arm and takes no "
                f"multiplier, but one is stated ({multiplier})"
            )
        return None, True
    if multiplier is None:
        raise ValueError(
            f"{path}: aux needs a `multiplier`, or `aux: none` to turn it off"
        )
    a = float(multiplier)
    if not (a > 0.0 and math.isfinite(a)):
        raise ValueError(f"{path}: aux.multiplier must be positive, got {a}")

    # The binding pair is the largest sigma_ij, over every pair the sweep can
    # read -- framework included, since a guest trial sees framework neighbours.
    i, j = np.unravel_index(int(np.argmax(sigma_ij)), sigma_ij.shape)
    widest = float(sigma_ij[i, j])
    if a * widest >= rc:
        raise ValueError(
            f"{path}: aux.multiplier {a} gives aux_cutoff = {a * widest:.6g} for "
            f"pair {types[i]}-{types[j]} (sigma_ij = {widest:.6g}), which is not "
            f"strictly shorter than rc = {rc:.6g}. An auxiliary at or beyond the "
            "physics cutoff costs work and buys nothing; use `aux: none` or "
            "`aux: {control: true}` if that is what you meant."
        )
    if a < AUX_FIRST_SHELL:
        warnings.warn(
            f"{path}: aux.multiplier {a} is below {AUX_FIRST_SHELL} sigma_ij. The "
            "LJ first shell runs to about 1.5 sigma, so a proposal radius inside "
            "it cannot see the structure it proposes into: sd(Delta_corr) blows "
            "up and acceptance collapses while the run still finishes at a "
            "plausible density. Watch aux_corr_sd in the log.",
            stacklevel=3,
        )
    return a, False


def load_forcefield(path: str | Path, *, allow_provisional: bool = False) -> ForceField:
    """Parse a force-field or species yaml. Guests take the low type ids.

    A block still marked `provisional` is a transcription nobody has checked
    against the primary source; loading one without `allow_provisional` raises,
    so a provisional number cannot reach a V-rung by accident.

    Every check in section 4.1 of multispecies_plan.md is here rather than
    downstream, because a malformed model file is a property of the file and M15
    puts the check at the earliest layer that can see the problem. Each one names
    the offending species, pair or key.
    """
    path = Path(path)
    text = path.read_bytes()
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a yaml mapping")

    schema = int(doc.get("schema", 1))
    if schema not in (1, 2):
        raise ValueError(f"{path}: unsupported schema {schema}; known: 1, 2")

    mixing = doc.get("mixing")
    if mixing not in ("lorentz-berthelot", "lorentz-berthelot-kij"):
        raise ValueError(
            f"{path}: mixing must be lorentz-berthelot or lorentz-berthelot-kij, "
            f"got {mixing!r}"
        )
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
    rc = float(doc["interaction"]["rc"])
    if not (rc > 0.0 and math.isfinite(rc)):
        raise ValueError(f"{path}: interaction.rc must be positive, got {rc}")

    units, reference = _read_units(path, doc, schema)

    status = _blocks_status(doc)
    stale = [b for b, s in status.items() if s != "confirmed"]
    if stale and not allow_provisional:
        raise ValueError(
            f"{path}: block(s) {', '.join(stale)} are still marked provisional. "
            "Confirm the numbers against the primary source and flip the status, "
            "or pass allow_provisional=True (--provisional on the CLI) to proceed "
            "with the status stamped into the output."
        )

    # --- the guest block, under either name -------------------------------
    has_species, has_guests = "species" in doc, "guests" in doc
    if has_species and has_guests:
        raise ValueError(
            f"{path}: state `species:` (a species file) or `guests:` (a MOF force "
            "field), not both"
        )
    if not (has_species or has_guests):
        raise ValueError(f"{path}: no `species:` or `guests:` block")
    guest_key = "species" if has_species else "guests"
    guest_block = doc[guest_key]
    frame_block = doc.get("framework") or {}

    guest_names = [k for k in guest_block if k != "status"]
    frame_names = [k for k in frame_block if k != "status"]
    if not guest_names:
        raise ValueError(f"{path}: the `{guest_key}` block is empty")
    clash = set(guest_names) & set(frame_names)
    if clash:
        raise ValueError(f"{path}: type name(s) {sorted(clash)} appear in both blocks")

    types = tuple(guest_names + frame_names)
    n, ng = len(types), len(guest_names)
    entries = [
        _entry_keys(path, guest_key, name, guest_block[name], _SPECIES_KEYS)
        for name in guest_names
    ] + [
        _entry_keys(path, "framework", name, frame_block[name], _FRAMEWORK_KEYS)
        for name in frame_names
    ]

    sigma = np.array([float(e["sigma"]) for e in entries], float)
    epsilon = np.array([float(e["epsilon"]) for e in entries], float)
    if np.any(sigma <= 0) or np.any(epsilon <= 0):
        bad = [t for t, s, e in zip(types, sigma, epsilon) if s <= 0 or e <= 0]
        raise ValueError(
            f"{path}: sigma and epsilon must be positive; offending: {bad}"
        )
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(epsilon))):
        raise ValueError(f"{path}: sigma and epsilon must be finite")

    # --- lambda ------------------------------------------------------
    lam = np.ones(n, float)
    for i, (name, e) in enumerate(zip(types, entries)):
        if "lambda" not in e:
            continue
        v = float(e["lambda"])
        if not (v > 0.0 and math.isfinite(v)):
            raise ValueError(f"{path}: lambda for {name!r} must be positive, got {v}")
        lam[i] = v

    # --- p_trial ------------------------------------------------------
    declared = [name for name, e in zip(guest_names, entries[:ng]) if "p_trial" in e]
    p_trial = np.zeros(n, float)
    if not declared:
        p_trial[:ng] = 1.0 / ng
    elif len(declared) != ng:
        missing = [name for name in guest_names if name not in declared]
        raise ValueError(
            f"{path}: p_trial is declared for {declared} but missing for "
            f"{missing}; state it for every species or for none (then uniform)"
        )
    else:
        for i, (name, e) in enumerate(zip(guest_names, entries[:ng])):
            v = float(e["p_trial"])
            if not (v > 0.0 and math.isfinite(v)):
                raise ValueError(
                    f"{path}: p_trial for {name!r} must be positive, got {v}"
                )
            p_trial[i] = v
        total = float(p_trial.sum())
        if abs(total - 1.0) > _P_TRIAL_TOL:
            raise ValueError(
                f"{path}: p_trial must sum to 1 within {_P_TRIAL_TOL:g}, got {total!r}"
            )

    # --- symbols -----------------------------------------------------------
    symbol = tuple(str(e.get("symbol", name)) for name, e in zip(types, entries))
    for name, sym in zip(types, symbol):
        if sym not in _ELEMENTS:
            raise ValueError(
                f"{path}: symbol {sym!r} for {name!r} is not a chemical element; "
                "declare `symbol:` with a real element used for trajectory I/O"
            )
    dupes = sorted({s for s in symbol if symbol.count(s) > 1})
    if dupes:
        raise ValueError(
            f"{path}: symbol(s) {dupes} are used by more than one type; a "
            "trajectory frame would be unreadable back into species"
        )

    k_ij = _read_k_ij(path, doc, str(mixing), types)
    sigma_ij = 0.5 * (sigma[:, None] + sigma[None, :])
    aux_multiplier, aux_control = _read_aux(path, doc, sigma_ij, rc, types)

    c = doc.get("core_floor", {}).get("c")
    if c is None:
        if schema >= 2:
            raise ValueError(
                f"{path}: a schema-2 species file must state core_floor.c; the "
                "pair floor is c * sigma_ij and there is no default"
            )
    else:
        c = float(c)
        if not (0.0 < c <= 1.0):
            raise ValueError(f"{path}: core_floor.c must be in (0, 1], got {c}")
        # THE AUXILIARY MUST REACH PAST THE FLOOR IT HAS TO RESPECT, and both
        # radii are multiples of the same sigma_ij, so the whole per-pair
        # condition c*sigma_ij < a*sigma_ij is the scalar c < a. Checked here
        # rather than in `_read_aux` only because `c` is parsed after it.
        #
        # Why it matters, in the words the untyped guard uses: admissibility is
        # read off the auxiliary-radius neighbour query as `all(d >= floor)`,
        # which means nothing while that query stops short of the floor. An
        # auxiliary inside the floor would report every trial admissible.
        if aux_multiplier is not None and aux_multiplier <= c:
            raise ValueError(
                f"{path}: aux.multiplier {aux_multiplier} is at or below "
                f"core_floor.c {c}, so the auxiliary radius c*sigma_ij would not "
                f"reach past the overlap floor it has to respect. Both are "
                f"multiples of sigma_ij, so the condition is simply "
                f"multiplier > c."
            )

    return ForceField(
        name=str(doc["name"]),
        path=path,
        schema=schema,
        units=units,
        reference=reference,
        types=types,
        sigma=sigma,
        epsilon=epsilon,
        lam=lam,
        p_trial=p_trial,
        symbol=symbol,
        k_ij=k_ij,
        n_guests=ng,
        rc=rc,
        core_floor_c=c,
        aux_multiplier=aux_multiplier,
        aux_control=aux_control,
        status=status,
        sha256=hashlib.sha256(text).hexdigest(),
        raw=doc,
    )


@dataclass(frozen=True)
class ReducedTables:
    """The force field as the engine sees it.

    For an Angstrom-Kelvin file that means everything divided through by the
    reference self-pair. For a `units: reduced` file it is a passthrough: the
    numbers in the yaml already are these numbers, `sigma_ref = eps_ref = 1`, and
    `T_kelvin` carries T* itself.
    """

    types: tuple[str, ...]
    n_guests: int
    sigma_star: np.ndarray  # (n_types, n_types)
    eps_star: np.ndarray  # (n_types, n_types)
    T_star: float
    rc_star: float
    sigma_ref: float  # angstrom (1.0 when reduced)
    eps_ref: float  # kelvin (1.0 when reduced)
    T_kelvin: float
    core_floor_c: float | None
    units: str
    reference: str | None
    lam: np.ndarray  # per type
    p_trial: np.ndarray  # per type
    symbol: tuple[str, ...]
    k_ij: np.ndarray  # (n_types, n_types)
    aux_cutoff_star: np.ndarray | None  # (n_types, n_types) or None when off
    aux_control: bool

    def z_star(self, fugacity_pa: float) -> float:
        """Reduced activity from a fugacity in Pa: z* = f sigma_ref^3 / (kB T)."""
        if self.units == "reduced":
            raise ValueError(
                "z_star converts a pressure through the reference scale, which a "
                "reduced file does not have. Give the activity directly "
                "(ensemble.z) or a reduced fugacity (ensemble.fugacity)."
            )
        return (
            fugacity_pa
            * (self.sigma_ref * ANGSTROM_M) ** 3
            / (KB_J_PER_K * self.T_kelvin)
        )

    def as_dict(self, ff: ForceField) -> dict[str, Any]:
        """The reduced force field as plain JSON-able data, for the run record.

        Carries the source path and its sha256 alongside the numbers, so a run
        directory says which file it was built from and whether that file has
        since changed.
        """
        return {
            "forcefield": {"name": ff.name, "path": str(ff.path), "sha256": ff.sha256},
            "status": ff.status,
            "types": list(self.types),
            "n_guests": self.n_guests,
            "units": self.units,
            "reference": self.reference,
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
            "lambda": self.lam.tolist(),
            "p_trial": self.p_trial.tolist(),
            "symbol": list(self.symbol),
            "k_ij": self.k_ij.tolist(),
            "aux_cutoff_star": (
                None if self.aux_cutoff_star is None else self.aux_cutoff_star.tolist()
            ),
            "aux_control": self.aux_control,
        }


def reduce_tables(
    ff: ForceField, T: float, *, reference: str | None = None
) -> ReducedTables:
    """Divide the force field through by the reference self-pair scale.

    The reference type's self-pair lands on sigma* = eps* = 1 exactly, which is
    what makes the single-species reduction test against the frozen engine
    possible at all.

    A `units: reduced` file is already there and is passed through untouched:
    sigma_ref = eps_ref = 1, no type is pinned to sigma* = 1, and `T` is
    read as T* rather than as kelvin. Naming a reference for such a file is an
    error, because there is nothing for it to mean.

    `reference` defaults to the one the file declares, and to "Xe" for the
    schema-1 MOF force fields, which is the behaviour every existing caller
    already relies on.
    """
    if T <= 0:
        raise ValueError("temperature must be positive")
    sigma_ij = ff.pair_sigma()
    eps_ij = ff.pair_epsilon()

    if ff.units == "reduced":
        if reference is not None:
            raise ValueError(
                f"{ff.path}: reduced files take no reference (asked for "
                f"{reference!r}); sigma_ref = eps_ref = 1 by definition"
            )
        sigma_ref = eps_ref = 1.0
        sigma_star, eps_star = sigma_ij, eps_ij
        T_star, rc_star = float(T), ff.rc
    else:
        ref = reference if reference is not None else (ff.reference or "Xe")
        r = ff.index(ref)
        sigma_ref = float(sigma_ij[r, r])
        eps_ref = float(eps_ij[r, r])
        sigma_star, eps_star = sigma_ij / sigma_ref, eps_ij / eps_ref
        T_star, rc_star = T / eps_ref, ff.rc / sigma_ref
        reference = ref

    aux = ff.aux_cutoff()
    return ReducedTables(
        types=ff.types,
        n_guests=ff.n_guests,
        sigma_star=sigma_star,
        eps_star=eps_star,
        T_star=T_star,
        rc_star=rc_star,
        sigma_ref=sigma_ref,
        eps_ref=eps_ref,
        T_kelvin=T,
        core_floor_c=ff.core_floor_c,
        units=ff.units,
        reference=reference,
        lam=ff.lam,
        p_trial=ff.p_trial,
        symbol=ff.symbol,
        k_ij=ff.k_ij,
        aux_cutoff_star=None if aux is None else aux / sigma_ref,
        aux_control=ff.aux_control,
    )


def lj_energy_kelvin(r_angstrom: np.ndarray, sigma: float, eps: float) -> np.ndarray:
    """4 eps ((sigma/r)^12 - (sigma/r)^6), eps and the result in kelvin.

    Same 12-6 shape as gcmc.forcefield.lennard_jones._lj_from_ir6, written out
    here so this module never imports the engine's own LJ; the reduction test
    compares the two and must not share an implementation.
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
    p.add_argument(
        "--T", type=float, required=True, help="temperature in kelvin (T* if reduced)"
    )
    p.add_argument(
        "--reference",
        default=None,
        help="type whose self-pair sets the scale; defaults to the file's own, "
        "then to Xe. Forbidden for a reduced file.",
    )
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
    print(f"{ff.name} ({ff.path}) sha256 {ff.sha256[:12]}  schema {ff.schema}")
    print(f"  types      : {', '.join(ff.types)}  (guests: {', '.join(ff.guests)})")
    print(f"  units      : {ff.units}")
    if ff.units == "reduced":
        print("  reference  : none (sigma_ref = eps_ref = 1 by definition)")
        print(f"  T*         : {red.T_star:.6f}")
        print(f"  rc*        : {red.rc_star:.6f}")
    else:
        print(
            f"  reference  : {red.reference}-{red.reference}"
            f"  sigma_ref = {red.sigma_ref:.5f} A  eps_ref/kB = {red.eps_ref:.4f} K"
        )
        print(f"  T          : {a.T:g} K   ->  T* = {red.T_star:.6f}")
        print(f"  rc         : {ff.rc:g} A  ->  rc* = {red.rc_star:.6f}")
    print(
        "  lambda     : "
        + ", ".join(f"{t}={v:g}" for t, v in zip(ff.guests, ff.lam[: ff.n_guests]))
    )
    print(
        "  p_trial    : "
        + ", ".join(f"{t}={v:g}" for t, v in zip(ff.guests, ff.p_trial[: ff.n_guests]))
    )
    if ff.aux_control:
        print("  auxiliary  : control arm (exact no-op)")
    elif ff.aux_multiplier is None:
        print("  auxiliary  : off")
    else:
        aux = ff.aux_cutoff()
        assert aux is not None
        print(
            f"  auxiliary  : a = {ff.aux_multiplier:g}  ->  aux_cutoff_ij in "
            f"[{aux.min():.4f}, {aux.max():.4f}]  (rc = {ff.rc:g})"
        )
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
    if ff.k_ij.any():
        print("  k_ij (non-zero):")
        for i in range(len(ff.types)):
            for j in range(i + 1, len(ff.types)):
                if ff.k_ij[i, j]:
                    print(
                        f"    {ff.types[i]}-{ff.types[j]:<8} {ff.k_ij[i, j]:+.6g}"
                        f"   eps_ij = {red.eps_star[i, j]:.6g}"
                    )

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

    Note this `kij` is the Peng-Robinson binary interaction parameter of the gas
    equation of state, read from the `eos` block. It is a different quantity from
    the force-field cross term `k_ij` of M1, which mixes epsilons; the two are
    stated in different blocks and are never mixed up by the code.

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

"""Configuration: one place that turns a params.json into a frozen Config and
derives every mode-dependent constant the rest of the code needs.

params.json has five blocks, one per axis. Inactive keys cannot appear (an LJ
temperature on a hard-core run, a pmf block under branch), so nothing leaks.
Validation is a per-section check plus the compatibility rules below, each with
a specific message.

Schema
------
potential, sampler and ensemble are required; auxiliary and run are optional
blocks (omitting `run` entirely takes every default). The machine-readable twin
of this block is examples/schema.json, which validation uses.

{
  "potential":  {"main": "hard_core" | "lennard_jones",
                 "T": float,                        # required iff main = lennard_jones, else forbidden
                 "tail": bool},                     # default True; LJ only; auto-off under a pore/QA matrix

  "_comment":   str,                                 # OPTIONAL anywhere; any key starting with
                                                     #   "_" is ignored, so a params file can
                                                     #   carry its own explanation
  "auxiliary":  {"type": "pmf" | "short_range",      # OPTIONAL block; omit = no aux
                 "g3_csv": str,                      # pmf only; required iff 3D
                 "cutoff": float,                    # short_range only, required for it:
                                                     #   the proposal/selection radius, e.g. 1.5.
                                                     #   lower_cutoff < cutoff <= upper_cutoff.
                                                     #   channel + lennard_jones + 3D bulk only.
                 "eta": float,                       # default 0.69 (2D)
                 "gamma_max": float, "delta_eta": float},

  "sampler":    {"type": "metropolis" | "rosenbluth" | "branch" | "channel" | "reweight",
                 "depth": int,                             # default 0; forbidden for metropolis;
                                                           #   channel needs depth >= 1; XOR full_repacking
                 "full_repacking": bool,                   # default False; repacks the whole cavity instead
                                                           #   of a fixed depth. Not metropolis. XOR depth.
                 "trials": int | "trial_density": float,   # trials default 30; forbidden for metropolis; XOR
                 "trial_box_side": float,                  # default 3.0; rosenbluth/branch only (cube trial
                                                           #   box); rejected for channel and metropolis
                 "channel_length": float | "cyclic",        # channel only: rod length along the sweep
                                                           #   axis, or "cyclic" for a wall-to-wall rod
                                                           #   whose ends meet through the boundary.
                                                           #   Legacy spellings "full" | "box" |
                                                           #   "full_box" | "span" are equivalent.
                 "channel_width": float},                  # channel only: square cross-section side

  "ensemble":   {"dimension": 2 | 3,                 # required
                 "type": "muVT" | "NVT" | "QA" | "confined",   # default muVT
                 "mu": float,                        # muVT: exactly one of mu or ONE target_* key
                 "target_eta" | "target_density" | "target_N",  # controller drives mu to it; not with sweep
                 "N": int | "packing_fraction": float,         # NVT only, exactly one; mu/sweep forbidden
                 "pore": {"geometry": "slit" | "cylinder",     # confined only
                          "H": float,                          #   slit gap; must exceed 2*run.radius
                          "R": float,                          #   cylinder radius; 2R must exceed 2*run.radius
                          "L_z": float},                       #   cylinder axis length, default ensemble.box
                 "box": float,                       # default by (dimension, run.kind): 2D 75 / 2D par 200
                                                     #   / 3D 20 / 3D par 50
                 "sweep": [float]},                  # LEGACY, unusable: the launcher does not expand it (it
                                                     #   is not a sweep axis), so it reaches here unexpanded
                                                     #   and raises. Sweep with a LIST on an axis instead.

  "run":        {"kind": "single" | "parallel",
                 "n_cell": int | [int, ...],         # PARALLEL, recommended: cells the box is cut into. One
                                                     #   cell is one core, so this IS the run's core count.
                                                     #   int is factored into the most cubic per-axis split;
                                                     #   a list is honoured verbatim. cell_length/buffer/
                                                     #   x_spacing/y_spacing are then DERIVED and must not
                                                     #   be given alongside it (either/or, nothing between).
                 "n_workers": int,                   # DEPRECATED (accepted + warned): cores are implied by
                                                     #   the cell count and cannot be set independently
                 "n_cores": int,                     # launcher only: TOTAL sweep core budget (= --cores N);
                                                     #   rejected on a parallel base (cores = cells there)
                 "sampling_window": int,             # default 1000
                 "target_cycles": float,             # default 1.0025e9 -- set it for real runs
                 "burn_in": int,                     # default 0
                 "traj_every": int,                  # default 0 (off)
                 "checkpoint_every": int,            # default 1
                 "log_flush_interval": float,        # default 0.0 = flush every row (unchanged). Seconds:
                                                     #   a TIME FLOOR between log flushes, so the log only
                                                     #   batches when rows arrive faster than this. Bounds
                                                     #   the loss window in time, not in rows. Only worth
                                                     #   setting for tiny sampling_window; measured, the log
                                                     #   is 0.02% of runtime at the default cadence
                 "seed": int, "initial_config": str,
                                                     #   a path (file or equilibrated library), or the
                                                     #   literal token "grid" / "grid:<eta>" to bootstrap
                                                     #   from a lattice at a reasonable fluid density
                                                     #   (0.35 in 3D, 0.45 in 2D) instead of an empty box
                 "matrix_config": str,               # QA only (required by it); file or folder-of-realizations
                 "radius": float,                    # default 0.5; diameter 2*radius is the hard-core floor
                 "move_factor": float,               # default 0.25
                 "upper_cutoff": float,              # default 3.7545 (2D hard/aux), 2.0 (3D hard/aux),
                                                     #   2.5 (LJ), 3*radius (reweight)
                 "lower_cutoff": float,              # default 0.9 (any LJ mode) | 2*radius (hard core)
                 "no_neigh": int,                    # default 40 (2D) / 75 (3D)
                 "cell_length": float, "buffer": float,   # MANUAL parallel geometry; both REQUIRED when
                                                     #   n_cell is absent (no defaults: the historical 30
                                                     #   could not fit any box <= 32). buffer must be >= the
                                                     #   interaction reach and cell_length+buffer must fit box
                 "control_gain": float, "control_gain_tau": float, "control_window": int,
                                                     # muVT target controller; defaults 0.5 / 20.0 / 20
                 "x_spacing": float, "y_spacing": float,  # DEPRECATED and INERT (accepted + warned): they
                                                     #   feed only the verbatim 2D grid_up_the_system, which
                                                     #   the engine no longer calls
                 "sweep_label": str}                 # written BY the launcher into each expanded run
}

Sweeps
------
Give a LIST to any sweep axis and launch with `python -m gcmc.launcher` (not
`gcmc run`): it expands the Cartesian product, one run per combination. The axes
are ensemble.{mu, packing_fraction, N, target_N, target_eta, target_density},
sampler.{depth, trial_density, trial_box_side, channel_length, channel_width},
run.seed, and ensemble.pore.{H, R}. A list that reaches from_params unexpanded
is an error naming the key.

For rosenbluth/branch, trial_box_side is the full side of the square trial box
(twice the legacy cavity radius) and the trial-count relation uses trial_box_side^2
as the span. The channel sampler does not use trial_box_side: its trial region is
an oriented cuboid of volume channel_length * channel_width^2, and trial_density is
spheres per unit volume, so the count is trial_density times that volume.

Sampled model
-------------
Hard-core runs sample hard spheres of diameter 2*radius (the exclusion floor is
that diameter). Every Lennard-Jones run, in every sampler (metropolis, rosenbluth,
branch, channel), samples truncated-and-shifted-free LJ with a hard core at
run.lower_cutoff, which defaults to 0.9 sigma. It is not pure LJ: pairs closer
than lower_cutoff are forbidden outright rather than merely improbable. One floor
for all samplers means a metropolis isotherm and a channel isotherm are comparing
the same distribution. The bulk tail correction (potential.tail, on by default in
bulk) is disabled automatically in a pore or a quenched-annealed matrix, where the
uniform space-filling assumption behind it does not hold.

Support matrix
--------------
Not every axis combines with every other. from_params enforces these rules and
raises a specific ValueError on any disallowed mix; they are collected here so the
boundaries are readable in one place.

Samplers x potential:
  metropolis, rosenbluth, branch   hard_core or lennard_jones
  channel                          hard_core or lennard_jones, depth >= 1
  reweight (psi6)                  hard_core only

An optional pmf auxiliary layers onto hard_core (-> aux) or lennard_jones
(-> ljaux, 3D only, needs auxiliary.g3_csv); reweight takes no auxiliary.

The short_range auxiliary is a different thing wearing the same block: the SAME
Lennard-Jones at a shorter radius, used to propose and select, with the mismatch
divided out in the acceptance (dual-cutoff CBMC). It does not change the
potential_type -- the run is still ljbranch, with aux_cutoff as a parameter of it
-- so it cannot disturb an existing run fingerprint. Supported for channel +
lennard_jones + 3D + bulk + serial only; every other combination is refused with
a message naming the reason.

Samplers x ensemble:
  muVT        every sampler
  NVT         every sampler except reweight paths that assume muVT targeting
  confined    grand-canonical samplers (give mu); 3D only; no matrix
  QA          hard_core or LJ+branch only; not metropolis; needs run.matrix_config

Dimension:
  reweight    2D only
  channel     2D and 3D (2D body is an oriented rectangle, not a cuboid)
  confined    3D only
  LJ + pmf    3D only

Parallelism (run.kind='parallel'):
  A parallel run is submitted in ONE of two ways, never a mixture:
    n_cell     -> the cell counts are the submission; per-axis cell lengths and
                  the buffer are derived, and one cell is one core.
    full geometry -> cell_length AND buffer are stated and the cell count falls
                  out as floor(box / (cell_length + buffer)) per axis.
  The counts are what the decomposition acts on: recovering them from a cell
  length by floor() is not exact in float64 and silently loses cells.
  supports metropolis, rosenbluth, and branch in both 2D and 3D. The 3D case
  is validated physics (cell_unify_plan.md U3), not just an accepted config.
  Rejected: reweight and QA (decomposition drops the psi6 field / frozen
  matrix), channel (serial only), and confined (decomposition drops the pore
  walls).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .potentials.base import Potential

import logging
import math
from dataclasses import dataclass

from . import cell_plan

log = logging.getLogger(__name__)


def default_side(two_dim: bool, parallel: bool) -> float:
    """Historical default box side per (dimensionality, parallel-mode) pair;
    the values the original scripts hard-coded, kept so configs without an
    explicit box reproduce the original systems."""
    return {
        (True, True): 200.0,
        (True, False): 75.0,
        (False, True): 50.0,
        (False, False): 20.0,
    }[(two_dim, parallel)]


_GAMMA_AUX_2D = {
    1: 0.193548,
    2: 0.049948,
    3: 0.022557,
    4: 0.010187,
    5: 0.007558,
    6: 0.002438,
    7: 0.001180,
    8: 0.000495,
    9: 0.000176,
}
_GAMMA_AUX_3D = {
    1: 0.2903,
    2: 0.1030,
    3: 0.0332,
    4: 0.0193,
    5: 0.0037,
    6: 0.0042,
    7: 0.0,
    8: 0.0,
    9: 0.0,
}


def _gamma_dict(pmf: bool, reweight: bool, two_dim: bool) -> dict[int, float]:
    """Per-nesting-level bias exponents gamma_k for k = 1..9: the tuned PMF
    schedules (2D and 3D differ; fitted in the original work) when the pmf
    auxiliary is on, unit exponents for psi6 reweighting, zero (bias off)
    otherwise. The engine applies gamma_k at depth k of the Rosenbluth
    selection."""
    if pmf:
        return dict(_GAMMA_AUX_2D if two_dim else _GAMMA_AUX_3D)
    if reweight:
        return dict.fromkeys(range(1, 10), 1.0)
    return dict.fromkeys(range(1, 10), 0.0)


_MAIN = {"hard_core", "lennard_jones"}
_SAMPLERS = {"metropolis", "rosenbluth", "branch", "reweight", "channel"}


def _derive_potential_type(potential: str, bias: str | None, sampler: str) -> str:
    """The engine's dispatch key, derived from three orthogonal axes: the base
    potential ("hard_sphere" | "lennard_jones"), any bias ("pmf" | "psi6" | None),
    and the sampler. This is the one place the combination is collapsed into the
    flat tag the rest of the code (and the run identity) uses, so the axes can be
    reasoned about independently while old fingerprints stay byte-for-byte stable.
    """
    if sampler == "reweight":
        return "reweight"
    if potential == "lennard_jones":
        if bias == "pmf":
            return "ljaux"
        if sampler in ("branch", "channel"):
            return "ljbranch"
        return "lj"
    if bias == "pmf":
        return "aux"
    return "hard_sphere"


def _as_bool(value: object, name: str) -> bool:
    """Coerce a JSON value to a real bool. JSON true/false pass through, but a
    quoted "True"/"False" (easy to write by hand) must not both read as truthy,
    so strings are parsed explicitly and anything ambiguous raises."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    raise ValueError(f"{name} must be a boolean (true/false), got {value!r}")


def _check(
    section: dict[str, Any],
    allowed: set[str],
    required: tuple[str, ...] | set[str] = (),
) -> None:
    """Validate one params section against its schema: unknown keys are an
    error (typos surface immediately, not as silently ignored physics) and
    required keys must be present.

    Keys beginning with an underscore are IGNORED, so a params file can carry its
    own explanation. JSON has no comments, and the alternative -- keeping the
    rationale for a run in a separate file from the run -- is how a job ends up
    being repeated a year later by someone who cannot tell what it was for. The
    prefix cannot collide with a real key because every schema key is a bare
    identifier. `examples/*.json` all use `_comment`.
    """
    extra = {k for k in set(section) - set(allowed) if not k.startswith("_")}
    if extra:
        raise ValueError(f"unknown keys {sorted(extra)}; allowed {sorted(allowed)}")
    missing = set(required) - set(section)
    if missing:
        raise ValueError(f"missing required keys {sorted(missing)}")


@dataclass(frozen=True)
class Config:
    """The complete, immutable description of one run.

    Physics identity first (potential family, bias, sampler, dimension, mu,
    ensemble, geometry, move parameters), then operational knobs (windows,
    cadences, paths). Frozen on purpose: nothing may mutate the physics
    mid-run, and the run fingerprint (hash_fingerprint.identity_dict) hashes
    a subset of these fields, so equality of identity implies resumability.
    Field-by-field meaning is documented inline below.
    """

    # identity-bearing
    potential: str  # "hard_sphere" | "lennard_jones"
    bias: str | None  # None | "pmf" | "short_range" | "psi6"
    sampler: str
    dimension: int
    chem_pot: float
    ensemble_type: str
    target_n: int | None
    box: float
    cavity_radius: float
    trial_box_side: float
    trial_density: float
    depth: int
    branch: bool
    metropolis: bool
    full_repacking: bool
    move_factor: float
    temp: float
    rc: float | None
    eta: float | None
    gamma_max: float | None
    delta_eta: float | None
    g3_csv: str | None
    # derived
    two_dim: bool
    parallel: bool
    beta: float
    fugacity: float
    radius: float
    diameter: float
    r_squared: float
    no_of_trial_molecules: int
    no_neigh: int
    upper_cutoff: float
    lower_cutoff: float
    volume_of_molecule: float
    total_volume: float
    lj_tail_pref: float | None
    gamma_dict: dict
    # execution (not identity-bearing)
    mode: str
    # Cores are implied, never stated: one cell is one core, so `n_workers` is
    # DERIVED (= the cell count) and is not part of the submission surface.
    n_workers: int
    # The submission as written (int total or per-axis list), kept for the log
    # and provenance only; `cell_counts` is what the decomposition acts on.
    n_cell: int | list[int] | None
    # Authoritative per-axis cell counts, None for a serial run. Handed to
    # decomposition.grid_up_nd directly: recovering them from a cell length by
    # floor() is not exact in float64 and silently loses cells (D7).
    cell_counts: tuple[int, ...] | None
    cell_length: float
    # Per-axis parallel cell length. None means cubic cells (every axis is
    # `cell_length`, bit-identical to the scalar path). When set it is a
    # length-`dimension` tuple, letting the box tile n_x x n_y x n_z with
    # independent counts (e.g. 4 x 4 x 5 = 80 cells in a cubic box).
    cell_length_sides: tuple[float, ...] | None
    buffer: float
    # RNG seed: part of the physics identity, so each seed is a distinct run
    # with its own directory (two seeds are two independent chains, never the
    # same experiment). None means an unseeded (nondeterministic) chain.
    seed: int | None
    x_spacing: float | None
    y_spacing: float | None
    sampling_window: int
    target_cycles: float
    burn_in: int
    traj_every: int
    checkpoint_every: int
    log_flush_interval: float
    initial_config: str | None
    # quenched-annealed matrix: frozen obstacle positions the fluid cannot
    # overlap but which are never inserted/deleted/moved/counted. Default empty
    # => ordinary bulk run, bit-identical to before. matrix_config is the file
    # path (loaded by the runner); matrix_positions holds the loaded coordinates.
    matrix_config: str | None = None
    matrix_positions: object = None  # np.ndarray (n_matrix, dim) or None
    # muVT target-mu control: drive mu so the running-mean N hits a target.
    # target_count is the target <N> (from target_N / target_eta / target_density);
    # None => ordinary fixed-mu muVT. The controller is Robbins-Monro on log-N.
    target_count: float | None = None
    control_gain: float = 0.5
    control_gain_tau: float = 20.0
    control_window: int = 20
    sweep_label: str | None = None  # tag from the launcher: the swept axis
    # confinement (ensemble_type == "confined"): hard slit or cylinder pore
    pore_geometry: str | None = None  # "slit" | "cylinder" | None
    pore_H: float | None = None  # slit gap (z), wall surfaces at 0 and H
    pore_R: float | None = None  # cylinder radius (axis along z, at centre)
    pore_axis_len: float | None = None  # cylinder axial length (periodic z)
    # channel sweep (sampler == "channel"): thin oriented cuboid trial region
    # short-range (dual-cutoff) auxiliary radius, or None when the auxiliary is
    # off. The PHYSICS cutoff is always upper_cutoff; this is only the radius the
    # proposal and the clique sum are evaluated at, with the mismatch divided out
    # in the acceptance. aux_cutoff == upper_cutoff is a legal no-op.
    aux_cutoff: float | None = None
    channel: bool = False
    channel_length: float = 0.0  # cuboid length along the sweep axis
    channel_width: float = 0.0  # cuboid cross-section (square side)
    channel_cyclic: bool = False  # channel_length == "cyclic": wall-to-wall rod
    # Per-axis orthorhombic box (A2.1). None means cubic: every axis is `box`,
    # and the scalar path runs exactly as before, so cubic fingerprints, RNG
    # streams and characterisation trajectories are unchanged. When set it is a
    # length-`dimension` tuple (Lx, Ly, Lz); `box` then holds min(sides) for the
    # legacy scalar guards, and volume/wrap/draws route through `box_dims`.
    box_sides: tuple[float, ...] | None = None
    # oriented rod, any orientation, exact PBC
    # (the periodic guard is lifted; the sweep
    # uses minimum-image distances + retention)

    @property
    def box_dims(self) -> list[float]:
        """Per-axis box lengths (Lx, Ly, Lz), the A2.1 accessor. Cubic
        (box_sides is None): the scalar side replicated `dimension` times,
        bit-identical to the old behaviour. Orthorhombic: the stored tuple."""
        if self.box_sides is not None:
            return list(self.box_sides)
        return [self.box] * self.dimension

    @property
    def cell_dims(self) -> list[float]:
        """Per-axis parallel cell lengths. Cubic (cell_length_sides is None):
        the scalar `cell_length` replicated `dimension` times, bit-identical to
        the old behaviour. Anisotropic: the stored tuple."""
        if self.cell_length_sides is not None:
            return list(self.cell_length_sides)
        return [self.cell_length] * self.dimension

    @property
    def potential_type(self) -> str:
        """Flat engine dispatch key derived from potential, bias and sampler.
        Kept as a property so existing fingerprints (which read potential_type)
        stay byte-identical after the underlying fields were split apart."""
        return _derive_potential_type(self.potential, self.bias, self.sampler)

    def build_potential(self) -> Potential | None:
        """Instantiate the interaction object for this run from the registry,
        forwarding exactly the constants each mode needs (e.g. volume for the
        LJ tail, the g3 table and trial count for the 3D aux bias); None for
        pure reweighting, which has no potential of its own."""
        from .potentials import make_potential

        uc, d = self.upper_cutoff, self.diameter
        match self.potential_type:
            case "hard_sphere":
                return make_potential("hard_sphere", diameter=d, upper_cutoff=uc)
            case "lj" | "ljbranch":
                return make_potential(
                    "lj",
                    beta=self.beta,
                    upper_cutoff=uc,
                    diameter=d,
                    volume=self.total_volume,
                )
            case "ljaux":
                return make_potential(
                    "ljaux3d",
                    beta=self.beta,
                    upper_cutoff=uc,
                    diameter=d,
                    side=self.box,
                    n_trials=self.no_of_trial_molecules,
                    g3_csv=self.g3_csv,
                )
            case "aux" if self.two_dim:
                return make_potential(
                    "aux2d", diameter=d, upper_cutoff=uc, eta=self.eta, sigma=1.0
                )
            case "aux":
                return make_potential(
                    "aux3d",
                    diameter=d,
                    upper_cutoff=uc,
                    side=self.box,
                    n_trials=self.no_of_trial_molecules,
                    g3_csv=self.g3_csv,
                )
            case "reweight":
                return None
            case other:
                raise ValueError(f"unknown potential type {other!r}")


def from_params(p: dict[str, Any]) -> Config:
    """Translate the user-facing nested params dict (sections: potential,
    auxiliary, sampler, ensemble, run) into a validated frozen Config,
    deriving every dependent constant (beta, cutoffs, gamma schedule, trial
    counts, tail prefactor inputs) in one place. Every key is checked against
    its section schema, so a config typo fails loudly here rather than
    changing the physics silently."""
    _check(
        p,
        {"potential", "auxiliary", "sampler", "ensemble", "run"},
        {"potential", "sampler", "ensemble"},
    )

    # ---- main potential ----
    pot = p["potential"]
    _check(pot, {"main", "T", "tail"}, {"main"})
    main = pot["main"]
    if main not in _MAIN:
        raise ValueError(f"unknown main potential {main!r}; allowed {sorted(_MAIN)}")
    lj = main == "lennard_jones"
    if lj and "T" not in pot:
        raise ValueError("Lennard-Jones requires potential.T")
    if not lj and "T" in pot:
        raise ValueError("only Lennard-Jones takes potential.T")

    # ---- sampler ----
    smp = p["sampler"]
    _check(
        smp,
        {
            "type",
            "depth",
            "full_repacking",
            "trials",
            "trial_density",
            "trial_box_side",
            "channel_length",
            "channel_width",
        },
        {"type"},
    )
    sampler = smp["type"]
    if sampler not in _SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; allowed {sorted(_SAMPLERS)}")
    metropolis = sampler == "metropolis"
    branch = sampler in ("branch", "channel")
    channel = sampler == "channel"
    reweight = sampler == "reweight"
    # full_repacking (default False) is the XOR opposite of the old use_depth:
    # instead of a fixed depth, an exchange repacks the whole cavity, removing
    # every sphere it finds and inserting one more (add), one fewer (remove), or
    # the same count (repack). depth is then meaningless and must not be given.
    full_repacking = _as_bool(
        smp.get("full_repacking", False), "sampler.full_repacking"
    )
    if full_repacking:
        if metropolis:
            raise ValueError(
                "full_repacking applies to the cavity samplers "
                "(rosenbluth/branch/channel), not metropolis"
            )
        if "depth" in smp:
            raise ValueError(
                "depth and full_repacking are mutually exclusive: "
                "full_repacking repacks the whole cavity and ignores "
                "depth, so give one or the other, not both"
            )

    # ---- ensemble ----
    ens = p["ensemble"]
    _check(
        ens,
        {
            "mu",
            "sweep",
            "dimension",
            "box",
            "type",
            "N",
            "packing_fraction",
            "target_eta",
            "target_density",
            "target_N",
            "pore",
        },
        {"dimension"},
    )
    dim = int(ens["dimension"])
    target_count = None  # set below iff a muVT target is requested
    _target_spec = None  # (kind, value) for a muVT target, finalized post-volume
    _pore_geometry = _pore_H = _pore_R = _pore_axis_len = None
    if dim not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    two_dim = dim == 2
    ens_type = ens.get("type", "muVT")
    if ens_type not in ("muVT", "NVT", "QA", "confined"):
        raise ValueError("ensemble.type must be 'muVT', 'NVT', 'QA', or 'confined'")
    nvt = ens_type == "NVT"
    qa = ens_type == "QA"
    confined = ens_type == "confined"
    if confined:
        # hard slit or cylinder pore. muVT physics (give mu, or a target) plus a
        # one-body hard-wall constraint. The pore is the only external field;
        # rejects a matrix (no fluid-in-matrix-in-pore here). 3D only.
        if "pore" not in ens:
            raise ValueError("ensemble.type='confined' requires an ensemble.pore block")
        if two_dim:
            raise ValueError("confined pores are 3D only")
        if "matrix_config" in p.get("run", {}):
            raise ValueError("a pore and a QA matrix cannot be combined")
        pore = ens["pore"]
        # the accessible pore must admit a whole sphere, so every gap is checked
        # against the actual diameter 2*radius, not a hardcoded 1. radius is parsed
        # again below for the derived constants; here we only need it for the wall
        # geometry checks, which run before that block.
        _pore_diam = 2.0 * float(p.get("run", {}).get("radius", 0.5))
        _pore_geometry = pore.get("geometry")
        if _pore_geometry == "slit":
            _pore_H = float(pore["H"])  # gap in z; walls at 0 and H
            if _pore_H <= _pore_diam:
                raise ValueError(
                    f"slit gap H must exceed the sphere diameter ({_pore_diam})"
                )
        elif _pore_geometry == "cylinder":
            _pore_R = float(pore["R"])  # radius; axis along z at centre
            _pore_axis_len = float(pore.get("L_z", ens["box"]))
            if 2 * _pore_R <= _pore_diam:
                raise ValueError(
                    f"cylinder diameter 2R must exceed the sphere "
                    f"diameter ({_pore_diam})"
                )
        else:
            raise ValueError("ensemble.pore.geometry must be 'slit' or 'cylinder'")
        # the wall lives inside the cubic periodic cell of side `box`; require a
        # forbidden margin >= 1 sigma so periodic images never interact across
        # the wall (then the existing cubic-periodic neighbour tree is exact and
        # no per-axis box change is needed).
        if isinstance(ens["box"], (list, tuple)):
            raise ValueError(
                "confined pores are defined inside a cubic cell; give a scalar "
                "ensemble.box"
            )
        L = float(ens["box"])
        if _pore_geometry == "slit":
            assert _pore_H is not None  # slit requires H, checked above
            if _pore_H > L - _pore_diam:
                raise ValueError(
                    f"slit gap H={_pore_H} must be <= box-{_pore_diam}="
                    f"{L - _pore_diam} so the forbidden margin separates "
                    f"periodic z-images"
                )
        if _pore_geometry == "cylinder":
            assert _pore_R is not None  # cylinder requires R, checked above
            if 2 * _pore_R > L - _pore_diam:
                raise ValueError(
                    f"cylinder diameter 2R={2 * _pore_R} must be <= box-"
                    f"{_pore_diam}={L - _pore_diam} so the forbidden margin "
                    f"separates periodic x,y-images"
                )
        _pore_axis_len = L  # cubic cell: cylinder axial period = box
    if "pore" in ens and not confined:
        raise ValueError("ensemble.pore is only valid with ensemble.type='confined'")

    if nvt:
        # NVT: fixed particle number, no chemical potential. Target given as an
        # exact integer N, or as a packing fraction converted to the nearest
        # integer N (held exactly thereafter). mu is meaningless here.
        if "mu" in ens or "sweep" in ens:
            raise ValueError("NVT takes no 'mu'/'sweep' (N is fixed, mu undefined)")
        has_n, has_pf = "N" in ens, "packing_fraction" in ens
        if has_n == has_pf:
            raise ValueError("NVT needs exactly one of 'N' or 'packing_fraction'")
        _swept = next(
            (k for k in ("N", "packing_fraction") if isinstance(ens.get(k), list)), None
        )
        if _swept is not None:
            raise ValueError(
                f"ensemble.{_swept} is a list (a sweep), but a single run got it "
                f"unexpanded. Launch sweeps with `python -m gcmc.launcher` (not "
                f"`gcmc run`); if you already did, the launcher on this machine "
                f"is an older version that does not expand '{_swept}' -- update "
                f"gcmc/launcher.py."
            )
        mu: float | None = 0.0  # unused under NVT; prefactor cancels in repacking
    else:
        # muVT: either a fixed `mu` (or a sweep), OR a target the controller
        # drives mu toward: exactly one of target_eta / target_density / target_N.
        if "N" in ens or "packing_fraction" in ens:
            raise ValueError(
                "'N'/'packing_fraction' are NVT-only; set ensemble.type='NVT'"
            )
        target_keys = [
            k for k in ("target_eta", "target_density", "target_N") if k in ens
        ]
        if len(target_keys) > 1:
            raise ValueError(f"give only one target, got {target_keys}")
        targeting = len(target_keys) == 1
        has_mu, has_sweep = "mu" in ens, "sweep" in ens

        if targeting:
            if has_sweep:
                raise ValueError("a target and a 'sweep' are mutually exclusive")
            for k in ("target_eta", "target_density", "target_N"):
                if isinstance(ens.get(k), list):
                    raise ValueError(f"ensemble.{k} must be a single value, not a list")
            # record the raw target; convert to a particle count <N> after the
            # box volume is known (computed further below). Starting mu is set
            # there too (explicit `mu` seed, or the ideal-gas guess).
            _target_spec = target_keys[0], float(ens[target_keys[0]])
            mu = float(ens["mu"]) if has_mu else None  # finalized after volume
        else:
            if has_mu and has_sweep:
                raise ValueError("ensemble takes either 'mu' or 'sweep', not both")
            if not has_mu and not has_sweep:
                raise ValueError(
                    "ensemble needs 'mu', a target "
                    "(target_eta/target_density/target_N), or 'sweep'"
                )
            if isinstance(ens.get("mu"), list):
                raise ValueError(
                    "ensemble.mu is a list (a sweep) but a single run got it "
                    "unexpanded. Launch sweeps with `python -m gcmc.launcher`; if "
                    "you did, gcmc/launcher.py on this machine is outdated."
                )
            if has_sweep:
                raise ValueError(
                    "this config has a 'sweep'; launch it with gcmc.launcher, "
                    "which expands it into one run per mu"
                )
            mu = float(ens["mu"])

    # ---- auxiliary ----
    # Two auxiliaries, and they are unrelated apart from sharing this block:
    #
    #   "pmf"          the tabulated g3(r, cos theta) / analytic hard-disk g(r)
    #                  bias. Biases WHICH trial is selected; the main potential
    #                  is untouched. Needs g3_csv (3D) and a gamma schedule.
    #   "short_range"  dual-cutoff CBMC (Vlugt, Martin, Smit, Siepmann, Krishna,
    #                  Mol. Phys. 94, 727 (1998)): the SAME Lennard-Jones,
    #                  truncated at `cutoff` < upper_cutoff, used to propose and
    #                  select; the mismatch against the full potential is divided
    #                  back out in the acceptance. Takes no g3 table and no gamma
    #                  schedule -- there is no exponent to tune, only a radius.
    aux_block = p.get("auxiliary")
    aux_type = None if aux_block is None else aux_block.get("type")
    pmf = aux_type == "pmf"
    short_range = aux_type == "short_range"
    aux_cutoff: float | None = None
    if aux_block is not None:
        _check(
            aux_block,
            {"type", "g3_csv", "eta", "gamma_max", "delta_eta", "cutoff"},
            {"type"},
        )
        if not (pmf or short_range):
            raise ValueError(
                f"unknown auxiliary.type {aux_type!r}; the auxiliaries are "
                f"'pmf' (tabulated g3 selection bias) and 'short_range' "
                f"(dual-cutoff: the same LJ at a shorter radius)"
            )
        if short_range:
            # keys that belong to the pmf bias and mean nothing here. Silently
            # ignoring them would let someone believe a gamma schedule is being
            # applied to a scheme that has no exponent.
            stray = {"g3_csv", "eta", "gamma_max", "delta_eta"} & set(aux_block)
            if stray:
                raise ValueError(
                    f"auxiliary.type='short_range' takes only 'cutoff'; "
                    f"{sorted(stray)} belong to the pmf bias (a tabulated table "
                    f"and its gamma exponents). The short-range auxiliary is the "
                    f"same potential at a shorter radius: it has no table and no "
                    f"exponent to tune."
                )
            if "cutoff" not in aux_block:
                raise ValueError(
                    "auxiliary.type='short_range' requires auxiliary.cutoff (the "
                    "radius used to propose and select, e.g. 1.5). The physics "
                    "cutoff stays run.upper_cutoff and is not changed by it."
                )
            aux_cutoff = float(aux_block["cutoff"])
        elif "cutoff" in aux_block:
            raise ValueError(
                "auxiliary.cutoff belongs to auxiliary.type='short_range'; the "
                "pmf bias is tabulated to its own range and reads run.upper_cutoff"
            )

    # ---- run ----
    run = p.get("run", {})
    _check(
        run,
        {
            "kind",
            "n_cell",
            "n_workers",
            "n_cores",
            "sampling_window",
            "target_cycles",
            "burn_in",
            "traj_every",
            "checkpoint_every",
            "seed",
            "initial_config",
            "matrix_config",
            "radius",
            "move_factor",
            "upper_cutoff",
            "lower_cutoff",
            "no_neigh",
            "cell_length",
            "buffer",
            "x_spacing",
            "y_spacing",
            "control_gain",
            "control_gain_tau",
            "control_window",
            "sweep_label",
            "log_flush_interval",
        },
    )

    # Deprecated run keys: accepted and warned for one release, then removed
    # (parallel_input_plan.md D5). They are IGNORED, not honoured -- keeping them
    # working would be the "something in between" the two submission modes that
    # D5 rules out.
    for _dep, _why in (
        (
            "n_workers",
            "cores are implied by the cell count (one cell is one core); "
            "set run.n_cell instead",
        ),
        ("x_spacing", "it fed only the retired verbatim 2D grid, and is inert"),
        ("y_spacing", "it fed only the retired verbatim 2D grid, and is inert"),
    ):
        if _dep in run:
            log.warning(
                "run.%s is deprecated and IGNORED: %s. It will be removed in a "
                "future release.",
                _dep,
                _why,
            )
    if int(run.get("traj_every", 0)) < 0:
        raise ValueError(
            "run.traj_every must be >= 0 (0 = save only the last "
            "frame; N >= 1 = append every N windows)"
        )
    kind = run.get("kind", "single")
    if kind not in ("single", "parallel"):
        raise ValueError("run.kind must be 'single' or 'parallel'")
    parallel = kind == "parallel"

    # ---- compatibility table ----
    if reweight and lj:
        raise ValueError("reweight requires a hard_core main potential")
    if parallel and reweight:
        raise ValueError("parallel does not wire the reweight (psi6) mode")
    if parallel and confined:
        raise ValueError(
            "parallel does not wire confined pores: the domain decomposition "
            "carries only the fluid in each cell and drops the pore walls, so "
            "the run would fill the forbidden region with no error. Use "
            "run.kind='single'."
        )
    if reweight and not two_dim:
        raise ValueError("reweight (psi6) is a 2D method; use dimension 2")
    if pmf and lj and two_dim:
        raise ValueError("the Lennard-Jones + pmf auxiliary mode is 3D only")
    if pmf and lj and "g3_csv" not in aux_block:
        raise ValueError("Lennard-Jones + pmf requires auxiliary.g3_csv")
    if pmf and branch:
        raise ValueError("branch takes no auxiliary potential")
    if pmf and reweight:
        raise ValueError(
            "reweight uses psi6, not a pmf auxiliary; remove the auxiliary block"
        )
    if pmf and not two_dim and "g3_csv" not in aux_block:
        raise ValueError("3D pmf auxiliary requires auxiliary.g3_csv")

    # ---- short-range auxiliary support matrix -----------------------------
    # Deliberately narrow. Every rejection below is a combination that is either
    # meaningless or simply untested; each is a separate, cheap relaxation later,
    # and refusing them now is what keeps "validated" honest. The reason is spelt
    # out in each message because these are the guards a user will actually hit.
    if short_range:
        if not lj:
            raise ValueError(
                "auxiliary.type='short_range' needs potential.main="
                "'lennard_jones'. It shortens an interaction range, and a hard "
                "core has no range to shorten -- its exclusion floor IS the "
                "interaction, so there is nothing to propose with and nothing "
                "to correct."
            )
        if sampler != "channel":
            raise ValueError(
                f"auxiliary.type='short_range' is supported for "
                f"sampler.type='channel' only (got {sampler!r}). The saving is "
                f"in the transfer-matrix sweep, whose DP window width IS the "
                f"interaction radius. 'branch' would gain almost nothing: its "
                f"clique adjacency is built at the overlap floor, not at the "
                f"cutoff, so the enumeration cost does not shrink. 'metropolis' "
                f"has a single trial and no selection, so there is no proposal "
                f"bias to divide out. 'rosenbluth' would be correct and is a "
                f"clean later extension, but its per-trial weight is a vector "
                f"operation, not a combinatorial one."
            )
        # 2D IS supported. The sweep is dimension-agnostic (it sorts on axis 0
        # and takes full Euclidean distances), the trial body is an oriented
        # rectangle instead of a cuboid, and the periodic guard already carries
        # (dim-1) widths. tests/unit/test_channel_2d.py pins the 2D sweeps
        # against brute-force clique enumeration and the 2D engine against the
        # exact P(N=0)/P(N=1) ratio, for both the fixed and the cyclic rod.
        #
        # WHAT 2D DOES NOT INHERIT, and it is worth knowing before trusting a
        # number: that engine-level 2D check is HARD-CORE only. 2D + LJ + channel
        # runs and gives a plausible density, but it has never been validated
        # against a reference at the engine level, in either direction. The
        # auxiliary is exact relative to whatever the 2D LJ channel already
        # samples -- its own gate is aux-on == aux-off -- so it cannot make that
        # worse, but it cannot vouch for it either. 2D LJ metropolis does run, so
        # a reference for closing that gap exists.
        if qa:
            raise ValueError(
                "auxiliary.type='short_range' does not support the QA frozen "
                "matrix: frozen obstacles must be dropped from the LJ sum in "
                "the member weights, the full-potential energy AND the new "
                "correction term, and QA + channel is not a supported "
                "combination to begin with."
            )
        if _pore_geometry is not None:
            raise ValueError(
                "auxiliary.type='short_range' does not support a confining pore "
                "yet. The algebra carries over (the tail is already disabled in "
                "a pore, so the correction adds cleanly), but pore + channel is "
                "untested."
            )
        # The numeric guard needs lower_cutoff and uc, which are resolved later;
        # it lives immediately after them (search: "short-range auxiliary radius").
    if float(run.get("log_flush_interval", 0.0)) < 0.0:
        raise ValueError(
            f"run.log_flush_interval is a time floor between log flushes in "
            f"seconds and cannot be negative (got "
            f"{run['log_flush_interval']}); 0.0 means flush every row"
        )
    if int(smp.get("depth", 0)) < 0:
        raise ValueError("depth must be >= 0")
    if metropolis:
        forbidden = {"depth", "trials", "trial_density", "trial_box_side"} & set(smp)
        if forbidden:
            raise ValueError(
                "metropolis is standard GCMC: it takes no "
                "depth, trials, trial_density, or trial_box_side"
            )
    if channel and "trial_box_side" in smp:
        raise ValueError(
            "the channel sampler has no trial_box_side: the trial "
            "region is an oriented cuboid, not a cube. Give "
            "trial_density (spheres per unit volume) with "
            "channel_length and channel_width instead."
        )
    if "trials" in smp and "trial_density" in smp:
        raise ValueError(
            "give either sampler.trials or sampler.trial_density, not both"
        )

    # ---- derived constants ----
    temp = float(pot["T"]) if lj else 1.0
    beta = 1.0 / temp
    # box may be a scalar (cubic, the legacy path) or a per-axis list/tuple of
    # `dim` lengths (A2.1, orthorhombic). Cubic keeps box_sides None so the
    # scalar path is untouched; per-axis stores the tuple and sets the scalar
    # `box` to min(sides) for the legacy 2*r_int minimum-image guard.
    box_sides: tuple[float, ...] | None = None
    if "box" in ens:
        _raw_box = ens["box"]
        if isinstance(_raw_box, (list, tuple)):
            if len(_raw_box) != dim:
                raise ValueError(
                    f"ensemble.box as a list must give one length per axis "
                    f"(dimension={dim}, got {len(_raw_box)})"
                )
            box_sides = tuple(float(v) for v in _raw_box)
            if len(set(box_sides)) == 1:
                # an isotropic list is just the cube; collapse so fingerprints
                # and the RNG path stay identical to a scalar box
                box_sides = None
                box = float(_raw_box[0])
            else:
                box = min(box_sides)
        else:
            box = float(_raw_box)
    else:
        box = default_side(two_dim, parallel)

    radius = float(run.get("radius", 0.5))
    diameter = 2.0 * radius  # physical particle size / LJ sigma
    disk = math.pi * radius * radius
    # One exclusion floor for the whole run. Hard cores exclude at the diameter.
    # Every LJ mode (metropolis, rosenbluth, branch, channel) shares a single hard
    # floor of 0.9 sigma unless the user overrides run.lower_cutoff, so the sampled
    # model is identical across samplers and cross-mode checks compare like with
    # like. The model each LJ run samples is therefore "truncated LJ plus a hard
    # core at lower_cutoff", not pure LJ; see the module docstring. Previously the
    # value depended on the sampler (0.85 for plain LJ, 0.9 for branch/channel),
    # so metropolis and the biased samplers converged to different densities.
    _lower_default = 0.9 if lj else diameter
    lower_cutoff = float(run.get("lower_cutoff", _lower_default))
    r_squared = lower_cutoff * lower_cutoff

    channel_cyclic = False
    if metropolis:
        trial_box_side = 0.0
        cav = 0.0
        channel_length = 0.0
        channel_width = 0.0
        n_trials = 1
        trial_density = 0.0
    elif channel:
        # The channel trial region is an oriented cuboid: one long axis of length
        # channel_length and a square cross-section channel_width x channel_width.
        # That is the exact body the trial spheres are dropped into (see the
        # channel branch of MC_engine.advance), so its volume is the natural
        # normaliser. trial_density is spheres per unit volume, hence the count is
        # simply density * volume. trial_box_side has no meaning here and was
        # rejected above.
        trial_box_side = 0.0
        cav = 0.0
        _clen = smp.get("channel_length", 0.0)
        if isinstance(_clen, (list, tuple)):
            raise ValueError(
                f"sampler.channel_length must be a single number (the cuboid "
                f"length along the sweep axis) or the string 'cyclic' for a "
                f"wall-to-wall rod; got a list of {len(_clen)}. The channel body "
                f"is one oriented cuboid, so it has one length; per-axis lists "
                f"apply to run.cell_length (parallel decomposition), not here."
            )
        # "cyclic" is the canonical spelling: the rod runs wall to wall, so its
        # two ends meet through the periodic boundary and the transfer-matrix
        # sweep closes on itself (ChannelSweepCyclic scores that seam with true
        # minimum-image distances instead of an axial window). The four legacy
        # spellings are still accepted so existing params files keep working;
        # they are exactly equivalent, and because the parser collapses all of
        # them to channel_length = 0.0 the run identity does not record which
        # word was typed. See test_channel_cyclic_spellings_share_one_run_id.
        channel_cyclic = isinstance(_clen, str) and _clen.strip().lower() in (
            "cyclic",
            "full",  # legacy
            "box",  # legacy
            "full_box",  # legacy
            "span",  # legacy
        )
        channel_width = float(smp.get("channel_width", 0.0))
        if not (diameter <= channel_width <= 2.0 * diameter + 1e-9):
            raise ValueError(
                f"channel_width must lie in [sigma, 2 sigma] = "
                f"[{diameter}, {2 * diameter}]; got {channel_width}"
            )
        if channel_cyclic:
            # Cyclic rod: the length is set per move by the orientation (wall
            # to wall, box for an axis line up to box*sqrt(3) for the body
            # diagonal), so there is no fixed channel_length and no fixed trial
            # count. trial_density (spheres per unit volume) fixes the linear
            # density; the engine draws density * chord_volume trials each move.
            # A nominal cap sized on the longest possible chord bounds arrays.
            channel_length = 0.0
            if "trials" in smp:
                raise ValueError(
                    "cyclic channel sets its length per move, so a fixed "
                    "trial count is ambiguous; give sampler.trial_density "
                    "(spheres per unit volume) instead of sampler.trials"
                )
            trial_density = float(smp.get("trial_density", 0.0))
            if trial_density <= 0.0:
                raise ValueError("cyclic channel needs trial_density > 0")
            # longest possible chord is the box body diagonal: box*sqrt(dim).
            # The cross-section carries (dim-1) widths, so the swept measure is
            # chord * w**(dim-1) -- an area in 2D, a volume in 3D.
            _maxchord = box * (float(dim) ** 0.5)
            n_trials = max(
                2,
                int(round(trial_density * _maxchord * channel_width ** (dim - 1))),
            )
        else:
            # A string that reached here is not one of the cyclic spellings, so
            # it is either a typo for "cyclic" or a malformed number. float()
            # would report `could not convert string to float`, which names
            # neither cause; say what the key actually accepts instead.
            if isinstance(_clen, str):
                raise ValueError(
                    f"sampler.channel_length={_clen!r} is not recognised: give a "
                    f'positive number for a fixed-length rod, or "cyclic" for a '
                    f"wall-to-wall rod whose ends meet through the periodic "
                    f'boundary (legacy spellings "full", "box", "full_box", '
                    f'"span" are also accepted)'
                )
            channel_length = float(_clen)
            if channel_length <= 0.0:
                raise ValueError(
                    "channel sampler needs channel_length > 0 "
                    '(or the string "cyclic" for a wall-to-wall '
                    "rod), and channel_width > 0"
                )
            # 3D body is a cuboid L x w x w (a volume); 2D body is a rectangle
            # L x w (an area). trial_density is spheres per unit measure in the
            # run's dimension, so the count is density x measure in both cases.
            chan_vol = channel_length * channel_width ** (dim - 1)
            if "trials" in smp:
                n_trials = int(smp["trials"])
                trial_density = round(n_trials / chan_vol, 6)
            else:
                trial_density = float(smp.get("trial_density", 0.0))
                n_trials = int(round(trial_density * chan_vol))
    else:
        trial_box_side = float(smp.get("trial_box_side", 3.0))
        cav = trial_box_side / 2.0
        channel_length = float(smp.get("channel_length", 0.0))
        channel_width = float(smp.get("channel_width", 0.0))
        # `trial_density` is a REDUCED density: trial molecules per unit measure
        # of the trial body, in units of one molecule's own measure. Both factors
        # must therefore be taken in the run's dimension. This used to read
        # `span = trial_box_side**2` and `disk = pi r^2` in 3D as well, i.e. an
        # area over an area for a cubic body, so a 3D `trial_density` was not the
        # density it claimed to be and did not scale with trial_box_side
        # correctly. 2D is unchanged and bit-identical; 3D counts change, so a 3D
        # config carried over from before must be restated (pin `trials` outright
        # if you need to reproduce an earlier trial count exactly).
        span = trial_box_side**dim
        unit = disk if two_dim else (4.0 / 3.0) * math.pi * radius**3
        if "trial_density" in smp:
            trial_density = float(smp["trial_density"])
            n_trials = int((trial_density * span) / unit)
        else:
            n_trials = int(smp.get("trials", 30))
            trial_density = round((n_trials * unit) / span, 3)

    no_neigh = int(run.get("no_neigh", 40 if two_dim else 75))
    if "upper_cutoff" in run:
        uc = float(run["upper_cutoff"])
    elif reweight:
        uc = 3.0 * radius
    elif lj:
        uc = 2.5
    else:
        uc = 3.7545 if two_dim else 2.0

    # volume is the product of the per-axis sides; cubic (box_sides None) gives
    # box**dim exactly, so the scalar path is bit-identical
    _sides = box_sides if box_sides is not None else (box,) * dim
    if two_dim:
        vol_mol = math.pi * radius**2
        total_vol = float(_sides[0] * _sides[1])
    else:
        vol_mol = (4.0 / 3.0) * math.pi * radius**3
        total_vol = float(_sides[0] * _sides[1] * _sides[2])

    # NVT target as an exact integer N
    target_n = None
    if nvt:
        if "N" in ens:
            target_n = int(ens["N"])
            if target_n < 0:
                raise ValueError("ensemble.N must be >= 0")
        else:
            pf = float(ens["packing_fraction"])
            if not 0.0 <= pf < 1.0:
                raise ValueError("packing_fraction must be in [0, 1)")
            target_n = int(round(pf * total_vol / vol_mol))

    # muVT target control: now that the box volume is known, convert the target
    # to a particle count <N>, and finalize the starting mu (ideal-gas guess if
    # the user gave no explicit seed).
    if _target_spec is not None:
        kind, val = _target_spec
        if kind == "target_N":
            target_count = val
        elif kind == "target_eta":
            target_count = val * total_vol / vol_mol
        else:  # target_density (rho* = N / V)
            target_count = val * total_vol
        if target_count <= 0:
            raise ValueError("the target must correspond to N > 0")
        if mu is None:
            mu = math.log(target_count / total_vol)  # ideal-gas cold start

    lj_tail_pref = None
    lj_tail_on = bool(pot.get("tail", True))
    if "tail" in pot and not lj:
        raise ValueError("potential.tail applies to lennard_jones only")
    # The bulk tail correction assumes a uniform fluid with g(r)=1 beyond rc
    # filling all space. In a pore or a quenched-annealed matrix the accessible
    # volume is neither uniform nor space-filling, so that integral is simply
    # wrong: it adds an N^2 energy the confined system never has. Disable it there
    # by default and say so; only raise if the user explicitly asked for the tail,
    # rather than silently dropping a setting they set on purpose.
    _confined_or_qa = (_pore_geometry is not None) or (
        run.get("matrix_config") is not None
    )
    if lj and _confined_or_qa:
        _where = (
            "a hard pore"
            if _pore_geometry is not None
            else "a quenched-annealed matrix"
        )
        if "tail" in pot and bool(pot["tail"]):
            raise ValueError(
                f"potential.tail=true is invalid with {_where}: the bulk LJ tail "
                f"correction assumes a uniform space-filling fluid, which a confined "
                f"or matrix geometry is not. Remove potential.tail (it defaults off "
                f"here) or run in bulk."
            )
        if lj_tail_on:
            log.warning(
                "LJ tail correction disabled: %s is not a uniform bulk fluid, "
                "so the bulk tail integral does not apply.",
                _where,
            )
        lj_tail_on = False

    # ---- short-range auxiliary radius (needs both cutoffs resolved) --------
    # The floor is deliberately NOT made configurable per-auxiliary: the clique
    # adjacency is built at `lower_cutoff`, so a separate aux floor would change
    # which cliques exist rather than only how they are weighted, and a clique
    # legal under the aux but overlapping under the main potential would carry
    # true weight zero. Admissibility is also read off the aux-radius neighbour
    # query as `all(d >= lower_cutoff)`, which only means anything while that
    # query reaches past the floor.
    if short_range:
        assert aux_cutoff is not None  # set wherever short_range is True
        # The RANGE check comes first, and the staged-delivery interlock second.
        # Ordering matters for the message the user gets: a radius below the
        # exclusion floor is invalid whatever the implementation status is, so
        # reporting it as "not implemented yet" would be actively misleading.
        if not (lower_cutoff < aux_cutoff <= uc):
            raise ValueError(
                f"auxiliary.cutoff must satisfy run.lower_cutoff < cutoff <= "
                f"run.upper_cutoff, i.e. {lower_cutoff} < cutoff <= {uc}; got "
                f"{aux_cutoff}. At or below the floor the auxiliary cannot see "
                f"the overlap test it has to respect; above the physics cutoff "
                f"it is not an auxiliary at all. cutoff == upper_cutoff is "
                f"allowed on purpose: it is the exact no-op control."
            )
    if lj:
        if two_dim:
            # 2D mean-field tail (g(r)=1 beyond rc), integrated over the plane:
            # U_tail = (4 pi / A) N^2 [ rc^-10/10 - rc^-4/4 ].
            x = uc**-10 / 10.0 - uc**-4 / 4.0
            lj_tail_pref = (4.0 * math.pi / total_vol * x) if lj_tail_on else 0.0
        else:
            x = uc**-9 / 3.0 - uc**-3
            lj_tail_pref = (
                ((8.0 / 3.0) * math.pi / total_vol * x) if lj_tail_on else 0.0
            )

    # three orthogonal axes; the flat engine tag is derived from them so nothing
    # downstream has to special-case the combinations by hand.
    potential = "lennard_jones" if lj else "hard_sphere"
    bias = (
        "pmf"
        if pmf
        else ("short_range" if short_range else ("psi6" if reweight else None))
    )
    ptype = _derive_potential_type(potential, bias, sampler)

    has_matrix_cfg = run.get("matrix_config") is not None
    if qa and not has_matrix_cfg:
        raise ValueError(
            "ensemble.type='QA' (quenched-annealed) requires run.matrix_config: "
            "an absolute path to the frozen matrix configuration (an NVT "
            "checkpoint.npz, .extxyz, or plain x y z file)."
        )
    if has_matrix_cfg and not qa:
        raise ValueError(
            "run.matrix_config is only valid with ensemble.type='QA'. A frozen "
            "matrix defines a quenched-annealed run; set ensemble.type='QA' "
            "(it is grand-canonical, so give a chemical potential mu, not N)."
        )
    if qa:
        # QA support is implemented only on the deep serial rosenbluth/branch
        # path. Reject every combination that would silently ignore the matrix
        # and produce fluid overlapping the frozen obstacles with no error.
        if ptype not in ("hard_sphere", "ljbranch"):
            raise ValueError(
                f"QA runs support hard-core and LJ+branch potentials only, not "
                f"{ptype!r}; the other modes pass kd-tree neighbour indices "
                f"downstream where frozen-matrix indices would be misread as "
                f"fluid particles"
            )
        if metropolis:
            raise ValueError(
                "QA runs are not supported with the 'metropolis' sampler: its "
                "insertion/translation overlap test is built on the fluid only "
                "and would let fluid overlap the frozen matrix. Use 'rosenbluth' "
                "or 'branch'."
            )
        if kind == "parallel":
            raise ValueError(
                "QA runs are not supported with run.kind='parallel': the domain "
                "decomposition carries only the fluid in each cell and would "
                "drop the frozen matrix. Use run.kind='single'."
            )
    if channel:
        import math as _m

        if parallel:
            raise ValueError("channel sampler is serial only")
        if not full_repacking and int(smp.get("depth", 0)) < 1:
            raise ValueError(
                "channel sampler needs depth >= 1 (depth 0 has no "
                "clique structure to sweep), or set full_repacking"
            )
        # periodic guard: for the FIXED-length rod, insertion and removal share
        # one oriented cuboid, so the whole pool lives in that body. Its space
        # diagonal must stay below the box minus the interaction range (sigma for
        # hard core, rc for LJ) so local distances equal periodic distances. The
        # cyclic rod lifts this guard on purpose and instead computes true
        # minimum-image distances in the sweep, so the guard does not apply.
        if not channel_cyclic:
            # The margin is the largest distance the SWEEP will ever read, which is
            # not necessarily the physics cutoff. The guarded sweep works on local
            # cuboid coordinates with no minimum image, and its masks are built at
            # `lower` (overlap) and at the interaction radius; with a short-range
            # auxiliary that radius is aux_cutoff, so no pair beyond aux_cutoff is
            # ever consulted and the guard only has to exclude wrapped images
            # closer than THAT. Everything evaluated at the full upper_cutoff --
            # the member weights, the tail, and the dual-cutoff correction's two
            # energies -- goes through the CellList or an explicit minimum image,
            # so none of it depends on this guard.
            #
            # What it buys is modest and worth stating plainly: the cuboid diagonal
            # is orientation-independent, so this relaxes the admissible rod LENGTH,
            # not the set of orientations. At box 12, w 1.4, uc 2.5 the longest
            # legal rod goes from 9.29 to 10.31 (+11%) at aux_cutoff 1.5. The
            # cyclic rod already lifts the guard entirely and spans the box, so this
            # matters only if a fixed-length rod is wanted for its own sake.
            guard_cut = uc if lj else diameter
            if lj and short_range:
                assert aux_cutoff is not None
                guard_cut = aux_cutoff
            # body diagonal: cuboid sqrt(L^2 + 2 w^2) in 3D, rectangle
            # sqrt(L^2 + w^2) in 2D. The cross-section carries (dim-1) widths.
            diag = _m.sqrt(channel_length**2 + (dim - 1) * channel_width**2)
            if diag > box - guard_cut:
                raise ValueError(
                    f"channel geometry violates the periodic guard: cuboid "
                    f"diagonal {diag:.3f} must be <= box - {guard_cut:.3f} = "
                    f"{box - guard_cut:.3f} (the margin is the largest distance "
                    f"the sweep reads, i.e. auxiliary.cutoff when a short-range "
                    f"auxiliary is configured, else run.upper_cutoff)"
                )

    # ---- fail fast on physically invalid runs, one specific message each ----
    _r_int = uc if lj else diameter
    if box < 2.0 * _r_int - 1e-12:
        _which = "upper_cutoff" if lj else "diameter"
        raise ValueError(
            f"box={box} is below twice the interaction range 2*{_r_int}="
            f"{2.0 * _r_int}: the minimum-image convention requires box >= 2*"
            f"{_which} so a particle never interacts with its own image"
        )
    move_factor = float(run.get("move_factor", 0.25))
    if move_factor <= 0.0:
        raise ValueError(
            "run.move_factor must be > 0 (it scales the trial "
            "displacement; zero freezes every translation)"
        )
    sampling_window = int(run.get("sampling_window", 1000))
    if sampling_window < 1:
        raise ValueError("run.sampling_window must be >= 1 exchange step")
    if not metropolis:
        _depth = int(smp.get("depth", 0))
        if n_trials < 1:
            raise ValueError(
                f"the trial pool is empty: the requested trial_density/trials "
                f"gives {n_trials} trial spheres. Raise trial_density (or trials) "
                f"so at least one sphere is placed"
            )
        if branch and n_trials < _depth + 1:
            raise ValueError(
                f"branch/channel needs at least depth+1={_depth + 1} trial "
                f"spheres to place a depth-{_depth} move, but only {n_trials} "
                f"were requested; raise trials or trial_density"
            )

    assert mu is not None  # finalized above (explicit, target seed, or NVT 0.0)
    # A2.2: per-axis boxes are threaded through the serial metropolis fluid path
    # (wrap, minimum image, uniform draws, volume). The sweep samplers place
    # trial bodies against a scalar side (Phase 5 territory), the confined pore
    # geometry is defined inside a cubic cell, and the parallel decomposition is
    # scalar throughout. Rather than half-convert those, refuse a non-cubic box
    # there with a specific message; cubic runs are unaffected.
    # An orthorhombic (per-axis) ensemble.box is a general feature: every sampler
    # and the parallel decomposition route wrap, insertion range and volume
    # through box_dims. The one genuine exception is a confined pore, whose slab/
    # cylinder geometry is defined inside a cubic cell; refuse that specific mix.
    if box_sides is not None and confined:
        raise ValueError(
            "confined pores are defined inside a cubic cell; give a scalar ensemble.box"
        )
    # ---- parallel cell geometry: two mutually exclusive submission modes ----
    # AUTO   (run.n_cell given): the cell COUNTS are the submission. Lengths and
    #        the buffer are derived here; one cell is one core, so no core count
    #        is ever stated twice (decisions D1-D6).
    # MANUAL (no run.n_cell): the user states the whole geometry and the count
    #        falls out of it, exactly as it always has. cell_length and buffer
    #        are then REQUIRED: the historical defaults (30 and 2) are unusable
    #        for any box <= 32 and were the landmine behind two broken examples.
    # Nothing in between: mixing the two is an error, not a merge (D5).
    _MANUAL_GEOM = ("cell_length", "buffer", "x_spacing", "y_spacing")
    cell_counts: tuple[int, ...] | None = None
    cell_length_sides: tuple[float, ...] | None = None
    n_cell_submitted: int | list[int] | None = None
    cell_length = 30.0
    buffer_v = 2.0

    if parallel:
        # the interaction reach a border ring has to contain; `radius` is a RUN
        # key, so reading it off `ens` (as this guard used to) always yielded the
        # 0.5 default and left the guard under-strict above it.
        # The reach a border ring must contain. For LJ that is the truncation
        # radius actually in force (`uc`, i.e. run.upper_cutoff or its default);
        # for hard core it is the exclusion diameter. Both of the old lookups
        # here were dead: `pot["rc"]` can never fire because "rc" is not in the
        # potential allowlist, and `ens.get("radius", 0.5)` always returned the
        # default because `radius` is a RUN key -- so the guard checked a fixed
        # LJ reach of 2.5 and a fixed hard-core reach of 1.0 whatever the config
        # said (recorded in cell_unify_plan.md's schema audit).
        _reach = float(uc) if lj else 2.0 * radius
        _sides_for_cells = list(box_sides) if box_sides is not None else [box] * dim

        if "n_cell" in run:
            _clash = [k for k in _MANUAL_GEOM if k in run]
            if _clash:
                raise ValueError(
                    f"run.n_cell sets the decomposition on its own, so "
                    f"{', '.join('run.' + k for k in _clash)} cannot be given "
                    f"alongside it. Either submit run.n_cell and let the cell "
                    f"lengths and buffer be derived, or omit run.n_cell and "
                    f"state run.cell_length and run.buffer yourself."
                )
            n_cell_submitted = (
                [int(v) for v in run["n_cell"]]
                if isinstance(run["n_cell"], (list, tuple))
                else int(run["n_cell"])
            )
            buffer_v = _reach
            _plan = cell_plan.plan_auto(
                n_cell_submitted,
                _sides_for_cells,
                buffer_v,
                min_cell=max(2.0 * cav, _reach),
                min_cell_desc=(
                    f"the larger of twice the cavity radius ({2.0 * cav:g}) and "
                    f"the interaction reach ({_reach:g})"
                ),
            )
        else:
            _missing = [k for k in ("cell_length", "buffer") if k not in run]
            if _missing:
                raise ValueError(
                    f"a parallel run needs either run.n_cell (and everything "
                    f"else is derived) or the full geometry; "
                    f"{', '.join('run.' + k for k in _missing)} "
                    f"{'is' if len(_missing) == 1 else 'are'} missing. "
                    f"run.n_cell is the recommended form: one cell is one core."
                )
            buffer_v = float(run["buffer"])
            if buffer_v < _reach - 1e-9:
                raise ValueError(
                    f"parallel buffer ({buffer_v}) must be >= the interaction "
                    f"range ({_reach}): border rings must contain every molecule "
                    f"that can interact with a live one"
                )
            _plan = cell_plan.plan_manual(
                run["cell_length"],
                buffer_v,
                _sides_for_cells,
                min_cell=max(2.0 * cav, _reach),
                min_cell_desc=(
                    f"the larger of twice the cavity radius ({2.0 * cav:g}) and "
                    f"the interaction reach ({_reach:g})"
                ),
            )

        cell_counts = _plan.n_ax
        # An isotropic set of lengths collapses to the scalar field so the cubic
        # path stays bit-identical (cell_length_sides None is the legacy route).
        if len(set(_plan.cell_length)) == 1:
            cell_length = float(_plan.cell_length[0])
        else:
            cell_length_sides = _plan.cell_length
            cell_length = min(_plan.cell_length)

        # Accuracy, not correctness: the depth>=1 cell proposal carries a bias
        # set by the cell/cavity ratio (+19% at 2.7 cavity widths, +0.9 +/- 0.6%
        # at 10; cell_unify_plan.md U3). Warn with the number so an operator can
        # price it; never refuse, because the ratio is a choice, not an error.
        # The depth>=1 cell-proposal caveat is reported ONCE, by
        # describe_parallel on stdout. It used to be a log.warning here, which
        # fired on every from_params call -- the launcher builds a config to size
        # the core budget before it launches anything, so a single sweep printed
        # it three times before the run even started.

    _raw_seed = run.get("seed")
    if isinstance(_raw_seed, (list, tuple)):
        raise ValueError(
            "run.seed is a list (a sweep) but a single run got it unexpanded. "
            "Launch sweeps with `python -m gcmc.launcher`; if you did, "
            "gcmc/launcher.py on this machine is outdated."
        )
    seed_val = int(_raw_seed) if _raw_seed is not None else None

    return Config(
        potential=potential,
        bias=bias,
        sampler=sampler,
        dimension=dim,
        chem_pot=mu,
        ensemble_type=ens_type,
        target_n=target_n,
        box=box,
        box_sides=box_sides,
        target_count=target_count,
        control_gain=float(run.get("control_gain", 0.5)),
        control_gain_tau=float(run.get("control_gain_tau", 20.0)),
        control_window=int(run.get("control_window", 20)),
        cavity_radius=cav,
        trial_box_side=trial_box_side,
        trial_density=trial_density,
        depth=(0 if (metropolis or full_repacking) else int(smp.get("depth", 0))),
        branch=branch,
        metropolis=metropolis,
        channel=channel,
        channel_length=channel_length,
        channel_width=channel_width,
        channel_cyclic=channel_cyclic,
        full_repacking=full_repacking,
        move_factor=move_factor,
        temp=temp,
        rc=(uc if lj else None),
        eta=(
            float(aux_block.get("eta", 0.69))
            if aux_block is not None and two_dim
            else None
        ),
        gamma_max=(
            float(aux_block["gamma_max"])
            if aux_block is not None and "gamma_max" in aux_block
            else None
        ),
        delta_eta=(
            float(aux_block["delta_eta"])
            if aux_block is not None and "delta_eta" in aux_block
            else None
        ),
        g3_csv=(aux_block.get("g3_csv") if pmf and not two_dim else None),
        aux_cutoff=aux_cutoff,
        two_dim=two_dim,
        parallel=parallel,
        beta=beta,
        fugacity=math.exp(beta * mu),
        radius=radius,
        diameter=diameter,
        r_squared=r_squared,
        no_of_trial_molecules=n_trials,
        no_neigh=no_neigh,
        upper_cutoff=uc,
        lower_cutoff=lower_cutoff,
        volume_of_molecule=vol_mol,
        total_volume=total_vol,
        lj_tail_pref=lj_tail_pref,
        gamma_dict=_gamma_dict(pmf, reweight, two_dim),
        mode=kind,
        n_workers=(_plan.n_cells if parallel else 1),
        n_cell=n_cell_submitted,
        cell_counts=cell_counts,
        cell_length=cell_length,
        cell_length_sides=cell_length_sides,
        buffer=buffer_v,
        seed=seed_val,
        x_spacing=(float(run["x_spacing"]) if "x_spacing" in run else None),
        y_spacing=(float(run["y_spacing"]) if "y_spacing" in run else None),
        sampling_window=sampling_window,
        target_cycles=float(run.get("target_cycles", 1.0025e9)),
        burn_in=int(run.get("burn_in", 0)),
        traj_every=int(run.get("traj_every", 0)),
        checkpoint_every=int(run.get("checkpoint_every", 1)),
        log_flush_interval=float(run.get("log_flush_interval", 0.0)),
        initial_config=run.get("initial_config"),
        matrix_config=run.get("matrix_config"),
        sweep_label=run.get("sweep_label"),
        pore_geometry=_pore_geometry,
        pore_H=_pore_H,
        pore_R=_pore_R,
        pore_axis_len=_pore_axis_len,
    )


def describe_parallel(cfg: Config) -> str:
    """Human-readable report of the decomposition a parallel run resolved to.

    Printed to STDOUT by the runner (so it lands in the scheduler's .out beside
    the traceback if anything fails), never into run.log, which is the CSV data
    file and must stay machine-parsable.

    Everything derived is shown next to the constraint that shaped it, because
    the whole point of submitting `n_cell` is that the cell geometry is no longer
    something the operator typed and can read back from their own params file.
    Per-axis quantities are printed per axis: the gap is `pitch - cell_length`
    and equals the buffer exactly in n_cell mode, but in manual mode it is
    whatever is left over after the floor divide and can differ per axis, which
    is worth seeing rather than assuming.
    """
    if cfg.cell_counts is None:
        return "decomposition: serial run (one core, no cells)"

    dim = cfg.dimension
    axes = "xyz"[:dim]
    sides = [float(v) for v in cfg.box_dims]
    n_ax = list(cfg.cell_counts)
    cell = [float(v) for v in cfg.cell_dims]
    pitch = [s / n for s, n in zip(sides, n_ax)]
    gap = [p - c for p, c in zip(pitch, cell)]

    reach = (
        float(cfg.upper_cutoff)
        if cfg.potential_type == "lennard_jones"
        else 2.0 * float(cfg.radius)
    )
    floor = max(2.0 * float(cfg.cavity_radius), reach)
    mode = (
        f"run.n_cell = {cfg.n_cell}"
        if cfg.n_cell is not None
        else "manual geometry (run.cell_length + run.buffer)"
    )

    def row(label: str, vals: list[float], fmt: str = "10.4f") -> str:
        body = "  ".join(f"{a}={v:{fmt}}" for a, v in zip(axes, vals))
        return f"  {label:<22}: {body}"

    lines = [
        "=" * 72,
        f"PARALLEL DECOMPOSITION  ({dim}D, {cfg.sampler}, depth={cfg.depth})",
        "=" * 72,
        f"  {'submitted as':<22}: {mode}",
        row("box side", sides),
        f"  {'cells per axis':<22}: "
        + " x ".join(str(n) for n in n_ax)
        + f"  = {cfg.n_workers} cells",
        f"  {'cores':<22}: {cfg.n_workers}   (one cell is one core)",
        row("cell pitch", pitch),
        row("cell length", cell),
        row("gap between cells", gap),
        f"  {'buffer (border ring)':<22}: {cfg.buffer:.4f}"
        f"   (interaction reach {reach:.4f})",
        f"  {'shortest cell edge':<22}: {min(cell):.4f}"
        f"   (floor {floor:.4f} = max(2*cavity {2.0 * cfg.cavity_radius:.4f},"
        f" reach {reach:.4f}))",
    ]
    if cfg.cavity_radius > 0.0:
        width = 2.0 * float(cfg.cavity_radius)
        lines.append(
            f"  {'cavity width':<22}: {width:.4f}"
            f"   -> shortest cell = {min(cell) / width:.2f} cavity widths"
        )
    lines.append("=" * 72)
    return "\n".join(lines)

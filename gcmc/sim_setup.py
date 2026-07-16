"""Configuration: one place that turns a params.json into a frozen Config and
derives every mode-dependent constant the rest of the code needs.

params.json has five blocks, one per axis. Inactive keys cannot appear (an LJ
temperature on a hard-core run, a pmf block under branch), so nothing leaks.
Validation is a per-section check plus the compatibility rules below, each with
a specific message.

Schema
------
{
  "potential":  {"main": "hard_core" | "lennard_jones",
                 "T": float},                       # T required iff main = lennard_jones

  "auxiliary":  {"type": "pmf",                      # OPTIONAL block; omit = no aux
                 "g3_csv": str,                      # required iff present and 3D
                 "eta": float, "gamma_max": float, "delta_eta": float},

  "sampler":    {"type": "metropolis" | "rosenbluth" | "branch" | "channel" | "reweight",
                 "depth": int, "use_depth": bool,    # forbidden for metropolis; channel needs depth >= 1
                 "trials": int | "trial_density": float,   # forbidden for metropolis
                 "trial_box_side": float,                  # rosenbluth/branch only (cube trial box); rejected for channel
                 "channel_length": float, "channel_width": float},  # channel only: oriented cuboid extents

  "ensemble":   {"mu": float | "sweep": [float],     # exactly one; sweep is launcher-only
                 "dimension": 2 | 3, "box": float,
                 "type": "muVT" | "NVT" | "QA" | "confined",   # default muVT
                 "N": int | "packing_fraction": float,         # NVT target
                 "pore": {"geometry": "slit" | "cylinder", ...}},  # confined only

  "run":        {"kind": "single" | "parallel", "n_workers": int,
                 "sampling_window": int, "target_cycles": float, "burn_in": int,
                 "traj_every": int, "checkpoint_every": int, "seed": int,
                 "initial_config": str, "matrix_config": str,   # matrix_config: QA only
                 "radius": float, "move_factor": float,
                 "upper_cutoff": float, "lower_cutoff": float, "no_neigh": int,
                 "cell_length": float, "buffer": float, "x_spacing": float, "y_spacing": float}
}

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

Samplers x ensemble:
  muVT        every sampler
  NVT         every sampler except reweight paths that assume muVT targeting
  confined    grand-canonical samplers (give mu); 3D only; no matrix
  QA          hard_core or LJ+branch only; not metropolis; needs run.matrix_config

Dimension:
  reweight    2D only
  channel     3D only
  confined    3D only
  LJ + pmf    3D only

Parallelism (run.kind='parallel'):
  not available for reweight or QA; channel is serial only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .potentials.base import Potential

import logging
import math
from dataclasses import dataclass

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
    required keys must be present."""
    extra = set(section) - set(allowed)
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
    bias: str | None  # None | "pmf" | "psi6"
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
    n_workers: int
    cell_length: float
    buffer: float
    x_spacing: float | None
    y_spacing: float | None
    sampling_window: int
    target_cycles: float
    burn_in: int
    traj_every: int
    checkpoint_every: int
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
    channel: bool = False
    channel_length: float = 0.0  # cuboid length along the sweep axis
    channel_width: float = 0.0  # cuboid cross-section (square side)
    channel_full_span: bool = False  # channel_length == "full": wall-to-wall
    # oriented rod, any orientation, exact PBC
    # (the periodic guard is lifted; the sweep
    # uses minimum-image distances + retention)

    @property
    def box_dims(self) -> list[float]:
        """Per-axis box lengths (cubic: the side replicated `dimension`
        times), the form scipy and the cell list expect."""
        return [self.box] * self.dimension

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
    aux_block = p.get("auxiliary")
    pmf = aux_block is not None
    if aux_block is not None:
        _check(aux_block, {"type", "g3_csv", "eta", "gamma_max", "delta_eta"}, {"type"})
        if aux_block["type"] != "pmf":
            raise ValueError("the only auxiliary potential is 'pmf'")

    # ---- run ----
    run = p.get("run", {})
    _check(
        run,
        {
            "kind",
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
        },
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
    if reweight and not two_dim:
        raise ValueError("reweight (psi6) is a 2D method; use dimension 2")
    if pmf and lj and two_dim:
        raise ValueError("the Lennard-Jones + pmf auxiliary mode is 3D only")
    if aux_block is not None and lj and "g3_csv" not in aux_block:
        raise ValueError("Lennard-Jones + pmf requires auxiliary.g3_csv")
    if pmf and branch:
        raise ValueError("branch takes no auxiliary potential")
    if pmf and reweight:
        raise ValueError(
            "reweight uses psi6, not a pmf auxiliary; remove the auxiliary block"
        )
    if aux_block is not None and not two_dim and "g3_csv" not in aux_block:
        raise ValueError("3D pmf auxiliary requires auxiliary.g3_csv")
    if parallel:
        buffer_v = float(run.get("buffer", 2.0))
        reach = float(pot["rc"]) if (lj and "rc" in pot) else (2.5 if lj else None)
        if reach is None:
            reach = 2.0 * float(ens.get("radius", 0.5))
        if buffer_v < reach - 1e-9:
            raise ValueError(
                f"parallel buffer ({buffer_v}) must be >= the interaction "
                f"range ({reach}): border rings must contain every molecule "
                f"that can interact with a live one"
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
    box = float(ens["box"]) if "box" in ens else default_side(two_dim, parallel)

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

    channel_full_span = False
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
        channel_full_span = isinstance(_clen, str) and _clen.strip().lower() in (
            "full",
            "box",
            "full_box",
            "span",
        )
        channel_width = float(smp.get("channel_width", 0.0))
        if not (diameter <= channel_width <= 2.0 * diameter + 1e-9):
            raise ValueError(
                f"channel_width must lie in [sigma, 2 sigma] = "
                f"[{diameter}, {2 * diameter}]; got {channel_width}"
            )
        if channel_full_span:
            # Full-span rod: the length is set per move by the orientation (wall
            # to wall, box for an axis line up to box*sqrt(3) for the body
            # diagonal), so there is no fixed channel_length and no fixed trial
            # count. trial_density (spheres per unit volume) fixes the linear
            # density; the engine draws density * chord_volume trials each move.
            # A nominal cap sized on the longest possible chord bounds arrays.
            channel_length = 0.0
            if "trials" in smp:
                raise ValueError(
                    "full-span channel sets its length per move, so a fixed "
                    "trial count is ambiguous; give sampler.trial_density "
                    "(spheres per unit volume) instead of sampler.trials"
                )
            trial_density = float(smp.get("trial_density", 0.0))
            if trial_density <= 0.0:
                raise ValueError("full-span channel needs trial_density > 0")
            _maxchord = box * (3.0**0.5)
            n_trials = max(
                2, int(round(trial_density * _maxchord * channel_width * channel_width))
            )
        else:
            channel_length = float(_clen)
            if channel_length <= 0.0:
                raise ValueError(
                    "channel sampler needs channel_length > 0 "
                    '(or the string "full" for a wall-to-wall '
                    "rod), and channel_width > 0"
                )
            chan_vol = channel_length * channel_width * channel_width
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
        span = trial_box_side**2
        if "trial_density" in smp:
            trial_density = float(smp["trial_density"])
            n_trials = int((trial_density * span) / disk)
        else:
            n_trials = int(smp.get("trials", 30))
            trial_density = round((n_trials * disk) / span, 3)

    no_neigh = int(run.get("no_neigh", 40 if two_dim else 75))
    if "upper_cutoff" in run:
        uc = float(run["upper_cutoff"])
    elif reweight:
        uc = 3.0 * radius
    elif lj:
        uc = 2.5
    else:
        uc = 3.7545 if two_dim else 2.0

    if two_dim:
        vol_mol = math.pi * radius**2
        total_vol = box**2
    else:
        vol_mol = (4.0 / 3.0) * math.pi * radius**3
        total_vol = box**3

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
    bias = "pmf" if pmf else ("psi6" if reweight else None)
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
        if two_dim:
            raise ValueError("channel sampler is 3D only")
        if not full_repacking and int(smp.get("depth", 0)) < 1:
            raise ValueError(
                "channel sampler needs depth >= 1 (depth 0 has no "
                "clique structure to sweep), or set full_repacking"
            )
        # periodic guard: for the FIXED-length rod, insertion and removal share
        # one oriented cuboid, so the whole pool lives in that body. Its space
        # diagonal must stay below the box minus the interaction range (sigma for
        # hard core, rc for LJ) so local distances equal periodic distances. The
        # full-span rod lifts this guard on purpose and instead computes true
        # minimum-image distances in the sweep, so the guard does not apply.
        if not channel_full_span:
            guard_cut = uc if lj else diameter
            diag = _m.sqrt(channel_length**2 + 2.0 * channel_width**2)
            if diag > box - guard_cut:
                raise ValueError(
                    f"channel geometry violates the periodic guard: cuboid diagonal "
                    f"{diag:.3f} must be <= box - cutoff = {box - guard_cut:.3f}"
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
    return Config(
        potential=potential,
        bias=bias,
        sampler=sampler,
        dimension=dim,
        chem_pot=mu,
        ensemble_type=ens_type,
        target_n=target_n,
        box=box,
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
        channel_full_span=channel_full_span,
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
        g3_csv=(
            aux_block.get("g3_csv") if aux_block is not None and not two_dim else None
        ),
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
        n_workers=int(run.get("n_workers", 1)),
        cell_length=float(run.get("cell_length", 30.0)),
        buffer=float(run.get("buffer", 2.0)),
        x_spacing=(float(run["x_spacing"]) if "x_spacing" in run else None),
        y_spacing=(float(run["y_spacing"]) if "y_spacing" in run else None),
        sampling_window=sampling_window,
        target_cycles=float(run.get("target_cycles", 1.0025e9)),
        burn_in=int(run.get("burn_in", 0)),
        traj_every=int(run.get("traj_every", 0)),
        checkpoint_every=int(run.get("checkpoint_every", 1)),
        initial_config=run.get("initial_config"),
        matrix_config=run.get("matrix_config"),
        sweep_label=run.get("sweep_label"),
        pore_geometry=_pore_geometry,
        pore_H=_pore_H,
        pore_R=_pore_R,
        pore_axis_len=_pore_axis_len,
    )

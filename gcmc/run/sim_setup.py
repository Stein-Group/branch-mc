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
  "potential":  {"main": "hard_core" | "lennard_jones",   # required
                 "T": float,                         # required iff main == lennard_jones (reduced T*)
                 "tail": bool,                       # default true; lennard_jones only
                 "species_file": str},               # OPTIONAL. Present = TYPED multi-species mode:
                                                     #   path to a schema-2 `units: reduced` species yaml
                                                     #   (ff/species_*.yaml). It owns sigma, epsilon,
                                                     #   lambda, p_trial, symbol, k_ij, rc, core_floor.c
                                                     #   and the auxiliary, so run.radius,
                                                     #   run.lower_cutoff, run.upper_cutoff and the
                                                     #   whole `auxiliary` block are then FORBIDDEN.
                                                     #   Absent = single-species scalar mode, unchanged.

  "auxiliary":  {"type": "pmf" | "short_range",      # OPTIONAL block; omit = no aux. Forbidden in
                                                     #   typed mode: the species file states `aux`.
                 "g3_csv": str,                      # pmf only; required iff 3D
                 "cutoff": float,                    # short_range only, required for it:
                                                     #   the proposal/selection radius, e.g. 1.5.
                                                     #   lower_cutoff < cutoff <= upper_cutoff.
                                                     #   channel + lennard_jones + 3D bulk only.
                 "eta": float,                       # default 0.69 (2D)
                 "gamma_max": float, "delta_eta": float},

  "sampler":    {"type": "metropolis" | "rosenbluth" | "branch" | "channel"
                         | "channel_cyclic" | "reweight",
                 "engine": "metropolis_dynamics" | "glauber_dynamics"
                           | "suwa_todo_dynamics",
                                                           # THREE DYNAMICS. The last two are
                                                           #   siblings: same window, same pool,
                                                           #   same exact occupancy ladder, and a
                                                           #   different kernel drawing k from it
                                                           #   (categorical vs the Suwa-Todo
                                                           #   geometric allocation). Both require
                                                           #   sampler.type = "channel".
                                                           # default metropolis_dynamics; ABSENT MEANS
                                                           #   metropolis_dynamics, which is the
                                                           #   backward-compatibility contract.
                                                           # THE DYNAMICS AXIS, not the proposal
                                                           #   axis `type`: metropolis_dynamics is
                                                           #   accept/reject, glauber_dynamics
                                                           #   draws the window's whole occupancy
                                                           #   from its exact conditional and has
                                                           #   no acceptance test at all (dN of
                                                           #   any sign and magnitude; add /
                                                           #   remove / repack stop being separate
                                                           #   moves). Same muVT ensemble, same z,
                                                           #   same U -- a different kernel.
                                                           # glauber_dynamics REQUIRES
                                                           #   type = "channel" and depth >= 1,
                                                           #   takes no full_repacking, refuses
                                                           #   both auxiliaries and
                                                           #   channel_cyclic, caps the LJ path at
                                                           #   4 trials per sigma of rod length,
                                                           #   and is serial only. It moves every
                                                           #   run id: pre-`engine` directories
                                                           #   are not resumed, by decision.
                 "kmax_margin": int,                       # default 0; glauber_dynamics ONLY, and
                                                           #   its only new tunable. kmax =
                                                           #   min(M, ceil(packing_bound) +
                                                           #   kmax_margin); M and kmax are both
                                                           #   DERIVED, so neither is a key.
                 # EVERY OTHER KEY BELONGS TO SOME SAMPLERS AND NOT OTHERS, and a
                 # sampler refuses the ones it does not read rather than ignoring
                 # them (`_SAMPLER_KEYS`). Two of them are identity fields, so an
                 # ignored key is not harmless: it gives the same chain a
                 # different run directory.
                 #
                 #   metropolis      no other key but `engine`
                 #   rosenbluth      depth, full_repacking, trials,
                 #   branch          trial_density, trial_box_side
                 #   reweight
                 #   channel         depth, full_repacking, trials,
                 #                   trial_density, channel_length,
                 #                   channel_width, kmax_margin
                 #   channel_cyclic  depth, full_repacking, trial_density,
                 #                   channel_width
                 #
                 # `engine` is a key of EVERY sampler: stating the default is
                 # always legal. WHICH VALUES a sampler admits is a different
                 # question -- only `channel` may set glauber_dynamics.
                 "depth": int,                             # default 0; channel needs >= 1; XOR full_repacking
                 "full_repacking": bool,                   # default False; repacks the whole cavity instead
                                                           #   of a fixed depth. XOR depth.
                 "trials": int | "trial_density": float,   # trials default 30; XOR
                 "trial_box_side": float,                  # default 3.0; the cube trial box
                 "channel_length": float | "cyclic",        # rod length along the sweep
                                                           #   axis, or "cyclic" for a wall-to-wall rod
                                                           #   whose ends meet through the boundary.
                                                           #   Legacy spellings "full" | "box" |
                                                           #   "full_box" | "span" are equivalent.
                 "channel_width": float},                  # square cross-section side
                 # A CANONICAL run (ensemble.type='NVT' with a species file) runs
                 # no trial cloud, so its sampler block is exactly
                 # {"type": "metropolis"} and nothing else is accepted.

  "ensemble":   {"dimension": 2 | 3,                 # required
                 "type": "muVT" | "NVT" | "semigrand" | "QA" | "confined",  # default muVT
                                                     #   NVT is CANONICAL: with a species file every
                                                     #   species count is fixed, not just the total.
                                                     #   semigrand fixes the TOTAL only and lets
                                                     #   composition move under the activity ratio.
                 "mu": float,                        # muVT: exactly one of mu or ONE target_* key.
                                                     #   TYPED mode: a MAP {species: float}, one entry per
                                                     #   species, and z_s = exp(beta mu_s)/lambda_s^dim
                 "fugacity": {str: float},           # TYPED only: z_s = beta f_s. lambda is not consulted
                 "z": {str: float},                  # TYPED only: the activity z_s directly
                                                     #   mu / fugacity / z are MUTUALLY EXCLUSIVE routes;
                                                     #   exactly one, with no species missing or unknown.
                                                     #   Typed mode takes no target_* and no sweep.
                 "target_eta" | "target_density" | "target_N",  # controller drives mu to it; not with sweep
                 "N": int | "packing_fraction": float,         # NVT/semigrand only, exactly one
                                                     #   NVT + species file (= CANONICAL): the composition
                                                     #   is the one in run.initial_config, which is then
                                                     #   REQUIRED; the move set becomes translation +
                                                     #   pair identity exchange, both of which conserve
                                                     #   every count, so two coexisting phases are the
                                                     #   equilibrium instead of a state that decays; no
                                                     #   state-point route may be given (the activity
                                                     #   ratio moves composition and here nothing can);
                                                     #   and sampling_window counts PROPOSALS rather
                                                     #   than exchange steps
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
                 "log_columns": list,                # which COLUMN GROUPS the run log carries. The set is
                 "log_drop": list,                   #   a function of the run -- a column is written when
                                                     #   the run can put a nonzero value in it -- and these
                                                     #   two narrow it further. Groups: core and species are
                                                     #   forced; state, moves, clique, clique_acc optional.
                                                     #   log_columns lists what to keep, log_drop what to
                                                     #   remove; giving both raises. Default is every group
                                                     #   the run can fill, which for a single-species muVT
                                                     #   run is exactly the historical layout.
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
Give a LIST to any sweep axis and launch with `python -m gcmc.run.launcher` (not
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
    from ..forcefield.base import Potential

import logging
import math
from dataclasses import dataclass

from ..forcefield.overlap import max_core_floor
from ..forcefield.species import (
    STATE_POINT_ROUTES,
    SpeciesSpec,
    build_species_spec,
    one_route,
)
from ..space import cell_plan
from ..space.body import cuboid_packing_bound
from . import guard_audit

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

#: THE DYNAMICS AXIS (section 3.4 of the design plan). `metropolis_dynamics`
#: reads redundant next to `sampler = "metropolis"` and is kept deliberately: it
#: names the DYNAMICS (accept/reject versus an exact conditional draw), while
#: `sampler` names the PROPOSAL (how a trial cloud is generated). Two different
#: things called "metropolis" in one params file is exactly the confusion the
#: value name exists to prevent, and every refusal below names the axis rather
#: than only the values.
#:
#: ABSENT MEANS `metropolis_dynamics`. That is the backward-compatibility contract and
#: it is stated here, in the schema and in the refusal messages -- not left
#: implied -- because every params file written before this field existed relies
#: on it.
_ENGINES = {"metropolis_dynamics", "glauber_dynamics", "suwa_todo_dynamics"}
_ENGINE_DEFAULT = "metropolis_dynamics"
#: THE TWO LADDER DYNAMICS, which are siblings and not a parent and a variant.
#: Both build the window, the pool and the exact occupancy ladder identically;
#: they differ in the kernel that draws k from it -- a categorical draw (a heat
#: bath) against the Suwa-Todo geometric allocation. Every refusal below that
#: exists BECAUSE there is a ladder therefore applies to both, and naming the
#: family once is what stops the second engine from inheriting half of them:
#: `kmax_margin`, `depth >= 1`, the auxiliary refusals, serial-only and the
#: lambda cap are all statements about the ladder, not about which kernel reads
#: it.
_LADDER_ENGINES = ("glauber_dynamics", "suwa_todo_dynamics")
# R16: `channel_cyclic` is its own sampler, not a value of `channel_length`.
# The two share a dynamic program and nothing else; the cyclic rod lifts the
# locality guard, pays a full minimum-image pair pass, and sizes its trial count
# per move from the orientation. Naming it here makes the cost visible at the
# config surface instead of hiding it in a length string.
_SAMPLERS = {
    "metropolis",
    "rosenbluth",
    "branch",
    "reweight",
    "channel",
    "channel_cyclic",
}

#: Which keys of the `sampler` block each sampler actually READS, and one line
#: saying what its trial region is so a refusal can explain itself.
#:
#: WHY THIS IS A TABLE. It used to be a union: `_check` accepted every sampler
#: key for every sampler, and each sampler then refused, one at a time and in
#: four different places, the subset someone had thought to write a message for.
#: The keys nobody had written a message for were accepted and ignored -- and
#: `channel_length` and `channel_width` are in `_IDENTITY_FIELDS`, so they were
#: not merely ignored, they reached the RUN ID. Measured, hard core, one seed,
#: 400 cycles:
#:
#:     rosenbluth, no channel keys       trajectory cec73cf7309c   id 65d0fb249188
#:     rosenbluth, channel_length = 5    trajectory cec73cf7309c   id c1ac02cead40
#:     rosenbluth, channel_length = 6    trajectory cec73cf7309c   id 362f95b2cc6a
#:     rosenbluth, channel_width  = 1.8  trajectory cec73cf7309c   id 73d66d2ff8c9
#:
#: One chain, four run directories. A sweep over a key the sampler cannot read
#: produces N directories of the same chain, and nothing in the output says so --
#: they would be averaged as independent measurements at different settings.
#: `channel_cyclic` had the same hole for `channel_length`, which is worse for
#: being the sampler whose whole point is that it has no fixed length.
#:
#: So the sampler names its keys once, here, and everything outside its own row
#: is refused by the same mechanism for every sampler. What remains as a separate
#: check is only what a table cannot express: CONFLICTS between two keys the
#: sampler does read (trials XOR trial_density, depth XOR full_repacking).
_CAVITY_KEYS = frozenset(
    {"trials", "trial_density", "trial_box_side", "depth", "full_repacking"}
)
#: `engine` is a key of EVERY sampler, because stating the default is always
#: legal and a reader should be able to write `engine: metropolis_dynamics` next to any
#: sampler and have it mean what it says. Which VALUES each sampler admits is a
#: different question, answered by the refusal table further down: only
#: `channel` may set `glauber_dynamics`. Refusing the key here instead would
#: give the reader "sampler.type='rosenbluth' does not read ['engine']", which
#: is true of the value and misleading about the key.
_ENGINE_KEY = frozenset({"engine"})
_SAMPLER_KEYS: dict[str, tuple[frozenset[str], str]] = {
    # Standard GCMC: one uniform trial insertion, no cavity, no trial cloud.
    "metropolis": (_ENGINE_KEY, "it proposes one uniform insertion and no cloud"),
    "rosenbluth": (
        _CAVITY_KEYS | _ENGINE_KEY,
        "its trial region is a cube of side trial_box_side",
    ),
    "branch": (
        _CAVITY_KEYS | _ENGINE_KEY,
        "its trial region is a cube of side trial_box_side",
    ),
    "reweight": (
        _CAVITY_KEYS | _ENGINE_KEY,
        "its trial region is a cube of side trial_box_side",
    ),
    # The rod: an oriented cuboid, so a cube side means nothing to it.
    # `kmax_margin` is the Glauber engine's one tunable and only this sampler can run
    # that engine, so only this sampler reads it.
    "channel": (
        frozenset(
            {"trials", "trial_density", "channel_length", "channel_width"}
            | {"depth", "full_repacking", "kmax_margin"}
        )
        | _ENGINE_KEY,
        "its trial region is an oriented cuboid of channel_length x channel_width",
    ),
    # R16: the cyclic rod spans the box, so its LENGTH is set per move by the
    # orientation and there is no fixed trial count either.
    "channel_cyclic": (
        frozenset({"trial_density", "channel_width", "depth", "full_repacking"})
        | _ENGINE_KEY,
        "it spans the box, so its length and trial count are set per move",
    ),
}


def assert_channel_guard(
    channel_length: float,
    channel_width: float,
    dim: int,
    span: float,
    cutoff: float,
    *,
    span_name: str = "box",
) -> None:
    """Refuse a channel body that does not fit, with a margin, inside `span`.

    The guarded sweep works on LOCAL cuboid coordinates and does no minimum
    image, so a body whose diagonal reaches within `cutoff` of the torus size
    can read a wrapped pair as a distant one. `span` is the size of that torus
    and `cutoff` the largest distance the sweep will read.

    ONE FUNCTION, TWO TORI. Serially the torus is the box. On the parallel path
    the body must lie inside the WORKER CELL, so the binding dimension is the
    cell and a config that passes the serial check can violate it per cell.
    G10 of the design plan wants that extension written while it is
    cheap rather than after the G9 references are pinned, so the span is a
    parameter and `span_name` is what the message calls it. The channel sampler
    is refused under `run.kind="parallel"` today, so `from_params` only ever
    passes the box; `tests/unit/test_glauber_config.py` calls this with a cell
    span directly, which is what keeps the branch honest rather than aspirational.

    THE DIAGONAL IS DIMENSION-CORRECT: sqrt(L^2 + 2 w^2) for a 3D cuboid,
    sqrt(L^2 + w^2) for a 2D rectangle, because the cross-section carries
    (dim - 1) widths. `channel_move.window_guard_ok` uses 2 w^2 unconditionally,
    which is STRICTER in 2D -- it can refuse a legal 2D body but never accept an
    illegal one -- and is left alone: relaxing a guard is the one direction that
    needs its own gate.
    """
    diag = math.sqrt(float(channel_length) ** 2 + (dim - 1) * float(channel_width) ** 2)
    limit = float(span) - float(cutoff)
    if diag > limit:
        raise ValueError(
            f"channel geometry violates the periodic guard: cuboid diagonal "
            f"{diag:.3f} must be <= {span_name} {float(span):.3f} - "
            f"{float(cutoff):.3f} = {limit:.3f} (the margin is the largest "
            f"distance the sweep reads: auxiliary.cutoff when a short-range "
            f"auxiliary is configured, the largest active pair floor on the "
            f"hard-core path under engine=glauber_dynamics, else "
            f"run.upper_cutoff)"
        )


def _check_sampler(smp: dict[str, Any], sampler: str) -> None:
    """Refuse any `sampler` key the chosen sampler does not read.

    Separate from `_check` because the allowed set depends on `sampler.type`,
    and because the message has to say WHY -- "unknown key" is true and useless
    when the key is a real key of a different sampler.
    """
    reads, why = _SAMPLER_KEYS[sampler]
    extra = sorted(set(smp) - reads - {"type"})
    if extra:
        raise ValueError(
            f"sampler.type={sampler!r} does not read {extra}: {why}. "
            f"It reads {sorted(reads) or 'no other key'}. "
            "Those keys belong to a different sampler; delete them rather than "
            "leaving them, because this run would not change if you did -- and "
            "channel_length/channel_width are identity fields, so leaving one "
            "gives the same chain a different run directory."
        )


#: Spellings of `channel_length` that used to select the cyclic rod. They are
#: refused now rather than aliased: a params file saying "full" and meaning
#: "topologically different, much costlier move" is exactly the surface R16
#: removes. Kept as data so the error can name what was typed.
_LEGACY_CYCLIC_SPELLINGS = ("cyclic", "full", "box", "full_box", "span")


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
        if sampler in ("branch", "channel", "channel_cyclic"):
            return "ljbranch"
        return "lj"
    if bias == "pmf":
        return "aux"
    return "hard_sphere"


def _short_range_unsupported(
    *, lj: bool, sampler: str, qa: bool, pore: bool
) -> str | None:
    """Why the short-range auxiliary cannot be used here, or None if it can.

    The support matrix for the dual-cutoff auxiliary, stated once. It returns a
    SENTENCE rather than raising, because its two callers do different things
    with the same verdict: a per-run `auxiliary` block is a request and an
    unsupported one raises this text, while a typed species file is a force field
    and instead logs it and runs with the auxiliary off. Both readings are honest
    only if they agree about what is supported, which one function guarantees and
    two copies would not.

    Every entry is a combination that is meaningless or untested, not one that is
    known wrong, and each is a separate cheap relaxation later. The reason is
    spelt out in each message because these are the guards a user actually hits.
    """
    if not lj:
        return (
            "auxiliary.type='short_range' needs potential.main='lennard_jones'. "
            "It shortens an interaction range, and a hard core has no range to "
            "shorten -- its exclusion floor IS the interaction, so there is "
            "nothing to propose with and nothing to correct."
        )
    if sampler not in ("channel", "channel_cyclic"):
        return (
            f"auxiliary.type='short_range' is supported for "
            f"sampler.type='channel' only (got {sampler!r}). The saving is in "
            f"the transfer-matrix sweep, whose DP window width IS the "
            f"interaction radius. 'branch' would gain almost nothing: its clique "
            f"adjacency is built at the overlap floor, not at the cutoff, so the "
            f"enumeration cost does not shrink. 'metropolis' has a single trial "
            f"and no selection, so there is no proposal bias to divide out. "
            f"'rosenbluth' would be correct and is a clean later extension, but "
            f"its per-trial weight is a vector operation, not a combinatorial one."
        )
    # 2D IS supported. The sweep is dimension-agnostic (it sorts on axis 0 and
    # takes full Euclidean distances), the trial body is an oriented rectangle
    # instead of a cuboid, and the periodic guard already carries (dim-1) widths.
    # tests/unit/test_channel_2d.py pins the 2D sweeps against brute-force clique
    # enumeration and the 2D engine against the exact P(N=0)/P(N=1) ratio, for
    # both the fixed and the cyclic rod.
    #
    # WHAT 2D DOES NOT INHERIT, and it is worth knowing before trusting a number:
    # that engine-level 2D check is HARD-CORE only. 2D + LJ + channel runs and
    # gives a plausible density, but it has never been validated against a
    # reference at the engine level, in either direction. The auxiliary is exact
    # relative to whatever the 2D LJ channel already samples -- its own gate is
    # aux-on == aux-off -- so it cannot make that worse, but it cannot vouch for
    # it either. 2D LJ metropolis does run, so a reference for closing that gap
    # exists.
    if qa:
        return (
            "auxiliary.type='short_range' does not support the QA frozen matrix: "
            "frozen obstacles must be dropped from the LJ sum in the member "
            "weights, the full-potential energy AND the new correction term, and "
            "QA + channel is not a supported combination to begin with."
        )
    if pore:
        return (
            "auxiliary.type='short_range' does not support a confining pore yet. "
            "The algebra carries over (the tail is already disabled in a pore, so "
            "the correction adds cleanly), but pore + channel is untested."
        )
    return None


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
    n_trial_molecules: int
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
    # floor() is not exact in float64 and silently loses cells.
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
    log_columns: tuple[str, ...] | None
    log_drop: tuple[str, ...] | None
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
    # Per-axis orthorhombic box. None means cubic: every axis is `box`,
    # and the scalar path runs exactly as before, so cubic fingerprints, RNG
    # streams and characterisation trajectories are unchanged. When set it is a
    # length-`dimension` tuple (Lx, Ly, Lz); `box` then holds min(sides) for the
    # legacy scalar guards, and volume/wrap/draws route through `box_dims`.
    box_sides: tuple[float, ...] | None = None
    # oriented rod, any orientation, exact PBC
    # (the periodic guard is lifted; the sweep
    # uses minimum-image distances + retention)
    # Everything species-indexed for a typed multi-species run, or None for
    # the single-species scalar path, which is then bit-identical to before. When
    # set, it -- not the scalar radius / lower_cutoff / upper_cutoff / volume_of_
    # molecule fields -- is the authority on sigma, epsilon, the pair floors, the
    # truncation, the auxiliary and the per-species state point.
    species: SpeciesSpec | None = None
    # ---- the DYNAMICS axis (design plan, section 3.4) -----------
    #
    # `metropolis_dynamics` (the default, and what an ABSENT key means -- that is the
    # backward-compatibility contract) is accept/reject dynamics: everything
    # shipped before this field existed. `glauber_dynamics` computes the exact
    # conditional distribution of a window's guest content and draws from it, so
    # its exchange path has no acceptance test at all.
    #
    # SAME ENSEMBLE, DIFFERENT KERNEL. muVT, the same z, the same U -- which is
    # why the existing physics gates apply to it verbatim as cross-engine
    # agreement rather than needing new reference numbers.
    engine: str = "metropolis_dynamics"
    # The Glauber engine's ONE new tunable: how far above the body's geometric
    # packing bound the occupancy ladder is truncated. Everything else derives.
    kmax_margin: int = 0
    # DERIVED, per section 1.3. `window_M` is the pool size (in-window guests
    # plus the fresh cloud, always exactly this many points) and `window_kmax`
    # the ladder truncation. Both are zero on a `metropolis_dynamics` run, which reads
    # neither. See `config_fields.DERIVED` for why they are derived rather than
    # separately hashed, and `tests/unit/test_glauber_config.py` for the
    # derivation as an assertion.
    window_M: int = 0
    window_kmax: int = 0

    @property
    def n_species(self) -> int:
        """1 for a single-species run, so callers need no None check."""
        return 1 if self.species is None else self.species.n

    @property
    def species_identity(self) -> tuple[str, ...] | None:
        """The typed half of the run fingerprint, or None.

        Carries the species-file hash (so k_ij, sigma, epsilon, the core floor,
        rc and the auxiliary multiplier are all covered by one digest), the route
        taken, and per species the thermal wavelength and the activity. Two runs
        whose lambdas differ are different experiments and must never share a
        directory, even when the recorded mu is the same number.
        """
        s = self.species
        if s is None:
            return None
        return (
            f"file={s.sha256}",
            f"route={s.route}",
            *(
                f"{n}:lam={format(float(lam), '.12g')}:z={format(float(z), '.12g')}"
                for n, lam, z in zip(s.names, s.lam, s.z)
            ),
        )

    def packing_fraction(self, counts: object) -> float:
        """eta from a particle count: `sum_s n_s v_s / V` when typed, and
        `n v / V` with the single molecular volume otherwise. One entry point, so
        a caller cannot pick the wrong convention."""
        if self.species is not None:
            return self.species.packing_fraction(counts, self.total_volume)
        return float(counts) * self.volume_of_molecule / self.total_volume  # type: ignore[arg-type]

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
        from ..forcefield import make_potential

        uc, d = self.upper_cutoff, self.diameter
        # THE TYPED ENERGY. Which samplers can integrate a species file, and why
        # the answer is per-SAMPLER rather than per potential_type.
        #
        # `potential_type` collapses `branch`, `channel` and `channel_cyclic` into
        # one tag, `ljbranch`, because they share a proposal shape. They do NOT
        # share a clique kernel, and that is what decides this:
        #
        #   rosenbluth      supported. Per-pair weights and per-pair overlap
        #                   floors, gated by the reduction and discrimination
        #                   tests in tests/engine/test_typed_energy.py.
        #   channel         supported. Both kernels take the mixture overlap
        #                   policy as a keyword: conflict pairs at c*sigma_ij and,
        #                   for the LJ kernel, in-window pairs weighted by the
        #                   per-pair 12-6 energy. That policy was written for the
        #                   MOF path and unit-tested there; what was missing was
        #                   `make_sweep` reaching for it.
        #   branch          supported. `count_and_sample` and
        #                   `ljbranch_weight_sum` now build adjacency at the
        #                   per-pair floor and evaluate the intra-clique factor
        #                   from the pair table. The clique ENUMERATION was
        #                   already type-blind, which is why this was a small
        #                   change to the graph and the pair energy rather than to
        #                   the DP.
        #   channel_cyclic  supported. The per-pair floor and the pair table are
        #                   OPTIONS on the cyclic sweeps rather than a seventh and
        #                   eighth class -- cyclic and multi-species are
        #                   independent properties of the conflict graph, and the
        #                   class lattice could not compose them. That is the graph
        #                   half of R7.
        #
        # An energetically DEGENERATE species file is allowed through every
        # sampler: one sigma, one epsilon and one core floor throughout means the
        # single-sigma potential represents that mixture exactly, the components
        # differing only in activity, which the engine does carry. Not a loophole
        # -- it is the true condition, and it is what the M21 arbiter runs on.
        if self.species is not None and self.potential_type in ("lj", "ljbranch"):

            def _all_one(table: Any) -> bool:
                flat = [float(v) for row in table for v in row]
                return all(abs(v - flat[0]) <= 1e-12 for v in flat)

            _degenerate = (
                _all_one(self.species.sigma)
                and _all_one(self.species.eps)
                and _all_one(self.species.core_floor)
            )
            # The table potential is returned for EVERY typed run, degenerate or
            # not. On a degenerate file it is numerically the scalar potential --
            # `table_pair_energy` at one sigma and one epsilon reduces to
            # `lj_pair_energy` bit for bit, which `test_pair_table.py` pins -- so
            # there is nothing to gain by branching, and one real thing to lose:
            # the degenerate file is the ONLY case in which the table path can be
            # checked against the untyped chain, and routing it around the table
            # would make that reduction gate untestable.
            from ..forcefield import make_potential as _mp

            return _mp(
                "lj_table",
                beta=self.beta,
                table=self.species.table,
                trial_type=0,
                upper_cutoff=uc,
            )
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
                    n_trials=self.n_trial_molecules,
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
                    n_trials=self.n_trial_molecules,
                    g3_csv=self.g3_csv,
                )
            case "reweight":
                return None
            case other:
                raise ValueError(f"unknown potential type {other!r}")


def _species_names_of(species_file: object) -> tuple[str, ...]:
    """The guest names a species file declares, without building the full spec.

    Used to synthesise the unit activities a canonical run needs in order to
    build a spec whose activities nothing reads, and to count species when
    deciding whether NVT is canonical or just NVT.

    A LIST is a sweep axis (one run per force-field variant, which is how the k_ij
    series is written) and the first entry answers for all of them: the variants
    differ in their pair parameters, never in which guests exist -- a sweep whose
    members declared different species would not be one sweep, and its run ids
    would not be comparable. Taking [0] rather than raising is what lets the
    ensemble be decided before the launcher expands the axis.
    """
    from ..forcefield.species_file import load_forcefield

    if isinstance(species_file, (list, tuple)):
        if not species_file:
            raise ValueError("potential.species_file is an empty list")
        species_file = species_file[0]
    return tuple(load_forcefield(str(species_file)).guests)


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
    _check(pot, {"main", "T", "tail", "species_file"}, {"main"})
    main = pot["main"]
    if main not in _MAIN:
        raise ValueError(f"unknown main potential {main!r}; allowed {sorted(_MAIN)}")
    lj = main == "lennard_jones"
    if lj and "T" not in pot:
        raise ValueError("Lennard-Jones requires potential.T")
    if not lj and "T" in pot:
        raise ValueError("only Lennard-Jones takes potential.T")
    # One switch selects the whole typed (multi-species) path. Present means
    # sigma, epsilon, lambda, p_trial, the core floor, rc and the auxiliary all
    # come from that file and may not be restated in `run`; absent means the
    # existing single-species scalar mode, unchanged down to the last bit.
    species_file = pot.get("species_file")
    typed = species_file is not None
    if typed and not lj:
        raise ValueError(
            f"potential.species_file describes a Lennard-Jones mixture "
            f"(interaction.form: lennard-jones-12-6), so potential.main must be "
            f"'lennard_jones', not {main!r}"
        )

    # ---- sampler ----
    smp = p["sampler"]
    _check(
        smp,
        {
            "type",
            "engine",
            "kmax_margin",
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
    # THE DYNAMICS, read before anything else about the sampler, because several
    # refusals below are conditional on it. Absent means `metropolis_dynamics`
    # (`_ENGINES`), which is the backward-compatibility contract.
    engine = str(smp.get("engine", _ENGINE_DEFAULT))
    if engine not in _ENGINES:
        raise ValueError(
            f"unknown sampler.engine {engine!r}; allowed {sorted(_ENGINES)}. "
            "`engine` selects the DYNAMICS (Metropolis accept/reject vs the "
            "Glauber engine's exact conditional draw); `sampler.type` selects "
            "how a trial cloud is proposed. They are different axes."
        )
    # THE FAMILY, not the name. Every check below that reads this flag is a
    # statement about the occupancy ladder, which both ladder dynamics build the
    # same way; the two places that care WHICH kernel draws from it name the
    # engine directly instead.
    ladder = engine in _LADDER_ENGINES
    # ---- the Glauber engine's refusal table (section 3.5) ---------------
    #
    # ERROR TEXT NAMES THE AXIS, NOT JUST THE VALUES. `engine` selects the
    # dynamics; `sampler` selects how a trial cloud is proposed. Both have a
    # value called "metropolis"-ish, so a bare "incompatible" message is
    # unreadable -- and an unreadable refusal is the failure mode here, which is
    # why `tests/unit/test_glauber_config.py` asserts on the MESSAGE and not only
    # on the raise.
    if ladder and sampler != "channel":
        _why = {
            "metropolis": (
                "sampler=metropolis proposes a single trial with no cloud, "
                "so the Glauber engine has nothing to sweep over"
            ),
            "rosenbluth": (
                "sampler=rosenbluth proposes a cube cavity and selects "
                "sequentially; there is no transfer matrix over it, so the "
                "local conditional cannot be computed exactly"
            ),
            "branch": (
                "sampler=branch enumerates cliques in a cube cavity, which "
                "has no 1D chain structure and no transfer matrix"
            ),
            "reweight": (
                "sampler=reweight is the 2D hexatic psi6 method, an "
                "estimator-layer scheme with no local conditional"
            ),
            "channel_cyclic": (
                "the cyclic rod spans the box wall to wall, so its exterior "
                "is not separable from its interior and the frozen-exterior "
                "argument the whole conditional is derived from has no "
                "meaning. This is a permanent geometric objection, not a "
                "performance one"
            ),
        }[sampler]
        raise ValueError(
            f"engine={engine} requires sampler=channel, got "
            f"sampler={sampler}.\n"
            "These are different axes: `engine` selects the dynamics "
            "(Metropolis accept/reject vs Glauber conditional draw), "
            "`sampler` selects how the trial cloud is proposed. "
            f"{_why}."
        )

    kmax_margin = int(smp.get("kmax_margin", 0))
    if kmax_margin < 0:
        raise ValueError(
            f"sampler.kmax_margin must be >= 0, got {kmax_margin}: it is how far "
            "ABOVE the body's geometric packing bound the ladder is truncated, "
            "and a negative margin would truncate below the bound -- which makes "
            "the kernel the identity on states the chain can actually reach and "
            "silently biases <N> low (design plan, section 3.7)."
        )
    if "kmax_margin" in smp and not ladder:
        raise ValueError(
            "sampler.kmax_margin only means something under a LADDER dynamics "
            '(sampler.engine="glauber_dynamics" or "suwa_todo_dynamics"): it '
            "sets where the occupancy ladder is truncated, and both build that "
            "ladder the same way. Metropolis dynamics has no ladder. Remove it, "
            "or set the engine."
        )
    # Only this sampler's keys, from here on. Every check below is one this
    # sampler can actually fail.
    _check_sampler(smp, sampler)
    metropolis = sampler == "metropolis"
    # Both rod kinds are channel samplers: they take the same cavity algebra and
    # the same DP. Only the mask policy and the guard differ.
    branch = sampler in ("branch", "channel", "channel_cyclic")
    channel = sampler in ("channel", "channel_cyclic")
    reweight = sampler == "reweight"
    # full_repacking (default False) is the XOR opposite of the old use_depth:
    # instead of a fixed depth, an exchange repacks the whole cavity, removing
    # every sphere it finds and inserting one more (add), one fewer (remove), or
    # the same count (repack). depth is then meaningless and must not be given.
    full_repacking = _as_bool(
        smp.get("full_repacking", False), "sampler.full_repacking"
    )
    if full_repacking and "depth" in smp:
        raise ValueError(
            "depth and full_repacking are mutually exclusive: full_repacking "
            "repacks the whole cavity and ignores depth, so give one or the "
            "other, not both"
        )

    # ---- ensemble ----
    ens = p["ensemble"]
    # Ahead of `_check`, so a params file written for the previous stage gets the
    # replacement named instead of "unknown keys [...]". A retired key is the one
    # case where the generic message is the wrong one: the reader already knows
    # the key, and what they need is what it became.
    if "fixed_composition" in ens:
        raise ValueError(
            "ensemble.fixed_composition is gone; the ensemble name carries the "
            "meaning now. Drop the key: ensemble.type='NVT' with a species file "
            "IS the canonical run it used to switch on. If you wanted the free "
            "composition that typed NVT used to give by default, that ensemble "
            "is now ensemble.type='semigrand'."
        )
    _check(
        ens,
        {
            "mu",
            "fugacity",
            "z",
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
    # `fugacity` and `z` are the two non-mu state-point routes of M10 and exist
    # only in typed mode; `mu` doubles as the scalar single-species route and the
    # typed map route, told apart by whether a species file is present.
    _typed_routes = [k for k in ("fugacity", "z") if k in ens]
    if _typed_routes and not typed:
        raise ValueError(
            f"ensemble.{_typed_routes[0]} is a typed multi-species state-point "
            "route and needs potential.species_file; a single-species run states "
            "a scalar ensemble.mu"
        )
    if not typed and isinstance(ens.get("mu"), dict):
        raise ValueError(
            "ensemble.mu is a map (one entry per species) but no "
            "potential.species_file was given, so there are no species to index. "
            "State a scalar mu, or add the species file."
        )
    dim = int(ens["dimension"])
    target_count = None  # set below iff a muVT target is requested
    _target_spec = None  # (kind, value) for a muVT target, finalized post-volume
    _pore_geometry = _pore_H = _pore_R = _pore_axis_len = None
    if dim not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    two_dim = dim == 2
    ens_type = ens.get("type", "muVT")
    if ens_type not in ("muVT", "NVT", "semigrand", "QA", "confined"):
        raise ValueError(
            "ensemble.type must be 'muVT', 'NVT', 'semigrand', 'QA', or 'confined'"
        )
    # `nvt` is the FIXED-N question, which both number-conserving ensembles answer
    # yes to: they take N or packing_fraction, they refuse a target, and the
    # engine runs the same number-conserving loop for both. What they disagree
    # about is whether COMPOSITION is fixed too, and that is `canonical` below.
    nvt = ens_type in ("NVT", "semigrand")
    semigrand = ens_type == "semigrand"
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

    # ---- CANONICAL vs SEMI-GRAND, and why they are two ensemble.type values --
    #
    # This was one type ("NVT") and a boolean (`fixed_composition`). It is two
    # names now, because one name for two ensembles is what actually went wrong:
    # typed NVT defaulted to SEMI-GRAND -- the repack draws each inserted
    # particle's species, so the total is fixed and composition is free -- and
    # three cluster campaigns were launched, and dissolved into a single phase,
    # under a params file that said `"type": "NVT"` and read like canonical. A
    # flag you have to remember to set is a worse guard than a name you have to
    # write, so:
    #
    #   type = "NVT"        every species count is fixed (CANONICAL). Moves are
    #                       translation + pair identity exchange, both of which
    #                       conserve every count, so a box held at x = 0.5
    #                       between two coexisting compositions cannot become
    #                       one phase and two slabs are the EQUILIBRIUM.
    #   type = "semigrand"  the total is fixed, composition is free under the
    #                       activity ratio. Right for a composition isotherm
    #                       x(Delta-mu); it cannot hold a two-phase box, because
    #                       a single phase costs no interface.
    #
    # `canonical` needs a species file: with nothing typed there is no
    # composition to hold, and NVT means what it always meant.
    # TWO conditions, not one, and the difference is a one-species species file.
    # `nvt_typed` is "NVT with a species file", which is what forbids a state
    # point and what makes the synthesised unit activities necessary further
    # down. `canonical` is "NVT with a COMPOSITION", which is what the engine
    # keys its move set on -- with one species there is nothing to exchange, the
    # pair move is a null proposal every time, and the repack is the better
    # sampler for the same distribution. Deriving both from the same count here
    # is what stops the engine, the log and this validator drifting apart on the
    # edge case; they answered it three different ways in the first draft.
    nvt_typed = ens_type == "NVT" and typed
    canonical = nvt_typed and len(_species_names_of(species_file)) > 1
    if semigrand and not typed:
        raise ValueError(
            "ensemble.type='semigrand' samples COMPOSITION at fixed total N, so "
            "it needs a species file (potential.species_file) -- with one "
            "species there is no composition and the ensemble is NVT."
        )

    if nvt:
        # NVT: fixed particle number, no chemical potential. Target given as an
        # exact integer N, or as a packing fraction converted to the nearest
        # integer N (held exactly thereafter). mu is meaningless here.
        #
        # ---- the two typed number-conserving ensembles ------------------------
        #
        # Typed + fixed N was for a long time neither supported nor refused.
        # `build_species_spec` calls `one_route`, which demands
        # ensemble.mu/fugacity/z for any typed run -- including this branch,
        # which refused `mu` outright. So a typed NVT params file died with "a
        # typed run needs exactly one state-point route: ensemble.mu", doing what
        # it said produced "NVT takes no 'mu'", and the two messages sent the
        # reader back and forth with no reachable configuration between them.
        #
        # WHAT THE ENSEMBLE ACTUALLY IS, because the first fix here got it wrong
        # and refused the combination on the grounds that "a mixture's canonical
        # state point is one count per species". That is true of the canonical
        # ensemble and this is not it. NVT in this engine forces `choice =
        # "repack"`: the only move removes k particles and inserts k, and a typed
        # insertion DRAWS its species and carries `z_s` in the one-body weight.
        # So total N is conserved exactly while composition relaxes under the
        # activity ratio -- which is the SEMI-GRAND ensemble, the standard
        # instrument for exactly this problem. Nothing needed inventing; the
        # combination only needed to stop being refused by two validators at
        # once.
        #
        # `mu` is therefore MEANINGFUL for a typed NVT run and meaningless for a
        # single-species one, where it is the same number for every particle and
        # cancels in the repack. Hence the split below rather than one rule.
        if typed:
            if "sweep" in ens:
                raise ValueError(
                    "ensemble.sweep is the scalar-mu sweep and a typed run has "
                    "one mu per species; sweep a typed state point by giving a "
                    "LIST OF MAPS to ensemble.mu/.fugacity/.z and launching with "
                    "gcmc.run.launcher"
                )
            if nvt_typed:
                # CANONICAL: N_A and N_B are both fixed, by the start file, and
                # nothing in the chain can change them. An activity ratio would
                # be a knob that appears to work and does nothing -- the failure
                # this codebase refuses -- so stating one is an error rather than
                # something quietly ignored.
                _given = [k for k in STATE_POINT_ROUTES if k in ens]
                if _given:
                    raise ValueError(
                        f"ensemble.type='NVT' with a species file holds every "
                        f"species count constant, so ensemble.{_given[0]} would "
                        "do nothing: the activity ratio is what MOVES "
                        "composition, and here it cannot. The composition is the "
                        "one in run.initial_config. Drop the state point, or set "
                        "ensemble.type='semigrand' to sample composition at "
                        "fixed total N."
                    )
                # WHERE THE COMPOSITION COMES FROM has to be somewhere, and a
                # generated start would make it whatever the placement happened
                # to draw -- an input nobody stated and nobody can reproduce
                # from the params file. Demanding the file is M15: the earliest
                # layer that can see the problem is this one.
                # THE SAMPLER IS NOT A CHOICE HERE, so it must not read like
                # one. A canonical run proposes translations and pair identity
                # exchanges and runs no trial cloud at all -- measured, not
                # assumed: channel at two geometries, rosenbluth, branch at
                # depth 2 and channel_cyclic all produce the byte-identical
                # final state (digest 1bda67b8fe30a574) from the same seed, and
                # FIVE DIFFERENT RUN IDS, because `sampler` is an identity
                # field. One physical chain in five directories is not a
                # cosmetic problem: a sweep over trial_density would look like
                # five settings and be one.
                #
                # So the sampler is pinned to the one name that is true of the
                # move set: no trial cloud. `_check_sampler` has already refused
                # every other key of the metropolis row, which is empty, so
                # `{"type": "metropolis"}` is the whole legal block.
                if sampler != "metropolis":
                    raise ValueError(
                        "ensemble.type='NVT' with a species file proposes "
                        "translations and pair identity exchanges, and runs no "
                        "trial cloud, so sampler.type must be 'metropolis' (got "
                        f"{sampler!r}). Any other sampler is configuration that "
                        "cannot change this run -- measured: channel, "
                        "rosenbluth, branch and channel_cyclic give the "
                        "identical chain here, and four different run ids. Use "
                        "ensemble.type='semigrand' if you want the repack, "
                        "which is the move that reads a sampler."
                    )
                if not p.get("run", {}).get("initial_config"):
                    raise ValueError(
                        "ensemble.type='NVT' with a species file fixes every "
                        "species count, so the composition IS an input and has "
                        "to be stated: give run.initial_config. Build one with "
                        "`python -m scripts.demixing start`. (Without it the "
                        "composition would be whatever the initial placement "
                        "drew, which is not a number this params file records.)"
                    )
            else:
                # SEMI-GRAND. Exactly one route, named error if not -- the same
                # call the muVT branch makes, so the two ensembles state a typed
                # state point the same way. Only the RATIO of activities does any
                # work here, since N is fixed; the common factor cancels in the
                # repack exactly as the single-species prefactor does.
                one_route(ens)
        elif "mu" in ens or "sweep" in ens:
            raise ValueError(
                f"ensemble.type={ens_type!r} takes no 'mu'/'sweep' (N is fixed, "
                "mu undefined)"
            )
        has_n, has_pf = "N" in ens, "packing_fraction" in ens
        if has_n == has_pf:
            raise ValueError(
                f"ensemble.type={ens_type!r} needs exactly one of 'N' or "
                "'packing_fraction'"
            )
        _swept = next(
            (k for k in ("N", "packing_fraction") if isinstance(ens.get(k), list)), None
        )
        if _swept is not None:
            raise ValueError(
                f"ensemble.{_swept} is a list (a sweep), but a single run got it "
                f"unexpanded. Launch sweeps with `python -m gcmc.run.launcher` (not "
                f"`gcmc run`); if you already did, the launcher on this machine "
                f"is an older version that does not expand '{_swept}' -- update "
                f"gcmc/run/launcher.py."
            )
        mu: float | None = 0.0  # unused under NVT; prefactor cancels in repacking
    else:
        # muVT: either a fixed `mu` (or a sweep), OR a target the controller
        # drives mu toward: exactly one of target_eta / target_density / target_N.
        if "N" in ens or "packing_fraction" in ens:
            raise ValueError(
                "'N'/'packing_fraction' belong to the fixed-N ensembles; set "
                "ensemble.type='NVT' (fixed counts) or 'semigrand' (fixed total)"
            )
        target_keys = [
            k for k in ("target_eta", "target_density", "target_N") if k in ens
        ]
        if len(target_keys) > 1:
            raise ValueError(f"give only one target, got {target_keys}")
        targeting = len(target_keys) == 1
        has_mu, has_sweep = "mu" in ens, "sweep" in ens

        if typed:
            # The typed state point is n numbers, one per species, and it is
            # validated by gcmc.forcefield.species once beta is known (search: "typed
            # species spec"). What is settled here is what typed mode does NOT
            # take.
            if targeting:
                raise ValueError(
                    f"ensemble.{target_keys[0]} drives one scalar mu, but a typed "
                    "run has one mu per species and composition is an output of "
                    "the chemical potentials, never an input (M2). Give "
                    "ensemble.mu / .fugacity / .z as a map instead."
                )
            if has_sweep:
                raise ValueError(
                    "ensemble.sweep is the scalar-mu sweep. Sweep a typed state "
                    "point by giving a LIST OF MAPS to ensemble.mu/.fugacity/.z "
                    "and launching with gcmc.run.launcher, which expands it."
                )
            one_route(ens)  # exactly one route, named error if not
            mu = None  # finalized from the species spec below
        elif targeting:
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
                    "unexpanded. Launch sweeps with `python -m gcmc.run.launcher`; if "
                    "you did, gcmc/run/launcher.py on this machine is outdated."
                )
            if has_sweep:
                raise ValueError(
                    "this config has a 'sweep'; launch it with gcmc.run.launcher, "
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
    if typed and aux_block is not None:
        raise ValueError(
            "the `auxiliary` block is the single-species auxiliary. In typed mode "
            "the auxiliary is stated in the species file as a per-pair multiplier "
            "(`aux: {multiplier: a}`, giving a*sigma_ij per M12), or `aux: none`, "
            "or `aux: {control: true}` for the exact no-op arm. Two places to "
            "state one cutoff is the bug class the config exists to prevent. The "
            "pmf bias is separately out of scope: its g3 table is three-body and "
            "breaks the pair factorisation the transfer-matrix sweep depends on."
        )
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
            "log_columns",
            "log_drop",
        },
    )

    # In typed mode sigma comes from the species file, the exclusion floor is
    # c*sigma_ij and the truncation is interaction.rc, all per pair. Letting the
    # run restate any of them as one scalar is how a config ends up with two
    # competing definitions of the core floor, which cost a session before; the
    # config is written so it is impossible to state one.
    if typed:
        _owned = {
            "radius": "sigma comes from the species file, per species",
            "lower_cutoff": "the floor is core_floor.c * sigma_ij, per pair",
            "upper_cutoff": "the truncation is interaction.rc in the species file",
        }
        _stated = sorted(set(_owned) & set(run))
        if _stated:
            raise ValueError(
                "run key(s) "
                + ", ".join(f"{k!r} ({_owned[k]})" for k in _stated)
                + " are owned by the species file in typed mode; remove them "
                "from `run` or drop potential.species_file"
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
    # `pmf` can only be true when an auxiliary block was supplied, but that
    # invariant lives in how `pmf` is derived rather than in the types, so the
    # checker cannot see it. Binding it to a non-optional name states the
    # invariant once and lets every guard below read plainly.
    aux: dict[str, Any] = aux_block if aux_block is not None else {}
    if pmf and lj and "g3_csv" not in aux:
        raise ValueError("Lennard-Jones + pmf requires auxiliary.g3_csv")
    if pmf and branch:
        raise ValueError("branch takes no auxiliary potential")
    if pmf and reweight:
        raise ValueError(
            "reweight uses psi6, not a pmf auxiliary; remove the auxiliary block"
        )
    # M15: fail at the earliest layer that can see the problem. This guard lived
    # in `metropolis_dynamics._advance_single_particle`, which meant the validator
    # ACCEPTED the combination and the engine refused it -- after the run directory was
    # created, the log opened and the first window begun. The feature matrix
    # found it on its first run, as a point that passed validation and then
    # raised.
    #
    # Only the HARD-CORE pmf auxiliary is refused. LJ + pmf (`ljaux`) is
    # supported under metropolis and is not an approximation: with a single
    # trial the selection bias cancels identically, so metropolis + pmf IS
    # metropolis on the plain LJ potential, and the engine unwraps it to exactly
    # that.
    # M15 again, and a worse case than the one below it. `_advance_single_particle`
    # is not typed-aware in ANY respect, and measured on the tree three separate
    # things go wrong at once, all of them SILENTLY:
    #
    #   1. `moves.draw_move` returns "swap" for n_species > 1, and the loop
    #      dispatched on `if move / if add / else`, so every swap proposal was
    #      executed as a DELETION. Measured over one window: 15 removes and 14
    #      swaps drawn, remove_step recorded 29.
    #   2. The pair energy is `lj_pair_energy`, which is sigma = epsilon = 1.
    #      The species file's per-pair sigma, epsilon and k_ij are never read, so
    #      the mixture is sampled as if it were one species.
    #   3. `state.species` is never updated alongside `state.positions`, so the
    #      two desync on the first accepted insertion or deletion -- 43 positions
    #      against 40 labels after a single window.
    #
    # None of the three raises. The run completes, the counters look ordinary,
    # and the isotherm is wrong. Five feature-matrix points exercised this and
    # reported PASS.
    #
    # So the combination is refused until the typed metropolis path exists, and
    # refused HERE rather than patched in the engine: half-implementing the swap
    # on top of a species-blind energy would fix the visible defect and leave the
    # two that produce plausible numbers.
    #
    # EXCEPT under the canonical ensemble, and the exception is exact rather than
    # convenient. Everything the refusal names is a property of
    # `_advance_single_particle`, and a canonical run never enters it: `advance` takes
    # its NVT branch, draws from `moves.draw_canonical_move`, and proposes a
    # translation or a species-aware pair identity exchange. `metropolis` there
    # is a name for "no trial cloud", which is what the canonical move set is.
    if typed and metropolis and not canonical:
        raise ValueError(
            "the metropolis sampler is not multi-species: it reads no per-pair "
            "sigma/epsilon/k_ij, proposes no identity swap, and does not carry "
            "the species column, so a typed run through it would silently "
            "sample a single-species system. Use sampler.type='rosenbluth' or "
            "'branch' for a species file, or drop potential.species_file for a "
            "single-species metropolis run. (ensemble.type='NVT' with a species "
            "file is the one exception: its move set is translation plus pair "
            "exchange and never reaches the metropolis loop.)"
        )
    # ---- typed + PARALLEL is refused until P7 wires it -------------------
    #
    # Same class of defect as typed + metropolis above, found the same way (by
    # running it rather than reading it), and it reported PASS in the feature
    # matrix for the same reason: the numbers are the numbers of a perfectly
    # valid run, just not the run that was requested.
    #
    # `gcmc/engine/cell_parallel.py` contains no species draw, no `p_trial`, no identity
    # swap and no `species_draw_correction` -- grep it. The cell kernel is
    # single-species, so every insertion is type 0. Measured on a three-species
    # request at box 16 over four windows: N grew 453 -> 788 and the composition
    # was (N, 0, 0) throughout. What ran was a valid single-species run of
    # species A at species A's activity; what was asked for was a mixture.
    #
    # The structural half of P7 does exist -- `grid_up_nd` splits a species
    # column and the reassembly puts it back, and that fixed the writeback which
    # dropped it -- so the plumbing is ready for the kernel. Only the kernel is
    # missing, which is exactly what P7 is.
    #
    # Refused HERE rather than half-wired in the kernel, for the reason the
    # metropolis guard gives: adding the species draw on top of a species-blind
    # energy and a missing swap would fix the visible symptom (the composition
    # stops being (N, 0, 0)) and leave the two that produce plausible numbers.
    if typed and parallel:
        raise ValueError(
            "run.kind='parallel' is not multi-species yet (P7): the cell kernel "
            "in gcmc/engine/cell_parallel.py draws no species, proposes no identity swap and "
            "reads no per-pair sigma/epsilon, so a typed parallel run inserts "
            "everything as the first species and silently samples a "
            "single-species system at that species' activity. The species column "
            "IS carried through the decomposition and the reassembly, so what is "
            "missing is the cell kernel, not the plumbing. Use run.kind='single' "
            "for a species file, or drop potential.species_file."
        )
    if pmf and metropolis and not lj:
        raise ValueError(
            "metropolis with a hard-core pmf auxiliary is not implemented; the "
            "pmf bias enters through the trial-cloud selection, and metropolis "
            "draws a single trial with no selection to bias. Use "
            "sampler.type='rosenbluth' for pmf runs, or drop the auxiliary "
            "block for plain hard-core metropolis."
        )
    if pmf and not two_dim and "g3_csv" not in aux:
        raise ValueError("3D pmf auxiliary requires auxiliary.g3_csv")

    # ---- short-range auxiliary support matrix -----------------------------
    # Deliberately narrow. Every rejection is a combination that is either
    # meaningless or simply untested; each is a separate, cheap relaxation later,
    # and refusing them now is what keeps "validated" honest.
    #
    # The matrix itself lives in `_short_range_unsupported`, ONE function, because
    # there are two callers: a per-run `auxiliary` block, which is a request and
    # is refused when unsupported, and a typed species file, which is a force
    # field and instead drops the auxiliary with a warning (see the typed section
    # above for why the two differ). Two copies of a support matrix is two
    # matrices, and the one that gets a new case added is not the other one.
    if short_range:
        _unsupported = _short_range_unsupported(
            lj=lj, sampler=sampler, qa=qa, pore=_pore_geometry is not None
        )
        if _unsupported is not None:
            raise ValueError(_unsupported)
        # The numeric guard needs lower_cutoff and uc, which are resolved later;
        # it lives immediately after them (search: "short-range auxiliary radius").
    # The log's column selection. Validated HERE rather than in the runner so a
    # typo is refused before the run directory exists -- M15, fail at the
    # earliest layer that can see the problem. `select_columns` owns the rules
    # (the two keys are mutually exclusive, the forced groups cannot be dropped,
    # an unknown name raises); this call is what makes them apply at parse time.
    _log_cols = run.get("log_columns")
    _log_drop = run.get("log_drop")
    for _key, _val in (("log_columns", _log_cols), ("log_drop", _log_drop)):
        if _val is not None and not isinstance(_val, (list, tuple)):
            raise ValueError(
                f"run.{_key} is a LIST of column-group names, got {type(_val).__name__}"
            )
    from .runner import select_columns as _select_columns

    _select_columns(
        [],
        include=None if _log_cols is None else tuple(_log_cols),
        drop=None if _log_drop is None else tuple(_log_drop),
    )
    if float(run.get("log_flush_interval", 0.0)) < 0.0:
        raise ValueError(
            f"run.log_flush_interval is a time floor between log flushes in "
            f"seconds and cannot be negative (got "
            f"{run['log_flush_interval']}); 0.0 means flush every row"
        )
    if int(smp.get("depth", 0)) < 0:
        raise ValueError("depth must be >= 0")
    # WHAT IS LEFT HERE is only a conflict between two keys the sampler DOES
    # read; everything about which keys belong to which sampler is `_SAMPLER_KEYS`
    # and `_check_sampler` above. The metropolis "takes no depth/trials/..." and
    # channel "has no trial_box_side" refusals used to live here, one hand-written
    # message per sampler per key, which is how channel_length/channel_width came
    # to have no message at all and be silently accepted by three samplers.
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

    # ---- typed species spec ------------------------------------------------
    # Needs beta (above) and the dimension; everything geometric below then
    # derives from it instead of from run.radius / run.lower_cutoff.
    species: SpeciesSpec | None = None
    if typed:
        # A CANONICAL run has no state point to state. `build_species_spec`
        # needs a route because every other typed run's composition is set by
        # one; here the composition is set by the starting configuration and the
        # activities never enter a single acceptance. A unit activity for every
        # species is the honest stand-in -- it makes the run identity blind to a
        # number that does no work, so two canonical runs differing only in a mu
        # nobody read hash the same, which is correct.
        _state_point = (
            {**ens, "z": dict.fromkeys(_species_names_of(species_file), 1.0)}
            if nvt_typed
            else ens
        )
        species = build_species_spec(
            str(species_file), _state_point, beta=beta, dimension=dim
        )
        mu = float(species.mu[0])
        # ---- the TYPED short-range auxiliary ---------------------
        #
        # THE DEFECT THIS CLOSES. `species.aux_cutoff` is the (T, T) matrix
        # `a * sigma_ij` built from the species file's `aux: {multiplier: a}`,
        # and for a long time nothing outside `species.py` read it. The engine's
        # selection radius comes from `pot_spec.aux_cutoff`, which only the
        # per-run `auxiliary` block sets and which a typed run may not use -- so
        # a species file could declare an auxiliary, pass every range check in
        # `species_file`, print itself in the run header, and be ignored. The
        # shipped `ff/species_three_reduced.yaml` declares `multiplier: 1.6` and
        # every typed run in the feature matrix was proposing at the full cutoff.
        #
        # Not a wrong answer -- the auxiliary is a proposal bias and dropping it
        # samples the same distribution, more slowly -- which is exactly why it
        # could sit there. A declared knob that does nothing is the failure mode
        # this file exists to refuse.
        #
        # WHY THE SCALAR MAX AND NOT THE PER-PAIR MATRIX. P5 asks for masks built
        # at `aux_cutoff_ij`. The scalar `max_ij aux_cutoff_ij` is used instead,
        # for now, and it is the conservative direction: every pair is proposed
        # into at a radius at least its own, so no pair is proposed at less than
        # the algebra assumes, and the correction is the same difference of two
        # radii the untyped path takes. Going per-pair is a SPEED refinement (the
        # DP window narrows for the small pairs) and needs a per-pair split
        # energy; it is not a correctness step, and doing it first would have
        # meant landing the correction and the optimisation in one commit.
        _typed_aux = species.aux_cutoff
        if _typed_aux is not None:
            _aux_why = _short_range_unsupported(
                lj=lj, sampler=sampler, qa=qa, pore=_pore_geometry is not None
            )
            if _aux_why is None:
                short_range = True
                aux_cutoff = species.max_aux_cutoff
            else:
                # WARN AND RUN, rather than raise. The per-run `auxiliary` block
                # is a request and an unsupported one is refused (see the support
                # matrix below); a species file is a FORCE FIELD, shared across
                # every run that uses that model, so refusing every branch or
                # pore run because the file mentions an auxiliary would make the
                # file unusable for the runs it is otherwise right for. Dropping
                # the auxiliary is safe -- it is a proposal bias -- so this is
                # the same shape as the LJ tail being disabled in a pore: say so
                # in the log and continue.
                log.warning(
                    "species file declares an auxiliary (aux_cutoff_ij up to "
                    "%.4f) but this run does not use one: %s. The auxiliary is a "
                    "proposal radius, so the sampled distribution is unaffected; "
                    "only the speed-up is lost.",
                    float(_typed_aux.max()),
                    _aux_why,
                )

    # `radius`, `diameter` and `lower_cutoff` are scalars in a typed run only as
    # CONSERVATIVE STAND-INS for the legacy geometric guards (pore gaps, cell
    # pitch floors, buffer minima, the periodic guard): each takes the largest
    # value over species or pairs, so a guard that passes on the stand-in passes
    # on every pair. Nothing physical reads them in typed mode -- the hot path
    # reads `species.table` and `species.core_floor` per pair -- and P3 asserts
    # that. Taking the max rather than a species-0 value is the whole point: a
    # stand-in that is not an upper bound would let a guard pass for the small
    # species and be violated by the large one.
    if species is not None:
        radius = 0.5 * float(species.sigma_self.max())
    else:
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
    if species is not None:
        lower_cutoff = species.max_core_floor
    else:
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
        # channel branch of metropolis_dynamics.advance), so its volume is the natural
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
        # The cyclic rod runs wall to wall, so its two ends meet through the
        # periodic boundary and the transfer-matrix sweep closes on itself
        # (ChannelSweepHardCoreCyclic scores that seam with true minimum-image distances
        # instead of an axial window).
        #
        # R16: the rod kind comes from the SAMPLER, not from this field, and the
        # four legacy spellings are REFUSED. The text here claimed for two stages
        # that they were "still accepted so existing params files keep working",
        # directly above the line that rejects them -- which is how three slow
        # tests came to be erroring on a retired spelling while reading as though
        # they were measuring physics. A comment that contradicts the line under
        # it is worse than none, because it is what a reader checks first.
        if isinstance(_clen, str) and _clen.strip().lower() in _LEGACY_CYCLIC_SPELLINGS:
            raise ValueError(
                f"sampler.channel_length = {_clen!r} used to select the cyclic "
                f"wall-to-wall rod. It is a different move, not a length, so it "
                f'is now its own sampler: set sampler.type = "channel_cyclic" '
                f"and drop channel_length."
            )
        channel_cyclic = sampler == "channel_cyclic"
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
            # `trials` and `channel_length` are refused for this sampler by
            # `_check_sampler`, which is where "which keys does channel_cyclic
            # read" is now stated once. They used to be refused here (trials) and
            # not at all (channel_length).
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
    if species is not None:
        uc = species.rc
    elif "upper_cutoff" in run:
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
    # Typed: the packing fraction is sum_s n_s v_s / V and the per-species v_s
    # lives on the spec. The scalar kept here is the largest species' volume, so
    # `eta` computed from it is an upper bound rather than a wrong number; use
    # Config.packing_fraction, which routes to the per-species sum when typed.
    if species is not None:
        vol_mol = float(species.volume_of_molecule.max())

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
    # A FIXED-N ENSEMBLE NEVER EVALUATES A TAIL. The mean-field term enters
    # exactly one place, the insertion/deletion acceptance, as tail*(2N +/- 1);
    # a number-conserving move has no such term, so the setting is read by
    # nothing. Measured: `tail: false` against the default, same seed, 400
    # cycles, byte-identical trajectory and two different run ids.
    #
    # Stated-only, like the pore/QA rule below it and for the same reason: the
    # default is left to derive what it always derived, so no existing run id
    # moves, while the value that used to split them can no longer be written.
    if lj and nvt and "tail" in pot:
        raise ValueError(
            f"potential.tail does nothing under ensemble.type={ens_type!r}: the "
            "mean-field tail enters only the insertion/deletion acceptance, and "
            "a fixed-N ensemble makes no insertion or deletion. Remove it."
        )
    # A SPECIES FILE MEANS NO TAIL, and the file itself already says so:
    # `species_file.load_forcefield` refuses `interaction.tail_correction: true`
    # outright ("no-tail on both sides"). But `potential.tail` defaults to True
    # here, so a typed run was picking up a tail its own force field had
    # forbidden -- and picking up the WRONG one: `lj_tail_pref` below is derived
    # at sigma = epsilon = 1, while the mixture tail is
    # sum_ij x_i x_j f(sigma_ij, eps_ij), a composition-dependent quantity that
    # changes as the run's composition moves.
    #
    # So it is forced off for typed runs, and an explicit `tail: true` raises
    # rather than being silently overridden -- the same shape as the pore/QA case
    # a few lines below, and for the same reason: a tail integral that does not
    # describe the system is not a small correction, it is an N^2 energy the
    # system does not have.
    if typed and lj:
        if pot.get("tail") is True:
            raise ValueError(
                "potential.tail: true is not available for a species file. The "
                "mean-field tail for a mixture is composition-dependent "
                "(sum_ij x_i x_j over the pair table) and is not implemented; "
                "the species-file format forbids tail corrections on its own "
                "side too (interaction.tail_correction must be false). Remove "
                "potential.tail or set it false."
            )
        lj_tail_on = False
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
    # ---- the ladder dynamics' remaining refusals (section 3.5) ----------
    #
    # ALL OF THEM ARE ABOUT THE LADDER and none about the kernel, which is why
    # `suwa_todo_dynamics` inherits every one without a line of its own. A
    # refusal that were really about the heat bath would have to name it.
    if ladder:
        if full_repacking or int(smp.get("depth", 0)) < 1:
            raise ValueError(
                f"engine={engine} requires sampler.depth >= 1 and takes "
                "no sampler.full_repacking. A ladder dynamics replaces add / remove "
                "/ repack with ONE move that draws the window's whole occupancy "
                "from its exact conditional, so dN is free and neither knob "
                "selects a move type any more; depth >= 1 is kept because it is "
                "the channel sampler's own precondition (depth 0 has no clique "
                "structure to sweep). Set depth: 1 and drop full_repacking."
            )
        if short_range:
            # SECTION 3.5, the config twin of the assertion in the engine
            # module. The dual-cutoff correction depends on the drawn
            # ARRANGEMENT, not on k, so it cannot fold into the ladder and would
            # reintroduce an acceptance test -- i.e. it would make a
            # "rejection-free" engine reject. A later opt-in hybrid (Glauber engine on
            # the rc_sel ladder plus one Metropolis step on the uc residual) is
            # exact but not rejection-free and must log its acceptance; it is out
            # of scope.
            raise ValueError(
                f"engine={engine} refuses the short-range (dual-cutoff) "
                "auxiliary. A ladder dynamics has no acceptance step to divide a "
                "shortened selection radius back out of, so it scores every "
                "pair at run.upper_cutoff; the dual-cutoff correction depends on "
                "the drawn arrangement rather than on the occupancy k, so it "
                "cannot fold into the occupancy ladder and would reintroduce a "
                "rejection. Drop the auxiliary block, or use "
                "engine=metropolis_dynamics."
            )
        if pmf:
            raise ValueError(
                f"engine={engine} refuses the pmf auxiliary: the pmf is a "
                "SELECTION bias divided back out in an acceptance ratio, and "
                "this engine has no acceptance ratio to divide it out of."
            )
        if ens_type == "NVT" and typed:
            raise ValueError(
                f"engine={engine} does not support the CANONICAL "
                "ensemble (ensemble.type='NVT' with a species file), which "
                "fixes every species count and not just the total. The window "
                "move resamples a whole block's occupancy from its exact "
                "conditional, and there is no way to do that while conserving a "
                "species MULTISET -- the transfer matrix draws a clique, not a "
                "permutation. Use ensemble.type='semigrand' (total N fixed, "
                "composition free), which is what this engine's fixed-N "
                "restriction actually samples."
            )
        if ens_type not in ("muVT", "NVT", "semigrand"):
            raise ValueError(
                f"engine={engine} does not support "
                f"ensemble.type={ens_type!r}. It is derived for muVT with a "
                "separable frozen exterior (and its fixed-N restriction, where "
                "the ladder collapses to k = k_in); a pore or a quenched matrix "
                "changes what 'outside the window' means and has its own gate."
            )
    # ---- M and kmax, DERIVED (section 1.3) --------------------------------
    #
    #     M    = round(trial_density * V_W)          per window body
    #     kmax = min(M, ceil(packing_bound(body)) + kmax_margin)
    #
    # BOTH ARE PHYSICS PARAMETERS, NOT ENGINE CONSTANTS: each changes the chain,
    # so neither may be a literal inside the engine module where it would be an
    # undeclared knob that escapes the run digest. They are declared here, and
    # `config_fields.DERIVED` records why they are derived from hashed inputs
    # rather than hashed themselves.
    #
    # M IS THE CONFIGURED TRIAL COUNT, deliberately the same number and not a
    # second derivation to drift: `n_trials` above is already
    # `round(trial_density * channel_length * channel_width**(dim-1))`, which is
    # exactly `round(trial_density * V_W)`. What the field adds is the MEANING --
    # on the Metropolis path `n_trial_molecules` is the size of the fresh
    # cloud, while here it is the size of the whole POOL (in-window guests plus
    # cloud), and the pool is always exactly M points because inadmissible
    # trials are kept at weight zero rather than dropped (section 2B.5).
    #
    # M MAY DEPEND ON THE BODY GEOMETRY -- legal, the body is drawn without
    # looking at the configuration -- but NEVER ON OCCUPANCY (section 2B.4).
    # That is asserted rather than asserted-in-a-comment: G1.11 builds the body
    # against configurations with different k_in and requires the same M.
    window_M = 0
    window_kmax = 0
    if ladder:
        window_M = int(n_trials)
        _bound = cuboid_packing_bound(channel_length, channel_width, dim, diameter)
        window_kmax = min(window_M, int(math.ceil(_bound)) + kmax_margin)
        if not (1 <= window_kmax <= window_M):
            raise ValueError(
                f"engine={engine} needs 1 <= kmax <= M, derived "
                f"kmax={window_kmax} and M={window_M} from trial_density="
                f"{trial_density:g} on a {channel_length:g} x {channel_width:g} "
                f"body (V_W = {channel_length * channel_width ** (dim - 1):g}). "
                "M is round(trial_density * V_W), so raise sampler.trial_density "
                "until the pool can hold at least one particle."
            )
        # WHY THERE IS NO "M IS BELOW THE PACKING BOUND" REFUSAL HERE. The
        # bound is a rigorous over-estimate and, for a thin body, a loose one
        # (`cuboid_packing_bound` says by how much), so requiring M above it
        # would demand a `trial_density` the lambda <= 4 rule below then
        # refuses -- two rules contradicting each other over a number neither
        # measures. What matters is whether the chain ever REACHES an occupancy
        # the pool cannot hold, and that is measured rather than predicted:
        # `cut_M` and `cut_k` count every such window, and
        # G3.4 asserts both are identically zero on the hard-core path. A
        # physical channel occupancy is far below M at any realistic
        # trial_density, so they are zero -- and if they are not, the counter
        # says so in the log instead of an estimate saying so at setup.
        # LAMBDA <= 4 TRIALS PER SIGMA OF CHANNEL LENGTH. A hard scoping rule,
        # not a tunable. With rc = uc forced (section 1.2) the DP's eviction
        # window widens, and the measured penalty is 1.3-2x at lambda <= 4 but
        # 11x at lambda = 8 (state counts: L=24, M=192, lambda=8 goes from 9099
        # states / 20 ms at rc 1.5 to 283339 / 217 ms at rc 3.0). LJ path only:
        # the hard-core sweep's eviction window is the contact distance and does
        # not widen.
        _lambda = window_M * diameter / channel_length
        if lj and _lambda > 4.0 + 1e-9:
            raise ValueError(
                f"engine={engine} refuses {_lambda:.2f} trials per sigma "
                f"of channel length (M={window_M} over L={channel_length:g} at "
                f"sigma={diameter:g}); the limit is 4. With the selection radius "
                "forced to run.upper_cutoff the transfer matrix's eviction "
                "window widens, and the measured cost is 1.3-2x at lambda <= 4 "
                "against 11x at lambda = 8. Lower sampler.trial_density to at "
                f"most {4.0 * channel_length / (diameter * channel_length * channel_width ** (dim - 1)):.4g} "
                "or lengthen the rod. This is a scoping rule, not a tunable."
            )
    if channel:
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
            # THE HEAT BATH ENFORCES AT THE RADIUS IT SCORES AT, which is
            # `upper_cutoff` on the LJ path (section 1.2: there is no acceptance
            # step to divide a shortened selection radius back out of, and the
            # short-range auxiliary is refused above so `guard_cut` is already
            # `uc` there) and the LARGEST ACTIVE PAIR FLOOR on the hard-core path.
            #
            # The second half is the only place the two engines' guards differ,
            # and it is a real gap rather than pedantry: `diameter` is the
            # SINGLE-COMPONENT floor, so a typed hard-core run whose largest
            # `c * sigma_ij` exceeds it was guarded at too short a radius. Under
            # Metropolis a wrapped pair beyond the guard is divided back out of
            # the acceptance; under either LADDER dynamics there is nothing to
            # divide it out of, so it has to be excluded geometrically.
            if ladder and not lj and species is not None:
                guard_cut = max(guard_cut, max_core_floor(species.table.core_floor))
            assert_channel_guard(
                channel_length, channel_width, dim, box, guard_cut, span_name="box"
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
    # NO TRANSLATION MEANS NO STEP SIZE. `nvt and not canonical` -- untyped NVT
    # and semigrand -- forces `choice = "repack"` in the engine and proposes no
    # translation at all, so `move_factor` scales nothing. Measured: two values,
    # same seed, 400 cycles, byte-identical trajectory, and TWO RUN IDS, because
    # `move_factor` is an identity field. Same defect as the foreign sampler keys
    # above, in a different block.
    #
    # Refused only when STATED, so no existing run id moves: the default still
    # derives whatever it derived, and it is the same number for every run of
    # this kind, so it can no longer split anything.
    if nvt and not canonical and "move_factor" in run:
        raise ValueError(
            f"run.move_factor does nothing under ensemble.type={ens_type!r}: its "
            "only move is the repack, which conserves N and proposes no "
            "translation, so there is no trial displacement to scale. Remove it. "
            "(ensemble.type='NVT' WITH a species file is the exception -- that "
            "move set does translate -- and muVT translates too.)"
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
        # THE FLOOR IS THE DEPTH PATH'S, NOT THE BRANCH SAMPLERS'. It read
        # `if branch and ...`, so `rosenbluth` was exempt -- and rosenbluth takes
        # the same depth path in `advance`, where a remove claims `depth+1` trial
        # slots via `rng.choice(n_trials, size=n_remove, replace=False)`. Every
        # accepted-but-impossible combination died there:
        #
        #     depth 1, trials 1     depth 2, trials <= 2     depth 3, trials <= 3
        #
        # measured, and it is not a rare tail -- the first exchange proposal of
        # the first window raises. So
        # the validator accepted a configuration that could not survive one
        # window, which is the one thing tier 2's green is supposed to mean it
        # cannot do. Nothing in the matrix covers it because every rosenbluth
        # point there is generated with a trial pool comfortably above its depth.
        #
        # Found while looking for a typed reference sampler for P8's speedup
        # claim: `rosenbluth` with `trials: 1` is metropolis in all but name and
        # is what the typed+metropolis refusal points at, so it is exactly the
        # configuration a reader following that message would write next.
        if n_trials < _depth + 1:
            raise ValueError(
                f"{sampler} needs at least depth+1={_depth + 1} trial spheres to "
                f"place a depth-{_depth} move, but only {n_trials} were "
                f"requested; raise trials or trial_density, or lower depth"
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
    # Nothing in between: mixing the two is an error, not a merge.
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

        # R17 (2026-08): the depth>=1 cell proposal was believed to carry a
        # density bias set by the cell/cavity ratio (+19% at 2.7 cavity widths;
        # cell_unify_plan.md U3), on the reasoning that the decomposition
        # satisfied global but not detailed balance. Later runs did not
        # reproduce the bias and detailed balance does hold, so the caveat is
        # gone and parallel is exact at every depth.
        #
        # `describe_parallel` still REPORTS the shortest cell in cavity widths.
        # That is now a performance figure, not an accuracy one: a small ratio
        # means more border work per live cell. Kept because it is useful for
        # sizing a job, and because a number an operator already recognises is
        # cheaper to reinterpret than to remove and reintroduce.

    _raw_seed = run.get("seed")
    if isinstance(_raw_seed, (list, tuple)):
        raise ValueError(
            "run.seed is a list (a sweep) but a single run got it unexpanded. "
            "Launch sweeps with `python -m gcmc.run.launcher`; if you did, "
            "gcmc/run/launcher.py on this machine is outdated."
        )
    seed_val = int(_raw_seed) if _raw_seed is not None else None

    cfg = Config(
        potential=potential,
        bias=bias,
        sampler=sampler,
        dimension=dim,
        chem_pot=mu,
        ensemble_type=ens_type,
        target_n=target_n,
        box=box,
        box_sides=box_sides,
        species=species,
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
        g3_csv=(
            aux_block.get("g3_csv")
            if pmf and not two_dim and aux_block is not None
            else None
        ),
        aux_cutoff=aux_cutoff,
        two_dim=two_dim,
        parallel=parallel,
        beta=beta,
        fugacity=math.exp(beta * mu),
        radius=radius,
        diameter=diameter,
        r_squared=r_squared,
        n_trial_molecules=n_trials,
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
        log_columns=None if _log_cols is None else tuple(_log_cols),
        log_drop=None if _log_drop is None else tuple(_log_drop),
        initial_config=run.get("initial_config"),
        matrix_config=run.get("matrix_config"),
        sweep_label=run.get("sweep_label"),
        pore_geometry=_pore_geometry,
        pore_H=_pore_H,
        pore_R=_pore_R,
        pore_axis_len=_pore_axis_len,
        engine=engine,
        kmax_margin=kmax_margin,
        window_M=window_M,
        window_kmax=window_kmax,
    )
    # G0.9 OF THE DESIGN PLAN: the periodic guard has never been
    # checked on the bulk channel path, so nobody knew for which configurations
    # its local-Euclidean assumption held. This records one line per channel
    # Config when `GCMC_GUARD_AUDIT` names a sink and does nothing otherwise --
    # which is how the scan reaches the three dozen channel configurations built
    # programmatically inside the test suite without a maintained list of them.
    # It is an audit, not a check: nothing here refuses a run.
    guard_audit.record(cfg)
    return cfg


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

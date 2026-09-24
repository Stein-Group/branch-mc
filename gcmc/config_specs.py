"""The sub-configs `Config` decomposes into (R8, second half).

Five of them are the SUBMISSION -- potential, sampler, ensemble, execution, IO --
and are the split R8 names. A sixth, `KernelSpec`, is the derived half: the
precomputed forms the hot loop actually reads. R8 did not name it because R8 is
about identity, and derived values are never hashed; it is needed all the same,
because a kernel that wants `beta` and `box_dims` must otherwise take the whole
`Config`. See `KernelSpec` for the full argument.


WHY A PROJECTION FIRST, AND NOT THE SPLIT ITSELF. Turning `Config` from a flat
72-field dataclass into a composition touches 278 attribute reads in the package,
three `dataclasses.replace` call sites, the checkpoint round-trip and the field
partition test -- all at once. That is the big-bang the reduction-gate discipline
 exists to avoid, and it is the single largest change in the plan.

So the dataclasses are created HERE first, as read-only projections of an
unchanged `Config`, with a round-trip test proving the five together lose
nothing. That buys the two things that actually matter:

  * **Modules get something narrow to depend on.** A layer-1 function can take a
    `PotentialSpec` instead of the whole `Config`, which is what breaks the
    god-object coupling -- and it can do so one module at a time, each migration
    its own commit with its own gate.
  * **Identity becomes typed, not just grouped.** `hash(potential + sampler +
    ensemble)` is now three objects rather than a filtered name list.

Nothing here is torn out later. When `Config` finally composes rather than
projects, these same classes become the members it holds; the projection
properties become one-line accessors. The direction of travel is the same, the
step is just smaller.

WHERE THE FIELD LISTS COME FROM. `config_fields`, and nowhere else. Each class
below is checked against its group at import time, so a field added to a group
without a matching attribute -- or the reverse -- fails immediately rather than
silently dropping out of the identity.

UNITS. Every quantity is engine-reduced. Nothing in this module has seen an
Angstrom or a Kelvin; `forcefield/units.py` is the single boundary where that
conversion happens.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from . import config_fields

if TYPE_CHECKING:
    from .run.sim_setup import Config


@dataclasses.dataclass(frozen=True, slots=True)
class PotentialSpec:
    """What interacts, and how strongly.

    Identity-bearing. Two runs differing in any field here are sampling
    different Hamiltonians -- or, for the auxiliary fields, the same ensemble
    through a different proposal distribution, which is still a different
    experiment (see the rule in `config_fields`).
    """

    potential: str
    temp: float
    rc: float | None
    lower_cutoff: float
    lj_tail_pref: float | None
    bias: str | None
    eta: float | None
    gamma_max: float | None
    gamma_dict: Any
    delta_eta: float | None
    x_spacing: float | None
    y_spacing: float | None
    volume_of_molecule: float
    g3_csv: str | None
    min_g3: float | None
    #: radius beyond which a neighbour contributes nothing to the 3D bias;
    #: None means every neighbour inside the potential cutoff contributes
    g3_r_cut: float | None
    #: which reader the g3 table is read with (`auxiliary.G3_READER`), or None
    #: where there is no table. Derived, not configurable; see
    #: `config_fields.POTENTIAL` for why it is still part of the identity.
    g3_reader: str | None
    aux_cutoff: float | None
    species_identity: Any


@dataclasses.dataclass(frozen=True, slots=True)
class SamplerSpec:
    """How a trial cloud is proposed and weighted.

    Identity-bearing. `trial_box_side` and `n_trial_molecules` are here, not
    in execution, because they change the Markov chain even though the
    equilibrium distribution is invariant under them -- and because they are
    documented launcher sweep axes, so two runs differing there must not collide.
    """

    sampler: str
    #: the DYNAMICS axis: metropolis_dynamics | glauber_dynamics. See the note in
    #: `config_fields.SAMPLER`; `sampler` and `engine` are different questions
    #: and the refusal messages in `sim_setup` say so by name.
    engine: str
    #: how far above the body's geometric packing bound `window_kmax` is set.
    #: The Glauber engine's only new tunable.
    kmax_margin: int
    depth: int
    full_repacking: bool
    n_trial_molecules: int
    trial_density: float
    trial_box_side: float
    cavity_radius: float
    move_factor: float
    channel_length: float
    channel_width: float
    channel_cyclic: bool
    radius: float
    #: fixed = the nine fitted pmf exponents; adaptive = solve one per
    #: Rosenbluth level for a target participation fraction. None means the key
    #: was absent, which is fixed as well -- see `config_fields.SAMPLER` for why
    #: this is identity-bearing and why absent and explicit "fixed" are not the
    #: same run.
    gamma_mode: str | None
    #: target participation fraction PR / k; None unless gamma_mode is adaptive
    pr_target: float | None
    #: participation | perplexity; None unless gamma_mode is adaptive
    gamma_criterion: str | None
    #: moments | grid; None unless gamma_mode is adaptive. Which solver returns
    #: the exponent: the closed-form expansion or the 17-point grid search.
    gamma_solver: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class EnsembleSpec:
    """Which ensemble, at what state point, in what box.

    Identity-bearing. `seed` is here rather than in execution: a different seed
    is a different chain, and two seeds must not share a run directory.
    `matrix_positions` is the quenched disorder realisation, hashed as a rounded
    coordinate digest so I/O noise cannot split one run into many.
    """

    ensemble_type: str
    chem_pot: Any
    fugacity: Any
    target_n: int | None
    target_count: float | None
    control_gain: float
    control_gain_tau: float
    control_window: int
    box: float
    box_sides: Any
    dimension: int
    pore_geometry: str | None
    pore_H: float | None
    pore_R: float | None
    pore_axis_len: float | None
    matrix_config: str | None
    matrix_positions: Any
    seed: int | None


@dataclasses.dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """How the same chain is computed: serial or across cells.

    NEVER identity-bearing. Two runs differing only here are the same
    experiment, belong in the same directory, and must be able to resume each
    other -- a run interrupted on eight cores and resumed on four is still that
    run.
    """

    mode: str
    parallel: bool
    n_cell: int | list[int] | None
    cell_counts: Any
    cell_length: float
    cell_length_sides: Any
    buffer: float
    n_workers: int


@dataclasses.dataclass(frozen=True, slots=True)
class IOSpec:
    """Cadence and paths: when to write, where to write, where to start.

    NEVER identity-bearing. `initial_config` is deliberately here rather than in
    the ensemble: a warm start changes how quickly a chain reaches equilibrium,
    not which equilibrium it reaches, so a warm-started run and a cold-started
    one at the same state point are the same experiment.
    """

    sampling_window: int
    burn_in: int
    target_cycles: float
    checkpoint_every: int
    traj_every: int
    log_flush_interval: float
    log_columns: tuple[str, ...] | None
    log_drop: tuple[str, ...] | None
    initial_config: str | None
    sweep_label: str | None
    #: record the ESJD counters in each window summary (and in the log)
    esjd: bool


@dataclasses.dataclass(frozen=True, slots=True)
class KernelSpec:
    """What the hot loop was handed, as opposed to what the user asked for.

    NEVER identity-bearing, and structurally incapable of becoming so: every
    field here is a function of fields already hashed elsewhere, so hashing one
    would double-count its inputs. That is the failure `config_fields` was
    written to end -- `sampler` reached the old digest four separate ways
    through this group.

    WHY THIS EXISTS AS A SIXTH SPEC. The other five are the SUBMISSION: `temp`,
    `rc`, a scalar `box` and a `box_sides` that is usually None. The kernels
    below L6 do not want any of that. They want `beta`, `r_squared` and a
    per-axis `box_dims` -- the precomputed forms, resolved once at setup so the
    hot path contains neither a divide nor a None check.

    Before this class, a kernel that wanted them had no narrow object to take
    and had to accept the whole `Config` -- which is exactly the god-object
    coupling R8 exists to break, reappearing one layer down. `cell_parallel.py` was
    where that became unavoidable: 11 of its 36 `cfg.` reads are derived, and no
    combination of the five could have supplied them.

    Kept as a projection like the other five, and checked against
    `config_fields.DERIVED` at import by the same mechanism, so the six-way
    partition of `Config` is stated once and verified once.
    """

    potential_type: str
    #: Glauber pool size M and ladder truncation kmax, derived per section 1.3
    #: of the design plan. Zero on a `metropolis_dynamics` run, which reads
    #: neither.
    window_M: int
    window_kmax: int
    beta: float
    branch: bool
    channel: bool
    metropolis: bool
    two_dim: bool
    diameter: float
    upper_cutoff: float
    no_neigh: int
    r_squared: float
    total_volume: float
    species: Any
    n_species: int
    box_dims: Any
    cell_dims: Any


#: Each class paired with the `config_fields` group it must mirror exactly.
#: Checked at import (see `_verify_against_groups`), so the two cannot drift.
_PAIRS: tuple[tuple[type, tuple[str, ...], str], ...] = (
    (PotentialSpec, config_fields.POTENTIAL, "POTENTIAL"),
    (SamplerSpec, config_fields.SAMPLER, "SAMPLER"),
    (EnsembleSpec, config_fields.ENSEMBLE, "ENSEMBLE"),
    (ExecutionSpec, config_fields.EXECUTION, "EXECUTION"),
    (IOSpec, config_fields.IO, "IO"),
    (KernelSpec, config_fields.DERIVED, "DERIVED"),
)


def _verify_against_groups() -> None:
    """Fail at import if a class and its group have drifted apart.

    At import rather than in a test, because the two are one declaration written
    twice and a mismatch is never intentional. A test would catch it too, but
    only for someone who ran the tests; an import-time check catches it for
    anyone who so much as loads the package.
    """
    for cls, group, name in _PAIRS:
        have = {f.name for f in dataclasses.fields(cls)}
        want = set(group)
        if have != want:
            missing = sorted(want - have)
            extra = sorted(have - want)
            raise AssertionError(
                f"{cls.__name__} does not mirror config_fields.{name}: "
                f"missing={missing} extra={extra}. They are one declaration "
                f"written twice and must be edited together."
            )


_verify_against_groups()


def _project(cls: type, cfg: Config) -> Any:
    """Build one sub-config by reading its fields off `cfg`.

    `getattr` rather than an explicit constructor call: the field list is
    already stated once on the dataclass, and restating it here would be a third
    copy to drift. Properties (`species_identity`) read exactly like fields, so
    the projection does not need to know which is which.
    """
    return cls(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cls)})


def potential_spec(cfg: Config) -> PotentialSpec:
    """The potential half of `cfg`. Identity-bearing."""
    return _project(PotentialSpec, cfg)


def sampler_spec(cfg: Config) -> SamplerSpec:
    """The sampler half of `cfg`. Identity-bearing."""
    return _project(SamplerSpec, cfg)


def ensemble_spec(cfg: Config) -> EnsembleSpec:
    """The ensemble half of `cfg`. Identity-bearing."""
    return _project(EnsembleSpec, cfg)


def execution_spec(cfg: Config) -> ExecutionSpec:
    """How `cfg` is computed. Never identity-bearing."""
    return _project(ExecutionSpec, cfg)


def io_spec(cfg: Config) -> IOSpec:
    """When and where `cfg` writes. Never identity-bearing."""
    return _project(IOSpec, cfg)


def kernel_spec(cfg: Config) -> KernelSpec:
    """The precomputed forms the hot loop reads. Never identity-bearing."""
    return _project(KernelSpec, cfg)


def physics_specs(cfg: Config) -> tuple[PotentialSpec, SamplerSpec, EnsembleSpec]:
    """The three identity-bearing sub-configs, in hash order.

    This is what the run fingerprint IS, once `hash_fingerprint` reads objects
    rather than a filtered name list. Returned as a tuple so the order is fixed
    by this function and not re-chosen at each call site.
    """
    return potential_spec(cfg), sampler_spec(cfg), ensemble_spec(cfg)

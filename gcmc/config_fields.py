"""Which Config fields are physics, which are execution, and which are derived.

WHY THIS FILE EXISTS (R8). The run fingerprint used to be a hand-curated tuple
of names in `hash_fingerprint`. Maintained lists drift, and this one had:

  * `lj_tail_pref` MISSING -- so `tail: true` and `tail: false`, two different
    Hamiltonians sampling different densities, hashed to the same id and would
    resume each other. A live collision, found by the partition test in S1.
  * `pore_axis_len` missing while `pore_H` and `pore_R` were present.
  * eight of twenty-five keys hashing as literal `null` on an ordinary run.
  * four keys DERIVED from others already in the set -- `potential_type` from
    `sampler`, which was itself hashed; `rc` and `upper_cutoff` are one number
    hashed twice.
  * two documented sweep axes (`trial_box_side`, `trials`) that changed the id
    only by ACCIDENT, leaking through derived keys.

None of that is a lapse of judgement. It is what a maintained list does.

So identity is no longer a list. It is a PARTITION, declared here once, and
`hash_fingerprint` derives the hashed set from it. A field is identity-bearing
because of which group it was declared in -- not because someone remembered.

THE RULE, stated once because the code previously contained two contradictory
versions of it:

    A field is PHYSICS if changing it changes the sampled ensemble OR the
    Markov chain used to sample it.

Proposal knobs count. That is why `trial_box_side` and the auxiliary cutoff are
physics even though the equilibrium distribution is invariant under them: two
runs differing there are not the same experiment and must not share a directory.
The old code included `aux_cutoff` on exactly this reasoning while excluding
`g3_csv` as "data, not a physics parameter" -- but a different g3 table changes
the proposal distribution too. `g3_csv` is physics, hashed by CONTENT so that
moving the file does not re-address the run.

WHY THE GROUPS ARE NAMED AFTER SUB-CONFIGS. They are the sub-configs. R8 splits
`Config` into `PotentialSpec`, `SamplerSpec`, `EnsembleSpec`, `ExecutionSpec`
and `IOSpec`; this module is that split expressed as field names before the
dataclasses exist. When they do, each tuple below becomes
`dataclasses.fields(...)` of the corresponding class and this file shrinks to
five one-line expressions. Doing it in this order means the grouping is written,
reviewed and TESTED against the real Config before any call site moves.
"""

from __future__ import annotations

# --------------------------------------------------------------- physics
#
# Hashed. Three groups, because they become three dataclasses -- and because the
# fingerprint is exactly "the physics sub-configs", never a subset chosen later.

POTENTIAL: tuple[str, ...] = (
    "potential",  # hard_core | lennard_jones
    "temp",  # reduced temperature
    "rc",  # interaction cutoff
    "lower_cutoff",  # hard-core floor
    "lj_tail_pref",  # mean-field tail correction on/off
    "bias",  # auxiliary flavour: pmf | short_range | None
    "eta",  # 2D pmf packing fraction
    "gamma_max",  # pmf exponent
    "gamma_dict",  # per-shell pmf exponents
    "delta_eta",  # pmf grid spacing
    "x_spacing",  # pmf grid
    "y_spacing",  # pmf grid
    "volume_of_molecule",  # sets the packing fraction the pmf is read at
    "g3_csv",  # 3D pmf table -- hashed by CONTENT digest, not path
    "aux_cutoff",  # short-range auxiliary radius
    # PROPERTY, and PHYSICS: the canonical hashable form of the force field --
    # a digest over sigma, epsilon, k_ij, the core floor, rc, the auxiliary
    # multiplier and the per-species state point. The `species` object it
    # summarises is DERIVED below: hashing an object is not meaningful, hashing
    # its digest is. Getting this backwards would silently drop the entire typed
    # model from the identity, so two different force fields would share a run
    # directory.
    "species_identity",
)

SAMPLER: tuple[str, ...] = (
    "sampler",  # metropolis | rosenbluth | branch | channel | channel_cyclic | reweight
    "depth",
    "full_repacking",
    "no_of_trial_molecules",
    "trial_density",
    "trial_box_side",
    "cavity_radius",
    "move_factor",
    "channel_length",
    "channel_width",
    "channel_cyclic",  # R16: derived from `sampler`, kept for the reduction check
    "radius",  # hard-core radius; `diameter` is its derived twin
)

ENSEMBLE: tuple[str, ...] = (
    "ensemble_type",  # muVT | NVT | QA | confined
    "chem_pot",
    "fugacity",  # z, the activity the chain actually uses
    "target_n",
    "target_count",
    "control_gain",  # mu-control feedback: changes the chain
    "control_gain_tau",
    "control_window",
    "box",
    "box_sides",
    "dimension",
    "pore_geometry",
    "pore_H",
    "pore_R",
    "pore_axis_len",
    "matrix_config",
    "matrix_positions",  # hashed as a rounded coordinate digest
    "seed",
)

PHYSICS: tuple[str, ...] = POTENTIAL + SAMPLER + ENSEMBLE

# --------------------------------------------------------------- execution
#
# Never hashed. These change HOW the same chain is computed, not WHAT is
# computed: two runs differing only here are the same experiment and belong in
# the same directory. That is the whole point of resume.

EXECUTION: tuple[str, ...] = (
    "mode",
    "parallel",
    "n_cell",
    "cell_counts",
    "cell_length",
    "cell_length_sides",
    "buffer",
    "n_workers",
)

IO: tuple[str, ...] = (
    "sampling_window",
    "burn_in",
    "target_cycles",
    "checkpoint_every",
    "traj_every",
    "log_flush_interval",
    "log_columns",
    "log_drop",
    "initial_config",
    "sweep_label",
)

# --------------------------------------------------------------- derived
#
# Never hashed, because hashing a function of other fields double-counts the
# fields it derives from. `sampler` used to reach the digest four separate ways
# through this group.

# NOTE. Five names here -- `potential_type`, `n_species`, `box_dims`, `cell_dims`
# and (in POTENTIAL) `species_identity` -- are PROPERTIES on Config, not dataclass
# fields. They are listed anyway because the old identity tuple hashed
# `potential_type`, and a group that silently omitted it would let it drift back
# in unnoticed. The totality test skips properties when checking for phantoms;
# see `PROPERTIES` below.
#
# THIS GROUP IS A SUB-CONFIG TOO. It mirrors `config_specs.KernelSpec`, the sixth
# spec, on the same import-time check as the other five. The five physics/
# execution/IO specs describe what the user ASKED FOR; this one describes what
# the hot loop was HANDED -- `beta` rather than `temp`, `r_squared` rather than
# `rc`, per-axis `box_dims` rather than a scalar side plus a None. Kernels below
# L6 need those and nothing else, which is why they get their own narrow object
# rather than the whole Config.
DERIVED: tuple[str, ...] = (
    "potential_type",  # PROPERTY: from potential + bias + sampler
    "beta",  # 1 / temp
    "branch",  # from sampler
    "channel",  # from sampler
    "metropolis",  # from sampler
    "two_dim",  # from dimension
    "diameter",  # 2 * radius, precomputed to keep the multiply out of the loop.
    # Q-R3, answered: an internal optimisation, never a user-facing knob. It
    # must not appear in params, in the schema, or in the identity.
    "upper_cutoff",  # the same number as `rc`
    "no_neigh",
    "r_squared",  # rc ** 2, precomputed for the hot loop
    "total_volume",  # a function of box and dimension
    "species",  # the SpeciesSpec object; `species_identity` above is what hashes
    "n_species",  # PROPERTY: len(species)
    # PROPERTIES, and classified here for the first time at the parallel
    # migration. Both were previously in NO group: not a deliberate omission but
    # the totality test's blind spot -- it checks stored FIELDS, and a property
    # classified nowhere is invisible to it. They are unambiguously derived:
    # `box_dims` is `box_sides` or the scalar `box` replicated `dimension` times,
    # `cell_dims` the same for `cell_length_sides` / `cell_length`. Both are the
    # A2.1 per-axis accessors every kernel reads instead of the scalar twins.
    "box_dims",  # PROPERTY: per-axis (Lx, Ly, Lz)
    "cell_dims",  # PROPERTY: per-axis parallel cell lengths
)

#: Names in the groups above that are Config PROPERTIES rather than dataclass
#: fields. Declared so the totality test can tell "I invented this name" from
#: "this is computed, not stored" -- two very different mistakes.
PROPERTIES: frozenset[str] = frozenset(
    {
        "potential_type",
        "species_identity",
        "n_species",
        "box_dims",
        "cell_dims",
    }
)

#: Every classified name. A Config field absent from this is a bug, caught by
#: `tests/unit/test_identity_partition.py` at the moment the field is added.
ALL: tuple[str, ...] = PHYSICS + EXECUTION + IO + DERIVED

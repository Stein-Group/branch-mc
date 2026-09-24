"""Which Config fields are physics, which are execution, and which are derived.

WHY THIS FILE EXISTS. The run fingerprint used to be a hand-curated tuple
of names in `hash_fingerprint`. Maintained lists drift, and this one had:

  * `lj_tail_pref` MISSING -- so `tail: true` and `tail: false`, two different
    Hamiltonians sampling different densities, hashed to the same id and would
    resume each other. A live collision, found by the partition test below.
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
    # 3D pmf g3 floor. PHYSICS by the rule at the top of this file: it does
    # not move the sampled ensemble (the bias is divided back out), but it
    # changes which trial gets selected, so two runs differing here are two
    # different proposal distributions and must not share a run directory.
    # None (the default 1/n_trials) and an explicit 1/n_trials are the same
    # floor but NOT the same identity, and that is correct: the second says
    # the floor was pinned deliberately.
    "min_g3",
    # WHICH READER the 3D pmf table is read with. DERIVED from the auxiliary
    # (there is one reader, `auxiliary.G3_READER`) and not a config key, but
    # identity-bearing all the same, which is the point of it: the reader
    # changed from interpolating four tabulated corners to taking the nearest
    # tabulated point, and the two select different trials. Neither can move an
    # equilibrium average, because the bias is divided back out of the
    # acceptance -- but they are two proposal distributions, and without this
    # marker the run before the change and the run after it would share a run id
    # and resume each other. None where there is no table to read (2D, or no
    # pmf auxiliary), so nothing but a 3D pmf run sees it at all.
    # THE BIAS CUTOFF, 3D pmf only. Physics by the same rule as `min_g3` and
    # `gamma`: it cannot move an equilibrium average, because the bias is
    # divided back out, and it changes which trial is selected. None means every
    # neighbour inside the potential cutoff contributes, which is what every run
    # before this key did, so no pre-existing run id moves.
    "g3_r_cut",  # None | radius in diameters
    "g3_reader",  # None | the value of auxiliary.G3_READER
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
    # THE DYNAMICS AXIS, and it is not the same axis as `sampler`.
    #
    # `sampler` says how a trial cloud is PROPOSED; `engine` says what is then
    # done with it -- a Metropolis accept/reject test (`metropolis_dynamics`) or an
    # exact conditional draw over the whole window (`glauber_dynamics`). Two
    # runs differing here sample the same muVT distribution through different
    # transition kernels, which is the textbook case of a field that is physics
    # by the rule at the top of this file: the ensemble is invariant, the Markov
    # chain is not.
    #
    # ADDING IT MOVES EVERY RUN ID, and that is accepted rather than worked
    # around. `hash_fingerprint._IDENTITY_FIELDS` derives from the CLASSES of
    # the physics sub-configs on purpose, so a new field in this group reaches
    # the digest without anyone remembering to list it -- which is the whole
    # design. Excluding the default to preserve old ids would mean fighting the
    # PHYSICS / config_specs / hash_fingerprint triangle that exists to stop
    # hand-maintained identity lists. Existing Metropolis params files still
    # validate and run correctly; they start a fresh run directory rather than
    # resuming an old one, and resuming a pre-`engine` run is explicitly not a
    # requirement. Precedent: `upper_cutoff` and `no_neigh` moved ids the same
    # way.
    "engine",  # metropolis_dynamics | glauber_dynamics
    # The Glauber engine's ONLY new tunable. Everything else about the window --
    # the pool size M and the truncation kmax -- derives (see DERIVED below).
    "kmax_margin",
    "depth",
    "full_repacking",
    "n_trial_molecules",
    "trial_density",
    "trial_box_side",
    "cavity_radius",
    "move_factor",
    "channel_length",
    "channel_width",
    "channel_cyclic",  # R16: derived from `sampler`, kept for the reduction check
    "radius",  # hard-core radius; `diameter` is its derived twin
    # ADAPTIVE GAMMA. `fixed` reads the nine fitted exponents in
    # `gamma_dict`; `adaptive` solves one per Rosenbluth level from the trial
    # weights to hit `pr_target`, a target participation fraction PR / k.
    #
    # PHYSICS by the rule at the top of this file, and for the same reason
    # `min_g3` is: neither moves the sampled ensemble, because the bias is
    # divided back out of the acceptance, and both change which trial is
    # selected. Two runs differing here are two proposal distributions and must
    # not share a run directory. That is also why `pr_target` cannot be tuned
    # from a run's own acceptance statistics: a constant of the run is a
    # constant, and a mid-run update makes the kernel a function of chain
    # history.
    #
    # SAMPLER and not POTENTIAL, although the bias it tunes is a potential key.
    # The exponent is a property of the SELECTION, which is what `sampler`
    # names: the same g3 table under two targets is one potential proposing two
    # different ways.
    # None (absent) and "fixed" are the same arithmetic and different run ids,
    # the convention `min_g3` set: absent keeps every pre-existing run id,
    # because `identity_dict` drops a None physics field, while pinning the
    # default says it was pinned on purpose.
    "gamma_mode",  # None (absent) | fixed | adaptive
    "pr_target",  # adaptive only: target PR / k, in (0, 1]
    # WHICH SPREAD the target is measured with: the participation ratio (a
    # second moment) or perplexity (an entropy). Physics for the same reason
    # `pr_target` is -- the same target under two criteria selects differently,
    # so the two are different proposal distributions. None on a fixed-gamma
    # run, where nothing reads it.
    "gamma_criterion",  # None | participation | perplexity
    # WHICH SOLVER returns the exponent: `moments`, the second-order closed
    # form, or `grid`, the 17-point search. Physics for the third time by the
    # same argument: the two land within about a percent of each other on the
    # target and neither can move an equilibrium average, but a percent in gamma
    # is a different trial selected, so two runs differing here are two proposal
    # distributions. Keeping it identity-bearing is also what makes the pilot
    # comparison legible: the two arms cannot land in one run directory and be
    # averaged together by accident. None on a fixed-gamma run.
    "gamma_solver",  # None | moments | grid
)

ENSEMBLE: tuple[str, ...] = (
    # muVT | NVT | semigrand | QA | confined. It carries the CANONICAL vs
    # SEMI-GRAND distinction that `fixed_composition` used to carry as a
    # separate boolean: at fixed N, NVT holds every species count and
    # semigrand holds only the total, and those are different chains sampling
    # different distributions, so the name is physics.
    "ensemble_type",
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
    # MEASUREMENT, not physics: it adds counters to the window summary and
    # changes nothing the chain does, so it belongs here and never reaches the
    # run fingerprint. Two runs differing only in whether ESJD was recorded are
    # the same experiment and must share a run directory.
    "esjd",
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
    # THE GLAUBER ENGINE's pool size M and truncation kmax (section 1.3 of
    # the design plan). DERIVED, and the plan's phrase "identity-bearing"
    # is satisfied through their INPUTS rather than by hashing them here.
    #
    # The plan is right that both change the chain, so two runs differing in
    # either must not share a directory. They cannot differ without one of
    # `trial_density`, `channel_length`, `channel_width`, `dimension`, `radius`
    # or `kmax_margin` differing -- every one of which is hashed -- because
    #
    #     window_M    = round(trial_density * V_W)
    #     window_kmax = min(window_M, ceil(packing_bound(body)) + kmax_margin)
    #
    # is a function of exactly those. Hashing M and kmax as well would
    # double-count their inputs, which is the specific failure this file was
    # written to end (`sampler` reached the old digest four separate ways
    # through this group). So the identity requirement is met and the triangle
    # is not fought; what makes it a REQUIREMENT rather than a hope is
    # `tests/unit/test_glauber_config.py`, which asserts the derivation and that
    # neither is a function of occupancy.
    "window_M",
    "window_kmax",
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

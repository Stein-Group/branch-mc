# S2 progress: the mechanical restructure

Commit 1 of S2. Pure moves and rewrites; no algorithm changed.

## The overlap convention moves to layer 1

`overlap_radius`, `overlaps`, `pair_overlaps`, `max_core_floor` and `OVERLAP_TOL`
left `gcmc/geometry.py` (L2) for a new `gcmc/potentials/overlap.py` (L1).

**Why, and why it is a correction to the plan.** Three layer-1 modules --
`species`, `pair_table`, `mof_field` -- had to reach UP into layer 2 to use these.
That was three of the four upward imports in the whole package, and the layer
ladder test is what surfaced it.

The resolution is not to permit the exception. It is that these were never
geometry: every one is a function of SIGMA. `overlap_radius` is `sigma - tol`,
`pair_overlaps` compares against `c * sigma_ij` from the pair table,
`max_core_floor` reduces that same table. Nothing in any of them knows about a
box, a cell list, a minimum image or a wrap -- which is what layer 2 is for.

`restructure_plan.md` section 5 placed `overlap.py` at L2. That was wrong, and it
is corrected here rather than in prose.

The split is now clean, and no module reaches up:

    L1  potentials.overlap   owns the FLOOR: what counts as an overlap
    L2  geometry             owns the DISTANCE: how far apart two things are
    L3  system               applies one to the other

## What the lint found on the way

`geometry.py` still carried its own `OVERLAP_TOL = 1e-12` after the import was
added -- F811, a redefinition. Two copies of the constant that the whole
convention is built on, which is precisely the duplication the move exists to
remove. Deleted; there is now one.

## Guards updated, not deleted

Two tests pinned `geometry` as the owner of the convention. Both were kept and
repointed, because their intent is unchanged and still valuable:

* `test_overlaps_table_delegates_rather_than_reimplementing` monkeypatches the
  central helper and requires the table wrapper to break with it -- a
  behavioural proof that the wrapper is a wrapper. Now patches
  `overlap.pair_overlaps`.
* `test_the_mof_layer_imports_geometry` -> `..._imports_the_overlap_convention`.
  Both MOF overlap sites were independent copies until P3; if one loses the
  import it has almost certainly grown a copy back. Now asserts the `overlap`
  import.

## A correction to an earlier finding

The first review listed `max_core_floor` as "defined in both geometry.py and
species.py". That was a name collision, not duplicated logic: `species` had a
one-line PROPERTY delegating to the geometry function. The delegation was
correct; only its direction across the ladder was wrong.

## Verification

    ruff                 0 errors, 0 unformatted
    mypy                 0 errors
    full suite           603 passed, 41 skipped, 2 xfailed
    layer ladder         5 passed -- and test_no_stale_allowances confirms the
                         three retired inversions are genuinely gone
    matrix digests       15/15 UNCHANGED

That last line is the gate that matters. A mechanical commit must not move a
single counter stream, and none moved.

    KNOWN_INVERSIONS  4 -> 1
    geometry.py       385 -> 335 lines

The one remaining inversion is `channel_move` (L4) -> `samplers.channel` and
`samplers.channel_cyclic` (L5): the P4 extraction still importing from what it
was extracted from. It clears when the shared pieces move down into
`channel_move`, later in S2.

## Next in S2

- `channel_move` inversion: shared pieces down into L4
- `Config` split into five sub-configs (R8) -- the big one
- fingerprint rebuilt as a structural partition, collapsing
  `test_identity_partition` to a two-line assertion
- file moves into the L0-L7 tree
- docstring contracts as each file moves

---

# S2 commit 2: body geometry moves to layer 2

## What moved

Six functions and a lookup table left `gcmc/samplers/channel.py` (L5) for a new
`gcmc/body.py` (L2), 358 lines:

    random_rotation_matrix   a uniform orientation
    local_from_box           box coordinates -> body-local coordinates
    ray_box_chord            where a line through the box meets the walls
    axial_coord_full         a particle's coordinate along a wall-to-wall rod
    cuboid_members_cl        which slots lie inside an oriented cuboid
    rod_members_full_cl      which slots lie inside a wall-to-wall rod
    _IMAGES                  the 27 periodic image offsets, precomputed

None of them ever knew anything about sampling. They answer purely geometric
questions -- where is the body, which particles are inside it -- and their
placement in the sampler was what forced `channel_move` (L4) to import from L5.

`samplers/channel.py`: 1,265 -> 964 lines.

## One inversion remains, and it is deferred on purpose

`channel_move` still imports the sweep CLASSES. `ChannelBody.sweep()` is a
factory picking one of four, which is a layer-5 decision inside a layer-4 module.

That is left for S5, not overlooked. R7 collapses those four classes (six,
counting the multi-species pair) into ONE dynamic program selected by a mask
policy. Afterwards `sweep()` needs no sampler class at all -- it hands a policy
object down, and the edge disappears. Building a seam now that the collapse then
removes would be work done twice. The `KNOWN_INVERSIONS` entry says S5 rather
than S2, so the allowlist records the real plan.

## The characterisation script had to flip its contract

`check_channel_full.py` asserted that seven spellings of `channel_length`
("cyclic", "CYCLIC", "full", "box", "full_box", "span", "FULL") all selected the
cyclic rod. R16 inverted that: they must now be REFUSED, with a message naming
the replacement.

The test was flipped rather than deleted, because the new contract needs a guard
as much as the old one did -- a silent alias would leave the old surface alive in
every params file that still uses it. Three inline params dicts in the same
script were converted to `sampler.type = "channel_cyclic"`.

## Verification

    ruff                 0 errors, 0 unformatted
    mypy                 0 errors
    full suite           603 passed, 42 skipped, 2 xfailed
    matrix digests       15/15 UNCHANGED
    characterisation     5 passed (channel_full, decomposition, cli)
    KNOWN_INVERSIONS     4 -> 1

Two mechanical commits, two clean digest checks. The layering is now:

    L1  potentials.overlap   the floor
    L2  geometry             distances
    L2  body                 oriented bodies and their membership
    L4  channel_move         the proposal
    L5  samplers.channel*    the sweep

---

# S2 commit 3: identity becomes structural (R8, first half)

R8 splits `Config` into five sub-configs. That rewrites every `cfg.X` access in
16,000 lines, so it is TWO commits, and this is the first: the grouping is
declared, the identity set is derived from it, and the whole thing is tested
against the real `Config` before any call site moves.

## `gcmc/config_fields.py`

The partition, as PACKAGE data rather than test data. Five groups --
`POTENTIAL`, `SAMPLER`, `ENSEMBLE` (= `PHYSICS`), `EXECUTION`, `IO` -- plus
`DERIVED`. All 72 Config fields classified; total, disjoint, no phantoms.

`hash_fingerprint._IDENTITY_FIELDS` is now literally `config_fields.PHYSICS`.
A field is identity-bearing because of the group it was declared in, not because
someone remembered to add it to a tuple. That indirection is the entire fix: all
eight findings from the S1 audit were produced by maintaining a list.

These groups ARE the sub-configs. When the dataclasses land, each tuple becomes
`dataclasses.fields(...)` of the corresponding class and the file shrinks to five
one-line expressions.

## Four special cases become one rule

`identity_dict` carried bespoke skip logic for `channel`, `aux_cutoff`,
`box_sides` and `species_identity`, each so runs predating a feature kept
byte-identical ids -- while EIGHT other inactive keys hashed as literal `null`,
because nobody had written a special case for them. Two conventions, applied by
accident of history.

R9 retires the compatibility requirement, so the rule is now the one that was
always correct: **an inactive key is absent, never null.** `False` and `0.0` are
not None and still hash, which is what keeps the `tail` collision fixed.

    keys hashed on an ordinary LJ run:  27
    nulls:                               0

## What the partition caught

Placing `species_identity` was not obvious, and getting it wrong would have been
severe. It is a computed PROPERTY, which suggested DERIVED -- but it is the
canonical hashable form of the FORCE FIELD: a digest over sigma, epsilon, k_ij,
the core floor, rc, the auxiliary multiplier and the per-species state point. In
DERIVED, two different force fields would have shared a run directory.

It is PHYSICS. The `species` object it summarises is DERIVED, because hashing an
object is not meaningful and hashing its digest is.

Also caught: four names invented in the S1 test version (`log_every`, `resume`,
`verbose`, `species_file`) that are not Config fields at all, and three that are
properties rather than stored fields. `CF.PROPERTIES` keeps those cases apart --
"I invented this name" and "this is computed, not stored" are different mistakes
with different fixes, and a test that conflates them tells you neither.

## Discrimination re-verified

Thirteen probes, one parameter at a time:

    MUST DIFFER   tail, T, mu, depth, trials, trial_box_side, sampler,
                  potential, dimension, box, seed        -- all differ
    MUST MATCH    sampling_window, checkpoint_every      -- both match

## Two tests shrank

`test_identity_partition.py`: 297 -> 165 lines. It no longer restates the
partition; it imports it and asserts properties OF it. The
`test_identity_equals_the_physics_half` xfail is gone -- true by construction.

`test_the_untyped_fingerprint_does_not_gain_a_null_key` stopped pinning a literal
25-name key set (which would have been a THIRD copy of the partition) and now
asserts the property it is named for: no key is null, and the hashed set is a
subset of `config_fields.PHYSICS`.

## Verification

    ruff                 0 errors, 0 unformatted
    mypy                 0 errors
    full suite           604 passed, 43 skipped, 1 xfailed  (was 603 / 2 xfailed)
    matrix digests       15/15 UNCHANGED
    identity partition   5 passed, no xfail

Run ids moved, by design (R9). Chains did not -- the digests prove it. That
separation is exactly what a fingerprint rebuild should look like.

Digest provenance, recorded so it is never lost again:

    1a24fe1c17aa   pre-S1, origin unknown (the failure this practice prevents)
    fbdc3125c74a   pre-S1
    0c683de8d49b   S1: lj_tail_pref + pore_axis_len added
    e864f492d3ba   S2: identity set becomes config_fields.PHYSICS

---

# S2 commit 4: the five sub-config dataclasses (R8, second half)

## Why a projection and not the split

Turning `Config` from a flat 72-field dataclass into a composition touches, all
at once: 278 attribute reads in the package, three `dataclasses.replace` call
sites, the checkpoint round-trip, and the field partition test. That is the
big-bang R15 exists to avoid, and it is the largest single change in the plan.

So `gcmc/config_specs.py` creates the five classes NOW as read-only projections
of an unchanged `Config`. That buys the two things that actually matter:

* **Modules get something narrow to depend on.** A layer-1 function can take a
  `PotentialSpec` instead of the whole `Config` -- which is what breaks the
  god-object coupling -- and can do so one module at a time, each migration its
  own commit with its own gate.
* **Identity becomes typed.** `hash(potential + sampler + ensemble)` is three
  objects, not a filtered name list.

Nothing here is torn out later. When `Config` composes rather than projects,
these same classes become the members it holds and the projection functions
become one-line accessors. Same direction, smaller step.

## Drift is impossible by construction

Each class is checked against its `config_fields` group **at import**, not in a
test. The two are one declaration written twice, and a mismatch is never
intentional; an import-time check catches it for anyone who loads the package,
not only for someone who runs the suite.

## The round-trip gate

`Config -> five sub-configs -> Config` must reproduce every stored field, and it
is asserted across the whole pairwise cover rather than one hand-picked config.
That matters: on any single mode most fields are None, and None survives any
bug. The cover exercises pore geometry, matrix coordinates, typed species,
auxiliaries, parallel cells and both rod kinds.

    39 passed

Two further tests pin the triangle. `test_identity_is_exactly_the_three_physics_specs`
asserts the hashed set equals the union of the three physics CLASSES -- so the
fingerprint, `config_fields` and `config_specs` are each pinned to the other two,
and any two agreeing while the third drifts is caught. Its complement,
`test_execution_and_io_never_reach_the_identity`, guards the more dangerous
direction: an execution field leaking into the identity would split one
experiment across many directories and quietly break resume, and would look like
nothing worse than a lot of run folders.

## A fixture had to move

`matrix_dir` lived in `tests/matrix/conftest.py`, and a conftest is visible only
to its own directory and below -- so `tests/unit` could not see it. Duplicating
the QA obstacle generator would have given two libraries of "the same" obstacles
that could silently diverge. It moved up to `tests/conftest.py`; the matrix
conftest keeps only its CLI options.

## Verification

    ruff                 0 errors, 0 unformatted
    mypy                 0 errors
    full suite           643 passed, 44 skipped, 1 xfailed
    matrix digests       15/15 UNCHANGED
    round trip           39 passed across the pairwise cover

---

# S2 commit 5: `hash_fingerprint` migrates onto the sub-configs

## A correction to the original review, first

I claimed in the first review that `Config` being a 72-field flat dataclass meant
"every module takes `cfg`, so every module transitively depends on the union of
all configuration." Measured:

    gcmc/potentials/pair_table.py     0 cfg. accesses
    gcmc/species.py                   0
    gcmc/neighbors.py                 0
    gcmc/geometry.py                  0
    gcmc/body.py                      0
    gcmc/trials.py                    0
    gcmc/moves.py                     0
    ---
    gcmc/hash_fingerprint.py         11
    gcmc/parallel.py                 31
    gcmc/MC_engine.py                93
    gcmc/runner.py                   98

**Layers 1 to 4 already take no Config at all.** The coupling is entirely in L6
and L7, which is where a broad config legitimately belongs. The claim was too
strong and the layers below the engine were already clean.

R8 is still worth doing, but its justification narrows honestly:

1. identity becomes structural (delivered, commit 3)
2. the engine, runner and parallel path get readable, narrow signatures
3. reading 5 named groups beats reading 72 flat fields

What it is NOT is a prerequisite for separating the lower libraries. Those are
already separate.

## The migration

`hash_fingerprint` goes from 11 `cfg.` accesses to zero.

`identity_dict` now iterates the FIELDS OF THE THREE SPEC OBJECTS rather than
getattr-ing names off `Config`. The name list is gone from the runtime path
entirely: identity is the fields of three dataclasses, and `_IDENTITY_FIELDS` is
derived from the same classes for the tests that still want names.

`run_paths` reads an `EnsembleSpec` and an `IOSpec`. That split is the point --
a directory name is built from the state point (what the run IS) plus one
presentational label, and nothing else. Eleven scattered `cfg.` reads left that
to be inferred; two narrow objects state it.

Six `getattr(cfg, name, default)` calls disappeared with it. Those were defensive
reads against a field that might not exist -- on a typed dataclass, it always
does.

## Verification

    ruff              0 errors          mypy  0 errors
    full suite        643 passed, 44 skipped, 1 xfailed
    matrix digests    15/15 UNCHANGED
    run ids           UNCHANGED (e18e3e01f3b3 before and after)

The run id holding is the interesting one. Switching from name-getattr to spec
objects could have reordered the digest input -- `sort_keys` is what makes it
invisible, and that was checked rather than assumed.

Directory naming verified across muVT, NVT and confined:

    runs/afc87e1c8fce__cyl_R3

## Next

`parallel` (31 accesses) is the next tractable migration, then the engine and
runner. After those, `Config` can compose rather than project and the flat field
list disappears.

---

# S2 commit 6: the sixth sub-config

R8 names five sub-configs. Migrating the first kernel onto them found that five
is not enough, and the shortfall is structural rather than an oversight in the
plan.

## What the count showed

`parallel.py` reads `cfg` 36 times. Classified against `config_fields`:

    POTENTIAL / SAMPLER / ENSEMBLE / EXECUTION / IO      25
    DERIVED                                              11

Eleven reads -- `beta`, `two_dim`, `r_squared`, `upper_cutoff`, `no_neigh`,
`total_volume`, `potential_type`, `box_dims`, `cell_dims` -- have no home in any
of the five. They are not obscure: they are what the hot loop actually computes
with. `beta` rather than `temp`, `r_squared` rather than `rc`, a resolved
per-axis `box_dims` rather than a scalar `box` beside a usually-None
`box_sides`.

So the five describe what the user SUBMITTED, and nothing described what the
kernel was HANDED. A kernel wanting the second had exactly one option: take the
whole `Config`. That is the god-object coupling R8 exists to break, reappearing
one layer down -- and it would have blocked the engine and runner migrations too,
harder, because both read more derived fields than `parallel` does.

## `KernelSpec`

A sixth frozen dataclass mirroring `config_fields.DERIVED`, on the same
import-time check as the other five, so the six-way partition is stated once and
verified once. Never identity-bearing, and not by convention: every field in it
is a function of a field already hashed elsewhere, so hashing one would
double-count its own inputs -- the exact defect `config_fields` was written to
end, where `sampler` reached the old digest four separate ways.

## Two properties that were classified nowhere

`box_dims` and `cell_dims` were in no group at all. Not a deliberate omission --
a blind spot in the totality test, which enumerates `dataclasses.fields(Config)`
and therefore cannot see a property. Both are unambiguously derived, and both are
now in `DERIVED` and in `PROPERTIES`.

Worth recording as a limit of that gate: `test_partition_is_total` proves every
stored FIELD is classified. A property is invisible to it in both directions,
and the only thing that surfaced these two was a module trying to read them.

## One assertion got stronger

`test_the_five_cover_every_stored_field` carried a `- set(CF.DERIVED)` exemption,
because no spec carried those. `KernelSpec` does, so the exemption is gone and
the test now says what it wants to say: the six specs cover the stored half of
`Config` with nothing exempted. An exemption is a place a field can hide.

## Verification

    ruff              0 errors, 0 unformatted
    mypy              0 errors
    full suite        578 passed, 47 skipped, 1 xfailed  (was 576; +2 new tests)
    matrix digests    15/15 UNCHANGED

---

# S2 commit 7: the two shared helpers, and where the specs belong

`parallel.py` cannot stop taking `cfg` while it forwards `cfg` to
`MC_engine._weighted_trials` and `MC_engine.make_weigher`. So those move first.

    make_weigher       2 cfg. accesses -> KernelSpec
    _weighted_trials   4 cfg. accesses -> KernelSpec + PotentialSpec + EnsembleSpec

Three specs for four reads looks like overkill until you see which four:
`potential_type` and `two_dim` are derived, `gamma_dict` is a potential knob,
`box` is an ensemble one. The signature is wider on paper and states a
dependency that eleven scattered `cfg.` reads left to be inferred -- and the
dependency turned out to be four fields out of 72.

`make_weigher` is called from 26 sites, mostly tests and characterisation
scripts; all now pass `config_specs.kernel_spec(cfg)`. That churn is the one-time
price of narrowing a shared helper, and it is paid once.

## The ladder moved the specs, not the engine

`MC_engine` (L6) importing `config_specs` (L7) failed `test_imports_only_go_down`
immediately. The fix is not an allowlist entry -- it is that the specs were
placed wrongly.

`config_fields` and `config_specs` sat at L7 on the reasoning "they describe
`Config`, and `Config` is L7". True, and beside the point. What they ARE is the
vocabulary in which a layer states which slice of configuration it needs, and a
vocabulary only works if every layer can speak it. That is the definition of L0.

They qualify structurally, not just rhetorically: `config_fields` imports nothing
in-package, `config_specs` imports only `config_fields`, and the `Config` they
are typed against is a TYPE_CHECKING forward reference with no runtime edge.
Nothing had to be bent.

This is the third time the ladder has corrected a placement the plan stated in
prose (after `overlap` at L1 and `body` at L2), and the pattern is consistent:
the plan reasons about what a module is ABOUT, the ladder reasons about what it
DEPENDS ON, and where they disagree the ladder has been right every time.

## One test was rewritten, and is stricter than before

`test_layer_zero_imports_nothing` said "L0 depends on nothing", which was
indistinguishable from "L0 depends on nothing above it" while L0 held one module.
It now reads as the second, and the guarantee the blanket ban gave for free -- L0
cannot be part of a cycle -- is asserted directly by a new
`test_layer_zero_is_acyclic` rather than approximated. Strictly more is checked.

This is deliberately NOT a weakened gate. The old rule forbade a real, needed
edge (`config_specs` -> `config_fields`) as collateral; the new pair forbids
everything the old one forbade that actually mattered, plus cycles.

## Verification

    ruff              0 errors, 0 unformatted
    mypy              0 errors
    full suite        581 passed, 45 skipped, 1 xfailed
    matrix digests    15/15 UNCHANGED
    KNOWN_INVERSIONS  1 (unchanged -- channel_move, still S5)

---

# S2 commit 8: `parallel.py` migrates, and the annotations were wrong

    gcmc/parallel.py   36 cfg. accesses -> 0

## Where the boundary is, and why it is there

`advance_parallel` KEEPS `Config` in its signature. It must: `runner` binds
either it or `MC_engine.advance` to the same name and calls whichever it bound,
so the two signatures are one signature. It is the projection boundary -- `cfg`
is read there and nowhere below.

`advance_cell` and `_run_cell` take four specs: `KernelSpec`, `SamplerSpec`,
`PotentialSpec`, `EnsembleSpec`.

That boundary is where the narrowing pays for itself, and the reason is specific
to this module rather than general tidiness. Every task tuple is PICKLED to a
separate process. `cfg` was not merely a wide parameter; it was a whole `Config`
-- species tables, matrix coordinates, IO paths -- serialised once per cell per
window, to be read for about twenty numbers. `IOSpec` and `ExecutionSpec` now do
not cross the process boundary at all, which is the structural statement that a
cell kernel cannot depend on where the run writes or how many cores it has.

Commit 5's correction still stands: L6/L7 are where a broad config legitimately
belongs, and this is not decoupling for its own sake. The win here is the worker
payload and a readable signature, not a dependency that needed breaking.

## The annotations had drifted on 24 of 72 fields

mypy caught it the moment a kernel actually consumed a spec:
`ExecutionSpec.n_workers` is `int | None`, `Config.n_workers` is `int`.

Checked systematically, 24 of the 72 spec annotations disagreed with `Config`.
The import-time check pins NAMES -- which is what keeps a field inside the
identity -- and says nothing about types, so the types, written by hand once,
were never checked by anything.

Eighteen were widened to `| None` where `Config` promises a value. Those fail
nothing; they make the next migration write a dead None branch or reach for
`# type: ignore`, and once that habit exists a genuine optional slips through
behind it. Six had drifted the dangerous way, narrowing what `Config` allows:

    n_cell           Config: int | list[int] | None   spec: int | None
    lj_tail_pref     Config: float | None             spec: bool
    rc               Config: float | None             spec: float
    target_count     Config: float | None             spec: int | None
    target_cycles    Config: float                    spec: int | None
    log_flush_interval  Config: float                 spec: int | None

`n_cell` lost the per-axis `list[int]` the launcher documents. `lj_tail_pref` --
a float coefficient multiplied into `delta_E` -- was declared `bool`; this commit
reads it in `parallel.advance_cell` as exactly that float.

All 24 corrected, and `test_spec_annotations_match_config` now pins spec
annotations to `Config`'s. It lives in the test rather than beside the name
check because it needs `Config`, and `config_specs` is L0 while `sim_setup` is
L7 -- the ladder refuses that edge, function-local import or not. (It found that
too, on the first attempt at putting it in the package.)

## Verification

    ruff              0 errors, 0 unformatted
    mypy              0 errors
    full suite        582 passed, 45 skipped, 1 xfailed
    matrix digests    15/15 UNCHANGED
    tier 2 matrix     21 failed, 30 passed, 21 skipped -- identical to baseline,
                      same four root causes at the same counts

Three mechanical commits, three clean digest checks. Not one counter stream moved
across the whole of commits 6-8.

## Next

`MC_engine` (86 accesses, was 92) and `runner` (110). Both now have a complete
vocabulary to migrate onto, which they did not before commit 6.

---

# S2 commit 9: the metropolis+pmf guard moves (S2 item 7, M15)

The fourth of the four bugs the generated matrix found on its first run, and the
only one S2 owned.

`sim_setup.from_params` ACCEPTED hard_core + metropolis + a pmf auxiliary;
`MC_engine._advance_metropolis` refused it with `NotImplementedError`. So the
combination passed validation, created a run directory, opened a log, began the
first window and then raised. M15 says fail at the earliest layer that can see
the problem, and the validator can see this one: it is three params-file keys.

Only the HARD-CORE pmf auxiliary is refused. LJ + pmf (`ljaux`) is supported
under metropolis and is not an approximation -- with a single trial the selection
bias cancels identically, so metropolis + pmf IS metropolis on the plain LJ
potential, and the engine unwraps it to exactly that. The refusal message names
`rosenbluth` as the replacement rather than saying "not implemented".

The engine's guard became an assertion rather than being deleted.
`_advance_metropolis` is also called directly by characterisation scripts that
build a `Config` by other means, and for those a loud failure still beats a
silent wrong answer.

    tier 2:  21 failed -> 19 failed.  Three root causes remain, all scheduled:
             swap (S3), reweight kdtree (S5), pmf-aux kdtree (S5).

## And the gate this broke, which matters more than the fix

Refusing those params removed 13 points from `accepted()`. All 13 are genuinely
hard_core + metropolis + aux_pmf; the guard is exactly as narrow as intended.

But the digest gate is parametrised on the PAIRWISE COVER, and the cover is
chosen greedily from `accepted()`. Removing 13 points re-ran the greedy choice,
which then selected 36 different points -- of which exactly 2 had digests.

    before:  15 passed, 21 skipped
    after:    2 passed, 34 skipped

Nothing failed. Every skip said "no committed digest; run --matrix-regen", which
reads like new features awaiting baselines rather than like thirteen existing
pins having silently stopped being checked. A commit could have moved any of
those thirteen counter streams and the gate would have said nothing.

**"digests 15/15 UNCHANGED" was only ever true while `accepted()` was
unchanged** -- and S2 item 7 is, by construction, a change to `accepted()`. The
protocol that has carried eight commits had a hole in it at exactly the commit
that needed it most.

The fix separates what tier 2 RUNS from what tier 2 PINS:

    point         the pairwise cover. SUPPOSED to move when accepted() moves.
    digest_point  the cover UNION every id in digests.json. Must not.

Every committed digest is now checked on every run, whatever the cover selects,
and the gate is independent of the validator. A digested point that stops being
accepted FAILS by name -- carried through collection as a `_MissingPoint` so it
cannot simply vanish -- with a message saying that retiring a mode means
deleting its digest in the same commit and saying so.

    restored:  15 passed, 34 skipped   (the 34 are new cover points with no
                                        digest yet, which is the honest state)

Nothing was regenerated. The same 15 digests from before commit 6 still hold,
and the record that chains have never moved is intact.

## Verification

    ruff              0 errors, 0 unformatted
    mypy              0 errors
    full suite        582 passed, 45 skipped, 1 xfailed
    matrix digests    15/15 UNCHANGED  (and now actually checked)
    tier 2            19 failed, 32 passed -- one fewer root cause

---

# S2 commits 10-12: the engine and the runner migrate (S2 items 2 and 3)

    gcmc/MC_engine.py   86 cfg. accesses -> 0
    gcmc/runner.py     110 cfg. accesses -> 1
    ---
    package            278 (measured at commit 4) -> 20, of which 19 are in
                       sim_setup itself, which owns Config

## The one that stays, and why it is not a miss

`runner` keeps `cfg.build_potential()`. That is a METHOD, not a field: it reads
the whole configuration to instantiate the interaction object, and no sub-config
is the right receiver for it. `run()` also still hands `cfg` to
`MC_engine.advance` / `parallel.advance_parallel`, which take a `Config` by
design (see commit 8 -- `runner` binds one or the other to the same name).

That is the honest shape. R8 was never about eliminating `Config`; it was about
making every module state which SLICE it needs. The two places that genuinely
need all of it still say so.

## `run()` projects twice, deliberately

`cfg` is rebuilt up to three times inside `run()` --
`dataclasses.replace(cfg, matrix_positions=...)` while the frozen matrix is
resolved. A single projection taken before that would carry
`matrix_positions=None` into every reader downstream: a QA run sampling with no
obstacles, which looks like a physics result rather than a bug.

So there is one projection immediately after `from_params` (used only to resolve
the matrix) and a second after the replaces settle. A projection is a snapshot of
a frozen object; re-taking it is how a snapshot follows a rebuild. Anything that
holds a spec across a `replace` is holding a stale one, and that is a general
hazard worth stating once here rather than rediscovering at S2 item 4.

## Eleven defensive `getattr(cfg, name, default)` reads disappeared

Six in `advance`, five in `runner`. Each guarded against a field that might not
exist; on a typed dataclass it always does, so the default was dead code -- and
worse than dead, because it would have swallowed a genuine rename instead of
failing on it. Commit 5 removed six of these from `hash_fingerprint` for the same
reason; this is the rest of them.

---

# S2 commit 11 (folded in): three copies of the pore predicate become one

Found while migrating the pore helpers, not looked for:

    MC_engine._inside_pore     the live array/point predicate        (L6)
    runner._inside_pore_seed   a verbatim single-point duplicate     (L7)
    MC_engine._pore_bounds     the same bounds a third time -- and DEAD,
                               called by nothing anywhere in the tree

Two live copies of a hard-wall test do not announce themselves. They agree until
one is edited, and the symptom of divergence is seeded configurations sitting
inside a wall that the engine then refuses to move -- which presents as a
sampling problem, not a geometry one. `tests/unit/test_initial_config_grid.py`
was asserting the seeder agreed with its own copy of the rule, which it could
not fail to do.

One predicate now, at **L2 `geometry.inside_pore`** -- the ladder's own
instruction when a shared thing forces an upward import: move it down to a layer
both callers can see. It is also just true that a wall test asks where a particle
is and reads no energy, which is what that module is for. It takes
`EnsembleSpec` + `SamplerSpec`, which is the honest pair: a pore is ensemble
geometry, and the clearance that makes it a hard wall is the sampler's particle
radius.

`_pore_bounds` was deleted rather than migrated. Note for the LOC target: ruff
does not flag an unused private module-level function, so dead code of this shape
is invisible to tier 0 and only surfaces when someone greps for a caller.

## Verification (commits 10-12)

    ruff              0 errors, 0 unformatted
    mypy              0 errors
    full suite        582 passed, 45 skipped, 1 xfailed
    matrix digests    15/15 UNCHANGED
    tier 0            13 passed
    tier 2            19 failed, 32 passed -- unchanged, same three root causes

Six mechanical commits since the digest gate was repaired, and not one counter
stream has moved.

---

# S2 commit 13: typed + metropolis is refused, and a fifth bug

Not a bug the matrix found. The matrix reported these five points PASSING; this
one was found while reading the dispatch on the way to S3.

## What `_advance_metropolis` did with a species file

Three things, none of which raised:

1. **Every swap proposal executed as a deletion.** `moves.draw_move` returns
   `"swap"` for n_species > 1 (M8, 20% of proposals). The loop dispatched on
   `if choice == "move" / if choice == "add" / else:`, and the bare `else` was
   the deletion branch. Measured over one window: 15 removes and 14 swaps drawn,
   `remove_step` recorded 29. So a typed metropolis run deleted at twice the
   intended rate and never once changed a particle's identity -- meaning the
   composition could not equilibrate between species at all.

2. **The typed force field was ignored.** The pair energy is `lj_pair_energy`,
   which is sigma = epsilon = 1. Per-pair sigma, epsilon and k_ij from the
   species file are never read anywhere in the function.

3. **The species column desynced from the positions.** `state.species` is never
   updated alongside `state.positions`, so the first accepted insertion or
   deletion breaks the alignment: 43 positions against 40 labels after a single
   window.

Any one of the three makes the run meaningless. Together they produce a run that
completes, logs ordinary-looking counters, and yields an isotherm for a system
nobody configured.

## Refused, not patched

`sim_setup.from_params` now rejects `species_file` + `sampler.type='metropolis'`.

Fixing only (1) -- adding a swap branch -- was the obvious move and would have
been the wrong one. It removes the visible defect and leaves the two that produce
plausible numbers, which is a strictly worse state than the loud one: a swap
acceptance built on `swap_log_acceptance` would be exactly right in its
activities and applied on top of an energy that does not know the species exist.

The dispatch was still made total: `elif choice == "remove"` with a final
`else: raise`, matching `advance`. A dispatch over a closed set of strings must
enumerate the set. A catch-all on the last branch turns every future addition to
that set into a silent mis-execution of whichever branch happens to be last --
which is precisely how this bug was born, since `"swap"` was added to
`moves.MS_MOVES` and this loop was never revisited.

An assertion now pins the species-alignment invariant too, for a `Config` built
by a route other than `from_params`.

## The digest gate caught the consequence, four commits after it was fixed

Refusing the mode retired 13 points, two of which had committed digests.
`test_counter_digest` failed with:

    lennard_jones-metropolis-d0-3d-muVT-single-typed has a committed digest but
    the validator no longer accepts it.

Under the gate as it stood before commit 9 those two would have silently dropped
out of the run set and been reported as skips. Both are retired deliberately, in
this commit, with the cause recorded in the new
`docs/decisions/DIGEST_PROVENANCE.md` -- which is now the required companion to
any change to `digests.json`.

**One thing that record makes explicit, because it is easy to misread.** Those
two digests were stable across every commit of S1 and S2. Their stability was
never evidence of correctness: a wrong calculation reproduces exactly as
faithfully as a right one. "15/15 unchanged" says the engine is deterministic
under refactoring and says nothing at all about whether what it computes is
right. The two claims have been carried together through this whole stage and
they are not the same claim.

## `tests/unit/test_typed_config.py` moved to rosenbluth

Its 21 tests exercise typed config PARSING -- fingerprints, species files, state-
point routes -- and used metropolis only as a convenient valid sampler. The
untyped fixture in the same file keeps metropolis, which is the useful contrast:
the refusal is about the pairing, not about either half of it.

## Verification

    ruff              0 errors, 0 unformatted
    mypy              0 errors
    full suite        579 passed, 45 skipped, 1 xfailed
    matrix digests    13/13 UNCHANGED (2 retired, cause recorded)
    tier 2            16 failed, 30 passed -- two root causes left, both S5-owned
                      plus `swap` in `advance`, which is S3

---

# S5 (brought forward): the two "dead KD-tree features" are neither dead nor KD-tree

    tier 2:  16 failed -> 8 failed.  Only `swap` is left.

## The finding, which contradicts the handover

`HANDOVER.md` section 4 said:

> The two `kdtree` failures are one root cause: the cell-list migration removed
> the KD-tree and **neither reweight nor the pmf auxiliary was ever ported**.
> That reframes R5 -- "one neighbour structure" is not cleanup, it is the repair
> of two features that currently cannot run at all.

Measured: **the port is fine.** `CellList.data` exists for exactly this purpose
-- a slot-indexed coordinate table that `idxs` index the way they indexed a
kd-tree's, with the same pad-index clamp -- and both features run correctly
against a `CellList` the moment the system holds a single particle. Seeded with
30 particles, a psi6 reweight run advances a full window and returns ordinary
counters.

What was missing is the **N = 0 branch**. `MC_engine.advance` passes
`tree = cl if cl.n_active else None`, so an empty exclusion set -- no fluid left
and no frozen matrix -- reaches every weigher as `None` with fully padded
neighbour rows. The distance-only weighers have always handled that: an infinite
distance contributes zero energy and unit weight, which is why the ordinary LJ
and hard-core paths can start from an empty box. The two GEOMETRY-AWARE weighers
could not:

    reweighting.calculate_order_parameter   dereferenced kdtree.data
    Aux3D.weight_bulk                       asserted kdtree is not None

Every run starts from an empty box unless it is warm-started, which is why this
presented as "the feature cannot run at all". It can; it cannot BEGIN.

**This shrinks R5 rather than reframing it.** "One neighbour structure" is
cleanup after all, not the repair of two dead features -- and the eight matrix
points were one unhandled state, not two unported features. Worth correcting in
the record because R5's priority was raised on the strength of the larger claim.

## The fixes are the no-neighbour limit, not fallbacks

**psi6.** Zero phasor sums with zero counts. Not a chosen default: it is what
the existing convention already says, because `psi6_magnitude` divides by
`counts` and maps the resulting 0/0 to the 0.3 liquid floor -- exactly how an
isolated particle in a sparse system is already treated. A test asserts the two
cases agree, so the empty box cannot drift into a second rule.

**pmf g3.** The bias is a THREE-BODY factor over (r, cos theta) triplets. With no
neighbours there is no triplet, the product runs over an empty set, and an empty
product is 1.

The reference neighbour needed more care than the weight. It is UNDEFINED here,
and the fix returns NaN to say so. **Zero would have been worse than undefined:**
`weight_pair` builds `vec_BA = ref_coords - trial_coords`, so a zero reference is
not neutral -- it is a real direction pointing at the origin, and the per-pair
recompute would have applied a definite three-body bias derived from nothing.
That is a silent wrong answer of exactly the shape commit 13 was about, and it is
the obvious fix. `weight_pair` reads the NaN and takes the matching no-bias
branch, so the bulk weight and the recompute cannot disagree on the same move --
which is asserted, because if they did the acceptance ratio would be
inconsistent rather than merely wrong.

`tests/unit/test_empty_environment_weighers.py` pins all of it at the weigher,
so a reintroduced dereference fails there rather than somewhere inside an engine
loop.

## Verification

    ruff              0 errors, 0 unformatted
    mypy              0 errors
    full suite        583 passed, 45 skipped, 1 xfailed   (+4 new tests)
    matrix digests    13/13 UNCHANGED
    tier 2            8 failed, 38 passed -- ONE root cause left

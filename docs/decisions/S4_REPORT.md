# S4 report

S4's stated scope is P5 (the multi-species channel sweep with the auxiliary) and
P6 (structure in and out), with one addition written into the gate:

> **Gate S4** is Gate P5 and Gate P6 as written, with one addition: the auxiliary
> control arm must be an **exact** no-op, not statistically close. If it is not,
> the correction is inserted in the wrong place, which is the open Q6.
> -- `restructure_plan.md`

Phase 1 does that addition, because it is the only part of S4 that is checkable
before any new code exists, and because Q6 has been open since
`multispecies_plan.md` was written. It ended up also rewriting the tier-2 gate,
for reasons that are the substance of this report.

---

## Phase 1: the auxiliary control arm

### The claim that was already in the tree

`MC_engine.py` carries a VALIDATION RECORD comment block whose first line is:

> bit-identical to the no-auxiliary engine at aux_cutoff == upper_cutoff, over
> 2D/3D x fixed/cyclic rod x depth-2/full_repacking x 3 seeds

Four of the five silent defects found during S2 and S3 were claims in comments
that measurement contradicted, so this one was measured rather than accepted.
It holds -- and measuring it turned out to be worth doing for a different reason
than expected.

### Arm A: the plumbing arm, and why it settles nothing

Configure `auxiliary.type = "short_range"` with `cutoff == run.upper_cutoff`, run
it against the same params with no `auxiliary` block at all, and compare counter
streams and final positions byte for byte. Over 2D/3D x fixed/cyclic rod x
depth-1/depth-2/full_repacking x 3 seeds: **identical, every point.**

That is worth one test. The auxiliary block changes three things that could have
leaked into the chain -- `bias` becomes `"short_range"`, `aux_cutoff` becomes a
float, and the fixed rod's periodic guard swaps `upper_cutoff` for `aux_cutoff`
as its margin -- and none of them do.

But it says nothing about Q6. The engine sets

    aux_on = ljbranch and rc_sel < uc

so at `rc_sel == uc` the correction block is never entered. Arm A therefore
proves that the short-circuit short-circuits. The comment above that line says
so in as many words -- "the no-op case does no arithmetic at all and bit-identity
is structural" -- which is exactly what makes the control arm as written
vacuous with respect to the question it is supposed to settle.

### Arm B: the algebraic arm

The module is recompiled from its own source with that one line widened to
`aux_on = ljbranch and _aux_rc is not None`, so at the no-op radius the two shell
energies are really computed, the difference is really taken, and the term really
reaches `neg_log_acc`. The correction evaluated between 1 and 59 times per point
(read back out of the window summary's `aux_corr_n`, and asserted positive, so
the arm cannot go vacuous silently), returned exactly `0.0` with exactly `0.0`
spread every time, and the chain was **bit-identical to aux-off at all 30
points.**

**Q6 is settled, and this is what settles it.** An exact zero added in the right
place changes nothing. An exact zero added in the wrong place still changes the
acceptance, the RNG draw that follows it, or -- the specific hazard here -- the
`None`-ness of the tail term, since the insertion reads
`delta_E_lj = (delta_E_lj or 0.0) + delta_corr` and three call sites downstream
branch on `delta_E_lj is not None`. It is not wrong: `aux_on` implies `ljbranch`
implies `lj`, so the tail is always a float by then. That is now measured rather
than reasoned.

Exactness is the right demand rather than merely a strict one, and the reason is
in `_lj_set_energy_split`: both radii are applied to ONE neighbour query, as two
`_lj_energy` calls over the same array in the same order. At `rc_short ==
rc_full` those are the identical call, so the difference is `0.0` and not
`1e-17`. A tolerance here would have been the wrong instrument.

Both arms live in `tests/engine/test_aux_noop_control.py`, 60 points, 9 seconds.
Two axis cells are deliberately absent and the file says why: 2D +
`full_repacking` evaluates the correction zero times at that density, so
including it would add green rows that test nothing.

### The recompile, and why it is not scaffolding in the engine

`aux_on` is a local, so it cannot be monkeypatched. The alternative -- a test
hook in `MC_engine` -- puts scaffolding in the hot loop to serve a control arm.
So the test reads the module source, substitutes one line, and executes the
result as a second module, asserting the line appears exactly once. If someone
edits that flag the test fails loudly and asks to be re-derived, rather than
silently testing the unmodified engine.

---

## Phase 2: the cyclic rod at depth >= 2 does not run

Arm A's sweep refused two of its points with

    ValueError: Cannot take a larger sample than population when replace is False

raised from inside numpy, at 2D cyclic depth 2, with no auxiliary involved. Not
skipped: chased, because a supported configuration crashing is worse than
whatever the arm was measuring.

`ChannelBody.build` recomputes the cyclic rod's trial count per move from the
chord where the line meets the box, floored at the literal **2**. That is
`depth + 1` at depth 1, and it was never revisited. An oblique line clipped to
the box leaves an arbitrarily short chord -- the slab intersection goes to zero
as the line grazes a corner -- so at depth >= 2 the count falls below the group
size, and the reverse-move mirror (`keep_mask`, which claims one trial slot per
removed molecule) asks `rng.choice` for more slots than exist.

Measured over 200k orientations at box 12, width 1.2, density 0.6:

| | depth 2 | depth 3 | depth 4 |
|---|---|---|---|
| 2D | 2.7% | 5.2% | 8.7% |
| 3D | 3.5% | 6.8% | 10.8% |

Three percent of exchange proposals is a crash within the first few dozen. The
cyclic rod at depth >= 2 had never been run by anything.

**Why the validator missed it.** `sim_setup` checks
`n_trials >= depth + 1` against the CONFIGURED count, and that check is precisely
why the plain depth path in the engine has no runtime guard of its own -- the two
`full_repacking` paths guard `max(n_add, n_remove) > n_trials` and bail, the depth
path relies on the validator. The one sampler that recomputes the count per move
was therefore the one sampler whose guarantee had quietly stopped holding.

**Why tier 2 missed it.** The matrix does cover `channel_cyclic` x depth 3, with
a pinned digest. It survived because `trial_density = 1.5` and
`channel_width = 1.5` put the crash threshold at a chord of 1.19 against a median
of 12.1. The point was one unlucky orientation from failing and its digest was
pinning a chain in that state.

**The fix.** The floor becomes `max(2, n_add, n_remove)`, passed in from the
engine. Legal because `n_trials` is a proposal parameter and detailed balance
needs only that forward and reverse agree on it for the same body: both compute
it from (body, group size), and the group size is `depth + 1` for all three move
types -- an add proposes (d+1, d), a remove (d, d+1), a repack (d+1, d+1). It
reduces to the literal 2 at depth 0 and 1, which the digests confirm.

Bailing instead would also have been balanced, and was rejected: it would drop a
few percent of orientations from the sampler rather than proposing into them, and
the cyclic rod exists to span the box.

`tests/engine/test_cyclic_rod_short_chord.py` pins both halves -- the arithmetic
over thousands of deliberately grazing chords for group sizes 1 through 5, and
the engine running 500+ exchange proposals at depth 2 and 3 in both dimensions,
which puts the probability of not meeting a short chord at about 1e-8.

---

## Phase 3: the tier-2 gate was measuring less than it claimed

Re-running tier 2 after the fix moved one digest, as expected. Reading the
committed file to check that only one had moved is what found the rest.

Five defects, all in the gate rather than the engine, all recorded with their
full argument in `DIGEST_PROVENANCE.md` under "S4". In brief:

1. **`_counter_digest` did not hash the counters.** It kept top-level integers;
   the counters and clique tallies are nested dicts and were dropped. What it
   hashed was `summary["N"]`, three integers per point. The tell was in the
   committed file and went unread -- five of thirteen ids shared one digest,
   because the hashed quantity was `N = 0, 0, 0`.

2. **Five pinned points ran zero moves.** `_advance_point` started from
   `ChainState.fresh` -- an empty box -- and never called the seeder every real
   run uses. NVT conserves N, so those chains proposed nothing for three windows.
   It now calls `runner._init_config`.

3. **The guard against exactly that was `assert pos.shape[0] >= 0`.** A shape is
   never negative. Replaced by a positive-move-count assertion, over every
   ensemble rather than muVT alone.

4. **`--matrix-regen` was declared and never read.** Four failure messages told
   the reader to run an inert flag, leaving hand-editing JSON as the only route
   to re-baseline a pin.

5. **`expected.json` was committed and never compared.** `load_expected` and
   `dump_expected` had zero callers. It was stale by 14 verdicts -- 9 from M15's
   metropolis+pmf guard, 5 from S2 commit 13's typed-metropolis refusal -- both
   intended, neither visible.

### What this does to the S2 and S3 claims

"13/13 digests unchanged", repeated through this file's predecessors, now reads
as: the per-window N series was unchanged. That is a real invariant -- an
off-by-one in an acceptance moves N -- and much weaker than the sentence it was
used to support.

The independent evidence for those stages is unaffected and is what the claims
should have rested on: `test_p4_refactor_identity` compares full counter streams
AND final positions against a frozen snapshot over fourteen modes, and
`test_typed_energy` does the same for typed-against-untyped. Those two were doing
the work the digests were being credited for.

### The gate now

    43 pins, 43 distinct digests, no collisions   (52 after phase 10)
    every pin hashes N + all 12 counters + all 15 clique tallies, per window
    every point starts where a real run starts
    a point that proposes nothing fails
    the partition file is compared, and regenerating it works

Phase 10 moves the count again -- it retires one point, regenerates the eight
`-parallel` pins that had never run the parallel kernel, and pins ten new cover
members that appeared when 24 typed+parallel points left the accepted set. The
final gate is 52 pins, 52 distinct digests.

---

## The lesson, stated once

Three of the five phase-3 defects and both phase-1 findings are the same shape: a
gate that passes without discriminating. A digest over a quantity that does not
vary. An assertion that cannot be false. A control arm whose flag short-circuits
before the code it controls. A pin on a chain that never moves.

None of these fail. That is the whole problem: a suite reports the count of tests
that passed, and every one of these contributed to that count. The only defence
that worked, here and at the M21 arbiter before it, is to ask of each gate what
value it would have to see to fail, and then check that value is reachable. Three
of the six were caught by reading the committed artifact rather than the code --
five ids sharing a digest, an inert command-line flag, a file with no readers.

---

## Phase 4: the typed auxiliary was declared and ignored

`species.aux_cutoff` is the `(T, T)` matrix `a * sigma_ij` that a species file's
`aux: {multiplier: a}` builds. Nothing outside `species.py` read it.

The engine's selection radius comes from `pot_spec.aux_cutoff`, which only the
per-run `auxiliary` block sets -- and a typed run may not use that block, by a
deliberate guard whose message says so: "in typed mode the auxiliary is stated in
the species file as a per-pair multiplier". So a species file could declare an
auxiliary, pass every range check in `species_file`, print itself in the run
header, be hashed into the run identity, and do nothing. The shipped
`ff/species_three_reduced.yaml` declares `multiplier: 1.6`, so every typed run in
the feature matrix was proposing at the full cutoff.

Not a wrong answer, which is exactly why it could sit there: the auxiliary is a
proposal bias, so dropping it samples the same distribution more slowly. A
declared knob that does nothing is still the failure mode this codebase refuses.

**Wired.** `sim_setup` resolves `aux_cutoff` from `species.max_aux_cutoff` when
the sampler supports an auxiliary, and the correction's two shell energies become
per-pair through a new `_lj_energy_typed`. That second half is not optional: run
the untyped correction on a mixture and it computes the sigma = 1 shell difference
of a system with no sigma = 1 pair in it, which is a wrong number in the
acceptance rather than a missing speed-up.

**The scalar max, not the per-pair matrix.** P5 asks for masks at
`aux_cutoff_ij`. `max_ij aux_cutoff_ij` is used instead and it is the
conservative direction -- every pair is proposed into at a radius at least its own
-- so the correction is the same difference of two radii the untyped path takes.
Per-pair is a SPEED refinement (the DP window narrows for the small pairs) needing
a per-pair split energy; landing it here would have put the correction and the
optimisation in one commit.

**Warn, do not refuse, when the sampler cannot use it.** The per-run `auxiliary`
block is a request, and an unsupported one raises. A species file is a FORCE
FIELD, shared by every run that uses that model, so refusing every branch or pore
run because the file mentions an auxiliary would make the file unusable for the
runs it is otherwise right for. Dropping the auxiliary is safe, so `sim_setup`
logs which condition blocked it and continues -- the same shape as the LJ tail
being disabled in a pore. The support matrix itself moved into one function,
`_short_range_unsupported`, because it now has two callers and two copies of a
support matrix is two matrices.

**Measured, then landed.** On a three-species channel run at box 12, 40 windows,
three seeds:

    auxiliary OFF   <N> = 1104.1 +- 6.4
    auxiliary ON    <N> = 1103.7 +- 0.8    correction mean +0.885, sd 0.806

The correction is far from zero, so the agreement is a result and not a
tautology. That is Gate P5's "auxiliary on at a legal `a` gives the same density
as auxiliary off, within block error".

Two smaller things fell out of the same work. `table_pair_energy_vec` annotated
its two type-id arguments `FloatArray`; numpy refuses a float array as a fancy
index, so the annotation described a call that cannot run -- corrected when mypy
saw the second caller. And `aux.multiplier <= core_floor.c` was unchecked: both
radii are multiples of the same `sigma_ij`, so the whole per-pair condition is
the scalar `c < a`, and without it an auxiliary inside the overlap floor would
report every trial admissible.

---

## Phase 5: the same vacuity, twice more

### `aux: {control: true}` was the auxiliary OFF

The species-file block documented as "the exact no-op arm the tuning sweep uses
as its own reference" returned `aux_cutoff() is None` -- byte-for-byte the value
`aux: none` returns. So the typed control arm produced a run with no auxiliary
machinery in it: a reference arm that IS the arm it is the control for, and a unit
test that pinned exactly that (`assert ff.aux_cutoff() is None`) and read as
coverage.

It now returns `rc` for every pair, which turns the machinery on at a radius where
the correction is identically zero.

### The bulk engine's flag, and why it was flipped rather than worked around

Phase 1 worked around `aux_on = ljbranch and rc_sel < uc` by recompiling the
module inside the test. That was the right first move -- it answered Q6 without
touching the engine -- and the wrong permanent state: the shipped gate would still
have been the vacuous one, and the real gate would have lived in a test's string
substitution.

The flag now reads `aux_on = ljbranch and _aux_rc is not None`. The cost falls
only on the control arm: every production run configures `rc_sel < uc`, where the
flag was already True, so the extra work happens only for a run whose whole
purpose is to do that work and get zero. `test_aux_noop_control.py` lost its
recompile and became one arm instead of two, covering untyped and typed.

### The MOF driver had it too

`_channel_move_ms` guarded its correction on `rc_aux is not None and rc_sel < rc`.
Third copy, same shape. The flag is flipped, and the driver now has a chain-level
control arm at all: `test_run_mof_stream_integrity.py` compares the whole
per-window CSV between an auxiliary-off run and a `rc_aux == rc` run, column by
column, and asserts first that the correction was EVALUATED -- because a mean of
0 and an sd of 0 are what a correct no-op reports AND what a correction that never
ran reports. `aux_corr_n` was added to the stream for exactly that reason.

Building it turned up a mislabel worth stating: `MofRunSpec.sweep_rc_aux` is
compared against `fieldobj.rc`, which is REDUCED (2.927 for `forcefield_A`), while
the file's `interaction.rc` is 12.0 Angstrom. The spec field carries no unit in
its name or comment, and the existing test calls `sweep_rc_aux = 2.5` the "noop"
spec -- 2.5 reduced against 2.927 reduced is a real short-range auxiliary. That
test only compares run ids so the mislabel cost it nothing, but a control arm
built on the same reading would have compared two auxiliaries and called their
agreement a no-op. The first draft of this gate did exactly that and reported
`aux_corr_mean = -0.0104`, which is how the units were found.

---

## Phase 6: the species column did not survive the window

The worst defect of the stage, and it was three lines away from the one above.

`advance` wrote the species column back under `if state.species is not None`. A
fresh `ChainState` has `species is None`. So on a real multi-species run: the cell
list was seeded from `species_of` (which materialises zeros), the window drew
species per trial, swapped identities and weighted every pair by its own
`sigma_ij` and `eps_ij` -- and then this line discarded the result. The next
window re-seeded from zeros, so **every particle was relabelled species 0 at each
window boundary.**

A three-species run at equal activities reported composition (1, 0, 0) in every
window, while accepting ~65 identity swaps per window whose effect was erased.
Per-species occupancy is the entire output of a multi-species run. The chain
itself was never wrong -- inside a window the physics is right -- which is why
nothing looked implausible.

**Why nothing caught it**, which is the reusable part:

  * the feature matrix's species invariant reads
    `if species is not None: assert aligned`. Guarded on the very thing that was
    missing, so it skipped itself on exactly the broken runs and reported green.
  * the typed statistical gates run through `advance_metropolis_mof_ms`, a
    different sampler that keeps its own column.
  * the engine tests that do check the column build a state that already has one.
    `ChainState.fresh` is the start every real run gets, and no test of this path
    used it.

The condition is now "is this a typed run" (`n_species > 1`), the matrix
invariant asserts PRESENCE first for any point with a species file, and
`tests/engine/test_species_survives_the_window.py` pins two things from
`ChainState.fresh`: that a three-species run reports more than one species, and
-- the sharp one -- that a particle which neither moved nor was swapped has the
same label after the boundary as before it. That second test needs the identity
swap suppressed, for the same reason the M21 arbiter does: a swap relabels a
stationary particle legitimately, so with it active 35 of 304 untouched particles
changed label and the test could not tell that from a boundary reset.

---

## Phase 7: Gate P5's p_trial independence, on the sweep path

With the column fixed, composition through the channel samplers is a meaningful
observable for the first time, so the M21 arbiter can be pointed at them. It is
not a duplicate of the `branch` arbiter: `branch` enumerates cliques and returns
row indices, while the sweep runs a DP over an axially ordered pool and recovers
the drawn rows from a suffix pass, sorting the pool on top of the permutation the
engine already applied. The drawn-clique-to-`pool.species` correspondence is
established by entirely separate code.

Degenerate three-species mixture, `p_trial = [0.7, 0.2, 0.1]`, swap suppressed,
80 windows, pooled over two opposite starts:

| | corrected | uncorrected |
|---|---|---|
| `channel_cyclic` | [0.393, 0.338, 0.269] | [0.705, 0.201, 0.094] |
| `channel` | [0.447, 0.299, 0.254] | [0.702, 0.186, 0.112] |

The uncorrected arms reproduce `p_trial` to 0.6% and 1.4%. That is the sharpest
form this measurement takes anywhere in the suite: the broken answer does not
merely miss uniform, it lands on the proposal distribution.

**A false alarm on the way there, recorded because it would recur.** The first
attempt used `T = 1.0`, `box = 10.0` -> <N> ~ 690, and 30 windows. It read
[0.640, 0.186, 0.174] pooled -- close to `p_trial`, nowhere near uniform, and
indistinguishable from a broken correction. It was not. With swap suppressed
labels relax only by particle turnover, so the relaxation time scales with
N / (accepted exchanges per window); at that density and length the C-start arm
still held 25% C, i.e. had barely left its initial state, and what the run
measured was the recent INSERTION FLUX, which is `p_trial`-weighted by
construction. The tell is the two starts DISAGREEING -- the converged arms above
agree to about 0.01.

So the rule is stronger than "pool two starts": never read this arbiter until the
two starts have met. Written into the test file, next to the numbers.

---

## Phase 8: Gate P5's other bullets

**"A degenerate three-species mixture reproduces the single-species channel
density within block error."** Met, and the state point needs stating because the
obvious reading of it is wrong. Three labels at ONE chemical potential do not
reproduce the single-species run -- they give three times the ideal density. The
labels are bookkeeping, so the grand partition function of a mixture of identical
species depends only on `sum_s z_s`; matching that sum is the identity, which
means `z_s = z / 3`, i.e. `mu_s = mu - kT ln 3`.

At T = 2.0, box 8, mu = -0.5 against mu_s = -2.6972, cyclic rod, depth 1:

    one species    <N> = 291.2 +- 0.9
    degenerate x3  <N> = 293.8 +- 1.8      +0.89%, 1.3 sigma

Agreement within error. Stated with the caveat the data actually supports: the
three seeds gave ratios 1.0025, 1.0117 and 1.0125 -- all positive -- so a real
bias of order half a percent is not EXCLUDED by this run length, only unsupported
by it. A longer run is the way to settle that, not a wider tolerance.

The gate is worth more than its size suggests: it tests the per-species activity
(M10), the species-draw correction (M21) and the identity swap simultaneously,
against an untyped reference that shares none of that code.

**"An `a` landing on or above `rc` raises and names the binding pair."** Already
held, in `species_file._read_aux`, with a test. S4 added the missing companion:
`aux.multiplier <= core_floor.c` was unchecked, and both radii are multiples of
the same `sigma_ij`, so the whole per-pair condition is the scalar `c < a`.

**"Channel and metropolis agree on <N_s> where metropolis still mixes."**
RETIRED, and it cannot be un-retired: typed + metropolis is refused at the
validator (S2 commit 13), because `_advance_metropolis` executed every swap as a
deletion, ignored the per-pair force field and desynced the species column. There
is no typed metropolis to compare against. The gate's own text calls it a
"consistency check only, not truth", so nothing that depended on it is lost -- but
the plan should not keep listing a gate whose reference the plan itself removed.

**"Delete the moves-per-window flag (M7)."** Not S4 work. The flag
(`sweep_moves_per_window`) is on the MOF driver's spec, not on the engine's
submission surface, and M7 says it becomes implied by the sampler choice "on the
MOF driver" -- which is the S5/S6 MOF merge. Left where it is, noted here so the
next reader does not go looking for it in the engine.

---

## Phase 9: P6, structure in and out

The plan's description was accurate and understated the checkpoint half.

**The trajectory.** `io.write_frame` built its `Atoms` object as
`Atoms("H" * len(pos), ...)`, so every particle in every frame was hydrogen
whatever its type was. A three-species trajectory therefore recorded the
coordinates of a mixture and none of its composition, and a warm start from it
would restore a pure species-0 system at the density of a mixture. Frames now
carry the species file's own per-type symbols -- which are validated against the
periodic table and checked for duplicates when the file loads, precisely so this
round trip is unambiguous.

Single-species frames keep writing `"H"`. Not laziness:
`tests/engine/test_stageA_traj_identity.py` compares trajectory files against
frozen references, and a symbol change would move every one of them to
disambiguate nothing.

**The reader.** `load_config_species` is the other half. Three things raise
rather than defaulting, and each is a silent wrong answer otherwise: an unknown
symbol (naming it and listing the known ones), a bare coordinate table
(composition is not recoverable from positions, so all-type-0 is a guess dressed
as a default), and a count that disagrees with the positions read from the same
file.

**The checkpoint, which is the sharper example.** `io.save_checkpoint` grew a
`species` parameter at D2. `_read_npz` reads the array back and `load_checkpoint`
returns it in its dict. And `runner._save` never passed it, while the resume path
never read it -- a parameter with no caller and a return value with no reader,
facing each other across a gap. A resumed multi-species run therefore restarted
with every particle relabelled species 0.

Both ends are wired, and a pre-S4 typed checkpoint is now REFUSED with a message
rather than silently resumed: continuing would report a composition the chain
never had, and there is nothing in the file to recover it from.

**A structural note.** `_init_config` was split so that "which file a warm start
reads" (`_resolve_start_file`) is decided once and read twice -- once for
positions, once for the species column. The library search is the expensive and
error-prone half (it globs a tree and matches mu, box, dimension and T out of file
NAMES), and two copies of it would let a typed run take its coordinates from one
frame and its labels from another.

**Gate P6, as stated in the plan:** positions and species round-trip through a
trajectory frame and through a checkpoint; a schema-2 checkpoint still loads; a
mislabelled symbol raises and names it; a bare coordinate table raises in typed
mode. All in `tests/unit/test_species_round_trip.py` (9 tests, sub-second) plus
an end-to-end run-then-resume in `tests/engine/test_run_resume.py`.

The end-to-end test is not redundant with the unit tests, and the reason is the
shape of the defect: both halves of the io layer were already correct in
isolation, and both are what the unit tests cover. Only a test that runs
`runner.run` twice sees the gap between them.

**CIF in is NOT done.** The plan lists it as one bullet of P6 ("through ASE and
the same symbol mapping, with an optional name map in the config for CIF labels
that do not match species names"). It is a separate reader with its own label
conventions, the MOF path already has CIF machinery that should be reused rather
than duplicated, and no current run needs it. Left for the S6 MOF work, recorded
here so the gate is not reported as fully met.

Positions come back through extxyz to 1e-7, not bit-identically, because extxyz
is a decimal text format -- the tolerance in the test is stated against ASE's own
eight-decimal writer, so it is a claim about the format rather than about the
physics. The checkpoint path IS bit-identical, npz being binary, and is asserted
as such.

---

## Phase 10: the parallel axis was never running

Found while checking whether the species-column fix (phase 6) needed a twin in
`parallel.py`. It did, and the twin turned out to be the smaller half.

**`advance_parallel` had the same writeback condition** -- `if state.species is
None: spc = None` -- so a typed parallel run never built a column. Measured on a
three-species request at box 16: N grew 453 -> 788 over four windows and
`state.species` was still None.

**But fixing that only exposed the real problem.** With a column materialised,
the composition came out `(N, 0, 0)` anyway, because `gcmc/parallel.py` contains
no species draw, no `p_trial`, no identity swap and no `species_draw_correction`.
Grep it; the cell kernel is single-species. A typed parallel run therefore
inserted everything as type 0 and produced a perfectly valid single-species chain
of species A at species A's activity -- a different question from the one asked,
answered without complaint.

So typed + parallel is REFUSED, exactly as typed + metropolis was at S2 commit 13
and for the same reason: half-wiring the species draw on top of a species-blind
cell kernel would fix the visible symptom (the composition stops being `(N, 0, 0)`)
and leave the ones that produce plausible numbers. `advance_parallel` also raises
if handed a typed config, so a script calling it directly gets an error rather
than a number. The structural half of P7 does exist -- `grid_up_nd` splits a
species column and the reassembly puts it back -- so what P7 adds is the kernel,
not the plumbing.

**And the reason nothing caught it is worse than the defect.** `_advance_point`
called `MC_engine.advance` for every matrix point. `kind` is an axis with two
values, `single` and `parallel`, and the harness ran the serial engine for both.
Nine pinned points carried a label that said parallel and a stream that was
serial. The runner has dispatched on `exe.mode == "parallel"` all along; the
matrix simply did not, so an axis with two values was testing one.

That is the fourth instance in this stage of the same shape -- a gate whose name
describes something it does not do -- and the one with the widest blast radius,
because it means no statement about the parallel path made on tier 2's authority
was ever true. `tests/engine/test_parallel_parity.py` does exercise the real
kernel and is where the parallel path's actual evidence lives; tier 2 was adding
nine green rows on top of it that measured the serial engine twice.

The eight surviving parallel pins are regenerated and are the first honest
measurement of those modes. `DIGEST_PROVENANCE.md`, "S4 phase 5", has the table.

---

## Closing tally, and the one rule that would have caught most of it

Ten phases. Two of them were the work S4 was scheduled to do (P5's gates, P6's
structure IO). The other eight were defects found while trying to do it, and
eleven of the twelve share one shape.

**The engine defects (4).** The cyclic rod could not run at depth >= 2 at all.
The typed short-range auxiliary was declared and ignored. The species column was
discarded at every window boundary, and again at every resume. Typed + parallel
sampled a single species under a mixture's label.

**The gate defects (8), and this is the part worth carrying forward.**

| what it claimed | what it did |
|---|---|
| counter digest | hashed the per-window `N` and no counters |
| 13 pinned points | 5 of them ran zero moves |
| "not empty" guard | `assert pos.shape[0] >= 0` |
| `--matrix-regen` | declared, never read |
| `expected.json` | committed, never compared |
| aux control arm | short-circuited before the correction (3 copies) |
| `aux: {control: true}` | returned no auxiliary at all |
| species alignment guard | `if species is not None:` -- skipped itself |
| `kind` axis, 2 values | ran the serial engine for both |

Not one of these failed. Every one of them contributed to a passing count, and
several were cited as evidence for claims they could not support -- "13/13
digests unchanged" carried the whole of S2 and S3.

**The rule.** Ask of every gate what value it would have to SEE in order to fail,
and then check that value is reachable. A digest over a quantity that does not
vary, an assertion that cannot be false, a control arm whose flag short-circuits
before the code it controls, a guard conditioned on the thing being missing, an
axis whose second value is never dispatched: each of these has a reachable-value
answer of "none", and each is visible in a minute once the question is asked.

**Where they were found.** Five by reading a committed artifact rather than code
-- five ids sharing one digest, an inert command-line flag, a file with no
readers, a test asserting that a control arm configures nothing, a pin labelled
parallel. Two by reading a signature against its call sites, which is where a
parameter with no caller shows up. Four by running the thing and looking at the
number.

None by reading the code that contained them. That is not an argument against
reading code; it is an argument that a gate's own text is the least reliable
description of it, and the artifacts it produces are the most reliable.

---

## Phase 11: composition never reached a file

S4's remaining work is the LJ A-B demixing runs, and the first thing checked was
not the physics but what a run WRITES. Composition is the entire output of a
multi-species run -- phase 6 says so in as many words while fixing the boundary
reset that destroyed it -- and it appeared in no artifact of any run.

    window summary   {N, packing_fraction, counters, n_cav}   plus clique, aux_corr
    run log          one `N` column

Every composition number in this suite is read out of `state.species` by a test
holding the live object: the M21 arbiter bincounts it, the typed state-point gate
bincounts it, phase 6's own regression test bincounts it. Nothing outside a
python session could see one. A cluster job at k_AB = 0.4 would have run for
hours and written a log in which A and B are indistinguishable.

**Why it survived phase 6.** That phase fixed the column and gated it, and the
gates it wrote were in-process ones -- which is the correct instrument for a
window-boundary reset and blind to this. The one place composition IS written is
`scripts/mof/run_mof.py`, whose CSV has carried `N_<guest>` columns all along, so
the typed statistical work that went through the MOF driver had the observable
and the engine's own runner did not. Same shape as the third reason phase 6 gives
for nothing catching the reset: the typed gates run through a different sampler.

**Wired.** `advance` emits `n_by_species`, a dict of counts keyed by species
name, present exactly when the chain carries a species column and absent on a
single-species run (inactive keys are absent, never null). The runner appends one
`N_<name>` column per species to the log header, indexed BY NAME so the value
under `N_B` is species B's count whatever order the summary was built in.

Three smaller decisions, each with a reason that is not taste:

  * **Raw integers, never a mole fraction.** The clique columns state the rule
    already: raw per-window sums add exactly over any span of rows, while a
    per-window fraction becomes a mean of means as soon as the reporting interval
    changes.
  * **Appended, not placed beside `N`.** The log is fixed-width and read
    positionally by `scripts/sweetspot_plot.py` and the characterisation
    harnesses. Inserting in the middle would silently reinterpret every existing
    reader's columns; appending leaves the single-species header byte-identical
    and makes the typed one a strict extension.
  * **`stats["n_by_species"]`, not `stats.get(...)`.** On a typed run the engine
    always emits it, so a missing key is a defect and a KeyError names the window
    it happened in. The clique block above it defaults to zeros because
    `advance_parallel` genuinely does not carry one; composition has no such
    excuse, and defaulting would write `(0, 0)` into a file that otherwise looks
    complete.

`_log_row` also gained a length check. `zip` stops at the shorter of its
arguments, so a row carrying more cells than the header declared would drop its
tail -- the per-species columns being the tail -- and still write a well-formed
line.

### The near-miss, which is this stage's own lesson in miniature

The composition was first emitted as a LIST. `tests/matrix/test_smoke._int_leaves`
recurses into dicts and hashes integers; a list is neither, so it walked past --
the exact traversal defect that lost the nested counters for the whole of S2 and
S3, reappearing in the gate phase 3 had just rebuilt to prevent it. The
composition would have been in the summary, in the log, and absent from the
digest that pins the stream.

It surfaced by accident: a different helper, `test_aux_noop_control._flatten`,
raised `TypeError` on the same value. The fix is the dict, and the traversal now
RAISES on a list, tuple or array rather than skipping it, so the next summary key
in an unhandled shape stops the suite instead of quietly not being pinned.

Eleven typed pins moved as a result, and the cause is proven rather than argued:
re-hashing each stream with the new key removed reproduces the committed digest
exactly, 11 of 11, and the 41 untyped pins never moved. `DIGEST_PROVENANCE.md`,
"S4 phase 11", has the table.

---

## Phase 12: an observable for demixing, with a null that is exact

A composition series says how much of each species is in the box. It does not say
whether they are MIXED, and P8's claim is about arrangement:

> at strong `k_ij`, show the mixture separates, and that the channel sampler
> reaches the separated state from a mixed warm start in fewer cycles

So `gcmc/demixing.py` computes, over the pairs within `r_cut`,

    alpha = 1 - n_unlike / E[n_unlike],
    E[n_unlike] = n_pairs * (1 - sum_s n_s (n_s - 1) / (N (N - 1)))

That expectation is the marginal probability that a uniform random permutation of
the label multiset gives one pair two different labels, and expectations add
whether or not pairs are independent. So **the null is exactly zero by
derivation**, known before any run exists, and the parameter reads 1 when no
unlike pair is in contact and negative when unlike contacts are enriched.

This is the property the stage has been asking of every gate, stated for an
observable instead: the value it would have to see in order to say "not demixed"
is reachable, and is a number rather than a judgement. A parameter built on the
count of LIKE neighbours instead -- the obvious alternative -- moves with density
at fixed composition, so a denser arm would read as a more demixed one.

The null MEAN is exact; its SPREAD is not, because pairs sharing a particle are
correlated. `permutation_band` measures it by relabelling the run's own final
configuration: same positions, same label multiset, shuffled assignment. That is
the exact null distribution of the statistic for the system in hand, so the
tolerance is a measurement rather than a guess.

The gates are closed forms rather than thresholds, which is what makes them able
to fail informatively:

    two slabs        alpha = 1 - (2 / 3 N_side) / p_unlike     geometry, to 1%
    checkerboard     alpha = 1 - 1 / p_unlike                  the floor, to 1%
    relabelled       inside its own permutation band
    pair counter     against an O(N^2) minimum-image sweep, exactly

The lattice spacing in those fixtures is pinned with an assertion, and the reason
is a fixture that lied: at spacing 1.0 with `r_cut = 1.5` the twelve
face-diagonal neighbours at 1.414 are inside the shell, and on a checkerboard
those are all LIKE, so the alternating case reads POSITIVE. The statistic was
right and the question was wrong -- the more dangerous of the two.

`PartialRDF` accumulates `g_ab(r)` from the same pair pass, with the diagonal and
off-diagonal normalisations kept apart (`N_a - 1` against `N_b`); at equal
composition they differ by nearly a factor of two, and one shared normalisation
would put the two on different vertical scales while both still looked like a
g(r). Its gate is that random labelling makes all three coincide at 1.

Neighbours come from `CellList`, not from a cKDTree, so the module is on the
right side of R5. It does overlap `rdf.py` -- that module's `g11` is this one's
one-species case -- and the overlap is recorded in the docstring as an S5 merge
in the R4 sense rather than left for someone to discover.

---

## Phase 13: the pilot, and three things it found

The runs themselves are cluster work. What was run here is the pilot that decides
the state point, and it did what a pilot is for.

### The state point is calibrated, not chosen

At T* = 1.0 and equal chemical potentials the shipped binary model is NOT
equimolar: `eps_B = 0.8` against `eps_A = 1.0` puts it at x_A = 0.70, and a 30%
minority cannot form a phase to be seen separating. Measured at box 10 over 40
windows:

    mu_A = mu_B = -3.2                 <N> = 660   rho* = 0.660   x_A = 0.70
    mu_A = -3.2, mu_B = -2.35          <N> = 678   rho* = 0.678   x_A = 0.491
    mu_A = -3.4, mu_B = -2.55          <N> = 670   rho* = 0.670   x_A = 0.504

The third is the series' state point. The offset is held FIXED across k: retuning
per k would hold composition still and let the state point drift, which is the
wrong thing to hold still in an ensemble whose composition is the observable.

### The order parameter separates the arms, and the null arm reads zero

Box 10, same mixed start, 70 windows:

    k_AB = 0     alpha = 0.003    permutation band sd 0.018    x_A = 0.50
    k_AB = 0.5   alpha = 0.17     permutation band sd 0.049    x_A = 0.13

The null arm sits inside its own band, which is the measurement that makes the
0.17 mean something. The pipeline works end to end: run, log, trajectory,
composition series, order parameter, band, partial g_ab.

### Finding 1: at strong k the composition is TRAPPED, and two samplers proved it

From the same equimolar start at k = 0.5, over the same 140k cycles:

    channel sweep         x_A: 0.50 -> 0.13     N -> 340
    rosenbluth (d0, t1)   x_A: 0.50 -> 0.96     N -> 600

Opposite corners. Both are valid chains and neither has converged: at strong k
the mixed state is unstable, both phases are long-lived, and 140k cycles is not
enough to cross between them. Reading either number as the equilibrium
composition would be reading the start.

This is S4 phase 7's rule arriving in a second setting -- *never read this
observable until arms from opposite ends have met* -- so it is built into the
experiment rather than written down as advice. `run.initial_config` is now a
swept launcher axis, `scripts.demixing start --x-first` writes the lopsided
starts, and the series runs each k from x_A = 0.9, 0.5 and 0.1. A k is reported
only where the outer pair has met; the 0.5 arm is separately the mixed start P8's
speedup claim is stated against.

**What this does to P8's second claim.** "In fewer cycles than metropolis" cannot
be measured against metropolis at all: typed + metropolis is refused at the
validator (S2 commit 13) and there is no typed metropolis to compare against --
the same hole that retired Gate P5's last bullet. The refusal's own message names
the replacement, `rosenbluth`, and `rosenbluth` at depth 0 with one trial IS
single-particle insertion, so it is the honest reference. The comparison is not
yet a comparison, though: the two arms above are in different basins, so what
they currently differ in is which phase they found, not how fast they found it.
It becomes measurable once the bracket converges, and the report should say
"against single-particle insertion", not "against metropolis".

### Finding 2: typed + NVT was neither supported nor refused

Trying to run the mixture at fixed N -- the obvious way to see two slabs at a
composition that cannot drift -- produced

    a typed (multi-species) run needs exactly one state-point route: ensemble.mu

from `build_species_spec`, which demands a route for any typed run. Doing what it
says produces `NVT takes no 'mu'/'sweep'`. Two validators, each naming the other's
refusal as the fix, with no configuration in between.

Refused by name now, with the reason: `ensemble.N` is ONE number and a mixture's
canonical state point is one count per species, which no route states. Accepting
it would mean choosing a composition on the user's behalf; inventing a spelling
for `N_s` would put a state-point convention in a validator. Same shape as typed
+ metropolis and typed + parallel, and the third member of that family.

### Finding 3: the depth trial floor did not cover `rosenbluth`

Looking for the typed reference sampler found a configuration the validator
accepts and the engine cannot run. `advance` places a depth-d move by claiming
`d + 1` trial slots through `rng.choice(n_trials, size=n_remove, replace=False)`,
and `sim_setup` guarded that with

    if branch and n_trials < _depth + 1:

where `branch` means `branch`, `channel` or `channel_cyclic`. `rosenbluth` takes
the same path and was exempt. Measured, and it is not a tail like the cyclic
rod's short chord:

    depth 1, trials 1     depth 2, trials <= 2     depth 3, trials <= 3

each raises on the FIRST exchange proposal of the first window. The validator was
accepting configurations that could not survive one window, which is the one
thing tier 2's green is supposed to mean it cannot do. Nothing in the matrix
covers it because every generated rosenbluth point carries a comfortable trial
pool -- and `rosenbluth` with one trial is exactly what a reader following the
typed+metropolis message writes next.

The guard is now the depth path's rather than three samplers', and it names the
sampler that was asked for.
`tests/engine/test_trial_floor_holds_for_every_sampler.py` pins the VALIDATOR AND
THE ENGINE AGREEING, sampler by sampler and depth by depth, rather than pinning
either side alone -- a test that only asserted the refusal would keep passing if
the engine changed underneath it. Its last case bypasses the guard after
validation and asserts the numpy error still arrives, so the refusal is protecting
against a crash rather than stating a preference.

### The figure, designed before the runs as P8 asks

Four panels, and the third is the claim:

1. **x_A against k**, three markers per k, one per start. Filled where the 0.9
   and 0.1 arms agree, hollow where they have not met -- the existing
   frozen-chain convention, carrying the convergence audit inside the result
   rather than beside it.
2. **alpha against cycles**, one line per k from the mixed start, with the
   permutation band shaded. This is the panel that shows the approach as well as
   the plateau, which is what a claim about "reaching" a state needs.
3. **alpha plateau against k**, block error over the plateau, with the null band.
   The demixing curve.
4. **g_AA, g_AB, g_BB at the largest converged k.** The structural evidence that
   panel 3's number is demixing and not a counting artefact: like pairs enhanced
   at contact, unlike suppressed.

A fifth panel -- the sampler comparison -- is designed but not yet measurable;
see finding 1.

---

## Phase 14: a mixture at fixed density, and a correction to phase 13

Phase 13 refused typed + NVT and gave a reason. The reason was wrong, and the
refusal is now gone.

**What phase 13 said.** "`ensemble.N` is ONE number and a mixture's canonical
state point is one count per species, which no route states." True of the
canonical ensemble, and this engine's NVT is not the canonical ensemble. The
NVT path forces `choice = "repack"`: the only move removes k particles and
inserts k, and a typed insertion DRAWS its species carrying `z_s` in the one-body
weight. Total N is conserved exactly while composition relaxes under the activity
ratio -- which is the SEMI-GRAND ensemble, and it is the standard instrument for
exactly the problem S4 is trying to measure.

So nothing needed inventing. The combination only needed to stop being refused by
two validators at once, and the refusal I wrote was a third one on top of a
misreading. Recorded here rather than quietly deleted because the shape is the
one this stage keeps meeting: a confident sentence about what the code does,
contradicted by the code.

**Why it matters for the science, which is the point.** The muVT arm answers
"does the mixture come apart" with a composition collapse: at k = 0.5 the pilot
went from N = 670 at x_A = 0.50 to N = 340 at x_A = 0.13. The box escapes by
evaporating one species, and the density falls with it. That is a real result and
it is the transition curve the muVT series measures -- but there is no interface
in it, so there is nothing to look at, and the P8 claim is partly a claim about a
picture.

At fixed N the escape route is closed. Density and total particle count cannot
move, so at strong k the system has to separate IN SPACE. Measured at box 16,
N = 2744 (rho* = 0.670, the density the muVT arm settles at), k = 0.5, from the
equimolar start:

    cycles     10k    160k    310k    460k    610k    910k   1200k
    x_A      0.501   0.497   0.469   0.430   0.386   0.355   0.324
    alpha   +0.014  +0.135  +0.226  +0.305  +0.402  +0.526  +0.561
    profile  (0.75)  0.191   0.247   0.343   0.429   0.388   0.529

N was 2744 in every window of that run and the density never moved, which is the
whole point of the arm. Both observables climb, and they climb in the right ORDER
-- alpha first, the profile after -- which is what growth from local clusters to
a macroscopic domain looks like and is not something either number could show
alone. The last frame's profile along the axis that separated:

    x_A per slab   0.13 0.18 0.29 0.35 0.32 0.56 0.61 0.49
                   0.41 0.40 0.36 0.35 0.30 0.23 0.13 0.08

An A-poor region and an A-rich one, with a width of several sigma between them.
Neither number had saturated at 1.2e6 cycles, so the production target of 4e6 is
sized to finish the job rather than to be safe. (The 0.75 in the first column is
the lattice-binning artefact documented with the function; alpha on that same
frame reads 0.014, which is how to tell.)

For contrast, the muVT arm at the same k over 140k cycles went to x_A = 0.13 with
N falling from 670 to 340. Same physics, different escape route: one box comes
apart in space, the other empties.

### Two gates, and they fail in different directions

`tests/engine/test_semigrand_arbiter.py` is the M21 arbiter transplanted to a
third sampler path, for the reason phase 7 gives for the second: the species-draw
bookkeeping is separate code on each path.

**Equal activities.** Degenerate species are one species wearing three labels, so
composition must be uniform for any `p_trial`. The model file's is (0.7, 0.2,
0.1), so the uncorrected answer is a third of the axis away. Measured, 343
particles, both starts, 150 windows:

    from uniform    x = [0.338, 0.320, 0.343]
    from (0.7,0.2,0.1)   x = [0.346, 0.323, 0.331]

Per-window multinomial sd is 0.025, so both sit inside one. The skewed arm
started ON the wrong answer and left it, reaching uniform by about window 40.

**Unequal activities.** With identical species the semi-grand composition is
multinomial with `p_s = z_s / sum_t z_t`, EXACTLY -- the labels do no physics, so
density, temperature and force field drop out. `mu_C = -kT ln 2` therefore
predicts (0.4, 0.4, 0.2) and nothing else:

    from uniform      x = [0.391, 0.428, 0.182]    within 1.0 sd
    from (0.1,0.2,0.7)   x = [0.406, 0.396, 0.199]    within 0.2 sd

The second arm started at 70% C and finished at 19.9%. This is the arm that
catches an activity ratio dropped, inverted, or applied once per repack instead
of once per particle -- at equal activities every one of those errors is the
identity, so the first gate is blind to all three. Both gates require the two
starts to MEET before either is read, which is phase 7's rule and is asserted
before the comparison against the prediction rather than after.

### A second observable, because alpha cannot see a phase

`alpha` counts contacts inside `r_cut`. A fluid of small like-clusters and a box
split cleanly in half read the SAME alpha, and only one of them is two phases.
`demixing.composition_profile` bins the mole fraction along an axis and
`profile_amplitude` reports the largest slab-to-slab swing over all axes -- 0 for
a uniform mixture, 1 for two pure slabs, maximum over axes because nothing in a
periodic cubic box picks out a direction.

Its gate is the pair that makes the distinction load-bearing: two slabs read
exactly 1.0, randomly dealt labels read inside the binomial slab band, and a
configuration of alternating 2x2x2 BLOCKS reads a strongly positive alpha (five
null sd) with a flat profile. That third case is the one a run reported on alpha
alone would call separated.

On the fixed-N pilot the two disagree early and agree late, which is the whole
argument for carrying both: at 240k cycles alpha was 0.18 with a flat profile
(local segregation, no interface), and by 1.2e6 cycles alpha was 0.56 with a
profile swinging from 0.08 to 0.61 (a domain). A run reported on alpha alone
could not have told those two states apart. Both are in the analysis JSON per
frame, so the figure shows the two curves and says which one the claim rests on.

One artefact is documented next to the function because it appears in the first
frame of every run: a lattice start binned into a bin count that does not divide
its site count aliases, reading a few tenths of amplitude on a perfectly mixed
configuration. It decays within a window or two, and `alpha` on the same frame
reads zero, which is how to tell.

### What is now true for the cluster

Both arms are launchable, and the launch has been REHEARSED rather than reasoned
about: `scripts/submit_demixing.sbatch` was executed end to end against a shrunk
base file -- models regenerated, three starts written, launcher expanded and ran
the runs, analysis JSON written per directory -- and at box 16 with two windows
it already separated the arms (`alpha` = 0.18 at k = 0.5 against 0.0008 at k = 0,
null sd 0.009).

Timing is measured, not estimated: one window of 2000 cycles at N = 2744 costs
about 5 s on one core, so the 4e6-cycle target is about 3 hours per run and the
24 h wall is ample. 18 runs per arm, one core each.

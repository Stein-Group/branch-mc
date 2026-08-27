# Counter-digest provenance

Every deliberate change to `tests/matrix/digests.json`, with its cause. The
digest gate is only worth anything if a removed or regenerated pin has a stated
reason, so this file is append-only and a commit that touches `digests.json`
without adding an entry here is incomplete.

The distinction that matters:

    REGENERATED   the same mode, a different counter stream. Requires a stated
                  physics or engine change. Has not happened since S1.
    RETIRED       the mode itself no longer exists. Requires a stated reason for
                  the mode's removal. The stream is not "unchanged" or "changed";
                  there is nothing left to run.

---

## S1: baseline

15 points digested, one per member of the pairwise cover as it stood.

## S2 commits 6-12: unchanged

Nine mechanical commits (the sixth sub-config, the shared helpers, `parallel`,
`MC_engine`, `runner`, the pore-predicate merge). 15/15 unchanged throughout.
That is the whole claim of those commits and it held.

## S2 commit 13: two RETIRED

    lennard_jones-metropolis-d0-3d-muVT-single-typed        00ea759508712719
    lennard_jones-metropolis-d0-3d-confined-single-typed    fb1cfab25d0355f6

**Cause.** `sim_setup.from_params` now refuses `species_file` +
`sampler.type='metropolis'`. The mode is gone, so its streams cannot be checked.

**Why the mode was refused.** `_advance_metropolis` is not multi-species in any
respect, and three things went wrong at once, none of them raising:

1. `moves.draw_move` returns `"swap"` for n_species > 1; the loop dispatched on
   `if move / if add / else`, so every swap proposal executed as a DELETION.
   Measured over one window: 15 removes and 14 swaps drawn, `remove_step`
   recorded 29.
2. The pair energy is `lj_pair_energy`, sigma = epsilon = 1. The species file's
   per-pair sigma, epsilon and k_ij are never read.
3. `state.species` is never updated with `state.positions`, so the two desync on
   the first accepted exchange -- 43 positions against 40 labels after one
   window.

**What this means about the two retired digests.** They pinned the counter
streams of a mode that was computing the wrong thing. They were stable, and
their stability was never evidence of correctness -- a wrong calculation
reproduces exactly as faithfully as a right one. Worth stating plainly, because
"15/15 unchanged" reads like a correctness claim and is not one: it says the
engine is deterministic under refactoring, and nothing whatsoever about whether
what it computes is right.

**How this was caught.** By `test_counter_digest`, which failed with "has a
committed digest but the validator no longer accepts it". Under the gate as it
stood before commit 9 these two would have silently dropped out of the run set
and reported as skips.

## S3: unchanged (13/13), and one OTHER pin re-baselined

No digest moved. The typed insert path, the identity swap and the
`ljbranch_weight_sum` signature change all landed with the counter streams
untouched, which is the claim that the single-species chain is bit-identical.

A DIFFERENT pin did move, recorded here because it is the same kind of object:

    tests/unit/test_locked_invariants.py :: ljbranch_weight_sum

    from  84f3d9613a3226393bb7a39b034a91396f35890dfea1c8ba821e2dd4611681c3
    to    4dc9820e1d4ad518214b86c8cb5635840e4d7cef7c1115eb226f58d85e06e379

Two deliberate changes at once, separable and both stated in the test's own
provenance comment:

1. The hash function moved from `ast.dump` to `ast.unparse`. `ast.dump` output
   gained a `type_params` field on FunctionDef in 3.12, so the same unmodified
   source hashed differently on 3.11, 3.12 and 3.13 and only the CI version
   matched -- while `pyproject` promises `>=3.11`. On the UNMODIFIED body the new
   function gives `7ad0b90206a8142d7f7067c1...` identically on all three.
2. The body gained a fourth return value, `chosen_rows`. The typed insert path
   needs the drawn clique's row indices, because M21 decides provenance by pool
   row and never by comparing coordinates. `idx` was already computed on every
   iteration; the change is to stop discarding it.

The reservoir test, its rng consumption and the log-space accumulation order are
untouched, so no weight and no draw moved. The counter digests are the evidence.

## S3 phase 2 (the typed energy): unchanged (13/13)

No digest moved. `build_potential` now returns `LennardJonesTable` for a typed
`lj` run, the trial-cloud weights and the overlap floor became per-pair, and the
identity swap gained a real energy difference -- all with the counter streams
untouched, because every one of those paths is guarded on a typed potential being
present and an untyped run takes the expression that was already there.

The independent evidence is stronger than the digests here, and it is worth
naming: `tests/engine/test_typed_energy.py` asserts that a typed run on a
one-species unit file reproduces the UNTYPED run bit for bit -- same counters
every window, same final positions, same N series. The digests say the untyped
path did not move; that test says the typed path lands exactly on top of it.

## S3 phase 3 (the per-species activity): unchanged (13/13)

No digest moved. The exchange acceptance now carries
`sum_ins log z_s - sum_del log z_s` in place of one scalar `beta * mu`, and the
single-species reduction is exact: the sums collapse to `net * log z_0`, which is
`+log z_0` / `-log z_0` / `0` for add / remove / repack -- the three cases the
scalar expression covered. `log z_0 == beta * mu` bit-for-bit when the state point
came in by the mu route at unit thermal wavelength, which is checked rather than
assumed by the reduction test in `tests/engine/test_typed_energy.py`.

The untyped path keeps the literal `beta * mu` expression it always had; the typed
term is reached only when a species file is present. That is what makes the
bit-identity structural rather than lucky.

## S3 phase 4 (the typed sweeps): unchanged (13/13)

No digest moved, across the channel dispatch, the typed branch kernels, the two
filled cyclic cells and the `table_pair_energy_vec` vectorisation.

`ljbranch_weight_sum`'s LOCKED BODY moved for the second time. Its provenance
comment in `tests/unit/test_locked_invariants.py` carries the full entry; in
short: three keyword-only arguments (`species`, `floor`, `table`), all defaulting
to None, and on that default path every expression is the one that was there. The
clique enumeration, the reservoir draw and its rng consumption are untouched --
what changed is the graph handed to it and the weight of a pair, not how cliques
are walked.

The vectorisation is worth a line of its own because it LOOKS like a numerical
change and is not: `table_pair_energy_vec` gathers `sigma[si, sj]` and evaluates
the same expression in the same order as the scalar `table_pair_energy`, which is
asserted element-by-element against the scalar form rather than assumed. Tier 2
fell from 106 s to 68 s with every digest unchanged.

## S3 phase 5 (tier 3's first run): unchanged (13/13)

No digest moved. Three `slow` tests were repaired -- they had been erroring since
R16 on `sampler.channel_length = "full"`, a spelling the validator refuses -- and
one stale comment in `sim_setup` was corrected. No engine code changed, which is
why the digests are untouched and why this was a test repair rather than a physics
change.

Recorded here because it is the same class of hazard this file exists for: a pin,
a gate or a test that is never exercised says nothing, and its failure mode is
silence rather than noise. Tier 3 had never been run against this tree.

## S4: ALL 13 REGENERATED, 30 added, and the reason is that the gate was not the gate

This is the largest entry in this file and the one most worth reading, because
what changed is not a counter stream: it is what the digest had been measuring.

### 1. The digest did not hash the counters

`_counter_digest` walked the window summary and kept every value passing
`isinstance(val, int)`. The summary is not flat. The counters live in
`summary["counters"]` and the clique tallies in `summary["clique"]`, both dicts,
so both failed that test and were dropped. What survived the filter was
`summary["N"]` -- one integer per window.

So for the whole of S2 and S3 the "counter digest" hashed three particle counts
and no counters, while being cited in this file as the bit-identity evidence for
every migration in both stages.

The tell was in the committed file and went unread: five of the thirteen ids
shared one digest. A hash collision across five points is not a coincidence, and
the value was `sha256("0.N=0|1.N=0|2.N=0")`.

**What the old claims are still worth.** "13/13 unchanged" across S2 and S3 now
reads as: the per-window N series was unchanged. That is a real invariant and a
non-trivial one -- an off-by-one in an acceptance moves N -- but it is much
weaker than the sentence it was used to support. The independent evidence for
those stages is unaffected and is what should be leaned on instead:
`tests/engine/test_p4_refactor_identity.py` compares full counter streams AND
final positions against a frozen snapshot over fourteen modes, and
`tests/engine/test_typed_energy.py` does the same for typed-against-untyped.
Those two were doing the work the digests were being credited for.

`_int_leaves` now recurses, so `counters` and `clique` are hashed. Floats stay
out for the reason the old docstring gave and which still holds: they are not
bit-reproducible across BLAS builds. Every engine counter is an integer, so the
int-only RULE lost nothing -- the TRAVERSAL lost the dicts.

### 2. Five of the thirteen pinned points ran zero moves

`_advance_point` built its start with `ChainState.fresh`, i.e. an empty box, and
never called the seeder every real run uses. NVT conserves N: a chain that starts
with no particles proposes nothing and stays empty. Three windows, every counter
zero, for five pinned points -- `hard_core-branch-dfull_repacking-3d-NVT-parallel`,
`hard_core-reweight-d3-2d-NVT-single`,
`hard_core-rosenbluth-d2-3d-NVT-single-aux_pmf`,
`lennard_jones-channel-d1-2d-NVT-single-aux_short_range-len_6.0`,
`lennard_jones-channel_cyclic-d4-3d-NVT-single-aux_short_range`.

They were green because nothing can go wrong in a chain that never moves.

`_advance_point` now calls `runner._init_config`, which is what every real run
calls: NVT gets its lattice of `target_n`, a confined pore gets particles inside
the pore, a QA run gets fluid clearing the obstacles, and bulk muVT still starts
empty because that branch returns an empty array. Nothing is invented that
production would not do -- the matrix had been testing a start no run uses.

### 3. The guard against exactly this was a tautology

`_assert_invariants` carried, under a comment about muVT chains that never fill:

    assert pos.shape[0] >= 0, f"{pid}: negative particle count"

A shape is never negative. It is replaced by a count of proposed moves over the
window set, asserted positive, and stated over every ensemble rather than muVT
alone -- NVT was the ensemble actually broken, and a guard that only watches the
mode someone suspected is how this was missed.

### 4. `--matrix-regen` did not exist

The option was declared, its help text described the two files it would rewrite,
and four failure messages told the reader to run it. Nothing anywhere called
`config.getoption("--matrix-regen")`. The documented way to re-baseline a pin was
inert, which leaves hand-editing JSON as the only route -- the habit every
comment in this file is written to prevent. It is implemented now, in
`tests/matrix/conftest.py` (pytest collects hooks from conftest files only), and
merges rather than replaces so a regen run under the pairwise cover cannot
silently drop pins outside it.

### 5. `expected.json` was committed and never compared

Same shape from the other direction: `_enumerate` documents the file as the
record that makes a validator change show up as a diff, `load_expected` and
`dump_expected` were written for it, and both had zero callers. It was stale by
14 verdicts, all accounted for and all intended:

    9  hard_core-metropolis-d0-*-aux_pmf         M15 moved the metropolis+pmf
                                                 guard into the validator
    5  lennard_jones-metropolis-d0-*-typed       S2 commit 13 refused typed +
                                                 metropolis (see that entry)

`test_the_committed_partition_is_still_the_live_one` compares it now, reporting
flips, appearances and disappearances separately -- a vanished point is the
quietest of the three and the one that shrinks coverage.

### 6. One genuine engine fix, and it is why S4 started here

`ChannelBody.build` floored the cyclic rod's chord-derived trial count at the
literal 2. That is `depth + 1` at depth 1 and was never revisited. An oblique
line clipped to the box leaves an arbitrarily short chord, so at depth >= 2 the
count fell below the group size and the engine died in

    keep_mask[rng.choice(n_trials, size=n_remove, replace=False)] = False
    ValueError: Cannot take a larger sample than population when replace is False

on 2.7% of exchange proposals in 2D and 3.5% in 3D, measured over 200k
orientations. Three percent is a crash within the first few dozen proposals: the
cyclic rod at depth >= 2 had never been run by anything.

The floor is now `max(2, n_add, n_remove)`, passed from the engine. It is the
same number on both sides of every move -- an add proposes (d+1, d), a remove
(d, d+1), a repack (d+1, d+1) -- so detailed balance is untouched, and it reduces
to the literal 2 at depth 0 and 1. `tests/engine/test_cyclic_rod_short_chord.py`
pins both halves: the arithmetic on grazing chords, and the engine running long
enough that those orientations are certain.

**How it hid.** The validator checks `n_trials >= depth + 1` against the
CONFIGURED count, and that check is why the plain depth path has no runtime
guard. The one sampler that recomputes the count per move was therefore the one
sampler whose guarantee had quietly stopped holding. The matrix's own
`channel_cyclic-d3` point survived only because `trial_density = 1.5` and
`channel_width = 1.5` put the crash threshold at a chord of 1.19 out of a median
12.1 -- it was one unlucky orientation from failing, and its pinned digest was
recording a chain in that state.

### The resulting file

43 pins, 43 distinct digests, no collisions (52 after phase 5 below). Up from
13 pins of which 5 were the
empty stream and all 13 hashed three integers. The 30 additions are the current
pairwise cover, which had never been pinned; every one of the original 13 is
REGENERATED, and the causes are 1, 2 and 6 above compounded -- the hashed
quantity changed, the starting configuration changed for NVT/confined/QA, and
the depth>=2 cyclic chain changed.

Only one pin moved for reason 6 alone, and it was seen before the other changes
landed: `lennard_jones-channel_cyclic-d3-3d-muVT-single-aux_short_range` went
`d15736b6487f19c6 -> a0bb3ffb25f8b037` with the trial-count floor as the sole
edit in the tree. Recorded because it is the only clean attribution available in
this entry, and a compound re-baseline with no clean attribution anywhere in it
would be worth distrusting.

## S4 phase 4 (the typed auxiliary and the species column): EIGHT REGENERATED

Eight pins moved, all typed, and nothing else did. Both causes are engine fixes.

    lennard_jones-branch-d0-2d-muVT-parallel-typed          41abb090 -> e0c4402f
    lennard_jones-branch-d3-3d-confined-single-typed        38263b6d -> 1171a9c9
    lennard_jones-branch-d4-3d-QA-single-typed              edc333a6 -> 8aa9747c
    lennard_jones-branch-dfull_repacking-3d-QA-single-typed edd2e51b -> 9d7382a2
    lennard_jones-channel-d1-2d-QA-single-len_6.0-typed     67832626 -> 2b77372c
    lennard_jones-channel_cyclic-d2-3d-QA-single-typed      a14c49a2 -> 4ac59008
    lennard_jones-channel_cyclic-d4-3d-confined-single-typed f2b2ab0e -> 2ccd0f94
    lennard_jones-rosenbluth-dfull_repacking-3d-muVT-single-typed
                                                            ed4a19d1 -> b2b3624d

That the 35 untyped pins are byte-identical is the evidence that both fixes are
confined to the typed path, which is what they were meant to be.

**Cause 1: the species column was discarded at every window boundary.** `advance`
wrote `state.species` back under `if state.species is not None`, and a fresh
`ChainState` has None. So a multi-species run drew species per trial, swapped
identities, weighted every pair by its own sigma_ij and eps_ij -- and then threw
the column away. The next window re-seeded the cell list from `species_of`, which
materialises zeros, so every particle was relabelled species 0 at the boundary. A
three-species run at equal activities reported composition (1, 0, 0) in every
window while accepting ~65 identity swaps per window whose effect was erased.

Per-species occupancy is the entire output of a multi-species run, so this was
the most consequential defect of the stage. The chain itself was never wrong --
inside a window the physics is right -- which is why nothing looked implausible.
The condition is now "is this a typed run" (`n_species > 1`). Every typed pin
moves because window k+1 now starts from real species instead of zeros.

The gate that should have caught it read
`if species is not None: assert species.shape[0] == pos.shape[0]` -- guarded on
the very thing that was missing. It now asserts PRESENCE first for any point with
a species file. `tests/engine/test_species_survives_the_window.py` pins it from
`ChainState.fresh`, which is the start every real run gets and the one thing no
existing test of this path used.

**Cause 2: the typed short-range auxiliary now exists.** `species.aux_cutoff` --
the (T, T) matrix `a * sigma_ij` from the species file's `aux: {multiplier: a}` --
was read by nothing outside `species.py`. The shipped
`ff/species_three_reduced.yaml` declares `multiplier: 1.6`, so every typed run in
the feature matrix was proposing at the full cutoff while the file, the header
print and the run identity all said otherwise. Not a wrong answer, which is
exactly why it could sit there: the auxiliary is a proposal bias and dropping it
samples the same distribution more slowly.

Wired at S4 through `pot_spec.aux_cutoff` (the scalar `max_ij aux_cutoff_ij`,
which is the conservative direction), with the correction's two shell energies
made per-pair via `_lj_energy_typed`. Only the two typed CHANNEL pins could move
for this cause -- the support matrix leaves branch and rosenbluth without an
auxiliary -- and both of those moved for cause 1 as well, so the two are not
separable in this table. They were separated by measurement before landing:
aux-on gives <N> = 1103.7 +- 0.8 against aux-off 1104.1 +- 6.4 on a
three-species channel run, with the correction itself far from zero
(mean +0.885, sd 0.806), so the agreement is a result rather than a tautology.

**One pin gained a column rather than changing.** `run_mof`'s CSV now streams
`aux_corr_n` beside `aux_corr_mean` and `aux_corr_sd`. Not a matrix digest, but
recorded here for the same reason as the locked-body hashes: a mean of 0 and an
sd of 0 are what a correct no-op arm reports AND what a correction that never ran
reports, so without the count the MOF control arm could not be told from a
vacuous one -- which is how the third copy of the `rc_sel < rc` short-circuit went
unnoticed. See `S4_REPORT.md` phase 5.

## S4 phase 5 (the parallel axis was never running): 8 REGENERATED, 1 RETIRED, 10 added

    RETIRED
    lennard_jones-branch-d0-2d-muVT-parallel-typed          e0c4402f268a864e

**Cause.** `sim_setup` now refuses `species_file` + `run.kind='parallel'`. The
mode is gone, so its stream cannot be checked.

**Why the mode was refused.** `gcmc/parallel.py` contains no species draw, no
`p_trial`, no identity swap and no `species_draw_correction` -- grep it. The cell
kernel is single-species, so every insertion is type 0. Measured on a
three-species request at box 16 over four windows: N grew 453 -> 788 with
composition (N, 0, 0) throughout. What ran was a valid single-species chain of
species A at species A's activity; what was asked for was a mixture. Same class
as S2 commit 13's typed metropolis and found the same way -- by running it.

The structural half of P7 exists (`grid_up_nd` splits a species column, the
reassembly puts it back, and S4 fixed the writeback that dropped it whenever the
caller had not supplied one). What is missing is the cell kernel, which is what
P7 is. `advance_parallel` also raises now if handed a typed config, so a direct
caller cannot get the plausible number either.

    REGENERATED (8)
    hard_core-branch-dfull_repacking-3d-NVT-parallel        745655e9 -> 7e8b2404
    hard_core-metropolis-d0-2d-muVT-parallel                2f0bf63f -> 704b580b
    hard_core-rosenbluth-d4-2d-NVT-parallel-aux_pmf         89c68ee8 -> fb0db82a
    hard_core-rosenbluth-d4-2d-muVT-parallel-aux_pmf        b4c7f480 -> f98390e2
    lennard_jones-branch-d1-3d-NVT-parallel                 9c6e97a9 -> 284f7038
    lennard_jones-rosenbluth-d2-3d-NVT-parallel-aux_pmf     ae506d83 -> 672d07b7
    lennard_jones-rosenbluth-d3-3d-muVT-parallel-aux_pmf    c28b10cb -> e24d1013
    lennard_jones-rosenbluth-dfull_repacking-3d-muVT-...    e2003424 -> c217283c

**Cause: every one of these is a `-parallel` point, and none of them had ever run
the parallel kernel.** `_advance_point` called `MC_engine.advance` for every
point. `kind` is an axis with two values -- `single` and `parallel` -- and the
harness ran the serial engine for both. The runner dispatches on
`exe.mode == "parallel"`; the matrix did not, so nine pinned points carried a
label that said parallel and a stream that was serial.

`_advance_point` dispatches the same way the runner does now, so these eight
digests are the first honest measurement of those modes. The chains are DIFFERENT
code, not a changed one: nothing about `parallel.py` moved except the species
writeback, and none of these eight is typed.

**Ten points added.** Refusing typed + parallel removed 24 points from the
accepted set, which re-ran the greedy pairwise cover; ten of its new members had
no digest. 52 pins, 52 distinct digests, no collisions.

**`expected.json` moved by 38 verdicts**, all accounted for:

    24  typed + parallel        refused at S4 (P7 not wired)
     9  metropolis + hard-core pmf   M15 moved the guard into the validator
     5  typed + metropolis      S2 commit 13

The last two were already stale before S4 and invisible because nothing compared
the file; see the S4 entry above.

---

## S4 phase 11 -- per-species occupancy enters the digest

**Eleven pins moved, all typed, and the cause is proven rather than argued.**

`MC_engine.advance` now emits `n_by_species` in the window summary -- a dict of
per-species counts, keyed by species name, present only when the chain carries a
species column. `_int_leaves` walks it, so composition is now part of every typed
point's hashed stream. Nothing else about those runs changed: no acceptance term,
no draw order, no weight.

The proof, rather than the assertion: for each of the eleven, the summaries were
re-hashed with `n_by_species` REMOVED and compared against the committed value.

    11 reproduce the committed digest exactly, 0 do not

So the chains are bit-identical to their pre-S4.11 selves and only the hashed
content grew. The 41 untyped pins are unchanged, which is the other half of the
same statement: a moved chain would have moved them too.

    lennard_jones-branch-d0-2d-QA-single-typed                b3f55e3ecb50258a -> b6f1631172276be2
    lennard_jones-branch-d2-3d-muVT-single-typed              (see digests.json)
    lennard_jones-branch-d3-3d-confined-single-typed
    lennard_jones-branch-d4-3d-QA-single-typed
    lennard_jones-branch-dfull_repacking-3d-QA-single-typed   9d7382a29e0edec1 -> 6a3b73c9c9885dd9
    lennard_jones-channel-d1-2d-QA-single-len_6.0-typed       2b77372c5e038ab8 -> 8a9dc662161e611c
    lennard_jones-channel-d3-2d-muVT-single-len_6.0-typed     476045762c1c9852 -> 26bf8572f4a178b0
    lennard_jones-channel_cyclic-d2-3d-QA-single-typed        4ac590089336293f -> 0878f0488d94a5a2
    lennard_jones-channel_cyclic-d4-3d-confined-single-typed  2ccd0f94d715d45b -> 764aa29518ae0f03
    lennard_jones-channel_cyclic-dfull_repacking-3d-QA-typed  616846466c69ba08 -> 78dbff3a4af5b833
    lennard_jones-rosenbluth-dfull_repacking-3d-muVT-typed    b2b3624d78a0652b -> 3ef5fa5914df5443

52 pins, 52 distinct digests, no collisions. `expected.json` did not move: the
two validator changes landing in the same stage -- typed + NVT refused by name,
and the depth trial floor extended to `rosenbluth` -- rejected nothing the
validator was accepting. Both configurations died in the engine before; the
accepted set is what it was.

**The near-miss, recorded because it is the S4 lesson defending itself.** The
composition was first emitted as a LIST. `_int_leaves` recurses into dicts and
hashes integers; a list is neither, so it was dropped in silence -- the same
traversal defect that lost the nested counters for the whole of S2 and S3, in a
gate that had just been rebuilt to prevent it. The tell was a test helper
elsewhere raising `TypeError` on the same value. The traversal now RAISES on a
list, tuple or array instead of walking past it, so the next summary key that
arrives in an unhandled shape stops the suite rather than quietly not being
pinned.

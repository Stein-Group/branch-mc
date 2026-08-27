# S3: the typed insert path, the identity swap, and the M21 verdict

Headline: **M21 passes.** The derivation the plan called the highest-risk item in
the project has been measured, and it holds. The measurement, its negative
control, and the two things the plan did not anticipate are below.

    tier 0   green
    tier 1   586 passed, 45 skipped, 1 xfailed
    tier 2   8 failed, 38 passed  -- ONE named cause, see section 6
    digests  13/13 UNCHANGED

---

## 1. What was broken

Two defects, one loud and one silent, in the same feature.

**The swap move did not exist.** `moves.draw_move` has emitted `"swap"` for
n_species > 1 since M8. `MC_engine.advance` had no branch for it and fell through
to a raise, so every multi-species run crashed within a handful of steps. That
was 8 of the 21 original tier-2 failures.

**Insertions were always species 0.** Removals took any species, so every species
but the first could leave the box and never arrive. Add and remove were not each
other's mirror -- and that mirror is the one thing the grand-canonical acceptance
assumes. Probed from an all-type-1 configuration, 61 type-0 particles appeared in
a single window.

The second is worse than the first and nothing reported it. Fixing only the swap
would have traded a loud crash for a silent detailed-balance violation, which is
why it was not done that way.

## 2. What was wired

    trial species        drawn from `p_trial` with the position (M19)
    provenance           a `pool_row` column on `TrialSet`, stamped AFTER the
                         pool permutation
    the M21 correction   `moves.species_draw_correction` into all three
                         acceptance sites
    the identity swap    a real branch, `moves.swap_log_acceptance`, no
                         combinatorial term
    commit               `commit_exchange(..., chosen_species)`

**The permutation trap, and how it is closed.** M21 decides provenance by POOL
ROW and explicitly never by comparing coordinates -- a reselected member and a
trial atom that lands on it are indistinguishable by position. The engine does
`pool = pool[rng.permutation(len(pool))]` and the recorded trap is that this
threw the permutation away.

Carrying it as a `TrialSet` column rather than a separate variable is what makes
it survive: the Rosenbluth selection deletes and masks the pool between levels,
so a row number held anywhere outside the struct is stale after the first level.
`geometry.find_neighbors` threads the species column through the same `[good]`
survivor filter as the coordinates for the same reason -- only that function
knows which trials survived.

Two facts had to be recovered by measurement rather than reading:

* `ljbranch_weight_sum` computes the clique's row indices and discarded them.
  Recovering them outside would have meant either an exact-float coordinate
  lookup (forbidden) or a second copy of the clique enumeration. It now returns
  them, with a deliberate re-baseline of its locked-body hash -- see section 5.
* `TrialSet.delete` mutates IN PLACE and returns self, so `select_forward`
  shrinks the CALLER's pool. Indexing it afterwards with pre-selection row
  numbers is wrong. The species column is snapshotted before the selection runs.
  This surfaced as an `IndexError` only because the pool happened to be short;
  in a larger pool it would have been a wrong answer.

## 3. Gate A: the degenerate control arm

A single-species run must be bit-identical to before. It is: **13/13 counter
digests unchanged**, across every commit of this stage. The `typed_insert` switch
is what buys that -- when it is false nothing draws, branches, or costs anything,
which matters because an `rng` draw consumes the stream and one extra draw would
move every characterisation trajectory in `tests/ref` and the whole P4 snapshot.

## 4. Gate B: the arbiter

Three species identical in sigma, epsilon, lambda and mu, every k_ij absent, and
`p_trial = [0.7, 0.2, 0.1]`. The mixture is one species wearing three labels, so
the stationary composition must be uniform for any `p_trial` whatsoever, because
`p_trial` is a proposal knob and a proposal knob cannot move an equilibrium
distribution.

    depth 0, 600 windows, pooled over two opposite starts

        M21 ON    [0.346, 0.311, 0.344]    max|x - 1/3| = 0.023
        M21 OFF   [0.692, 0.188, 0.120]    max|x - 1/3| = 0.359   (= p_trial)

The correction reduces the bias by a factor of ~16 and leaves the composition
uniform to about 2%. Removing it makes the composition track `p_trial` almost
exactly. **Verdict: M21 passes**, and it holds at depth 1 as well as depth 0 --
asserted separately, because depth >= 1 selects a clique from a finite pool, which
is sampling without replacement and a strictly harder claim than a single `p_s`
draw.

### 4a. The first run of the arbiter measured nothing, and looked like a pass

With the identity-swap move active, both arms came out uniform inside the
4-sigma band -- the correct one and the deliberately-broken one alike:

        M21 ON    [0.341, 0.330, 0.329]    "UNIFORM, M21 behaves"
        M21 OFF   [0.341, 0.329, 0.330]    also uniform

In a degenerate mixture every activity is equal, so every swap is accepted and
labels random-walk to uniform no matter how insertions are booked. **The swap
move masks the effect the arbiter exists to detect.** The plan did not anticipate
this, and it could not have: swap did not exist when the arbiter was written.

A test that cannot fail reports success. The arbiter now suppresses swap, and the
negative control is a test in its own right -- `test_the_arbiter_can_actually_fail`
asserts that removing the correction DOES break the composition, and that the two
arms are separated by at least a factor of three. Without that assertion the pass
above would be worth nothing, and for one run it was worth nothing.

### 4b. Convergence is the binding constraint, and the arms differ in HOW they fail

Without swap, composition relaxes only through particle turnover, which is slow;
neither arm is fully converged at any run length affordable in a test. What
separates them is the shape of the residual, and it is unambiguous:

    M21 OFF   both starting compositions converge toward the SAME non-uniform
              answer, ~p_trial. That is a bias.
    M21 ON    the two starts BRACKET uniform symmetrically and are still
              approaching. That is slow relaxation.

    depth 0, 600 windows, last quarter
        from all-A (favoured)     [0.395, 0.284, 0.321]
        from all-C (disfavoured)  [0.312, 0.328, 0.360]

So the assertions are stated on the pooled average of two opposite starts, which
cancels the residual to first order, and on the ratio between the arms. Asserting
one start against uniform would have measured the run length.

An earlier, shorter run reported "M21 FAILS" on exactly that mistake: a single
all-A start at 60 windows gave [0.654, 0.233, 0.113], which reads as tracking
`p_trial`. It was not converged, and the start was the favoured species. The
verdict was wrong and the run that produced it was too short by an order of
magnitude.

## 5. The locked-body hash was unportable, and is now not

`test_ljbranch_weight_sum_body_is_unchanged` hashed `ast.dump(node)`. That output
gained a `type_params` field on FunctionDef in 3.12, so the same unmodified
source hashed three different ways:

    3.11  53dcf3c246dc...    3.12  84f3d9613a32...    3.13  1413c4f53423...

Only 3.12 -- the version CI pins -- matched the committed value, while
`pyproject` promises `>=3.11`. The gate failed on correct code on two of three
supported interpreters, which is the worst failure mode a gate can have: the only
way to make it pass is to stop believing it.

It now hashes `ast.unparse(node)`, the canonical source round-trip: formatting-,
comment- and line-number-insensitive, which is what the test wanted, and verified
identical on 3.11, 3.12 and 3.13 before the re-baseline rather than after. The
re-baseline bundles that change with the added `chosen_rows` return value, and the
two are separated in the provenance comment so a future reader can tell which
moved what. No weight and no draw changed; the digests are the evidence.

## 6. What is left, precisely

All 8 remaining tier-2 failures are one cause, and it is now named rather than
presenting as a crash:

> multi-species Lennard-Jones is not wired end to end. [...] this method returns
> a single-sigma LennardJones, so the per-pair sigma/epsilon/k_ij in the species
> file would never reach the engine and every species would interact identically.

`Config.build_potential` returns a plain `LennardJones` for every `lj` /
`ljbranch` run. `potentials.lj_table` reads `species.table` per pair but only the
MOF path selects it. So the BOOKKEEPING half of P3 is done and measured; the
ENERGY half is not, and a typed run would have produced an isotherm for an
equal-sized mixture.

**This corrects a comment in `sim_setup`** which states that "the hot path reads
`species.table` and `species.core_floor` per pair -- and P3 asserts that". That
is the intent. It is not what the code does, and measuring it is how the gap was
found.

It is refused in `build_potential` rather than in the validator, deliberately.
The params are valid and the Config is correct -- it carries the pair table, the
per-species activities and the right conservative stand-ins, and
`tests/unit/test_typed_config.py` asserts exactly that on files whose pairs
differ. What does not exist is an engine that can integrate it, so the refusal
names the one function that has to change.

An energetically degenerate species file is allowed through, and that is the true
condition rather than a loophole: if every pair shares one sigma, one epsilon and
one core floor, the single-sigma potential represents that mixture exactly -- the
components differ only in activity, which the engine does carry. It is also what
makes the arbiter runnable.

**Why tier 2 is red rather than green.** Refusing typed runs in the validator
would have removed those 8 points from `accepted()` and turned tier 2 green. That
would have been green-by-redefinition. The handover's own standard is that green
means "no known bugs", and there is one: the pair table does not reach the engine.
It stays red until that is wired.

### The next step, concretely

`potentials.lj_table.LennardJonesTable` is per-trial stateful -- it carries a
`trial_type` and reads `table_pair_energy(r, trial_type, neighbour_type, table)`.
The engine's weighers call `potential.weight_pair(dists)` for the whole trial
cloud at once and pass no types. So wiring it is not a one-line change in
`build_potential`: the weigher interface has to carry the trial's species and the
neighbours' species through `_weighted_trials`, `select_forward` and the DP's
intra-clique factor. The species columns those need are now all in place, which is
the part S3 delivered.

---

# S3 phase 2: the typed energy reaches the engine (conventional path)

    tier 1   592 passed, 45 skipped, 1 xfailed   (+6 new gates)
    tier 2   7 failed, 39 passed  -- ONE cause, the sweeps, named
    digests  13/13 UNCHANGED

The per-pair sigma, epsilon and k_ij from the species file now reach the hot loop
for `sampler.type='rosenbluth'`. That is the scope fence `potentials/lj_table.py`
sets for itself, quoted in its own docstring: "this is the CONVENTIONAL move path
only. The sweep (`ljbranch`) is untouched this phase; its AST-locked kernels
(`count_and_sample`, `ljbranch_weight_sum`) stay exactly as they are."

## What was wired

    build_potential           returns LennardJonesTable for a typed `lj` run
    trial-cloud weights       per (trial type, neighbour type) from the table
    translate move            typed dU and a per-pair overlap floor
    identity swap             a REAL dU -- relabelling now changes the energy
    admissibility             core_floor[t_trial, t_neigh], not one scalar
    the mean-field tail       forced OFF for typed runs

Three things worth stating rather than leaving in the diff.

**The trial cloud needed a new shape of call.** `LennardJonesTable.trial_type` is
per-instance, which suits the MOF path -- one guest inserted at a time. The
exchange move weights an entire cloud in one call and, after phase 1, every trial
in it carries its own drawn species. A per-instance type cannot express that, and
one potential object per type would put a Python loop over trials in the hot
path. `pair_u_grid` / `weight_cloud` group by (trial type, neighbour type)
instead: a short loop over a small product, vectorised inside each group.

**Neighbour types come from `species_at`, never `species[idxs]`.** A query pads
its rows with the pad index, one past the slot table, so the direct index raises
`IndexError` -- which it did, on the first run. `species_at` clamps, and the
clamp is neutral because a pad carries an infinite distance and so can never
reach a weight or an overlap verdict. That is the same contract `matrix_at` and
`CellList.data` already document, and the accessor existed for exactly this.

**The tail had to go off, and the species file already said so.**
`species_file.load_forcefield` refuses `interaction.tail_correction: true`
outright, but `potential.tail` defaults to True here, so a typed bulk run was
picking up a tail its own force field had forbidden -- and the wrong one:
`lj_tail_pref` is derived at sigma = epsilon = 1, while the mixture tail is
`sum_ij x_i x_j f(sigma_ij, eps_ij)`, composition-dependent and moving as the run
proceeds. It is forced off, and an explicit `tail: true` now raises rather than
being silently overridden.

## The gates, and why one of them alone would be worthless

**Reduction.** A typed run on a one-species unit file reproduces the untyped run
BIT FOR BIT -- identical counters every window, identical final positions,
identical N series [57, 81, 106, 117]. Asserted on the whole engine, not just the
pair energy.

**Discrimination.** Reduction alone is satisfied by a typed path that ignores the
table completely, because at one unit species there is nothing to ignore -- and
that is precisely the state phase 1 found the engine in. So the second gate is
that differing pairs produce differing physics: sigma 1.3 must not give the same
weight as sigma 1.0; an A trial and a B trial at identical distances against
identical neighbours must not get the same weight; B-B at 1.2 must overlap where
A-A does not. Without those, "reduces correctly" would be evidence of nothing.

The pair is the point. The reduction gate protects every existing single-species
result; the discrimination gate is the one that would have caught the original
defect. Neither is sufficient alone and the first is the one that feels like
enough.

## What is left: the sweeps, and why not now

All 7 remaining tier-2 failures are `branch` / `channel` / `channel_cyclic` with
a species file, refused by name:

> multi-species is not wired for the sweep samplers [...] The sweep builds its
> clique adjacency at a single scalar overlap floor and its intra-clique factor
> at sigma = epsilon = 1, both inside AST-locked kernels
> (samplers.branch.count_and_sample and ljbranch_weight_sum), so a typed sweep
> would silently flatten the force field.

Concretely, four things need per-pair parameters: `_build_adj` (adjacency is
non-overlap at one `lower`), `_intra_lj_factor` (intra-clique energy at
sigma = 1), `count_and_sample` (deliberately unpinned, so free to change) and
`ljbranch_weight_sum` (pinned, and re-baselining it again needs a stated cause).
The channel DP in `samplers/channel.py` and `channel_cyclic.py` needs the same.

**It is deferred on the plan's own reasoning, not on caution.** R7 collapses
those four sweep classes (six with the multi-species pair) into ONE dynamic
program at S5, selected by a mask policy -- the same reason the
`channel_move -> samplers.channel` inversion is still in `KNOWN_INVERSIONS`.
Threading a pair table through four classes that S5 replaces with one is work
done twice, and the seam would be built and then removed. The right order is R7
first, then type the single DP that results.

`test_the_sweep_refuses_a_species_file` pins the refusal for all three samplers,
so if the table is later wired into `build_potential` for `ljbranch` without the
kernels being touched, that test says so rather than the run quietly integrating
a single-sigma model.

**Tier 2 stays red at 7, deliberately.** Refusing typed sweeps in the validator
would remove those points from `accepted()` and turn tier 2 green. That is
green-by-redefinition. Green means "no known bugs"; there is one, it is named,
and it stays visible.

---

# S3 phase 3: the per-species activity, and the gate that could not see it

    tier 0   green -- and now includes the R14 docstring gate
    tier 1   595 passed, 45 skipped, 1 xfailed
    tier 2   7 failed, 39 passed -- unchanged; still the sweeps
    digests  13/13 UNCHANGED

## The defect

The exchange acceptance carried a single scalar `beta * mu` for the NET particle
a move adds or removes. In a typed run that scalar is `float(species.mu[0])` --
species 0's chemical potential -- so **every species was exchanged at the first
one's activity**. `species.z`, the authoritative state point (M10), reached only
the swap acceptance.

This is the third of the four changes `multispecies_plan.md` section 17 insists
are one change and not four: *fold `z_s` into the one-body weights*. The plan's
warning was exact -- *any subset gives a sampler that runs and is wrong*.

## Why two passing gates could not see it

The arbiter runs on species with EQUAL activities, so the term is identically
zero there. The reduction gate runs on ONE species, where it is zero for the same
reason. Between them they covered every other part of the typed path and were
blind to this one by construction.

**And the swap move hid it from the direct test too.** Degenerate species at
mu = [1.5, 0, -1.5]:

    swap active      x_s = [0.569, 0.311, 0.121]   looks correct
    swap suppressed  x_s = [0.323, 0.346, 0.332]   uniform: z_s never used

Swap equilibrates the composition at the correct activities, so the exchange's
error leaves no trace in the stationary distribution it produces. That is the
SECOND time in this stage that a passing gate meant nothing until an
equilibrating move was switched off -- the first was the arbiter itself.

After the fix, swap still suppressed:

    measured   x_s = [0.418, 0.321, 0.261]
    z_s demand x_s = [0.419, 0.327, 0.254]

## `moves.species_activity_term`

    sum_{inserted} log z_s  -  sum_{deleted} log z_s

Three things worth stating.

**It shares its provenance split with the M21 correction.** Both need the same
three-way partition -- reselected members contribute to neither sum -- so
`_partition_rows` is factored out and both call it. A second copy would be a
second place for the reselection case to be got wrong, and that case is not rare:
12 of 214 in one measured `hs_branch` window.

**A REPACK carries this term.** A repack removes k and inserts k, so N is
conserved and the single-species engine correctly has no activity term at all --
which is exactly why the typed one is easy to omit there. The k that leave and the
k that arrive need not be the same species: **an exchange that conserves particle
number does not conserve composition.**

**It reduces to the scalar cases exactly.** One species collapses the sums to
`net * log z_0`: `+log z_0` for an add, `-log z_0` for a remove, `0` for a repack.
The reduction is bit-identical and the digests say so.

## Gate S3's statistical arm is now complete

The plan names four. Their state:

| Gate | Status |
|---|---|
| degenerate mixture equals single species | green, bit-identical |
| ideal-gas `<N_s> = z_s V` per species | green, swap suppressed |
| swap stationary composition vs binomial | green (the arbiter) |
| uniform swap destination draw | green (`test_move_split`) |

`tests/engine/test_typed_state_point.py` holds the new two, and every test in it
suppresses swap. `test_swap_masks_the_exchange` pins the masking itself, so
nobody removes the suppression as unnecessary complexity and restores a test that
cannot fail.

## R14 finally has a gate

`restructure_plan.md` section 6 lists "docstring coverage" in the tier-0 row, and
Gate S2 asks for 100% on public functions. The check was never written, so the
requirement had no teeth and coverage had drifted to 96% unnoticed.

`tests/unit/test_docstring_coverage.py` is now in tier 0 and in the CI job. It
demands 100% of MODULE- and CLASS-level public definitions -- a closure inside a
function is an implementation detail whose contract is the enclosing docstring,
and demanding one for each would produce a hundred restatements that bury the
real contracts. Twelve definitions needed writing; two further gaps the gate found
on its first run were a module with no docstring at all
(`potentials/__init__.py`, whose imports ARE its job) and `species.py` not stating
its units convention.

Underscore-private definitions are outside the count but not outside the
practice: most in this package carry long docstrings, because a private function
with a subtle invariant is where a contract earns its keep. The exemption is about
what a gate can sensibly demand.

## Deferred, with reasoning: the L0-L7 file moves

S2 item 5 in the handover. **It should wait for S5, and the plan's own section 5
is why.** That target layout is not a set of moves; it embeds decisions that later
stages own:

    sampling/sweep.py        "ONE transfer-matrix DP (R7)"   -- R7 is S5
    forcefield/pairtable.py  "hard-core is eps=0"            -- R4 is S5
    system/particles.py      "the unification"               -- R3
    engine/window.py         one move loop, serial or per-cell -- R6 is S5

Moving `samplers/channel.py` and `channel_cyclic.py` into `sampling/` when R7 is
about to collapse them into one file is the same work-done-twice argument that
defers the typed sweeps, and this one additionally touches every import in the
package and in the tests.

Two places section 5 is already superseded, worth recording before someone
follows it literally:

- It puts the config classes under `run/config/` at L7. The ladder moved
  `config_fields` and `config_specs` to **L0** at S2 commit 7, because kernels
  below L6 depend on them and an L7 placement was an upward import. L0 is right
  and section 5 is stale on this point.
- `forcefield/units.py` is listed as "THE boundary (R10)". There is no `units.py`;
  the conversion lives in `species_file`. The boundary is real and singular, the
  filename is aspirational.

The purpose section 5 served -- *this is the tree to read* -- has in any case
already been met by other means: the tree was read at S2/S3, and the reading is
what produced five bugs, four of them silent.

---

# S3 phase 4: TIER 2 IS GREEN

    tier 0   18 passed
    tier 1   596 passed, 45 skipped, 1 xfailed
    tier 2   46 passed, 0 failed          <- the cluster gate
    digests  13/13 UNCHANGED

Every feature the validator accepts runs, and runs right. That was the finish
line, and it is met without weakening a gate, xfailing a failure, widening a
tolerance or regenerating a pin.

## A correction to what phase 2 claimed

Phase 2 said the sweep samplers were refused because "the sweep builds its clique
adjacency at a single scalar overlap floor and its intra-clique factor at
sigma = epsilon = 1, both inside AST-locked kernels". That was true of the BRANCH
sampler's kernels and **false of the channel sweeps**, and I did not check before
writing it.

`ChannelSweepMS` and `ChannelSweepLJMS` already existed, complete: per-pair
conflict floors AND per-pair 12-6 intra-clique energies via
`table_pair_energy_sq`. They were written for the MOF path, are unit-tested in
`test_channel_sweep_ms.py` / `test_channel_sweep_lj_ms.py`, and the only thing
missing was `ChannelBody.make_sweep` selecting them. One typed matrix point was
failing because of a dispatch, not because of a kernel.

## What each sampler actually needed

    rosenbluth      done in phase 2
    channel         a dispatch. `make_sweep` gained `floor` / `table` and picks
                    the MS classes when they are given.
    branch          real work: `_build_adj` decides conflict at the per-pair
                    floor, `_intra_lj_factor` reads the pair table.
                    `count_and_sample` was already unpinned;
                    `ljbranch_weight_sum` needed a second deliberate
                    re-baseline, recorded in the locked-invariants provenance.
    channel_cyclic  the missing cells.

## The two missing cells, and why the fix is a parameter

The sweep classes read as {hard-core, LJ} x {open, multi-species, cyclic} -- six
classes. But **cyclic and multi-species are independent properties of the
conflict graph**, so the honest shape is 2x2x2, and two cells were simply absent:
there was no multi-species cyclic sweep at all. The class lattice could not
express the combination, which is why it looked like six variants rather than a
gap.

Adding `ChannelSweepCyclicMS` and `ChannelSweepLJCyclicMS` would have made it
eight classes and left the same lattice. Instead the per-pair floor (and the
table, for LJ) became OPTIONS on the two cyclic classes, which collapses the
graph axis and makes the combination free.

**That is the graph half of R7, delivered.** The weight half -- integer counts
against log-space sums -- deliberately stays two classes, and the reason is a
measurement already in the tree: `_place_logfactor` carries a note that it is
INLINED in `_forward` and `_suffix` because the call plus its attribute lookups
cost 1.19x at ~10^6 calls per 250 sweeps. A weight policy reached through a
method call in the inner loop would pay exactly that. R7 has to answer this
before it can collapse further, and the plan's write-up does not mention it.

## One performance finding, fixed

Tier 2 went from 16 s to 106 s when the typed points started actually running --
6 points that used to raise in microseconds now do real work, so most of that is
honest. But two of them took 53 s and 36 s, and that was a genuine defect: my
first `_intra_lj_factor` grouped pairs by type combination in a Python double
loop, and that function is called once per CLIQUE while
`ljbranch_weight_sum` enumerates every clique. A loop over type pairs was sitting
inside the clique enumeration.

`table_pair_energy_vec` replaces it with one fancy-index gather --
`sigma[si, sj]` -- which is the same arithmetic in the same order and therefore
bit-identical to the scalar form, asserted rather than assumed. Tier 2: 106 s ->
68 s. The same gather replaced the same loop in `LennardJonesTable.pair_u_grid`.

Worth stating as a rule rather than an anecdote: a per-pair table lookup is a
gather, and writing it as a loop over type combinations is only acceptable where
the enclosing call is rare. Once per trial cloud is rare. Once per clique is not.

## What P3 delivered, end to end

    bookkeeping   insertion species drawn from p_trial, provenance by pool row
                  through the permutation, the M21 correction  -- MEASURED by the
                  arbiter
    activities    per-species z_s in the exchange acceptance, including repack
                  -- MEASURED with swap suppressed
    energy        per-pair sigma/epsilon/k_ij in every sampler: weights, overlap
                  floors, intra-clique factors, translate dU, swap dU
                  -- gated by reduction AND discrimination
    identity swap implemented, with a real dU

and the single-species chain is bit-identical throughout: 13/13 counter digests
unchanged across all four phases.

---

# S3 phase 5: tier 3's runnable half, run for the first time

The characterisation arm of tier 3 needs the history tags and is out of scope
(the git history is a separate project). The STATISTICAL arm needs nothing but
time, and **it had never been run against this tree**. Running it took 45
minutes and found three broken tests.

    first run   3 failed, 28 passed, 5 skipped   (45m43s)

## What was broken, and for how long

All three failed the same way, and not on physics:

    ValueError: sampler.channel_length = 'full' used to select the cyclic
    wall-to-wall rod. It is a different move, not a length, so it is now its own
    sampler: set sampler.type = "channel_cyclic" and drop channel_length.

    tests/engine/test_item01_detailed_balance.py::test_channel_cyclic_n0_ratio
    tests/engine/test_item01_detailed_balance.py::test_channel_cyclic_tetris_n0_ratio
    tests/unit/test_channel_2d.py::test_channel_2d_occupancy_ratio[True]

R16 made the cyclic rod its own sampler and retired the four `channel_length`
spellings that used to select it. `check_channel_full.py` was updated at S2
commit 2 -- its contract was deliberately inverted, and that is recorded. These
three were missed, and because they are `slow` they have been **erroring rather
than measuring ever since**, while reading in the test list as though the cyclic
rod's detailed balance was under test.

It was not. After the fix, it is:

    10 passed in 695.88s

That includes `test_channel_cyclic_n0_ratio` and
`test_channel_cyclic_tetris_n0_ratio`, which check P(N=0)/P(N=1) against the
exact equilibrium ratio to 5%, and both arms of the 2D occupancy ratio. **The
cyclic rod's detailed balance is verified here for the first time** -- the seam
is neither double counted nor dropped. The physics was fine; the config vocabulary
had moved out from under the tests.

## The comment that made it possible

Directly above the line that raises, `sim_setup` said:

> The four legacy spellings are still accepted so existing params files keep
> working; they are exactly equivalent [...] See
> test_channel_cyclic_spellings_share_one_run_id.

Every clause of that was false by the time it was read: the spellings are
refused, not accepted; and the test it cites does not exist. It stood for two
stages immediately above the `raise` that contradicts it.

That is worth more than a note about tidiness. A comment is what a reader checks
FIRST, and this one said the exact opposite of the code -- so anyone auditing
whether R16 had been followed through would have concluded, correctly according to
the comment, that legacy params still worked and nothing needed updating.
Corrected, with the history recorded rather than quietly deleted.

## Why this was worth 45 minutes

Three of the four tiers were green and had been for a while. The one that had
never been run held three tests that could not pass, on the sampler with the most
delicate geometry in the package -- the one whose seam has to be counted exactly
once. Nothing in tiers 0-2 could have found this, because the fault was in tests
that tiers 0-2 do not collect.

The general lesson for the CI design: **a tier that is never run is not a tier.**
Tier 3 is nightly in `.github/workflows/ci.yml`, which is the right cadence, but
nightly only starts counting once there is a repository to run it in.

## The re-run, green

    31 passed, 5 skipped, 714 deselected   (51m32s)

Tier 3's statistical arm is green for the first time. The five skips are the tests
that genuinely need the frozen tags (`ENGINEA_ROOT` and friends) plus the RASPA2
reference; they skip with instructions rather than failing, which is the behaviour
S1 designed and it works.

## Tier status, complete

    tier 0   green   18 passed (static, ladder, identity, R14, locked bodies)
    tier 1   green   596 passed
    tier 2   GREEN   46 passed, 0 failed        <- the cluster gate
    tier 3   green   31 passed, 5 skipped        <- statistical arm, first run
             ---     characterisation arm needs the history tags (out of scope)
    digests  13/13 UNCHANGED

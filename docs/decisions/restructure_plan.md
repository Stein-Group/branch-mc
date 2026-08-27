# restructure_plan.md

Append-only decision record for the consolidation of BRANCH-MC into one clean
package. Same protocol as `multispecies_plan.md`, `mof_raspa_plan.md` and
`cell_unify_plan.md`: decisions get an R-number and are never edited in place. A
superseded decision keeps its number and gains a pointer to the one that
replaced it.

Revision 1. Nothing here supersedes an M-, A- or U-decision; where this plan and
an older one disagree, the disagreement is called out explicitly and given an
R-number.

Goal: one package that handles hard spheres, Lennard-Jones, multi-species
Lennard-Jones and a rigid framework with a real force field as **one code path
with different numbers in one table** — not as four modes. Lean enough to read
end to end, documented well enough that reading it is sufficient, and defended
by a test suite that exercises every feature on every commit so no job has to be
run by hand to find out whether a release still works.

Non-goal: preserving compatibility with any run directory produced before this
plan (R9).

---

## 0. Baseline, measured

Everything below was measured on the uploaded `BRANCH_MC` tree, 2026-08-24,
numpy 2.4.4 / scipy 1.17.1 / ase 3.29.0 / Python 3.12. Numbers, not impressions.

| Quantity | Value |
|---|---|
| Default suite | 594 passed, 1 skipped, 1 xfailed, **101.96 s** |
| ruff, `gcmc/` only | 3 errors |
| ruff, repo-wide | 72 errors; 14 files need reformatting |
| mypy | 5 errors (1 missing `types-PyYAML` stub, 4 real) |
| CI | **none** — no `.github/`, no workflow of any kind |
| `gcmc/` | 16,859 LOC |
| `tests/` excluding `ref/` | 16,399 LOC |
| `tests/ref/engine0` + `engineA` | 14,726 LOC (two frozen package copies) |
| `orig/` | 5,116 LOC + a 416 KB CSV |
| Python LOC, whole tree | 57,570 |

Structural findings:

- **No import cycles** at module level. The layering is already almost right;
  the coupling that breaks it is `Config`, not the module graph.
- **Five callables hold ~4,400 lines**: `sim_setup.from_params` (1,207),
  `MC_engine.advance` (1,160), `decomposition.grid_up_the_system` (645),
  `parallel.advance_cell` (499), `neighbors.CellList` (864, class).
- **`Config` is a 72-field flat dataclass.** Every module takes `cfg`, so every
  module transitively depends on the union of all configuration. This is the
  single thing that makes "one library, one job" impossible today.
- **Almost no textual copy-paste.** A clone detector over `gcmc/` found a
  maximum of 5 duplicated 8-line windows, all inside `potentials/aux.py`. The
  duplication is *parallel implementations that diverged*, not copies — which is
  why the fix is consolidation and not deletion.

Parallel implementations found:

| Concept | Implementations |
|---|---|
| Channel sweep | `ChannelSweep`, `ChannelSweepMS`, `ChannelSweepLJ`, `ChannelSweepLJMS`, `ChannelSweepCyclic`, `ChannelSweepLJCyclic`, plus `mof_channel.ChannelWindowMS` and `channel_move.ChannelBody` |
| Lennard-Jones | `potentials/lennard_jones.py`, `potentials/lj_table.py`, `potentials/pair_table.py` |
| Metropolis loop | `MC_engine._advance_metropolis`, `mof_metropolis.advance_metropolis_mof` |
| Move loop | `MC_engine.advance`, `parallel.advance_cell` |
| Species concept | `species.SpeciesSpec`, `species_file.ForceField`, `topology.Species` |
| `max_core_floor` | defined in both `geometry.py` and `species.py` |
| Neighbour search | `neighbors.CellList` (core) and `scipy.spatial.cKDTree` (`mof_metropolis`) |
| Runner | `run/runner.py` and `scripts/mof/run_mof.py` |

Stale claims confirmed:

- `examples/schema.json` enforces `depth: maximum: 3`. The code enforces only
  `depth >= 0`. **The schema rejects runs the engine handles.**
- README states "Depth is capped at 3" three times, and carries the
  parallel-depth bias caveat in footnote 2; `sim_setup.py:1674` repeats it.
- `scripts/visualize_channel_moves.py` patches `MC_engine.cuboid_members`; the
  engine imports `cuboid_members_cl`. The script has been saving nothing.
  Recorded in `multispecies_plan.md` §10, never fixed.
- `check_cli.py` is xfailed against a launcher API that no longer exists.
- Zero `TODO`/`FIXME` markers in the tree, which is precisely why the above is
  invisible: every stale claim reads as authoritative prose.

Plan status inherited from `multispecies_plan.md`:

| Phase | State |
|---|---|
| P1 species file | done, gated, 43 tests |
| P2 config surface | done |
| P4 channel extraction | done, 14/14 bit-identical |
| P3 hot path | ~80% — column wired everywhere, **typed insert path not wired** |
| P5, P6, P7, P8 | not started |

---

## 1. Decisions

### R1 `orig/` and the frozen engines live in history, not in the tree

The repository's first commits build the real lineage, and each is tagged. The
working tree at HEAD carries none of them.

    commit 1   the original monolith, at repo root      tag v0-original
    commit 2   the engine0 snapshot                     tag v1-engine0
    commit 3   the engineA snapshot                     tag v2-engineA
    commit 4   delete all three; the clean package      tag v3

`git worktree add /tmp/orig v0-original` materialises any of them from local
objects, no network. GitHub renders each tag with a browsable file tree, so
"version 0" exists publicly and permanently without being carried forward.

This removes **~19,800 LOC** from the working tree through one mechanism.

Three consequences, all of which are work:

1. Eighteen `tests/scripts/check_*.py` hardcode `../../orig`. They gain an
   `ORIG_ROOT` environment variable and skip with a reason when it is unset. CI
   does one `git worktree add`; locally it is `make orig`.
2. **`orig/g3_analysis_data.csv` is not history — it is live data.** 416 KB, read
   at runtime by the 3D pmf auxiliary through `auxiliary.g3_csv`. It moves to a
   shipped `data/` directory. This is the one item that would silently break a
   real run if it went into a tag with everything else.
3. This must happen before anything is pushed. Retrofitting a root commit onto
   existing history needs `git rebase --root` and rewrites every hash.

### R2 There is no MOF workstream

MOF is not an extension. It is the general case of which hard spheres and bulk
Lennard-Jones are reductions. Hard-core → LJ → LJ + force field is one axis with
two knobs, not three features:

| System | Pair interaction | Who moves |
|---|---|---|
| hard-core bulk | eps = 0, floor = sigma | everyone |
| LJ bulk | eps = 1, sigma = 1 | everyone |
| QA matrix | eps = 0 fluid-obstacle, floor > 0 | fluid only |
| multi-species LJ | full per-pair table | everyone |
| MOF + force field | full per-pair table | guests only |

Every row is the same simulation with different numbers in one table and a
different mobility column. No row is a mode. `gcmc/mof_*.py` and `scripts/mof/`
were written before the substrate that makes this true existed, and are migrated
onto it rather than kept beside it.

### R3 Mobility and species are orthogonal properties of a particle

Today `CellList.is_matrix` is one boolean meaning two things at once: *frozen*
and *contributes no energy*. The second is enforced in the hot loop by writing
`np.inf` over the distance so the Boltzmann factor comes out 1
(`MC_engine._is_matrix_neighbor`, `_lj_singleparticle_weight`).

It splits into two independent columns:

- **mobility** — bookkeeping. May this be inserted, removed, moved, counted in N?
- **species id** — physics. Which row of the pair table?

The `np.inf` overwrite then disappears. QA becomes "a frozen species whose eps
with the fluid is zero and whose sigma is not" — a number in a table, not a
branch in the hot loop. MOF becomes "a frozen species whose eps is real."

`multispecies_plan.md` §11 already reached this conclusion and stopped short of
acting on it: *"that flag means 'enters the overlap test, never the energy sum'
… which is orthogonal to type. A framework atom has a species too."* R3 acts on
it.

### R4 Hard-core is `eps = 0`, not a class

`exp(-beta * 0)` is exactly `1.0`, so a hard-core run through the pair table is
bit-identical to the current unit-weight path by construction, not by tolerance.
The overlap test is already separate and already per-pair (P3 part 2).

This collapses `HardSphere`, `LennardJones` and `LennardJonesTable` into one
table-driven potential. The existing gate — `table_pair_energy_sq` at one type
bit-identical to `lj_pair_energy_sq` — already proves half of it.

**This has a hot-path cost that must be measured, not argued.** A table lookup
per pair replaces a scalar. `neighbors` stores species as int64 specifically "so
it indexes the (T, T) core-floor and pair tables without a cast on the hot path",
which suggests it was anticipated, but anticipation is not a measurement. See
Gate S5.

Note this requires the species-file validator to admit `eps = 0` for a
hard-core species. Section 4.1 of `multispecies_plan.md` currently requires
epsilon positive. That rule narrows to "positive for any species that carries an
energy"; a species declaring `interaction: hard_core` states `eps = 0`
explicitly and is validated for a positive sigma instead.

### R5 One neighbour structure

`CellList`, everywhere. `mof_metropolis`'s `cKDTree` path is deleted, not ported
— `multispecies_plan.md` P3 already forbade porting it. `tests/engine/
test_no_kdtree_in_hot_path.py` exists; its scope widens from the core engine to
the whole package.

### R6 One move loop

`advance` (1,160 lines) and `advance_cell` (499) are the same algorithm over two
tori. P4 built the seam and proved it 14/14 bit-identical; `advance_cell` still
does not call it. Serial versus parallel becomes a geometry argument, exactly as
`channel_move` already separates `side` (proposal torus) from `box` (physics).

### R7 One sweep

Six sweep classes are a 2x3 matrix (hard-core/LJ x plain/multi-species/cyclic)
expressed as an inheritance lattice in one 1,420-line file. They become one DP
with two policies:

- hard-core versus LJ is a **weight policy** (integer counts versus log-space
  Boltzmann sums)
- cyclic versus open is a **mask policy** (the seam)
- single-species is `T = 1`

The DP itself is already type-blind once masks and per-bit log-factors are built
(`multispecies_plan.md` P5), so this is a packaging change to code that is
already general.

### R8 `Config` splits, and identity becomes structural

Five sub-configs:

    PotentialSpec    force field reference, T, tail, auxiliary
    SamplerSpec      type, depth, trials, cavity/channel geometry
    EnsembleSpec     muVT / NVT / QA, mu or N, box, dimension, framework
    ExecutionSpec    serial/parallel, cells, buffer, workers
    IOSpec           cadence, paths, checkpointing

The run fingerprint is then the hash of the first three and never the last two.
Not a curated list — a partition. A new physics knob is identity-bearing because
of *where it was declared*.

A CI test asserts the partition is total: every field belongs to exactly one
side, so a new field cannot land in neither.

### R9 Backward compatibility with existing run ids is abandoned

Dipesh, 2026-08-24: the published paper is addressed by older commits; the second
paper is its own thing. No existing run directory needs to resume.

This is load-bearing rather than merely permissive. Four special cases in
`identity_dict` exist solely to keep old digests stable (`channel`,
`aux_cutoff`, `box_sides`, `species_identity` skip logic). All four are deleted.
The frozen digest pins in the test suite are regenerated once, deliberately, and
labelled as regenerated.

### R10 One units boundary

Exactly one place converts a physical-units force field into engine-reduced
units, and the reference scale enters the fingerprint. Two conversion sites is
the bug class M3 was written to prevent, and with MOF first-class it becomes
easy to grow a second one.

`forcefield/units.py` is that place. Nothing downstream of it sees Ångström or
Kelvin.

### R11 The tail correction is derived, not declared

Bulk LJ needs it; a pore must not have it. Today `pair_table` hard-raises and
`sim_setup` auto-offs under a matrix — two mechanisms for one fact. It becomes:
off iff a frozen species is present. `potential.tail` remains settable only to
force it off in a bulk run.

### R12 The layer ladder is enforced, not documented

Imports may only go down. A ~30-line CI test walks the AST, holds the ladder as
data, and fails any upward import.

This is the decision that keeps the package lean *after* it is made lean. Nothing
prevents re-bloating today, and nothing did.

Layer numbers are **not** in package names — `l1_forcefield` would churn every
import on any later move and read badly in tracebacks. Each package's
`__init__.py` states its layer in the first docstring line; the CI test holds the
authoritative ladder.

### R13 The smoke matrix is generated, not written

A hand-written list of modes goes stale the first time a feature is added.
The CI matrix enumerates the legal product of potential x sampler x depth x
ensemble x dimension x execution **from the validator itself**, and runs each for
2–4 windows.

Consequence, which is the point: a new feature cannot be added without appearing
in the matrix, because the validator is what decides the matrix.

`tests/scripts/gen_p4_snapshot.py` is the right primitive already — the engine
frozen against itself, 14 modes in five seconds. It is simply pointed at a
fraction of the surface. `_harness.REF_MODES` covers 14 serial modes and misses
parallel, reweight, pmf, NVT, QA x sampler, 2D channel, cyclic channel, typed
multi-species, the whole MOF path, migrate, launcher sweep and resume.

### R14 A function without a contract does not ship

Module docstrings in this codebase are its best feature: they record *why*,
including the measurements that forced each design. **They are preserved
verbatim.** What is missing is the layer below them.

Every public function gains a numpydoc contract: Parameters, Returns, Raises,
**Units**, Invariants. Understanding a signature must not require reading the
essay above it.

Units are stated on every physical quantity. R10 gives one boundary, so every
function is unambiguously `reduced` or `Å-K` and only one module sees both.

Enforced by a docstring-coverage gate in CI tier 0.

### R15 Every merge is a reduction gate

Each specialisation being merged away is a provable reduction of the general
case. So:

1. Build the general form alongside the specific one.
2. Prove it reduces — **bit-identically**, not statistically.
3. Delete the specific one.
4. One merge per commit. Never two at once.

This is not new discipline; it is what P4 did (14/14) and what `pair_table` did
against `lj_pair_energy_sq`. R15 makes it the rule for the whole consolidation.

### R17 Parallel is exact at every depth; the density-bias caveat is retired

Dipesh, 2026-08-24, on being asked directly whether the measurement was current:
*"The cap was bogus. Parallel is tested and works perfectly. Detailed balance is
satisfied. I've since run this properly and the bias isn't there."*

What this retires, in three places:

- `README.md` footnote 2: *"parallel depth >= 1 is approximate -- the cell
  decomposition satisfies global but not detailed balance, leaving a density
  bias set by the cavity/cell ratio (~+19% at cell = 2.7 cavity widths, +0.9% at
  10x). Use `cell_length >= 20x` the cavity, or serial mode."*
- `README.md` "Known limitations", the same claim in short form.
- `sim_setup.py`, the comment carrying the same numbers, citing
  `cell_unify_plan.md` U3.

Two distinct claims were welded together in that footnote and both are gone: a
STRUCTURAL one (detailed balance is not satisfied under decomposition) and a
MEASUREMENT (+19% at 2.7 widths). The structural claim was the reasoning behind
the measurement; later runs did not reproduce the bias.

**Why this needed an explicit decision rather than a quiet edit.** R17 deletes a
number with error bars on the strength of newer unpublished runs. Deleting a
measurement because it is inconvenient, or because a related claim (the depth
cap) turned out to be bogus, is exactly the drift this plan exists to stop -- so
the removal is attributed, dated, and quotes the sentence that authorised it. If
the bias is ever seen again, this entry is where to start.

U3 in `cell_unify_plan.md` is NOT edited: it is an append-only record of what was
measured at the time, and rewriting history there would destroy the audit trail.
It is superseded by this entry, not wrong about what it observed.

`describe_parallel` still prints the shortest cell in cavity widths. That number
is now a PERFORMANCE figure -- a small ratio means more border work per live cell
-- and is kept on those grounds.

---

## 2. Invariants this work must not break

Carried from `multispecies_plan.md` §2, minus the ones R9 retires.

- `ljbranch_weight_sum` in the clique sampler keeps its AST hash.
- `_advance_metropolis` stays bit-identical for cubic single-species runs
  **until** R4/R5 merge it, at which point the reduction gate is the bar.
- `table_pair_energy_sq` at one type stays bit-identical to `lj_pair_energy_sq`.
- The overlap convention stays centralised in one module. One convention, one
  place. A pair floor changes what number is handed to it, never where the
  comparison lives.
- No inline pair-energy formula outside the force-field layer; the second AST
  guard already forbids it.
- No frozen pin is widened to make a failing test pass.

Retired by R9: schema-2 checkpoint loading, and the stability of any existing
run id.

Added here:

- No upward import (R12).
- No `cKDTree` anywhere in `gcmc/` (R5).
- No physics field outside the three identity-bearing sub-configs (R8).

---

## 3. Stages

Ordered against the stated research plan: read the tree line by line, run the LJ
A-B demixing jobs, then MFM-300(In) on the cluster.

The restructure lands **before** the reading, so the tree is read once. The
semantic merges land **after** the demixing science, so the first physics is not
blocked behind them.

    S1  safety net + deletions + history tags      no physics risk
    S2  mechanical restructure                     no behaviour change
    --  READ THE TREE HERE
    S3  close P3: the typed insert path            physics
    S4  P5 + P6  ->  LJ A-B demixing runs          physics
    S5  semantic merges: MOF, sweeps, hard-core    physics
    S6  P7 parallel channel + MFM-300(In)          physics

### S1. Safety net, deletions, history

Nothing else starts before this. It is the instrument every later stage is
measured with.

- Four CI tiers (§6), including the generated smoke matrix (R13).
- History tags and `orig/` removal (R1), `ORIG_ROOT`, `g3_analysis_data.csv`
  relocated to `data/`.
- Free deletions (§7).
- Stale claims corrected: the schema depth cap, the README depth cap, the
  `visualize_channel_moves` patch point, `check_cli` either fixed or deleted.
- Repo-wide lint and format clean; the four real mypy errors fixed;
  `types-PyYAML` added to the dev extra.

**Gate S1.** Suite green at or under the current 101.96 s. Smoke matrix green,
and its enumeration provably covers every mode `from_params` accepts — asserted
by comparing the generated matrix against the validator's own accept set, not by
eye. Zero ruff errors repo-wide. Zero mypy errors. `orig/` absent from the tree
and retrievable by tag in one command.

### S2. Mechanical restructure

Pure moves and mechanical splits. No algorithm changes.

- Files into the L0–L7 layout (§5).
- `Config` split into five sub-configs (R8).
- Fingerprint rebuilt as a structural partition (§4).
- The ladder test (R12).
- Docstring contracts written as each file moves (R14) — this is the cheapest
  moment to write them, because every file is being touched anyway.

This touches every import in 16,399 LOC of tests. It must be one atomic change,
gated by S1's matrix.

**Gate S2.** Full suite green. Smoke matrix bit-identical to S1 — same
positions, same counter streams, all modes. The fingerprint is the one
deliberate exception: run ids change once, by design (R9), and the change is
recorded with the old and new digests for the modes in the matrix. Ladder test
green. Docstring coverage 100% on public functions.

*This is the tree to read.*

### S3. Close P3

The typed insert path, which `multispecies_plan.md` §17 leaves as the one
remaining item and insists is **one change, not four**: draw a trial's species
from `p_s` when the cloud is built, assemble `chosen_species` from the pool by
row, fold `z_s` into the one-body weights, add the M21 term to `neg_log_acc`.
Any subset gives a sampler that runs and is wrong.

Two things to carry in from the record:

- **M21 is a derivation that has never been measured.** The plan says so in
  capitals. It is the highest-uncertainty item in the project. Its arbiter —
  degenerate species with skewed `p_trial`, where the stationary composition must
  be uniform for any `p_s` — is already written and must be run **first**.
- **The permutation is currently discarded.** M21 decides provenance by pool row,
  but `pool = pool[rng.permutation(len(pool))]` throws the permutation away. The
  caller must apply it to the row bookkeeping in the same breath. Wiring the
  insert path without this is wrong at a rate set by the member-reselection
  frequency — measured at 12 of 214 in one `hs_branch` window, which is exactly
  the size of error that looks like a plausible isotherm.

**Gate S3** (Gate P3 of the multispecies plan): the arbiter first, then the four
statistical gates — degenerate mixture equals single species, ideal-gas limit
`<N_s> = z_s V` per species with Poisson tolerance, swap stationary composition
against binomial error, and the uniform swap-destination draw. Single-species
bit-identity is covered by the smoke matrix.

### S4. P5 + P6, and the demixing runs

P5 (multi-species channel sweep with the auxiliary) and P6 (structure in and
out) as written in `multispecies_plan.md`, unchanged. P6 matters here rather
than later: a warm start is what makes a demixing run at useful density
tractable at all.

**Gate S4** is Gate P5 and Gate P6 as written, with one addition: the auxiliary
control arm must be an **exact** no-op, not statistically close. If it is not,
the correction is inserted in the wrong place, which is the open Q6.

At the end of S4 the LJ A-B demixing jobs run. Sweeps go out as launcher array
jobs, which work today.

### S5. Semantic merges

The consolidation proper, each merge its own commit with its own reduction gate
(R15), in this order because each makes the next smaller:

1. `pair_table` absorbs `lennard_jones` and `lj_table` (R4). Gate: bit-identity
   at one type, already half-proven.
2. `pair_table` absorbs `hard_sphere` as `eps = 0` (R4). Gate: bit-identity of
   every hard-core mode in the smoke matrix, **plus a hot-path timing** against
   the S1 baseline. A regression over 10% is a finding, not a pass.
3. The particle table: mobility and species split (R3). Gate: every QA mode in
   the matrix bit-identical, with the `np.inf` path deleted.
4. `mof_field` absorbs into the pair table plus a cell-list query; `cKDTree`
   deleted (R5). Gate: `test_mof_field` and `test_mof_ref_energy` pass
   unmodified against the merged path.
5. `mof_metropolis` absorbs into the one metropolis loop. Gate: the MOF
   metropolis tests pass unmodified; the frozen `_advance_metropolis` reduction
   is bit-identical.
6. Six sweeps become one (R7). Gate: all fourteen smoke-matrix channel modes
   bit-identical.
7. `mof_channel` absorbs into `channelbody`. Gate: `test_mof_channel` unmodified.
8. `advance_cell` calls the shared loop (R6). Gate: every parallel mode in the
   matrix bit-identical.
9. `run_mof.py` absorbs into `runner`. Gate: a MOF run checkpoints, resumes and
   reproduces an unkilled run bit-identically — the property `run_mof` was
   written for.

**Gate S5.** `gcmc/` under 10,000 LOC. Zero parallel implementations from the §0
table remain. Smoke matrix bit-identical throughout, except where a gate above
names an expected change. Hot-path timing within 10% of the S1 baseline, or the
regression explained.

### S6. P7 and MFM-300(In)

P7 (parallel channel: species through `grid_up_nd`, the rod floor from M4, the
eroded-cell draw) then the framework runs.

**One arithmetic check must happen before S6 is scheduled, and it is currently
unknown**: how many atoms MFM-300(In) has at the supercell size actually used,
against `CellList`'s growth strategy and the per-worker pickle budget the
weak-scaling work put at roughly 125 MB. If the framework is large enough to
break either, R3's single-table design needs a frozen-species storage variant —
same interface, different backing. That is a design branch, not a surprise, and
it is cheap to decide once the number exists.

---

## 4. The fingerprint, rebuilt

### What is wrong with it now, measured

Probed by mutating one parameter at a time against a reference LJ-branch config
and comparing run ids.

**A live collision:**

    potential.tail  True -> False     SAME run id

Two different Hamiltonians, one directory, and they will resume each other.
`lj_tail_pref` is a `Config` field and is simply absent from `_IDENTITY_FIELDS`.

**Eight of twenty-five keys hash as literal `null`** on an ordinary LJ run:
`seed`, `eta`, `gamma_max`, `delta_eta`, `target_n`, `pore_geometry`, `pore_H`,
`pore_R`. They contribute nothing but make the digest depend on which inactive
features happen to exist. The treatment is also inconsistent: `channel`,
`aux_cutoff`, `box_sides` and `species_identity` get careful skip logic when
inactive; these eight do not.

**Four keys are derived from others already in the set:**

| Key | Derived from |
|---|---|
| `potential_type` | `potential` + `bias` + `sampler`, and `sampler` is hashed too |
| `branch` | `sampler` |
| `channel` | `sampler` |
| `upper_cutoff` | identical value to `rc`; 2.5 is hashed twice |

**Absent but physics-bearing:** `lj_tail_pref`, `pore_axis_len` (while `pore_H`
and `pore_R` are present), `diameter`, `g3_csv`.

**A contradiction in the file's own reasoning.** `g3_csv` is excluded because it
is "data, not a physics parameter". `aux_cutoff` is included because "it changes
the proposal distribution, hence the Markov chain, even though the sampled
ensemble is unchanged by construction." A different g3 table changes the
proposal distribution too. One of the two rules is wrong; §4.2 below picks one.

**`channel_cyclic` is a dead field.** It exists on `Config`, but the schema
rejects the key, so the cyclic rod is unreachable from a params file. Either it
becomes settable or the field and its classes go.

**Two documented sweep axes work by accident.** `trial_box_side` and `trials` do
change the id, but neither is in the identity list; they leak through derived
keys (`cavity_radius`, `no_neigh`). It happens to be correct today. Nothing
guarantees it stays correct.

Also on the record, from `multispecies_plan.md` §9: the pin
`test_the_untyped_fingerprint_does_not_gain_a_null_key` asserted a digest
`1a24fe1c17aa` that is not reproducible from any variant of the identity dict,
and its provenance is lost.

### The rebuild

A hand-curated tuple produced every item above and will drift again. The fix is
mechanical, not editorial (R8):

    identity  =  hash(PotentialSpec + SamplerSpec + EnsembleSpec)
    excluded  =  ExecutionSpec, IOSpec, control              by construction

Rules, each of which kills one finding:

1. **Inactive keys are absent, never null.** A sub-config only carries the fields
   its mode has, so there is nothing to skip and no null to hash. The four
   backward-compatibility skips die with R9.
2. **Derived values are never hashed.** Only the axes are: `potential`, `bias`,
   `sampler`. `potential_type`, `branch`, `channel`, `metropolis` and
   `two_dim` are properties, not fields.
3. **One name per quantity.** `rc` and `upper_cutoff` are the same number and
   become one field.
4. **The rule for proposal knobs, stated once:** anything that changes the
   Markov chain is identity-bearing, even when the sampled ensemble is invariant.
   That makes `trial_box_side`, `trials`, `aux_cutoff` and the **content digest**
   of the g3 table identity-bearing — the digest, not the path, so moving the
   file does not re-address the run.
5. **Frozen coordinates keep the existing digest treatment**, rounded to 1e-6 so
   I/O noise cannot split one run into many. This extends unchanged from the QA
   matrix to a MOF framework.
6. **One CI test asserts the partition is total.** Every field of every
   sub-config is on exactly one side. A new field cannot land in neither.

Digest pins are regenerated once at S2 and labelled as regenerated, with the
measurement recorded, so the `1a24fe1c17aa` situation does not repeat.

---

## 5. Target layout

Roughly 35 files, none over about 400 lines, one job each.

    gcmc/
      types/          L0   vocabulary
        arrays.py            FloatArray, IntArray, BoolArray, RNG
        protocols.py         NeighborQueryable, Weigher, Proposal
      forcefield/     L1   what interacts, and how strongly
        speciesfile.py       YAML load + validate; schema 2 canonical,
                             schema 1 read through an adapter
        units.py             Å-K -> reduced. THE boundary (R10)
        pairtable.py         (T,T) sigma, eps, core_floor; hard-core is eps=0
        auxiliary.py         short-range multiplier, g3 table
        eos.py               Peng-Robinson, Carnahan-Starling (validation refs)
      space/          L2   where things are
        box.py               per-axis box, wrap, minimum image
        celllist.py          the one neighbour structure
        overlap.py           the one overlap convention
        cavity.py            membership queries
        partition.py         cell plan + grid_up_nd
      system/         L3   the configuration being sampled
        particles.py         positions + species + mobility   <- R3
        chainstate.py        counters, step, RNG
        energy.py            total and delta energy
      proposals/      L4   what move to try
        select.py            move-type choice, swap destination (M8)
        translate.py
        cavitybody.py
        channelbody.py       the P4 extraction
      sampling/       L5   how to weight a trial cloud
        trialset.py
        weighers.py
        rosenbluth.py
        clique.py            branch enumeration
        sweep.py             ONE transfer-matrix DP (R7)
        reweight.py          psi6
      engine/         L6   the move loop
        window.py            advance() — serial or per-cell (R6)
        acceptance.py        the acceptance algebra, including M21
        parallel.py          worker orchestration only
      run/            L7   making a params.json runnable
        config/
          potential.py  sampler.py  ensemble.py  execution.py  io.py
          schema.py
        identity.py          the fingerprint (§4)
        io.py                checkpoint, trajectory, log
        runner.py            the one loop
        launcher.py          sweeps
      analysis/       off-ladder; imports L0–L3 only
        rdf.py  widom.py  blocking.py  isotherm.py

    data/             g3 table and other shipped assets (R1)
    ff/               force fields and species files
    tests/            unit / engine / matrix / characterisation
    docs/             Sphinx source

The four load-bearing files:

- **`system/particles.py`** — the unification. Every system from a hard-disk
  fluid to MFM-300(In) is a row count in one table.
- **`forcefield/pairtable.py`** — the only place a pair interaction is defined.
- **`sampling/sweep.py`** — six classes become one DP with two policies.
- **`engine/window.py`** — one move loop; serial versus per-cell is an argument.

---

## 6. CI/CD

Four tiers, split by wall clock so the fast ones can run on every push.

| Tier | Runs on | Budget | Content |
|---|---|---|---|
| **0** | every push | ~30 s | ruff, ruff format, mypy, docstring coverage, the ladder test, schema-vs-docstring, the identity-partition test |
| **1** | every push | ~2 min | the current default suite |
| **2** | every push | ~5 min | the **generated** smoke matrix (R13) |
| **3** | nightly | ~40 min | characterisation tier against the tags, statistical gates, everything marked `slow` |

Tier 2 is the one that replaces running jobs by hand. Each mode runs 2–4 windows
and asserts **completion and invariants**, not physics: species totals conserved,
slot alignment held, no negative or non-finite weight reaching the sweep, the
counter stream reproducing its committed digest. Physics lives in tiers 1 and 3.

Three details that will otherwise bite:

- **`OMP_NUM_THREADS` is pinned** in the runner before any timing is trusted.
  Already flagged in the weak-scaling work; it becomes a CI environment
  assertion, not a habit.
- **The equilibrated library needs a synthetic fixture.** The folder is empty in
  the repo and full in deployment. `tests/unit/test_initial_config.py` already
  builds a small library in a tmpdir and tests the resolver against it — that
  pattern extends to cover the typed and framework cases. Separately, a
  deployment-time check validates the *real* library's shape without CI ever
  needing the files.
- **Tier 3 does `git worktree add`** for the three tags (R1), so the
  characterisation oracles keep working with none of that code in the tree.

---

## 7. Deletion ledger

Absorbed by the merges (S5):

| File | LOC | Absorbed by |
|---|---:|---|
| `mof_metropolis.py` | 511 | the one metropolis loop |
| `mof_channel.py` | 486 | `channelbody` |
| `mof_field.py` | ~250 | `pairtable` + a cell-list query |
| `potentials/lj_table.py` | ~170 | `pairtable` |
| `potentials/lennard_jones.py` | 139 | `pairtable` at one type |
| `potentials/hard_sphere.py` | 82 | `pairtable` with eps = 0 |
| `io_framework.py` | 58 | `run/io` |
| `topology.py` | 134 | unused; the species column supersedes it |
| `scripts/mof/run_mof.py` | 367 | `runner` |
| **Subtotal** | **~2,200** | |

Plus most of the 2,188 LOC of `tests/unit/test_mof_*.py`, which become
parametrised cases of existing tests rather than a parallel suite.

Deleted outright (S1):

| Item | Size |
|---|---|
| `tests/ref/engine0` + `engineA` | 14,726 LOC → tags |
| `orig/` | 5,116 LOC → tag (CSV to `data/`) |
| `decomposition.py` legacy 2D helpers | ~1,000 LOC; the engine calls only `grid_up_nd` (79 lines) — the rest is reachable only from `check_decomposition.py`, which moves to the tag-driven tier |
| `scripts/sweetspot_plot_bkup.py` | 361 LOC; 53-line diff from the live file, and the source of most of the 72 lint errors |
| `gcmc/cluster_diff.txt`, `cluster_diff_sampler.txt` | scratch diffs inside the package |
| `mnt/user-data/outputs/gcmc/` | a leaked agent output directory, committed |
| eight plan `.md` files at repo root | gitignored yet committed; they move to `docs/decisions/` |

---

## 8. Documentation

- **Module docstrings preserved verbatim.** They are the best thing in the
  codebase and they record why, including the measurements that forced each
  choice.
- **Function contracts** in numpydoc: Parameters, Returns, Raises, Units,
  Invariants (R14).
- **Sphinx + autodoc + napoleon**, not Doxygen. Doxygen renders Python poorly;
  the existing `Doxyfile` produces a weak result. If Doxygen is a hard
  requirement, `doxypypy` bridges it, but the recommendation is Sphinx.
- **The append-only decision records stay** and move to `docs/decisions/`. The
  M/A/U/R numbering is the project's institutional memory and is worth more than
  most of the code.
- **Comments are audited during S2**, when every file is being touched anyway.
  The specific class to hunt: prose that reads as authoritative and is stale.
  Parallelisation works at any depth; the tree says otherwise in four places.

Reading target: `gcmc/` goes from 16,859 LOC to **under 10,000**. That is a tree
that can be read end to end in a sitting or two, which is the actual requirement.

---

## 9. Open questions

- **Q-R1.** MFM-300(In) atom count at the supercell actually used. Blocks the S6
  scheduling decision and, in the bad case, forces a frozen-species storage
  variant behind the R3 interface. Cheap to answer, so answer it before S5 ends.
- **Q-R2.** `channel_cyclic` is unreachable from a params file. Make it settable,
  or delete the field and the two cyclic classes? The cyclic rod is permanently
  rejected under parallel (M4) but is legal serially, so this is a real feature
  that has been dark. Needs a decision before R7 collapses the sweeps.
- **Q-R3.** Does `diameter` differ from `lower_cutoff` on any legal config? If
  not, R8 folds them into one field; if so, both are identity and the difference
  needs documenting. Ten minutes with the validator settles it.
- **Q-R4.** Q5 and Q6 from `multispecies_plan.md` remain open and are not
  resolved here. Q6 in particular gates S4: the auxiliary control arm must be
  exact, not close.

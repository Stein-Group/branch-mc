# S1 status report

Measured on the working tree, 2026-08-24. Every number below was produced by
running the thing, not by inspection.

## Gates

| Gate | Before | After |
|---|---|---|
| ruff, repo-wide | 72 errors | **0** |
| ruff format | 14 files unformatted | **0** |
| mypy | 5 errors | **0** |
| Default suite | 594 passed / 101.96 s | **603 passed / 146 s** |
| Tier 0 (ladder + identity) | did not exist | **9 passed, 1 xfailed** |
| Tier 2 (feature matrix) | did not exist | **30 passed, 20 failed** (see below) |
| CI | none | **4 tiers written** |

The default suite grew by 9 tests and by ~45 s. The extra time is the two new
tier-0 modules plus the identity work; it stays inside the tier-1 budget.

## Bugs found and FIXED

1. **Run-directory collision.** `potential.tail` was absent from the identity
   set, so `tail: true` and `tail: false` -- two different Hamiltonians -- hashed
   to the same id and would resume each other. `lj_tail_pref` and
   `pore_axis_len` are now identity-bearing. Run ids moved once, deliberately
   (R9). One pin regenerated: `fbdc3125c74a` -> `0c683de8d49b`, with the reason
   recorded inline so this digest keeps its provenance.

2. **The schema rejected runs the engine handles.** `examples/schema.json`
   enforced `depth: maximum: 3`; the code enforces only `depth >= 0`. Cap
   removed; the matrix now exercises depth 4 on every sampler that takes one,
   and 89 accepted points use it.

3. **The channel visualiser had been saving nothing.** It patched
   `MC_engine.cuboid_members`, a name the engine stopped importing at the
   cell-list migration. It ran, exited zero, and wrote an empty trace every
   time. Repointed at `channel_move.cuboid_members_cl`, the live call, with the
   signature change (CellList, not a coordinate array) carried through.

4. **Four `Optional`-narrowing errors** in `sim_setup` and `MC_engine`, fixed by
   making the "pmf implies an auxiliary block exists" invariant explicit rather
   than silencing the checker.

## Bugs found and PINNED (not S1 work)

The generated matrix found four broken features in its first eight seconds.
None was covered by any existing test.

| Failure | Points | Stage |
|---|---|---|
| `RuntimeError: unhandled move choice 'swap'` | 11 | S3 |
| Reweight: `kdtree is None` | 5 | S5 |
| pmf auxiliary: `kdtree is None` | 3 | S5 |
| metropolis+pmf accepted by validator, refused by engine | 1 | S2 |

The middle two are one root cause: the cell-list migration removed the KD-tree
and neither reweight nor the pmf auxiliary was ever ported. **This reframes R5**
-- "one neighbour structure" is not cleanup, it is the repair of two features
that currently cannot run at all.

The first means every multi-species run crashes within a handful of steps:
`moves.py` emits `"swap"`, `MC_engine.advance` has no branch for it.

## Architecture finding

The layer ladder found a flaw in the target layout itself. Three of the four
upward imports are `species` and `pair_table` reaching into `geometry` for
`max_core_floor` and the overlap predicate -- both functions of sigma, hence
FORCE FIELD concepts. `restructure_plan.md` section 5 placed `overlap.py` at L2;
it belongs at L1. After the move: L2 computes distances, L1 owns the floor, L3
applies the test, nothing reaches up.

Only four upward imports exist in the whole package. All four are in
`KNOWN_INVERSIONS`, each tagged with the stage that removes it, and
`test_no_stale_allowances` fails when an entry stops being needed -- so the list
can only shrink.

## Deletions

| Item | Effect |
|---|---|
| `scripts/sweetspot_plot_bkup.py` | 361 LOC; was the source of 24 of 72 lint errors |
| `gcmc/cluster_diff*.txt` | scratch diffs inside the package |
| `mnt/user-data/outputs/` | a leaked agent output directory, committed |
| `orig/` | 5,116 LOC out of the tree, retrievable from tag `v0-original` |

`orig/g3_analysis_data.csv` was moved to `data/` FIRST: it is read at runtime by
the 3D pmf auxiliary, so it is shipped data, not history. This was the one item
that would have silently broken a real run.

Characterisation verified both ways: **passes** with `ORIG_ROOT` set and `orig/`
absent from the tree; **skips with instructions** when it is not set. The skip
lives in the harness, not in the scripts -- they run as subprocesses, where a
skip is indistinguishable from a crash.

## Second pass: the rest of the deletions

**`tests/ref/engine0` and `engineA` to tags.** 14,726 LOC out. The resolver
generalised from `_orig.py` to `_frozen.py`, which now handles all three
artifacts through one `Frozen(tag, env, legacy)` record and generates its own
instructions -- so a message can never name a tag and an env var that do not
belong together.

Critically, `tests/ref/*.npz` and the timing JSONs **stayed**. They are frozen
DATA the oracles compare against, not code, and 620 KB total. Same reasoning
that kept `g3_analysis_data.csv` out of the v0 tag.

Sixty-seven default-tier tests compare against engineA. They now skip as a
MODULE with instructions when the tag is not materialised, rather than failing:

    without the tags  ->  536 passed, 42 skipped
    with the tags     ->  603 passed, 38 skipped

**`decomposition.py`: 1,216 -> 171 lines.** A dependency analysis showed the
live set is exactly `grid_up_nd -> Cell, Decomposition`. The other seven
functions (1,041 lines) are the verbatim 2D transcript, reachable only from
`check_decomposition.py`. They moved to `tests/scripts/_legacy_decomposition.py`
-- test-support code, which is what they always were. The parity oracle passes
against the extracted transcript.

Note a correction: an earlier draft of this report claimed `grid_up_the_system`
was called by `sim_setup`. It is not -- `sim_setup:123` is a COMMENT mentioning
it, and the grep that found it did not distinguish. Only `check_decomposition`
ever called it.

The NPY002 lint exemption moved with the code. `gcmc/decomposition.py` is now
clean of it, and the exemption is scoped to the transcript with a note that
nothing there is maintained.

**`check_cli.py` fixed, not deleted.** It had been xfailed against a launcher
API that changed three ways: the swept axis moved INTO the params dict, the
second positional argument became a core budget rather than a value list, and
`ensemble.sweep` is now rejected outright rather than merely ignored. The
per-run snapshot filename also changed from `params_mu_<value>.json` to
`params_<label>.json`. All four updated; the xfail is gone. Worth keeping
because the launcher sweep spawns subprocesses, which the feature matrix
cannot cover.

**Decision records moved to `docs/decisions/` and are now TRACKED.** They were
gitignored at the repo root and committed anyway -- the project's institutional
memory was one `git clean` from gone.

**`build_history.sh`** builds the four commits and three tags. Run once, before
the first push: retrofitting a root commit onto existing history needs
`git rebase --root` and rewrites every hash.

## Where the lines went

| | Before | After |
|---|---:|---:|
| `gcmc/` | 16,859 | **15,853** |
| `tests/` incl. ref | 31,125 | **19,274** |
| whole tree | 57,570 | **39,369** |

18,201 lines out of the working tree, none of it lost -- every artifact is one
`git worktree add` away.

## R16: channel_cyclic promoted to a first-class sampler

**A correction first.** I claimed in Q-R2 that `channel_cyclic` was unreachable
from a params file. That was wrong, and the error was mine: I checked whether the
schema accepted a `channel_cyclic` KEY and concluded the feature was dark. It was
reached all along by `sampler.channel_length: "cyclic"` -- or by four
undocumented aliases, "full", "box", "full_box", "span".

Worse: the feature-matrix axis used `channel_length: "full"`, which is one of
those aliases. So **115 of 185 accepted channel points were cyclic**, and 11 of
the 12 channel points in the pairwise cover -- all labelled `len_full`, as if the
axis were a rod length. The matrix was exercising the expensive method heavily
under a name that said otherwise.

That makes the surface WORSE than "dark", and R16 more worth doing: a
topologically different, substantially costlier move was selected by one of five
magic strings on a length field.

What changed:

* `gcmc/samplers/channel_cyclic.py`, 210 lines. `channel.py`: 1,421 -> 1,269.
  The two cyclic classes override only `__init__` -- they build overlap masks,
  eviction masks and axial ordering from periodic geometry, then hand off to the
  DP they inherit unchanged. That is why R7 ("one DP") and R16 ("own file") do
  not conflict: one implementation of the DP, cyclic as a mask policy.
* `sampler.type: "channel_cyclic"` added to the schema and `_SAMPLERS`.
* The five legacy `channel_length` spellings now RAISE, naming the replacement.
  Not aliased: a params file that says "full" and means "different move" is the
  surface this removes. `channel_length` is numeric only.
* The matrix axis became `(6.0, 10.0)` -- two genuine rod lengths. The cyclic rod
  now appears on the SAMPLER axis: 115 points, 7 in the cover, under its own name.

## The back-translation shim

Renaming the sampler broke the two frozen-engineA parity oracles, because a
frozen engine cannot parse a config key introduced after it was frozen -- and
that failure looks exactly like a physics regression, which is the worst possible
disguise for a spelling change.

`_harness._back_translate` fixes it generally: surface changes are declared once,
and the frozen engine is handed the spelling it knows. An entry may only ever be
a rename; anything that changes what the engine COMPUTES must regenerate the
oracle instead.

One detail worth keeping. The shim emits `"full"`, not `"cyclic"`: engineA accepts
only ("full", "box", "full_box", "span") -- "cyclic" was added later as the
preferred name and the frozen engine has never heard of it. **Translate into the
vocabulary the frozen engine had, not the most recent spelling the current engine
deprecated.**

All 29 frozen-trajectory oracles pass, so the chain is bit-identical through the
rename.

## S1 complete

    ruff              0 errors, 0 unformatted
    mypy              0 errors
    full suite        603 passed, 40 skipped, 2 xfailed   (tags materialised)
                      536 passed, clean skips             (tags absent)
    tier 0            green
    tier 2            30 passed, 21 failed (the pinned bugs), 21 skipped
    characterisation  green, resolved from tags

Remaining: run `build_history.sh`, push, then S2.

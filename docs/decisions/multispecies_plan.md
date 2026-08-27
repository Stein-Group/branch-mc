# multispecies_plan.md

Append-only decision record for the multi-species Lennard-Jones channel work.
Same protocol as `mof_raspa_plan.md` and `cell_unify_plan.md`: decisions get an
M-number and are never edited in place. A superseded decision keeps its number
and gains a pointer to the one that replaced it.

Revision 2. M5 is superseded by M11. M9 through M15 are new. Everything from
revision 1 not marked superseded stands.

Goal: an n-species truncated Lennard-Jones fluid with independently chosen sigma,
epsilon, thermal wavelength and chemical potential per species, an independently
chosen cross term per pair, sampled by the standard channel sweep, running serial
and parallel, with the short-range auxiliary available, started from an
equilibrated structure on disk.

Nothing here is written for two species. Two is the first case that gets run, not
the case the code knows about. Every structure is `(T,)` or `(T, T)` and every
loop runs over species ids. An `if n_species == 2`, or an `A` and `B` in a
signature, is a defect.

---

## 1. Decisions

### M1 Cross terms are `k_ij`

Sigma keeps Lorentz, `sigma_ij = (sigma_i + sigma_j)/2`. Epsilon becomes

    eps_ij = (1 - k_ij) * sqrt(eps_i * eps_j)

with `k_ij` symmetric, zero diagonal, default zero everywhere. Positive `k_ij`
weakens a cross term, negative strengthens it. At `k = 0` the table is exactly
Lorentz-Berthelot, so every existing force field keeps its current meaning, and
RASPA2 accepts the same statement, which keeps the external validation route
open.

`units.py::load_forcefield` currently raises on anything other than
`mixing: lorentz-berthelot`. That widens to accept `lorentz-berthelot-kij`.
Files without a `k_ij` block keep working untouched.

The diagonal is copied, never computed: at `i == j`, `(1 - 0) * sqrt(e*e)` must
return the stored epsilon bit for bit, or the single-species reduction gate
fails.

### M2 Composition is set by chemical potentials

The ensemble is muVT with an independent `mu_s` per species. Mole fractions are
an output, measured per window, never an input. No semi-grand ensemble, no
composition controller.

Worth remembering when reading results: at fixed `mu` a strongly demixing mixture
can sit in more than one composition, and which one it finds depends on the
starting structure and the chain length. That is physics. It is also why P6 is
scheduled before P7.

### M3 No reference species; the species file declares its own units

Rejected: the MOF path's convention where one guest sets `sigma_ref` and
everything else is measured against it. That pins one species to `sigma* = 1` and
makes "every sigma is free" false.

The species file declares its unit system:

    units: reduced          # sigma, epsilon, rc, lambda, T, mu already reduced
    units: angstrom-kelvin  # sigma in A, epsilon/kB in K; needs a reference

Reduced files carry `sigma_ref = 1`, `eps_ref = 1` by definition and are not
rescaled on load. Every sigma is free; none is pinned. Angstrom-Kelvin is the
existing behaviour, still requires a `reference`, and is what
`ff/forcefield_A.yaml` and `forcefield_B.yaml` keep using. The MOF workstream is
untouched by this entire plan.

The frozen single-species case is one species with `sigma = 1`, `epsilon = 1`,
`lambda = 1` in a reduced file. That is what the bit-identity gates pin against.

### M4 Parallel channel: cell-eroded fixed rod, serial stays the reference

1. The rod lives entirely inside the owning cell. The cuboid centre is drawn from
   the cell eroded by the rod's bounding-sphere radius
   `r_rod = 0.5 * sqrt(L_long^2 + 2*w^2)`, because the rotation is free and the
   erosion must hold for every orientation. Every particle the sweep can touch is
   then owned by the cell.
2. `buffer >= rc` as now, so the rod's interaction reach lies inside
   cell + border.
3. The wall-to-wall cyclic rod stays rejected under parallel, permanently. It
   spans the box by construction and cannot be owned by a cell.
4. Serial is the exact reference. A cell proposal satisfies global balance only
   under full translation invariance, which a cell cannot provide, so parallel
   channel is approximate for the same reason `depth >= 1` under parallel already
   is, and more strongly, because a rod is a larger body than a cavity. The bias
   is measured, not argued, and reported as a curve against
   `2*r_rod / cell_length` in the same shape as the existing depth table
   (+19% at 2.7 cavity widths, +1.3 +/- 0.8% at 6.7, +0.9 +/- 0.6% at 10).

Standing consequence: rule 1 puts a floor on the cell pitch,

    pitch >= sqrt(L_long^2 + 2*w^2) + rc

At `L_long = 6`, `w = 2`, `rc = 2.5` that is about 9.4, so a box of 20 supports
two cells per axis and a box of 50 supports five. Parallel channel is a large-box
feature. It will not accelerate the current sweetspot geometry and is not
expected to.

### M5 SUPERSEDED BY M11

Revision 1 fixed `Lambda_s = 1` for every species. Superseded.

### M6 Trial species probabilities are a proposal knob

Each trial in the channel cloud independently draws its species from a fixed
vector `p_s`, as `mof_channel.assign_species` does. `p_s` must not depend on
current occupancy: the trial density per species has to be a property of the
move, not of the configuration, or detailed balance fails. The Rosenbluth
correction carries the species-draw probability, so a non-uniform `p_s` changes
variance and nothing else.

`p_s` is never used to steer composition (see M2) and is never adjusted mid-run.

### M7 `sweep-moves-per-window` is deleted from the submission surface

In the main engine the channel move is the sampler, selected by
`sampler.type = "channel"`, so a moves-per-window flag has nothing to mean. On
the MOF driver it becomes implied by the sampler choice. Any residual internal
cadence is a module-level constant, per house rule, not a CLI argument.

### M8 Identity swap is on iff there is more than one species

The four-way `move / add / remove / swap` split from `mof_metropolis` is ported
into the main engine, guarded so a single-species run takes the existing
three-way 0.5 / 0.25 / 0.25 split, draws no species and no swap randomness, and
consumes the RNG stream in exactly the old order.

For `n > 2` the swap proposal draws the new species uniformly from the other
`n - 1`. That proposal is symmetric, so the ratio stays
`z_s' / z_s * exp(-beta dU)` with no proposal correction, exactly as in the
two-species case. Drawing from `p_s` instead would need a `p_s / p_s'` factor;
we are not doing that, and the uniform draw is asserted in the test suite so
nobody optimises it into a silent bias later.

Without swaps, composition in a demixing mixture relaxes only through insertion
and deletion, which is the slow mode this sampler exists to defeat.

### M9 Everything species-indexed lives in one species file (resolves Q1)

One YAML per model. Per species: `sigma`, `epsilon`, `lambda`, `p_trial`, and the
element `symbol` used for trajectory I/O. Per pair: `k_ij`. Globally: `units`,
`rc`, the auxiliary multiplier, `core_floor.c`.

No species-indexed quantity appears in `params.json`. The only per-species thing
in the run config is the state point (`mu`, fugacity or `z`), because that is an
experiment and not a model parameter, and it is the axis you sweep. The split is:
the species file is the model, `params.json` is the experiment.

Naming: `ff/species_<name>.yaml`. It is a superset of what the MOF force-field
files carry, and `load_forcefield` reads both, so there is one loader and one
validator, not two.

### M10 The fugacity route lands with P2 (resolves Q2)

Three mutually exclusive state-point routes, all per species, exactly one
present:

    ensemble.mu:        {A: ..., B: ...}   ->  z_s = exp(beta * mu_s) / lambda_s^d
    ensemble.fugacity:  {A: ..., B: ...}   ->  z_s = beta * f_s
    ensemble.z:         {A: ..., B: ...}   ->  z_s given directly

`d` is the dimension. The fugacity relation is the ideal-gas one, which is what
the MOF path already does under `z* = f sigma_ref^3 / (kB T)`; there is no
fugacity coefficient in reduced mode. For real-gas non-ideality, supply `mu`.

Gotcha worth writing down: the fugacity and `z` routes define the activity
directly, so `lambda_s` is inert under them. Only the `mu` route consults it.
Supplying `lambda` alongside `fugacity` or `z` is not an error, since the file is
the model and may be used with either route, but the run log records which route
was taken and whether lambda was consulted, so a later reader cannot
misinterpret a recorded `mu`.

### M11 Per-species thermal wavelength, settable (supersedes M5)

`lambda_s` is a per-species positive number in the species file, default 1.0. It
enters only through `z_s = exp(beta mu_s) / lambda_s^d` on the `mu` route.

All lambdas at 1 recovers the existing convention exactly and is what the
single-species bit-identity gates use. Setting them differently shifts each
`mu_s` by `d * ln(lambda_s)`, a constant offset per species, harmless as long as
it is recorded. It is part of the physics identity and goes into the run
fingerprint: two runs whose lambdas differ are different experiments and must
never share a directory or resume one another.

### M12 The short-range auxiliary is per-pair and strictly shorter (resolves Q3)

The auxiliary is a single dimensionless multiplier `a` in the species file, not a
per-species radius:

    aux_cutoff_ij = a * sigma_ij,    with  max_ij (a * sigma_ij) < rc

One knob, and it is automatically shorter for the smaller pairs, which is the
behaviour you want: the proposal radius should track the pair it proposes into. A
per-species radius list would let someone state a set that is longer for one pair
than the physics cutoff.

The inequality is strict. Equality is reachable only through `aux: none`
(auxiliary off) or `aux: {control: true}`, the exact no-op arm the tuning sweep
needs as its own reference. Anything else landing on or above `rc` is a hard
error naming the binding pair, because an auxiliary that is not shorter than the
physics cutoff costs work, buys nothing, and does it silently.

Keep the existing warning below `1.4 sigma_ij`: the LJ first shell runs to about
`1.5 sigma`, so a proposal radius inside it cannot see the structure it proposes
into, `sd(Delta_corr)` explodes and acceptance collapses while the run completes
at a plausible density. Watch `aux_corr_sd` in the log.

This does not resolve the open question in `guide_cutoff_plan.md` about the exact
insertion point of the acceptance correction. It fixes only the multi-species
generalisation of the cutoff itself, so the two workstreams do not collide. The
periodic guard in P5 is written so `aux_cutoff` remains the binding margin
whenever the auxiliary is on.

### M13 n species, not two

Stated as a decision so it is testable rather than aspirational. Every array is
type-indexed and sized from the species file. The test suite runs a
three-species case for every physics gate, because three is the smallest number
that catches "the second species is special" bugs: an asymmetric `k_ij` matrix, a
swap with more than one destination, and a cross term that is weak for one pair
and strong for another.

### M14 Test budget

Every test in the default suite runs on a tiny fixed-seed system and finishes
fast: boxes of side 6 to 10, `N` of order 20 to 60, chains of at most a few
thousand cycles, one process, a few tens of MB at most. The full suite stays near
the current wall time, which is the reason people actually run it.

Statistical gates take their tolerance from the analytic variance of the quantity
under test, never from a number tuned until it passed. Where a gate genuinely
needs more samples than the budget allows, it is marked `slow`, excluded from the
default run, and listed in section 5 so it is not forgotten.

### M15 Checks fail loudly at the earliest layer that can see the problem

Validation belongs in the loader when it is a property of the file, in
`from_params` when it is a property of the submission, and as a hot-path
assertion only when it is a property that can drift at runtime. A check that
cannot name the offending species, pair or key is not finished. Section 4 is the
required set.

---

## 2. Invariants this work must not break

Pre-existing and non-negotiable. Any phase that appears to need one broken is
wrong and needs a new M-decision, not a widened pin.

- `ljbranch_weight_sum` in `samplers/branch.py` keeps its AST hash. Nothing here
  touches it.
- `orig/` stays byte-identical.
- `_advance_metropolis` stays bit-identical for cubic single-species runs.
- `table_pair_energy_sq` at one type stays bit-identical to `lj_pair_energy_sq`.
- Schema-2 checkpoints still load and report `species = None`.
- The overlap convention stays centralised in `geometry.py`. One convention, one
  place. The pair floor changes what number is handed to it, never where the
  comparison lives.
- No inline pair-energy formula outside the force-field layer; the second AST
  guard already forbids it.
- No frozen pin is widened to make a failing test pass.

---

## 3. Phases

Gated build order. A phase does not start until the previous gate is measured.

### P1. Species file: cross terms, units, lambda, p_trial, auxiliary

Files: `scripts/mof/units.py`, new `ff/species_*.yaml`.

- Widen the `mixing` check for `lorentz-berthelot-kij`; parse `k_ij` as a
  symmetric map over unordered pairs, defaulting to zero.
- Add `units: reduced | angstrom-kelvin` (M3). Reduced files skip reduction and
  need no `reference`.
- Per-species `lambda` (default 1.0), `p_trial` (default uniform), `symbol`.
- Global auxiliary multiplier `a` (M12).
- `ForceField.pair_epsilon()` applies M1 with an exact diagonal copy.
- `reduce_tables` becomes a passthrough for reduced files.
- Ship two files: a three-species reduced one for the test suite, a two-species
  one for the first production runs.

**Gate P1**, unit only, no chain, all sub-second:
- `k = 0` reproduces the current Lorentz-Berthelot table element for element.
- Diagonal is bit-identical to the input epsilons at any `k`.
- A one-species reduced file with `sigma = eps = lambda = 1` yields a `PairTable`
  whose `table_pair_energy_sq` is bit-identical to `lj_pair_energy_sq`.
- Hand-computed `eps_ij` for one asymmetric pair matches the built table.
- A three-species file with two different `k_ij` values builds a symmetric table
  with the right diagonal and three distinct off-diagonal values.
- Every malformed file in 4.1 raises with the offending name in the message.
- `test_mof_units.py` and `test_pair_table.py` pass unmodified.

### P2. Configuration surface and state points

Files: `gcmc/sim_setup.py`, `gcmc/launcher.py`, `examples/schema.json`, the
`from_params` docstring (pinned by `test_schema_matches_code.py`).

- New `potential.species_file` key. Present means typed multi-species mode;
  absent means the existing single-species scalar mode. One switch selects the
  whole typed path.
- The three exclusive routes of M10, each a map with exactly one entry per
  species. A missing or unknown species is an error naming it.
- A scalar `mu` stays legal and means single species, unchanged.
- `run.radius` and `run.lower_cutoff` are forbidden in typed mode. Sigma comes
  from the species file and the floor is `c * sigma_ij`. Two competing
  definitions of the core floor is the bug class that cost a session before; the
  config must make it impossible to state one.
- `volume_of_molecule` becomes per-species; packing fraction becomes
  `sum_s n_s v_s / V`; the log gains `N_s` and `x_s` columns per species.
- Launcher: the state-point maps become shaped axes in the sense `box` and
  `cell_length` already are. One map is one value; a list of maps is a sweep.
  Reuse `_shaped_grid`'s disambiguation rather than inventing a second
  convention. `seed` is already an axis, so sixteen chains is `seed: [1..16]`.
- Run fingerprint gains the species-file hash, the route taken, and the lambdas.

**Gate P2**, config only, no physics, all sub-second:
- Every rejected combination in 4.2 raises with the offending key named.
- A single-species params file produces a `RunConfig` field for field identical
  to today's.
- Two routes at once, a route with a missing species, and a route with an unknown
  species each raise.
- Fingerprint differs when lambda differs, when `k_ij` differs, and when the
  route differs; identical for two runs differing only in `windows` or
  `checkpoint_every`.
- `test_schema_matches_code.py` passes against the updated docstring and schema.

### P3. Species through the hot path

Files: `gcmc/neighbors.py`, `gcmc/geometry.py`, `gcmc/MC_engine.py`.
`gcmc/state.py` already has the column and `gcmc/io.py` is already schema 3.

The long pole, and the only phase touching the hot loop.

- `neighbors.CellList` gains a slot-aligned species column, maintained through
  `insert`, `remove`, `move`, `reactivate`, `bulk_insert`, `_grow`, `rebuild`,
  and through the tombstone protocol on every early-exit path. The existing
  `matrix` flag stays what it is and is not reused for species.
- The list cutoff becomes `max(rc, max_ij core_floor_ij)`. Cell width keys to it,
  so the 27-cell stencil argument still holds.
- Overlap tests take `core_floor[si, sj]`. The convention `d <= floor - 1e-12`
  does not change; only the number handed to it does.
- Port the four-way move split (M8) with the single-species guard.
- Do not port `mof_metropolis`'s KD-tree neighbour path. The point of this phase
  is that the typed engine runs on the same single cell list as the frozen one.

**Gate P3**, Gate 1 for the project:
- Single-species cubic runs are bit-identical to the frozen characterisation
  trajectories. Bit-identical, not statistically equal.
- Brute force: for a random 200-particle three-species configuration in a box of
  8, cell-list total energy matches an `O(N^2)` minimum-image reference to the
  last bit and the overlap verdict matches exactly. Sub-second, and the single
  most valuable test in this plan.
- Slot alignment: after a scripted sequence of 500 inserts, removes, moves,
  tombstones and reactivations with a fixed seed, every live slot's species
  matches an independently maintained reference dict. Pure bookkeeping, no
  physics, very fast, catches the failure that would otherwise surface as a wrong
  isotherm three phases later.
- Degenerate mixture: three species identical in sigma, epsilon, lambda and mu
  with all `k_ij = 0` reproduce the single-species `N` distribution within block
  error over a short fixed-seed run.
- Ideal-gas limit: at low `mu`, `<N_s> = z_s V` per species. `N_s` is Poisson, so
  the tolerance is `4 * sqrt(z_s V / n_indep)`, derived rather than tuned.
- Swap stationary composition: with all species identical except `mu`, the
  composition matches `z_s / sum_t z_t` within binomial error.

### P4. Extract the channel move (refactor, no behaviour change)

Files: `gcmc/MC_engine.py`, `gcmc/parallel.py`.

`advance_cell` re-implements the move loop inline. Copying the channel branch
into it would leave two divergent copies of the hardest move in the codebase.
Factor the channel proposal, acceptance and commit into one callable that both
the serial engine and `advance_cell` invoke, taking the geometry (box torus or
cell torus) as a parameter.

Scheduled ahead of P5 deliberately. Doing it after means doing it twice.

**Gate P4**: pure refactor, so the bar is bit-identity. Every existing serial
channel characterisation trajectory reproduces exactly, and the parallel branch,
rosenbluth and metropolis paths reproduce exactly. One moved bit means the
refactor is wrong.

### P5. Multi-species channel sweep, with the auxiliary

Files: `gcmc/MC_engine.py`, `gcmc/samplers/channel.py` (mostly read-only),
`gcmc/mof_channel.py` (source of the orchestration pattern).

- Wire `ChannelSweepLJMS` / `ChannelSweepMS` in place of the single-type sweeps
  when a species file is present. The DP needs no change: it is type-blind once
  the masks and per-bit log-factors are built, and it already takes a `(T, T)`
  floor matrix and a `PairTable`.
- Trial cloud draws a species per trial from fixed `p_s` (M6, M9). One-body
  weights fold `z_s`. The Rosenbluth correction carries the species-draw
  probability.
- Auxiliary per M12: the sweep's LJ masks are built at `aux_cutoff_ij` when it is
  on, at `rc` when off, and the mismatch is divided out in acceptance exactly as
  the single-species dual-cutoff path does.
- Periodic guard becomes per-species: the margin is the largest distance the
  sweep will read, which is `max_ij core_floor_ij` for the count sweep and
  `max_ij aux_cutoff_ij` (auxiliary on) or `rc` (off) for the LJ sweep.
- Delete the moves-per-window flag (M7).

**Gate P5**, Gate 2 territory:
- Degenerate three-species mixture reproduces the single-species channel density
  within block error.
- `p_s` independence: three trial vectors at fixed `mu` give the same composition
  within block error. If they do not, the Rosenbluth species correction is wrong.
  Sharpest available test of that term, and cheap, so it runs by default.
- Auxiliary control arm reproduces the auxiliary-off chain bit-identically. Not
  statistically. The control arm is meant to be an exact no-op, and if it is not,
  the correction is inserted in the wrong place (this is the gate that bears on
  Q6).
- Auxiliary on at a legal `a` gives the same density as auxiliary off, within
  block error, on a tiny system. Speed is not tested here; correctness is.
- An `a` landing on or above `rc` raises and names the binding pair.
- Channel and metropolis agree on `<N_s>` where metropolis still mixes.
  Consistency check only, not truth. Using another sampler as the reference is
  what injected frozen-chain excursions last time.

### P6. Structure in and out

Files: `gcmc/io.py`, `gcmc/runner.py`.

Both directions currently lose species: `write_frame` labels every particle `"H"`
and `load_config_file` calls `get_positions()` and discards the symbols.

- Write real symbols from the species file, so a trajectory frame is a valid warm
  start for the same model.
- Read symbols back into `state.species`. A symbol not in the species file is a
  loud error naming it and listing the known species, never a silent type 0.
- A file with no species column loads only in single-species mode. In typed mode
  it is an error, because guessing composition from a bare coordinate table is
  exactly the silent wrong answer this codebase refuses.
- CIF in: through ASE and the same symbol mapping, with an optional name map in
  the config for CIF labels that do not match species names.
- Wrapping stays off; the file is authoritative in-box coordinates.

**Gate P6**: round-trip a three-species configuration through extxyz and through
CIF, recovering positions bit-identically and species exactly. Schema-2
checkpoints still load. A mislabelled symbol raises and names it. A bare
coordinate table raises in typed mode. All file operations on 40-particle
configurations, so effectively free.

### P7. Parallel channel

Files: `gcmc/sim_setup.py`, `gcmc/cell_plan.py`, `gcmc/decomposition.py`,
`gcmc/parallel.py`.

- Carry the species column through `grid_up_nd` and the reassembly. Particles
  reorder between windows, so species travels with coordinates and never by
  index. Most likely place for a silent scrambling bug, and the gate below exists
  to catch it.
- `cell_plan` gains the rod floor from M4: an infeasible request errors listing
  the nearest usable cell counts, matching existing planner behaviour, never a
  silent reduction.
- Draw the rod centre from the eroded cell.
- Remove the `channel sampler is serial only` rejection in `sim_setup`. Keep
  rejecting `channel_cyclic` under parallel, permanently.
- Watch the per-cell pickled payload: species arrays and the pair table add to
  the roughly 125 MB per worker already flagged in the weak-scaling work. Pin
  `OMP_NUM_THREADS` before reading any timing.

**Gate P7**:
- Species conservation: totals per species invariant across a decomposition round
  trip, and a tagged-particle test confirming every particle keeps its own
  species through split, zero-step advance, and reassembly. Tiny, deterministic,
  in the default suite.
- One cell reproduces the serial chain bit-identically.
- Rod floor: a request that cannot fit raises and lists the nearest usable
  counts; one that just fits is accepted. Planner arithmetic only, no chain.
- Bias curve against serial at four values of `2*r_rod / cell_length`, plus one
  three-species arm (resolves Q4), reported with block error. Marked `slow`: a
  cluster measurement, not a unit test. Publish the curve, do not average it
  away.

### P8. Validation

- There is no equation of state for an asymmetric multi-component Lennard-Jones
  mixture, so the external oracle is RASPA2 with explicit `k_ij`, same box, same
  truncation, same core floor, same state point. Same discipline as the MOF
  Q-series.
- Internal consistency: channel versus metropolis where metropolis still mixes,
  as a check and not as a reference.
- The scientific claim: at strong `k_ij`, show the mixture separates and that the
  channel sampler reaches the separated state from a mixed warm start in fewer
  cycles than metropolis. Design that figure before the runs, not afterwards from
  whatever happens to exist.
- Frozen-chain gating carries over: `n_indep < 20` or fewer than 200 accepted
  insertions stays out of axis scaling and appears as hollow markers.

---

## 4. Required checks

### 4.1 Species file (loader)

Sigma and epsilon positive. Lambda positive. `p_trial` all positive and summing
to 1 within tolerance, or absent (then uniform). `k_ij` symmetric, zero diagonal,
every key naming two known species, no pair stated twice in both orders with
different values. Symbols unique and valid element names. Species names unique
and disjoint from framework names. `core_floor.c` in `(0, 1]` and not null. `rc`
positive. Auxiliary multiplier satisfying M12 against the largest `sigma_ij`.
`units` present and one of the two accepted values. `reference` present iff units
are Angstrom-Kelvin.

### 4.2 Submission (`from_params`)

Exactly one state-point route. Exactly one entry per species in it. No unknown
species. `run.radius` and `run.lower_cutoff` absent in typed mode. Channel with
`depth >= 1` or `full_repacking`. Under parallel: not cyclic, rod fits the eroded
cell, `buffer >= rc`. Species file present iff typed mode. Unknown keys are
already a hard error per block and stay that way.

### 4.3 Runtime assertions

Species column length equals position count at every window boundary. Cell-list
species column stays slot-aligned, asserted in the same consistency helper that
already checks border frozenness, so it costs one call per window and not one per
move. Species totals conserved across decomposition. Every pair floor consulted
finite and positive. No negative or non-finite one-body weight reaches the sweep.

---

## 5. Tests, and what they cost

Default suite, all fast, all fixed-seed, tiny systems:

| Test | Kind | System |
|---|---|---|
| Table build, `k = 0` reduces to LB | unit | none |
| Diagonal bit-identity at any `k` | unit | none |
| One-species table reduces to locked LJ | unit | none |
| Three-species asymmetric `k_ij` table shape | unit | none |
| Every malformed species file raises | unit | none |
| Every rejected submission raises | unit | none |
| Fingerprint sensitivity (lambda, `k_ij`, route) | unit | none |
| Cell-list energy vs `O(N^2)` reference | physics | 200 particles, box 8 |
| Slot-species alignment under 500 mutations | unit | 60 particles |
| Single-species bit-identity | regression | frozen trajectories |
| Degenerate mixture equals single species | physics | box 8, few thousand cycles |
| Ideal-gas limit `<N_s> = z_s V` | physics | box 10, low `mu` |
| Swap stationary composition vs analytic | physics | box 8 |
| Swap destination draw is uniform | unit | none |
| `p_s` independence of composition | physics | box 8, three `p_s` values |
| Auxiliary control arm is an exact no-op | regression | box 8 |
| Auxiliary on matches auxiliary off | physics | box 8 |
| extxyz and CIF species round trip | unit | 40 particles |
| Decomposition species conservation | unit | 100 particles |
| One-cell parallel equals serial | regression | box 12 |
| Rod-floor planner arithmetic | unit | none |

Marked `slow`, excluded from the default run, listed so they are not lost: the
parallel bias curve (P7, four ratios plus the three-species arm), the RASPA2
comparison (P8), the demixing-speedup figure (P8). Cluster measurements with
their own submission scripts, not pytest.

Statistical tolerances come from the analytic variance of the quantity: Poisson
for `N_s` in the ideal-gas limit, binomial for compositions, block error for
densities. A tolerance that cannot be derived means the test is measuring the
wrong thing.

---

## 6. Order and session map

    P1 + P2      one session; independent, cheap
    P4           next; refactor only, must be bit-identical
    P3           the long pole; Gate 1 for the project
    P5           depends on P3 and P4
    P6           any time after P3; good session filler
    P7           depends on P4, P5, P6
    P8           last, and the figure design starts before the runs do

P6 before P7 matters more than it looks. A parallel channel run at a useful box
size needs an equilibrated warm start, and generating one from an empty box at
high density is the problem this sampler exists to solve. Do not schedule P7
against a cold start.

---

## 7. Open questions

- **Q5.** Does the auxiliary multiplier belong in the species file (model) or the
  sampler block (experiment)? M9 puts it in the species file because it is
  sigma-indexed, but it is arguably a sampling knob you would sweep. If the
  tuning sweep in P5 wants it as an axis, revisit and record as M16.
- **Q6.** `guide_cutoff_plan.md` still has the unresolved question of the exact
  insertion point for the acceptance correction. M12 does not answer it. The
  auxiliary control-arm gate in P5 is what catches a wrong answer, so P5 is not
  signed off until that gate is exact rather than merely close.

---

## 8. Revision 3

M17 is new. M16 stays reserved for Q5 as revision 2 promised, so the gap is
deliberate. P1 is measured and signed off; nothing from revisions 1 or 2 is
superseded.

### M17 `schema` tells the two file kinds apart, not a heuristic

Revision 2 left a contradiction between M3 and Gate P1. Section 4.1 requires
`reference` present iff units are Angstrom-Kelvin, but `ff/forcefield_A.yaml`
carries no `reference` key at all -- its reference is an argument to
`reduce_tables` -- and Gate P1 requires `test_mof_units.py` to pass unmodified.
Enforcing 4.1 as written would have invalidated both shipped MOF force fields.

The discriminator is the `schema` key, which both files already carry:

    schema: 1   MOF force field. `guests:` + `framework:`. `units` absent and
                defaulting to angstrom-kelvin. `reference` supplied by the
                caller, defaulting to Xe. Exactly today's behaviour, unchanged.
    schema: 2   species file. `species:`. `units` mandatory. `reference`
                required iff angstrom-kelvin, forbidden iff reduced.
                `core_floor.c` mandatory.

So the iff-rule of 4.1 binds at schema 2, which is where it was aimed, and the
half that always bites -- a reduced file must not name a reference, because
there is nothing for one to mean -- binds at both. `reduce_tables` now defaults
its `reference` to the file's own declaration and falls back to Xe, so every
existing caller keeps its current behaviour without being touched.

Two consequences worth writing down. `_blocks_status` now reports only blocks
the file actually carries, otherwise a species file with no framework would fail
the provisional gate on a block it does not have. And a schema-2 file states
`status: confirmed` on its species block meaning only that the numbers are the
model's own definition; there is no primary source to confirm a reduced model
against, and the comment in each shipped file says so.

Also settled while implementing, because the plan did not say and a silent
choice here would have been a second convention:

- `k_ij` is a nested map, partner under species, either order accepted. Not an
  `"A-B"` string key, so a species name may contain any character without
  inventing a delimiter escape. Restating a pair in both orders is legal iff the
  values agree; disagreeing is the error 4.1 asks for.
- `k >= 1` is refused, naming the pair. It would drive `eps_ij` to zero or
  negative, which is not a weakened cross term but a different potential.
- `p_trial` is per type over the whole type list, framework entries exactly
  `0.0`, guests summing to 1. One array, one indexing convention, no separate
  guest-only vector to keep aligned. Declared for every species or for none;
  a partial declaration raises and names the missing ones.
- `lambda` and `p_trial` are refused inside a `framework:` entry. A rigid atom
  has no thermal wavelength and the trial cloud never proposes one.
- `symbol` defaults to the type name and is validated against the real element
  list, so a species named `A` must declare one. Symbols must be unique across
  all types, or a trajectory frame could not be read back into species (P6).
- The auxiliary binding pair is taken over every pair including framework ones,
  since a guest trial reads framework neighbours.
- In reduced mode `ReducedTables.T_kelvin` carries `T*` itself and `z_star`
  raises: there is no reference scale to convert a pressure through. Use the `z`
  or `fugacity` route of M10.

### Gate P1: measured, passed

`tests/unit/test_species_file.py`, 43 tests, 1.8 s, no chain.

- `k = 0` reproduces the Lorentz-Berthelot table byte for byte, both with the
  `k_ij` block absent and with an all-zero block under the kij mixing rule.
- The diagonal is bit-identical to the input epsilons at k = 0, 0.25, -0.9 and
  0.999. It is copied, not computed, so this holds by construction rather than
  by the `sqrt(fl(e*e)) == e` rounding theorem.
- `ff/species_one_reduced.yaml` gives a `PairTable` whose `table_pair_energy_sq`
  is bit-identical to `lj_pair_energy_sq` over six squared distances.
- Hand-computed cross terms match: `eps_AB = 0.75 sqrt(0.8) = 0.670820393249937`,
  `eps_BC = 1.15 sqrt(1.04) = 1.172774488126341`, `eps_AC = sqrt(1.3)` from the
  absent-and-therefore-zero path, `sigma_AB = 1.075`.
- The three-species table is symmetric with the right diagonal and three
  distinct off-diagonals from two stated `k` values.
- Every malformed file in 4.1 raises with the offending species, pair or key in
  the message: 21 parametrised mutations plus 8 named cases.
- `test_mof_units.py` (12 tests) and `test_pair_table.py` pass unmodified.

Full suite after P1: 474 passed, up from 431, same 6 pre-existing failures and
same 1 collection error (section 9). ruff clean, ruff format clean, mypy clean
on the changed module.

Files: `scripts/mof/units.py` rewritten; `ff/species_one_reduced.yaml`,
`ff/species_three_reduced.yaml`, `ff/species_binary_reduced.yaml`,
`tests/unit/test_species_file.py` new.

---

## 9. Snapshot lag: resolved

Revision 3 recorded six failures and one collection error as a tests/examples lag
behind gcmc/. All seven are now fixed in the tree and the record is closed:
test_channel_2d imports ChannelSweepCyclic, test_item29 expects the 2D channel to
build, per-axis box and launcher expectations were updated, and the schema
docstring matches the code. The loader also moved from scripts/mof/units.py to
gcmc/species_file.py with a re-export shim (M18), because gcmc.sim_setup validates
a species file at from_params time and the engine package must not import
scripts/, which is not in packages.find.

Four of the six remaining failures were in the new Gate P2 test file rather than
in the code, and are worth recording because the same mistake is easy to repeat:

- Three route tests called `_typed(ensemble={"z": ...})`, which MERGES over a base
  that already carries `mu`. Every one of them was therefore exercising the
  two-routes-at-once rejection, not the route it named. `species.py` was correct
  to raise. `_typed` now treats a None VALUE as a deletion, and the route tests
  pass `"mu": None`.
- `run={"windows": ...}` names a key the schema does not have; the chain-length
  knobs are `sampling_window`, `burn_in` and `target_cycles`. Gate P2 in section 3
  says "windows", which is the plan being loose, not the schema being wrong.

One test assertion was replaced rather than satisfied, and that needs a human
look. `test_the_untyped_fingerprint_does_not_gain_a_null_key` asserted
`run_id == "1a24fe1c17aa"`. That digest is not reproducible from any variant of
the identity dict: not the current one, not with `species_identity: None`
restored, not with `seed` removed, and tests/ref/engineA gives `7c1f010d0f1b`
because `seed` became identity-bearing after engineA was frozen. It is now an
explicit assertion on the 25-key identity SET -- which is what actually catches a
new key leaking in and silently re-addressing every existing run directory --
plus the measured digest `fbdc3125c74a`, labelled a regression pin and not
evidence about historical ids. If the original constant came from a real
measurement, that provenance is lost and should be restored.

---

## 10. P4: measured, passed

Scope, now settled: the five things that are channel-specific, which are the five
things `advance_cell` lacks. The rest of the move loop is not duplicated between
`advance` and `advance_cell`, so there was nothing there to unify.

`gcmc/channel_move.py`, 303 lines:

  1. `draw_rotation` -- the orientation, drawn before the centre, and consuming
     nothing from the stream when the channel sampler is off
  2. `draw_center_offset` -- the anchor offset in the body's own frame, with the
     cyclic rod's degenerate axial direction handled in one place
  3. `ChannelBody.build` / `.members` -- the chord, the derived trial count, and
     membership; one object so that insertion and removal reading the same body
     with the same rotation is structural rather than something a reader verifies
     by eye across four hundred lines
  4. `ChannelBody.trial_cloud`
  5. `ChannelBody.make_sweep` -- one sweep serving both directions, so the
     feasibility test is symmetric by construction

`side` (proposal torus) and `box` (physics) are separate parameters, which is what
makes the parallel path of M4 expressible; conflating them is the bug M4 exists to
prevent.

THE ORACLE. `tests/ref/traj_*.npz` cannot gate this phase: stage B made particle
state index-based and `test_stageA_traj_identity` already records that only
`metropolis` still reproduces those bytes, so checking a refactor against them
proves nothing. `tests/scripts/gen_p4_snapshot.py` therefore freezes the engine
against ITSELF -- fourteen modes, four windows, final positions AND the per-window
counter stream, because two chains can end in the same place having taken
different routes. Generated on the pre-refactor tree, which is the only thing that
makes it meaningful. Five seconds for all fourteen, which is what made a
bit-identity refactor of an 1186-line function tractable at all.

Result: 14/14 bit-identical, verified after each of the five seams separately.
Full suite 538 passed, 1 skipped, 0 failed. `check_channel_full` (6 s),
`check_channel_frontier` (30 s) and `check_channel_lj` (128 s) all green, and all
three were timed on the pre-refactor tree first. Repo-wide mypy errors fell from
ten to four; the six that went were in the block the extraction removed. ruff
clean. `advance` is 1118 lines, down from 1186, and imports seven fewer names
from `samplers.channel` -- the import list is the evidence the extraction is
complete rather than partial.

Not done, and deliberately: `advance_cell` does not yet CALL any of this. P4's
gate is bit-identity of the existing paths, and wiring the parallel channel on is
P7, which needs the rod floor and the eroded-cell draw. The point of doing the
extraction now is that P5 and P7 both edit one copy.

### Found while working, not fixed

`scripts/visualize_channel_moves.py` patches `MC_engine.cuboid_members` to install
its recorder. The engine imports `cuboid_members_cl`, not `cuboid_members`, so the
patch has been setting an attribute nothing reads: `_LAST` never populates and the
script loops to MAX_WINDOWS saving nothing. Pre-existing, unrelated to this work.
P4 gives it a clean patch point in `channel_move.ChannelBody.members`.

---

## 11. P3, part 1: the species column on the cell list

Started. The column and its two bookkeeping gates are in; the overlap floor, the
list cutoff and the four-way move split are not. Recording the split so the next
session does not have to re-derive where the line falls.

### Done

`gcmc/neighbors.py`. `CellList` gains a slot-aligned `species` column, int64 so
it indexes the (T, T) core-floor and pair tables without a cast on the hot path.
All-zero is the single-species case, so a run that never sets it reads exactly
what it read before the column existed -- which is why the P4 snapshot still
holds unchanged.

Maintained through `__init__`, `_grow`, `bulk_insert`, `insert` and
`from_positions`. NOT touched by `remove`, `reactivate`, `move` or `rebuild`, and
that is the correct behaviour rather than an omission:

- `remove` tombstones; the species must survive, because `reactivate` has to
  restore the original type and not a default.
- `move` is a translation; a particle does not change what it is by moving.
- `rebuild` reorders the CSR and frees tombstones, but `_sorted` holds slot ids
  and slots do not move, so a slot-indexed column stays aligned for free.

`insert` writes the species unconditionally rather than leaving the slot at its
previous tenant's value. A reclaimed slot carrying a stale species is the one way
this column can go wrong while every coordinate-level test still passes.

`species_at` mirrors `matrix_at`, clamping the pad index. Pads carry an infinite
distance so whichever species they read can never reach a weight or a verdict;
the accessor is what keeps that true at call sites that have not thought about
pads. `species` is deliberately not folded into the existing `matrix` flag: that
flag means "enters the overlap test, never the energy sum, never inserted,
removed, moved or counted", which is orthogonal to type. A framework atom has a
species too.

A `bulk_insert` species column of the wrong length raises rather than
broadcasting.

### Gates measured

`tests/unit/test_species_hot_path.py`, 7 tests, 0.95 s.

- Slot alignment under 500 mutations against an independently maintained dict,
  checking the live SET and the species of every live slot at every step.
  Instrumented to confirm the mix does the work: 11 inserts landed on a
  previously tombstoned slot, 30 rebuilds, capacity grew 32 -> 162, 117 live at
  the end. Without those numbers the test could pass while exercising nothing.
- `_grow` carries the column. A column missing from that copy list is zeroed on
  the first reallocation, which reads as "everything is species 0" -- a
  plausible-looking single-species run, not a crash.
- Brute force: 200 particles, three species, box 8, cell-list total energy
  -445.52758333220686 against an O(N^2) minimum-image reference at
  -445.5275833322073, relative difference 1.0e-15. That is summation
  associativity, not physics: both sums run the same pairs through the same
  per-pair function in a different order. The plan asks for "to the last bit",
  which is not achievable between two different summation orders and is not the
  right bar; the bar is the tolerance of the summation.
- Overlap verdicts agree pair for pair with the per-pair floor. A boolean, so no
  tolerance to hide behind. The configuration is drawn without a rejection floor
  precisely so overlapping pairs exist, and the test asserts they do.

The reference energy goes through `table_pair_energy_sq`. An independent
implementation of the LOOP is what makes it a reference; an independent
implementation of the POTENTIAL would be a second thing that can be wrong, and
the second AST guard forbids it.

Full suite 545 passed, 1 skipped, 0 failed. ruff and mypy clean on both files.

### Not done, and what each one needs

- **Overlap tests take `core_floor[si, sj]`.** `gcmc/geometry.py`. The convention
  `d <= floor - 1e-12` does not change; only the number handed to it does. The
  centralisation invariant means this is one edit in one place, but every caller
  currently passes a scalar and has to start passing a pair lookup.
- **List cutoff becomes `max(rc, max_ij core_floor_ij)`.** Worth noting the
  numbers: for `species_three_reduced` the floors run 0.81 to 1.035 against
  rc 2.5, so the max is still rc and the 27-cell stencil argument is untouched at
  this parameter set. It only binds for a mixture whose largest pair floor
  exceeds the interaction cutoff, which no planned run has. Cheap to do, and it
  should be done anyway so the invariant is structural rather than incidental.
- **The four-way move split (M8).** `gcmc/MC_engine.py`. The hard part is not the
  swap move, it is the single-species guard: the existing three-way 0.5/0.25/0.25
  split must draw no species and no swap randomness and consume the RNG stream in
  exactly the old order. The P4 snapshot is the instrument -- fourteen modes, five
  seconds -- and it will catch a reordered draw immediately.

---

## 12. P3, part 2: the per-pair overlap floor, centralised

The invariant said the overlap convention lives in `geometry.py`, one convention
in one place, and that the pair floor changes what number is handed to it and
never where the comparison lives. It had already drifted. Four independent
implementations of `d <= floor - 1e-12` were live at once:

    gcmc/potentials/pair_table.py:138   with its own _OVERLAP_TOL constant
    gcmc/mof_field.py:125               batched, on squared distances
    gcmc/mof_field.py:156               scalar
    gcmc/mof_metropolis.py:116          scalar

They all happened to hold 1e-12, so nothing was measurably wrong today. That is
exactly the state the hard-won-lessons entry describes, and it is why the entry
exists: the copies agree until they do not, and then the disagreement is
measurable but not attributable.

### Done

`gcmc/geometry.py` gains the per-pair form and the reach:

- `pair_overlaps(d, si, sj, core_floor)` -- `d <= core_floor[si, sj] - OVERLAP_TOL`.
  Scalars or arrays for either species index, so the hot-path shape (one trial
  type against a neighbour column) needs no separate function.
- `max_core_floor(core_floor)` -- the largest pair floor. Two callers need this
  number and neither should compute it inline: the query reach, via
  `overlap_radius`, and the cell-list cutoff.
- `overlap_radius` now accepts an array of floors as well as one diameter,
  because a caller working in SQUARED distances needs the threshold rather than
  the verdict and would otherwise write the subtraction itself.

All four sites now route through it. `pair_table._OVERLAP_TOL` is deleted and
`overlaps_table` is a delegating wrapper; the comment claiming the duplicate
avoided importing geometry into a potential was wrong on the facts -- geometry
imports nothing but numpy at module scope, its one scipy use being a local import
-- so that cost was imaginary and the second copy was not.

`SpeciesSpec.max_core_floor` also routes through the helper rather than
recomputing `core_floor.max()`.

### The list cutoff was already done

`SpeciesSpec.list_cutoff` landed with P2 and is already `max(rc, max_core_floor)`.
Nothing to do. Worth keeping the measurement from part 1 next to it: for
`species_three_reduced` the floors run 0.81 to 1.035 against rc 2.5, so the rc
term wins and the 27-cell stencil argument is untouched at this parameter set. The
max only binds for a mixture whose largest pair floor exceeds the interaction
cutoff, which no planned run has.

### Gate

`tests/unit/test_overlap_convention.py`, 9 tests, 0.8 s.

The load-bearing one is a source scan: the tolerance literal appears in
`geometry.py` and nowhere else in `gcmc/`, with docstrings and comments stripped
first so prose about the convention is not mistaken for an implementation of it,
and with the match narrowed to a tolerance SUBTRACTED FROM a floor, sigma,
diameter or cutoff, so an unrelated 1e-12 (an allclose atol, a mol-fraction sum
check) is not reported as guilty. Verified to bite by planting the old
`mof_metropolis` line back and watching it fail with the file and line named.

The rest pin the behaviour the four copies could have disagreed about: the
per-pair form equals the global form exactly at one species; a pair exactly at
its floor is admissible for every pair; the floor is read the right way round,
tested against a deliberately ASYMMETRIC table so a transposed read cannot hide
behind the symmetry the engine's table happens to have; broadcasting over a
neighbour column; and `overlaps_table` is a real delegation, proved by breaking
the central helper and requiring the wrapper to break with it.

One of these caught my own error rather than the code's: the first version of the
right-way-round test picked a distance below BOTH floors, so it passed whichever
way the read went. It now picks one that sits between them.

Full suite 554 passed, 1 skipped, 0 failed. `check_channel_full` and
`check_ljbranch` green. ruff and mypy clean on every file touched.

### Pre-existing lint, not mine

`gcmc/sim_setup.py` has three ruff findings from the P2 typed path (one unsorted
import block, two f-strings without placeholders, lines 216, 967, 968). Left
alone rather than fixed, because that file was being edited concurrently and a
formatting change from me would collide.

### P3's remaining item

The four-way move split (M8), and nothing else. The hard part is not the swap
move; it is the single-species guard. The existing three-way 0.5/0.25/0.25 split
must draw no species, draw no swap randomness, and consume the RNG stream in
exactly the old order, and `_advance_metropolis` must stay bit-identical for
cubic single-species runs. The instrument is `gen_p4_snapshot.py`: fourteen modes,
five seconds, and it fails on a reordered draw immediately rather than as a
statistical discrepancy weeks later.

---

## 13. P3, part 3: M8, the move split

The selection half of M8 is in and shared. The swap HANDLER in the main engine is
not, and cannot be until the typed energy path exists; see the end of this
section for why that is the right boundary rather than an unfinished job.

### Done

`gcmc/moves.py`, new. `draw_move(rng, n_species)` and
`draw_swap_target(rng, current, species_ids)`, plus `swap_log_acceptance`. Three
loops now call one definition: `MC_engine.advance`,
`MC_engine._advance_metropolis` and `mof_metropolis`. The last of those had the
only implementation; it now imports the shared one and re-exports `_MS_MOVES` /
`_MS_PROBS` under their old private names so its golden pin and tests are
undisturbed.

The single-species guard is structural, not an optimisation. `rng.choice`
consumes different amounts of the stream for different argument shapes, so a
four-way call with `p = [0.4, 0.2, 0.2, 0.0]` would be a different chain even
though it can never return "swap". At `n_species == 1` the frozen three-way call
is made with the frozen list and the frozen probability vector, and the P4
snapshot confirms all fourteen modes are still bit-identical.

### A false claim corrected

`mof_metropolis` carried the comment that the swap budget is "carved out of the
translate budget so insertion and deletion keep equal probability". That is not
what the numbers do. Translation pays 0.1 and add and remove pay 0.05 each:

    0.5, 0.25, 0.25   x (1 - p_swap) = x 0.8  ->  0.4, 0.2, 0.2   plus swap 0.2

The multi-species vector is the single-species one scaled UNIFORMLY, so every
ratio among the non-swap moves is preserved, not just add against remove. That is
both stronger and simpler than the claim it replaces, and it is the property the
derivations actually need: the grand-canonical ratio wants
P(add) == P(remove), and the translate share relative to those two sets how much
configurational relaxation happens between exchanges, so a split that shifted it
would change the sampler's character as well as its algebra.

Found by writing the assertion the old comment implied and watching it fail
(0.09999999999999998 against 0.2). Recorded because the comment had been true-
sounding and unchallenged, and the corrected version is now the pinned one.

### A latent bug fixed on the way

`draw_swap_target` first took a species COUNT and drew from `range(n)`.
`mof_metropolis` derives its candidates from `sorted(z_star)`, the state-point
dict keys, so a run over a SUBSET of a force field's guests has non-contiguous
ids -- `range(n)` would then swap into a species the run does not contain and
miss one it does. The signature now takes the actual ids. Not reachable with FF-A
as currently driven (its guests are ids 0 and 1), so nothing measured was wrong,
but it is the exact failure class M13 exists to prevent and it was one keyword
away from being permanent.

### Gate

`tests/unit/test_move_split.py`, 11 tests, 8.8 s.

- The single-species split consumes the stream exactly as before. Checked by
  comparing 300 draws AND then requiring the two generators to still be in
  lockstep on `random(20)`: comparing the labels alone would not catch a call
  that returned the same sequence while eating a different number of values.
- A single-species run can never draw a swap (2000 draws).
- The multi-species split is the single one scaled by (1 - p_swap), element for
  element.
- Frequencies match, with the tolerance from the binomial standard error
  `sqrt(n p (1-p))` at 4 sd, derived rather than tuned.
- The swap destination is uniform over the others, over 120,000 draws from each
  of four starting species, and never returns the current one. M8 asks for this
  explicitly: uniform is what makes the proposal symmetric, so the acceptance is
  the activity ratio with no `p_s / p_s'` correction, and a well-meaning change
  to draw from `p_trial` would introduce a bias with nothing else in the code to
  catch it.
- Two species forces the destination and draws NOTHING, verified by stream
  lockstep. One candidate is no decision, and a value consumed there would shift
  every later draw.
- Non-contiguous ids are respected.
- The acceptance is the activity ratio alone, is antisymmetric at dU = 0, and
  reduces to the Boltzmann factor at equal activities.

Full suite 565 passed, 1 skipped, 0 failed. Repo-wide mypy errors still 4, all
pre-existing.

### Not done, and why this is the boundary

The swap HANDLER in `MC_engine`. `mof_metropolis` has one, and it needs three
things the main engine's loops do not yet have: a per-particle species column in
the chain state that the loop mutates, a typed energy evaluation at a fixed
position for two different species, and `log_z` per species. The first arrives
with the state/engine wiring, the second with the typed potential in the hot
loop, the third exists already on `SpeciesSpec`.

Porting the handler before those exist would mean stubbing all three, and a
stubbed swap that silently no-ops is worse than no swap: the split would draw it,
the counter would tick, and the composition would relax through insertion and
deletion only -- which is the slow mode M8 exists to defeat, presenting as a
plausible run rather than as a failure. So the split is wired and the handler
waits, and `draw_swap_target` raises rather than returning `current` precisely so
a premature call cannot become a quiet no-op.

Remaining for Gate P3, in order: the species column on `ChainState` through the
engine loops, the typed energy path, then the swap handler, then the four
statistical gates (degenerate mixture, ideal-gas limit, swap stationary
composition, single-species bit-identity -- the last already covered by the P4
snapshot).

---

## 14. P3, part 4: the column through `advance`, and a finding that changes P5

### Done

`advance` builds its cell list with `state.species_of` and materialises the
column back from `cl.species[alive[:N]]` -- the same slot gather the positions
come from, so the two cannot come apart. `commit_exchange` grows an optional
`chosen_species` argument, written at INSERT time rather than patched afterwards,
because a reclaimed slot arrives carrying its previous tenant's type and a second
pass over `alive` would be correct only until an insert triggered a rebuild
partway through.

A single-species run leaves `state.species` as None rather than writing zeros. io
then emits no species array, so a schema-2 checkpoint and a schema-3
single-species checkpoint stay identical in content and the fingerprint gains no
null key. P4 snapshot: 14/14 still bit-identical.

### M19 The species of an inserted particle is a property of the POOL ATOM

Found by writing the obvious test -- a particle that ends a window where it
started keeps its type -- and having it fail. It is not a test bug.

`pool = trial_set.mask(km).merge(mol_in_cav)`. The pool is the surviving trial
cloud PLUS the tombstoned cavity members, deliberately, because a member's own
position is a guaranteed-valid placement and is where a large part of the DP's
acceptance comes from. When the DP selects such an atom, the engine RE-INSERTS it
rather than reactivating it: as far as the cell list is concerned that is a new
particle, and it takes whatever species the insertion supplies.

Measured in one `hs_branch` window: 214 particles ended at an unchanged position,
and 12 of them were re-inserted members. So roughly 6% of apparently-stationary
particles per window are new particles wearing an old position.

The consequence for P5 is concrete, and it is not what the plan currently says.
M6 says each trial draws its species from `p_s`. That is right for a TRIAL CLOUD
atom and wrong for a tombstoned member, whose species is already fixed and must
be carried forward. Drawing fresh for every insertion would leak composition
toward whichever species the draw favours at a rate set by the member-selection
frequency, with total density unchanged -- a mixture drifting for a reason that
looks nothing like a bug.

So `chosen_species` must be assembled PER POOL ATOM: `p_s`-drawn for a trial-cloud
atom, carried over for a member. `TrialSet.merge` has to keep a species column
through the merge and the permutation, alongside the coordinates, for the same
reason the cell list does -- the pool is permuted (`pool[rng.permutation(len(pool))]`)
before the sweep, so anything held by index is scrambled.

This also bears on the Rosenbluth species correction of M6: the correction carries
the species-draw probability, and a member atom had no draw. Whether the member's
term is 1 or `p_s` is a detailed-balance question, not a bookkeeping one, and it
should be settled before P5 is written rather than discovered in a composition
residual. Recorded as the first thing P5 must resolve.

### Gate

`tests/engine/test_species_through_advance.py`, 3 passed and 1 xfailed.

- A single-species run leaves the column None, and the accessor still gives the
  right length on demand.
- The column comes back aligned with the positions, right length and dtype.
- Exchange insertions are still type 0. The documented gap, asserted so it cannot
  change unnoticed; when P5 lands this test SHOULD fail, and its message says so.
- The survivor-identity test is `xfail(strict=False)` with M19 as the reason, so
  it XPASSes the moment P5 fixes it. Following the repo's existing convention for
  a real, unresolved finding: visible and runnable, not hidden.

Full suite 568 passed, 1 skipped, 1 xfailed, 0 failed. mypy still 4 pre-existing.

### Open design question, not decided unilaterally

`grid_up_nd` returns `(main, border, origin)` per cell, built from boolean masks
over the input positions, and is pinned against `orig/` by
`tests/scripts/check_decomposition.py`. Carrying species means applying the same
masks, which is trivial, but it needs a return-shape decision: a conditional
5-tuple is ugly, a second function duplicates the origin draw and its RNG
consumption, and returning the masks themselves changes what the parity script
compares. That is a design call on a function with a frozen pin, so it waits for
a decision rather than being made mid-session. It is P7's first bullet and blocks
the decomposition species-conservation gate.

---

## 15. M20 The decomposition return shape, and species through the parallel split

### M20 `Cell` and `Decomposition` are NamedTuples

Decided on the brief "simpler, more robust, no performance cost". Two facts
settled it, both measured rather than assumed:

1. **No caller unpacks.** The only production consumer, `advance_parallel`, read
   `cell[0]`, `cell[1]`, `cell[2]`. The two test callers use `c[0]` and
   `cells, rest = ...`. So per-cell arity can grow without touching a single
   positional read.
2. **`grid_up_nd` has no `orig/` pin.** Section 14 said it did. Wrong:
   `check_decomposition.py` pins `grid_up_the_system`, a different function in the
   same module. That removed the constraint that had made this look hard.

So:

    class Cell(NamedTuple):
        main; border; origin; main_species=None; border_species=None

    class Decomposition(NamedTuple):
        cells; rest; rest_species=None

One unconditional shape, always. Rejected alternatives and why: a conditional
3-or-5-tuple makes the return type depend on an argument, which is the opposite of
robust; a second `grid_up_nd_labelled` duplicates the origin draw and therefore
the RNG consumption, so the two functions would silently give different
decompositions from the same seed. Performance is unaffected because a NamedTuple
IS a tuple -- `cell[0]` compiles to the same index -- and the reassembly path adds
no indirection.

The outer tuple gained a third field, so the two `cells, rest = ...` call sites
became `cells, rest, rest_species = ...`. Two edits, both covered by existing
tests.

### Species through the parallel split

`grid_up_nd` takes `species=None` and slices it with the SAME boolean masks as the
coordinates. `advance_cell` takes `main_species` / `border_species`, feeds them to
its cell list, and returns the surviving interior's types read from the same slots
the coordinates come from. `_run_cell` threads them through the worker argument
tuple. `advance_parallel` passes `state.species` and reassembles in cell order with
`rest` last, asserting length agreement rather than trusting it.

None in, None out at every level, so a single-species run is unchanged:
`test_parallel_parity` and `test_parallel_submission` pass untouched, and
`check_decomposition.py` is green.

### Gate

`tests/unit/test_decomposition_species.py`, 8 tests, 0.14 s.

- `cell[0] is cell.main` and friends, so the positional reads provably did not
  change meaning.
- Single species gives None in every species field.
- A wrong-length column raises, naming both counts.
- Species TOTALS are invariant across the split.
- Every molecule keeps its OWN type through the split, tagged by coordinate, which
  is exact because the split copies rather than perturbs. Borders are checked too:
  a scrambled border would corrupt the energies a worker computes without ever
  showing up in the totals.
- The round trip reassembles the exact multiset of (position, species) pairs.
- A guard on the guard: shuffling the input column and re-splitting must make the
  identity check fail, so a test that could pass on scrambled data is ruled out.

Full suite 576 passed, 1 skipped, 1 xfailed, 0 failed. mypy still 4 pre-existing.

### A geometry fact I had backwards

I asserted `borders <= mains` on the reasoning that a border molecule is "ordinary
fluid that merely sits still", as `advance_cell`'s docstring puts it. It fails. The
relationship is the reverse: because the gap between neighbouring cells is at least
`buffer` on every axis, a cell's border ring of width `buffer` lies entirely inside
the GAP, so every border molecule is owned by no cell -- which is what
`grid_up_nd`'s docstring means by `rest` being a superset of all borders.

This matters for the totals gate. Excluding borders from the composition count is
not merely conservative, it is required: a border molecule is already counted once,
in `rest`. Mains plus rest is a partition; mains plus borders would double count.
The corrected statement is now the pinned one.

### Still open for Gate P3

The four statistical gates, all of which need the typed insert path (M19) first:
degenerate mixture, ideal-gas limit per species, swap stationary composition, and
the swap handler that the last of those exercises. Single-species bit-identity is
already covered by the P4 snapshot.

---

## 16. `TrialSet` carries species, and M21 the species-draw correction

### Done: the column through `TrialSet`

`gcmc/trials.py`. `species` is a row-aligned int64 field, defaulting to all-zeros,
added to `__init__`, the fancy `__getitem__`, `mask`, `delete`, `copy` and `merge`
-- every method, as the class docstring demands of any new field. It sits LAST in
the constructor signature because every existing caller builds a `TrialSet`
positionally with up to six arguments; inserting it earlier would have silently
landed one of those in the wrong field.

Two things deliberately unchanged: an INTEGER index still returns the historical
five-tuple without species, because the Rosenbluth loop unpacks exactly five; and
the default of zeros means every existing construction means what it meant.

**Gate** `tests/unit/test_trialset_species.py`, 11 tests, 0.16 s. Each method
checked with per-row-distinguishable fields, so a misalignment is a mismatch and
not a coincidence. The load-bearing test is the real pool pipeline,

    pool = trials.mask(km).merge(members)
    pool = pool[rng.permutation(len(pool))]

with the two sides of the merge given DISJOINT species sets, so any leakage across
the merge or the permutation shows as a coordinate carrying the wrong type. Plus a
guard on the guard confirming the permutation genuinely reorders, since otherwise
that test would pass with the column held anywhere at all.

`check_trials.py` green, full suite 587 passed, 1 skipped, 1 xfailed.

### M21 (DERIVATION, NOT YET MEASURED) The species-draw correction is a
### three-way partition, and a reinserted member contributes nothing

M6 says the Rosenbluth correction carries the species-draw probability. Section 14
showed that is only half true. Writing out which atoms are DRAWN in which
direction gives a three-way partition rather than a per-insertion factor.

Notation: the body holds members M; the move tombstones `n_remove` of them
(`M_rem`, the code's `disk_in`) and the DP selects a clique K from the shared pool
`P = T' u M_rem`, where `T'` is the surviving trial cloud. Positions and species of
a trial atom are both drawn; a member's are not, they are part of the current
configuration.

    K n M_rem      a member reselected at its own position. Same particle, same
                   species, present in BOTH configurations. Drawn in NEITHER
                   direction, so it contributes 1. Measured at 12 of 214
                   unchanged-position particles in one hs_branch window.

    K \ M_rem      a genuine insertion, from the trial cloud. Its species was
                   drawn forward with probability p_s. In the reverse move it is a
                   member, so it is not drawn. Contributes p_s to the FORWARD
                   proposal density.

    M_rem \ K      a genuine deletion. Not drawn forward; in the reverse move it
                   must be regenerated as a trial draw. Contributes p_s to the
                   REVERSE proposal density.

so the correction to the log acceptance is

    sum_{M_rem \ K} log p_s  -  sum_{K \ M_rem} log p_s

and NOT `- sum_{K} log p_s`. The difference is exactly the reselected members. It
enters alongside the existing `k_trials` term, which is the position half of the
same trial-density measure: `k_trials = -log((n_trials - n_remove)/n_add)` in the
add branch, cancelling identically in `repack` where `n_add == n_remove`. By the
same argument the species correction does NOT cancel in a repack unless the
composition of K matches that of M_rem, which is the point: a repack is how the
sampler changes composition at fixed N.

Assumption this rests on, stated so it can be checked rather than inherited: the
sweep's one-body weight folds `z_s` and NOT `z_s p_s`. If a future implementation
folds `p_s` into the DP weight instead, these factors are already accounted for
and adding them again double counts. One or the other, never both.

**This is a derivation. It has not been measured, and it is not to be trusted until
it is.** Two arbiters, in order of sharpness:

1. **Degenerate species with non-uniform `p_trial`.** All species identical in
   sigma, epsilon, lambda and mu, all `k_ij = 0`, so the mixture is one species
   relabelled and the stationary composition must be uniform, `x_s = 1/T`, for
   ANY `p_s`. A wrong correction makes composition track `p_s` instead. Cheap,
   exact in expectation, and it isolates the species bookkeeping from every other
   term because the energies are identical by construction.
2. **`p_s` independence at a real state point** (Gate P5): three trial vectors at
   fixed mu give the same composition within block error.

Arbiter 1 should be run FIRST and is the one that will localise a mistake; arbiter
2 confirms it survives asymmetry. If either fails, the term above is wrong and the
partition is where to look, not the activities.

### Not done

The typed insert path itself: drawing a trial's species from `p_s` when the cloud
is built, assembling `chosen_species` from the pool by row, folding `z_s` into the
one-body weights, and adding the M21 correction to `neg_log_acc`. All four are one
change and belong together -- wiring any subset gives a sampler that runs and is
wrong -- so they wait for the derivation above to be reviewed rather than being
half-landed.

---

## 17. M21 as code, and its arbiter written first

No acceptance was wired this session. Two things landed instead: the derived
correction as a pure function with its structure pinned, and the measurement that
will judge it, written before the code it judges.

### The correction is now reviewable as code

`moves.species_draw_correction(log_p_trial, pool_species, member_rows, chosen_rows)`.
Provenance is decided by POOL ROW, never by comparing coordinates: a reselected
member and a trial atom that happens to land on it are indistinguishable by
position, and the row index is exact. That means the caller must keep the
permutation it applies to the pool and apply it to the row bookkeeping in the same
breath -- `pool = pool[rng.permutation(len(pool))]` currently discards it.

Two things written down that the prose version left implicit:

**Sign convention.** The function returns a contribution to log(acceptance ratio).
The engine assembles `neg_log_acc`, which is minus that, so a caller adds
`-species_draw_correction(...)`, exactly as it already does with `corr`. Stated
because an inverted combinatorial prefactor in this acceptance has cost a session
once already and the two conventions meet within a few lines of each other.

**A hand-checkable sign consequence, measured not argued.** At uniform `p_s = 1/T`
the sum collapses to `(n_deleted - n_inserted) log(1/T) = +dN log T`. Adding a
particle to a T-species mixture means the proposal had to guess its species among
T options, so the forward proposal is T times less likely and the ratio gains a
factor of T. Verified against the closed form at T = 2, 3, 5 over random exchanges.

**Gate** 7 new tests in `tests/unit/test_move_split.py` (18 total, 9 s). They pin
the STRUCTURE of the derived expression, not its correctness, and specifically the
two mistakes most likely when wiring it: a reselected member charged `p_s`, and the
sign inverted. Also that a one-for-one exchange within a species cancels for any
`p_s`, that provenance is by row and not by species (two rows may hold the same
species, and only the row decides whether an atom was drawn), and that an empty
exchange is free.

### The arbiter, written before the code it judges

`tests/engine/test_species_bookkeeping_arbiter.py`, marked `slow`.

Make the species physically indistinguishable -- identical sigma, epsilon, lambda,
mu, all `k_ij = 0` -- and the mixture is one species wearing T labels, so the
stationary composition MUST be uniform for ANY `p_trial`, because `p_trial` is a
proposal knob and a proposal knob cannot move an equilibrium distribution. Set
`p_trial` far from uniform and the candidates separate:

    correct bookkeeping        ->  x_s = 1/T within binomial error
    p_s charged per insertion  ->  x_s tracks p_s

Energies are identical by construction, so nothing but the species terms can
produce a deviation. That isolation is why this runs before the `p_s`-independence
test at a real state point: if both fail, this one says the fault is in the
bookkeeping and not in the activities. Tolerance is the binomial `sqrt(N p (1-p))`
at 4 sd, derived per M14.

The composition run itself is a skip with the reason attached, because it needs the
typed insert path. What DOES run now is the part that can: two self-checks proving
the arbiter can discriminate at all.

- The two candidate expressions differ on 100% of random exchanges, by 5.6 in
  log-acceptance on average (median 5.9), measured on a skewed `p = [0.7, 0.2, 0.1]`.
  Thresholds in the test are set well below those so it is not brittle, while still
  failing if the two expressions ever converge -- which would mean one had been
  edited into the other.
- Skew is what makes the arbiter sharp. At uniform `p_trial` the two candidates
  differ only by a term in the net count, which an isotherm absorbs into an
  apparent shift in mu; under skew the disagreement is unevenly distributed across
  species, which is a composition error and nothing can absorb it. This is why the
  arbiter specifies a skewed vector rather than the default, and it is asserted
  rather than asserted-in-a-comment.

A discriminating test that cannot discriminate is worse than no test, because it
reports success. That is the whole reason those two self-checks exist.

Full suite 594 passed, 1 skipped, 1 xfailed, 0 failed. mypy still 4 pre-existing.

### The state of Gate P3

Wired and gated: the species column on `CellList`, on `ChainState` through
`advance`, on `TrialSet` through every method, through the decomposition split and
reassembly; the per-pair overlap floor centralised; the move split with its
single-species guard.

Not wired, and it is one change rather than four: draw a trial's species from
`p_s` when the cloud is built, assemble `chosen_species` from the pool by row,
fold `z_s` into the one-body weights, add the M21 term to `neg_log_acc`. Any subset
gives a sampler that runs and is wrong at a rate set by the reselection frequency.

The four statistical gates follow immediately after, and the arbiter above should
be the first of them to run.

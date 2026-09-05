# gcmc — grand-canonical Monte Carlo with BRANCH-MC insertion

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Grand-canonical (and canonical, semi-grand and quenched-annealed) Monte Carlo of
simple fluids, built around **BRANCH-MC**: a cavity-biased, multi-particle
insertion/deletion scheme that keeps acceptance usable at densities where
single-particle insertion stops working. Hard spheres and Lennard-Jones, 2D and
3D, one component or a mixture, serial or domain-decomposed, with
checkpoint/resume and a parameter-sweep launcher.

> One JSON file describes a run. One command runs or resumes it. A run is
> reproducible to the bit across a walltime kill: a single RNG is checkpointed
> and restored, so the Markov chain is continuous.

**What it is good for.** Adsorption isotherms, equations of state, mixture
composition and demixing, fluids in a quenched disordered matrix, fluids in a
hard pore, and guests in a rigid framework read from a CIF.

**What it is not.** Not molecular dynamics, not a general force-field engine, and
not polyatomic: the current engine is single-site. See
[Known limitations](#known-limitations) before you plan a study around it.

---

## Contents

- [Install & quick start](#install--quick-start)
- [The mental model](#the-mental-model)
- [What you can run — the capability matrix](#what-you-can-run--the-capability-matrix)
- [The four axes explained](#the-four-axes-explained)
  - [Potential](#1-potential)
  - [Sampler](#2-sampler)
  - [Depth — what BRANCH-MC actually is](#3-depth--what-branch-mc-actually-is)
  - [Ensemble](#4-ensemble)
- [Quenched-annealed (QA) matrix runs](#quenched-annealed-qa-matrix-runs)
- [Running: single, sweep, resume](#running-single-sweep-resume)
- [Outputs & analysis](#outputs--analysis)
- [Validation & ground truth](#validation--ground-truth)
- [Known limitations](#known-limitations)
- [Repository map](#repository-map)
- [Testing](#testing)
- [Design history](#design-history)
- [Citing](#citing)
- [Contributing](#contributing)
- [Licence](#licence)

---

## Install & quick start

Requires **Python 3.11+**. NumPy, SciPy, ASE and PyYAML are installed with the
package.

```bash
pip install -e .            # from a checkout, editable
pip install -e ".[dev]"     # plus pytest, ruff and mypy, to run the test suite
```

A minimal hard-sphere run in 3D at fixed chemical potential. Save this as
`params.json` — any key starting with an underscore is ignored by the validator,
so `_comment` is how a params file carries its own explanation:

```json
{
  "_comment": "hard spheres, 3D, fixed mu",
  "potential": { "main": "hard_core" },
  "sampler":   { "type": "rosenbluth", "depth": 1 },
  "ensemble":  { "type": "muVT", "mu": 4.3, "dimension": 3, "box": 8.0 },
  "run":       { "kind": "single", "radius": 0.5,
                 "sampling_window": 2000, "target_cycles": 100000 }
}
```

```bash
gcmc run params.json --root runs          # installed console script
python -m gcmc run params.json --root runs  # same thing, no install needed
```

That writes `runs/<hash>__mu4p3/` holding the trajectory, the checkpoint, the
params actually used and the log. Re-running the same command **resumes** from
the checkpoint; pointed at a finished run it returns immediately. The directory
name begins with a hash of everything that defines the chain, so two runs that
differ in any physical setting can never share a directory — and two that differ
only in, say, how often they checkpoint, always will.

`examples/schema.json` is the full annotated schema: every key, what it means,
what it is mutually exclusive with. It is the reference the validator is written
against, and a test asserts the two agree.

---

## The mental model

Every run is defined by **four independent axes**. Most combinations are valid;
the validator rejects the ones that are not, with a message explaining why. The
axes are:

| Axis | Question it answers | Choices |
|---|---|---|
| **Potential** | How do particles interact? | `hard_core`, `lennard_jones` (+ optional `pmf` auxiliary) |
| **Sampler** | How are moves proposed/weighted? | `metropolis`, `rosenbluth`, `branch`, `channel`, `channel_cyclic`, `reweight` |
| **Depth** | How many particles per insertion/deletion? | `0`, `1`, `2`, `3` |
| **Ensemble** | What is held fixed? | `muVT` (fixed μ), `NVT` (fixed N — and with a species file, fixed N *per species*), `semigrand` (fixed total N, free composition), `QA` (μVT in a frozen matrix) |

Plus an execution choice — **serial** (`kind: single`, exact at all depths) or
**parallel** (`kind: parallel`, domain decomposition).

---

## What you can run — the capability matrix

The headline table. ✓ = supported and validated; ✓* = supported with a caveat
(see footnotes); ✗ = rejected by the validator.

| Potential | Sampler | Depth 0 | Higher Depth | NVT | Parallel | QA matrix |
|---|---|:---:|:---:|:---:|:---:|:---:|
| hard_core | metropolis | ✓ | n/a¹ | ✓ | ✓ | ✗⁴ |
| hard_core | rosenbluth | ✓ | ✓ | ✓ | ✓* ² | ✓ |
| hard_core | branch | ✓ | ✓ | ✓ | ✓* ² | ✓ |
| lennard_jones | metropolis | ✓ | n/a¹ | ✓ | ✓ | ✗⁴ |
| lennard_jones | rosenbluth | ✓ | ✓ | ✓ | ✓* ² | ✗⁵ |
| lennard_jones | branch (= LJ+branch) | ✓ | ✓ | ✓ | ✓* ² | ✓ |
| lennard_jones + pmf | rosenbluth | ✓ | ✓ | ✓ | ✓* ² | ✗⁵ |
| hard_core / LJ + pmf | metropolis | ✓³ | n/a¹ | ✓ | ✓ | ✗ |
| 2D hard disk / LJ | rosenbluth/branch | ✓ | ✓ | ✓ | ✓* ² | ✗⁵ |
| 2D + reweight (ψ₆) | reweight | ✓ | ✓ | — | ✓* ² | ✗⁵ |

**Footnotes**
1. **metropolis is depth-0 by definition** — single-particle moves. "Depth" is a BRANCH-MC concept; it does not apply.
2. **parallel is exact at every depth** — the cell decomposition satisfies detailed balance. This footnote previously claimed the opposite ("global but not detailed balance", a density bias of ~+19% at cell = 2.7 cavity widths) and advised `cell_length ≥ 20×` the cavity. That claim was superseded by later runs and removed in 2026-08; see R17 in `docs/decisions/restructure_plan.md`. Cell size is now a performance choice, not an accuracy one.
3. **metropolis + pmf: LJ only, and it reduces to plain metropolis-LJ** — with a single trial the selection bias cancels identically, so the auxiliary is a no-op (handled automatically). **hard_core + pmf + metropolis is REJECTED**: there the pmf is the whole interaction beyond exclusion, so "the bias cancels" is not the same statement, and the path was never implemented. 
4. **QA + metropolis is rejected** — the metropolis path builds its overlap test on the fluid only and would silently let fluid overlap the frozen matrix.
5. **QA is hard_core or LJ+branch only** — the LJ-rosenbluth, pmf, 2D, and reweight paths pass kd-tree neighbour indices downstream, where frozen-matrix indices would be misread as fluid. QA also requires serial `single` mode (the parallel decomposition drops the matrix) and μVT (NVT has no insertion for the matrix to constrain).

---

## The four axes explained

### 1. Potential

- **`hard_core`** — hard spheres (3D) / hard disks (2D). Overlap is forbidden;
  there is no energy, only exclusion. The exclusion floor equals the diameter.
- **`lennard_jones`** — truncated LJ; requires a reduced temperature `T`. A mean-field
  tail correction is applied; `potential.tail` toggles it.
- **`+ pmf` auxiliary** *(optional block)* — a potential-of-mean-force auxiliary
  used to bias selection. The bias cancels exactly in the acceptance, so the
  sampled distribution is unchanged; it only improves efficiency. Validated to
  leave the equation of state invariant.
- **`+ short_range` auxiliary** *(optional block)* — dual-cutoff CBMC (Vlugt et
  al., Mol. Phys. 94, 727): the same LJ truncated at `auxiliary.cutoff` is used to
  propose and select, and the mismatch against `run.upper_cutoff` is divided back
  out in the acceptance. `channel` and `channel_cyclic` only, where the saving is
  real (the DP window width IS the interaction radius). Setting the cutoff equal
  to `upper_cutoff` is the **exact no-op control arm**: the correction runs, its
  value is exactly zero, and the chain must reproduce the auxiliary-off chain bit
  for bit. Watch `aux_corr_sd` in the log — below about 1.4 σ the correction's
  spread explodes and acceptance collapses while the run still finishes at a
  plausible density.
- **`species_file`** *(optional)* — a multi-species (mixture) run. The file
  declares each species' σ, ε, λ, `p_trial` and element symbol, the per-pair
  `k_ij`, the core-floor fraction `c` and optionally the auxiliary multiplier;
  `ensemble.mu` (or `fugacity`) then takes one entry per species. The engine
  draws a species per trial from `p_trial`, divides that proposal knob back out
  of the acceptance, proposes identity swaps, and weights every pair by its own
  σ_ij/ε_ij. `p_trial` is a proposal knob only: the stationary composition is set
  by the activities and nothing else, which is measured rather than asserted
  (`tests/engine/test_species_bookkeeping_arbiter.py`).

  **Not supported in typed mode, each refused with a message rather than run:**
  `metropolis` (the path reads no per-pair parameters and proposes no swap),
  `run.kind = "parallel"` (the cell kernel is single-species — this is stage P7),
  and the per-run `auxiliary` block (a typed run states its auxiliary in the
  species file, so there is one place for one cutoff). Species survive a
  trajectory frame and a checkpoint; a frame written by a different force field,
  or a bare coordinate table, is refused rather than mapped to the first species.

### 2. Sampler

- **`metropolis`** — textbook single-particle insertion/deletion/translation.
  The reference path; simplest, always correct, slowest in dense fluids.
- **`rosenbluth`** — cavity-biased Rosenbluth insertion. A cloud of trial
  positions is generated in a local cavity and one is chosen by its weight,
  with the Rosenbluth normalization preserving detailed balance.
- **`branch`** — full BRANCH-MC. For LJ this is **LJ+branch**: a genuine
  configurational-bias scheme over the *k-cliques* of the cavity's overlap
  graph, with the complete Rosenbluth sum under the real LJ energy (no hard-core
  auxiliary). This is the package's headline method for dense-fluid efficiency.
- **`channel`** — BRANCH-MC over an oriented rod instead of a spherical cavity.
  Trials are dropped into a thin cuboid (`channel_width` square cross-section)
  and the clique sum is evaluated by a transfer-matrix sweep along the rod's long
  axis, replacing exponential clique enumeration with a linear-cost DP. Insertion
  and removal share one rod (same centre, same rotation), which is what makes the
  move reversible. 3D only; needs `depth >= 1` or `full_repacking`.
- **`reweight`** — 2D hexatic (ψ₆) reweighting; 2D only.

#### Channel rod length: fixed vs full-span

`channel_length` takes either a number or the string `"full"`.

- **number** — a fixed-length rod. It must satisfy the periodic guard
  `sqrt(channel_length² + 2·channel_width²) ≤ box − cutoff`, so no trial pair
  ever wraps the box and the sweep can use plain local distances.
- **`"full"`** — a *full-span* rod that runs wall to wall along a freely drawn
  orientation. Its length is set per move by where the line meets the box:
  `box` for an axis-aligned line, up to `box·√3` down the body diagonal. This is
  the "tetris" move: an entire line of spheres, boundary to boundary, is replaced
  in one step. The periodic guard is lifted; instead the sweep scores every pair
  at its true minimum-image distance and evicts trials from the DP state by
  *retention* (a trial stays live until the last trial that interacts with it has
  been processed). Retention is what enforces the **seam** — when a rod's two
  ends are periodic neighbours, the entry-end trial stays in state across the
  whole sweep, so a clique closing through the boundary is counted exactly once,
  never twice, never dropped.

  Full-span requires `trial_density` (spheres per unit volume): the rod's length,
  and hence its trial count, changes from move to move, so a fixed
  `sampler.trials` is rejected. There is no orientation cap and no bandwidth cap;
  orientations are drawn Haar-uniform.

```json
{ "sampler": { "type": "channel", "full_repacking": true,
               "trial_density": 2.5,
               "channel_length": "full", "channel_width": 1.0 } }
```

### 3. Depth — what BRANCH-MC actually is

**Depth is the number of particles moved together in one insertion/deletion.**
A grand-canonical move at depth *d*:

- **add:** insert *(d+1)* particles, remove *d* → net ΔN = +1
- **remove:** insert *d*, remove *(d+1)* → net ΔN = −1
- **repack** *(NVT only):* insert *(d+1)*, remove *(d+1)* → net ΔN = 0

Depth 0 is the "mirror scheme" (single insertion/deletion, validated against the
hard-sphere equation of state). Depth ≥ 1 is the original BRANCH-MC proposal:
by coordinating several particles in a cavity, it can open space for an
insertion that single-particle moves would almost always reject — the source of
its acceptance advantage at high density. **Any depth >= 0 is supported.** The
cap of 3 that this line claimed until 2026-08 was never enforced by the code
(`sim_setup` validates only `depth >= 0`); it was enforced by `examples/schema.json`,
which therefore rejected runs the engine handles. The schema cap is gone and the
generated feature matrix exercises depth 4 on every sampler that takes one.

### 4. Ensemble

- **`muVT`** *(default)* — grand canonical. Fixed chemical potential μ; N
  fluctuates. The code's `mu` is the **total** chemical potential (activity
  z = exp(βμ)).
- **`NVT`** — canonical. Fixed particle number; specify exactly one of `N` or
  `packing_fraction` (the latter converts to the nearest integer N, held exactly
  thereafter). Moves are translations plus number-conserving **repacking**; the
  chemical-potential prefactor cancels identically, so NVT is exact, not an
  approximation of μVT.
- **`semigrand`** — semi-grand. The **total** particle count is fixed and the
  composition is not: the repack draws the species of everything it inserts, so
  the mixture relaxes toward the composition its activity ratio implies. Needs a
  species file and exactly one state-point route; only the **ratio** of the
  activities does any work, since the common factor cancels in the repack.

  With a species file, `NVT` means something stronger — every species count is
  held at what `run.initial_config` had, and the move set becomes translation
  plus a pair identity exchange, neither of which can change a count. That is
  the only ensemble here that can hold a two-phase box: in `muVT` and
  `semigrand` a single phase costs no interface and the composition goes there.

  These were one name and a flag (`NVT` plus `fixed_composition: true`) for one
  stage, which made semi-grand the ensemble you got by forgetting something.
  Three cluster campaigns were launched that way before the cause was found, so
  they are two names now and a file still carrying the flag is refused by name.
- **`QA`** — μVT in a frozen matrix (see next section).


---

## Quenched-annealed (QA) matrix runs

A QA run is a fluid equilibrating at fixed μ inside a **frozen, disordered set
of obstacles** — hard-sphere fluid adsorbed in a porous hard-sphere matrix. The
matrix is generated once, frozen, and never inserted, deleted, moved, or counted;
it enters only the overlap test. This is the standard quenched-annealed
adsorption problem (Madden–Glandt; benchmarked by Meroni 1996, Fernaud 1999,
Lomba 1993).

**The workflow has two stages:**

1. **Generate the matrix** — run an ordinary **NVT** hard-sphere simulation at
   the target matrix packing fraction, equilibrate, and use its `checkpoint.npz`
   as the frozen configuration.
2. **Run the fluid** — a μVT run whose `matrix_config` points at that
   checkpoint. Sweep μ to trace the adsorption isotherm.

```json
// qa_isotherm.json — μ-sweep at one fixed matrix
{
  "potential": { "main": "hard_core" },
  "sampler":   { "type": "rosenbluth", "depth": 0 },
  "ensemble":  { "mu": [-5.22, -4.04, -3.09, 0.23, 4.05], "dimension": 3, "box": 12.0 },
  "run": {
    "kind": "single",
    "n_cores": 5,
    "sampling_window": 2000,
    "matrix_config": "/abs/path/to/matrix_realization_01/checkpoint.npz"
  }
}
```

**`matrix_config`** is an absolute path to the frozen configuration. It accepts
`.npz` (a checkpoint's `positions`), `.extxyz`, or plain `x y z` text. The
matrix's content is hashed into the run identity, so each realization gets its
own run directory — generate ≥10 matrix seeds and aggregate across them
(quenched-disorder averaging is over matrices, not just over fluid moves).

**Chemical-potential convention.** Literature reports the *excess* chemical
potential; this code's `mu` is the *total*. They differ by the ideal term:

> **βμ_excess = βμ_code − ln(ρ_fluid)**

This is calibrated by the Widom free-volume check (`gcmc/analysis/widom.py`): in the
dilute limit βμ_excess → −ln φ, where φ is the fraction of random insertions
that clear the matrix.

**Depth guidance for QA.** Depth ≥ 1 works at moderate-to-high fluid density
(verified: seeded above and below it converges to the same density). At the very
dilute end the cavity proposal cannot bootstrap from a near-empty box, so the
dilute isotherm tail (low μ) should use **depth 0**.

---

## Running: single, sweep, resume

**Single run / resume** — one command, idempotent:

```bash
python -m gcmc run params.json --root runs
```

**Parameter sweep** — any axis given as a list expands into one run per value.
Sweepable axes: `mu`, `packing_fraction`, `N`, plus trial-cloud knobs. The
launcher enforces a core budget and runs as many combinations concurrently as
fit:

```bash
python -m gcmc.run.launcher base.json --cores 8 --root runs
```

The budget refuses to start if `(combinations) × (cores per run) > cores`.
For a parallel base, cores-per-run = `run.n_workers`; otherwise 1.

**Reproducibility.** A run directory is named `<identity-hash>__<tag>`. The hash
covers everything physically meaningful (potential, sampler, depth, μ or N,
cutoffs, ensemble, and the matrix content for QA), so distinct physics never
collide and identical physics resumes in place.

---

## Outputs & analysis

Each run directory contains:

- **`trajectory.extxyz`** — fluid configurations (one frame per write; frame
  size = current N).
- **`checkpoint.npz`** — positions + step + counters + RNG state; the only file
  read on resume.
- **`params.json`**, **`run.log`** — the exact inputs and the per-window counters.

  **The log's columns are a function of the run**: a column is written when the
  run can put a nonzero value in it. A muVT run gets add/remove/translate; an NVT
  run gets the repack counters instead, since that is the only move it proposes;
  a typed run adds one `N_<species>` column per species (raw counts, never mole
  fractions — they sum exactly over any span of rows, which a per-window fraction
  does not) and, in muVT, the identity-swap counters. A single-species muVT log is
  byte-identical to what it has always been.

  To narrow it further, name column GROUPS in the run block — `core`, `species`,
  `state`, `moves`, `clique`, `clique_acc`:

  ```jsonc
  "run": { "log_drop": ["clique", "clique_acc"] }     // remove these
  "run": { "log_columns": ["state", "moves"] }        // keep only these
  ```

  `core` and `species` are forced: without the first a row cannot be placed in
  time, and the second is the entire output of a mixture run. The two keys are
  mutually exclusive, and an unknown group name raises when the params are read.

**Radial distribution functions** (`gcmc/analysis/rdf.py`) — `RDFAccumulator` computes
fluid–fluid g₁₁(r) and, for QA runs, fluid–matrix g₀₁(r):

```python
from gcmc.analysis.rdf import RDFAccumulator

acc = RDFAccumulator(box=12.0, dimension=3, r_max=6.0, n_bins=200)
acc.set_matrix(matrix)  # QA only; omit for bulk
for frame in trajectory[burn_in:]:
    acc.accumulate(frame.get_positions())
r, g11, g01 = acc.finalize()
```

**Mixture structure** (`gcmc/analysis/demixing.py`) — the demixing order parameter and the
species-resolved g_ab(r). `alpha` is zero for labels arranged as randomly as the
positions allow, one when no unlike pair is in contact, negative when unlike
contacts are enriched; its null is exact by derivation, and
`permutation_band` measures the null's spread by relabelling the configuration in
hand.

```python
from gcmc import demixing

a = demixing.order_parameter_of(pos, species, box=16.0, dim=3, n_species=2)
mean, sd = demixing.permutation_band(pos, species, 16.0, 3, 2, rng)  # the null
```

`python -m scripts.demixing analyse <run_dir>` does this for a finished run and
writes the composition series, the order parameter per frame, the partial g_ab
and the null band as one JSON.

---

## Validation & ground truth

The package is checked against analytic and cross-method references, not just
against itself. The suite is **786 tests** in four tiers; `docs/TESTING.md` says
what each tier covers and how to run it.

| Check | Reference | Result |
|---|---|---|
| Hard-sphere EOS, 3D | Carnahan–Starling ⟨N⟩ = 292.9 at βμ = 4.3, V = 512 | within ~2% |
| Density invariant under sampler / trial knobs | all samplers agree | ✓ |
| LJ cross-sampler agreement | ⟨N⟩ band 298.5–303.5 | ✓ |
| Depth 0/1/2/3 EOS (rosenbluth & branch) | CS (depth-1 anchor 290.3) | ✓ |
| LJ+branch clique enumeration & weight symmetry | exact forward/reverse | ✓ |
| Parallel ↔ serial (2D & 3D) | η = 0.2984 | ✓ (depth 0) |
| NVT conserves N; recovers μVT equilibrium | — | ✓ |
| QA Widom dilute limit ρ/z → φ | free-volume fraction | ✓ |
| RDF normalisation (uncorrelated → g = 1; hard core) | — | ✓ |
| pmf auxiliary leaves EOS invariant (2D & 3D) | bias cancels | ✓ |
| Channel sweep clique counts and weights | brute-force enumeration | exact / 1e-10 |
| Mixture composition follows the activity ratio | analytic | ✓ |
| Framework energies | RASPA2 on the same Hamiltonian | rel ≤ 1e-6 |

Two conventions the suite is built on, both worth knowing before trusting a
number from it:

- **Nothing is asserted that cannot fail.** Every gate is written so that there
  is a stated value it would have to see to go red. Where a check turned out to
  be incapable of failing, it was replaced rather than kept for the green tick.
- **A pinned result is never regenerated to make a suite pass.** Counter-stream
  digests and golden trajectories move only with a written, reproduced cause;
  `docs/decisions/` records each one.

---

## Known limitations

- **QA depth ≥ 1 drains at the extreme-dilute tail** (⟨N⟩ ≈ 1–3) where the
  cavity proposal cannot grow from a near-empty box. Use depth 0 for the dilute
  isotherm points; depth ≥ 1 is correct at production density. A detailed-balance
  fix for the matrix-blocked trial slots across the full dilute range is pending.
- **QA is serial, μVT, hard_core or LJ+branch only** — enforced by the validator.
- **Any depth >= 0 is supported**, serially and in parallel. Cost, not
  correctness, is what bounds it in practice: the cavity holds `depth + 1`
  particles, so the trial cloud and the DP bandwidth both grow with it.
- **Full-span channel with LJ is cheap at small depth, expensive at
  `full_repacking`.** Cost is set by the DP bandwidth, i.e. how many trials sit
  inside one interaction window. Hard spheres see a window of σ, so the rod stays
  narrow and full-span tetris is inexpensive at any length. LJ sees a window of
  `rc = 2.5σ`, which admits many non-overlapping centres at once; combined with
  the unbounded `kmax` of `full_repacking` on a wall-to-wall rod, the state count
  can blow up. Use `depth = 1..3` for LJ full-span, or shorten the rod with a
  numeric `channel_length`. This is a cost limitation, not a correctness one, and
  by design there is no bandwidth cap: aborting a move on a
  configuration-dependent cost threshold would break detailed balance.

---

## Repository map

The directory a module sits in **is** its layer, and imports may only go down
the list. `tests/unit/test_layer_ladder.py` holds the same table as data and
fails if a module is added without someone deciding where it belongs, so the two
statements of the architecture cannot drift apart quietly.

| Layer | Directory | Role |
|---|---|---|
| L0 | `gcmc/_types.py`, `gcmc/config_fields.py`, `gcmc/config_specs.py` | The vocabulary every layer speaks: type aliases and the six sub-configs a layer uses to state which slice of configuration it needs |
| L1 | `forcefield/` | What interacts and how strongly; knows no geometry. Potential registry, per-pair tables, the overlap floor, species and their force-field files |
| L2 | `space/` | Where things are; knows no physics. Distances and periodic images, the persistent cell list, oriented bodies, domain decomposition |
| L3 | `system/` | The configuration being sampled — the chain state, where L1 and L2 meet |
| L4 | `proposals/` | What move to try: the move alphabet, the `TrialSet` currency, the channel exchange geometry both loops share |
| L5 | `sampling/` | How to weight a trial cloud: rosenbluth, branch (clique enumeration), channel (transfer-matrix DP sweep), reweighting, weight policies |
| L6 | `engine/` | The move loops. `MC_engine.advance` is one window; `parallel` is the same window per cell; `framework` is the framework run |
| L7 | `run/` | Making a `params.json` runnable: validation into a frozen `Config`, run/resume, on-disk formats, the run id, the migrator, the sweep launcher |
| — | `analysis/` | **Off the ladder on purpose**, at layer 3: g(r), Widom insertion, the demixing order parameter. They read a *finished* configuration, and nothing in the engine may import them — that is how analysis code stays out of a hot loop |

---

## Testing

```bash
pip install -e ".[dev]"
pytest -q                                        # the default suite, ~4 min
pytest tests/matrix -m matrix -q --timeout=600   # the generated feature matrix
```

786 tests in four tiers. **[`docs/TESTING.md`](docs/TESTING.md)** says what each
tier proves, which tier a given change needs, and how to materialise the frozen
history tags that the characterisation tier compares against.

Two conventions worth knowing before you trust a green tick here: nothing is
asserted that cannot fail, and a pinned result is never regenerated to make a
suite pass.

---

## Design history

`docs/decisions/` holds the working plans and decision records — what was
considered, what was measured, and why several obvious-looking ideas were
rejected. They are development documents rather than polished prose, and they
are tracked deliberately: a project's institutional memory should not be one
`git clean` away from gone.

Where a comment in the source says a choice was made "on measurement", the
measurement is in there.

---

## Citing

If this code contributes to published work, please cite it. `CITATION.cff` in
the repository root carries the machine-readable metadata; GitHub renders a
ready-made citation from it.

---

## Contributing

Issues and pull requests are welcome. Before opening one:

- run `pytest -q` and `pytest tests/matrix -m matrix -q` — tier 2 is the gate
  that matters, and a mechanical change must not move a single pinned digest;
- run `ruff check . && ruff format . && mypy`;
- if you add a module, place it on the layer ladder in
  `tests/unit/test_layer_ladder.py`. That is deliberately the only way to make
  the ladder test pass: a new file should be placed by someone who thought about
  where it goes, not defaulted onto the tree.

If a change moves a pinned digest or a golden trajectory, say so in the pull
request and say why. That is not a blocker — it is the one thing that must never
happen silently.

---

## Licence

MIT. See [`LICENSE`](LICENSE).

---

*Hard spheres and Lennard-Jones, 2D and 3D, grand-canonical / canonical /
semi-grand / quenched-annealed, one component or a mixture, serial or parallel —
one schema, one command, reproducible to the bit.*

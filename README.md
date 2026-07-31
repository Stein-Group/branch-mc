# gcmc — Grand Canonical Monte Carlo with BRANCH-MC insertion

A modular Python package for grand-canonical (and canonical) Monte Carlo of
simple fluids, built around **BRANCH-MC**: a cavity-biased, multi-particle
insertion/deletion scheme that pushes acceptance in dense fluids where naive
single-particle insertion fails. It runs hard spheres and Lennard-Jones in 2D
and 3D, serially or by domain decomposition, with checkpoint/resume and a
parameter-sweep launcher — and now supports **quenched-annealed (QA)** fluids
adsorbed in a frozen disordered matrix.

> One JSON file describes a run. One command runs or resumes it. A run is
> reproducible to the bit across a walltime kill: a single RNG is checkpointed
> and restored, so the Markov chain is continuous.

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

---

## Install & quick start

Requires Python 3.10+, NumPy, SciPy, and ASE.

```bash
pip install -e .          # installs the gcmc package + deps
# or, for a bare dependency install without the package:
pip install numpy scipy ase
```

A minimal hard-sphere run in 3D at fixed chemical potential:

```json
// params.json
{
  "potential": { "main": "hard_core" },
  "sampler":   { "type": "rosenbluth", "depth": 1 },
  "ensemble":  { "mu": 4.3, "dimension": 3, "box": 8.0 },
  "run":       { "kind": "single", "sampling_window": 2000, "target_cycles": 1e7 }
}
```

```bash
python -m gcmc run params.json --root runs
```

This creates `runs/<hash>__mu4p3/` containing the trajectory, checkpoint, params,
and log. Re-running the same command resumes from the checkpoint; pointed at a
finished run it returns immediately.

---

## The mental model

Every run is defined by **four independent axes**. Most combinations are valid;
the validator rejects the ones that are not, with a message explaining why. The
axes are:

| Axis | Question it answers | Choices |
|---|---|---|
| **Potential** | How do particles interact? | `hard_core`, `lennard_jones` (+ optional `pmf` auxiliary) |
| **Sampler** | How are moves proposed/weighted? | `metropolis`, `rosenbluth`, `branch`, `reweight` |
| **Depth** | How many particles per insertion/deletion? | `0`, `1`, `2`, `3` |
| **Ensemble** | What is held fixed? | `muVT` (fixed μ), `NVT` (fixed N), `QA` (μVT in a frozen matrix) |

Plus an execution choice — **serial** (`kind: single`, exact at all depths) or
**parallel** (`kind: parallel`, domain decomposition).

---

## What you can run — the capability matrix

The headline table. ✓ = supported and validated; ✓* = supported with a caveat
(see footnotes); ✗ = rejected by the validator.

| Potential | Sampler | Depth 0 | Depth 1–3 | NVT | Parallel | QA matrix |
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
2. **parallel depth ≥ 1 is approximate** — the cell decomposition satisfies global but not detailed balance, leaving a density bias set by the cavity/cell ratio (~+19% at cell = 2.7 cavity widths, +0.9% at 10×). Use `cell_length ≥ 20×` the cavity, or serial mode (exact at all depths). Depth-0 parallel is exact.
3. **metropolis + pmf reduces to plain metropolis-LJ** — with a single trial the selection bias cancels identically, so the auxiliary is a no-op (handled automatically).
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
its acceptance advantage at high density. **Depth is capped at 3.**

### 4. Ensemble

- **`muVT`** *(default)* — grand canonical. Fixed chemical potential μ; N
  fluctuates. The code's `mu` is the **total** chemical potential (activity
  z = exp(βμ)).
- **`NVT`** — canonical. Fixed particle number; specify exactly one of `N` or
  `packing_fraction` (the latter converts to the nearest integer N, held exactly
  thereafter). Moves are translations plus number-conserving **repacking**; the
  chemical-potential prefactor cancels identically, so NVT is exact, not an
  approximation of μVT.
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

This is calibrated by the Widom free-volume check (`gcmc/widom.py`): in the
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
python -m gcmc.launcher base.json --cores 8 --root runs
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

**Radial distribution functions** (`gcmc/rdf.py`) — `RDFAccumulator` computes
fluid–fluid g₁₁(r) and, for QA runs, fluid–matrix g₀₁(r):

```python
from gcmc.rdf import RDFAccumulator
acc = RDFAccumulator(box=12.0, dimension=3, r_max=6.0, n_bins=200)
acc.set_matrix(matrix)                 # QA only; omit for bulk
for frame in trajectory[burn_in:]:
    acc.accumulate(frame.get_positions())
r, g11, g01 = acc.finalize()
```

---

## Validation & ground truth

The package is checked against analytic and cross-method references, not just
self-consistency. 32 physics tests in `tests/test_physics.py`:

| Check | Reference | Result |
|---|---|---|
| Hard-sphere EOS, 3D | Carnahan–Starling ⟨N⟩ = 292.9 at βμ=4.3, V=512 | within ~2% |
| Density invariant under sampler / trial knobs | all samplers agree | ✓ |
| LJ cross-sampler agreement | ⟨N⟩ band 298.5–303.5 | ✓ |
| Depth 0/1/2/3 EOS (rosenbluth & branch) | CS (depth-1 anchor 290.3) | ✓ |
| LJ+branch clique enumeration & weight symmetry | exact forward/reverse | ✓ |
| Parallel ↔ serial (2D & 3D) | η = 0.2984 | ✓ (depth 0) |
| NVT conserves N; recovers μVT equilibrium | — | ✓ |
| QA Widom dilute limit ρ/z → φ | free-volume fraction | ✓ |
| RDF normalization (uncorrelated → g=1; hard core) | — | ✓ |
| pmf auxiliary leaves EOS invariant (2D & 3D) | bias cancels | ✓ |

---

## Known limitations

- **Parallel depth ≥ 1 is approximate** (global not detailed balance). Use a
  large cell/cavity ratio or serial mode. Depth-0 parallel is exact.
- **QA depth ≥ 1 drains at the extreme-dilute tail** (⟨N⟩ ≈ 1–3) where the
  cavity proposal cannot grow from a near-empty box. Use depth 0 for the dilute
  isotherm points; depth ≥ 1 is correct at production density. A detailed-balance
  fix for the matrix-blocked trial slots across the full dilute range is pending.
- **QA is serial, μVT, hard_core or LJ+branch only** — enforced by the validator.
- **Depth is capped at 3.**
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

| Module | Role |
|---|---|
| `sim_setup.py` | Parse & validate `params.json` → a frozen `Config`; all the rejection rules live here |
| `MC_engine.py` | The Markov chain: move proposal, Rosenbluth weights, acceptance (serial) |
| `samplers/` | Weight policies (`base.py`), branch/clique machinery (`branch.py`), Rosenbluth loop, reweighting |
| `geometry.py`, `trials.py`, `topology.py` | Spatial queries, the trial-set struct, clique/graph helpers |
| `decomposition.py`, `parallel.py` | Domain decomposition and the concurrent cell driver |
| `runner.py` | Run/resume orchestration, initial config, checkpoint loop |
| `launcher.py` | Parameter-sweep expansion with a core budget |
| `io.py`, `hash_fingerprint.py`, `state.py`, `migrate.py` | Checkpoint/trajectory I/O, run identity, live state, schema migration |
| `widom.py` | QA Widom free-volume / infinite-dilution chemical potential |
| `rdf.py` | g₁₁(r) and g₀₁(r) accumulation |
| `tests/test_physics.py` | The validation suite (the ground truth above) |

---

*Hard spheres and Lennard-Jones, 2D and 3D, grand-canonical / canonical /
quenched-annealed, serial or parallel — one schema, one command, reproducible
to the bit.*

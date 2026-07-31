"""Launcher: a Cartesian sweep over chemical potential and the cavity trial
knobs, with a hard core budget.

Sweepable axes and where they live in the base params:
    ensemble.mu                 (single value or a list; muVT)
    ensemble.packing_fraction   (single value or a list; NVT)
    ensemble.N                  (single value or a list; NVT)
    sampler.trial_density
    sampler.trial_box_side

Any axis given as a JSON list is swept; scalars are fixed. The launcher takes
the Cartesian product of all swept axes, so two list axes build a heatmap
(e.g. 4 trial_density x 4 trial_box_side = 16 runs). Each combination becomes a
concrete params.json with scalars in place, a distinct id, and its own run
directory, so combinations never collide and restart is just re-running.

Core budget (required, --cores N, or run.n_cores in the base):
    cores_per_run = run.n_workers   if run.kind == "parallel"   else 1
    total_needed  = (number of combinations) * cores_per_run
The launcher refuses to start if total_needed > N, then runs at most
floor(N / cores_per_run) combinations concurrently so the node is never
oversubscribed; the rest start as earlier ones finish.

Examples (serial unless noted):
    1 mu, 1 density, 1 box        -> 1  * 1  = 1 core
    16 mu                         -> 16 * 1  = 16 cores
    4 density x 4 box             -> 16 * 1  = 16 cores
    1 mu, parallel n_workers=30   -> 1  * 30 = 30 cores

    python -m gcmc.launcher base.json --cores 16 --root runs
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# (axis label, params section, key). Order fixes the product / id ordering.
_AXES = (
    ("mu", "ensemble", "mu"),
    ("packing_fraction", "ensemble", "packing_fraction"),
    ("N", "ensemble", "N"),
    ("target_N", "ensemble", "target_N"),
    ("target_eta", "ensemble", "target_eta"),
    ("target_density", "ensemble", "target_density"),
    ("depth", "sampler", "depth"),
    ("seed", "run", "seed"),
    ("trial_density", "sampler", "trial_density"),
    ("trial_box_side", "sampler", "trial_box_side"),
    ("channel_length", "sampler", "channel_length"),
    ("channel_width", "sampler", "channel_width"),
    # The short-range auxiliary radius. Sweeping it is how the speedup/acceptance
    # trade-off gets measured, and the sweep carries its own control: the arm at
    # cutoff == run.upper_cutoff is the exact no-op, so one launch produces both
    # the tuning curve and the reference it must agree with.
    ("aux_cutoff", "auxiliary", "cutoff"),
)

# nested pore axes (ensemble.pore.{H,R}); handled separately since they live one
# level deeper than the flat _AXES entries.
_PORE_AXES = (("poreH", "H"), ("poreR", "R"))


def _as_list(v: Any) -> list[Any]:
    """Wrap a scalar into a one-element list; pass lists/tuples through --
    how a sweep axis and a fixed value share one code path."""
    return list(v) if isinstance(v, (list, tuple)) else [v]


# box and cell_length are "shaped": a value may itself be a per-axis list (an
# orthorhombic box / anisotropic cell), which collides with the list = sweep
# convention. Disambiguate by nesting: a FLAT numeric list is one shaped value
# (anisotropic), a list that CONTAINS a list is a sweep over box-specs.
#   box: 50                       -> one cubic box
#   box: [45, 45, 56.25]          -> one anisotropic box (NOT swept)
#   box: [[45,45,56],[50,50,60]]  -> sweep over two anisotropic boxes
#   box: [[40,40,40],[50,50,50]]  -> sweep over two cubic boxes
_SHAPED_AXES = (
    ("box", "ensemble", "box"),
    ("cell_length", "run", "cell_length"),
)


def _shaped_grid(v: Any) -> list[Any]:
    """Sweep grid for a shaped axis (box / cell_length). A nested list sweeps
    over its elements; a flat numeric list is a single anisotropic value."""
    if isinstance(v, (list, tuple)):
        if any(isinstance(x, (list, tuple)) for x in v):
            return list(v)  # sweep over box-specs (each scalar or per-axis list)
        return [list(v)]  # one anisotropic value, not a sweep
    return [v]  # one scalar (cubic) value


def _shaped_label(val: Any) -> str:
    """Short id fragment for one shaped value: 50 -> '50', [45,45,56.25] ->
    '45x45x56.25'."""
    if isinstance(val, (list, tuple)):
        return "x".join(_fmt_num(x) for x in val)
    return _fmt_num(val)


def _fmt_num(x: Any) -> str:
    f = float(x)
    return str(int(f)) if f.is_integer() else str(f)


def _matrix_files_in(folder: str) -> list[str]:
    """A matrix folder (e.g. an NVT runs root) holds one subdirectory per
    realization, each with a trajectory.extxyz (the frozen matrix; its last
    frame is used). Return the per-realization trajectory paths, sorted for a
    deterministic pairing with the swept target list. Also accepts a folder of
    bare .extxyz/.npz files."""

    root = Path(folder).resolve()
    # case A: subdirectories each containing trajectory.extxyz
    subs = sorted(str(q) for q in root.glob("*/trajectory.extxyz"))
    if subs:
        return subs
    # case B: subdirectories each containing checkpoint.npz
    subs = sorted(str(q) for q in root.glob("*/checkpoint.npz"))
    if subs:
        return subs
    # case C: a flat folder of matrix files
    flat = sorted(
        [str(q) for q in root.glob("*.extxyz")] + [str(q) for q in root.glob("*.npz")]
    )
    return flat


def expand_sweep(
    base: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return (combos, labels): one concrete params dict per Cartesian
    combination of the swept axes, with the matching short id suffixes. A
    scalar axis contributes one value; a list axis is swept; absent axes are
    not varied."""
    present, grids = [], []
    for label, section, key in _AXES:
        sec = base.get(section, {})
        if key in sec:
            present.append((label, section, key))
            grids.append(_as_list(sec[key]))
    for label, section, key in _SHAPED_AXES:
        sec = base.get(section, {})
        if key in sec:
            present.append((label, section, key))
            grids.append(_shaped_grid(sec[key]))
    pore = base.get("ensemble", {}).get("pore", {})
    for label, key in _PORE_AXES:
        if key in pore:
            present.append((label, "ensemble.pore", key))  # nested marker
            grids.append(_as_list(pore[key]))
    if not present:  # nothing sweepable present: one run
        return [copy.deepcopy(base)], ["single"]

    # only axes with more than one distinct value are "swept"; those name the
    # run dirs. single-valued axes are still applied but kept out of the label.
    varying = {i for i, g in enumerate(grids) if len({repr(x) for x in g}) > 1}
    combos, labels = [], []
    for values in itertools.product(*grids):
        p = copy.deepcopy(base)
        parts = []
        for i, ((label, section, key), val) in enumerate(zip(present, values)):
            if section == "ensemble.pore":
                p.setdefault("ensemble", {}).setdefault("pore", {})[key] = val
            else:
                p.setdefault(section, {})[key] = val
            if i in varying and section != "ensemble.pore":
                shaped = (label, section, key) in _SHAPED_AXES
                shown = _shaped_label(val) if shaped else val
                parts.append(f"{label}{shown}")  # pore size is in the dir prefix
        lbl = "_".join(parts)
        if lbl:
            p.setdefault("run", {})["sweep_label"] = lbl
        combos.append(p)
        labels.append(lbl or "single")

    # matrix-folder pairing: if run.matrix_config is a FOLDER (not a file), pair
    # each swept run with one realization matrix from that folder, by index --
    # combo[i] uses matrix[i] (target[i] <-> matrix[i]). This makes a 16-target
    # sweep pick up 16 matrices and run them one-to-one.
    mcfg = base.get("run", {}).get("matrix_config")
    if mcfg is not None and Path(mcfg).is_dir():
        files = _matrix_files_in(mcfg)
        if not files:
            raise ValueError(
                f"run.matrix_config is a folder ({mcfg}) but no matrix files "
                f"were found in it (expected */trajectory.extxyz, */checkpoint.npz, "
                f"or *.extxyz/*.npz)."
            )
        if len(files) < len(combos):
            raise ValueError(
                f"sweep needs {len(combos)} matrices (one per swept value) but "
                f"the folder {mcfg} has only {len(files)}. Provide at least as "
                f"many realizations as swept values, or shorten the sweep."
            )
        for i, p in enumerate(combos):
            p.setdefault("run", {})["matrix_config"] = files[i]
            # the realization index in the label makes each run dir identifiable
            labels[i] = (
                f"{labels[i]}_m{i:02d}" if labels[i] != "single" else f"m{i:02d}"
            )

    return combos, labels


def cores_per_run(base: dict[str, Any]) -> int:
    """Cores one run of this params file consumes.

    One cell is one core, so for a parallel run this is the CELL COUNT and is
    never stated separately (parallel_input_plan.md D4/D6). The count is asked
    of the config layer rather than reconstructed here, so the launcher and the
    engine can never disagree about how many cores a run needs.
    """
    run = base.get("run", {})
    if run.get("kind") != "parallel":
        return 1
    # NB: `run.n_cores` is the SWEEP BUDGET (combinations x cores-per-run), which
    # is a different quantity from cores-per-run and is read by main() below. It
    # is NOT rejected here. An earlier version of this function refused it on a
    # parallel base, which made the two checks contradict each other: omit it and
    # main() says "cores is required: pass --cores N or set run.n_cores", supply
    # it and this function refused it. The thing decision D4 retires is stating
    # cores PER RUN, and there has never been a key for that other than the
    # deprecated run.n_workers.
    from .sim_setup import from_params

    # A swept base is not a single run: `run.seed: [1, 2, ...]` and friends are
    # still lists here, and from_params rightly refuses an unexpanded sweep. Ask
    # the first expanded combo instead. expand_sweep returns [base] unchanged
    # when nothing is swept, so this covers both cases with one path.
    combos, _ = expand_sweep(base)
    return from_params(combos[0]).n_workers


def plan(
    base: dict[str, Any], cores: int
) -> tuple[list[dict[str, Any]], list[str], int, int]:
    """Validate the budget; return (combos, labels, per_run, concurrency)."""
    combos, labels = expand_sweep(base)
    per = cores_per_run(base)
    total = len(combos) * per
    if total > cores:
        raise ValueError(
            f"core budget exceeded: {len(combos)} run(s) x {per} core(s) each "
            f"= {total} cores needed, but only {cores} available. One cell is "
            f"one core, so a parallel run costs its cell count: reduce the "
            f"sweep, lower run.n_cell, or request more cores."
        )
    return combos, labels, per, max(1, cores // per)


def launch_sweep(
    base: dict[str, Any],
    cores: int,
    *,
    root: str = "runs",
    launch_dir: str = "launch",
    threads: int = 1,
    pin: bool = True,
    wait: bool = True,
) -> list[int] | list[subprocess.Popen[bytes]]:
    """Expand the sweep and run every combination as its own `python -m gcmc
    run` subprocess within the core budget: per-run params snapshots under
    `launch_dir`, BLAS/OMP thread caps exported so numpy cannot oversubscribe,
    optional taskset pinning for serial runs (with a probe-and-fallback, see
    inline comment), and a simple keep-the-budget-full scheduler when
    `wait=True`. Returns exit codes (wait) or Popen handles (no wait)."""
    combos, labels, per, concurrency = plan(base, cores)
    os.makedirs(launch_dir, exist_ok=True)
    # serial runs take one core and can be pinned; parallel runs manage their
    # own workers, so they are not taskset-pinned
    taskset = shutil.which("taskset") if (pin and per == 1) else None

    next_core = [0]

    def start(params: dict[str, Any], label: str) -> subprocess.Popen[bytes]:
        pf = Path(launch_dir) / f"params_{label}.json"
        with open(pf, "w") as f:
            json.dump(params, f, indent=2, sort_keys=True)
        env = dict(os.environ)
        for v in _THREAD_VARS:
            env[v] = str(threads)
        cmd = [sys.executable, "-m", "gcmc", "run", str(pf), "--root", root]
        if taskset is not None:
            # prefix taskset, but fall back to unpinned if affinity is refused
            # (e.g. restricted cgroups): pinning is an optimisation, not a
            # correctness requirement, so a node that forbids it must not fail
            core = str(next_core[0] % cores)
            next_core[0] += 1
            probe = subprocess.run([taskset, "-c", core, "true"], capture_output=True)
            if probe.returncode == 0:
                cmd = [taskset, "-c", core, *cmd]
        return subprocess.Popen(cmd, env=env)

    pending = list(zip(combos, labels))
    if not wait:
        return [start(p, l) for p, l in pending]

    done: list[int] = []
    running: list[tuple[subprocess.Popen[bytes], int]] = []
    used = 0
    while pending or running:
        while pending and used + per <= cores and len(running) < concurrency:
            p, l = pending.pop(0)
            running.append((start(p, l), per))
            used += per
        pr, c = running.pop(0)
        done.append(pr.wait())
        used -= c
    return done


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper over `launch_sweep`; see --help for the knobs."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(prog="python -m gcmc.launcher")
    ap.add_argument("base", help="base params.json; sweepable axes may be lists")
    ap.add_argument(
        "--cores",
        type=int,
        default=None,
        help="cores available this job (required; or run.n_cores)",
    )
    ap.add_argument("--root", default="runs")
    ap.add_argument("--launch-dir", default="launch")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without launching",
    )
    args = ap.parse_args(argv)

    with open(args.base) as f:
        base = json.load(f)
    cores = args.cores
    if cores is None:
        cores = base.get("run", {}).get("n_cores")
    if not isinstance(cores, int) or cores < 1:
        ap.error("cores is required: pass --cores N or set run.n_cores in the base")

    combos, labels, per, concurrency = plan(base, cores)
    print(
        f"{len(combos)} run(s) x {per} core(s) = {len(combos) * per} of "
        f"{cores} cores; up to {concurrency} concurrent"
    )
    if args.dry_run:
        for l in labels:
            print("  ", l)
        return 0

    result = launch_sweep(
        base,
        cores,
        root=args.root,
        launch_dir=args.launch_dir,
        threads=args.threads,
        pin=not args.no_pin,
        wait=not args.no_wait,
    )
    if args.no_wait:
        print(f"launched {len(result)} runs")
    else:
        print(f"finished {len(result)} runs, return codes {result}")
        if any(result):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

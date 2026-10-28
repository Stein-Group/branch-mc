"""Launcher: a Cartesian sweep over chemical potential and the cavity trial
knobs, with a hard core budget.

Sweepable axes and where they live in the base params:
    ensemble.mu                 (single value or a list; muVT)
    ensemble.fugacity / .z      (a list of {species: value} maps; typed runs)
    ensemble.pressure           (framework runs, with ensemble.y)
    ensemble.packing_fraction   (single value or a list; NVT)
    ensemble.N                  (single value or a list; NVT)
    sampler.trials              (absolute trial-pool size; XOR trial_density)
    sampler.trial_density
    sampler.trial_box_side
    sampler.channel_length
    sampler.channel_width
    engine.type                 (metropolis_dynamics / glauber_dynamics /
                                 suwa_todo_dynamics; any sampler that can
                                 supply the dynamics a pool)
    engine.depth                (metropolis_dynamics only)
    engine.kmax_margin          (ladder dynamics only)
    auxiliary.gamma             (fixed gamma; each arm a scalar or an array of
                                 up to 9 per-level exponents)
    run.seed
    potential.species_file      (the model axis: a list of species files)
    run.initial_config          (the start axis: a list of starting
                                 configurations, each a path to a file or to an
                                 equilibrated library directory, or the literal
                                 token "grid". The start is not part of the run
                                 id, so the arms are kept apart by the label
                                 alone and the label is built from as much of
                                 each path as it takes to separate them.)

Any axis given as a JSON list is swept; scalars are fixed. The launcher takes
the Cartesian product of all swept axes, so two list axes build a heatmap
(e.g. 4 trial_density x 4 trial_box_side = 16 runs, or 10 gamma x 8
trial_box_side = 80). Each combination becomes a
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

    python -m gcmc.run.launcher base.json --cores 16 --root runs
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# (axis label, params section, key). Order fixes the product / id ordering.
_AXES = (
    ("mu", "ensemble", "mu"),
    # typed and framework state points: a list of maps (fugacity, z) or of
    # total pressures (framework runs, with ensemble.y) is one run per entry
    ("f", "ensemble", "fugacity"),
    ("z", "ensemble", "z"),
    ("p", "ensemble", "pressure"),
    ("packing_fraction", "ensemble", "packing_fraction"),
    ("N", "ensemble", "N"),
    ("target_N", "ensemble", "target_N"),
    ("target_eta", "ensemble", "target_eta"),
    ("target_density", "ensemble", "target_density"),
    ("depth", "engine", "depth"),
    ("seed", "run", "seed"),
    # `trials` (absolute count) and `trial_density` (count per unit measure of
    # the trial body) are the two ways to say how big the trial pool is, and
    # sim_setup refuses both at once. Both are sweepable: a k-sweep holds the
    # budget fixed while the geometry varies, a density sweep holds the
    # candidates-per-unit-length fixed instead. Omitting `trials` here made a
    # list-valued sampler.trials survive expansion unexpanded, and every run
    # then died in from_params on int(list).
    ("trials", "sampler", "trials"),
    ("trial_density", "sampler", "trial_density"),
    ("trial_box_side", "sampler", "trial_box_side"),
    ("channel_length", "sampler", "channel_length"),
    ("channel_width", "sampler", "channel_width"),
    # THE DYNAMICS AXIS. Sweepable because the only useful comparison between
    # `metropolis_dynamics` and `glauber_dynamics` is at the SAME state point, the
    # same box and the same trial cloud, and tau_N per CPU-second is a ratio
    # that means nothing otherwise. Without this axis a head-to-head is two
    # launches whose roots have to be kept apart by hand, which is how a
    # comparison ends up between two different experiments. Inert for every
    # params file that does not set it: a single-valued axis is applied and
    # kept out of the label.
    ("engine", "engine", "type"),
    # The short-range auxiliary radius. Sweeping it is how the speedup/acceptance
    # trade-off gets measured, and the sweep carries its own control: the arm at
    # cutoff == run.upper_cutoff is the exact no-op, so one launch produces both
    # the tuning curve and the reference it must agree with.
    ("aux_cutoff", "auxiliary", "cutoff"),
    # THE MODEL AXIS. Every other entry above is a knob of the experiment; this
    # one sweeps the force field itself, which is what the k_ij demixing series
    # is: `eps_ij = (1 - k_ij) sqrt(eps_i eps_j)`, and a run at a different k is
    # a different model rather than the same model at a different setting. The
    # species file says so itself -- "a run at a different k gets its own file
    # rather than an override" (M1, `ff/species_binary_reduced.yaml`) -- so the
    # sweep is over files, and the file's sha256 is already in the run identity,
    # which is what stops two k values from ever sharing a run directory.
    ("model", "potential", "species_file"),
    # THE START AXIS, and it is a convergence instrument rather than a
    # convenience. A composition at fixed mu can be trapped: measured at
    # k_AB = 0.5, two samplers from the SAME equimolar start settled on opposite
    # compositions and neither had converged. The only way to know is to run the
    # same state point from opposite starts and require them to MEET (the rule
    # the species-draw arbiter needs), which means the start is a
    # swept axis of the experiment and not a fixed input to it.
    ("start", "run", "initial_config"),
)

#: Axes whose value is a PATH. The id fragment is the file's stem rather than
#: the path, so a run directory is `modelbinary_k0.3` and not a flattened
#: absolute path with slashes in it.
_FILE_AXES = (
    ("model", "potential", "species_file"),
    ("start", "run", "initial_config"),
)

# nested pore axes (ensemble.pore.{H,R}); handled separately since they live one
# level deeper than the flat _AXES entries.
_PORE_AXES = (("poreH", "H"), ("poreR", "R"))


def _map_label(val: dict[str, Any]) -> str:
    """Id fragment for a per-species state point: `{A: -0.05, B: 0.05}` becomes
    `A-0.05_B0.05`.

    A typed state point is a MAP, and `repr` of a dict carries braces, quotes,
    colons and spaces -- every one of which is either illegal or awkward in a
    directory name, and the run directory is how a human finds a run. Rendered in
    the map's own key order, which is the species file's order, so two runs of one
    model always render their species the same way round.
    """
    return "_".join(f"{k}{_fmt_num(v)}" for k, v in val.items())


#: State-point axes whose values span decades (Pa, activities): labelled in %g
#: so a folder reads `fCH4_1e+06` / `p2.5e+05`, not `fCH41000000`.
_STATE_AXES = (
    ("f", "ensemble", "fugacity"),
    ("z", "ensemble", "z"),
    ("p", "ensemble", "pressure"),
)


def _state_label(val: Any) -> str:
    if isinstance(val, dict):
        return "_".join(f"{k}_{float(v):g}" for k, v in val.items())
    return f"{float(val):g}"


def _file_fragment(val: Any, depth: int) -> str:
    """The last `depth` path components of a path-valued axis, the file itself
    rendered as its stem. A leading `species_` is dropped because every file on
    the model axis carries it and a prefix shared by every value distinguishes
    nothing. The filesystem root is not a component anyone can read, so it is
    left out."""
    parts = [q for q in Path(str(val)).parts if q not in ("/", "\\", "")]
    if not parts:
        return ""
    stem = Path(parts[-1]).stem
    parts[-1] = stem[len("species_") :] if stem.startswith("species_") else stem
    return "_".join(parts[-depth:])


def _file_label_grid(vals: list[Any]) -> list[str]:
    """Fragments for one file axis, decided with the whole axis in view so that
    two distinct paths never render the same.

    The stem alone is the fragment in the ordinary case and every sweep that
    already ran keeps the directory name it had. It is not enough in general,
    and the start axis is where that bites: a sweep over per-run output folders
    is a list of paths whose stems are all `final`, or all `trajectory`, and the
    directory they differ in is the part the stem throws away. Two arms sharing
    a fragment share their `sweep_label`, and for `run.initial_config` that is
    fatal rather than cosmetic -- the start is deliberately NOT part of the run
    id (a warm start reaches the same equilibrium), so the label is the only
    thing keeping the arms in separate directories, and without it they would
    write one checkpoint and one log between them.

    So the fragment grows leftwards one path component at a time until the axis
    separates, at one common depth so the names stay comparable, and a digest of
    the full path is the last resort for paths that differ only in characters a
    fragment drops.
    """
    uniq = list(dict.fromkeys(str(v) for v in vals))
    deepest = max((len(Path(u).parts) for u in uniq), default=1)
    for depth in range(1, deepest + 1):
        frag = {u: _file_fragment(u, depth) for u in uniq}
        if len(set(frag.values())) == len(uniq):
            return [frag[str(v)] for v in vals]
    frag = {
        u: f"{_file_fragment(u, 1)}x{hashlib.sha1(u.encode()).hexdigest()[:6]}"
        for u in uniq
    }
    return [frag[str(v)] for v in vals]


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


# `auxiliary.gamma` is shaped like box and cell_length -- one value may be an
# ARRAY, the per-level exponents -- but it takes the OPPOSITE convention, and
# the difference is deliberate rather than an inconsistency.
#
# For a box, the natural thing to write is one anisotropic box, and sweeping
# boxes is the rare case, so a flat list reads as one value. For gamma the
# natural thing to write is a SWEEP: ten exponents crossed with eight trial box
# sides is the heatmap this axis exists for, and `[0.0, 0.05, 0.1]` means three
# arms to everyone who writes it. So here the outer list is ALWAYS the sweep and
# each element is one arm, a scalar (applies at every level) or an array (the
# first levels, the rest unbiased):
#   gamma: 0.1                       -> one arm, 0.1 at every level
#   gamma: [0.0, 0.05, 0.1]          -> three arms, each a scalar
#   gamma: [[0.29,0.10,0.03], 0.1]   -> two arms, one array and one scalar
#   gamma: [[0.29,0.10,0.03]]        -> ONE arm, that array
# The last line is the price of the convention: to pin a single array without
# sweeping it, wrap it. A bare flat list is a sweep of scalars.
_GAMMA_AXES = (("gamma", "auxiliary", "gamma"),)


def _gamma_grid(v: Any) -> list[Any]:
    """Sweep grid for the gamma axis: the outer list is the sweep, a scalar is
    one arm."""
    return list(v) if isinstance(v, (list, tuple)) else [v]


#: Longest gamma fragment rendered value by value. A nine-level arm spelled out
#: is about ninety characters, and the run directory is how a human finds a run,
#: so past this the tail becomes a digest of the full array: the fragment stays
#: readable, stays unique, and stays the same across relaunches because the
#: digest is of the values and nothing else.
_GAMMA_LABEL_MAX = 28


def _gamma_label(val: Any) -> str:
    """Id fragment for one gamma arm: 0.1 -> '0.1', [0.29, 0.103, 0.0332] ->
    '0.29x0.103x0.0332', and a long array -> '0.193548x9Lx4f1c2a'.

    'x', the shaped axis's separator, and NOT a hyphen: `run_paths` renders a
    directory name by replacing '-' with 'm' and '.' with 'p', so a hyphenated
    arm would land on disk as `gamma0p29m0p1`, which reads as a subtraction. The
    separator has to survive that substitution to stay readable, and 'x' does.
    """
    if not isinstance(val, (list, tuple)):
        return _fmt_num(val)
    full = "x".join(_fmt_num(x) for x in val)
    if len(full) <= _GAMMA_LABEL_MAX:
        return full
    dig = hashlib.sha1(full.encode()).hexdigest()[:6]
    return f"{_fmt_num(val[0])}x{len(val)}Lx{dig}"


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
    for label, section, key in _GAMMA_AXES:
        sec = base.get(section, {})
        if key in sec:
            present.append((label, section, key))
            grids.append(_gamma_grid(sec[key]))
    pore = base.get("ensemble", {}).get("pore", {})
    for label, key in _PORE_AXES:
        if key in pore:
            present.append((label, "ensemble.pore", key))  # nested marker
            grids.append(_as_list(pore[key]))
    if not present:  # nothing sweepable present: one run
        return [copy.deepcopy(base)], ["single"]

    # A file axis renders as a fragment of the path, and that fragment has to
    # separate the axis by itself (see _file_label_grid), which cannot be
    # decided one value at a time inside the product below.
    file_frags = {
        i: dict(zip((str(v) for v in g), _file_label_grid(list(g))))
        for i, (a, g) in enumerate(zip(present, grids))
        if a in _FILE_AXES
    }

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
                axis = (label, section, key)
                if axis in _SHAPED_AXES:
                    shown = _shaped_label(val)
                elif axis in _GAMMA_AXES:
                    shown = _gamma_label(val)
                elif axis in _FILE_AXES:
                    shown = file_frags[i][str(val)]
                elif axis in _STATE_AXES:
                    shown = _state_label(val)
                elif isinstance(val, dict):
                    shown = _map_label(val)
                else:
                    shown = val
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

    seen: dict[str, int] = {}
    for lbl in labels:
        seen[lbl] = seen.get(lbl, 0) + 1
    clash = sorted(l for l, n in seen.items() if n > 1)
    if clash:
        raise ValueError(
            f"{len(clash)} sweep label(s) are shared by more than one "
            f"combination: {clash[:5]}. The label names the run directory and "
            f"nothing else separates two runs that agree on every identity-"
            f"bearing field, so these arms would overwrite each other's "
            f"checkpoint and log. Almost always a value listed twice on one "
            f"axis; drop the repeat."
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
    # THE CPUS THIS PROCESS MAY USE, not 0..cores-1. Under SLURM a job's
    # allowed set is whatever the scheduler handed it (22-43, say). Pinning to
    # absolute ids 0..15 either fails the probe below and runs unpinned, or, on
    # a node without cpuset confinement, lands every job sharing the node on the
    # SAME sixteen cpus, so each run's wall time depends on how many neighbours
    # it had. Pin inside the allowed set instead.
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        allowed = list(range(cores))
    if len(allowed) < min(cores, len(combos)):
        log.warning(
            "the core budget is %d but this process may run on only %d cpus "
            "(%s); runs will share cpus and their wall times will not compare",
            cores,
            len(allowed),
            allowed,
        )

    next_core = [0]

    def start(params: dict[str, Any], label: str) -> subprocess.Popen[bytes]:
        # The snapshot name must be unique to its CONTENT, not just its label.
        # `label` only encodes the swept axes ("mu15"), so two jobs launched
        # from different base files into the same launch_dir would write the
        # same `params_mu15.json`. The child reads it a second or so later
        # (after importing numpy/scipy/ase), by which time the second job can
        # have overwritten it, and the first job's runs would silently take the
        # other job's physics. A digest of the
        # content makes the names disjoint, and identical content writing the
        # same file is harmless. Written to a temp name and renamed, so a
        # reader never sees a half-written file.
        text = json.dumps(params, indent=2, sort_keys=True)
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        pf = Path(launch_dir) / f"params_{label}__{digest}.json"
        tmp = pf.with_name(f"{pf.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, pf)
        env = dict(os.environ)
        for v in _THREAD_VARS:
            env[v] = str(threads)
        cmd = [sys.executable, "-m", "gcmc", "run", str(pf), "--root", root]
        if taskset is not None:
            # prefix taskset, but fall back to unpinned if affinity is refused
            # (e.g. restricted cgroups): pinning is an optimisation, not a
            # correctness requirement, so a node that forbids it must not fail
            core = str(allowed[next_core[0] % min(cores, len(allowed))])
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
    ap = argparse.ArgumentParser(prog="python -m gcmc.run.launcher")
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

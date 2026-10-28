"""Runner: the one loop that makes a params.json runnable.

Fresh and restart are the same path. Compute the run id, look for a checkpoint,
load it if present or initialise if not, then run `while step < target`. A
finished run re-entered is a no-op, so requeues and array jobs are idempotent.
The single generator's state is checkpointed, so a restart continues the exact
Markov chain rather than reseeding.

A lockfile (pid, host, time) in the run directory stops two jobs resuming the
same id; a stale lock past the timeout is taken over.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._types import RNG, FloatArray, IntArray

if TYPE_CHECKING:
    from ..system.state import ChainState
    from .hash_fingerprint import RunPaths
    from .sim_setup import Config

from .. import config_specs
from ..analysis.energy import configuration_energy
from ..engine import metropolis_dynamics
from ..forcefield.lj_table import LennardJonesTable
from ..space import geometry
from ..system.state import ChainState
from . import hash_fingerprint, io, sim_setup

# Module logger is `logger`, not `log`: run() binds `log` to the io.Log run-log
# writer, so the getLogger idiom's usual `log` name is already taken here.
logger = logging.getLogger(__name__)


def _save_pore_geometry(cfg: Config, directory: str | Path) -> None:
    """Write the wall geometry as an extxyz of marker points so the pore can be
    loaded next to the fluid trajectory in OVITO. Slit: two wall planes tiled
    with markers; cylinder: the lateral surface tiled in (theta, z). Markers use
    element 'W' (wall) so they are visually distinct from the fluid."""
    # Sub-configs, projected once per call.
    ens = config_specs.ensemble_spec(cfg)
    import numpy as _np

    L = ens.box
    pts: list[tuple[float, float, float]] = []
    n = 40
    if ens.pore_geometry == "slit":
        assert ens.pore_H is not None  # slit config carries H
        z0 = (L - ens.pore_H) / 2.0
        xs = _np.linspace(0, L, n)
        ys = _np.linspace(0, L, n)
        for zc in (z0, z0 + ens.pore_H):  # both wall surfaces
            for x in xs:
                for y in ys:
                    pts.append((x, y, zc))
    else:
        assert ens.pore_R is not None  # cylinder config carries R
        R = ens.pore_R
        cx = cy = L / 2.0
        th = _np.linspace(0, 2 * _np.pi, n, endpoint=False)
        zs = _np.linspace(0, L, n)
        for z in zs:
            for t in th:
                pts.append((cx + R * _np.cos(t), cy + R * _np.sin(t), z))
    arr = _np.asarray(pts, float)
    path = Path(directory) / "pore_geometry.extxyz"
    with open(path, "w") as f:
        f.write(f"{len(arr)}\n")
        f.write(
            f'Lattice="{L} 0 0 0 {L} 0 0 0 {L}" '
            f"Properties=species:S:1:pos:R:3 geometry={ens.pore_geometry}\n"
        )
        for p in arr:
            f.write(f"W {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


# ---------------------------------------------------------------- grid start
# `initial_config: "grid"` is a LITERAL TOKEN, not a path: it means "bootstrap
# from a commensurate lattice at a reasonable fluid density" instead of from an
# empty box. It exists because the cavity-biased exchange move cannot grow a
# depth>=1 chain out of N=0 in reasonable walltime at high mu, and warm-starting
# from the equilibrated library is not always possible (a new box, a new T, a
# fresh MOF). The token is checked BEFORE the file/directory dispatch, so a
# directory that happens to be named "grid" is unreachable by that name; point
# `initial_config` at "./grid" or an absolute path if you really have one.
_GRID_TOKENS = ("grid",)

# Default target packing fraction of the bootstrap lattice, per dimension.
# Both sit in the dense-but-unambiguously-fluid range (hard-disk freezing
# eta ~ 0.70, hard-sphere freezing eta ~ 0.494), so the lattice melts within a
# few windows instead of being metastably stuck in a crystal, and neither is so
# dense that the first removals are all rejected. 0.35 in 3D is the same working
# density the neighbour benchmarks seed at (rho = 0.65 -> eta = 0.34).
_GRID_DEFAULT_ETA = {2: 0.45, 3: 0.35}

# Lattice spacing must EXCEED the contact distance, not merely equal it: the
# hard-core test forbids r <= diameter (potentials/hard_sphere.weight_pair), so
# a lattice sitting exactly at contact would start every pair overlapping, and
# L/n in float64 can land a hair under the target anyway. 1% clearance costs
# ~3% in density in 3D and removes both problems.
_GRID_CLEARANCE = 1.01


def _sphere_volume(diameter: float, dimension: int) -> float:
    """Volume (3D) or area (2D) of one particle of the given diameter -- the
    numerator of the packing fraction. Only the two dimensions the engine
    supports are defined."""
    r = diameter / 2.0
    if dimension == 2:
        return math.pi * r**2
    if dimension == 3:
        return 4.0 / 3.0 * math.pi * r**3
    raise ValueError(f"grid bootstrap supports dimension 2 or 3, not {dimension}")


def _parse_grid_spec(spec: str) -> tuple[bool, float | None]:
    """Recognise the grid token and its optional explicit packing fraction.

    Returns (is_grid, eta). `eta` is None for a bare "grid" (use the default for
    the dimension, or the muVT target density when one is set). Accepted forms,
    case- and whitespace-insensitive: "grid", "grid:0.3", "grid=0.3"."""
    s = spec.strip().lower()
    if s in _GRID_TOKENS:
        return True, None
    for sep in (":", "="):
        head, _, tail = s.partition(sep)
        if head.strip() in _GRID_TOKENS and tail.strip():
            try:
                eta = float(tail)
            except ValueError:
                raise ValueError(
                    f"initial_config {spec!r}: the part after {sep!r} must be a "
                    f"packing fraction, e.g. 'grid{sep}0.30'"
                ) from None
            if eta <= 0.0:
                raise ValueError(
                    f"initial_config {spec!r}: packing fraction must be positive"
                )
            return True, eta
    return False, None


def _grid_sites(
    box_dims: list[float], dimension: int, diameter: float, target_eta: float
) -> FloatArray:
    """A periodic-commensurate simple-cubic lattice filling the box at the
    density closest to `target_eta` that an integer number of lattice planes
    allows -- which may be a little above it as well as below.

    The per-axis site count is round(L_i / s) for the isotropic spacing s that
    realises the target, then decremented until the ACTUAL spacing L_i / n_i
    clears the contact distance. Rounding rather than flooring matters: a box
    that is 6.99 ideal spacings wide would otherwise drop to 6 and start 30%
    below the requested density. The spacing is recomputed from L_i / n_i so the
    lattice is commensurate with the periodic box -- a lattice built on the ideal
    spacing would leave a short seam at the wrap, and that seam is exactly where
    overlaps appear."""
    floor = diameter * _GRID_CLEARANCE
    s = (_sphere_volume(diameter, dimension) / target_eta) ** (1.0 / dimension)
    counts: list[int] = []
    for length in box_dims:
        n = max(1, int(round(length / s)))
        while n > 1 and length / n < floor:
            n -= 1
        if length / n < floor:
            raise ValueError(
                f"cannot build a grid start: box side {length} is shorter than "
                f"the contact clearance {floor:.4f} (diameter {diameter}), so not "
                f"even one lattice plane fits"
            )
        counts.append(n)
    axes = [
        (np.arange(n) + 0.5) * (length / n)
        for n, length in zip(counts, box_dims, strict=True)
    ]
    return np.array(list(itertools.product(*axes)), dtype=float).reshape(-1, dimension)


def _grid_config(cfg: Config, spec: str) -> FloatArray:
    """Build the `initial_config: "grid"` start.

    Density precedence: an explicit "grid:<eta>" wins; otherwise a muVT target
    density (run.target_N / target_eta / target_density) is honoured, so the
    chain starts where the controller is trying to drive it; otherwise the
    per-dimension default. NVT ignores all of this and delegates to the exact-N
    lattice, because repacking conserves N and the starting count MUST be the
    target, not a density that happens to be nearby.

    Sites that are physically unavailable are then dropped -- overlapping a
    frozen QA obstacle, or lying outside a hard pore -- so the token behaves
    sensibly in a matrix or under confinement instead of starting inside a wall.

    The achieved density is the closest an integer lattice can get, not the
    target exactly; it is logged next to the target, and a gap wider than 15%
    (a box only a few diameters across, or one where the next denser lattice
    would overlap) is warned about rather than left for the reader to notice."""
    # Sub-configs, projected once per call.
    kern = config_specs.kernel_spec(cfg)
    samp = config_specs.sampler_spec(cfg)
    ens = config_specs.ensemble_spec(cfg)
    if ens.target_n is not None:
        return _nvt_grid(cfg)

    _, eta = _parse_grid_spec(spec)
    dim = ens.dimension
    box_dims = kern.box_dims
    volume = float(np.prod(box_dims))
    v_particle = _sphere_volume(kern.diameter, dim)
    source = "requested"
    if eta is None:
        target_count = ens.target_count
        if target_count is not None and target_count > 0:
            eta = float(target_count) * v_particle / volume
            source = "muVT target density"
        else:
            eta = _GRID_DEFAULT_ETA[dim]
            source = f"default for {dim}D"

    eta_max = v_particle / (kern.diameter * _GRID_CLEARANCE) ** dim
    if eta > eta_max:
        raise ValueError(
            f"initial_config {spec!r}: packing fraction {eta:.4f} is above the "
            f"densest lattice this bootstrap can build without overlap "
            f"({eta_max:.4f} in {dim}D at diameter {kern.diameter}). Start lower "
            f"and let GCMC compress, or warm-start from an equilibrated file."
        )

    pts = _grid_sites(box_dims, dim, kern.diameter, eta)
    n_lattice = len(pts)

    mpos = ens.matrix_positions
    if mpos is not None and len(mpos) > 0:
        # QA: a lattice site inside an obstacle is not a configuration, it is an
        # instant overlap. Drop those sites; the pores keep the rest.
        from scipy.spatial import cKDTree as _KD

        tree_m = _KD(np.asarray(mpos, float) % np.asarray(box_dims), boxsize=box_dims)
        clear = tree_m.query(pts % np.asarray(box_dims), k=1)[0] > (
            kern.diameter * _GRID_CLEARANCE
        )
        pts = pts[clear]

    if ens.pore_geometry is not None:
        pts = np.array(
            [p for p in pts if geometry.inside_pore(p, ens, samp)], dtype=float
        )
        pts = pts.reshape(-1, dim)

    if len(pts) == 0:
        raise ValueError(
            f"initial_config {spec!r}: every one of the {n_lattice} lattice sites "
            f"was rejected (obstacle overlap or outside the pore). Lower the "
            f"packing fraction, or warm-start from a file."
        )

    achieved = len(pts) * v_particle / volume
    logger.info(
        "grid initial config: %d molecules, packing fraction %.4f (target %.4f, %s%s)",
        len(pts),
        achieved,
        eta,
        source,
        f"; {n_lattice - len(pts)} sites dropped as unavailable"
        if len(pts) != n_lattice
        else "",
    )
    # Warn only about LATTICE quantisation, measured before the filters: sites
    # dropped for obstacle overlap or confinement are excluded volume doing its
    # job, are already itemised in the line above, and blaming them on lattice
    # planes would be a false explanation.
    lattice_eta = n_lattice * v_particle / volume
    if abs(lattice_eta - eta) > 0.15 * eta:
        logger.warning(
            "grid lattice packs at %.4f, %+.0f%% off the %.4f asked for: an "
            "integer number of lattice planes cannot land closer in a box of %s "
            "at diameter %g. GCMC will relax it either way; warm-start from a "
            "file if the exact starting density matters.",
            lattice_eta,
            100.0 * (lattice_eta - eta) / eta,
            eta,
            "x".join(f"{length:g}" for length in box_dims),
            kern.diameter,
        )
    return pts


def _nvt_grid(cfg: Config) -> FloatArray:
    """NVT: EXACTLY target_n non-overlapping particles on a grid. Repacking
    conserves N, so the starting count must be the target; the configuration
    then equilibrates in place from this lattice."""
    # Sub-configs, projected once per call.
    kern = config_specs.kernel_spec(cfg)
    ens = config_specs.ensemble_spec(cfg)
    n = ens.target_n
    assert n is not None  # only called when target_n is set
    if n == 0:
        return np.empty((0, ens.dimension))
    # smallest cubic grid with at least n valid sites, spacing >= diameter
    import math as _m

    per = max(2, int(_m.ceil(n ** (1.0 / ens.dimension))))
    while True:
        step = ens.box / per
        if step < kern.diameter:  # too tight: would overlap
            raise ValueError(
                f"cannot place {n} particles in box {ens.box} without "
                f"overlap (grid spacing {step:.3f} < diameter "
                f"{kern.diameter}); reduce N/packing_fraction or enlarge box"
            )
        g = (np.arange(per) + 0.5) * step
        pts = np.array(list(itertools.product(g, repeat=ens.dimension)))
        if len(pts) >= n:
            return pts[:n]
        per += 1


def _init_config(cfg: Config, rng: RNG) -> FloatArray:
    """Initial fluid configuration for a fresh run: the literal token "grid"
    builds a reasonable-density lattice (see `_grid_config`); any other
    `initial_config` value is a path -- a single file, or the pre-equilibrated
    library matched on mu, box, dimension, and temperature for LJ (details in
    the comments below); with no `initial_config` at all, bulk muVT starts empty
    and the constrained ensembles seed a valid start of their own."""
    # Sub-configs, projected once per call.
    kern = config_specs.kernel_spec(cfg)
    samp = config_specs.sampler_spec(cfg)
    ens = config_specs.ensemble_spec(cfg)
    io_ = config_specs.io_spec(cfg)
    # start from a pre-equilibrated configuration when one is given, else one
    # seed molecule, as the original starts (insertion needs an occupied
    # cavity, so an empty system never grows).
    #
    # initial_config is either a directory or a file, decided by whether it has
    # a file extension (not by os.path.isdir, which silently mis-resolves when a
    # job's working directory differs from where it was launched). A directory is
    # the equilibrated library, nested by system, temperature (LJ only) and box:
    #     equilibrated/lj/T_0.85/box_20/mu_0.0_lj_3d_box20.xyz
    #     equilibrated/hard_disk_2d/box_200/mu_1.00_hard_2d_box_200.xyz
    #     equilibrated/NVT/box_20/...           matrix/ is loaded via matrix_config
    # The per-run file is picked by matching chemical potential, box, dimension
    # and (for LJ) temperature, so every mu in a sweep loads its own start. Use an
    # absolute path to be safe across SLURM / subprocess working directories.
    spec = io_.initial_config
    # The grid token is resolved first: it is a keyword, so it must never reach
    # the suffix test below (it has no extension and would be mistaken for the
    # equilibrated-library directory, failing with a confusing "not found").
    if spec and _parse_grid_spec(spec)[0]:
        return _grid_config(cfg, spec)
    if not spec:
        if ens.pore_geometry is not None:
            # confined pore: seed a few non-overlapping particles INSIDE the pore
            # so the depth>=1 proposal has an anchor (it cannot grow from N=0).
            from scipy.spatial import cKDTree as _KD

            box = ens.box
            dim = ens.dimension
            placed: list[FloatArray] = []
            for _ in range(20 * 300):
                if len(placed) >= 20:
                    break
                pt = rng.uniform(0, box, dim)
                if not geometry.inside_pore(pt, ens, samp):
                    continue
                if placed:
                    tp = _KD(np.array(placed) % box, boxsize=[box] * dim)
                    if tp.query(pt % box, k=1)[0] < kern.diameter:
                        continue
                placed.append(pt)
            return np.array(placed) if placed else np.empty((0, dim))
        mpos = ens.matrix_positions
        if mpos is not None and len(mpos) > 0 and ens.target_n is None:
            # quenched-annealed muVT: seed a few valid fluid particles (the
            # depth>=1 proposal needs an anchor and cannot grow from N=0), each
            # clearing the matrix and the others. GCMC then fills the pores.
            from scipy.spatial import cKDTree as _KD

            box = ens.box
            tree_m = _KD(np.asarray(mpos, float) % box, boxsize=[box] * ens.dimension)
            placed = []
            target_seed = 20  # small; enough to bootstrap insertion
            for _ in range(target_seed * 200):
                if len(placed) >= target_seed:
                    break
                pt = rng.uniform(0, box, ens.dimension)
                if tree_m.query(pt % box, k=1)[0] < kern.diameter:
                    continue  # overlaps matrix
                if placed:
                    tp = _KD(np.array(placed) % box, boxsize=[box] * ens.dimension)
                    if tp.query(pt % box, k=1)[0] < kern.diameter:
                        continue  # overlaps an already-placed fluid sphere
                placed.append(pt)
            return np.array(placed) if placed else np.empty((0, ens.dimension))
        if ens.target_n is not None:
            # NVT: exactly target_n particles on a lattice (verbatim, extracted
            # to _nvt_grid so the "grid" token can reach the same code).
            return _nvt_grid(cfg)
        # Bulk muVT with no initial_config: start EMPTY. Insertion centres on a
        # uniform random point and needs nothing present, so N=0 -> 1 grows on
        # its own; removal and displacement only ever act on existing particles.
        # No lattice by DEFAULT: this method places trials at uniform random
        # points, so a gridded start is foreign to it and an empty box is the
        # honest zero. Set initial_config to a pre-equilibrated file (e.g. the
        # metastable liquid) or to the token "grid" when growing from empty is
        # too slow -- at high mu with depth >= 1 the cavity proposal needs
        # something to nucleate around, so an empty box can crawl.
        return np.empty((0, ens.dimension))

    path = _resolve_start_file(cfg)
    assert path is not None  # `spec` is set and is not the grid token
    pos = io.load_config_file(path, ens.dimension)
    logger.info("loaded initial config: %d molecules from %s", len(pos), path)
    return pos


def _resolve_start_file(cfg: Config) -> Path | None:
    """WHICH file a warm start reads, or None when the start is not a file.

    Split out of `_init_config` so that "which file" is decided once and read
    twice: once for positions, once for the species column. The library
    search below is the expensive and error-prone half -- it globs a tree and
    matches mu, box, dimension and T out of file NAMES -- and having two copies
    of it would let a typed run load its coordinates from one frame and its
    labels from another.

    None means there is no file to read: the grid token builds a lattice, and an
    absent `initial_config` is dispatched by ensemble (pore seed, QA seed, NVT
    lattice, or an empty box). Those starts carry no species information, so a
    typed run built from them starts all-type-0 and the engine assigns species as
    it inserts -- which is correct, and different from a warm start silently
    doing the same thing.
    """
    io_ = config_specs.io_spec(cfg)
    kern = config_specs.kernel_spec(cfg)
    ens = config_specs.ensemble_spec(cfg)
    pot_spec = config_specs.potential_spec(cfg)
    spec = io_.initial_config
    if not spec or _parse_grid_spec(spec)[0]:
        return None
    if Path(spec).suffix:  # has an extension -> file
        path = Path(spec).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"initial_config file not found: {path}\n"
                f"  (resolved from cwd={Path.cwd()}). Use an absolute path if the "
                f"run is launched from a different directory than the config."
            )
        return path

    # A directory points at the equilibrated library, which is nested by system,
    # (temperature for LJ) and box, with per-mu files whose decimal formatting
    # differs between families, e.g.
    #     equilibrated/lj/T_0.85/box_20/mu_0.0_lj_3d_box20.xyz
    #     equilibrated/hard_disk_2d/box_200/mu_1.00_hard_2d_box_200.xyz
    #     equilibrated/NVT/box_20/mu_..._box_20.xyz
    # Rather than reconstruct an exact filename (the formats are inconsistent),
    # search the tree and match on the values parsed from each candidate: chemical
    # potential and box always, dimension and (for LJ) temperature when present.
    import re as _re

    root = Path(spec).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"initial_config directory not found: {root} (resolved from "
            f"cwd={Path.cwd()}). Point it at the equilibrated library folder or "
            f"at a single .xyz file."
        )
    is_lj = kern.potential_type in ("lj", "ljbranch", "ljaux")
    all_xyz = [str(q) for q in root.rglob("*.xyz")]
    if not all_xyz:
        raise FileNotFoundError(
            f"no .xyz configurations found anywhere under initial_config directory "
            f"{root}. Expected files like mu_<mu>_<family>_<dim>d_box<L>.xyz."
        )

    def _num(pattern: str, text: str) -> float | None:
        m = _re.search(pattern, text)
        return float(m.group(1)) if m else None

    matches = []
    for f in all_xyz:
        name = Path(f).name
        f_mu = _num(r"mu_(-?\d+(?:\.\d+)?)", name)
        f_box = _num(r"box_?(\d+(?:\.\d+)?)", f)  # from the path (box_<L> dir)
        f_dim = _num(r"(\d)d", name)
        f_T = _num(r"[/\\]T_(\d+(?:\.\d+)?)[/\\]", f)
        if f_mu is None or abs(f_mu - ens.chem_pot) > 1e-6:
            continue
        if f_box is not None and abs(f_box - float(ens.box)) > 1e-6:
            continue
        if f_dim is not None and int(f_dim) != ens.dimension:
            continue
        if is_lj and f_T is not None and abs(f_T - float(pot_spec.temp)) > 1e-6:
            continue
        matches.append(f)

    if len(matches) == 0:
        sample = sorted(str(Path(f).relative_to(root)) for f in all_xyz)[:12]
        raise FileNotFoundError(
            f"no equilibrated config under {root} matches this run "
            f"(mu={ens.chem_pot}, box={ens.box}, dimension={ens.dimension}"
            + (f", T={pot_spec.temp}" if is_lj else "")
            + "). "
            f"{len(all_xyz)} .xyz files are present, for example: {sample}. "
            f"Check that a matching mu/box (and T for LJ) exists, or point "
            f"initial_config at the exact file."
        )
    if len(matches) > 1:
        rels = sorted(str(Path(m).relative_to(root)) for m in matches)[:8]
        raise ValueError(
            f"initial_config is ambiguous: {len(matches)} files under {root} match "
            f"mu={ens.chem_pot}, box={ens.box}"
            + (f", T={pot_spec.temp}" if is_lj else "")
            + f": {rels}. Point "
            f"initial_config at a more specific subdirectory or at the exact file."
        )

    return Path(matches[0])


def _init_species(cfg: Config, n: int) -> IntArray | None:
    """The species column of a warm start, or None when there is nothing to read.

    None in three cases, and only one of them is interesting: a single-species
    run has no column by design; a start that is not a file (grid, pore seed, QA
    seed, NVT lattice, empty box) has no labels to read; and a typed run reading
    a file gets its column checked rather than guessed -- see
    `io.load_config_species`, which raises on an unknown symbol, on a bare
    coordinate table and on a count mismatch.
    """
    kern = config_specs.kernel_spec(cfg)
    if kern.species is None or kern.n_species <= 1 or n == 0:
        return None
    path = _resolve_start_file(cfg)
    if path is None:
        return None
    return io.load_config_species(path, symbols=kern.species.symbol, n_expected=n)


def acquire_lock(paths: RunPaths, timeout: float = 1) -> None:
    """Take the run-directory lock (pid, host, time written into the lock
    file) or raise if a lock younger than `timeout` seconds exists. Prevents
    two jobs resuming the same run id and interleaving one Markov chain; a
    stale lock (crashed job) expires by age rather than needing manual
    cleanup."""
    if paths.lock.exists():
        age = time.time() - paths.lock.stat().st_mtime
        if age < timeout:
            with open(paths.lock) as lf:
                holder = lf.read().strip()
            raise RuntimeError(f"run is locked ({age:.0f}s old): {holder}")
    paths.lock.write_text(f"{os.getpid()} {socket.gethostname()} {time.time():.0f}")


def release_lock(paths: RunPaths) -> None:
    """Remove the lock file; already-gone is fine (idempotent)."""
    paths.lock.unlink(missing_ok=True)


def _assert_start_is_legal(cfg: Config, pos: FloatArray, what: str) -> None:
    """Reject a starting configuration that already contains a hard-core overlap.

    Applies to every entry point -- a warm-start .xyz, a grid bootstrap, or a
    resumed checkpoint -- because none of them is trusted and an overlap is
    unrecoverable once the chain is running (see geometry.assert_no_overlap).
    Skipped for a soft potential, where an overlap is merely expensive.
    """
    # Sub-configs, projected once per call.
    pot_spec = config_specs.potential_spec(cfg)
    ens = config_specs.ensemble_spec(cfg)
    if pot_spec.potential != "hard_sphere":
        return
    sides = ens.box_sides if ens.box_sides is not None else (ens.box,) * ens.dimension
    geometry.assert_no_overlap(
        pos, pot_spec.lower_cutoff, np.asarray(sides, dtype=float), what=what
    )


def _save(
    paths: RunPaths, cfg: Config, state: ChainState, params: dict[str, Any]
) -> None:
    """Checkpoint the full chain state (positions, step, rng, counters, live
    mu and its history, and the species column) through io's atomic
    write-and-rotate.

    THE SPECIES COLUMN WAS NOT BEING PASSED. `io.save_checkpoint` grew a
    `species` parameter at D2 and had no caller; `_read_npz` reads the array back
    and `load_checkpoint` returns it, and the resume path below never looked at
    it. So a typed run's checkpoint carried no species, and a resumed run started
    with every particle relabelled species 0 -- the same loss as the window
    boundary (see `metropolis_dynamics.advance`), one level up and across a restart.

    None for a single-species run, which is what keeps a schema-2 checkpoint and
    a schema-3 single-species checkpoint identical in content.
    """
    io.save_checkpoint(
        paths,
        positions=state.positions,
        step=state.step,
        rng=io.rng_state(state.rng),
        params=params,
        mu=state.mu,
        n_history=state.n_history,
        species=state.species,
    )


def _find_matrix_snapshot(
    params: dict[str, Any], root: str, dimension: int
) -> str | None:
    """On resume with the external matrix gone, locate the run directory whose
    params match and return its matrix.npz snapshot path (or None)."""
    import json

    target = json.dumps(params, sort_keys=True)
    for snap in Path(root).glob("*/matrix.npz"):
        pj = snap.parent / "params.json"
        if pj.exists():
            try:
                if json.dumps(json.loads(pj.read_text()), sort_keys=True) == target:
                    return str(snap)
            except Exception:
                continue
    return None


#: Largest allowed difference between the running U and a full recompute, in
#: units of max(1, |U|). Roundoff over millions of moves stays far below it.
ENERGY_DRIFT_TOL = 1e-6


def _check_energy(state: ChainState, full: float, paths: RunPaths) -> None:
    """Compare the running U against a full recompute and resynchronise it.

    A difference beyond roundoff means an accepted move's energy change was
    booked wrongly, which is a defect, so it raises with both numbers.
    """
    assert state.energy is not None
    if abs(state.energy - full) > ENERGY_DRIFT_TOL * max(1.0, abs(full)):
        raise RuntimeError(
            f"{paths.directory}: running U = {state.energy!r} but a full "
            f"recompute gives {full!r} at step {state.step}. An engine booked "
            "a wrong energy change; the run is stopped rather than logging it."
        )
    state.energy = full


def run(
    params: dict[str, Any] | str, root: str = "runs"
) -> tuple[ChainState, RunPaths]:
    """Run (or resume) one simulation to its configured number of windows.

    The full lifecycle lives here: build the Config, load or snapshot the
    frozen matrix, resolve the run directory from the physics fingerprint,
    take the lock, resume from a checkpoint when one exists (restoring the rng
    so the chain continues bit-identically) or build a fresh initial
    configuration, then loop windows: advance the engine, adjust mu when a
    target N is configured, log a row, checkpoint and write trajectory frames
    on their cadences. Ctrl-C/kill between windows leaves a resumable
    directory behind; that is the property everything in here serves."""
    if isinstance(params, str):
        with open(params) as f:
            params_dict: dict[str, Any] = json.load(f)
    else:
        params_dict = params
    params = params_dict  # keep the historical local name below
    if "framework" in params_dict:
        raise ValueError(
            "a params file with a `framework` block is a MOF run: run it with "
            "`python -m gcmc run` or gcmc.framework.driver.run_framework"
        )

    cfg = sim_setup.from_params(params_dict)
    # Sub-configs. Projected HERE, straight after the Config is built, and
    # again below once the frozen matrix has been folded in.
    #
    # Twice, not once, because `cfg` is genuinely rebuilt in between:
    # `dataclasses.replace(cfg, matrix_positions=...)` happens up to three times
    # while the matrix is resolved, and a projection taken before that would
    # carry `matrix_positions=None` into every reader downstream -- a QA run
    # silently sampling with no obstacles. A projection is a SNAPSHOT of a frozen
    # object, and re-taking it is how a snapshot follows a rebuild.
    #
    # This first one exists only to resolve the matrix itself.
    ens = config_specs.ensemble_spec(cfg)
    if ens.matrix_config is not None:
        import dataclasses

        import numpy as _np

        mpath = Path(ens.matrix_config).resolve()
        if mpath.is_dir():
            raise IsADirectoryError(
                f"run.matrix_config is a folder ({mpath}). A folder of matrix "
                f"realizations is only valid as a SWEEP: give a list target "
                f"(target_N/target_eta/target_density) and launch with "
                f"`python -m gcmc.run.launcher`, which pairs each swept value with one "
                f"matrix from the folder. For a single run, point matrix_config "
                f"at one trajectory.extxyz / checkpoint.npz file."
            )
        # On a fresh launch we load the external matrix to compute the
        # content-addressed run dir, then snapshot it inside. On a resume the
        # snapshot in the (params-determined) run dir is authoritative, so the
        # external file may be gone -- find the run dir by probing the snapshot.
        if mpath.exists():
            mpos = io.load_matrix(mpath, ens.dimension)
            cfg = dataclasses.replace(cfg, matrix_positions=mpos)
            paths = hash_fingerprint.run_paths(cfg, root)
            io.ensure_run_dir(paths, params)
            snap = paths.directory / "matrix.npz"
            if snap.exists():
                cfg = dataclasses.replace(
                    cfg, matrix_positions=io.load_matrix(snap, ens.dimension)
                )
            else:
                _np.savez(snap, positions=_np.asarray(mpos, float))
            logger.info("loaded frozen matrix: %d obstacles from %s", len(mpos), mpath)
        else:
            # external matrix gone: this must be a resume. Find the existing run
            # dir whose params.json matches and load its matrix snapshot.
            found = _find_matrix_snapshot(params_dict, root, ens.dimension)
            if found is None:
                raise FileNotFoundError(
                    f"matrix_config not found: {mpath}, and no existing run "
                    f"snapshot under {root!r} to resume from. Restore the matrix "
                    f"file or the run directory."
                )
            mpos = io.load_matrix(found, ens.dimension)
            cfg = dataclasses.replace(cfg, matrix_positions=mpos)
            paths = hash_fingerprint.run_paths(cfg, root)
            io.ensure_run_dir(paths, params)
            logger.info("resumed frozen matrix: %d obstacles from %s", len(mpos), found)
    else:
        paths = hash_fingerprint.run_paths(cfg, root)
        io.ensure_run_dir(paths, params)
        if ens.pore_geometry is not None:
            _save_pore_geometry(cfg, paths.directory)
    # Re-projected: `cfg` may have been rebuilt three times above to fold in the
    # frozen matrix, and every reader from here down must see the final one.
    kern = config_specs.kernel_spec(cfg)
    ens = config_specs.ensemble_spec(cfg)
    exe = config_specs.execution_spec(cfg)
    io_ = config_specs.io_spec(cfg)
    samp = config_specs.sampler_spec(cfg)

    if exe.mode == "parallel":
        # stdout, not run.log: this belongs in the scheduler's .out next to any
        # traceback, and run.log is machine-parsable CSV. Placed after the last
        # of the three run_paths branches so it prints on every path, including
        # a resume and a frozen-matrix run.
        print(sim_setup.describe_parallel(cfg), flush=True)

    acquire_lock(paths)

    try:
        ckpt = io.load_checkpoint(paths)
        if ckpt is None:
            seed = params.get("run", {}).get("seed")
            state = ChainState.fresh(ens.dimension, seed=seed)
            state.positions = _init_config(cfg, state.rng)
            # The species column of the same start. None unless this is a
            # multi-species run reading a FILE -- a generated start (grid, pore
            # seed, QA seed, NVT lattice, empty box) has no labels to carry, and
            # the engine assigns species as it inserts.
            state.species = _init_species(cfg, len(state.positions))
            _assert_start_is_legal(cfg, state.positions, "initial_config")
        else:
            gen = io.set_rng_state(np.random.default_rng(), ckpt["rng"])
            _ckpt_species = ckpt.get("species")
            state = ChainState(
                positions=ckpt["positions"],
                step=int(ckpt["step"]),
                rng=gen,
                mu=ckpt.get("mu"),
                n_history=list(ckpt.get("n_history", [])),
                species=(
                    None
                    if _ckpt_species is None
                    else np.asarray(_ckpt_species, dtype=np.int64)
                ),
            )
            # A typed run whose checkpoint carries no species column is a
            # checkpoint written before the column was saved. Refusing is the
            # only honest option: continuing would silently relabel every
            # particle species 0 and report a composition the chain never had.
            if kern.n_species > 1 and state.species is None and len(state.positions):
                raise ValueError(
                    f"{paths.checkpoint} has no species column but this is a "
                    f"{kern.n_species}-species run. It was written before the "
                    "column was checkpointed; resuming would relabel every "
                    "particle as species 0. Start a fresh run, or warm-start from "
                    "a trajectory frame, which now carries real symbols."
                )
            _assert_start_is_legal(cfg, state.positions, "checkpoint")
            io.trim_log(paths.log, state.step / 1e6)
            if io_.traj_every >= 1:
                io.trim_trajectory(paths.trajectory, state.step)

        potential = cfg.build_potential()
        weigher = metropolis_dynamics.make_weigher(
            config_specs.kernel_spec(cfg), potential
        )
        # THE DYNAMICS DISPATCH. Two axes, resolved in the order that makes
        # each one legible: the execution mode picks serial or parallel, and
        # within serial the `engine` field picks the dynamics.
        #
        # THE TWO LADDER DYNAMICS are SIBLINGS of `metropolis_dynamics.advance`
        # and never an extra branch inside it: `advance` is 2912 lines and its
        # acceptance algebra is built for a different proposal, while neither
        # ladder dynamics has any acceptance algebra at all. `sim_setup` refuses
        # both under `run.kind="parallel"` (the channel sampler is serial only),
        # so the nesting below cannot lose a case.
        #
        # THEY DISPATCH TO TWO MODULES AND ONE LOOP. `suwa_todo_dynamics.advance`
        # delegates to `glauber_dynamics.advance`, which reads `engine` and
        # picks the kernel that draws k from the ladder. Dispatching by name
        # here rather than collapsing the two branches keeps this table a
        # complete list of the dynamics this project has -- which is what a
        # reader comes to it for.
        if exe.mode == "parallel":
            from ..engine import cell_parallel

            advance = cell_parallel.advance_parallel
        elif samp.engine == "glauber_dynamics":
            from ..engine import glauber_dynamics

            advance = glauber_dynamics.advance
        elif samp.engine == "suwa_todo_dynamics":
            from ..engine import suwa_todo_dynamics

            advance = suwa_todo_dynamics.advance
        else:
            advance = metropolis_dynamics.advance
        # The species column names (None on a single-species run, which logs N).
        _species_names = (
            None if kern.species is None or kern.n_species <= 1 else kern.species.names
        )
        # U is tracked for Lennard-Jones runs; a hard core has U = 0.
        _pot_spec = config_specs.potential_spec(cfg)
        _lj_run = _pot_spec.potential == "lennard_jones"
        _e_table = potential if isinstance(potential, LennardJonesTable) else None

        def _full_energy() -> float:
            return configuration_energy(
                state.positions,
                kern.box_dims,
                ens.dimension,
                kern.upper_cutoff,
                species=state.species if _e_table is not None else None,
                matrix=ens.matrix_positions,
                table=_e_table,
                tail_pref=_pot_spec.lj_tail_pref,
            )

        if _lj_run:
            state.energy = _full_energy()
        log = io.RunLog(paths.log)
        info = {
            "step": state.step,
            "mu": ens.chem_pot,
            "potential": kern.potential_type,
            "box": ens.box,
        }
        # per-type element symbols for the trajectory (None writes "H")
        _traj_symbols = None if kern.species is None else kern.species.symbol

        # target-mu control (muVT only): drive mu so the running-mean N hits
        # ens.target_count; state.mu and state.n_history are checkpointed.
        targeting = ens.target_count is not None
        if targeting and state.mu is None:
            state.mu = ens.chem_pot

        win = 0
        t_prev = time.perf_counter()
        while state.step < io_.target_cycles:
            stats = advance(state, cfg, potential, weigher)
            if _lj_run:
                assert state.energy is not None and stats["dU"] is not None
                state.energy += stats["dU"]

            if targeting:
                state.n_history.append(stats["N"])
                if len(state.n_history) > ens.control_window:
                    state.n_history = state.n_history[-ens.control_window :]
                avg_n = sum(state.n_history) / len(state.n_history)
                # Robbins-Monro on log N: mu += gamma (ln N_target - ln <N>), with
                # gain gamma0 / (1 + win / tau) so mu settles
                assert ens.target_count is not None and state.mu is not None
                gamma = ens.control_gain / (1.0 + win / ens.control_gain_tau)
                state.mu += gamma * (
                    math.log(ens.target_count) - math.log(max(avg_n, 0.5))
                )

            win += 1
            checkpoint_now = win % io_.checkpoint_every == 0
            if checkpoint_now and _lj_run:
                _check_energy(state, _full_energy(), paths)
            if state.step >= io_.burn_in:
                row: dict[str, Any] = {io.LOG_CYCLE: state.step / 1e6}
                if _species_names is None:
                    row["N"] = stats["N"]
                else:
                    counts = stats["n_by_species"]
                    for name in _species_names:
                        row[f"N_{name}"] = counts[name]
                if _lj_run:
                    row["U"] = state.energy
                if targeting:
                    row["mu"] = state.mu
                row.update(stats["columns"])
                now = time.perf_counter()
                row["wall_s"] = now - t_prev
                t_prev = now
                log.write(row)
            else:
                t_prev = time.perf_counter()
            if checkpoint_now:
                log.flush()
                _save(paths, cfg, state, params)
            if io_.traj_every >= 1 and win % io_.traj_every == 0:
                info["step"] = state.step
                info["mu"] = state.mu if state.mu is not None else ens.chem_pot
                io.write_frame(
                    paths,
                    state.positions,
                    two_dim=kern.two_dim,
                    box=ens.box_sides if ens.box_sides is not None else ens.box,
                    info=info,
                    append=True,
                    species=state.species,
                    symbols=_traj_symbols,
                )

        log.close()
        _save(paths, cfg, state, params)
        info["step"] = state.step
        if io_.traj_every == 0:
            # save-last-frame mode: one overwritten snapshot
            io.write_frame(
                paths,
                state.positions,
                two_dim=kern.two_dim,
                box=ens.box_sides if ens.box_sides is not None else ens.box,
                info=info,
                append=False,
                species=state.species,
                symbols=_traj_symbols,
            )
        elif win % io_.traj_every != 0:
            # append mode: the last window was not on the cadence, so its frame
            # has not been written yet -- append it so the final state is saved
            io.write_frame(
                paths,
                state.positions,
                two_dim=kern.two_dim,
                box=ens.box_sides if ens.box_sides is not None else ens.box,
                info=info,
                append=True,
                species=state.species,
                symbols=_traj_symbols,
            )
        return state, paths
    finally:
        release_lock(paths)

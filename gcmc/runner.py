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

from ._types import RNG, FloatArray

if TYPE_CHECKING:
    from .hash_fingerprint import RunPaths
    from .sim_setup import Config
    from .state import ChainState

from . import MC_engine, geometry, hash_fingerprint, io, sim_setup
from .state import ChainState

# Module logger is `logger`, not `log`: run() binds `log` to the io.Log run-log
# writer, so the getLogger idiom's usual `log` name is already taken here.
logger = logging.getLogger(__name__)

# One fixed width per column, shared by the header and every data row, so the
# labels sit directly above their values and the log reads as a clean table.
_LOG_COLS = [
    ("MC_cycle(M)", 13),
    ("pckng_frctn", 13),
    ("N", 7),
    ("add_acc", 9),
    ("add_att", 9),
    ("add_stp", 9),
    ("rm_acc", 9),
    ("rm_stp", 9),
    ("mv_acc", 9),
    ("mv_stp", 9),
    ("wall_s", 11),
    ("mu", 12),
    ("avgN", 11),
    ("avg_n_cav", 12),
    ("density", 14),
    # CLIQUE COMPOSITION (MC_engine._clq): raw per-window integer SUMS, one
    # triple per direction -- A = add, R = remove. clq*_n counts scored moves,
    # clq*_pos sums the clique sizes, clq*_new sums the positions in those
    # cliques that were NOT a removed member put back. Never log a ratio here:
    # summing these columns over any span of rows and dividing at the end is
    # exact, whereas a per-window fraction would become a mean of means as soon
    # as the reporting interval changed. (Contrast avg_n_cav, two columns up,
    # which IS a running mean of per-window means -- do not copy it.)
    ("clqA_n", 9),
    ("clqA_pos", 10),
    ("clqA_new", 10),
    ("clqR_n", 9),
    ("clqR_pos", 10),
    ("clqR_new", 10),
    # ACCEPTED-only repeats of pos/new (MC_engine._clq "*_acc"). The plain
    # columns above are counted where the acceptance is computed and so include
    # rejections; these four are incremented at the commit. Turnover per
    # ACCEPTED move -- clqA_new_a / clqA_pos_a -- is the quantity that says how
    # much of the configuration a successful move actually replaced, and
    # clq*_new_a / wall_s is the throughput in positions rather than in moves.
    ("clqA_pos_a", 11),
    ("clqA_new_a", 11),
    ("clqR_pos_a", 11),
    ("clqR_new_a", 11),
]
_LOG_WIDTHS = [w for _, w in _LOG_COLS]


def _log_header() -> str:
    """Fixed-width column header for the run log, prefixed with '#' without
    disturbing the column alignment (so numpy.loadtxt-style readers skip it)."""
    h = "".join(name.rjust(w) for name, w in _LOG_COLS)
    return "#" + h[1:]  # mark as a comment without shifting the columns


def _log_row(cells: list[str]) -> str:
    """One fixed-width log row from the per-window cells, aligned to the
    header columns."""
    return "".join(str(c).rjust(w) for c, w in zip(cells, _LOG_WIDTHS))


LOG_HEADER = _log_header()


def _inside_pore_seed(pt: FloatArray, cfg: Config) -> bool:
    """Is a single point inside the hard pore? The centre clearance is the
    configurable sphere radius cfg.radius, not a hardcoded 0.5."""
    L = cfg.box
    cl = cfg.radius
    if cfg.pore_geometry == "slit":
        assert cfg.pore_H is not None  # slit config carries H
        z0 = (L - cfg.pore_H) / 2.0
        return bool(z0 + cl <= pt[2] <= z0 + cfg.pore_H - cl)
    cx = cy = L / 2.0
    import math as _m

    assert cfg.pore_R is not None  # cylinder config carries R
    return bool(_m.hypot(pt[0] - cx, pt[1] - cy) <= cfg.pore_R - cl)


def _save_pore_geometry(cfg: Config, directory: str | Path) -> None:
    """Write the wall geometry as an extxyz of marker points so the pore can be
    loaded next to the fluid trajectory in OVITO. Slit: two wall planes tiled
    with markers; cylinder: the lateral surface tiled in (theta, z). Markers use
    element 'W' (wall) so they are visually distinct from the fluid."""
    import numpy as _np

    L = cfg.box
    pts: list[tuple[float, float, float]] = []
    n = 40
    if cfg.pore_geometry == "slit":
        assert cfg.pore_H is not None  # slit config carries H
        z0 = (L - cfg.pore_H) / 2.0
        xs = _np.linspace(0, L, n)
        ys = _np.linspace(0, L, n)
        for zc in (z0, z0 + cfg.pore_H):  # both wall surfaces
            for x in xs:
                for y in ys:
                    pts.append((x, y, zc))
    else:
        assert cfg.pore_R is not None  # cylinder config carries R
        R = cfg.pore_R
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
            f"Properties=species:S:1:pos:R:3 geometry={cfg.pore_geometry}\n"
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
    if getattr(cfg, "target_n", None) is not None:
        return _nvt_grid(cfg)

    _, eta = _parse_grid_spec(spec)
    dim = cfg.dimension
    box_dims = cfg.box_dims
    volume = float(np.prod(box_dims))
    v_particle = _sphere_volume(cfg.diameter, dim)
    source = "requested"
    if eta is None:
        target_count = getattr(cfg, "target_count", None)
        if target_count is not None and target_count > 0:
            eta = float(target_count) * v_particle / volume
            source = "muVT target density"
        else:
            eta = _GRID_DEFAULT_ETA[dim]
            source = f"default for {dim}D"

    eta_max = v_particle / (cfg.diameter * _GRID_CLEARANCE) ** dim
    if eta > eta_max:
        raise ValueError(
            f"initial_config {spec!r}: packing fraction {eta:.4f} is above the "
            f"densest lattice this bootstrap can build without overlap "
            f"({eta_max:.4f} in {dim}D at diameter {cfg.diameter}). Start lower "
            f"and let GCMC compress, or warm-start from an equilibrated file."
        )

    pts = _grid_sites(box_dims, dim, cfg.diameter, eta)
    n_lattice = len(pts)

    mpos = getattr(cfg, "matrix_positions", None)
    if mpos is not None and len(mpos) > 0:
        # QA: a lattice site inside an obstacle is not a configuration, it is an
        # instant overlap. Drop those sites; the pores keep the rest.
        from scipy.spatial import cKDTree as _KD

        tree_m = _KD(np.asarray(mpos, float) % np.asarray(box_dims), boxsize=box_dims)
        clear = tree_m.query(pts % np.asarray(box_dims), k=1)[0] > (
            cfg.diameter * _GRID_CLEARANCE
        )
        pts = pts[clear]

    if getattr(cfg, "pore_geometry", None) is not None:
        pts = np.array([p for p in pts if _inside_pore_seed(p, cfg)], dtype=float)
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
            cfg.diameter,
        )
    return pts


def _nvt_grid(cfg: Config) -> FloatArray:
    """NVT: EXACTLY target_n non-overlapping particles on a grid. Repacking
    conserves N, so the starting count must be the target; the configuration
    then equilibrates in place from this lattice."""
    n = cfg.target_n
    assert n is not None  # only called when target_n is set
    if n == 0:
        return np.empty((0, cfg.dimension))
    # smallest cubic grid with at least n valid sites, spacing >= diameter
    import math as _m

    per = max(2, int(_m.ceil(n ** (1.0 / cfg.dimension))))
    while True:
        step = cfg.box / per
        if step < cfg.diameter:  # too tight: would overlap
            raise ValueError(
                f"cannot place {n} particles in box {cfg.box} without "
                f"overlap (grid spacing {step:.3f} < diameter "
                f"{cfg.diameter}); reduce N/packing_fraction or enlarge box"
            )
        g = (np.arange(per) + 0.5) * step
        pts = np.array(list(itertools.product(g, repeat=cfg.dimension)))
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
    spec = cfg.initial_config
    # The grid token is resolved first: it is a keyword, so it must never reach
    # the suffix test below (it has no extension and would be mistaken for the
    # equilibrated-library directory, failing with a confusing "not found").
    if spec and _parse_grid_spec(spec)[0]:
        return _grid_config(cfg, spec)
    if not spec:
        if getattr(cfg, "pore_geometry", None) is not None:
            # confined pore: seed a few non-overlapping particles INSIDE the pore
            # so the depth>=1 proposal has an anchor (it cannot grow from N=0).
            from scipy.spatial import cKDTree as _KD

            box = cfg.box
            dim = cfg.dimension
            placed: list[FloatArray] = []
            for _ in range(20 * 300):
                if len(placed) >= 20:
                    break
                pt = rng.uniform(0, box, dim)
                if not _inside_pore_seed(pt, cfg):
                    continue
                if placed:
                    tp = _KD(np.array(placed) % box, boxsize=[box] * dim)
                    if tp.query(pt % box, k=1)[0] < cfg.diameter:
                        continue
                placed.append(pt)
            return np.array(placed) if placed else np.empty((0, dim))
        mpos = getattr(cfg, "matrix_positions", None)
        if (
            mpos is not None
            and len(mpos) > 0
            and getattr(cfg, "target_n", None) is None
        ):
            # quenched-annealed muVT: seed a few valid fluid particles (the
            # depth>=1 proposal needs an anchor and cannot grow from N=0), each
            # clearing the matrix and the others. GCMC then fills the pores.
            from scipy.spatial import cKDTree as _KD

            box = cfg.box
            tree_m = _KD(np.asarray(mpos, float) % box, boxsize=[box] * cfg.dimension)
            placed = []
            target_seed = 20  # small; enough to bootstrap insertion
            for _ in range(target_seed * 200):
                if len(placed) >= target_seed:
                    break
                pt = rng.uniform(0, box, cfg.dimension)
                if tree_m.query(pt % box, k=1)[0] < cfg.diameter:
                    continue  # overlaps matrix
                if placed:
                    tp = _KD(np.array(placed) % box, boxsize=[box] * cfg.dimension)
                    if tp.query(pt % box, k=1)[0] < cfg.diameter:
                        continue  # overlaps an already-placed fluid sphere
                placed.append(pt)
            return np.array(placed) if placed else np.empty((0, cfg.dimension))
        if getattr(cfg, "target_n", None) is not None:
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
        return np.empty((0, cfg.dimension))

    if Path(spec).suffix:  # has an extension -> file
        path = Path(spec).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"initial_config file not found: {path}\n"
                f"  (resolved from cwd={Path.cwd()}). Use an absolute path if the "
                f"run is launched from a different directory than the config."
            )
        pos = io.load_config_file(path, cfg.dimension)
        logger.info("loaded initial config: %d molecules from %s", len(pos), path)
        return pos

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
    is_lj = cfg.potential_type in ("lj", "ljbranch", "ljaux")
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
        if f_mu is None or abs(f_mu - cfg.chem_pot) > 1e-6:
            continue
        if f_box is not None and abs(f_box - float(cfg.box)) > 1e-6:
            continue
        if f_dim is not None and int(f_dim) != cfg.dimension:
            continue
        if is_lj and f_T is not None and abs(f_T - float(cfg.temp)) > 1e-6:
            continue
        matches.append(f)

    if len(matches) == 0:
        sample = sorted(str(Path(f).relative_to(root)) for f in all_xyz)[:12]
        raise FileNotFoundError(
            f"no equilibrated config under {root} matches this run "
            f"(mu={cfg.chem_pot}, box={cfg.box}, dimension={cfg.dimension}"
            + (f", T={cfg.temp}" if is_lj else "")
            + "). "
            f"{len(all_xyz)} .xyz files are present, for example: {sample}. "
            f"Check that a matching mu/box (and T for LJ) exists, or point "
            f"initial_config at the exact file."
        )
    if len(matches) > 1:
        rels = sorted(str(Path(m).relative_to(root)) for m in matches)[:8]
        raise ValueError(
            f"initial_config is ambiguous: {len(matches)} files under {root} match "
            f"mu={cfg.chem_pot}, box={cfg.box}"
            + (f", T={cfg.temp}" if is_lj else "")
            + f": {rels}. Point "
            f"initial_config at a more specific subdirectory or at the exact file."
        )

    chosen_file = matches[0]
    pos = io.load_config_file(chosen_file, cfg.dimension)
    logger.info("loaded initial config: %d molecules from %s", len(pos), chosen_file)
    return pos


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
    if cfg.potential != "hard_sphere":
        return
    sides = cfg.box_sides if cfg.box_sides is not None else (cfg.box,) * cfg.dimension
    geometry.assert_no_overlap(
        pos, cfg.lower_cutoff, np.asarray(sides, dtype=float), what=what
    )


def _save(
    paths: RunPaths, cfg: Config, state: ChainState, params: dict[str, Any]
) -> None:
    """Checkpoint the full chain state (positions, step, rng, counters, live
    mu and its history) through io's atomic write-and-rotate."""
    io.save_checkpoint(
        paths,
        positions=state.positions,
        step=state.step,
        rng=io.rng_state(state.rng),
        counters=state.counters,
        params=params,
        mu=state.mu,
        n_history=state.n_history,
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

    cfg = sim_setup.from_params(params_dict)
    if cfg.matrix_config is not None:
        import dataclasses

        import numpy as _np

        mpath = Path(cfg.matrix_config).resolve()
        if mpath.is_dir():
            raise IsADirectoryError(
                f"run.matrix_config is a folder ({mpath}). A folder of matrix "
                f"realizations is only valid as a SWEEP: give a list target "
                f"(target_N/target_eta/target_density) and launch with "
                f"`python -m gcmc.launcher`, which pairs each swept value with one "
                f"matrix from the folder. For a single run, point matrix_config "
                f"at one trajectory.extxyz / checkpoint.npz file."
            )
        # On a fresh launch we load the external matrix to compute the
        # content-addressed run dir, then snapshot it inside. On a resume the
        # snapshot in the (params-determined) run dir is authoritative, so the
        # external file may be gone -- find the run dir by probing the snapshot.
        if mpath.exists():
            mpos = io.load_matrix(mpath, cfg.dimension)
            cfg = dataclasses.replace(cfg, matrix_positions=mpos)
            paths = hash_fingerprint.run_paths(cfg, root)
            io.ensure_run_dir(paths, params)
            snap = paths.directory / "matrix.npz"
            if snap.exists():
                cfg = dataclasses.replace(
                    cfg, matrix_positions=io.load_matrix(snap, cfg.dimension)
                )
            else:
                _np.savez(snap, positions=_np.asarray(mpos, float))
            logger.info("loaded frozen matrix: %d obstacles from %s", len(mpos), mpath)
        else:
            # external matrix gone: this must be a resume. Find the existing run
            # dir whose params.json matches and load its matrix snapshot.
            found = _find_matrix_snapshot(params_dict, root, cfg.dimension)
            if found is None:
                raise FileNotFoundError(
                    f"matrix_config not found: {mpath}, and no existing run "
                    f"snapshot under {root!r} to resume from. Restore the matrix "
                    f"file or the run directory."
                )
            mpos = io.load_matrix(found, cfg.dimension)
            cfg = dataclasses.replace(cfg, matrix_positions=mpos)
            paths = hash_fingerprint.run_paths(cfg, root)
            io.ensure_run_dir(paths, params)
            logger.info("resumed frozen matrix: %d obstacles from %s", len(mpos), found)
    else:
        paths = hash_fingerprint.run_paths(cfg, root)
        io.ensure_run_dir(paths, params)
        if getattr(cfg, "pore_geometry", None) is not None:
            _save_pore_geometry(cfg, paths.directory)
    if cfg.mode == "parallel":
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
            state = ChainState.fresh(cfg.dimension, seed=seed)
            state.positions = _init_config(cfg, state.rng)
            _assert_start_is_legal(cfg, state.positions, "initial_config")
        else:
            gen = io.set_rng_state(np.random.default_rng(), ckpt["rng"])
            state = ChainState(
                positions=ckpt["positions"],
                step=int(ckpt["step"]),
                rng=gen,
                counters=dict(ckpt["counters"]),
                mu=ckpt.get("mu"),
                n_history=list(ckpt.get("n_history", [])),
            )
            _assert_start_is_legal(cfg, state.positions, "checkpoint")
            io.trim_log(paths.log, state.step / 1e6)
            if cfg.traj_every >= 1:
                io.trim_trajectory(paths.trajectory, state.step)

        potential = cfg.build_potential()
        weigher = MC_engine.make_weigher(cfg, potential)
        if cfg.mode == "parallel":
            from . import parallel

            advance = parallel.advance_parallel
        else:
            advance = MC_engine.advance
        # flush_every is a memory cap; min_interval is the real control, and at
        # its default of 0.0 every row is flushed exactly as before.
        log = io.Log(
            paths.log,
            LOG_HEADER,
            flush_every=1000,
            min_interval=cfg.log_flush_interval,
        )
        info = {
            "step": state.step,
            "mu": cfg.chem_pot,
            "potential": cfg.potential_type,
            "box": cfg.box,
        }

        # target-mu control (muVT only): drive mu so the running-mean N hits
        # cfg.target_count. state.mu carries the live mu (init from cfg on a
        # fresh run; preserved across resume). state.n_history is the rolling
        # window of N feeding the running mean the controller steers.
        targeting = cfg.target_count is not None
        if targeting and state.mu is None:
            state.mu = cfg.chem_pot
        V = cfg.total_volume

        win = 0
        while state.step < cfg.target_cycles:
            _t = time.perf_counter()
            stats = advance(state, cfg, potential, weigher)
            win_wall = time.perf_counter() - _t
            for k, v in stats["counters"].items():
                state.counters[k] += v

            # running mean of N over the control window
            state.n_history.append(stats["N"])
            if len(state.n_history) > cfg.control_window:
                state.n_history = state.n_history[-cfg.control_window :]
            avg_n = sum(state.n_history) / len(state.n_history)

            # running mean of cavity occupancy over the same window (diagnostic)
            state.ncav_history.append(stats.get("n_cav", 0.0))
            if len(state.ncav_history) > cfg.control_window:
                state.ncav_history = state.ncav_history[-cfg.control_window :]
            avg_ncav = sum(state.ncav_history) / len(state.ncav_history)

            if targeting:
                # Robbins-Monro update on log-N: mu += gamma * (lnN_target - lnN).
                # Gain decays as gamma0/(1+win/tau) so mu converges and settles.
                assert cfg.target_count is not None and state.mu is not None
                gamma = cfg.control_gain / (1.0 + win / cfg.control_gain_tau)
                avg_safe = max(avg_n, 0.5)  # guard log at near-empty
                state.mu += gamma * (math.log(cfg.target_count) - math.log(avg_safe))

            cur_mu = state.mu if state.mu is not None else cfg.chem_pot

            if state.step >= cfg.burn_in:
                cc = stats["counters"]
                # advance_parallel does not carry the clique accumulators; a
                # missing block logs as zeros rather than raising, so a parallel
                # run keeps the same column layout.
                clq = stats.get("clique") or dict.fromkeys(
                    ("A_n", "A_pos", "A_new", "R_n", "R_pos", "R_new",
                     "A_pos_acc", "A_new_acc", "R_pos_acc", "R_new_acc"), 0
                )
                log.write(
                    _log_row(
                        [
                            f"{state.step / 1e6:.6f}",
                            f"{stats['packing_fraction']:.8f}",
                            stats["N"],
                            cc["add_accept"],
                            cc["add_attempt"],
                            cc["add_step"],
                            cc["remove_accept"],
                            cc["remove_step"],
                            cc["move_accept"],
                            cc["move_step"],
                            f"{win_wall:.4f}",
                            f"{cur_mu:.6f}",
                            f"{avg_n:.4f}",
                            f"{avg_ncav:.4f}",
                            f"{avg_n / V:.8f}",
                            clq["A_n"],
                            clq["A_pos"],
                            clq["A_new"],
                            clq["R_n"],
                            clq["R_pos"],
                            clq["R_new"],
                            clq.get("A_pos_acc", 0),
                            clq.get("A_new_acc", 0),
                            clq.get("R_pos_acc", 0),
                            clq.get("R_new_acc", 0),
                        ]
                    )
                )

            win += 1
            if win % cfg.checkpoint_every == 0:
                _save(paths, cfg, state, params)
            if cfg.traj_every >= 1 and win % cfg.traj_every == 0:
                info["step"] = state.step
                info["mu"] = cur_mu
                io.write_frame(
                    paths,
                    state.positions,
                    two_dim=cfg.two_dim,
                    box=cfg.box_sides if cfg.box_sides is not None else cfg.box,
                    info=info,
                    append=True,
                )

        log.close()
        _save(paths, cfg, state, params)
        info["step"] = state.step
        if cfg.traj_every == 0:
            # save-last-frame mode: one overwritten snapshot
            io.write_frame(
                paths,
                state.positions,
                two_dim=cfg.two_dim,
                box=cfg.box_sides if cfg.box_sides is not None else cfg.box,
                info=info,
                append=False,
            )
        elif win % cfg.traj_every != 0:
            # append mode: the last window was not on the cadence, so its frame
            # has not been written yet -- append it so the final state is saved
            io.write_frame(
                paths,
                state.positions,
                two_dim=cfg.two_dim,
                box=cfg.box_sides if cfg.box_sides is not None else cfg.box,
                info=info,
                append=True,
            )
        return state, paths
    finally:
        release_lock(paths)

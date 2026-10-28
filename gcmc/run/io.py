"""IO: checkpoint, trajectory, log. One file, nothing clever.

Checkpoint is a single .npz holding the positions plus a small JSON blob with
the step counter, the RNG state, the params, and a schema version. It is written to a temp file and moved into place atomically,
which is atomic on POSIX, and the previous checkpoint is kept as a .bak so a
kill mid-write can never destroy the last good state. Restart reads only the
checkpoint, never the trajectory.

Trajectory is extxyz, written either as a single overwritten snapshot (space
saving, the default) or appended as a full trajectory. Params go into each
frame's info for eyeballing. It is analysis-only and never read on restart.

The run log has one row per window of raw observables and counts (`RunLog`),
appended to on resume; trim_log drops any rows past the checkpoint so the cycle
column stays monotonic after a mid-window kill. `read_log` reads it back.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._types import RNG, FloatArray, IntArray

if TYPE_CHECKING:
    from .hash_fingerprint import RunPaths
from ase import Atoms
from ase.io import write as ase_write

# Checkpoint schema version. Bumped 1 -> 2 when the muVT move selection stopped
# forcing "add" at N=0 (item 1) and the 3D translation drew an independent z
# component instead of a derived one (item 3). Both change how many random values
# a step consumes, so a schema-1 checkpoint replayed under schema 2 continues as a
# valid but different Markov chain; the run id and stored schema make that visible
# rather than silently mixing two streams.
#
# Bumped 2 -> 3 (Phase 4, A4.4) when the checkpoint grew an optional per-particle
# species column. Unlike 1 -> 2 this is a pure format extension, not a change
# to how the main engine consumes randomness: a single-species run writes no
# species array and its stream is unchanged, so a schema-2 checkpoint replays
# bit-identically under schema-3 code and load accepts both. The species stream
# lives in the multi-species MOF loop, which is a separate sampler from the main
# muVT engine; the schema records that the on-disk format can now carry species.
SCHEMA = 3


# ---------------------------------------------------------------- run dir
def ensure_run_dir(paths: RunPaths, params: dict[str, Any]) -> None:
    """Create the run directory and (re)write the params snapshot atomically
    (tmp + atomic replace), so a reader never sees a half-written JSON."""
    paths.directory.mkdir(parents=True, exist_ok=True)
    tmp = paths.params.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(params, indent=2, sort_keys=True))
    tmp.replace(paths.params)


# ---------------------------------------------------------------- RNG state
def rng_state(generator: RNG) -> Any:
    """The generator's full internal state as a JSON-serialisable dict; saving
    and restoring this is what makes a restarted chain bit-identical."""
    return generator.bit_generator.state


def set_rng_state(generator: RNG, state: dict[str, Any]) -> RNG:
    """Restore a state captured by `rng_state` into `generator` (returned)."""
    generator.bit_generator.state = state
    return generator


# ---------------------------------------------------------------- checkpoint
def save_checkpoint(
    paths: RunPaths,
    *,
    positions: FloatArray,
    step: int,
    rng: dict[str, Any],
    params: dict[str, Any],
    mu: float | None = None,
    n_history: list[float] | None = None,
    species: IntArray | None = None,
    schema: int = SCHEMA,
) -> None:
    """Atomic write: positions array plus a JSON metadata blob. Keeps the
    previous checkpoint as .bak. `rng` is the generator state dict. `mu` and
    `n_history` carry the target-mu controller state so a resume continues the
    search rather than restarting it. `species` is the optional per-particle
    type column for a multi-species run; a single-species run leaves it None and
    the file carries no species array, which is what makes a schema-2 checkpoint
    and a schema-3 single-species checkpoint identical in content."""
    meta = json.dumps(
        {
            "schema": int(schema),
            "step": int(step),
            "n": len(positions),
            "rng": rng,
            "params": params,
            "mu": mu,
            "n_history": list(n_history) if n_history is not None else [],
        }
    )
    # the temp name ends in .npz so np.savez never appends a suffix; a stale
    # temp from a killed job (old naming) is deleted, never promoted.
    for stale_name in (".tmp", ".tmp.npz"):
        stale = paths.checkpoint.with_name(paths.checkpoint.name + stale_name)
        stale.unlink(missing_ok=True)
    tmp = paths.checkpoint.with_name(paths.checkpoint.name + ".tmp.npz")
    pos_arr = np.asarray(positions, dtype=float)
    meta_arr = np.array(meta)
    if species is None:
        np.savez(tmp, positions=pos_arr, meta=meta_arr)
    else:
        np.savez(
            tmp,
            positions=pos_arr,
            meta=meta_arr,
            species=np.asarray(species, dtype=np.int64),
        )
    if paths.checkpoint.exists():
        paths.checkpoint.replace(paths.checkpoint_bak)
    tmp.replace(paths.checkpoint)


def _read_npz(path: str | Path) -> dict[str, Any]:
    """Decode one checkpoint file: positions as a real array, everything else
    from the embedded JSON `meta` blob (allow_pickle stays False so a
    checkpoint can never execute code on load)."""
    with np.load(path, allow_pickle=False) as data:
        positions = data["positions"]
        meta = json.loads(str(data["meta"]))
        # schema-2 checkpoints have no species array; absence reads as None, i.e.
        # single-species. allow_pickle stays False, so this is a plain int array.
        species = data["species"] if "species" in data.files else None
    return {
        "positions": positions,
        "step": meta["step"],
        "n": meta["n"],
        "rng": meta["rng"],
        "params": meta["params"],
        "schema": meta["schema"],
        "mu": meta.get("mu"),
        "n_history": meta.get("n_history", []),
        "species": species,
    }


def load_checkpoint(paths: RunPaths) -> dict[str, Any] | None:
    """Return the checkpoint dict, falling back to the .bak if the primary is
    missing or unreadable, or None if neither is usable."""
    for path in (paths.checkpoint, paths.checkpoint_bak):
        if path.exists():
            try:
                return _read_npz(path)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------- trajectory
def write_frame(
    paths: RunPaths,
    positions: FloatArray,
    *,
    two_dim: bool,
    box: float | Sequence[float],
    info: dict[str, Any],
    append: bool,
    species: IntArray | None = None,
    symbols: Sequence[str] | None = None,
) -> None:
    """Write one configuration to the extxyz trajectory.

    append=False overwrites the file, leaving a single snapshot (the
    save-last-frame mode). append=True adds the frame to a growing trajectory.
    The caller (runner) decides which from the single `traj_every` knob:
    traj_every == 0 -> overwrite (last frame only); traj_every >= 1 -> append.

    SPECIES. With `species` and `symbols` given, each particle is written
    under its own element symbol from the species file, so a frame is a valid
    warm start for the same model: without it a typed trajectory cannot be read
    back into species.

    Left at "H" when `species` is None, which is the single-species case. That is
    not laziness: `tests/engine/test_stageA_traj_identity.py` compares trajectory
    files against frozen references, and a symbol change would move every one of
    them for no gain -- a one-species run has nothing to disambiguate.
    """
    pos = np.asarray(positions, dtype=float)
    if two_dim:
        pos = np.column_stack([pos, np.zeros(len(pos))])
    # box is a scalar for a cubic cell or a per-axis sequence for an orthorhombic
    # one. ASE always wants three axis lengths, so 2D pads a third.
    #
    # In 2D that third axis is ZERO and its boundary is OPEN (pbc "T T F"). The
    # padded axis is not a physical direction: there is no periodicity along it
    # and no length to be periodic over, so writing a real side there (this used
    # to repeat the x side "so the cell stays non-degenerate") describes a cubic
    # 3D box that was never simulated. Any viewer or analysis tool reading the
    # header would wrap along z, compute a volume from it, or count spurious
    # periodic images. A zero vector with pbc False is the honest statement, and
    # it is what extxyz uses for a genuinely 2D system.
    #
    # Safe for the readers: both load paths take get_positions()[:, :dimension]
    # and never consult the cell or pbc, so the equilibrated warm-start library
    # keeps loading either way.
    b = np.atleast_1d(np.asarray(box, dtype=float))
    pbc: bool | tuple[bool, bool, bool] = True
    if two_dim:
        y = float(b[1]) if b.size > 1 else float(b[0])
        cell = [float(b[0]), y, 0.0]
        pbc = (True, True, False)
    elif b.size == 1:
        cell = [float(b[0])] * 3
    else:
        cell = [float(x) for x in b]
    atom_symbols: str | list[str]
    if species is None:
        atom_symbols = "H" * len(pos)
    else:
        if symbols is None:
            raise ValueError(
                "write_frame was given a species column but no symbol table; "
                "the type ids alone cannot be written as element symbols"
            )
        col = np.asarray(species, dtype=np.int64).reshape(-1)
        if len(col) != len(pos):
            raise ValueError(
                f"write_frame: {len(col)} species labels for {len(pos)} "
                "positions. The two must come from the same slot order or the "
                "frame mislabels every particle."
            )
        if len(col) and (col.min() < 0 or col.max() >= len(symbols)):
            raise ValueError(
                f"write_frame: species ids span [{col.min()}, {col.max()}] but "
                f"the symbol table has {len(symbols)} entries"
            )
        atom_symbols = [symbols[int(t)] for t in col]
    atoms = Atoms(atom_symbols, positions=pos, cell=cell, pbc=pbc)
    for k, v in info.items():
        atoms.info[k] = v
    ase_write(paths.trajectory, atoms, format="extxyz", append=append)


# ---------------------------------------------------------------- log
#: First column of every run log: the cumulative MC step in millions.
LOG_CYCLE = "MC_cycle"


def _cell(value: Any) -> str:
    """One log cell: integers as integers, floats to ten significant digits."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return format(float(value), ".10g")


#: Longest time a written row waits before the log is flushed.
FLUSH_SECONDS = 1.0


class RunLog:
    """The run log: a `#` header naming the columns, then one whitespace
    separated row per window.

    Rows reach the disk at most once per `FLUSH_SECONDS`, at every checkpoint
    (the caller's `flush()`) and on `close()`, so a one-step window does not pay
    a filesystem write per row. A killed run loses only rows past its last
    checkpoint, which a resume drops and rewrites anyway.

    The header is written with the first row, so the columns are whatever the
    first window reports, in that order. Reopening an existing log (a resumed
    run) reads its header, and a row whose columns differ is refused rather than
    appended under the wrong names.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cols: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size:
            self._cols = log_columns(self.path)
            if not self._cols or self._cols[0] != LOG_CYCLE:
                raise ValueError(
                    f"{self.path} is a run log in an older format (its first "
                    f"column is not {LOG_CYCLE}). Finish that run with the code "
                    "that started it, or delete the run folder to start afresh."
                )
        self._f = open(self.path, "a")  # noqa: SIM115  (closed by close())
        self._last_flush = time.monotonic()

    def write(self, row: dict[str, Any]) -> None:
        """Append one row; the first row of a new file writes the header."""
        cols = list(row)
        widths = [max(len(c) + 1, 12) for c in cols]
        if self._cols is None:
            self._cols = cols
            head = " ".join(c.rjust(w) for c, w in zip(cols, widths))
            self._f.write("#" + head[1:] + "\n")
        elif cols != self._cols:
            raise ValueError(
                f"{self.path} has columns {self._cols} and this window produced "
                f"{cols}; the run's configuration no longer matches its log"
            )
        cells = (_cell(v).rjust(w) for v, w in zip(row.values(), widths))
        self._f.write(" ".join(cells) + "\n")
        if time.monotonic() - self._last_flush >= FLUSH_SECONDS:
            self.flush()

    def flush(self) -> None:
        """Push every written row to the file."""
        self._f.flush()
        self._last_flush = time.monotonic()

    def close(self) -> None:
        """Flush and release the file handle."""
        self._f.close()


def log_columns(path: str | Path) -> list[str]:
    """The column names of a run log (its last `#` line before the data)."""
    names: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                names = line[1:].split()
            elif line.strip():
                break
    return names


def read_log(path: str | Path) -> dict[str, np.ndarray]:
    """A run log as {column: array}, one entry per window."""
    names = log_columns(path)
    if not names:
        raise ValueError(f"{path}: no '#' header line naming the columns")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size == 0:
        return {n: np.empty(0) for n in names}
    if data.shape[1] != len(names):
        raise ValueError(
            f"{path}: the header names {len(names)} columns and the rows have "
            f"{data.shape[1]}"
        )
    return {n: data[:, i] for i, n in enumerate(names)}


def trim_trajectory(path: str | Path, max_step: int) -> None:
    """Drop trajectory frames whose stored step exceeds the checkpoint step, so
    a resumed append-mode run does not duplicate frames written after the last
    checkpoint but before a kill. No-op if the file is absent. Frames are extxyz
    blocks: a count line, a comment line carrying `step=<int>`, then that many
    atom lines; whole frames with step <= max_step are kept.

    The copy is streamed frame by frame into a temporary file rather than reading
    the whole trajectory into memory, so trimming a multi-gigabyte run costs one
    frame of memory instead of the entire file."""
    path = Path(path)
    if not path.exists():
        return
    import re

    tmp = path.with_name(path.name + ".tmp")
    with open(path) as fin, open(tmp, "w") as fout:
        while True:
            header = fin.readline()
            if header == "":
                break  # EOF
            try:
                natoms = int(header.strip())
            except ValueError:
                fout.write(header)  # stray line: pass it through
                continue
            comment = fin.readline()
            atoms = [fin.readline() for _ in range(natoms)]
            m = re.search(r"step=(\d+)", comment)
            step = int(m.group(1)) if m else 0
            if step <= max_step:
                fout.write(header)
                fout.write(comment)
                fout.writelines(atoms)
    tmp.replace(path)


def trim_log(path: str | Path, max_cycle_millions: float) -> None:
    """Drop data rows whose leading MC_cycle value (in millions) exceeds the
    checkpoint, keeping the header. No-op if the file is absent."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text().splitlines(keepends=True)
    kept = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            kept.append(ln)
            continue
        try:
            cyc = float(s.split()[0])
        except (ValueError, IndexError):
            kept.append(ln)
            continue
        if cyc <= max_cycle_millions * (1.0 + 1e-12):
            kept.append(ln)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(kept))
    tmp.replace(path)


def _coerce_positions(
    pos: Any, dimension: int, n_expected: int | None, path: str | Path
) -> FloatArray:
    """Turn whatever a checkpoint stored under `positions` into a clean
    (N, dimension) array, or raise loudly. Handles the layouts seen in the wild:
      - (N, dim)               : the canonical form, used as-is
      - (dim, N)               : transposed -> transpose back
      - (1, N*dim) / (N*dim,)  : flattened -> reshape to (-1, dim)
    `n_expected` (from meta['n'], if present) disambiguates. We never silently
    return a 1-row array from a flattened blob: that is the bug that made a
    415-sphere matrix load as a single obstacle and run as bulk GCMC."""
    pos = np.asarray(pos, dtype=float)
    pos = np.squeeze(pos)
    # flat vector of all coordinates -> (N, dim)
    if pos.ndim == 1:
        if pos.size % dimension != 0:
            raise ValueError(
                f"{path}: positions is a flat array of length {pos.size}, not "
                f"divisible by dimension {dimension}; cannot reshape to coords"
            )
        pos = pos.reshape(-1, dimension)
    elif pos.ndim == 2:
        r, c = pos.shape
        if c == dimension:
            pass  # (N, dim) already
        elif r == dimension and c != dimension:
            pos = pos.T  # (dim, N) -> (N, dim)
        elif r == 1 and c % dimension == 0:
            pos = pos.reshape(-1, dimension)  # (1, N*dim) flattened
        elif c % dimension == 0 and r == 1:
            pos = pos.reshape(-1, dimension)
        elif c > dimension:
            pos = pos[:, :dimension]  # extra columns (e.g. 2D pad)
        else:
            raise ValueError(
                f"{path}: positions has shape {(r, c)}, cannot interpret as "
                f"(N, {dimension}) coordinates"
            )
    else:
        raise ValueError(
            f"{path}: positions has {pos.ndim} dimensions; "
            f"expected a 1- or 2-D coordinate array"
        )
    if n_expected is not None and len(pos) != n_expected:
        raise ValueError(
            f"{path}: loaded {len(pos)} positions but the checkpoint meta says "
            f"n={n_expected}. The file layout is not understood; refusing to run "
            f"a quenched-annealed simulation with the wrong matrix."
        )
    if len(pos) < 2:
        raise ValueError(
            f"{path}: matrix loaded as {len(pos)} obstacle(s). A frozen matrix "
            f"needs many spheres; a single point means the file layout was "
            f"misread. Check that the matrix checkpoint stores positions as "
            f"an (N, {dimension}) array."
        )
    return pos


def load_matrix(path: str | Path, dimension: int) -> FloatArray:
    """Load a frozen matrix configuration for a quenched-annealed run. Accepts
    whatever an NVT matrix-generation run produced:
      - .npz   : a checkpoint (reads the `positions` array)
      - .extxyz/.xyz via ASE if the header looks like extxyz, else plain floats
    Returns an (N, dimension) array. The file is authoritative in-box coords."""
    ext = Path(path).suffix.lower()
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            pos = np.asarray(data["positions"], dtype=float)
            # the expected number of particles, if the checkpoint recorded it
            n_expected = None
            if "meta" in data:
                try:
                    n_expected = int(json.loads(str(data["meta"])).get("n"))
                except Exception:
                    n_expected = None
        pos = _coerce_positions(pos, dimension, n_expected, path)
        return np.ascontiguousarray(pos)
    if ext == ".extxyz":
        from ase.io import read as ase_read

        # take the LAST frame only (index="-1"): a matrix trajectory may have
        # many frames but the frozen matrix is the final equilibrated config.
        atoms = ase_read(path, index="-1", format="extxyz")
        assert not isinstance(atoms, list)  # index="-1" yields a single frame
        return np.ascontiguousarray(
            np.asarray(atoms.get_positions(), dtype=np.float64)[:, :dimension]
        )
    # plain coordinate file (x y z per line), same as load_config_file
    return load_config_file(path, dimension)


def load_config_species(
    path: str | Path,
    *,
    symbols: Sequence[str],
    n_expected: int,
) -> IntArray:
    """Read the species column of a warm-start frame back into type ids.

    THE OTHER HALF OF `write_frame`'s symbols. A reader that takes
    `get_positions()` and discards `get_chemical_symbols()` restores the
    coordinates of a mixture and none of its composition, which is worse than
    failing: the run continues as a pure species-0 system at the density of a
    mixture.

    `symbols` is the species file's own per-type symbol tuple, so the mapping
    that reads a frame is the mapping that wrote it.

    THREE THINGS RAISE RATHER THAN GUESSING, and each is a silent wrong answer
    otherwise:

      * a symbol not in the file. Naming it and listing the known ones, never a
        silent type 0 -- a typo in one row would otherwise move one particle to
        the wrong species and nothing would say so.
      * a bare coordinate table (no symbols at all). Composition cannot be
        inferred from coordinates, so "all species 0" is a guess dressed as a
        default.
      * a count that disagrees with the positions already loaded. The two are
        read from the same file, so a mismatch means the file changed underneath
        or the parse took different frames -- either way the labels would not
        belong to the coordinates.

    Duplicate symbols cannot reach here: `species_file` refuses a file where two
    types share one, with the reason being exactly this round trip.
    """
    path = Path(path)
    table = {str(sym): i for i, sym in enumerate(symbols)}
    if len(table) != len(symbols):
        raise ValueError(
            f"the symbol table has duplicates ({list(symbols)}), so a frame "
            "cannot be read back into types unambiguously"
        )
    try:
        from ase.io import read as ase_read

        atoms = ase_read(str(path), index="-1")
    except Exception as exc:
        raise ValueError(
            f"{path}: could not be parsed as (extended) XYZ, so it carries no "
            f"species column ({exc}). A typed run cannot warm-start from a bare "
            f"coordinate table: composition is not recoverable from positions, "
            f"and defaulting every particle to {symbols[0]!r} would be a guess. "
            f"Use a trajectory frame written by this code."
        ) from None
    assert not isinstance(atoms, list)  # a single frame was requested
    found = [str(x) for x in atoms.get_chemical_symbols()]
    if len(found) != n_expected:
        raise ValueError(
            f"{path}: {len(found)} symbols for {n_expected} positions loaded "
            "from the same file; the labels would not belong to the coordinates"
        )
    unknown = sorted({sym for sym in found if sym not in table})
    if unknown:
        raise ValueError(
            f"{path}: symbol(s) {unknown} are not species of this model. Known "
            f"symbols: {sorted(table)}. A frame written for a different force "
            f"field is not a warm start for this one, and mapping the unknown "
            f"ones to type 0 would silently change the composition."
        )
    return np.asarray([table[sym] for sym in found], dtype=np.int64)


def load_config_file(path: str | Path, dimension: int) -> FloatArray:
    """Load an equilibrated starting configuration. Two formats are accepted and
    chosen automatically from the first line:

      (1) Extended XYZ, the format the equilibrated library is stored in: the
          first line is the atom count, the second is a comment carrying
          `Lattice=...` and `Properties=species:S:1:pos:R:3`, then one
          `species x y z` row per atom. Read through ASE (last frame if several).

      (2) A bare coordinate table: whitespace-separated floats `x y z`, one
          molecule per line (z present and 0 for 2D). Read with numpy.

    Returns an (N, dimension) array using the first `dimension` spatial columns.
    The file is taken as authoritative in-box coordinates (nothing is wrapped)."""
    with open(path) as f:
        first = f.readline().strip()
    is_xyz = False
    if first:
        try:
            int(first)  # an atom-count first line marks (extended) XYZ
            is_xyz = True
        except ValueError:
            is_xyz = False
    if is_xyz:
        try:
            from ase.io import read as ase_read

            atoms = ase_read(path, index="-1")  # ASE picks xyz/extxyz by header
        except Exception as e:
            raise ValueError(  # noqa: B904 -- replaced two lines below
                f"{path}: first line is an atom count so this looks like an XYZ "
                f"file, but it could not be parsed as XYZ/extended-XYZ ({e}). "
                f"Expected: an integer count, a comment line, then rows of "
                f"`species x y z`."
            )
        assert not isinstance(atoms, list)  # a single frame was requested
        pos = np.asarray(atoms.get_positions(), dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] < dimension:
            raise ValueError(
                f"{path}: parsed {pos.shape} positions but need at least "
                f"{dimension} spatial columns for a {dimension}D run"
            )
        return np.ascontiguousarray(pos[:, :dimension])

    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:  # a single line -> one molecule
        data = data.reshape(1, -1)
    if data.shape[1] < dimension:
        raise ValueError(
            f"{path}: has {data.shape[1]} numeric columns, need {dimension} for a "
            f"{dimension}D run. If this is an XYZ file, its first line must be the "
            f"atom count so it is read as XYZ rather than a bare number table."
        )
    return np.ascontiguousarray(data[:, :dimension])

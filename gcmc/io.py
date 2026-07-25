"""IO: checkpoint, trajectory, log. One file, nothing clever.

Checkpoint is a single .npz holding the positions plus a small JSON blob with
the step counter, the move counters, the RNG state, the params, and a schema
version. It is written to a temp file and moved into place atomically,
which is atomic on POSIX, and the previous checkpoint is kept as a .bak so a
kill mid-write can never destroy the last good state. Restart reads only the
checkpoint, never the trajectory.

Trajectory is extxyz, written either as a single overwritten snapshot (space
saving, the default) or appended as a full trajectory. Params go into each
frame's info for eyeballing. It is analysis-only and never read on restart.

Log is the plain counters table, opened in append so a resumed run continues
its rows; trim_log drops any rows past the checkpoint so the cycle column stays
monotonic after a mid-window kill.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ._types import RNG, FloatArray, IntArray

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
# species column (D2). Unlike 1 -> 2 this is a pure format extension, not a change
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
    counters: dict[str, int],
    params: dict[str, Any],
    mu: float | None = None,
    n_history: list[float] | None = None,
    species: IntArray | None = None,
    schema: int = SCHEMA,
) -> None:
    """Atomic write: positions array plus a JSON metadata blob. Keeps the
    previous checkpoint as .bak. `rng` is the generator state dict. `mu` and
    `n_history` carry the target-mu controller state so a resume continues the
    search rather than restarting it. `species` (D2) is the optional per-particle
    type column for a multi-species run; a single-species run leaves it None and
    the file carries no species array, which is what makes a schema-2 checkpoint
    and a schema-3 single-species checkpoint identical in content."""
    meta = json.dumps(
        {
            "schema": int(schema),
            "step": int(step),
            "n": len(positions),
            "counters": counters,
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
        "counters": meta["counters"],
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
) -> None:
    """Write one configuration to the extxyz trajectory.

    append=False overwrites the file, leaving a single snapshot (the
    save-last-frame mode). append=True adds the frame to a growing trajectory.
    The caller (runner) decides which from the single `traj_every` knob:
    traj_every == 0 -> overwrite (last frame only); traj_every >= 1 -> append.
    """
    pos = np.asarray(positions, dtype=float)
    if two_dim:
        pos = np.column_stack([pos, np.zeros(len(pos))])
    # box is a scalar for a cubic cell (unchanged) or a per-axis sequence for an
    # orthorhombic cell (Phase 2). ASE wants three axis lengths; in 2D the padded
    # z axis reuses the first side as a dummy so the cell stays non-degenerate.
    b = np.atleast_1d(np.asarray(box, dtype=float))
    if b.size == 1:
        cell = [float(b[0])] * 3
    elif two_dim:
        cell = [float(b[0]), float(b[1]), float(b[0])]
    else:
        cell = [float(x) for x in b]
    atoms = Atoms("H" * len(pos), positions=pos, cell=cell, pbc=True)
    for k, v in info.items():
        atoms.info[k] = v
    ase_write(paths.trajectory, atoms, format="extxyz", append=append)


# ---------------------------------------------------------------- log
class Log:
    """Append-only CSV log with buffered writes: the header is written once
    per file lifetime, rows are flushed every `flush_every` writes so an HPC
    kill loses at most one buffer, and reopening appends (a resumed run
    continues the same file)."""

    def __init__(self, path: str | Path, header: str, flush_every: int = 1) -> None:
        """Open (or reopen) `path` for appending; write `header` only if the
        file is new or empty."""
        self.path = Path(path)
        self.flush_every = flush_every
        self._buf: list[str] = []
        new = (not self.path.exists()) or self.path.stat().st_size == 0
        self._f = open(path, "a")  # noqa: SIM115  (held open; closed by close())
        if new:
            self._f.write(header if header.endswith("\n") else header + "\n")
            self._f.flush()

    def write(self, line: str) -> None:
        """Buffer one row (newline added if missing); flush on threshold."""
        self._buf.append(line if line.endswith("\n") else line + "\n")
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write the buffer through to disk (fsync-free; the OS flush is
        enough for log data whose loss is tolerable)."""
        if self._buf:
            self._f.writelines(self._buf)
            self._f.flush()
            self._buf.clear()

    def close(self) -> None:
        """Flush and release the file handle."""
        self.flush()
        self._f.close()


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
    """Drop data rows whose leading MC-cycle value (in millions) exceeds the
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
        if cyc <= max_cycle_millions:
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

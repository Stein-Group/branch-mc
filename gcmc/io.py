"""IO: checkpoint, trajectory, log. One file, nothing clever.

Checkpoint is a single .npz holding the positions plus a small JSON blob with
the step counter, the move counters, the RNG state, the params, and a schema
version. It is written to a temp file and moved into place with os.replace,
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
import os
import numpy as np
from ase import Atoms
from ase.io import write as ase_write

SCHEMA = 1


# ---------------------------------------------------------------- run dir
def ensure_run_dir(paths, params):
    os.makedirs(paths.directory, exist_ok=True)
    tmp = paths.params + ".tmp"
    with open(tmp, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    os.replace(tmp, paths.params)


# ---------------------------------------------------------------- RNG state
def rng_state(generator):
    return generator.bit_generator.state


def set_rng_state(generator, state):
    generator.bit_generator.state = state
    return generator


# ---------------------------------------------------------------- checkpoint
def save_checkpoint(paths, *, positions, step, rng, counters, params, schema=SCHEMA):
    """Atomic write: positions array plus a JSON metadata blob. Keeps the
    previous checkpoint as .bak. `rng` is the generator state dict."""
    meta = json.dumps({
        "schema": int(schema),
        "step": int(step),
        "n": int(len(positions)),
        "counters": counters,
        "rng": rng,
        "params": params,
    })
    # the temp name ends in .npz so np.savez never appends a suffix; a stale
    # temp from a killed job (old naming) is deleted, never promoted.
    for stale in (paths.checkpoint + ".tmp", paths.checkpoint + ".tmp.npz"):
        if os.path.exists(stale):
            os.remove(stale)
    tmp = paths.checkpoint + ".tmp.npz"
    np.savez(tmp, positions=np.asarray(positions, dtype=float),
             meta=np.array(meta))
    if os.path.exists(paths.checkpoint):
        os.replace(paths.checkpoint, paths.checkpoint_bak)
    os.replace(tmp, paths.checkpoint)


def _read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        positions = data["positions"]
        meta = json.loads(str(data["meta"]))
    return {"positions": positions, "step": meta["step"], "n": meta["n"],
            "counters": meta["counters"], "rng": meta["rng"], "params": meta["params"],
            "schema": meta["schema"]}


def load_checkpoint(paths):
    """Return the checkpoint dict, falling back to the .bak if the primary is
    missing or unreadable, or None if neither is usable."""
    for path in (paths.checkpoint, paths.checkpoint_bak):
        if os.path.exists(path):
            try:
                return _read_npz(path)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------- trajectory
def write_frame(paths, positions, *, two_dim, box, info, append):
    """Write one configuration to the extxyz trajectory.

    append=False overwrites the file, leaving a single snapshot (the
    save-last-frame mode). append=True adds the frame to a growing trajectory.
    The caller (runner) decides which from the single `traj_every` knob:
    traj_every == 0 -> overwrite (last frame only); traj_every >= 1 -> append.
    """
    pos = np.asarray(positions, dtype=float)
    if two_dim:
        pos = np.column_stack([pos, np.zeros(len(pos))])
    atoms = Atoms("H" * len(pos), positions=pos, cell=[box] * 3, pbc=True)
    for k, v in info.items():
        atoms.info[k] = v
    ase_write(paths.trajectory, atoms, format="extxyz", append=append)


# ---------------------------------------------------------------- log
class Log:
    def __init__(self, path, header, flush_every=1):
        self.path = path
        self.flush_every = flush_every
        self._buf = []
        new = (not os.path.exists(path)) or os.path.getsize(path) == 0
        self._f = open(path, "a")
        if new:
            self._f.write(header if header.endswith("\n") else header + "\n")
            self._f.flush()

    def write(self, line):
        self._buf.append(line if line.endswith("\n") else line + "\n")
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if self._buf:
            self._f.writelines(self._buf)
            self._f.flush()
            self._buf.clear()

    def close(self):
        self.flush()
        self._f.close()


def trim_trajectory(path, max_step):
    """Drop trajectory frames whose stored step exceeds the checkpoint step, so
    a resumed append-mode run does not duplicate frames written after the last
    checkpoint but before a kill. No-op if the file is absent. Frames are extxyz
    blocks: a count line, a comment line carrying `step=<int>`, then that many
    atom lines. We keep whole frames with step <= max_step."""
    if not os.path.exists(path):
        return
    import re
    with open(path) as f:
        lines = f.readlines()
    kept = []
    i = 0
    n = len(lines)
    while i < n:
        try:
            natoms = int(lines[i].strip())
        except (ValueError, IndexError):
            # not a frame header; keep the stray line and advance
            kept.append(lines[i]); i += 1
            continue
        block = lines[i:i + natoms + 2]          # header + comment + atoms
        m = re.search(r"step=(\d+)", block[1]) if len(block) > 1 else None
        step = int(m.group(1)) if m else 0
        if step <= max_step:
            kept.extend(block)
        i += natoms + 2
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(kept)
    os.replace(tmp, path)


def trim_log(path, max_cycle_millions):
    """Drop data rows whose leading MC-cycle value (in millions) exceeds the
    checkpoint, keeping the header. No-op if the file is absent."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.readlines()
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
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(kept)
    os.replace(tmp, path)


def _coerce_positions(pos, dimension, n_expected, path):
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
                f"divisible by dimension {dimension}; cannot reshape to coords")
        pos = pos.reshape(-1, dimension)
    elif pos.ndim == 2:
        r, c = pos.shape
        if c == dimension:
            pass                                   # (N, dim) already
        elif r == dimension and c != dimension:
            pos = pos.T                            # (dim, N) -> (N, dim)
        elif r == 1 and c % dimension == 0:
            pos = pos.reshape(-1, dimension)       # (1, N*dim) flattened
        elif c % dimension == 0 and r == 1:
            pos = pos.reshape(-1, dimension)
        elif c > dimension:
            pos = pos[:, :dimension]               # extra columns (e.g. 2D pad)
        else:
            raise ValueError(
                f"{path}: positions has shape {(r, c)}, cannot interpret as "
                f"(N, {dimension}) coordinates")
    else:
        raise ValueError(f"{path}: positions has {pos.ndim} dimensions; "
                         f"expected a 1- or 2-D coordinate array")
    if n_expected is not None and len(pos) != n_expected:
        raise ValueError(
            f"{path}: loaded {len(pos)} positions but the checkpoint meta says "
            f"n={n_expected}. The file layout is not understood; refusing to run "
            f"a quenched-annealed simulation with the wrong matrix.")
    if len(pos) < 2:
        raise ValueError(
            f"{path}: matrix loaded as {len(pos)} obstacle(s). A frozen matrix "
            f"needs many spheres; a single point means the file layout was "
            f"misread. Check that the matrix checkpoint stores positions as "
            f"an (N, {dimension}) array.")
    return pos


def load_matrix(path, dimension):
    """Load a frozen matrix configuration for a quenched-annealed run. Accepts
    whatever an NVT matrix-generation run produced:
      - .npz   : a checkpoint (reads the `positions` array)
      - .extxyz/.xyz via ASE if the header looks like extxyz, else plain floats
    Returns an (N, dimension) array. The file is authoritative in-box coords."""
    import os
    ext = os.path.splitext(path)[1].lower()
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
        atoms = ase_read(path, format="extxyz")
        return np.ascontiguousarray(np.asarray(atoms.get_positions(), float)[:, :dimension])
    # plain coordinate file (x y z per line), same as load_config_file
    return load_config_file(path, dimension)


def load_config_file(path, dimension):
    """Load an equilibrated starting configuration from a plain coordinate file:
    one molecule per line, whitespace-separated floats `x y z` (z present and 0
    for 2D systems). Returns an (N, dimension) array, using the first
    `dimension` columns and wrapping nothing (the file is taken as authoritative
    in-box coordinates)."""
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:                       # a single line -> one molecule
        data = data.reshape(1, -1)
    if data.shape[1] < dimension:
        raise ValueError(f"{path}: has {data.shape[1]} columns, need {dimension}")
    return np.ascontiguousarray(data[:, :dimension])

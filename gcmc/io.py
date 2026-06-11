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

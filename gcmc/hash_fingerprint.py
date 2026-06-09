"""Run identity and on-disk layout.

Saving simulation identity params in filename or log/traj header is fragile.
Files/experiment name can be renamed/regex fails
When saved in header, abrupt timeout/job cancellation can corroupt log/traj files, 
especially if they are written frequently.

Robust solution is to define each run with a hash of simulation identity params.

A run is addressed by a deterministic hash of the parameters that make it a
distinct experiment, never by parsing a name. Identity-bearing parameters are
the physics and the algorithm: potential type and its physics constants, mu,
temperature, box, dimension, cavity radius, trial density, depth, the BRANCH and
use-depth switches, and the move factor. Execution controls (target length,
worker count, IO cadence, parallel mode) are excluded, which is what lets a
finished run resume as a no-op and a longer target reuse the same directory.

The g3 table path is treated as data, not a physics parameter, so it is left out
of the hash; pointing at a different table is the caller's responsibility.

Each run lives in its own directory `runs/<id>__mu<mu>/`, where the id is the
hash and the mu tag is only there for eyeballing.
"""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os

_IDENTITY_FIELDS = (
    "potential_type", "sampler", "dimension", "chem_pot", "box", "cavity_radius",
    "trial_density", "depth", "branch", "use_depth", "move_factor",
    "temp", "rc", "eta", "gamma_max", "delta_eta",
)


def _canon(value):
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def identity_dict(cfg):
    return {k: _canon(getattr(cfg, k)) for k in _IDENTITY_FIELDS}


def run_id(cfg):
    blob = json.dumps(identity_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class RunPaths:
    directory: str
    checkpoint: str
    checkpoint_bak: str
    trajectory: str
    log: str
    params: str
    lock: str


def run_paths(cfg, root="runs"):
    rid = run_id(cfg)
    mu_tag = format(cfg.chem_pot, ".4g").replace("-", "m").replace(".", "p")
    d = os.path.join(root, f"{rid}__mu{mu_tag}")
    return RunPaths(
        directory=d,
        checkpoint=os.path.join(d, "checkpoint.npz"),
        checkpoint_bak=os.path.join(d, "checkpoint.bak.npz"),
        trajectory=os.path.join(d, "trajectory.extxyz"),
        log=os.path.join(d, "run.log"),
        params=os.path.join(d, "params.json"),
        lock=os.path.join(d, "lock"),
    )

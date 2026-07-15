"""Run identity and on-disk layout.

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

# upper_cutoff and no_neigh are physics-bearing (interaction range, energy
# truncation), so they are identity. Adding them changes every run id relative
# to the pre-fix layout; runs produced by the biased engine should not be
# resumed anyway.
_IDENTITY_FIELDS = (
    "potential_type", "sampler", "dimension", "chem_pot", "box", "cavity_radius",
    "trial_density", "depth", "branch", "use_depth", "move_factor",
    "temp", "rc", "eta", "gamma_max", "delta_eta", "lower_cutoff",
    "upper_cutoff", "no_neigh", "ensemble_type", "target_n",
    "pore_geometry", "pore_H", "pore_R",
    "channel", "channel_length", "channel_width",
)


def _canon(value):
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def identity_dict(cfg):
    # channel geometry only identifies a run when the channel sampler is active;
    # for every other mode these keys are omitted so the fingerprint is unchanged
    # from before the channel sampler existed.
    active_channel = getattr(cfg, "channel", False)
    skip = () if active_channel else ("channel", "channel_length", "channel_width")
    return {k: _canon(getattr(cfg, k)) for k in _IDENTITY_FIELDS if k not in skip}


def _matrix_digest(cfg):
    mpos = getattr(cfg, "matrix_positions", None)
    if mpos is None or len(mpos) == 0:
        return ""
    import numpy as np
    # round to a stable precision so floating noise in I/O does not change the id
    rounded = np.round(np.asarray(mpos, float), 6)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:12]


def run_id(cfg):
    ident = identity_dict(cfg)
    mdig = _matrix_digest(cfg)
    if mdig:
        ident["matrix"] = mdig
    blob = json.dumps(ident, sort_keys=True, separators=(",", ":"))
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
    def _fmt(x):
        return format(x, ".4g").replace("-", "m").replace(".", "p")
    # The eyeball tag is the parameter being swept. The launcher injects
    # cfg.sweep_label naming the varying axis/axes (e.g. "mu4.3", "depth3",
    # "target_N256", "pore.H7"); use it for ANY mode. Single (non-swept) runs
    # have no label, so fall back to the ensemble's natural variable.
    label = getattr(cfg, "sweep_label", None)
    confined = getattr(cfg, "pore_geometry", None) is not None
    if label:
        tag = label.replace("ensemble.", "").replace("sampler.", "").replace("run.", "")
        tag = tag.replace("-", "m").replace(".", "p")
    elif getattr(cfg, "ensemble_type", "muVT") == "NVT":
        tag = f"N{cfg.target_n}"
    elif getattr(cfg, "target_count", None) is not None:
        tag = "tgtN" + _fmt(cfg.target_count)
    elif confined:
        tag = ""                       # the pore prefix below is the descriptor
    else:
        tag = "mu" + _fmt(cfg.chem_pot)
    if confined:
        # prefix the pore geometry+size so confined runs are self-describing
        g = cfg.pore_geometry
        sz = _fmt(cfg.pore_H) if g == "slit" else _fmt(cfg.pore_R)
        prefix = f"{'slit_H' if g=='slit' else 'cyl_R'}{sz}"
        tag = f"{prefix}_{tag}" if tag else prefix
    if _matrix_digest(cfg):
        tag = "qa_" + tag      # quenched-annealed (frozen matrix present)
        # fold the matrix realization id into the name so a target sweep (same
        # target, different matrices) gives eyeball-distinguishable dirs. Use the
        # realization's source-dir name (e.g. ".../35baa13fed99__N507/traj...").
        mc = getattr(cfg, "matrix_config", None)
        if mc:
            import os as _os
            real = _os.path.basename(_os.path.dirname(_os.path.abspath(mc)))
            real = real.split("__")[0] if real else ""   # the short hash part
            if real:
                tag = f"{tag}_{real[:8]}"
    d = os.path.join(root, f"{rid}__{tag}")
    return RunPaths(
        directory=d,
        checkpoint=os.path.join(d, "checkpoint.npz"),
        checkpoint_bak=os.path.join(d, "checkpoint.bak.npz"),
        trajectory=os.path.join(d, "trajectory.extxyz"),
        log=os.path.join(d, "run.log"),
        params=os.path.join(d, "params.json"),
        lock=os.path.join(d, "lock"),
    )

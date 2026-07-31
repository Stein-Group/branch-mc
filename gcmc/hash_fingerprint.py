"""Run identity and on-disk layout.

A run is addressed by a deterministic hash of the parameters that make it a
distinct experiment, never by parsing a name. Identity-bearing parameters are
the physics and the algorithm: potential type and its physics constants, mu,
temperature, box, dimension, cavity radius, trial density, depth, the BRANCH and
use-depth switches, the move factor, and the RNG seed (each seed is a distinct
independent chain, so it gets its own run directory). Execution controls (target
length, worker count, IO cadence, parallel mode) are excluded, which is what lets
a finished run resume as a no-op and a longer target reuse the same directory.

The g3 table path is treated as data, not a physics parameter, so it is left out
of the hash; pointing at a different table is the caller's responsibility.

Each run lives in its own directory `runs/<id>__mu<mu>/`, where the id is the
hash and the mu tag is only there for eyeballing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sim_setup import Config
import dataclasses
import json
from dataclasses import dataclass

# upper_cutoff and no_neigh are physics-bearing (interaction range, energy
# truncation), so they are identity. Adding them changes every run id relative
# to the pre-fix layout; runs produced by the biased engine should not be
# resumed anyway.
_IDENTITY_FIELDS = (
    "potential_type",
    "sampler",
    "dimension",
    "chem_pot",
    "seed",
    "box",
    "box_sides",
    "cavity_radius",
    "trial_density",
    "depth",
    "branch",
    "full_repacking",
    "move_factor",
    "temp",
    "rc",
    "eta",
    "gamma_max",
    "delta_eta",
    "lower_cutoff",
    "upper_cutoff",
    "no_neigh",
    "ensemble_type",
    "target_n",
    "pore_geometry",
    "pore_H",
    "pore_R",
    "channel",
    "channel_length",
    "channel_width",
    "aux_cutoff",
)


def _canon(value: object) -> object:
    """Canonical form of one identity value: floats are rendered at 12
    significant digits so that 0.1 read back from JSON and 0.1 typed in a
    script hash identically; everything else passes through."""
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def identity_dict(cfg: Config) -> dict[str, object]:
    """The physics-identity subset of the config: every parameter that changes
    the sampled ensemble or the move set, and NOTHING operational (paths,
    logging cadence, walltime). Two runs with equal identity dicts are the
    same physical experiment and may share a run directory / resume each
    other; that is the whole contract of the fingerprint."""
    # channel geometry only identifies a run when the channel sampler is active;
    # for every other mode these keys are omitted so the fingerprint is unchanged
    # from before the channel sampler existed.
    active_channel = getattr(cfg, "channel", False)
    skip = () if active_channel else ("channel", "channel_length", "channel_width")
    # the short-range auxiliary radius identifies a run only when that auxiliary
    # is on; with it off the key is omitted so every fingerprint predating the
    # auxiliary is byte-for-byte what it was. Note aux_cutoff DOES belong to the
    # identity when set: it changes the proposal distribution, hence the Markov
    # chain, even though the sampled ensemble is unchanged by construction.
    if getattr(cfg, "aux_cutoff", None) is None:
        skip = (*skip, "aux_cutoff")
    # box_sides is part of the physics identity only for an orthorhombic box;
    # cubic runs (box_sides None) omit it so the fingerprint is byte-identical to
    # before per-axis boxes existed.
    if getattr(cfg, "box_sides", None) is None:
        skip = (*skip, "box_sides")
    return {k: _canon(getattr(cfg, k)) for k in _IDENTITY_FIELDS if k not in skip}


def _matrix_digest(cfg: Config) -> str:
    """Short digest of the frozen obstacle coordinates (empty string when
    there is no matrix): the quenched disorder realisation is part of the
    physics identity, but hashing raw floats would let I/O noise split one
    run into many, hence the round-to-1e-6 first."""
    mpos = getattr(cfg, "matrix_positions", None)
    if mpos is None or len(mpos) == 0:
        return ""
    import numpy as np

    # round to a stable precision so floating noise in I/O does not change the id
    rounded = np.round(np.asarray(mpos, float), 6)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:12]


def run_id(cfg: Config) -> str:
    """12-hex-digit id of the run's physics identity (sorted-key JSON of
    `identity_dict`, plus the matrix digest when present, through SHA-1).
    Stable across sessions, machines, and config file formatting."""
    ident = identity_dict(cfg)
    mdig = _matrix_digest(cfg)
    if mdig:
        ident["matrix"] = mdig
    blob = json.dumps(ident, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class RunPaths:
    """The fixed file layout of one run directory: checkpoint (+ .bak for
    crash-safe rotation), extxyz trajectory, CSV log, params snapshot, and
    the lock file that serialises writers. Fields are pathlib.Path objects;
    every stdlib/numpy/ase consumer accepts them directly, and rendering for
    logs or JSON is an explicit str(...) at the boundary."""

    directory: Path
    checkpoint: Path
    checkpoint_bak: Path
    trajectory: Path
    log: Path
    params: Path
    lock: Path

    def __post_init__(self) -> None:
        # Accept plain strings from older call sites and coerce once here, so
        # every consumer downstream can rely on Path semantics. object.__setattr__
        # is the sanctioned escape hatch for a frozen dataclass's own init.
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, Path):
                object.__setattr__(self, f.name, Path(v))


def run_paths(cfg: Config, root: str = "runs") -> RunPaths:
    """RunPaths under `root` for this config: directory name is a
    human-readable tag (mode, dimension, and the swept variable via
    cfg.sweep_label) joined with the run id, so directories sort usefully AND
    collide never."""
    rid = run_id(cfg)

    def _fmt(x: float) -> str:
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
        assert cfg.target_count is not None
        tag = "tgtN" + _fmt(cfg.target_count)
    elif confined:
        tag = ""  # the pore prefix below is the descriptor
    else:
        tag = "mu" + _fmt(cfg.chem_pot)
    if confined:
        # prefix the pore geometry+size so confined runs are self-describing
        g = cfg.pore_geometry
        if g == "slit":
            assert cfg.pore_H is not None  # slit config carries H
            sz = _fmt(cfg.pore_H)
        else:
            assert cfg.pore_R is not None  # cylinder config carries R
            sz = _fmt(cfg.pore_R)
        prefix = f"{'slit_H' if g == 'slit' else 'cyl_R'}{sz}"
        tag = f"{prefix}_{tag}" if tag else prefix
    if _matrix_digest(cfg):
        tag = "qa_" + tag  # quenched-annealed (frozen matrix present)
        # fold the matrix realization id into the name so a target sweep (same
        # target, different matrices) gives eyeball-distinguishable dirs. Use the
        # realization's source-dir name (e.g. ".../35baa13fed99__N507/traj...").
        mc = getattr(cfg, "matrix_config", None)
        if mc:
            real = Path(mc).resolve().parent.name
            real = real.split("__")[0] if real else ""  # the short hash part
            if real:
                tag = f"{tag}_{real[:8]}"
    d = Path(root) / f"{rid}__{tag}"
    return RunPaths(
        directory=d,
        checkpoint=d / "checkpoint.npz",
        checkpoint_bak=d / "checkpoint.bak.npz",
        trajectory=d / "trajectory.extxyz",
        log=d / "run.log",
        params=d / "params.json",
        lock=d / "lock",
    )

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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .sim_setup import Config
import dataclasses
import json
from dataclasses import dataclass

from .. import config_specs

# upper_cutoff and no_neigh are physics-bearing (interaction range, energy
# truncation), so they are identity. Adding them changes every run id relative
# to the pre-fix layout; runs produced by the biased engine should not be
# resumed anyway.
#: The hashed set IS the fields of the three physics sub-configs. Derived
#: from the CLASSES, never listed: a field is identity-bearing because of the
#: dataclass it was declared on, not because someone remembered to add it here.
#: That indirection is the entire fix -- the eight findings in `config_fields`'s
#: docstring were all produced by maintaining a list.
#:
#: `config_fields.PHYSICS` says the same thing as names, and `config_specs`
#: checks the two against each other at import, so the triangle cannot drift.
_IDENTITY_FIELDS: tuple[str, ...] = tuple(
    f.name
    for cls in (
        config_specs.PotentialSpec,
        config_specs.SamplerSpec,
        config_specs.EnsembleSpec,
    )
    for f in dataclasses.fields(cls)
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
    # ONE rule, replacing four backward-compatibility special cases.
    #
    # This function once carried bespoke skip logic for `channel`,
    # `aux_cutoff`, `box_sides` and `species_identity`, each existing so that
    # runs predating a feature kept byte-identical ids. Meanwhile EIGHT other
    # inactive keys hashed as literal `null`, because nobody had written a
    # special case for them. Two conventions, applied by accident of history.
    #
    # R9 retires the compatibility requirement, so the rule can be the one that
    # was always correct: an inactive key is ABSENT, never null. A field that is
    # None contributes nothing distinguishing, and a run where it is set differs
    # from one where it is not by the presence of the key itself.
    #
    # `False` and `0.0` are NOT None and are hashed. That matters: `tail: false`
    # must reach the digest, which is the collision the partition test found.
    out: dict[str, Any] = {}
    for spec in config_specs.physics_specs(cfg):
        for field in dataclasses.fields(spec):
            value = getattr(spec, field.name)
            if value is not None:
                out[field.name] = _canon(value)
    return out


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

    # Read the naming inputs off the sub-configs rather than off `cfg`.
    # Every field this function needs is EnsembleSpec except `sweep_label`,
    # which is IOSpec -- and that split is the point: a directory name is built
    # from the state point (what the run IS) plus one presentational label, and
    # nothing else. Narrowing the read makes that visible instead of leaving it
    # to be inferred from eleven scattered `cfg.` accesses.
    ens = config_specs.ensemble_spec(cfg)
    io = config_specs.io_spec(cfg)

    def _fmt(x: float) -> str:
        return format(x, ".4g").replace("-", "m").replace(".", "p")

    # The eyeball tag is the parameter being swept. The launcher injects
    # cfg.sweep_label naming the varying axis/axes (e.g. "mu4.3", "depth3",
    # "target_N256", "pore.H7"); use it for ANY mode. Single (non-swept) runs
    # have no label, so fall back to the ensemble's natural variable.
    label = io.sweep_label
    confined = ens.pore_geometry is not None
    if label:
        tag = label.replace("ensemble.", "").replace("sampler.", "").replace("run.", "")
        tag = tag.replace("-", "m").replace(".", "p")
    elif ens.ensemble_type == "NVT":
        tag = f"N{ens.target_n}"
    elif ens.target_count is not None:
        tag = "tgtN" + _fmt(ens.target_count)
    elif confined:
        tag = ""  # the pore prefix below is the descriptor
    else:
        tag = "mu" + _fmt(ens.chem_pot)
    if confined:
        # prefix the pore geometry+size so confined runs are self-describing
        g = ens.pore_geometry
        if g == "slit":
            assert ens.pore_H is not None  # slit config carries H
            sz = _fmt(ens.pore_H)
        else:
            assert ens.pore_R is not None  # cylinder config carries R
            sz = _fmt(ens.pore_R)
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

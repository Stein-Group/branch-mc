"""Checkpointed, resumable multi-species MOF GCMC runs (Q3).

The V2/V3 drivers run a fixed number of windows in memory and print at the end;
that is fine for smoke tests but S3 record runs are killed by walltime, so they
must checkpoint and resume bit-identically and stream N_s per window. This is
that wrapper around `advance_metropolis_mof_ms`. It never touches the frozen
main-engine dispatch; it reuses the same crash-safe checkpoint format (schema 3
carries the species column) and the same RNG-state round-trip that makes a
resumed run identical to an unkilled one.

A run is identified by its physics, not by how many windows have been done:
forcefield + framework (or bulk box) + T + guests + per-species z* + seed +
window_cycles. Total `windows` is a target that grows leg to leg, so two legs of
the same run share a directory and the second resumes the first. Change any
physics field and you get a new directory; you can never silently mix streams.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gcmc import io
from gcmc.hash_fingerprint import RunPaths
from gcmc.mof_channel import channel_repack_move_ms
from gcmc.mof_metropolis import advance_metropolis_mof_ms, mof_energy_decomposition
from scripts.mof.system_setup import SystemSetup, setup_system


@dataclass
class MofRunSpec:
    forcefield: Path
    T: float
    z_star_by_name: dict[str, float]  # reduced activity per guest species name
    seed: int
    windows: int  # total target (a leg extends toward it)
    window_cycles: int
    cif: Path | None = None
    bulk_box: float | None = None
    reference: str = "Xe"
    species_probs_by_name: dict[str, float] | None = None
    move_factor: float = 1.0
    n_neigh: int = 40
    checkpoint_every: int = 50
    # channel-sweep repack move (A5.6): number of whole-window heat-bath repacks
    # per advance window (0 = conventional sampling only)
    sweep_moves_per_window: int = 0
    sweep_L_long: float = 4.0
    sweep_w: float = 0.9
    sweep_n_trials: int = 12
    root: Path = field(default_factory=lambda: Path("runs/mof"))


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _identity(spec: MofRunSpec) -> dict[str, object]:
    """Canonical, physics-only provenance. Excludes `windows` (a target that
    grows) and IO knobs (checkpoint_every), includes everything that changes the
    Markov chain."""
    guests = sorted(spec.z_star_by_name)
    return {
        "ff_sha": _sha(spec.forcefield),
        "system": _sha(spec.cif) if spec.cif is not None else f"bulk{spec.bulk_box:g}",
        "reference": spec.reference,
        "T": float(spec.T),
        "guests": guests,
        "z_star": {g: float(spec.z_star_by_name[g]) for g in guests},
        "species_probs": (
            {g: float(spec.species_probs_by_name[g]) for g in guests}
            if spec.species_probs_by_name is not None
            else None
        ),
        "seed": int(spec.seed),
        "window_cycles": int(spec.window_cycles),
        "move_factor": float(spec.move_factor),
        "n_neigh": int(spec.n_neigh),
        # sweep-move settings change the Markov chain and RNG consumption, so they
        # are part of the physics identity (a run with a different sweep config is
        # a different chain and must not resume this one)
        "sweep_moves_per_window": int(spec.sweep_moves_per_window),
        "sweep_L_long": float(spec.sweep_L_long),
        "sweep_w": float(spec.sweep_w),
        "sweep_n_trials": int(spec.sweep_n_trials),
    }


def mof_run_id(spec: MofRunSpec) -> str:
    blob = json.dumps(_identity(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        directory=run_dir,
        checkpoint=run_dir / "checkpoint.npz",
        checkpoint_bak=run_dir / "checkpoint.npz.bak",
        trajectory=run_dir / "traj.extxyz",
        log=run_dir / "windows.csv",
        params=run_dir / "params.json",
        lock=run_dir / "lock",
    )


@dataclass
class MofRunResult:
    spec: MofRunSpec
    system: SystemSetup
    paths: RunPaths
    run_id: str
    positions: np.ndarray
    species: np.ndarray
    n_by_type: dict[int, int]
    windows_done: int
    resumed_from: int


def run_mof(spec: MofRunSpec) -> MofRunResult:
    sysm = setup_system(
        spec.forcefield,
        spec.T,
        cif=spec.cif,
        bulk_box=spec.bulk_box,
        reference=spec.reference,
    )
    table, fieldobj = sysm.table, sysm.field
    guests = sorted(spec.z_star_by_name)
    z_star = {table.index(g): spec.z_star_by_name[g] for g in guests}
    n_species = len(z_star)
    guests_ids = sorted(z_star)
    species_probs = (
        {table.index(g): spec.species_probs_by_name[g] for g in guests}
        if spec.species_probs_by_name is not None
        else None
    )

    rid = mof_run_id(spec)
    paths = _paths(spec.root / f"mof_{sysm.label.replace(' ', '_')}_{rid}")
    identity = _identity(spec)
    io.ensure_run_dir(paths, identity)

    ck = io.load_checkpoint(paths)
    if ck is not None:
        if ck["schema"] not in (2, 3):
            raise ValueError(f"unreadable checkpoint schema {ck['schema']}")
        pos = np.ascontiguousarray(ck["positions"], float)
        spc = ck["species"]
        if spc is None:
            if n_species > 1:
                raise ValueError(
                    "multi-species resume needs a species column; this checkpoint "
                    "has none (it was written by a single-species run)"
                )
            spc = np.zeros(len(pos), dtype=np.int64)
        else:
            spc = np.ascontiguousarray(spc, np.int64)
        step0 = int(ck["step"])
        rng = io.set_rng_state(np.random.default_rng(), ck["rng"])
    else:
        pos = np.empty((0, sysm.box.shape[0]))
        spc = np.empty(0, dtype=np.int64)
        step0 = 0
        rng = np.random.default_rng(spec.seed)

    resumed_from = step0
    new_log = not paths.log.exists() or step0 == 0
    # fixed, rectangular decomposition columns from the guest set (windows with a
    # species absent report 0.0 for its terms rather than a ragged row)
    hg_cols = [f"u_hg_{g}" for g in guests]
    gg_cols = [
        f"u_gg_{a}_{b}" for a, b in itertools.combinations_with_replacement(guests, 2)
    ]
    decomp_cols = [*hg_cols, *gg_cols, "u_total"]
    res = None
    cum_evals = 0
    with paths.log.open("w" if new_log else "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(
                [
                    "window",
                    *[f"N_{g}" for g in guests],
                    "N_total",
                    *decomp_cols,
                    "n_evals",
                    "cum_n_evals",
                    "wall_s",
                ]
            )
            fh.flush()
        for w in range(step0, spec.windows):
            t0 = time.perf_counter()
            res = advance_metropolis_mof_ms(
                pos,
                spc,
                rng,
                field=fieldobj,
                table=table,
                z_star=z_star,
                beta=sysm.beta,
                box=sysm.box,
                volume=sysm.volume,
                n_cells=sysm.n_cells,
                upper_cutoff=fieldobj.rc,
                move_factor=spec.move_factor,
                n_neigh=spec.n_neigh,
                window=spec.window_cycles,
                species_probs=species_probs,
            )
            wall = time.perf_counter() - t0
            pos, spc = res.positions, res.species
            assert spc is not None and res.n_by_type is not None
            # A5.6: interleave whole-window heat-bath repacks (multi-species only;
            # a single species has nothing to recompose and the move is a no-op)
            if spec.sweep_moves_per_window and n_species > 1:
                for _ in range(spec.sweep_moves_per_window):
                    pos, spc, _moved = channel_repack_move_ms(
                        pos,
                        spc,
                        rng,
                        field=fieldobj,
                        table=table,
                        z_star=z_star,
                        beta=sysm.beta,
                        box=sysm.box,
                        rc=fieldobj.rc,
                        guest_types=guests_ids,
                        L_long=spec.sweep_L_long,
                        w=spec.sweep_w,
                        n_trials=spec.sweep_n_trials,
                    )
            counts = [res.n_by_type[table.index(g)] for g in guests]
            decomp = mof_energy_decomposition(
                pos,
                spc,
                field=fieldobj,
                table=table,
                beta=sysm.beta,
                upper_cutoff=fieldobj.rc,
            )
            cum_evals += res.n_evals
            writer.writerow(
                [
                    w + 1,
                    *counts,
                    res.n,
                    *[f"{decomp.get(c, 0.0):.10g}" for c in decomp_cols],
                    res.n_evals,
                    cum_evals,
                    f"{wall:.6g}",
                ]
            )
            fh.flush()
            if (w + 1) % spec.checkpoint_every == 0:
                io.save_checkpoint(
                    paths,
                    positions=pos,
                    step=w + 1,
                    rng=io.rng_state(rng),
                    counters=res.counters,
                    params=identity,
                    species=spc,
                )

    # final checkpoint so a resumed leg always lands on a boundary
    final_step = spec.windows
    io.save_checkpoint(
        paths,
        positions=pos,
        step=final_step,
        rng=io.rng_state(rng),
        counters=res.counters if res is not None else {},
        params=identity,
        species=spc,
    )
    n_by_type = (
        res.n_by_type
        if res is not None and res.n_by_type is not None
        else {
            table.index(g): int(np.count_nonzero(spc == table.index(g))) for g in guests
        }
    )
    return MofRunResult(
        spec=spec,
        system=sysm,
        paths=paths,
        run_id=rid,
        positions=pos,
        species=spc,
        n_by_type=n_by_type,
        windows_done=final_step,
        resumed_from=resumed_from,
    )

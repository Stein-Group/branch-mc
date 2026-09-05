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

from gcmc.engine.framework import (
    advance_metropolis_mof_ms,
    channel_repack_move_ms,
    mof_energy_decomposition,
)
from gcmc.run import io
from gcmc.run.hash_fingerprint import RunPaths
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
    # short-range (dual-cutoff CBMC) auxiliary for the sweep move: PROPOSAL radius
    # in reduced units. None = no auxiliary (the move stays a heat bath). Setting
    # it equal to rc is an exact no-op and is the control arm. The physics cutoff
    # is always the forcefield rc; this only shortens the proposal.
    sweep_rc_aux: float | None = None
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
        # the auxiliary changes the acceptance test and the RNG consumption, so an
        # aux-on leg must never resume an aux-off directory (BLOCKER-2 class bug)
        "sweep_rc_aux": (
            None if spec.sweep_rc_aux is None else float(spec.sweep_rc_aux)
        ),
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
    # The pair keys must be built in the order mof_energy_decomposition emits them,
    # which is TYPE-ID order (guests at the low ids), not alphabetical order of the
    # names. Building them alphabetically silently wrote 0.0 into every cross-species
    # column via decomp.get(col, 0.0) -- u_gg_Xe_Kr was emitted, u_gg_Kr_Xe was
    # asked for -- while u_total still contained the term, so the printed components
    # did not sum to the printed total. Ordering by table index makes the two agree
    # by construction; the assertion below is what would have caught it.
    guests_by_type = sorted(guests, key=table.index)
    gg_cols = [
        f"u_gg_{a}_{b}"
        for a, b in itertools.combinations_with_replacement(guests_by_type, 2)
    ]
    decomp_cols = [*hg_cols, *gg_cols, "u_total"]
    sweep_cols = (
        [
            "sweep_attempts",
            "sweep_accepts",
            "sweep_k_mean",
            "sweep_k_max",
            # The auxiliary correction's COUNT, not just its moments. A mean and
            # an sd of 0 are what a correct no-op arm reports AND what a
            # correction that never ran reports, so without this column the
            # control arm cannot be told from a vacuous one -- which is how the
            # `rc_aux is not None and rc_sel < rc` short-circuit went unnoticed.
            "aux_corr_n",
            "aux_corr_mean",
            "aux_corr_sd",
        ]
        if spec.sweep_moves_per_window and n_species > 1
        else []
    )
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
                    *sweep_cols,
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
            pos, spc = res.positions, res.species
            assert spc is not None and res.n_by_type is not None
            # A5.6: interleave whole-window heat-bath repacks (multi-species only;
            # a single species has nothing to recompose and the move is a no-op)
            sweep_stats: dict[str, float] = {}
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
                        rc_aux=spec.sweep_rc_aux,
                        stats=sweep_stats,
                    )
            # wall_s is timed AFTER the sweep moves, not before them: they are part
            # of the window's cost and V5's secondary per-CPU-second metric is
            # meaningless if the sampler being benchmarked is outside the clock.
            wall = time.perf_counter() - t0
            counts = [res.n_by_type[table.index(g)] for g in guests]
            decomp = mof_energy_decomposition(
                pos,
                spc,
                field=fieldobj,
                table=table,
                beta=sysm.beta,
                upper_cutoff=fieldobj.rc,
            )
            # The stream is only trustworthy if the terms it prints add up to the
            # total it prints. A column name the decomposition never emits would
            # otherwise be written as a silent 0.0 and the loss would show up
            # nowhere -- which is exactly how the cross-species pair term was lost.
            _parts = sum(decomp.get(c, 0.0) for c in decomp_cols if c != "u_total")
            _tot = decomp.get("u_total", 0.0)
            assert abs(_parts - _tot) <= 1e-8 * max(1.0, abs(_tot)), (
                f"decomposition columns do not sum to u_total ({_parts!r} vs "
                f"{_tot!r}); a column name is missing from the emitted keys "
                f"{sorted(decomp)} and is being written as 0.0"
            )
            cum_evals += res.n_evals
            sweep_vals: list[object] = []
            if sweep_cols:
                n_corr = sweep_stats.get("corr_n", 0.0)
                s1 = sweep_stats.get("corr_sum", 0.0)
                s2 = sweep_stats.get("corr_sumsq", 0.0)
                mean = s1 / n_corr if n_corr else 0.0
                var = max(s2 / n_corr - mean * mean, 0.0) if n_corr else 0.0
                n_att = sweep_stats.get("attempts", 0.0)
                sweep_vals = [
                    int(n_att),
                    int(sweep_stats.get("accepts", 0.0)),
                    f"{(sweep_stats.get('k_sum', 0.0) / n_att if n_att else 0.0):.4g}",
                    int(sweep_stats.get("k_max", 0.0)),
                    int(n_corr),
                    f"{mean:.6g}",
                    f"{var**0.5:.6g}",
                ]
            writer.writerow(
                [
                    w + 1,
                    *counts,
                    res.n,
                    *[f"{decomp.get(c, 0.0):.10g}" for c in decomp_cols],
                    *sweep_vals,
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

"""Run one framework params file: checkpointed, resumable, one folder per run.

`python -m gcmc run params.json` lands here when the file has a `framework`
block (see `gcmc.framework.params` for the schema). BRANCH_MC engines run in
process; the raspa2_* engines write a RASPA2 case into the run folder, run
`$RASPA_DIR/bin/simulate` and parse its stream, so both codes leave the same
`run.log` behind.

The run folder (`<root>/<id>__<label>`, as for every other run):
  params.json        the params file exactly as run
  state_point.json   what it resolved to: fugacities, phi, z*, box, window sizes
  run.log            one row per window (BRANCH) or printed cycle (RASPA2)
  checkpoint.npz     BRANCH only; a resubmission continues from it
  raspa2/            RASPA2 only: the case and its Output/

Resuming: the id is the run's physics, so resubmitting the same file finds the
same folder. A BRANCH run continues from its checkpoint (rows a killed leg had
streamed past it are dropped first); a RASPA2 case that finished is only
re-parsed, one that was killed is started again.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..engine.glauber_dynamics import ladder_columns
from ..engine.rigid_framework import (
    advance_metropolis_mof_ms,
    framework_ladder_move,
    ladder_window_moves_ms,
    mof_energy,
)
from ..forcefield.lj_table import FrameworkField
from ..forcefield.pair_table import PairTable
from ..forcefield.species_file import ReducedTables
from ..run import io
from ..run.hash_fingerprint import RunPaths
from ..run.runner import ENERGY_DRIFT_TOL, acquire_lock, release_lock
from ..run.sim_setup import (
    assert_channel_guard,
    assert_channel_width,
    cavity_trial_count,
    channel_trial_count,
    ladder_window,
)
from .params import LADDER_ENGINES, FrameworkRun, identity, parse, run_dir
from .system import SystemSetup, setup_system


@dataclass
class FrameworkResult:
    """What a framework run left behind: its folder, how far it got, the final
    guest count per species and (BRANCH engines) the final configuration in
    reduced units, so tests and callers need not re-read the checkpoint."""

    directory: Path
    step: int  # windows done (BRANCH) or production cycles (RASPA2)
    n_by_species: dict[str, int]
    positions: np.ndarray | None = None  # BRANCH only
    species: np.ndarray | None = None

    @property
    def n(self) -> int:
        """Total guests at the end (the `N=` of the `done:` line)."""
        return sum(self.n_by_species.values())


def paths_for(directory: Path) -> RunPaths:
    """The file layout of a framework run folder: the bulk runner's names
    (checkpoint.npz, run.log, params.json, lock), so tooling reads both alike."""
    return RunPaths(
        directory=directory,
        checkpoint=directory / "checkpoint.npz",
        checkpoint_bak=directory / "checkpoint.bak.npz",
        trajectory=directory / "trajectory.extxyz",
        log=directory / "run.log",
        params=directory / "params.json",
        lock=directory / "lock",
    )


def ladder_geometry(
    engine: str,
    sampler: dict[str, Any],
    kmax_margin: int,
    box: np.ndarray,
    rc: float,
    table: PairTable,
    guest_ids: list[int],
) -> tuple[float, float, int, int]:
    """(channel_length, channel_width, M, kmax) of the window, by the bulk params
    file's own derivation, so a window a bulk run would refuse (periodic guard,
    transfer-matrix frame, cavity cost budget) is refused identically here.
    `sampler` is the params file's sampler block."""
    dim = len(box)
    diameter = float(max(table.sigma[t, t] for t in guest_ids))
    branch = sampler["type"] == "branch"
    if branch:
        L = W = 0.0
        n_trials = cavity_trial_count(
            sampler["trial_density"], sampler["trial_box_side"], dim, diameter
        )
    else:
        L, W = sampler["channel_length"], sampler["channel_width"]
        n_trials = channel_trial_count(sampler["trial_density"], L, W, dim)
        assert_channel_width(W, diameter)
        assert_channel_guard(L, W, dim, float(np.min(box)), rc)
    return ladder_window(
        engine=engine,
        ladder=True,
        branch_cavity=branch,
        n_trials=n_trials,
        channel_length=L,
        channel_width=W,
        trial_box_side=sampler.get("trial_box_side", 0.0),
        trial_density=sampler["trial_density"],
        kmax_margin=kmax_margin,
        channel_full=False,
        dim=dim,
        box=float(np.min(box)),
        uc=rc,
        diameter=diameter,
        lj=True,
    )


def preload_guests(
    counts: dict[int, int],
    rng: np.random.Generator,
    *,
    field: FrameworkField,
    table: PairTable,
    box: np.ndarray,
    max_tries: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Random sequential addition of `counts[type]` guests (reduced units). A
    point is kept only if it clears the framework floor and c * sigma_ij against
    every guest placed; no energy test, because this is an initial condition the
    chain is meant to relax. Raises if the count does not fit."""
    box = np.asarray(box, float)
    order = [t for t, n in sorted(counts.items()) for _ in range(int(n))]
    order = [order[i] for i in rng.permutation(len(order))]
    P = np.empty((0, len(box)))
    S = np.empty(0, dtype=np.int64)
    tries = 0
    for t in order:
        while True:
            tries += 1
            if tries > max_tries:
                raise ValueError(
                    f"run.initial_config {counts} does not fit: {len(S)} placed "
                    f"after {max_tries} tries"
                )
            p = rng.uniform(0.0, 1.0, len(box)) * box
            if field.overlaps(p, t):
                continue
            if len(P):
                dr = P - p
                dr -= box * np.round(dr / box)
                d = np.sqrt(np.einsum("ij,ij->i", dr, dr))
                if np.any(d < table.core_floor[t, S]):
                    continue
            P = np.vstack([P, p])
            S = np.append(S, np.int64(t))
            break
    return P, S


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def run_framework(
    params: dict[str, Any] | str | Path, root: str | Path = "runs"
) -> FrameworkResult:
    """Run or resume one framework params file (a dict or a path to one)."""
    doc: dict[str, Any] = (
        params if isinstance(params, dict) else json.loads(Path(params).read_text())
    )
    fr = parse(doc)
    paths = paths_for(run_dir(fr, root))
    io.ensure_run_dir(paths, doc)
    acquire_lock(paths)
    try:
        if fr.is_raspa:
            from .raspa2 import run_raspa2

            return run_raspa2(fr, paths)
        return _run_branch(fr, paths)
    finally:
        release_lock(paths)


def _run_branch(fr: FrameworkRun, paths: RunPaths) -> FrameworkResult:
    ff = fr.forcefield
    sysm: SystemSetup = setup_system(
        ff.path,
        fr.T,
        cif=fr.cif,
        bulk_box=fr.empty_box,
        reference=ff.guests[0],
        replicate=fr.replicate,
    )
    table, fieldobj = sysm.table, sysm.field
    red = cast(ReducedTables, sysm.red)
    guests = fr.guests
    z_star = {table.index(g): red.z_star(f) for g, f in fr.fugacity_pa.items()}
    guest_ids = sorted(z_star)
    windows = math.ceil(fr.target_cycles / fr.sampling_window)
    e_p = fr.move_probs[1]

    ladder = None
    n_window_moves = 0
    plan: dict[str, Any] = {}
    if fr.engine in LADDER_ENGINES:
        L, W, M, kmax = ladder_geometry(
            fr.engine,
            fr.sampler,
            fr.kmax_margin,
            sysm.box,
            fieldobj.rc,
            table,
            guest_ids,
        )
        ladder = framework_ladder_move(
            field=fieldobj,
            table=table,
            z_star=z_star,
            beta=sysm.beta,
            box=sysm.box,
            rc=fieldobj.rc,
            engine=fr.engine,
            sampler=fr.sampler["type"],
            channel_length=L,
            channel_width=W,
            trial_density=fr.sampler["trial_density"],
            M=M,
            kmax=kmax,
            kmax_margin=fr.kmax_margin,
        )
        # the exchange share of the window as window moves, the rest single-
        # particle moves; two blocks per window (see gcmc.framework.params)
        n_window_moves = round(e_p * fr.sampling_window)
        body = (
            [fr.sampler["trial_box_side"]] * 3
            if fr.sampler["type"] == "branch"
            else [L, W, W]
        )
        plan = {
            "pool_M": M,
            "kmax": kmax,
            "body_sigma": body,
            "body_angstrom": [b * sysm.sigma_ref for b in body],
            "window_moves_per_row": n_window_moves,
            "single_particle_moves_per_row": fr.sampling_window - n_window_moves,
        }
    _write_json(
        paths.directory / "state_point.json",
        {
            "fugacity_pa": fr.fugacity_pa,
            "pressure_pa": fr.pressure_pa,
            "y": fr.y,
            "fugacity_coefficient": fr.phi,
            "z_star": {g: red.z_star(f) for g, f in fr.fugacity_pa.items()},
            "sigma_ref_angstrom": sysm.sigma_ref,
            "T_star": 1.0 / sysm.beta,
            "box_sigma": [float(b) for b in sysm.box],
            "box_angstrom": [float(b) * sysm.sigma_ref for b in sysm.box],
            "rc_sigma": float(fieldobj.rc),
            "framework_mass_amu": sysm.mass_amu,
            "cif_cells": sysm.n_cells,
            "move_probabilities": dict(
                zip(
                    ("translation", "exchange", "identity_swap"),
                    fr.move_probs,
                    strict=True,
                )
            ),
            "rows_target": windows,
            "ladder": plan or None,
        },
    )

    ident = identity(fr)
    ck = io.load_checkpoint(paths)
    if ck is not None:
        pos = np.ascontiguousarray(ck["positions"], float)
        spc = np.ascontiguousarray(
            ck["species"] if ck["species"] is not None else np.zeros(len(pos)), np.int64
        )
        step0 = int(ck["step"])
        rng = io.set_rng_state(np.random.default_rng(), ck["rng"])
        # rows a killed leg streamed past its last checkpoint are replayed now
        io.trim_log(paths.log, step0 * fr.sampling_window / 1e6)
    else:
        rng = np.random.default_rng(fr.seed)
        pos = np.empty((0, 3))
        spc = np.empty(0, dtype=np.int64)
        step0 = 0
        if fr.preload:
            pos, spc = preload_guests(
                {table.index(g): n for g, n in fr.preload.items()},
                rng,
                field=fieldobj,
                table=table,
                box=sysm.box,
            )

    def full_energy() -> tuple[float, float]:
        return mof_energy(
            pos,
            spc,
            field=fieldobj,
            table=table,
            beta=sysm.beta,
            upper_cutoff=fieldobj.rc,
        )

    u_host, u_guest = full_energy()
    S = fr.sampling_window
    n_single = S - n_window_moves
    log = io.RunLog(paths.log)
    t_prev = time.perf_counter()
    for w in range(step0, windows):
        row: dict[str, Any] = {io.LOG_CYCLE: (w + 1) * S / 1e6}
        columns: dict[str, Any] = {}
        if n_single:
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
                move_factor=1.0,
                n_neigh=40,
                window=n_single,
                exchange=ladder is None,
                move_probs=fr.move_probs,
                count_every_move=True,
            )
            assert res.species is not None
            pos, spc = res.positions, res.species
            u_host += res.du_host
            u_guest += res.du_guest
            columns.update(res.counters)
            columns["guest_pairs_evaluated"] = res.n_evals
        if ladder is not None and n_window_moves:
            stats: dict[str, float] = {}
            pos, spc = ladder_window_moves_ms(
                ladder, pos, spc, rng, n_window_moves, stats
            )
            u_host += stats["dU_host"]
            u_guest += stats["dU_guest"]
            columns.update(ladder_columns(stats, fixed_n=False, full_span=False))
        checkpoint_now = (w + 1) % fr.checkpoint_every == 0 or w + 1 == windows
        if checkpoint_now:
            h, g = full_energy()
            scale = max(1.0, abs(h + g))
            if abs(u_host - h) + abs(u_guest - g) > ENERGY_DRIFT_TOL * scale:
                raise RuntimeError(
                    f"{paths.directory}: running U (host {u_host!r}, guest "
                    f"{u_guest!r}) but a full recompute gives ({h!r}, {g!r}) at "
                    f"row {w + 1}. An engine booked a wrong energy change."
                )
            u_host, u_guest = h, g
        for g_name in guests:
            row[f"N_{g_name}"] = int(np.count_nonzero(spc == table.index(g_name)))
        row["U_host_guest"] = u_host
        row["U_guest_guest"] = u_guest
        row.update(columns)
        now = time.perf_counter()
        row["wall_s"] = now - t_prev
        t_prev = now
        log.write(row)
        if checkpoint_now:
            log.flush()
            io.save_checkpoint(
                paths,
                positions=pos,
                step=w + 1,
                rng=io.rng_state(rng),
                params=ident,
                species=spc,
            )
    log.close()
    return FrameworkResult(
        directory=paths.directory,
        step=max(windows, step0),
        n_by_species={g: int(np.count_nonzero(spc == table.index(g))) for g in guests},
        positions=pos,
        species=spc,
    )

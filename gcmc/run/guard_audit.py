"""The periodic guard, audited over configurations rather than enforced in one.

WHAT THE GUARD IS. The channel sweep takes distances on LOCAL cuboid
coordinates and does no minimum-image wrapping (`sampling.channel`, module
docstring). That is only equal to the periodic distance when the whole body,
plus a cutoff margin, fits inside the smallest box dimension:

    sqrt(L_long^2 + 2 w^2)  <=  min(box) - cutoff

`proposals.channel_move.window_guard_ok` is that predicate and is the single
definition; nothing here re-derives it.

WHY AN AUDIT MODULE EXISTS AT ALL. The predicate has exactly two call sites in
the shipped tree -- the MOF window builder and the test-only bulk window
builder -- and `metropolis_dynamics.advance` calls neither. So the bulk channel sampler
has always taken local Euclidean distances under an UNCHECKED precondition, and
nobody knows for which shipped configurations it held. G0.9 of
the design plan asks for the scan that finds out, and this module is the
scan.

THE RADIUS THE AUDIT USES IS NOT THE RADIUS THE NEW ENGINE ENFORCES, and
conflating the two is budgeted in the plan's risk table at one day.

    AUDIT at `rc_sel`   The existing bulk sweep scores pairs at the SELECTION
                        radius, so `rc_sel` is the radius at which its
                        local-Euclidean assumption has to hold. Auditing at
                        `upper_cutoff` reports violations of a precondition the
                        shipped code never had.
    ENFORCE at `uc`     The Glauber engine scores at the full interaction
                        cutoff (it has no acceptance step to divide a shortened
                        radius back out of), so it enforces at `upper_cutoff`.

A configuration that passes at `rc_sel` and fails at `uc` is therefore NOT a
past defect. It is a configuration the new engine refuses, and this module
reports it as such -- `advisory`, never `violation`.

THE HARD-CORE CUTOFF is the largest active pair floor, not the interaction
cutoff: with no interaction beyond contact, the radius at which pairs are
scored IS the floor. `build_channel_window_ms` already makes that choice
(`cutoff = rc if lj else max_floor`) and this module reproduces it from the
same inputs.

TWO DIMENSIONS, ONE FORMULA. `window_guard_ok` uses `2 w^2` unconditionally,
which is the 3D cuboid diagonal. In 2D the body is L x w and the diagonal is
sqrt(L^2 + w^2), so the shipped predicate is STRICTER than necessary there.
Stricter is safe -- it can refuse a legal 2D configuration but never accept an
illegal one -- and it is left alone: changing it would relax a guard, which is
the one direction a guard must never move without its own gate.

THE AUDIT SINK. Set `GCMC_GUARD_AUDIT=<path>` and every Config built by
`from_params` that uses a channel sampler appends one JSON line there. That is
how the scan reaches the channel cases built programmatically inside the test
suite -- there are three dozen of them and no params file names any -- without
a maintained list of test modules, which is exactly the kind of list this
project has already been bitten by. Unset (the default) the hook is one
`os.environ.get` per run and writes nothing.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..proposals.channel_move import window_guard_ok

if TYPE_CHECKING:
    from .sim_setup import Config

#: Environment variable naming the JSONL sink; unset means "do not audit".
SINK_ENV = "GCMC_GUARD_AUDIT"


@dataclass(frozen=True)
class GuardCheck:
    """One (body, box, cutoff) verdict, with the numbers behind it."""

    diag: float  # sqrt(L_long^2 + 2 w^2), the body diagonal the guard uses
    limit: float  # min(box) - cutoff
    cutoff: float
    ok: bool

    @property
    def margin(self) -> float:
        """limit - diag: positive passes, negative is the amount it fails by."""
        return self.limit - self.diag


def check(L_long: float, w: float, box: Any, cutoff: float) -> GuardCheck:
    """The guard verdict plus the two numbers that produced it.

    `window_guard_ok` is the authority on the boolean; this only adds the
    diagnostics, so an audit line can say by how much a configuration failed
    rather than only that it did.
    """
    b = np.atleast_1d(np.asarray(box, dtype=np.float64))
    diag = math.sqrt(float(L_long) ** 2 + 2.0 * float(w) ** 2)
    limit = float(np.min(b)) - float(cutoff)
    return GuardCheck(
        diag=diag,
        limit=limit,
        cutoff=float(cutoff),
        ok=bool(window_guard_ok(float(L_long), float(w), b, float(cutoff))),
    )


def scoring_cutoffs(cfg: Config) -> tuple[float, float]:
    """(rc_sel, uc) -- the radius the SHIPPED sweep scores at, and the radius
    the new engine will.

    On the LJ path `rc_sel` is the auxiliary radius when one is configured and
    `upper_cutoff` otherwise, exactly as `metropolis_dynamics.advance` derives
    it. On the
    hard-core path neither radius is an interaction range: pairs conflict at the
    floor and nowhere else, so both are the largest active pair floor -- the
    same choice `build_channel_window_ms` makes.
    """
    lj = cfg.potential_type in ("lj", "ljbranch", "ljaux")
    if not lj:
        sp = cfg.species
        if sp is None:
            floor = float(cfg.lower_cutoff)
        else:
            f = np.asarray(sp.table.core_floor, dtype=np.float64)
            p = np.asarray(
                [i for i, t in enumerate(sp.p_trial) if float(t) > 0.0], dtype=np.int64
            )
            floor = float(f[np.ix_(p, p)].max()) if len(p) else float(cfg.lower_cutoff)
        return floor, floor
    uc = float(cfg.upper_cutoff)
    rc_sel = uc if cfg.aux_cutoff is None else float(cfg.aux_cutoff)
    return rc_sel, uc


def audit(cfg: Config, *, label: str | None = None) -> dict[str, Any] | None:
    """One audit record for a channel Config, or None when the guard cannot
    apply.

    None for two reasons, both of them "the guard is not a statement about this
    run": the sampler is not a channel, or the body has no fixed length. The
    second is `channel_length = "full"`, whose body is a per-move sub-segment of
    the chord and is held inside the periodic images by `rod_image_length`, per
    orientation. A single fits-inside test on a configured length cannot express
    that and would be answering a question this run does not ask.
    """
    if not cfg.channel:
        return None
    if cfg.channel_length <= 0.0:
        return None
    rc_sel, uc = scoring_cutoffs(cfg)
    at_sel = check(cfg.channel_length, cfg.channel_width, cfg.box_dims, rc_sel)
    at_uc = check(cfg.channel_length, cfg.channel_width, cfg.box_dims, uc)
    return {
        "label": label,
        "sampler": cfg.sampler,
        "potential_type": cfg.potential_type,
        "dimension": cfg.dimension,
        "box_dims": [float(v) for v in cfg.box_dims],
        "channel_length": float(cfg.channel_length),
        "channel_width": float(cfg.channel_width),
        "typed": cfg.species is not None,
        # the SHIPPED precondition
        "at_rc_sel": asdict(at_sel),
        # what the NEW engine will enforce; advisory only
        "at_uc": asdict(at_uc),
        "violation": not at_sel.ok,
        "advisory_refused_by_glauber": at_sel.ok and not at_uc.ok,
    }


def record(cfg: Config) -> None:
    """Append one audit line to the sink, if a sink is configured.

    Called from `from_params` so that EVERY channel Config the process builds is
    audited, wherever it was built. Silent and cheap when the sink is unset,
    which is every production run; failures to write are swallowed on purpose --
    an audit is never allowed to break a run.
    """
    path = os.environ.get(SINK_ENV)
    if not path:
        return
    try:
        rec = audit(cfg)
        if rec is None:
            return
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    except Exception:  # an audit must never break a run
        pass


def scan_params_files(paths: list[Path]) -> list[dict[str, Any]]:
    """Audit every params file under `paths` that names a channel sampler.

    Static: it builds each Config through `from_params` (the only authority on
    what is legal) and asks the guard. No simulation runs, which is what makes
    this a tier-0 check. Files that are not params files, or that `from_params`
    refuses for unrelated reasons, are skipped -- a scan is not a validator.
    """
    from . import sim_setup

    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in paths:
        files = (
            sorted(root.rglob("*.json")) + sorted(root.rglob("*.jsonc"))
            if root.is_dir()
            else [root]
        )
        for f in files:
            if f in seen:
                continue
            seen.add(f)
            try:
                doc = json.loads(f.read_text())
            except Exception:  # not JSON, or annotated JSONC
                continue
            if not isinstance(doc, dict) or "sampler" not in doc:
                continue
            smp = doc.get("sampler")
            if not isinstance(smp, dict) or smp.get("type") != "channel":
                continue
            for params in _expand(doc):
                try:
                    cfg = sim_setup.from_params(params)
                except Exception as exc:  # a scan is not a validator
                    # EXCEPT when the refusal IS the guard. `sim_setup` already
                    # refuses a channel body that does not fit -- see the note
                    # in `summarise` -- so silently skipping refused files would
                    # make this scan circular: it would report "all pass"
                    # precisely because the failures never reach it.
                    if "periodic guard" in str(exc):
                        out.append(
                            {
                                "label": str(f),
                                "sampler": smp.get("type"),
                                "refused_at_config_time": True,
                                "message": str(exc),
                                "violation": False,
                                "advisory_refused_by_glauber": False,
                            }
                        )
                    continue
                rec = audit(cfg, label=str(f))
                if rec is not None:
                    out.append(rec)
    return out


def _expand(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """One params file may name a sweep; the guard is a per-run property.

    Only the axes that can change the guard are expanded -- the box, the body
    length and width -- because expanding the full launcher product here would
    make a static scan quadratic in the number of species files for no change in
    verdict.
    """
    axes = (
        ("ensemble", "box"),
        ("sampler", "channel_length"),
        ("sampler", "channel_width"),
    )
    out = [doc]
    for block, key in axes:
        nxt: list[dict[str, Any]] = []
        for d in out:
            vals = d.get(block, {}).get(key)
            if isinstance(vals, list):
                for v in vals:
                    c = {**d, block: {**d[block], key: v}}
                    nxt.append(c)
            else:
                nxt.append(d)
        out = nxt
    # a species-file list is a sweep too, and the floor it sets is what the
    # hard-core cutoff reads; take the first entry only (see the docstring)
    fixed: list[dict[str, Any]] = []
    for d in out:
        pot = d.get("potential", {})
        sf = pot.get("species_file")
        if isinstance(sf, list) and sf:
            d = {**d, "potential": {**pot, "species_file": sf[0]}}
        ini = d.get("run", {}).get("initial_config")
        if isinstance(ini, list) and ini:
            d = {**d, "run": {**d["run"], "initial_config": ini[0]}}
        mu = d.get("ensemble", {}).get("mu")
        if isinstance(mu, list) and mu:
            d = {**d, "ensemble": {**d["ensemble"], "mu": mu[0]}}
        fixed.append(d)
    return fixed


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The G0.9 verdict: how many audited, how many violate, which advisories.

    `violations` is the list that decides whether the plan's 2C group stays
    empty. `advisories` is the list of configurations the Glauber engine will refuse
    but the shipped engine accepts, which is advance notice and nothing more.

    THE PLAN'S PREMISE FOR G0.9 DOES NOT HOLD ON THIS TREE, and that is the
    scan's most important finding rather than a reason not to run it. Section
    3.5 says "`sim_setup` never validates `channel_length` against box size".
    It does: `from_params` computes the body diagonal
    sqrt(L^2 + (dim-1) w^2) and refuses anything above `box - guard_cut`, with
    `guard_cut` the auxiliary radius when a short-range auxiliary is configured,
    `upper_cutoff` on the LJ path and the hard-core diameter otherwise -- i.e.
    EXACTLY the radius at which that config's sweep scores pairs, which is the
    radius the plan says an audit should use.

    So no shipped configuration can violate the guard, and the scan's verdict is
    structural rather than empirical: the group-2C item the plan budgeted for
    cannot exist. What the Glauber engine adds is not the check but two tightenings
    of it -- `upper_cutoff` instead of the auxiliary radius (moot, because the
    dual-cutoff auxiliary is refused outright), and the largest active PAIR
    FLOOR instead of the single-component diameter on a typed hard-core run,
    which is the one place the shipped guard was measured at too short a radius.
    """
    viol = [r for r in records if r["violation"]]
    adv = [r for r in records if r["advisory_refused_by_glauber"]]
    refused = [r for r in records if r.get("refused_at_config_time")]
    return {
        "audited": len(records),
        "violations": viol,
        "n_violations": len(viol),
        "advisories": adv,
        "n_advisories": len(adv),
        "refused_at_config_time": len(refused),
        "guard_is_enforced_by_sim_setup": True,
        "verdict": "all pass at rc_sel" if not viol else "VIOLATION at rc_sel",
    }


def load_sink(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL audit sink back, de-duplicated on the geometry that matters.

    A test suite builds the same channel Config dozens of times; the guard
    verdict depends only on (dimension, box, L, w, cutoff), so the distinct
    records are what a reader wants and the raw line count says nothing.
    """
    seen: dict[tuple, dict[str, Any]] = {}
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        key = (
            r.get("dimension"),
            tuple(r.get("box_dims", ())),
            r.get("channel_length"),
            r.get("channel_width"),
            r.get("at_rc_sel", {}).get("cutoff"),
            r.get("at_uc", {}).get("cutoff"),
        )
        seen.setdefault(key, r)
    return list(seen.values())

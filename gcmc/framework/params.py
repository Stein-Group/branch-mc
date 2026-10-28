"""Framework params files: guests in a rigid MOF, in the same five blocks as
every other params file, run by the same `python -m gcmc run` / launcher /
submit_serial.sbatch.

A params file is a framework run when it has a `framework` block. Physical units
throughout: the species file is an angstrom/kelvin force field (ff/*.yaml), so
T is in kelvin and fugacities and pressures in Pa. Sampler lengths are in sigma
of the first guest the species file lists, as bulk lengths are in sigma.

{
  "_comment": str,                         # OPTIONAL anywhere ("_" keys ignored)
  "potential": {"main": "lennard_jones",
                "T": float,                # kelvin
                "species_file": str},      # ff/*.yaml: sigma, epsilon, rc, shift,
                                           #   core floor and EOS all come from it.
                                           #   Paths (here and framework.cif) may be
                                           #   relative to the BRANCH_MC checkout
  "framework": {"cif": str,                # the structure; its supercell is the box
                "replicate": [int, int, int]},   # OPTIONAL; default: the smallest
                                           #   supercell with every edge >= 2 rc
             | {"empty_box": float},       # OR no host: the same guests and force
                                           #   field in an empty cubic box (angstrom),
                                           #   the bulk-fluid reference (BRANCH only)
  "engine":    {"type": "metropolis_dynamics"     # conventional single-particle GCMC
                      | "glauber_dynamics" | "suwa_todo_dynamics"   # BRANCH kernels
                      | "raspa2_conventional" | "raspa2_cbmc" | "raspa2_cfcmc",
                "move_probabilities": {"translation": float,       # OPTIONAL
                                       "exchange": float,
                                       "identity_swap": float},
                                           # probability that one move is a
                                           #   translation, an exchange (insert or
                                           #   delete / a window move / a CBMC or
                                           #   CFCMC swap) or, for a mixture, an
                                           #   identity swap. Sum 1. Default
                                           #   {0.5, 0.5} for one species and
                                           #   {0.4, 0.4, 0.2} for a mixture:
                                           #   translation and exchange attempted
                                           #   equally often, as in RASPA's usual
                                           #   inputs and every engine here.
                "kmax_margin": int,        # glauber/suwa_todo over channel only
                "trials": int,             # raspa2_cbmc only; default 10
                "wang_landau_cycles": int, # raspa2_cfcmc only; required
                "lambda_bins": int},       # raspa2_cfcmc only; default 20
  "sampler":   {"type": "metropolis"}                        # metropolis_dynamics
             | {"type": "branch", "trial_box_side": float,   # glauber/suwa_todo
                "trial_density": float}
             | {"type": "channel", "channel_length": float,
                "channel_width": float, "trial_density": float},
                                           # absent for raspa2_*
  "ensemble":  {"type": "muVT", "dimension": 3,
                "fugacity": {species: Pa}            # one entry per guest, or
                | "pressure": Pa, "y": {species: float}},   # Peng-Robinson
  "run":       {"kind": "single",
                "n_cores": int,            # launcher core budget, as for LJ sweeps
                "seed": int,
                "initial_config": "empty" | {species: int},   # default "empty"
                "sampling_window": int,    # moves per log row (raspa2_*: cycles)
                "target_cycles": float,    # total moves (raspa2_*: production cycles)
                "burn_in": int,            # raspa2_* only: initialization cycles
                "checkpoint_every": int,   # rows between checkpoints, default 10
                "sweep_label": str}        # written by the launcher
}

Sweeps: give a LIST to ensemble.fugacity (a list of maps), ensemble.pressure,
run.seed, engine.type or any sampler length, exactly as in an LJ params file.

How a BRANCH window (glauber/suwa_todo) spends its `sampling_window` moves: the
exchange share as window moves, the rest as translations and identity swaps in
their stated ratio, run as two blocks per window (each is a valid kernel, so the
composition samples the same ensemble). metropolis_dynamics draws every move
independently. RASPA2 counts in its own cycles, max(20, N) moves each.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..forcefield.species_file import ForceField, load_forcefield, peng_robinson

BRANCH_ENGINES = ("metropolis_dynamics", "glauber_dynamics", "suwa_todo_dynamics")
LADDER_ENGINES = ("glauber_dynamics", "suwa_todo_dynamics")
RASPA_ENGINES = ("raspa2_conventional", "raspa2_cbmc", "raspa2_cfcmc")
MOVE_KEYS = ("translation", "exchange", "identity_swap")


@dataclass(frozen=True)
class FrameworkRun:
    """One framework run, validated. Paths are resolved; numbers are floats."""

    params: dict[str, Any]  # the params file as given (the run folder's params.json)
    forcefield: ForceField
    T: float
    cif: Path | None  # None for an empty box
    empty_box: float | None  # angstrom
    replicate: tuple[int, int, int] | None
    engine: str
    move_probs: tuple[float, float, float]  # translation, exchange, identity swap
    sampler: dict[str, Any]  # empty for raspa2_*
    kmax_margin: int
    trials: int
    wang_landau_cycles: int
    lambda_bins: int
    fugacity_pa: dict[str, float]  # per guest
    pressure_pa: float  # total; the summed fugacity on the fugacity route
    y: dict[str, float]  # gas mole fractions
    phi: dict[str, float]  # fugacity coefficients (1 on the fugacity route)
    seed: int
    preload: dict[str, int] | None
    sampling_window: int
    target_cycles: float
    burn_in: int
    checkpoint_every: int
    sweep_label: str | None

    @property
    def guests(self) -> list[str]:
        """Guest species of this run, in the order the state point lists them."""
        return list(self.fugacity_pa)

    @property
    def is_raspa(self) -> bool:
        """True for the raspa2_* engines, which run RASPA2 instead of BRANCH_MC."""
        return self.engine in RASPA_ENGINES


REPO = Path(__file__).resolve().parents[2]


def resolve(path: Any, key: str) -> Path:
    """A path as written: absolute, or relative to where the run is launched,
    or else relative to the BRANCH_MC checkout (so `ff/forcefield_A_shift.yaml`
    and `structures/MgMOF-74.cif` work from any run directory). The run id
    hashes file contents, so where a file was found never changes the id."""
    p = Path(str(_scalar(path, key))).expanduser()
    for cand in (p, REPO / p):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(
        f"{key} {p} not found here or in the BRANCH_MC checkout ({REPO})"
    )


def is_framework(params: dict[str, Any]) -> bool:
    """A params file is a framework run exactly when it has a `framework` block."""
    return "framework" in params


def _section(params: dict[str, Any], name: str) -> dict[str, Any]:
    sec = params.get(name, {})
    if not isinstance(sec, dict):
        raise ValueError(f"{name} must be a JSON object")
    return {k: v for k, v in sec.items() if not str(k).startswith("_")}


def _check(
    sec: dict[str, Any], name: str, allowed: set[str], required: set[str]
) -> None:
    unknown = sorted(set(sec) - allowed)
    if unknown:
        raise ValueError(
            f"{name}.{unknown[0]} is not a framework-run key; {name} takes "
            f"{sorted(allowed)}"
        )
    missing = sorted(required - set(sec))
    if missing:
        raise ValueError(f"{name}.{missing[0]} is required for a framework run")


def _scalar(v: Any, key: str) -> Any:
    if isinstance(v, (list, tuple)):
        raise ValueError(
            f"{key} is a list: a sweep. Launch with `python -m gcmc.run.launcher` "
            "(or submit_serial.sbatch), which expands it into one run per value"
        )
    return v


def _move_probs(eng: dict[str, Any], n_species: int) -> tuple[float, float, float]:
    raw = eng.get("move_probabilities")
    if raw is None:
        return (0.5, 0.5, 0.0) if n_species == 1 else (0.4, 0.4, 0.2)
    if not isinstance(raw, dict) or set(raw) - set(MOVE_KEYS):
        raise ValueError(
            f"engine.move_probabilities is a map over {list(MOVE_KEYS)}, got {raw!r}"
        )
    t, e, s = (float(raw.get(k, 0.0)) for k in MOVE_KEYS)
    if min(t, e, s) < 0 or abs(t + e + s - 1.0) > 1e-9:
        raise ValueError(
            f"engine.move_probabilities must be non-negative and sum to 1, got {raw}"
        )
    if e == 0.0:
        raise ValueError(
            "engine.move_probabilities.exchange is 0: nothing would change N, and "
            "a muVT run would silently be canonical"
        )
    if n_species == 1 and s > 0:
        raise ValueError("identity_swap needs a mixture; this run has one guest")
    return (t, e, s)


def parse(params: dict[str, Any]) -> FrameworkRun:
    """Validate a framework params file; every error names the key to fix."""
    allowed_top = {"potential", "framework", "engine", "sampler", "ensemble", "run"}
    extra = sorted(k for k in params if k not in allowed_top and not k.startswith("_"))
    if extra:
        raise ValueError(
            f"{extra[0]} is not a framework-run block; a framework params file has "
            f"{sorted(allowed_top)}"
        )
    pot = _section(params, "potential")
    _check(
        pot, "potential", {"main", "T", "species_file"}, {"main", "T", "species_file"}
    )
    if pot["main"] != "lennard_jones":
        raise ValueError("potential.main must be 'lennard_jones' for a framework run")
    ff = load_forcefield(resolve(pot["species_file"], "potential.species_file"))
    if ff.units == "reduced":
        raise ValueError(
            "potential.species_file is a reduced-units file; a framework run needs "
            "the physical (angstrom/kelvin) force field, e.g. ff/forcefield_A_shift.yaml"
        )
    T = float(_scalar(pot["T"], "potential.T"))

    fw = _section(params, "framework")
    _check(fw, "framework", {"cif", "replicate", "empty_box"}, set())
    if ("cif" in fw) == ("empty_box" in fw):
        raise ValueError("framework takes exactly one of cif or empty_box")
    cif = resolve(fw["cif"], "framework.cif") if "cif" in fw else None
    empty_box = float(fw["empty_box"]) if "empty_box" in fw else None
    if empty_box is not None and "replicate" in fw:
        raise ValueError("framework.replicate applies to a cif, not an empty_box")
    rep = fw.get("replicate")
    if rep is not None:
        if not (isinstance(rep, (list, tuple)) and len(rep) == 3):
            raise ValueError(f"framework.replicate is [nx, ny, nz], got {rep!r}")
        rep = tuple(int(r) for r in rep)

    ens = _section(params, "ensemble")
    _check(ens, "ensemble", {"type", "dimension", "fugacity", "pressure", "y"}, set())
    if ens.get("type", "muVT") != "muVT" or int(ens.get("dimension", 3)) != 3:
        raise ValueError("a framework run is ensemble.type 'muVT', dimension 3")
    if ("fugacity" in ens) == ("pressure" in ens):
        raise ValueError(
            "give the state point as exactly one of ensemble.fugacity "
            "({species: Pa}) or ensemble.pressure (Pa) with ensemble.y"
        )
    if "fugacity" in ens:
        if "y" in ens:
            raise ValueError("ensemble.y goes with ensemble.pressure, not fugacity")
        fug = _scalar(ens["fugacity"], "ensemble.fugacity")
        if not isinstance(fug, dict) or not fug:
            raise ValueError("ensemble.fugacity is a map {species: Pa}")
        fugacity = {str(k): float(v) for k, v in fug.items()}
        P = sum(fugacity.values())
        y = {k: v / P for k, v in fugacity.items()}
        phi = dict.fromkeys(fugacity, 1.0)
    else:
        P = float(_scalar(ens["pressure"], "ensemble.pressure"))
        yraw = ens.get("y")
        if not isinstance(yraw, dict) or not yraw:
            raise ValueError(
                "ensemble.pressure needs ensemble.y {species: mole fraction}"
            )
        y = {str(k): float(v) for k, v in yraw.items()}
        if abs(sum(y.values()) - 1.0) > 1e-9:
            raise ValueError(f"ensemble.y must sum to 1, got {y}")
        pr = peng_robinson(ff, T, P, y)
        phi = {k: float(pr.phi[k]) for k in y}
        fugacity = {k: float(pr.fugacity_pa[k]) for k in y}
    unknown = sorted(set(fugacity) - set(ff.guests))
    if unknown:
        raise ValueError(f"{unknown[0]} is not a guest of {ff.path} ({ff.guests})")
    # species in the force field's own order, whatever order the file (or a
    # launcher snapshot, which sorts keys) lists them in: columns, folder labels
    # and RASPA2 component numbers then never depend on JSON key order
    order = [g for g in ff.guests if g in fugacity]
    fugacity = {g: fugacity[g] for g in order}
    y = {g: y[g] for g in order}
    phi = {g: phi[g] for g in order}
    if min(fugacity.values()) <= 0:
        raise ValueError("every fugacity / mole fraction must be positive")

    eng = _section(params, "engine")
    etype = str(_scalar(eng.get("type", "metropolis_dynamics"), "engine.type"))
    if etype not in BRANCH_ENGINES + RASPA_ENGINES:
        raise ValueError(
            f"engine.type {etype!r}: one of {list(BRANCH_ENGINES + RASPA_ENGINES)}"
        )
    allowed_eng = {"type", "move_probabilities"}
    if etype in LADDER_ENGINES:
        allowed_eng |= {"kmax_margin"}
    elif etype == "raspa2_cbmc":
        allowed_eng |= {"trials"}
    elif etype == "raspa2_cfcmc":
        allowed_eng |= {"wang_landau_cycles", "lambda_bins"}
    _check(
        eng,
        "engine",
        allowed_eng,
        {"wang_landau_cycles"} if etype == "raspa2_cfcmc" else set(),
    )
    probs = _move_probs(eng, len(fugacity))
    trials = int(eng.get("trials", 10)) if etype == "raspa2_cbmc" else 1
    if etype == "raspa2_cbmc" and trials < 2:
        raise ValueError(
            "engine.trials >= 2 for raspa2_cbmc (1 is raspa2_conventional)"
        )

    smp = _section(params, "sampler")
    if etype in RASPA_ENGINES:
        if empty_box is not None:
            raise ValueError("the RASPA2 arms need a framework.cif")
        if smp:
            raise ValueError(
                f"engine.type {etype} is a RASPA2 run and takes no sampler block"
            )
    elif etype == "metropolis_dynamics":
        _check(smp, "sampler", {"type"}, set())
        if smp.get("type", "metropolis") != "metropolis":
            raise ValueError(
                "engine.type metropolis_dynamics is conventional GCMC: sampler.type "
                "'metropolis' (BRANCH and channel windows need glauber_dynamics or "
                "suwa_todo_dynamics)"
            )
        smp = {"type": "metropolis"}
    else:
        stype = smp.get("type")
        if stype == "branch":
            _check(
                smp,
                "sampler",
                {"type", "trial_box_side", "trial_density"},
                {"type", "trial_box_side", "trial_density"},
            )
        elif stype == "channel":
            _check(
                smp,
                "sampler",
                {"type", "channel_length", "channel_width", "trial_density"},
                {"type", "channel_length", "channel_width", "trial_density"},
            )
        else:
            raise ValueError(
                f"engine.type {etype} needs sampler.type 'branch' or 'channel', "
                f"got {stype!r}"
            )
        smp = {
            k: (v if k == "type" else float(_scalar(v, f"sampler.{k}")))
            for k, v in smp.items()
        }
    if "kmax_margin" in eng and smp.get("type") != "channel":
        raise ValueError("engine.kmax_margin applies over sampler.type 'channel' only")

    run = _section(params, "run")
    _check(
        run,
        "run",
        {
            "kind",
            "n_cores",
            "seed",
            "initial_config",
            "sampling_window",
            "target_cycles",
            "burn_in",
            "checkpoint_every",
            "sweep_label",
        },
        {"seed", "target_cycles"},
    )
    if run.get("kind", "single") != "single":
        raise ValueError("a framework run is run.kind 'single' (one core each)")
    if "burn_in" in run and etype not in RASPA_ENGINES:
        raise ValueError(
            "run.burn_in applies to RASPA2 runs (its initialization cycles); a "
            "BRANCH_MC run keeps every row and the analysis drops its own burn-in"
        )
    start = run.get("initial_config", "empty")
    preload = None
    if isinstance(start, dict):
        preload = {str(k): int(v) for k, v in start.items()}
        bad = sorted(set(preload) - set(fugacity))
        if bad:
            raise ValueError(
                f"run.initial_config names {bad[0]}, not a guest of this run"
            )
    elif start != "empty":
        raise ValueError(
            "run.initial_config is 'empty' or a map {species: count} for a "
            "framework run (guests placed at random in the pores)"
        )
    window = int(run.get("sampling_window", 1000))
    if window < 1:
        raise ValueError("run.sampling_window must be >= 1")

    return FrameworkRun(
        params=params,
        forcefield=ff,
        T=T,
        cif=cif,
        empty_box=empty_box,
        replicate=rep,
        engine=etype,
        move_probs=probs,
        sampler=smp,
        kmax_margin=int(eng.get("kmax_margin", 0)),
        trials=trials,
        wang_landau_cycles=int(eng.get("wang_landau_cycles", 0)),
        lambda_bins=int(eng.get("lambda_bins", 20)),
        fugacity_pa=fugacity,
        pressure_pa=P,
        y=y,
        phi=phi,
        seed=int(_scalar(run["seed"], "run.seed")),
        preload=preload,
        sampling_window=window,
        target_cycles=float(run["target_cycles"]),
        burn_in=int(run.get("burn_in", 0)),
        checkpoint_every=int(run.get("checkpoint_every", 10)),
        sweep_label=run.get("sweep_label"),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _canon(v: Any) -> Any:
    if isinstance(v, float):
        return format(v, ".12g")
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return v


def identity(fr: FrameworkRun) -> dict[str, Any]:
    """Everything that makes the chain what it is, files by content hash, and
    nothing operational (target length, checkpoint cadence, cores, label). Two
    runs with equal identity share a folder and the second resumes the first."""
    return _canon(
        {
            "T": fr.T,
            "forcefield": _sha(fr.forcefield.path),
            "cif": _sha(fr.cif) if fr.cif is not None else None,
            "empty_box": fr.empty_box,
            "replicate": list(fr.replicate) if fr.replicate else None,
            "engine": fr.engine,
            "move_probs": list(fr.move_probs),
            "sampler": fr.sampler,
            "kmax_margin": fr.kmax_margin,
            "trials": fr.trials,
            "wang_landau_cycles": fr.wang_landau_cycles,
            "lambda_bins": fr.lambda_bins,
            "fugacity_pa": fr.fugacity_pa,
            "seed": fr.seed,
            "preload": fr.preload,
            "sampling_window": fr.sampling_window,
            "burn_in": fr.burn_in,
        }
    )


def run_id(fr: FrameworkRun) -> str:
    """12-hex-digit SHA-1 of `identity`: the folder name's stable half."""
    blob = json.dumps(identity(fr), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def run_dir(fr: FrameworkRun, root: str | Path) -> Path:
    """`<root>/<id>__<label>`, the layout of every other run: the launcher's
    sweep label names the swept value, a lone run is `single`."""
    tag = (fr.sweep_label or "single").replace("-", "m").replace(".", "p")
    return Path(root) / f"{run_id(fr)}__{tag}"

"""Shared setup for the neighbour-refactor harness (bench + reference trajectories).

Nothing here is imported by the package; it exists so `bench_neighbors.py` and
`gen_reference_traj.py` agree on the configurations they measure and freeze.

Test-only code, so scipy is used freely.
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from gcmc import config_specs
from gcmc.engine import metropolis_dynamics
from gcmc.run import sim_setup
from gcmc.system.state import ChainState

REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref")

# Frozen engines, for benchmarking against the current one in the same process.
# The container's clock drifts between runs, so a speedup measured against a
# stored number is not a measurement; measured back to back, it is.
import _frozen

ENGINES = {"current": "gcmc", "stageA": "gcmcA", "baseline": "gcmc0"}


#: Which tagged artifact each frozen engine name lives in (R1). The packages
#: themselves are no longer in the tree; only their trajectories are.
_FROZEN = {"gcmcA": _frozen.ENGINEA, "gcmc0": _frozen.ENGINE0}


def load_engine(name):
    """(sim_setup, metropolis_dynamics, ChainState) of a frozen or current package.

    A frozen engine is retrieved from its tag rather than the tree. Callers that
    cannot tolerate a skip should check `_frozen.root(...)` first; everything in
    the characterisation tier is skipped as a module when the artifact is absent.

    Raises
    ------
    _frozen.OrigUnavailable
        If the named engine's worktree is not available, carrying the two
        commands that would create it.
    """
    import importlib

    pkg = ENGINES[name]
    if pkg != "gcmc":
        _frozen.add_to_path(_FROZEN[pkg])
    importlib.import_module(pkg)
    return (
        importlib.import_module(pkg + ".sim_setup"),
        importlib.import_module(pkg + ".metropolis_dynamics"),
        importlib.import_module(pkg + ".state").ChainState,
    )


def lattice_seed(box, n, dim, rng, min_dist=1.0, jitter=0.12):
    """`n` non-overlapping centres on a jittered simple-cubic lattice in [0, box).

    The lattice pitch is chosen so the jittered spheres cannot touch: pitch is the
    smallest that fits `n` sites, and the jitter is capped at (pitch - min_dist)/2
    per axis so the worst case (two neighbours displaced toward each other on the
    same axis) still clears `min_dist`.
    """
    per = int(math.ceil(n ** (1.0 / dim)))
    pitch = box / per
    if pitch <= min_dist:
        raise ValueError(f"cannot fit {n} spheres of diameter {min_dist} in box {box}")
    amp = min(jitter, 0.5 * (pitch - min_dist) / math.sqrt(dim)) * 0.999
    grids = np.meshgrid(*([np.arange(per)] * dim), indexing="ij")
    sites = np.stack([g.ravel() for g in grids], axis=1) * pitch + 0.5 * pitch
    sel = rng.permutation(len(sites))[:n]
    pos = sites[sel] + rng.uniform(-amp, amp, size=(n, dim))
    return np.mod(pos, box)


def cs_mu(rho, radius=0.5):
    """Carnahan-Starling chemical potential (reduced, Lambda = 1) for hard spheres
    at number density `rho`, used only to keep a benchmark near stationary N."""
    eta = rho * (4.0 / 3.0) * math.pi * radius**3
    mu_ex = (8.0 * eta - 9.0 * eta**2 + 3.0 * eta**3) / (1.0 - eta) ** 3
    return math.log(rho) + mu_ex


def _back_translate(params):
    """Rewrite params into the config surface a FROZEN engine understands.

    A frozen engine is a fixed build of the package from a tag. It cannot parse
    a config key introduced after it was frozen, so any change to the config
    surface would otherwise silently break every parity oracle -- and the
    failure looks like a physics regression, which is the worst possible
    disguise for a spelling change.

    So surface changes are declared HERE, one entry each, and the frozen engine
    is handed the spelling it knows. The comparison stays honest because only
    the spelling is translated: both engines run the same Hamiltonian, the same
    move set and the same seed.

    An entry may only ever be a RENAME. If a change alters what the engine
    computes, the oracle must be regenerated rather than translated, and the
    commit must say so.

    Entries
    -------
    R16 (2026-08)
        The cyclic wall-to-wall rod became `sampler.type = "channel_cyclic"`.
        Before that it was `sampler.type = "channel"` plus a string
        `channel_length`. Identical Config apart from the `sampler` string,
        hence identical chain.

        The emitted spelling is "full", NOT "cyclic". engineA accepts only
        ("full", "box", "full_box", "span") -- "cyclic" was added later, as the
        preferred name, and a frozen engine has no idea it exists. This is the
        general rule for this function: translate into the vocabulary the
        FROZEN engine had, which is not necessarily the most recent spelling
        the current engine deprecated.

    The engine block (2026-09)
        `depth`, `full_repacking` and `kmax_margin` moved out of `sampler` into
        a top-level `engine` block, whose `type` is the dynamics that used to be
        `sampler.engine`. A frozen engine has no `engine` block at all and reads
        those keys where they used to be, so they go back. A RENAME and nothing
        more: the same values reach the same Config fields, and an absent
        dynamics meant metropolis_dynamics before the move exactly as it does
        after, so the key is dropped rather than spelled out.
    """
    import copy

    out = copy.deepcopy(params)
    smp = out.get("sampler", {})
    if smp.get("type") == "channel_cyclic":
        smp["type"] = "channel"
        smp["channel_length"] = "full"
    eng = out.pop("engine", None)
    if eng:
        for key, value in eng.items():
            if key == "type":
                # engineA's default is the same dynamics, so the default needs
                # no spelling; anything else it would not recognise at all.
                if value != "metropolis_dynamics":
                    smp["engine"] = value
            else:
                smp[key] = value
        out["sampler"] = smp
    return out


def build(params, seed, positions=None, matrix=None, engine=None):
    """cfg / potential / weigher / state for a params dict."""
    ss, eng, CS = (
        (sim_setup, metropolis_dynamics, ChainState) if engine is None else engine
    )
    if engine is not None:
        params = _back_translate(params)
    cfg = ss.from_params(params)
    if matrix is not None:
        cfg = dataclasses.replace(cfg, matrix_positions=np.asarray(matrix, float))
    pot = cfg.build_potential()
    wgh = eng.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = CS.fresh(cfg.dimension, seed=seed)
    if positions is not None:
        st.positions = np.asarray(positions, float)
    return cfg, pot, wgh, st


# ---------------------------------------------------------------------------
# Reference modes. Each entry: (params, seed, n_seed, matrix_n).
# Small boxes and short windows: these are correctness oracles, not benchmarks.
# ---------------------------------------------------------------------------

_HS = {"main": "hard_core"}
_LJ = {"main": "lennard_jones", "T": 0.85}


#: Keys that moved from the `sampler` block to the top-level `engine` block.
#: `engine` itself is in the list because it moved from a sampler key naming the
#: dynamics to that block's own `type`.
ENGINE_KEYS = ("engine", "depth", "full_repacking", "kmax_margin")


def split_engine(sampler):
    """(sampler block, engine block), splitting `sampler` on `ENGINE_KEYS`.

    `sampler` says WHERE the trial cloud comes from; `engine` says what is done
    with it, and the group size, the repacking switch and the ladder truncation
    are all the second question. `from_params` refuses the old spelling by name
    rather than aliasing it, so the split has to happen somewhere -- and it
    happens here, in the one constructor every mode in this file goes through,
    rather than in two hundred literals. `_back_translate` does the reverse for
    a frozen engine that only knows the old spelling.
    """
    smp = dict(sampler)
    eng = {}
    for key in ENGINE_KEYS:
        if key in smp:
            eng["type" if key == "engine" else key] = smp.pop(key)
    return smp, eng


def _p(potential, sampler, ensemble, run=None):
    smp, eng = split_engine(sampler)
    out = {
        "potential": dict(potential),
        "sampler": smp,
        "ensemble": dict(ensemble),
        "run": dict(run or {}),
    }
    if eng:
        out["engine"] = eng
    return out


REF_MODES = {
    "hs_branch": {
        "params": _p(
            _HS,
            {"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 3.0},
            {"mu": 3.0, "dimension": 3, "box": 10.0},
            {"sampling_window": 60},
        ),
        "seed": 11,
        "n_seed": 300,
    },
    "hs_rosenbluth": {
        "params": _p(
            _HS,
            {"type": "rosenbluth", "depth": 0, "trials": 30, "trial_box_side": 3.0},
            {"mu": 3.0, "dimension": 3, "box": 10.0},
            {"sampling_window": 60},
        ),
        "seed": 12,
        "n_seed": 300,
    },
    "hs_channel": {
        "params": _p(
            _HS,
            {
                "type": "channel",
                "depth": 1,
                "trial_density": 3.0,
                "channel_length": 6.0,
                "channel_width": 1.5,
            },
            {"mu": 3.0, "dimension": 3, "box": 12.0},
            {"sampling_window": 60},
        ),
        "seed": 13,
        "n_seed": 500,
    },
    "hs_channel_full": {
        "params": _p(
            _HS,
            {
                "type": "channel_cyclic",
                "depth": 1,
                "trial_density": 1.0,
                # R16: this mode was always the CYCLIC wall-to-wall rod --
                # "full" was one of five aliases for it. The mode name and the
                # frozen trajectory are unchanged; only the spelling that
                # selects it is. The Config it builds is identical apart from
                # `sampler`, so the chain and the npz oracle still match.
                "channel_width": 1.5,
            },
            {"mu": 3.0, "dimension": 3, "box": 12.0},
            {"sampling_window": 40},
        ),
        "seed": 14,
        "n_seed": 500,
    },
    "hs_full_repacking": {
        "params": _p(
            _HS,
            {
                "type": "channel",
                "full_repacking": True,
                "trial_density": 3.0,
                "channel_length": 6.0,
                "channel_width": 1.5,
            },
            {"mu": 3.0, "dimension": 3, "box": 12.0},
            {"sampling_window": 40},
        ),
        "seed": 15,
        "n_seed": 400,
    },
    "lj": {
        "params": _p(
            _LJ,
            {"type": "rosenbluth", "depth": 0, "trials": 30, "trial_box_side": 3.0},
            {"mu": -3.2, "dimension": 3, "box": 10.0},
            {"sampling_window": 40},
        ),
        "seed": 16,
        "n_seed": 400,
    },
    "ljbranch": {
        "params": _p(
            _LJ,
            {"type": "branch", "depth": 1, "trials": 20, "trial_box_side": 3.0},
            {"mu": -3.2, "dimension": 3, "box": 10.0},
            {"sampling_window": 30},
        ),
        "seed": 17,
        "n_seed": 400,
    },
    "lj_channel": {
        "params": _p(
            _LJ,
            {
                "type": "channel",
                "depth": 1,
                "trial_density": 1.2,
                "channel_length": 5.0,
                "channel_width": 1.2,
            },
            {"mu": -3.2, "dimension": 3, "box": 12.0},
            {"sampling_window": 30},
        ),
        "seed": 18,
        "n_seed": 700,
    },
    "lj_channel_full": {
        "params": _p(
            _LJ,
            {
                "type": "channel_cyclic",
                "depth": 1,
                "trial_density": 0.6,
                # R16: this mode was always the CYCLIC wall-to-wall rod --
                # "full" was one of five aliases for it. The mode name and the
                # frozen trajectory are unchanged; only the spelling that
                # selects it is. The Config it builds is identical apart from
                # `sampler`, so the chain and the npz oracle still match.
                "channel_width": 1.2,
            },
            {"mu": -3.2, "dimension": 3, "box": 12.0},
            {"sampling_window": 20},
        ),
        "seed": 19,
        "n_seed": 700,
    },
    "hs_matrix": {
        "params": _p(
            _HS,
            {"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 3.0},
            {"mu": 4.0, "dimension": 3, "box": 10.0},
            {"sampling_window": 40},
        ),
        "seed": 20,
        "n_seed": 150,
        "matrix_n": 60,
    },
    "hs_pore": {
        "params": _p(
            _HS,
            {"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 3.0},
            {
                "mu": 4.0,
                "dimension": 3,
                "box": 12.0,
                "type": "confined",
                "pore": {"geometry": "cylinder", "R": 3.0},
            },
            {"sampling_window": 40},
        ),
        "seed": 21,
        "n_seed": 0,
    },
    "hs_empty": {
        "params": _p(
            _HS,
            {"type": "branch", "depth": 1, "trials": 30, "trial_box_side": 3.0},
            {"mu": 3.0, "dimension": 3, "box": 8.0},
            {"sampling_window": 60},
        ),
        "seed": 22,
        "n_seed": 0,
    },
    "hd2d": {
        "params": _p(
            _HS,
            {"type": "rosenbluth", "depth": 0, "trials": 30, "trial_box_side": 3.0},
            {"mu": 3.0, "dimension": 2, "box": 20.0},
            {"sampling_window": 60},
        ),
        "seed": 23,
        "n_seed": 200,
    },
    "metropolis": {
        "params": _p(
            _HS,
            {"type": "metropolis"},
            {"mu": 3.0, "dimension": 3, "box": 10.0},
            {"sampling_window": 60},
        ),
        "seed": 24,
        "n_seed": 300,
    },
}

REF_WINDOWS = 3


def make_mode(name, engine=None):
    """(cfg, potential, weigher, state) for a reference mode, freshly seeded."""
    spec = REF_MODES[name]
    p = spec["params"]
    dim = p["ensemble"]["dimension"]
    box = p["ensemble"]["box"]
    rng = np.random.default_rng(1000 + spec["seed"])

    matrix = None
    if spec.get("matrix_n"):
        all_pts = lattice_seed(box, spec["n_seed"] + spec["matrix_n"], dim, rng)
        matrix = all_pts[: spec["matrix_n"]]
        pos = all_pts[spec["matrix_n"] :]
    elif spec["n_seed"] == 0:
        pos = np.empty((0, dim))
    else:
        pos = lattice_seed(box, spec["n_seed"], dim, rng)

    if name == "hs_pore":
        # confined: the seeder must place centres inside the cylinder
        cfg0 = (sim_setup if engine is None else engine[0]).from_params(p)
        pts = lattice_seed(box, 400, dim, rng)
        r = np.hypot(pts[:, 0] - box / 2, pts[:, 1] - box / 2)
        pos = pts[r <= cfg0.pore_R - cfg0.radius][:120]

    return build(p, spec["seed"], positions=pos, matrix=matrix, engine=engine)

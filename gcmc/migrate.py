"""One-shot migration of a legacy run into the new layout.

A legacy run is a single .extxyz trajectory in run-files/: its parameters live
in the filename and in the last frame's info dict (mc_cycle, the run signature),
its positions are the final configuration, and its cell side is the box. This
reads that frame, reconstructs a tagged params.json, and writes a new-format
checkpoint into the identity directory, so that

    python -m gcmc.migrate run-files/<run>.extxyz --temp 1.0
    python -m gcmc run runs/<id>__mu.../params.json

resumes the run from where it stopped.

What the legacy file does not record has to be supplied: the temperature (a CLI
argument in the original, not stored), and for a non-aux run whether it was hard
spheres or Lennard-Jones (the signature never distinguished them). Defaults are
the original's common case (T given explicitly, hard spheres for non-aux); pass
--potential lj / --temp / --g3-csv to override. The resumed run starts a fresh
random stream, exactly as any checkpoint resume does; the equilibrium it samples
does not depend on stream continuity.
"""

from __future__ import annotations
import argparse
import os
import re
import numpy as np

from . import config, identity, io
from .state import zero_counters

_TOKENS = [
    ("chem_pot",      r'mu_(-?\d+(?:\.\d+)?)_',     float),
    ("cavity_radius", r'_r_(-?\d+(?:\.\d+)?)_',      float),
    ("trial_density", r'rhok_(-?\d+(?:\.\d+)?)_',    float),
    ("no_trials",     r'_k_(\d+)_',                  int),
    ("two_dim",       r'_2d_(True|False)_',          lambda s: s == "True"),
    ("aux_pot",       r'_aux_pot_(True|False)_',     lambda s: s == "True"),
    ("gamma_max",     r'gammamax_(-?\d+(?:\.\d+)?)_', float),
    ("delta_eta",     r'deltaeta_(-?\d+(?:\.\d+)?)_', float),
]


def _parse_name(path):
    b = os.path.basename(path)
    out = {}
    for key, pat, cast in _TOKENS:
        m = re.search(pat, b)
        if m:
            out[key] = cast(m.group(1))
    out.setdefault("aux_pot", False)
    return out


def legacy_to_params(extxyz_path, *, temp, potential=None, depth=0, branch=False,
                     use_depth=True, move_factor=0.25, g3_csv=None):
    """Read a legacy trajectory and return (params_dict, positions, step)."""
    from ase.io import read
    frame = read(extxyz_path, index=-1)
    info = dict(frame.info)
    name = _parse_name(extxyz_path)

    def pick(key, default=None):
        return info.get(key, name.get(key, default))

    two_dim = bool(pick("two_dim", False))
    dim = 2 if two_dim else 3
    aux = bool(pick("aux_pot", False))
    side = float(frame.cell[0][0]) if frame.cell is not None else None
    if not side:
        raise ValueError(f"{extxyz_path}: no cell side in the trajectory; cannot "
                         "recover the box. Re-save with a cell or pass it in.")

    mu = float(pick("chem_pot"))
    cav = float(pick("cavity_radius"))
    k = int(pick("no_trials", pick("no_of_trial_molecules")))
    step = int(info.get("mc_cycle", 0))

    pot_type = potential or ("aux" if aux else "hard_sphere")
    pot_block = {"type": pot_type}
    if pot_type == "lj":
        pot_block["T"] = temp
    elif temp != 1.0:
        import warnings
        warnings.warn(f"{pot_type} runs are fixed at T=1.0; ignoring --temp {temp}")
    params = {
        "potential": pot_block,
        "ensemble": {"mu": mu, "dimension": dim, "box": side},
        "trials": {"k": k, "cavity_radius": cav},
        "algorithm": {"depth": depth, "branch": branch, "use_depth": use_depth},
        "move": {"move_factor": move_factor},
        "execution": {},
    }
    if pot_type == "aux":
        if "gamma_max" in info or "gamma_max" in name:
            pot_block["gamma_max"] = float(pick("gamma_max"))
        if "delta_eta" in info or "delta_eta" in name:
            pot_block["delta_eta"] = float(pick("delta_eta"))
        if not two_dim:
            if not g3_csv:
                raise ValueError("aux 3D run needs --g3-csv (the angular table "
                                 "path); it is not stored in the trajectory.")
            pot_block["g3_csv"] = g3_csv

    positions = np.asarray(frame.positions, float)
    positions = positions[:, :2] if two_dim else positions[:, :3]
    return params, positions, step


def migrate(extxyz_path, root="runs", *, temp, seed=0, **opts):
    """Convert one legacy trajectory and write a resumable checkpoint. Returns
    the run directory."""
    params, positions, step = legacy_to_params(extxyz_path, temp=temp, **opts)
    cfg = config.from_params(params)
    paths = identity.run_paths(cfg, root)
    io.ensure_run_dir(paths, params)

    gen = np.random.default_rng(seed)
    io.save_checkpoint(paths, positions=positions, step=step, rng=io.rng_state(gen),
                       counters=zero_counters(), params=params)
    return paths.directory


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m gcmc.migrate")
    ap.add_argument("trajectory", help="legacy .extxyz file")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="temperature (not stored in legacy files; 1.0 for "
                         "hard-sphere/aux, set the real value for LJ)")
    ap.add_argument("--potential", choices=["hard_sphere", "lj", "aux"], default=None,
                    help="override; default is aux if the name says so, else hard_sphere")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--move-factor", type=float, default=0.25)
    ap.add_argument("--g3-csv", default=None, help="angular table path for aux 3D")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    d = migrate(a.trajectory, root=a.root, temp=a.temp, potential=a.potential,
                depth=a.depth, move_factor=a.move_factor, g3_csv=a.g3_csv, seed=a.seed)
    print(f"migrated -> {d}")
    print(f"resume with: python -m gcmc run {os.path.join(d, 'params.json')} --root {a.root}")


if __name__ == "__main__":
    main()

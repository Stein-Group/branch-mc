"""Diff Config-derived constants against the original globals, per mode.

  python check_config.py <mode>

Builds the params dict equivalent to the mode's CLI and checks every derived
constant matches what the original sets at import. The box is left out of params
so the default_side preset is exercised and compared to the original's
side_of_box.
"""

import math
import os
import sys

import numpy as np

MODE = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
from _frozen import add_orig_to_path

ORIG = str(add_orig_to_path())  # R1: resolved from $ORIG_ROOT or the tag

ARGV = {
    "lj3d": [
        "--mu",
        "-4",
        "--rho_k",
        "400",
        "--d",
        "0",
        "--r",
        "1.5",
        "--LJ_pot",
        "--T",
        "0.85",
    ],
    "hs2d": [
        "--mu",
        "4",
        "--rho_k",
        "400",
        "--d",
        "0",
        "--r",
        "1.5",
        "--2d",
        "--T",
        "1.0",
    ],
    "hs3d": ["--mu", "4", "--rho_k", "400", "--d", "0", "--r", "1.5", "--T", "1.0"],
    "aux2d": [
        "--mu",
        "4",
        "--rho_k",
        "400",
        "--d",
        "0",
        "--r",
        "1.5",
        "--2d",
        "--aux_pot",
        "--T",
        "1.0",
    ],
    "aux3d": [
        "--mu",
        "4",
        "--rho_k",
        "400",
        "--d",
        "0",
        "--r",
        "1.5",
        "--aux_pot",
        "--T",
        "1.0",
    ],
}

PARAMS = {
    "lj3d": {
        "potential": {"main": "lennard_jones", "T": 0.85},
        "ensemble": {"mu": -4.0, "dimension": 3},
        "sampler": {
            "type": "rosenbluth",
            "trial_density": 400,
            "trial_box_side": 3.0,
            "depth": 0,
        },
        "run": {"lower_cutoff": 1.0},
    },  # original floors LJ at the diameter
    "hs2d": {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 4.0, "dimension": 2},
        "sampler": {
            "type": "rosenbluth",
            "trial_density": 400,
            "trial_box_side": 3.0,
            "depth": 0,
        },
    },
    "hs3d": {
        "potential": {"main": "hard_core"},
        "ensemble": {"mu": 4.0, "dimension": 3},
        "sampler": {
            "type": "rosenbluth",
            "trial_density": 400,
            "trial_box_side": 3.0,
            "depth": 0,
        },
    },
    "aux2d": {
        "potential": {"main": "hard_core"},
        "auxiliary": {"type": "pmf", "eta": 0.69},
        "ensemble": {"mu": 4.0, "dimension": 2},
        "sampler": {
            "type": "rosenbluth",
            "trial_density": 400,
            "trial_box_side": 3.0,
            "depth": 0,
        },
    },
    "aux3d": {
        "potential": {"main": "hard_core"},
        "auxiliary": {
            "type": "pmf",
            "g3_csv": os.path.join(ORIG, "g3_analysis_data.csv"),
        },
        "ensemble": {"mu": 4.0, "dimension": 3},
        "sampler": {
            "type": "rosenbluth",
            "trial_density": 400,
            "trial_box_side": 3.0,
            "depth": 0,
        },
    },
}

sys.argv = ["orig"] + ARGV[MODE]
if MODE == "aux3d":
    os.chdir(ORIG)
import cav_n_d_3d_g3_LJ as orig

sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gcmc.run import sim_setup as config

cfg = config.from_params(PARAMS[MODE])


def close(a, b, tag):
    ok = (
        (a == b)
        if isinstance(a, (int, bool)) and isinstance(b, (int, bool))
        else math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-15)
    )
    if not ok:
        raise AssertionError(f"[FAIL] {tag}: mine={a} orig={b}")
    print(f"  ok  {tag} = {a}")


close(cfg.beta, orig.BETA, "beta")
close(cfg.chem_pot, orig.chem_pot, "chem_pot")
close(int(cfg.two_dim), int(orig.two_dim_flag), "two_dim")
close(cfg.box, orig.side_of_box, "box (preset)")
close(cfg.radius, orig.radius_of_disk, "radius")
close(cfg.diameter, orig.diameter_of_disk, "diameter")
close(cfg.r_squared, orig.r_squared, "r_squared")
close(cfg.no_of_trial_molecules, orig.no_of_trial_molecules, "no_of_trial_molecules")
close(cfg.trial_density, orig.trial_density, "trial_density")
close(cfg.no_neigh, orig.no_neigh, "no_neigh")
close(cfg.upper_cutoff, orig.upper_cutoff, "upper_cutoff")
close(cfg.volume_of_molecule, orig.volume_of_molecule, "volume_of_molecule")
close(cfg.total_volume, orig.Total_volume, "total_volume")
close(cfg.fugacity, orig.fugacity, "fugacity")
close(cfg.depth, orig.depth, "depth")
close(cfg.move_factor, orig.move_factor, "move_factor")
close(cfg.sampling_window, orig.sampling_window, "sampling_window")
close(cfg.burn_in, orig.burn_in, "burn_in")
if MODE == "lj3d":
    close(cfg.lj_tail_pref, orig.lj_tail_pref, "lj_tail_pref")
for i in range(1, 10):
    close(cfg.gamma_dict[i], orig.gamma_dict[i], f"gamma_dict[{i}]")

# the potential builds and is callable
pot = cfg.build_potential()
if MODE != "reweight":
    r = np.full((3, 4), 1.5)
    _ = pot.weight_bulk(r) if not getattr(pot, "needs_geometry", False) else None
    print("  ok  build_potential ->", type(pot).__name__)

print(f"PASS {MODE}")

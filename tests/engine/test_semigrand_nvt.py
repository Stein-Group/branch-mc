"""A mixture at fixed N: the semi-grand path runs, conserves N, and moves.

WHAT THE ENSEMBLE IS. `sim_setup` used to leave typed + NVT unreachable -- the
species layer demanded `ensemble.mu` and the NVT branch refused it, so the two
messages pointed at each other. The combination is now accepted, and what it
gives is the SEMI-GRAND ensemble rather than the canonical one: the engine's NVT
path forces the repack move, which removes k particles and inserts k, and a typed
insertion draws its species carrying `z_s` in the one-body weight. Total N is
therefore conserved exactly while composition relaxes under the activity ratio.

That distinction is the reason this file exists rather than a comment. Two
properties have to hold together and each is worthless alone:

  * N EXACTLY constant, every window. Without it the ensemble is not NVT and the
    density -- the whole reason for running at fixed N -- is not fixed.
  * composition MOVING. A run that conserved N by never changing anything would
    satisfy the first check perfectly, which is what makes it the reachable-value
    half of the pair.

The statistical correctness of the composition it relaxes TO is a separate and
much slower question, and it lives in `test_semigrand_arbiter.py` (tier 3).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gcmc import io as gio
from gcmc import runner
from gcmc.sim_setup import from_params
from gcmc.species_file import load_forcefield

BOX = 8.0
N_SIDE = 7
MODEL = "ff/species_binary_k0.5.yaml"


class _FramePaths:
    """The one attribute `io.write_frame` reads off a RunPaths."""

    def __init__(self, trajectory) -> None:
        """Hold the destination."""
        self.trajectory = trajectory


def _start(tmp_path, x_first: float = 0.5):
    """An equimolar lattice start carrying a species column."""
    ff = load_forcefield(MODEL)
    g = (np.arange(N_SIDE) + 0.5) * (BOX / N_SIDE)
    pos = np.array([[x, y, z] for x in g for y in g for z in g])
    n_a = int(round(x_first * len(pos)))
    labels = np.concatenate(
        [np.zeros(n_a, dtype=np.int64), np.ones(len(pos) - n_a, dtype=np.int64)]
    )
    spc = np.random.default_rng(0).permutation(labels)
    path = tmp_path / "start.extxyz"
    gio.write_frame(
        _FramePaths(path),
        pos,
        two_dim=False,
        box=BOX,
        info={"step": 0},
        append=False,
        species=spc,
        symbols=ff.symbol,
    )
    return path, len(pos)


def _params(start, n, **ens_over):
    ens = {
        "type": "NVT",
        "dimension": 3,
        "box": BOX,
        "N": n,
        "mu": {"A": 0.0, "B": 0.0},
    }
    ens.update(ens_over)
    return {
        "potential": {"main": "lennard_jones", "T": 1.5, "species_file": MODEL},
        "sampler": {
            "type": "channel",
            "depth": 1,
            "trial_density": 1.0,
            "channel_width": 1.5,
            "channel_length": 3.0,
        },
        "ensemble": {k: v for k, v in ens.items() if v is not None},
        "run": {
            "kind": "single",
            "seed": 5,
            "traj_every": 0,
            "target_cycles": 12000,
            "sampling_window": 2000,
            "checkpoint_every": 3,
            "initial_config": str(start),
        },
    }


def _log(paths):
    lines = [ln for ln in paths.log.read_text().splitlines() if ln.strip()]
    names = lines[0].lstrip("#").split()
    rows = np.array([[float(x) for x in ln.split()] for ln in lines[1:]])
    return {n: rows[:, i] for i, n in enumerate(names)}


def test_a_mixture_at_fixed_n_conserves_n_and_still_mixes(tmp_path):
    start, n = _start(tmp_path)
    p = tmp_path / "sg.json"
    p.write_text(json.dumps(_params(start, n)))
    _state, paths = runner.run(str(p), root=str(tmp_path / "runs"))

    log = _log(paths)
    assert len(log["N"]) >= 5, "too few windows for the drift check to mean anything"
    assert set(log["N"].tolist()) == {float(n)}, (
        f"N took the values {sorted(set(log['N'].tolist()))}; NVT must hold it at "
        f"{n} exactly, and a density that moves is not the arm this run is for"
    )
    assert np.allclose(log["N_A"] + log["N_B"], log["N"])

    # THE REACHABLE-VALUE HALF. Everything above is satisfied by a chain that
    # never changes a label, which is precisely what a species-blind repack or a
    # dropped species draw would produce -- and it would look like a perfect NVT
    # run while doing so.
    assert log["N_A"].std() > 0, (
        "the composition never moved across the whole run: N_A was "
        f"{log['N_A'][0]:.0f} in every window. Conserving N by conserving every "
        "label is not sampling"
    )


def test_a_typed_nvt_run_without_a_state_point_is_refused_by_name(tmp_path):
    """The composition still needs an activity ratio, so the route is required.

    At fixed N only the RATIO of activities does any work -- the common factor
    cancels in the repack -- but the ratio is what sets the composition, so
    omitting it is under-specifying the run rather than choosing a default.
    """
    start, n = _start(tmp_path)
    with pytest.raises(ValueError, match="exactly one state-point route"):
        from_params(_params(start, n, mu=None))


def test_a_single_species_nvt_run_still_refuses_mu(tmp_path):
    """`mu` became meaningful for a typed NVT run and stays meaningless for an
    untyped one, where it is the same number for every particle and cancels."""
    with pytest.raises(ValueError, match="NVT takes no"):
        from_params(
            {
                "potential": {"main": "hard_core"},
                "sampler": {
                    "type": "branch",
                    "depth": 1,
                    "trials": 10,
                    "trial_box_side": 3.0,
                },
                "ensemble": {
                    "type": "NVT",
                    "dimension": 3,
                    "box": 12.0,
                    "N": 100,
                    "mu": 2.0,
                },
                "run": {"kind": "single", "radius": 0.5, "seed": 1},
            }
        )


def test_a_typed_nvt_sweep_spelling_is_refused_with_the_replacement_named(tmp_path):
    start, n = _start(tmp_path)
    params = _params(start, n)
    params["ensemble"]["sweep"] = {"start": 1.0, "stop": 2.0, "num": 2}
    with pytest.raises(ValueError, match="LIST OF MAPS"):
        from_params(params)

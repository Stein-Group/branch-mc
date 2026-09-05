"""run_id identifies the physics of a run, not its serialization: dict key
order and float spelling must not move it, and operational knobs (how often to
checkpoint, how long to run) must not either. RunPaths coerces str fields to
Path so every downstream consumer sees pathlib semantics.
"""

from __future__ import annotations

from pathlib import Path

from gcmc.run import hash_fingerprint as hf
from gcmc.run.hash_fingerprint import RunPaths
from gcmc.run.sim_setup import from_params

_BASE = {
    "potential": {"main": "hard_core"},
    "sampler": {"type": "branch", "depth": 1, "trials": 10, "trial_box_side": 3.0},
    "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
    "run": {"seed": 1},
}


def _rid(params) -> str:
    return hf.run_id(from_params(params))


def test_run_id_ignores_key_order_and_float_spelling():
    reordered = {
        "run": {"seed": 1},
        "ensemble": {"box": 12.0, "dimension": 3, "mu": 2e0},
        "sampler": {"trial_box_side": 3.0, "trials": 10, "depth": 1, "type": "branch"},
        "potential": {"main": "hard_core"},
    }
    assert _rid(_BASE) == _rid(reordered)


def test_run_id_ignores_operational_keys():
    op = {**_BASE, "run": {"seed": 1, "checkpoint_every": 5, "target_cycles": 9.9}}
    assert _rid(_BASE) == _rid(op)


def test_run_id_changes_with_physics():
    assert _rid(_BASE) != _rid({**_BASE, "ensemble": {**_BASE["ensemble"], "mu": 3.0}})


def test_run_paths_coerces_str_fields_to_path():
    rp = RunPaths(
        directory="runs/x",
        checkpoint="runs/x/checkpoint.npz",
        checkpoint_bak="runs/x/checkpoint.npz.bak",
        trajectory="runs/x/trajectory.extxyz",
        log="runs/x/log.csv",
        params="runs/x/params.json",
        lock="runs/x/lock",
    )
    assert isinstance(rp.directory, Path) and rp.checkpoint.name == "checkpoint.npz"

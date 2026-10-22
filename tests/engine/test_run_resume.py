"""End-to-end run then resume, small enough for the default tier. The first
process writes a checkpoint at the target step; the second resumes and must not
re-run the completed window (its step is unchanged), landing in the same run
directory with the expected files.

The params are built here rather than read from examples/. This test used to open
examples/3d_hard_sphere_branch.json, which was deleted when the examples were
consolidated into example.jsonc, and the test has been failing on a missing file
ever since. An end-to-end test of the run/resume protocol has no business
depending on which example configs happen to be shipped, so the dependency is
gone rather than the file restored.

The stop condition is `state.step < target_cycles`, so target_cycles = 0.004
means "one window and stop": step lands on sampling_window exactly, whatever the
physics does. checkpoint_every = 1 forces the checkpoint at that window boundary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gcmc.run import runner

_WINDOW = 2000

_PARAMS = {
    "potential": {"main": "hard_core"},
    "sampler": {"type": "branch", "trials": 30, "trial_box_side": 3.0},
    "engine": {"depth": 1},
    "ensemble": {"mu": 3.0, "dimension": 3, "box": 12.0, "type": "muVT"},
    "run": {
        "kind": "single",
        "radius": 0.5,
        "seed": 1,
        "traj_every": 1,
        "target_cycles": 0.004,
        "sampling_window": _WINDOW,
        "checkpoint_every": 1,
    },
}


def _write_params(dst: Path) -> Path:
    out = dst / "params.json"
    out.write_text(json.dumps(_PARAMS))
    return out


def test_run_then_resume_is_idempotent(tmp_path):
    params = _write_params(tmp_path)
    root = tmp_path / "runs"

    state1, paths1 = runner.run(str(params), root=str(root))
    assert state1.step == _WINDOW
    assert paths1.checkpoint.exists()
    assert paths1.params.exists() and paths1.trajectory.exists()

    state2, paths2 = runner.run(str(params), root=str(root))
    # resume lands in the same directory and does not advance past the target
    assert paths2.directory == paths1.directory
    assert state2.step == _WINDOW


def test_cli_run_exits_zero_and_prints_done_line(tmp_path):
    params = _write_params(tmp_path)
    root = tmp_path / "runs"
    proc = subprocess.run(
        [sys.executable, "-m", "gcmc", "run", str(params), "--root", str(root)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.startswith(f"done: step={_WINDOW}")


# ---------------------------------------------------------------- typed resume
#
# The defect this covers: `io.save_checkpoint` grew a `species` parameter at D2,
# `_read_npz` reads the array back, `load_checkpoint` returns it -- and
# `runner._save` never passed it while the resume path never read it. A parameter
# with no caller and a return value with no reader, facing each other across a
# gap, so a resumed multi-species run restarted with every particle relabelled
# species 0 and reported a composition it had never sampled.
#
# End to end through `runner.run`, twice, because that is the only place the gap
# was: both halves of the io layer were already correct in isolation and both are
# unit-tested in `tests/unit/test_species_round_trip.py`. A test of either half
# alone would have passed throughout.

_TYPED_PARAMS = {
    "potential": {
        "main": "lennard_jones",
        "T": 2.0,
        "species_file": "ff/species_three_reduced.yaml",
    },
    "sampler": {
        "type": "channel",
        "channel_length": 4.0,
        "trial_density": 1.0,
        "channel_width": 1.5,
    },
    "engine": {"depth": 1},
    "ensemble": {
        "mu": {"A": 0.0, "B": 0.0, "C": 0.0},
        "dimension": 3,
        "box": 8.0,
        "type": "muVT",
    },
    "run": {
        "kind": "single",
        "seed": 1,
        "traj_every": 1,
        "target_cycles": 0.0006,
        "sampling_window": 600,
        "checkpoint_every": 1,
    },
}


def test_a_typed_run_carries_its_species_across_a_resume(tmp_path):
    """The species column survives the checkpoint and the restart.

    Asserted three ways, because each catches a different failure:

      * the checkpoint FILE holds a species array. Without this the next two
        could pass on a run that never restarted.
      * the resumed state's column is aligned with its positions and still a
        mixture. A column dropped on load reads as all-species-0, which is
        aligned and the right length -- so length alone is not enough.
      * the resumed column is byte-identical to the one that was saved. The
        resume does no windows (the target is already met), so anything other
        than equality means the column was rebuilt rather than restored.
    """
    import numpy as np

    params = tmp_path / "typed.json"
    params.write_text(json.dumps(_TYPED_PARAMS))
    root = tmp_path / "runs"

    state1, paths1 = runner.run(str(params), root=str(root))
    assert state1.species is not None, "the typed run finished with no species column"
    saved = np.asarray(state1.species)
    assert len(saved) == len(state1.positions)
    assert np.unique(saved).size > 1, (
        f"the run ended with species {np.unique(saved).tolist()} present; a "
        "three-species run at equal activities should be a mixture, and a "
        "single-species result would make the resume check vacuous"
    )

    from gcmc.run import io as gio

    ckpt = gio.load_checkpoint(paths1)
    assert ckpt is not None and ckpt["species"] is not None, (
        "the checkpoint file carries no species array, so nothing could be "
        "restored from it"
    )
    assert np.asarray(ckpt["species"]).tolist() == saved.tolist()

    state2, paths2 = runner.run(str(params), root=str(root))
    assert paths2.directory == paths1.directory
    assert state2.step == state1.step, "the resume re-ran a completed window"
    assert state2.species is not None, (
        "the resumed run has no species column: the checkpoint's array was "
        "written but not read back, so every particle is now species 0"
    )
    assert np.asarray(state2.species).tolist() == saved.tolist(), (
        "the resumed species column differs from the saved one although no "
        "window ran; it was rebuilt rather than restored"
    )

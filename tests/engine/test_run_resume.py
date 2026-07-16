"""End-to-end run then resume, small enough for the default tier. The first
process writes a checkpoint at the target step; the second resumes and must not
re-run the completed window (its step is unchanged), landing in the same run
directory with the expected files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from gcmc import runner

_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "3d_hard_sphere_branch.json"
)


def _write_params(dst: Path) -> Path:
    p = json.loads(_EXAMPLE.read_text())
    p["run"]["target_cycles"] = 0.004
    p["run"]["sampling_window"] = 2000
    p["run"]["checkpoint_every"] = 1
    out = dst / "params.json"
    out.write_text(json.dumps(p))
    return out


def test_run_then_resume_is_idempotent(tmp_path):
    params = _write_params(tmp_path)
    root = tmp_path / "runs"

    state1, paths1 = runner.run(str(params), root=str(root))
    assert state1.step == 2000
    assert paths1.checkpoint.exists()
    assert paths1.params.exists() and paths1.trajectory.exists()

    state2, paths2 = runner.run(str(params), root=str(root))
    # resume lands in the same directory and does not advance past the target
    assert paths2.directory == paths1.directory
    assert state2.step == 2000


def test_cli_run_exits_zero_and_prints_done_line(tmp_path):
    params = _write_params(tmp_path)
    root = tmp_path / "runs"
    proc = subprocess.run(
        [sys.executable, "-m", "gcmc", "run", str(params), "--root", str(root)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert proc.stdout.startswith("done: step=2000")

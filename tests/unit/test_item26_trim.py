"""Item 26: trim_trajectory streams frame by frame.

The trim must keep whole extxyz frames whose step is <= max_step and drop the
rest, now without loading the whole trajectory into memory. This builds a small
multi-frame file, trims it, and checks exactly the right frames survive, plus the
absent-file no-op.
"""

import os
import tempfile

from gcmc import io


def _frame(natoms, step):
    lines = [f"{natoms}\n", f"Lattice=... step={step}\n"]
    for k in range(natoms):
        lines.append(f"C {k}.0 {k}.0 {step}.0\n")
    return "".join(lines)


def _steps_in(path):
    import re

    steps = []
    with open(path) as f:
        for ln in f:
            m = re.search(r"step=(\d+)", ln)
            if m:
                steps.append(int(m.group(1)))
    return steps


def test_trim_keeps_frames_up_to_max_step():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "traj.extxyz")
        with open(path, "w") as f:
            f.write(_frame(3, 10))
            f.write(_frame(2, 20))
            f.write(_frame(4, 30))
            f.write(_frame(1, 40))
        io.trim_trajectory(path, max_step=25)
        assert _steps_in(path) == [10, 20], _steps_in(path)
        # the kept frames must still be intact: 3-atom then 2-atom
        with open(path) as f:
            lines = f.readlines()
        assert lines[0].strip() == "3" and lines[5].strip() == "2"
        assert len(lines) == (3 + 2) + (2 + 2)


def test_trim_absent_file_is_noop():
    io.trim_trajectory("/tmp/definitely_not_here_98765.extxyz", max_step=5)


def test_trim_all_or_none():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.extxyz")
        with open(path, "w") as f:
            f.write(_frame(2, 5))
            f.write(_frame(2, 6))
        io.trim_trajectory(path, max_step=100)
        assert _steps_in(path) == [5, 6]
        io.trim_trajectory(path, max_step=0)
        assert _steps_in(path) == []


if __name__ == "__main__":
    test_trim_keeps_frames_up_to_max_step()
    test_trim_absent_file_is_noop()
    test_trim_all_or_none()
    print("PASS test_item26_trim")

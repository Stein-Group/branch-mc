"""_init_config resolves run.initial_config two ways: a file path is loaded
directly, and a directory is treated as the equilibrated library. An empty
library directory must fail loudly rather than start a silent empty run.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc.run import runner
from gcmc.run.sim_setup import from_params

_BASE = {
    "potential": {"main": "hard_core"},
    "sampler": {"type": "branch", "trials": 10, "trial_box_side": 3.0},
    "engine": {"depth": 1},
    "ensemble": {"mu": 2.0, "dimension": 3, "box": 12.0},
}


def test_init_config_file_arm_loads_xyz(tmp_path):
    f = tmp_path / "init.xyz"
    f.write_text("2\ncomment\nX 1.0 1.0 1.0\nX 2.0 2.0 2.0\n")
    cfg = from_params({**_BASE, "run": {"seed": 1, "initial_config": str(f)}})
    pos = runner._init_config(cfg, np.random.default_rng(0))
    assert pos.shape == (2, 3)


def test_init_config_directory_arm_raises_on_empty_library(tmp_path):
    empty = tmp_path / "library"
    empty.mkdir()
    cfg = from_params({**_BASE, "run": {"seed": 1, "initial_config": str(empty)}})
    with pytest.raises(FileNotFoundError, match=r"no \.xyz configurations found"):
        runner._init_config(cfg, np.random.default_rng(0))

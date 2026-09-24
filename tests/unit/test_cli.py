"""The command line's contract with the shell.

Two things a caller depends on and neither is checked anywhere else: the exact
`done: ...` line the cluster sweep scripts grep for, and the promise that a bad
params file is a USER error -- a readable message and exit 2 -- rather than a
traceback. `sim_setup`'s messages are written to be read; a traceback wrapped
around one hides it in frames the reader cannot act on.
"""

from __future__ import annotations

import json

import pytest

from gcmc.__main__ import main


def _params(**over):
    p = {
        "potential": {"main": "hard_core"},
        "sampler": {"type": "branch", "trials": 20, "trial_box_side": 3.0},
        "engine": {"depth": 1},
        "ensemble": {"type": "muVT", "dimension": 3, "box": 10.0, "mu": 1.0},
        "run": {
            "kind": "single",
            "radius": 0.5,
            "seed": 1,
            "sampling_window": 100,
            "target_cycles": 100,
        },
    }
    p.update(over)
    return p


def test_a_run_prints_the_line_the_sweep_scripts_grep_for(tmp_path, capsys):
    f = tmp_path / "params.json"
    f.write_text(json.dumps(_params()))
    assert main(["run", str(f), "--root", str(tmp_path / "runs")]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith("done: step=") and " N=" in last and " dir=" in last


@pytest.mark.parametrize(
    ("what", "fragment"),
    [
        ("missing-file", "No such file"),
        ("bad-params", "missing required keys"),
        ("bad-value", "unknown sampler"),
    ],
)
def test_a_user_error_is_a_message_and_exit_2_not_a_traceback(
    tmp_path, capsys, what, fragment
):
    if what == "missing-file":
        target = str(tmp_path / "nope.json")
    else:
        f = tmp_path / "params.json"
        if what == "bad-params":
            f.write_text(json.dumps({"potential": {"main": "hard_core"}}))
        else:
            p = _params()
            p["sampler"]["type"] = "definitely-not-a-sampler"
            f.write_text(json.dumps(p))
        target = str(f)

    with pytest.raises(SystemExit) as exc:
        main(["run", target, "--root", str(tmp_path / "runs")])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("error: "), f"not a clean message: {err!r}"
    assert fragment in err
    assert "Traceback" not in err

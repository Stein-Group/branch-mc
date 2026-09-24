"""Per-species occupancy reaches the run's FILES, not just `state.species`.

The gap this closes: composition is the entire output of a multi-species run,
and until S4 phase 11 no run wrote it anywhere. `metropolis_dynamics.advance` returned
`{N, packing_fraction, counters, n_cav}` and the log had one `N` column, so a
cluster run of a mixture produced a file in which A and B are indistinguishable.
Every composition number in this suite was read out of `state.species` by a test
holding the object in memory -- which is exactly the arrangement that hid the
window-boundary reset of phase 6, since an in-process reader can look at the
column at a moment when a file reader has nothing to look at.

End to end through `runner.run` rather than through `advance`, for the reason
phase 9 gives about the checkpoint: two halves can each be correct and the seam
between them empty. `advance` emitting the key and the log declaring the column
are separate halves.

What each assertion would have to see to fail:

  * the header check fails if the columns are not declared -- the state before
    this change.
  * `sum(N_s) == N` fails if the column and the positions come apart, which is
    the phase-6 defect writing itself into a file.
  * the mixture check fails if the run reports everything as species 0, which is
    what a boundary reset, a dropped resume column or a species-blind kernel all
    produce. Without it the two above pass happily on `(N, 0, 0)`.
  * the single-species header check fails if the layout moved for runs that have
    no composition, which every existing positional reader depends on.
"""

from __future__ import annotations

import json

import pytest

from gcmc.run import runner

_TYPED = {
    "potential": {
        "main": "lennard_jones",
        "T": 2.0,
        "species_file": "ff/species_three_reduced.yaml",
    },
    "sampler": {
        "type": "channel_cyclic",
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
        "traj_every": 0,
        "target_cycles": 0.0018,
        "sampling_window": 600,
        "checkpoint_every": 1,
    },
}

_UNTYPED = {
    "potential": {"main": "hard_core"},
    "sampler": {"type": "branch", "trials": 30, "trial_box_side": 3.0},
    "engine": {"depth": 1},
    "ensemble": {"mu": 3.0, "dimension": 3, "box": 12.0, "type": "muVT"},
    "run": {
        "kind": "single",
        "radius": 0.5,
        "seed": 1,
        "traj_every": 0,
        "target_cycles": 0.004,
        "sampling_window": 2000,
        "checkpoint_every": 1,
    },
}


def _run(params: dict, tmp_path, name: str):
    """Run to completion and return (header line, data rows as token lists)."""
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(params))
    _state, paths = runner.run(str(p), root=str(tmp_path / "runs"))
    lines = [ln for ln in paths.log.read_text().splitlines() if ln.strip()]
    header, rows = lines[0], [ln.split() for ln in lines[1:]]
    assert rows, "the run wrote no data rows, so nothing below is being tested"
    return header, rows


def test_typed_run_logs_one_column_per_species(tmp_path):
    header, rows = _run(_TYPED, tmp_path, "typed")
    names = header.lstrip("#").split()

    assert names[-3:] == ["N_A", "N_B", "N_C"], (
        f"the typed log declares no per-species columns; header is {names}"
    )
    n_i = names.index("N")
    a_i, b_i, c_i = (names.index(x) for x in ("N_A", "N_B", "N_C"))

    for row in rows:
        assert len(row) == len(names), (
            f"row has {len(row)} fields against {len(names)} columns: "
            "a fixed-width row was truncated against its header"
        )
        n, a, b, c = (int(row[i]) for i in (n_i, a_i, b_i, c_i))
        assert a + b + c == n, (
            f"composition ({a}, {b}, {c}) sums to {a + b + c} against N = {n}"
        )

    # The reachable-value check. Everything above is satisfied by a run that
    # reports (N, 0, 0) in every window, which is precisely what the boundary
    # reset did for the whole of P3.
    present = {i for row in rows for i in range(3) if int(row[(a_i, b_i, c_i)[i]]) > 0}
    assert len(present) > 1, (
        f"only species {sorted(present)} ever appears in the log. Three species "
        "at equal activities is a mixture; a single-label result means the "
        "column is being reset rather than reported"
    )


def test_single_species_header_is_unchanged(tmp_path):
    header, rows = _run(_UNTYPED, tmp_path, "untyped")
    assert header == runner.LOG_HEADER, (
        "the single-species log layout moved. Every positional reader of these "
        "files -- the sweetspot plot, the characterisation harnesses -- indexes "
        "columns by number"
    )
    assert not [n for n in header.split() if n.startswith("N_")], (
        "a run with no species column declared per-species columns anyway"
    )
    assert len(rows[0]) == len(header.lstrip("#").split())


def test_a_column_with_no_value_raises_and_names_itself():
    """The row cannot silently lose its tail.

    The earlier arrangement passed a positional list of cells alongside a
    positional list of widths, and `zip` stops at the shorter of the two -- so a
    row carrying more cells than the header declared dropped them and still
    wrote a well-formed line, the per-species columns being the tail. Header and
    row are now built from ONE column list with values looked up by key, which
    makes that failure unreachable; what remains reachable is a declared column
    the window did not produce, and that raises with the column named.
    """
    cols = [("N", 7, "core", "N"), ("wall_s", 11, "core", "wall")]
    assert runner._log_row(cols, {"N": 3, "wall": "1.0"}).split() == ["3", "1.0"]
    with pytest.raises(KeyError, match="wall_s"):
        runner._log_row(cols, {"N": 3})

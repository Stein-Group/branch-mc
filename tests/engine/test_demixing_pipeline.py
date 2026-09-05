"""The demixing pipeline end to end: a typed run, then the analysis that reads
only what the run left on disk.

The two halves are the ones this stage keeps finding gaps between -- something
written correctly and something read correctly, with nothing joining them. Here
the log's per-species columns and the trajectory's species symbols are written by
`runner.run`, and `scripts.demixing.analyse_run` reads them back through a
different code path (the header by name, the frames through ase and the species
file's own symbol table). Only running both sees whether they meet.

Marked slow-ish by size rather than by marker: one small typed run of three
windows, a few seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gcmc.run import runner
from scripts.demixing import analyse_run, read_log, write_models

_PARAMS = {
    "potential": {
        "main": "lennard_jones",
        "T": 1.0,
        "species_file": "ff/species_binary_k0.5.yaml",
    },
    "sampler": {
        "type": "channel",
        "depth": 1,
        "trial_density": 1.0,
        "channel_width": 1.5,
        "channel_length": 3.0,
    },
    "ensemble": {
        "type": "muVT",
        "dimension": 3,
        "box": 8.0,
        "mu": {"A": -3.4, "B": -2.55},
    },
    "run": {
        "kind": "single",
        "seed": 3,
        "initial_config": "grid",
        "sampling_window": 400,
        "target_cycles": 1000,
        "traj_every": 1,
        "checkpoint_every": 1,
    },
}


def test_the_series_is_generated_from_the_anchor(tmp_path):
    """Six models, each differing from the anchor in exactly one number.

    The check is on `k_ij` itself rather than on the file text: a series written
    from a template that forgot to substitute would still produce six files with
    six names.
    """
    from gcmc.forcefield.species_file import load_forcefield
    from scripts.demixing import K_SERIES

    paths = write_models(tmp_path)
    assert len(paths) == len(K_SERIES)
    for k, path in zip(K_SERIES, paths):
        ff = load_forcefield(str(path))
        assert ff.k_ij[0, 1] == k, f"{path} carries k = {ff.k_ij[0, 1]}, not {k}"
        assert ff.k_ij[0, 0] == 0.0, "the diagonal of k_ij must stay zero"
        assert tuple(ff.types) == ("A", "B")
        # eps_AB = (1 - k) sqrt(eps_A eps_B), which is the whole point of the axis
        assert ff.pair_epsilon()[0, 1] == (1 - k) * np.sqrt(1.0 * 0.8)


def test_regenerating_the_series_reproduces_the_committed_files_byte_for_byte(
    tmp_path,
):
    """The generator is idempotent, and that is a correctness property here.

    `scripts/submit_demixing.sbatch` regenerates the series at the top of every
    job so a stale checkout cannot run the wrong model. A species file's sha256
    is part of the run identity, so a generator that produced equivalent-but-not-
    identical bytes -- a reordered mapping, a different float repr, a changed
    header -- would give every run a new id, silently orphaning every checkpoint
    and resuming nothing.
    """
    from pathlib import Path

    for path in write_models(tmp_path):
        committed = Path("ff") / path.name
        assert committed.exists(), f"{committed} is generated but not committed"
        assert path.read_bytes() == committed.read_bytes(), (
            f"{committed} differs from a fresh generation; regenerating it in a "
            "job would change the run identity of every run that uses it"
        )


def test_a_run_and_its_analysis_agree(tmp_path):
    params = tmp_path / "demix.json"
    params.write_text(json.dumps(_PARAMS))
    _state, paths = runner.run(str(params), root=str(tmp_path / "runs"))

    log = read_log(paths.log)
    assert {"N_A", "N_B"} <= set(log), f"log columns are {sorted(log)}"
    assert np.allclose(log["N_A"] + log["N_B"], log["N"])

    res = analyse_run(paths.directory, n_shuffle=8)
    assert res["species"] == ["A", "B"]
    assert res["k_ij"] == 0.5, "the analysis read the wrong model's cross term"

    # The frames and the log are two independent records of the same chain, so
    # the LAST frame's composition must be the last window's. This is the join
    # the test exists for: the log column and the trajectory symbol table are
    # written by different code and read by different code.
    assert res["frame_n_by_species"][-1] == [
        int(log["N_A"][-1]),
        int(log["N_B"][-1]),
    ], "the final frame's composition disagrees with the final log row"

    assert len(res["alpha"]) == len(res["frame_step"])
    assert all(np.isfinite(a) for a in res["alpha"])
    assert res["alpha_null"]["sd"] > 0, (
        "the permutation band has no spread, so nothing could be judged against it"
    )
    g = np.asarray(res["rdf_g"])
    assert g.shape[0] == g.shape[1] == 2 and g.ndim == 3
    assert (g >= 0).all()


def test_the_analysis_refuses_a_log_with_no_composition(tmp_path):
    """A run from before the per-species columns existed is refused by name.

    Silently returning an empty composition series would make an old run look
    like a converged one with nothing to say.
    """
    import pytest

    d = tmp_path / "old_run"
    d.mkdir()
    (d / "params.json").write_text(json.dumps(_PARAMS))
    (d / "run.log").write_text("#      N\n    100\n")
    with pytest.raises(ValueError, match="no per-species columns"):
        analyse_run(d)


def test_prepare_makes_what_a_base_file_names_and_refuses_what_it_cannot(tmp_path):
    """`prepare` is load-bearing: the sbatch calls it instead of a hardcoded loop.

    Two properties, and the second is the one worth having. It writes the models
    and starts the base file's `_models` / `_starts` blocks describe -- and it
    CHECKS that every path the params actually name now exists. A base file that
    sweeps a model nobody generated used to fail eighteen runs later, inside the
    launcher, after the job had been queued.
    """
    import json as _json

    from scripts.demixing import prepare

    base = tmp_path / "base.json"
    start = tmp_path / "start.extxyz"
    spec = {
        "_models": {
            "anchor": "ff/species_symmetric_reduced.yaml",
            "out": str(tmp_path),
        },
        "_starts": {
            "model": "ff/species_symmetric_reduced.yaml",
            "box": 8.0,
            "eta": 0.35,
            "n": 200,
            "files": {str(start): 0.5},
        },
        "potential": {"species_file": [str(tmp_path / "species_symmetric_k0.yaml")]},
        "run": {"initial_config": [str(start)]},
    }
    base.write_text(_json.dumps(spec))
    made = prepare(base)
    assert len(made["models"]) == 6 and len(made["starts"]) == 1
    assert start.exists()
    assert int(start.read_text().splitlines()[0]) == 200, (
        "the start does not carry the exact particle count ensemble.N would state"
    )

    spec["potential"]["species_file"].append(str(tmp_path / "not_generated.yaml"))
    base.write_text(_json.dumps(spec))
    with pytest.raises(FileNotFoundError, match=r"not_generated\.yaml"):
        prepare(base)

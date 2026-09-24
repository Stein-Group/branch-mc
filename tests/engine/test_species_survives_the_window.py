"""A typed run's species column must survive the window boundary.

THE DEFECT. `advance` materialised `state.species` under
`if state.species is not None`, and a fresh `ChainState` has `species is None`.
So on a real multi-species run the column was built inside the window -- species
drawn per trial, identities swapped, every pair weighted by its own sigma_ij and
eps_ij -- and then discarded at the end. The next window re-seeded the cell list
from `state.species_of`, which materialises zeros when the column is None, so
every particle was relabelled species 0 at each boundary.

The visible symptom was a three-species run at equal activities reporting
composition (1, 0, 0) in every window while accepting ~65 identity swaps per
window. Per-species occupancy is the entire output of a multi-species run, so
this made every branch and channel mixture run wrong in the only quantity it
exists to produce -- and the chain itself was fine, which is why nothing looked
implausible.

WHY IT SURVIVED THE GATES, because that is the reusable part:

  * the feature matrix's species invariant reads
    `if species is not None: assert aligned`, so it skipped itself on exactly the
    runs that were broken. A guard conditioned on the thing being present cannot
    report it missing.
  * the typed statistical gates run through `advance_metropolis_mof_ms`, a
    different sampler that keeps its own column.
  * the engine tests that do check the column build a state that already has one,
    which is the one thing no real run does. `ChainState.fresh` is the start every
    run gets, and no test of this path used it.

So the test below starts from `ChainState.fresh` and nothing else. The first
assertion is that the column exists at all; the second is the sharp one.
"""

from __future__ import annotations

from pathlib import Path

import _harness as H
import numpy as np
import pytest
import yaml

from gcmc.engine import metropolis_dynamics
from gcmc.proposals import moves
from gcmc.system.state import ChainState

_SPECIES_FF = Path(__file__).resolve().parents[2] / "ff" / "species_three_reduced.yaml"
_BOX = 12.0


def _typed_cfg(tmp_path: Path, sampler: str, depth: int):
    """A three-species run, built exactly as a real one is: no species column."""
    with _SPECIES_FF.open() as fh:
        doc = yaml.safe_load(fh)
    doc["aux"] = "none"  # the auxiliary is tested elsewhere; keep this one narrow
    ff = tmp_path / "species.yaml"
    with ff.open("w") as fh:
        yaml.safe_dump(doc, fh)

    smp: dict = {"type": sampler}
    eng: dict = {"depth": depth}
    if sampler in ("channel", "channel_cyclic"):
        smp["trial_density"] = 1.0
        smp["channel_width"] = 1.5
        if sampler == "channel":
            smp["channel_length"] = 6.0
    else:
        smp["trials"] = 24
        smp["trial_box_side"] = 3.0
    params = {
        "potential": {"main": "lennard_jones", "T": 0.85, "species_file": str(ff)},
        "sampler": smp,
        "engine": eng,
        "ensemble": {
            "mu": {"A": 1.0, "B": 1.0, "C": 1.0},
            "dimension": 3,
            "box": _BOX,
        },
        "run": {"sampling_window": 120},
    }
    cfg, potential, weigher, _ = H.build(params, 5)
    return cfg, potential, weigher


_MODES = [("channel", 1), ("channel", 2), ("channel_cyclic", 1), ("branch", 1)]


@pytest.mark.parametrize("sampler,depth", _MODES, ids=lambda v: str(v))
def test_a_fresh_typed_run_reports_more_than_one_species(
    sampler, depth, tmp_path
) -> None:
    """Three species at equal activities do not come out as one.

    Deliberately weak as a physics statement -- it says nothing about what the
    composition SHOULD be, which is the statistical gates' job -- and exactly
    strong enough to fail on the defect, which reported (1, 0, 0) forever.
    """
    cfg, potential, weigher = _typed_cfg(tmp_path, sampler, depth)
    state = ChainState.fresh(cfg.dimension, seed=5)
    state.positions = H.lattice_seed(_BOX, 250, 3, np.random.default_rng(5))

    for _ in range(3):
        metropolis_dynamics.advance(state, cfg, potential, weigher)

    assert state.species is not None, (
        f"{sampler} d{depth}: a three-species run left state.species as None, so "
        "every consumer -- io, the checkpoint, the next window's cell list -- "
        "reads zeros and the run reports a pure species-0 system"
    )
    species = np.asarray(state.species)
    assert species.shape[0] == len(state.positions), (
        f"{sampler} d{depth}: {species.shape[0]} labels for "
        f"{len(state.positions)} positions"
    )
    present = int(np.unique(species).size)
    assert present > 1, (
        f"{sampler} d{depth}: {present} species present out of "
        f"{cfg.n_species} at equal activities. Composition "
        f"{np.bincount(species, minlength=cfg.n_species).tolist()}"
    )


@pytest.mark.parametrize("sampler,depth", _MODES, ids=lambda v: str(v))
def test_an_untouched_particle_keeps_its_species_across_the_boundary(
    sampler, depth, tmp_path
) -> None:
    """The sharp one: the column is carried, not rebuilt.

    A particle that neither moved nor was swapped in window k+1 must have the
    species it had at the end of window k. Matching is by exact coordinate --
    an untouched particle's position is a copied float, and the per-pair core
    floor forbids two particles sharing a coordinate -- so this needs no
    tolerance and no identity bookkeeping of its own.

    The failure mode it catches is the whole defect: relabelling at the boundary
    reassigns species 0 to particles that were B or C, so most matched
    coordinates come back with a different label. It also catches the subtler
    version -- a column written in a DIFFERENT slot order from the positions --
    which would scramble labels among live particles while keeping the counts
    right, and which no composition test can see.

    THE IDENTITY SWAP IS SUPPRESSED, and it has to be. A swap changes a
    particle's species without moving it, which is precisely the observation this
    test reads as a failure -- with swap on, 35 of 304 stationary particles
    legitimately changed label and the test could not tell those from a boundary
    reset. Suppressing it is the same discipline the M21 arbiter needed for the
    same reason: the move that equilibrates the quantity under test also masks
    the defect in it. `test_a_fresh_typed_run_reports_more_than_one_species`
    above runs with swap ON, so the two together cover both.
    """
    cfg, potential, weigher = _typed_cfg(tmp_path, sampler, depth)
    state = ChainState.fresh(cfg.dimension, seed=5)
    state.positions = H.lattice_seed(_BOX, 250, 3, np.random.default_rng(5))

    metropolis_dynamics.advance(
        state, cfg, potential, weigher
    )  # let a mixture establish

    def no_swap(rng, _n_species):
        """The three-way split whatever the species count: swap suppressed."""
        return str(rng.choice(moves.SINGLE_MOVES, p=moves.SINGLE_PROBS))

    real_draw = metropolis_dynamics.moves.draw_move
    metropolis_dynamics.moves.draw_move = no_swap
    try:
        metropolis_dynamics.advance(state, cfg, potential, weigher)
        before = {
            tuple(row): int(s)
            for row, s in zip(np.asarray(state.positions), np.asarray(state.species))
        }
        assert len(before) == len(state.positions), "two particles share a coordinate"

        summary = metropolis_dynamics.advance(state, cfg, potential, weigher)
        after = {
            tuple(row): int(s)
            for row, s in zip(np.asarray(state.positions), np.asarray(state.species))
        }
    finally:
        metropolis_dynamics.moves.draw_move = real_draw

    assert summary["counters"]["swap_step"] == 0, (
        "the swap suppression did not take effect, so a legal relabel is "
        "indistinguishable from a boundary reset and this test proves nothing"
    )

    survivors = set(before) & set(after)
    assert len(survivors) > 20, (
        f"{sampler} d{depth}: only {len(survivors)} particles were untouched "
        "across a window, too few to test the boundary. Shorten "
        "run.sampling_window."
    )
    relabelled = sorted(
        (before[k], after[k]) for k in survivors if before[k] != after[k]
    )
    assert not relabelled, (
        f"{sampler} d{depth}: {len(relabelled)} of {len(survivors)} untouched "
        f"particles changed species across the window boundary, e.g. "
        f"{relabelled[:6]}. Either the column is being rebuilt from zeros or it "
        "is written in a different slot order from the positions."
    )

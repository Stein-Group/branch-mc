"""Gate S4's addition: the auxiliary control arm is an EXACT no-op.

WHAT THE PLAN ASKS FOR. `docs/decisions/restructure_plan.md`, Gate S4: "the
auxiliary control arm must be an **exact** no-op, not statistically close. If it
is not, the correction is inserted in the wrong place, which is the open Q6." So
nothing here has a tolerance in it. Two chains either produce the same bytes or
they do not.

WHY THE ARM HAD TO BE REPAIRED BEFORE IT COULD BE RUN. The engine used to set

    aux_on = ljbranch and rc_sel < uc

with a comment explaining that the no-op case therefore "does no arithmetic at
all" so "bit-identity is structural". Both true, and together they made this gate
unable to fail: configuring `auxiliary.cutoff == run.upper_cutoff` and comparing
against no auxiliary tested the short-circuit, never the correction whose
insertion point is the open question. The typed side had the same hole one layer
further down and worse -- `aux: {control: true}` returned `aux_cutoff() is None`,
byte-for-byte identical to `aux: none`, so the "control arm" was literally the
arm it was the control for.

Both are fixed (`MC_engine.advance`'s flag, `ForceField.aux_cutoff`), so the
correction now RUNS at the no-op radius and this file is a real gate. It asserts
three things per point, and the first is what keeps it honest:

    1. the correction was EVALUATED (`aux_corr_n > 0`). A control arm that runs
       zero corrections passes trivially, which is the failure this whole file
       is a response to.
    2. its value was exactly 0.0 with exactly 0.0 spread. Exact rather than
       small: `_lj_set_energy_split` applies both radii to ONE neighbour query,
       as two identical sums over the same array in the same order, so at
       rc_short == rc_full the difference is a float zero. A tolerance here would
       have been the wrong instrument.
    3. the chain is the auxiliary-off chain, byte for byte -- counter stream and
       final configuration.

WHY 3 IS NOT IMPLIED BY 2. An exact zero added in the right place changes
nothing. An exact zero added in the wrong place still moves the acceptance, the
RNG draw that follows it, or -- the specific hazard in this code --
the None-ness of the tail term, since the insertion reads
`delta_E_lj = (delta_E_lj or 0.0) + delta_corr` and three call sites downstream
branch on `delta_E_lj is not None`. That it is safe (aux_on implies ljbranch
implies lj, so the tail is a float by then) is now measured rather than argued.

The arm also covers the plumbing it always covered: the auxiliary changes `bias`,
`aux_cutoff` and the fixed rod's periodic-guard margin, and none of those may
reach the sampler either.

TWO AXIS CELLS ARE DELIBERATELY ABSENT from `_POINTS`, and saying which matters
more than the count of green rows. 2D + `full_repacking` evaluates the correction
zero times at this density -- the repacking move almost never completes an
exchange -- so it would be three green rows testing nothing. Nothing else is
excluded: the cyclic rod at depth 2 used to crash here and was fixed rather than
skipped (see `test_cyclic_rod_short_chord.py`).
"""

from __future__ import annotations

import copy
from pathlib import Path

import _harness as H
import numpy as np
import pytest
import yaml

from gcmc.engine import MC_engine

#: Axis points where the correction is measured to actually evaluate. See the
#: module docstring for the one that is deliberately not here.
_POINTS = [
    (3, "channel", "depth1"),
    (3, "channel", "depth2"),
    (3, "channel", "repack"),
    (3, "channel_cyclic", "depth1"),
    (3, "channel_cyclic", "depth2"),
    (3, "channel_cyclic", "repack"),
    (2, "channel", "depth1"),
    (2, "channel", "depth2"),
    (2, "channel_cyclic", "depth1"),
    (2, "channel_cyclic", "depth2"),
]
_SEEDS = (18, 19, 20)
_WINDOWS = 3

#: The correction's own diagnostics. Excluded from the byte comparison because
#: their PRESENCE is the difference between the arms -- the aux-off run has no
#: such keys at all -- and their values are separately asserted to be zero.
_DIAG = ("aux_corr_mean", "aux_corr_sd", "aux_corr_n")

_SPECIES_FF = Path(__file__).resolve().parents[2] / "ff" / "species_three_reduced.yaml"


def _params(dim: int, rod: str, deep: str) -> dict:
    """An LJ channel run at one axis point, with no auxiliary block."""
    sampler: dict = {"type": rod, "trial_density": 0.6, "channel_width": 1.2}
    if rod == "channel":
        sampler["channel_length"] = 5.0
    if deep == "repack":
        sampler["full_repacking"] = True
    else:
        sampler["depth"] = 1 if deep == "depth1" else 2
    return {
        "potential": {"main": "lennard_jones", "T": 0.85},
        "sampler": sampler,
        "ensemble": {"mu": -3.2, "dimension": dim, "box": 12.0},
        "run": {"sampling_window": 20},
    }


def _flatten(summary: dict) -> dict[str, float]:
    """The window summary as one flat name -> float map, diagnostics dropped."""
    out: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for sub, inner in value.items():
                out[f"{key}.{sub}"] = float(inner)
        elif key not in _DIAG:
            out[key] = float(value)
    return out


def _run(params: dict, seed: int, dim: int, n_seed: int) -> tuple:
    """(cfg, keys, counter stream, final positions, per-window correction stats)."""
    box = params["ensemble"]["box"]
    pos = H.lattice_seed(box, n_seed, dim, np.random.default_rng(1000 + seed))
    cfg, potential, weigher, state = H.build(params, seed, positions=pos)
    rows: list[list[float]] = []
    keys: list[str] | None = None
    evaluated = 0
    stats: list[tuple[float, float]] = []
    for _ in range(_WINDOWS):
        summary = MC_engine.advance(state, cfg, potential, weigher)
        n = int(summary.get("aux_corr_n", 0))
        evaluated += n
        if n:
            stats.append(
                (float(summary["aux_corr_mean"]), float(summary["aux_corr_sd"]))
            )
        flat = _flatten(summary)
        if keys is None:
            keys = sorted(flat)
        rows.append([flat[k] for k in keys])
    return (
        cfg,
        keys,
        np.asarray(rows, float),
        np.asarray(state.positions, float),
        evaluated,
        stats,
    )


def _assert_exact_no_op(label: str, off: tuple, on: tuple) -> None:
    """The three assertions the module docstring lists, in that order."""
    _, k_off, c_off, x_off, _, _ = off
    _, k_on, c_on, x_on, evaluated, stats = on

    assert evaluated > 0, (
        f"{label}: the correction was evaluated 0 times, so this point proves "
        "nothing. Either the axis point stopped completing exchanges or the "
        "no-op radius stopped switching the auxiliary on; do not leave it green."
    )
    for mean, sd in stats:
        assert mean == 0.0 and sd == 0.0, (
            f"{label}: the correction is {mean!r} +- {sd!r} at the no-op radius, "
            "where both shell energies come from one query and two identical "
            "sums. A non-zero value means the two radii are not being applied to "
            "the same pair list -- or, in typed mode, not with the same types."
        )

    assert k_off == k_on, f"{label}: the window summary changed shape"
    if c_off.tobytes() != c_on.tobytes():
        for w in range(len(c_off)):
            diff = {
                k: (c_off[w, i], c_on[w, i])
                for i, k in enumerate(k_off)
                if c_off[w, i] != c_on[w, i]
            }
            if diff:
                pytest.fail(f"{label}: counters diverge in window {w}: {diff}")
    assert x_off.shape == x_on.shape, (
        f"{label}: final N is {len(x_on)} with the auxiliary, {len(x_off)} without"
    )
    if x_off.tobytes() != x_on.tobytes():
        d = np.abs(x_off - x_on).max(axis=1)
        r = int(np.argmax(d))
        pytest.fail(
            f"{label}: positions diverge, worst row {r} by {d[r]:.3e}: "
            f"{x_off[r]} -> {x_on[r]}"
        )


@pytest.mark.parametrize("dim,rod,deep", _POINTS, ids=lambda v: str(v))
@pytest.mark.parametrize("seed", _SEEDS)
def test_the_untyped_no_op_auxiliary_moves_no_bits(dim, rod, deep, seed) -> None:
    """`auxiliary.cutoff == run.upper_cutoff` against no `auxiliary` block."""
    base = _params(dim, rod, deep)
    n_seed = 120 if dim == 2 else 400
    off = _run(copy.deepcopy(base), seed, dim, n_seed)
    assert off[0].aux_cutoff is None and off[0].bias is None
    assert off[4] == 0, "the auxiliary-off arm evaluated a correction"

    aux = copy.deepcopy(base)
    aux["auxiliary"] = {"type": "short_range", "cutoff": off[0].upper_cutoff}
    on = _run(aux, seed, dim, n_seed)
    assert on[0].aux_cutoff == on[0].upper_cutoff
    assert on[0].potential_type == off[0].potential_type == "ljbranch"

    _assert_exact_no_op(f"{dim}D {rod} {deep} s{seed}", off, on)


# --------------------------------------------------------------- typed mode
#
# The typed arm is not a duplicate. It reaches a DIFFERENT correction: the two
# shell energies go through `_lj_energy_typed`, gathering sigma_ij and eps_ij per
# pair, and the removed group's types come off the cell list at slots that have
# already been tombstoned. Run the untyped correction on a mixture and it would
# compute the sigma = 1 shell difference of a system with no sigma = 1 pair in
# it, which is a wrong number in the acceptance rather than a missing speed-up.
# So the exact-zero assertion is doing real work here: it fails if the types are
# misaligned with the distances, which is the one way this arithmetic can be
# wrong while still looking plausible.


def _typed_ff(tmp_path: Path, aux: object) -> str:
    """The shipped three-species file with its `aux` block replaced."""
    with _SPECIES_FF.open() as fh:
        doc = yaml.safe_load(fh)
    doc["aux"] = aux
    doc["name"] = f"{doc['name']}-aux-{'ctl' if aux != 'none' else 'off'}"
    out = tmp_path / f"species_{doc['name']}.yaml"
    with out.open("w") as fh:
        yaml.safe_dump(doc, fh)
    return str(out)


def _typed_params(ff_path: str, rod: str, depth: int) -> dict:
    sampler: dict = {
        "type": rod,
        "depth": depth,
        "trial_density": 1.0,
        "channel_width": 1.5,
    }
    if rod == "channel":
        sampler["channel_length"] = 6.0
    return {
        "potential": {
            "main": "lennard_jones",
            "T": 0.85,
            "species_file": ff_path,
        },
        "sampler": sampler,
        "ensemble": {
            "mu": {"A": 1.0, "B": 1.0, "C": 1.0},
            "dimension": 3,
            "box": 12.0,
        },
        "run": {"sampling_window": 40},
    }


@pytest.mark.parametrize("rod", ("channel", "channel_cyclic"))
@pytest.mark.parametrize("depth", (1, 2))
@pytest.mark.parametrize("seed", (31, 32, 33))
def test_the_typed_control_arm_moves_no_bits(rod, depth, seed, tmp_path) -> None:
    """`aux: {control: true}` in the species file against `aux: none`.

    The control block's whole documented purpose is to be "the exact no-op arm
    the tuning sweep uses as its own reference", and until S4 it produced a run
    with no auxiliary machinery at all. It now sets the per-pair radius to `rc`
    for every pair, which is the shortest description of "on, and exactly zero".
    """
    off = _run(_typed_params(_typed_ff(tmp_path, "none"), rod, depth), seed, 3, 300)
    assert off[0].aux_cutoff is None, "aux: none must leave no selection radius"
    assert off[4] == 0

    ctl = _run(
        _typed_params(_typed_ff(tmp_path, {"control": True}), rod, depth), seed, 3, 300
    )
    assert ctl[0].aux_cutoff == ctl[0].upper_cutoff, (
        "the control arm must configure the auxiliary AT the physics cutoff; "
        "anything else is either a real auxiliary or no auxiliary"
    )
    assert ctl[0].species is not None and ctl[0].n_species == 3

    _assert_exact_no_op(f"typed {rod} d{depth} s{seed}", off, ctl)

"""The pool's one-body weight: three implementations, one number.

Section 1.1 of the design plan puts four factors into every pool row's
weight -- the activity z_s, the window volume V_W, the species-draw probability
1/p_s, and the one-body energy (framework PLUS frozen exterior guests, never the
in-window guests) -- and says each enters EXACTLY ONCE. Three pieces of code
compute it:

    framework.py::_exterior_logweights          the shipped MOF reference
    channel_move.pool_one_body_logweights       the definition, per point
    channel_move.pool_one_body_logweights_batched   the engine's fast path

The first sits behind golden pins and is deliberately NOT rewritten to call the
others: a one-ulp change in its summation order would move them. So the
agreement is asserted here instead, which is a STRONGER statement than a shared
call would be -- a shared implementation makes them equal by construction, a
test makes the equality something that can fail.

The batched form exists because the per-point form does M cell-list queries in a
Python loop and at M = 17 that was most of what a window move cost. A fast path
with no oracle is how a factor of V_W goes missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcmc.forcefield.lj_table import FrameworkField
from gcmc.forcefield.pair_table import PairTable
from gcmc.forcefield.species_file import load_forcefield, reduce_tables
from gcmc.proposals.channel_move import (
    frozen_neighbours_from_array,
    frozen_neighbours_from_cell_list,
    pool_one_body_logweights,
    pool_one_body_logweights_batched,
)
from gcmc.space.neighbors import CellList


@pytest.fixture(scope="module")
def env():
    ff = load_forcefield("ff/forcefield_A.yaml")
    red = reduce_tables(ff, 298.0)
    table = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, ff.core_floor_c
    )
    rng = np.random.default_rng(0)
    L = 14.0
    box = np.array([L, L, L])
    fw = table.index("In")
    field = FrameworkField(
        rng.uniform(0, L, size=(40, 3)),
        np.full(40, fw),
        table,
        box,
        ff.rc / red.sigma_ref,
    )
    return {
        "field": field,
        "table": table,
        "box": box,
        "rc": ff.rc / red.sigma_ref,
        "beta": 1.0 / red.T_star,
        "xe": table.index("Xe"),
        "kr": table.index("Kr"),
    }


@pytest.mark.parametrize("with_field", [False, True])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("lj", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_the_batched_fast_path_matches_the_definition(env, typed, lj, seed, with_field):
    """The engine's batched form against the per-point definition.

    Over a real cell list, because that is what the engine hands it and because
    the cell list is where the two could differ: it returns neighbours in cell
    order rather than in array order, so the exterior sum is a sum over the SAME
    values in a DIFFERENT order. Floating-point addition is not associative, so
    the tolerance here is a few ulps rather than zero -- and that is stated
    rather than hidden, because a mismatch larger than that is a missing term
    and not a summation order.

    `with_field` adds the framework, which is how the rigid-framework driver
    calls the engine's window move: the host's overlap and energy must enter the
    batched form exactly as they enter the definition.
    """
    if with_field and not typed:
        pytest.skip("a framework is a typed table; the untyped path has no host")
    rng = np.random.default_rng(20 + seed)
    box, table, beta = env["box"], env["table"], env["beta"]
    gt = [env["xe"], env["kr"]] if typed else [0]
    tab = table if typed else None
    floor = table.core_floor if typed else 1.0
    radius = env["rc"] if lj else (float(np.max(table.core_floor)) if typed else 1.0)

    n_ext, n_pool = 60, 16
    ext_P = rng.uniform(0, box[0], size=(n_ext, 3))
    ext_s = np.asarray(rng.choice(gt, size=n_ext), dtype=np.int64)
    cl = CellList.from_positions(ext_P, box, 3, max(radius, 1.0))
    for i, sp in enumerate(ext_s):
        cl.species[i] = int(sp)
    pool_P = rng.uniform(0, box[0], size=(n_pool, 3))
    pool_s = np.asarray(rng.choice(gt, size=n_pool), dtype=np.int64)
    if with_field:
        # a quarter of the rows on top of framework atoms, so overlap rejection
        # is exercised, and a quarter just outside their floor, so the field's
        # steep repulsive energy is too
        fp = env["field"].pos
        pool_P[:4] = (fp[:4] + 0.3) % box
        pool_P[4:8] = (fp[4:8] + 0.75) % box

    log_z = {t: 0.3 * (1 + i) for i, t in enumerate(gt)}
    log_p = {t: float(np.log(1.0 / len(gt))) for t in gt}
    nt = 1 + max(gt)
    log_z_arr = np.zeros(nt)
    log_p_arr = np.zeros(nt)
    for t in gt:
        log_z_arr[t] = log_z[t]
        log_p_arr[t] = log_p[t]

    want_w, want_a = pool_one_body_logweights(
        pool_P,
        pool_s,
        beta=beta,
        log_z=log_z,
        log_volume=1.7,
        log_p=log_p,
        neighbours=frozen_neighbours_from_cell_list(cl, radius),
        floor=floor,
        field=env["field"] if with_field else None,
        table=tab,
        lj=lj,
    )
    got_w, got_a, _uh, _ug = pool_one_body_logweights_batched(
        pool_P,
        pool_s,
        cl=cl,
        radius=radius,
        beta=beta,
        log_z_arr=log_z_arr,
        log_volume=1.7,
        log_p_arr=log_p_arr,
        floor=floor,
        table=tab,
        lj=lj,
        field=env["field"] if with_field else None,
    )
    if with_field:
        assert not got_a.all(), "no row met the framework: the case is vacuous"
    assert (got_a == want_a).all(), (
        f"admissibility differs at rows {np.flatnonzero(got_a != want_a)}"
    )
    f = np.isfinite(want_w)
    assert np.array_equal(np.isfinite(got_w), f)
    if f.any():
        dev = np.abs(got_w[f] - want_w[f]).max()
        assert dev < 1e-11 * max(1.0, float(np.abs(want_w[f]).max())), (
            f"max deviation {dev:.3e}: that is a missing term, not a summation order"
        )


def test_the_window_volume_and_the_draw_probability_enter_once(env):
    """V_W and 1/p_s each shift every finite weight by exactly their own log.

    The arithmetic is trivial; the point is the CONTRACT. Section 1.1 says both
    enter exactly once, and the two ways to break that -- putting V_W in the
    member weight AND in the ladder, or forgetting to divide p_s back out -- are
    silent: the first is a constant factor per k that reads as a shifted
    isotherm, the second is a composition drift over long runs (G7.2). A test
    that pins the shift is what makes either visible immediately.
    """
    rng = np.random.default_rng(5)
    box, table, beta = env["box"], env["table"], env["beta"]
    gt = [env["xe"], env["kr"]]
    pool_P = rng.uniform(0, box[0], size=(9, 3))
    pool_s = np.asarray(rng.choice(gt, size=9), dtype=np.int64)
    kw = {
        "beta": beta,
        "log_z": dict.fromkeys(gt, 0.0),
        "neighbours": frozen_neighbours_from_array(
            np.empty((0, 3)), np.empty(0, dtype=np.int64), box, env["rc"]
        ),
        "floor": table.core_floor,
        "field": None,
        "table": table,
        "lj": True,
    }
    base, adm = pool_one_body_logweights(pool_P, pool_s, **kw)  # type: ignore[arg-type]
    with_v, _ = pool_one_body_logweights(pool_P, pool_s, log_volume=2.5, **kw)  # type: ignore[arg-type]
    with_p, _ = pool_one_body_logweights(
        pool_P,
        pool_s,
        log_p=dict.fromkeys(gt, np.log(0.25)),
        **kw,  # type: ignore[arg-type]
    )
    assert np.allclose(with_v[adm] - base[adm], 2.5)
    assert np.allclose(with_p[adm] - base[adm], -np.log(0.25))

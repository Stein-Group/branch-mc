"""The bulk engine's ladder window move inside a rigid framework.

WHAT IS NEW HERE IS ONLY THE HOST. The window move is `glauber_dynamics.LadderMove`,
the same object the bulk engine runs and pins; `rigid_framework.framework_ladder_move`
configures it with a `FrameworkField`, which enters the one-body weight of each
pool row and nowhere else. So the gates are about that one entry point and about
the two bodies the production runs use:

  * every committed configuration is legal against the framework and at the
    per-pair floor, for both bodies and both kernels;
  * dN takes both signs and magnitudes above one (the ladder, not a repack);
  * the pore ladder equals brute-force enumeration over the same pool, for the
    rod (transfer matrix) AND the cube cavity (clique algebra), with the framework
    energy inside the member weights;
  * the Suwa-Todo kernel's logged rejection rate never exceeds the heat bath's on
    the same ladders;
  * a window the bulk params file would refuse is refused here too, by the same
    function (`sim_setup.ladder_window` / `assert_channel_guard`).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from gcmc.engine.rigid_framework import framework_ladder_move, ladder_window_moves_ms
from gcmc.forcefield.lj_table import FrameworkField
from gcmc.forcefield.overlap import overlap_radius
from gcmc.forcefield.pair_table import PairTable, table_pair_energy_sq
from gcmc.forcefield.species_file import load_forcefield, reduce_tables
from gcmc.framework.driver import ladder_geometry
from gcmc.proposals.channel_move import (
    frozen_neighbours_from_array,
    pool_one_body_logweights,
)
from gcmc.sampling.channel import ladder_probs, window_ladder_logw

BODIES = {
    "channel": {"channel_length": 3.0, "channel_width": 1.0, "trial_density": 4.1},
    "branch": {"trial_box_side": 1.5, "trial_density": 1.5},
}
COMBOS = [(s, e) for s in BODIES for e in ("suwa_todo_dynamics", "glauber_dynamics")]


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
        rng.uniform(0, L, size=(50, 3)),
        np.full(50, fw),
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
        "gt": [table.index("Xe"), table.index("Kr")],
    }


def _seed_guests(env, n, seed=1):
    """n guests placed where they clear the framework, with random species."""
    rng = np.random.default_rng(seed)
    box, field, table = env["box"], env["field"], env["table"]
    gt = env["gt"]
    pos: list[np.ndarray] = []
    spc: list[int] = []
    floor = table.core_floor
    while len(pos) < n:
        p = rng.uniform(0, box[0], 3)
        s = int(rng.choice(gt))
        if field.overlaps(p, s):
            continue
        ok = True
        for q, sq in zip(pos, spc):
            d = q - p
            d -= box * np.round(d / box)
            if float(d @ d) <= overlap_radius(floor[s, sq]) ** 2:
                ok = False
                break
        if ok:
            pos.append(p)
            spc.append(s)
    return np.array(pos), np.array(spc, dtype=np.int64)


def _ladder(env, sampler, engine, **over):
    geo = {**BODIES[sampler], **over}
    smp = {"type": sampler, **geo}
    L, W, M, kmax = ladder_geometry(
        engine, smp, 0, env["box"], env["rc"], env["table"], env["gt"]
    )
    return framework_ladder_move(
        field=env["field"],
        table=env["table"],
        z_star=dict.fromkeys(env["gt"], 0.4),
        beta=env["beta"],
        box=env["box"],
        rc=env["rc"],
        engine=engine,
        sampler=sampler,
        channel_length=L,
        channel_width=W,
        trial_density=smp["trial_density"],
        M=M,
        kmax=kmax,
    )


@pytest.mark.parametrize("sampler,engine", COMBOS)
def test_the_move_commits_a_legal_configuration(env, sampler, engine):
    """No framework overlap, no guest-guest overlap at the PER-PAIR floor, the
    species column aligned to the positions, and the occupancy actually moved."""
    pos, spc = _seed_guests(env, 40, seed=3)
    move = _ladder(env, sampler, engine)
    stats: dict[str, float] = {}
    pos, spc = ladder_window_moves_ms(
        move, pos, spc, np.random.default_rng(11), 120, stats
    )
    assert len(pos) == len(spc)
    box, table, field = env["box"], env["table"], env["field"]
    for i in range(len(pos)):
        assert not field.overlaps(pos[i], int(spc[i])), "committed onto the framework"
    for i, j in itertools.combinations(range(len(pos)), 2):
        d = pos[i] - pos[j]
        d -= box * np.round(d / box)
        fl = table.core_floor[int(spc[i]), int(spc[j])]
        assert float(d @ d) > overlap_radius(fl) ** 2, "committed an overlapping pair"
    assert stats["sum_abs_dN"] > 0.0, "the occupancy never changed"
    assert stats["sum_pool"] == 120.0 * move.M


@pytest.mark.parametrize("sampler", list(BODIES))
def test_dN_takes_both_signs_and_magnitudes_above_one(env, sampler):
    move = _ladder(env, sampler, "suwa_todo_dynamics")
    pos, spc = _seed_guests(env, 40, seed=5)
    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(200):
        n0 = len(pos)
        pos, spc = ladder_window_moves_ms(move, pos, spc, rng, 1)
        seen.add(len(pos) - n0)
    assert min(seen) < 0 < max(seen), f"dN never took both signs: {sorted(seen)}"
    assert max(abs(d) for d in seen) > 1, "dN never exceeded one in magnitude"


def test_suwa_todo_never_repeats_more_than_the_heat_bath(env):
    """The allocation's self-transition rate is the minimum a pi-invariant
    kernel can attain, so over the same moves it keeps k_in less often."""
    unchanged = {}
    for engine in ("glauber_dynamics", "suwa_todo_dynamics"):
        move = _ladder(env, "branch", engine)
        pos, spc = _seed_guests(env, 40, seed=6)
        stats: dict[str, float] = {}
        ladder_window_moves_ms(move, pos, spc, np.random.default_rng(3), 300, stats)
        unchanged[engine] = stats["unchanged"]
    assert unchanged["suwa_todo_dynamics"] < unchanged["glauber_dynamics"]


def _pool(env, rng, center, R, L, w, M, pos, spc):
    """A pool in the body: current occupants first, then fresh trials."""
    from gcmc.space.body import cuboid_members

    box = env["box"]
    inside = cuboid_members(pos, center, R, 0.5 * L, 0.5 * w, box)
    n_new = M - len(inside)
    U = np.column_stack(
        [
            rng.uniform(-0.5 * L, 0.5 * L, n_new),
            rng.uniform(-0.5 * w, 0.5 * w, n_new),
            rng.uniform(-0.5 * w, 0.5 * w, n_new),
        ]
    )
    Pf = np.mod(center + U @ R.T, box)
    sf = np.asarray(rng.choice(env["gt"], size=n_new), dtype=np.int64)
    pool_P = np.vstack([pos[inside], Pf]) if len(inside) else Pf
    pool_s = np.concatenate([spc[inside].astype(np.int64), sf])
    outside = np.ones(len(pos), dtype=bool)
    outside[inside] = False
    return pool_P, pool_s, outside


def _brute_force(env, M, kmax, mw, dist2, pool_s):
    """sum over every subset of the pool of the clique weight times (M-k)!."""
    table, beta, rc = env["table"], env["beta"], env["rc"]
    want = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        for c in itertools.combinations(range(M), k):
            if any(mw[i] <= 0.0 for i in c):
                continue
            wgt = float(np.prod([mw[i] for i in c])) if k else 1.0
            ok = True
            for a, b in itertools.combinations(c, 2):
                d2 = dist2(a, b)
                fl = table.core_floor[pool_s[a], pool_s[b]]
                if d2 <= overlap_radius(fl) ** 2:
                    ok = False
                    break
                if d2 < rc * rc:
                    wgt *= math.exp(
                        -beta
                        * float(
                            table_pair_energy_sq(
                                np.array([d2]), int(pool_s[a]), int(pool_s[b]), table
                            )[0]
                        )
                    )
            if ok:
                want[k] += wgt * math.gamma(M - k + 1)
    return want / want.sum()


@pytest.mark.parametrize("sampler", list(BODIES))
def test_the_pore_ladder_matches_brute_force_enumeration(env, sampler):
    """The pore's ladder against enumeration over the same pool, for both bodies.

    The member weights come from the reference one-body weight WITH the
    framework, so this says the host entered each member weight exactly once and
    never the intra-pool term. The rod's sweep reads local cuboid coordinates and
    the cavity's reads minimum-image distances; each is enumerated in its own.
    """
    from gcmc.proposals.channel_move import ChannelBody
    from gcmc.proposals.trials import TrialSet
    from gcmc.space.body import local_from_box, oriented_cuboid_trials

    box, table, beta, rc = env["box"], env["table"], env["beta"], env["rc"]
    gt = env["gt"]
    pos, spc = _seed_guests(env, 25, seed=9)
    rng = np.random.default_rng(4)
    if sampler == "channel":
        L, w, M, kmax = 3.0, 0.9, 9, 4
    else:
        L = w = 1.5
        M, kmax = 9, 5
    center = box / 2.0
    _, R = oriented_cuboid_trials(rng, center, 1, L, w, box)
    pool_P, pool_s, outside = _pool(env, rng, center, R, L, w, M, pos, spc)
    logw, adm = pool_one_body_logweights(
        pool_P,
        pool_s,
        beta=beta,
        log_z=dict.fromkeys(gt, math.log(0.4)),
        log_volume=math.log(L * w * w),
        log_p=dict.fromkeys(gt, math.log(0.5)),
        neighbours=frozen_neighbours_from_array(
            pos[outside], spc[outside].astype(np.int64), box, rc
        ),
        floor=table.core_floor,
        field=env["field"],
        table=table,
        lj=True,
    )
    mw = np.where(adm, np.exp(np.clip(logw, -700, 700)), 0.0)
    body = ChannelBody.build(
        center,
        R,
        box,
        box,
        3,
        channel_length=L,
        channel_width=w,
        trial_density=M / (L * w * w),
        n_trials_base=M,
        n_trials_min=M,
        cavity=sampler == "branch",
        full=False,
        reach=rc,
    )
    sweep = body.make_sweep(
        TrialSet(coords=pool_P, energy_weight=mw, species=pool_s),
        kmax,
        ljbranch=True,
        lower=float(table.core_floor.min()),
        rc_sel=rc,
        beta=beta,
        floor=table.core_floor,
        table=table,
        admissible=adm,
    )
    got = ladder_probs(window_ladder_logw(sweep, kmax, pool_size=M))

    if sampler == "channel":
        U = local_from_box(pool_P, center, R, box)

        def dist2(a, b):
            d = U[a] - U[b]
            return float(d @ d)

    else:

        def dist2(a, b):
            d = pool_P[a] - pool_P[b]
            d -= box * np.round(d / box)
            return float(d @ d)

    want = _brute_force(env, M, kmax, mw, dist2, pool_s)
    dev = float(np.abs(got - want).max())
    assert dev < 1e-10, f"{sampler}: {got} vs {want}"


def test_a_window_the_bulk_engine_refuses_is_refused_here(env):
    """Same function, same refusal: a rod too long for the box (periodic guard),
    and a cavity over the per-move cost budget."""
    with pytest.raises(ValueError, match="periodic guard"):
        _ladder(env, "channel", "suwa_todo_dynamics", channel_length=13.0)
    with pytest.raises(ValueError, match="per window move"):
        _ladder(
            env, "branch", "suwa_todo_dynamics", trial_box_side=2.5, trial_density=4.0
        )

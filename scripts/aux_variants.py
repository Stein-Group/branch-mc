#!/usr/bin/env python3
"""Does the 3D bias point at good positions, and would a small change help?

    python aux_variants.py runs/SR_aux_3d/<dir>/params.json --windows 2

WHAT IS BEING TESTED. A selection bias is useful only if it RANKS the admissible
trials the way the move would want them ranked. `aux_pool_probe` says whether
there is a pool and whether the bias has spread; this says whether the spread
points the right way, and it compares variants of the bias on the same pools so
a change can be judged before a single production job is launched.

THE TARGET IT IS SCORED AGAINST. For a whole-cavity repack the move must place
SEVERAL molecules, so a good first choice is one that leaves room for the rest.
So the per-trial target here is

    room(i) = how many of the OTHER admissible trials survive if trial i is
              placed, i.e. #{ j != i : |x_i - x_j| >= sigma }

which is the level-2 pool this choice would hand on. It is the cheapest honest
statement of "does this choice paint the deeper levels into a corner", it needs
nothing but the pool's own coordinates, and it is exactly the quantity a greedy
bias can destroy. A bias that correlates NEGATIVELY with it is not weak; it is
pointed backwards, and at large trial-box side, where a move places many
molecules, that is worse than no bias at all.

Reported per variant: Spearman rho against room, and Var(log b). Rho says
whether the ranking is right, the variance says how hard the exponent will push
on it. A variant with a better rho at a smaller variance is strictly better: the
same information with less greed.

THE VARIANTS, all small and all legal -- the bias is divided back out of the
acceptance, so any strictly positive deterministic function of the geometry
leaves the sampled ensemble exactly unchanged, and only the variance moves.

    sum        the shipped form: sum over neighbours of log g3(r, cos), the
               angle taken against the NEAREST neighbour
    mean       the same divided by the neighbour count. The shipped form's
               magnitude grows with how many neighbours a trial has, so it
               partly ranks trials by local crowding rather than by geometry,
               and the exponent then pushes hardest exactly where the count is
               highest
    trunc      the sum restricted to r < 1.9. Measured on this table, pairs
               beyond r = 1.8 carry 4% of the total Var(log g3) while being
               66% of the pairs: they add cost and count-dependence, not
               ranking
    smoothref  the angle taken against the MEAN neighbour direction instead of
               the single nearest one. At high density several neighbours sit at
               nearly the same distance, so "nearest" flips between them under a
               tiny displacement of the trial and every cos flips with it. That
               makes the shipped bias discontinuous in position: spread that
               carries no information, which is the worst kind
    mean+trunc both of the above
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from gcmc import config_specs
from gcmc.engine import metropolis_dynamics as md
from gcmc.run import runner, sim_setup
from gcmc.system.state import ChainState

#: Where the table stops carrying angular information, measured on
#: data/g3_analysis_data.csv: sd(log g3) over cos falls from 2.8 at contact to
#: 0.26 by r = 1.975, and pairs beyond 1.8 hold 4% of the total variance.
R_TRUNC = 1.9


def _geometry(aux, dists, coords, idxs, kdtree):
    """(r, cos_nearest, cos_meandir, valid) per (trial, neighbour j>=1).

    Both angle conventions from one pass, because the only honest comparison of
    the two is on identical neighbour sets.
    """
    data = kdtree.data
    neigh = data[np.minimum(idxs, data.shape[0] - 1)]
    d = neigh - coords[:, None, :]
    d -= aux.side * np.round(d / aux.side)
    r = dists
    valid = np.isfinite(r)
    ba = d[:, 0, :]
    bc = d[:, 1:, :]
    n_ba = np.sqrt(np.einsum("ij,ij->i", ba, ba))
    dot = np.einsum("ij,ikj->ik", ba, bc)
    rr = r[:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_near = np.clip(dot / (n_ba[:, None] * rr), -1.0, 1.0)
        # the MEAN neighbour direction: sum of unit vectors over every in-range
        # neighbour, which moves continuously as the trial moves
        unit = np.where(
            valid[:, :, None], d / np.where(r[:, :, None] > 0, r[:, :, None], 1.0), 0.0
        )
        mean_dir = unit.sum(axis=1)
        n_md = np.sqrt(np.einsum("ij,ij->i", mean_dir, mean_dir))
        n_md = np.where(n_md > 1e-12, n_md, 1.0)
        dot_m = np.einsum("ij,ikj->ik", mean_dir, bc)
        cos_mean = np.clip(dot_m / (n_md[:, None] * rr), -1.0, 1.0)
    return rr, cos_near, cos_mean, valid[:, 1:]


def _bias(aux, rr, cos_t, keep, clip=None):
    """log b per trial from one (r, cos) set, floored as the shipped bias is.

    `clip` bounds each PAIR's contribution to +/- that many log units. The
    shipped floor is one-sided: a single bad angle multiplies a trial's weight
    by min_g3 and vetoes it outright, while no single good angle can carry a
    trial the same distance the other way. A symmetric clip makes one pair's
    opinion bounded in both directions, which is a softer floor and costs one
    call.
    """
    g3 = aux._interp.at(rr, cos_t)
    np.maximum(g3, aux.min_g3, out=g3)
    lg = np.log(g3)
    if clip is not None:
        np.clip(lg, -clip, clip, out=lg)
    lg = np.where(keep, lg, 0.0)
    n = np.maximum(keep.sum(axis=1), 1)
    return lg.sum(axis=1), n


def _nearest_m(keep, m):
    """Keep only the m nearest in-range neighbours per trial. The query returns
    distances sorted ascending, so a cumulative count is the whole test."""
    return keep & (np.cumsum(keep, axis=1) <= m)


#: Print order, so a reader compares like with like. `sum` first because it is
#: the shipped bias and every other line is judged against it.
ORDER = (
    "sum",
    "trunc@2.0",
    "trunc@1.9",
    "trunc@1.8",
    "trunc@1.6",
    "trunc@1.4",
    "nearest4",
    "nearest6",
    "nearest10",
    "clip1",
    "clip2",
    "clip2+trunc@1.6",
    "clip1+trunc@1.8",
    "clip1+nearest10",
    "mean",
    "mean+trunc@1.6",
    "smoothref",
)


def variants(aux, dists, coords, idxs, kdtree):
    """Every variant on ONE neighbour set, the only fair comparison there is."""
    rr, cos_near, cos_mean, keep = _geometry(aux, dists, coords, idxs, kdtree)
    out = {}
    s_sum, n_all = _bias(aux, rr, cos_near, keep)
    out["sum"] = s_sum
    out["mean"] = s_sum / n_all
    out["smoothref"] = _bias(aux, rr, cos_mean, keep)[0]
    for rc in (2.0, 1.9, 1.8, 1.6, 1.4):
        out[f"trunc@{rc}"] = _bias(aux, rr, cos_near, keep & (rr < rc))[0]
    for m in (4, 6, 10):
        out[f"nearest{m}"] = _bias(aux, rr, cos_near, _nearest_m(keep, m))[0]
    for c in (1.0, 2.0):
        out[f"clip{c:g}"] = _bias(aux, rr, cos_near, keep, clip=c)[0]
    k16 = keep & (rr < 1.6)
    out["clip2+trunc@1.6"] = _bias(aux, rr, cos_near, k16, clip=2.0)[0]
    s16, n16 = _bias(aux, rr, cos_near, k16)
    out["mean+trunc@1.6"] = s16 / n16
    # the two best single changes, combined: dropping the uninformative far
    # pairs and bounding what any one pair can veto
    k18 = keep & (rr < 1.8)
    out["clip1+trunc@1.8"] = _bias(aux, rr, cos_near, k18, clip=1.0)[0]
    out["clip1+nearest10"] = _bias(aux, rr, cos_near, _nearest_m(keep, 10), clip=1.0)[0]
    return out


def room(coords, side, diameter):
    """How many of the other trials survive if this one is placed."""
    d = coords[:, None, :] - coords[None, :, :]
    d -= side * np.round(d / side)
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    np.fill_diagonal(r, np.inf)
    return (r >= diameter).sum(axis=1).astype(float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("params", type=Path)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument(
        "--min-pool",
        type=int,
        default=6,
        help="ignore pools smaller than this: a rank correlation "
        "over three points is noise",
    )
    a = ap.parse_args(argv)

    cfg = sim_setup.from_params(json.loads(a.params.read_text()))
    pot = cfg.build_potential()
    aux = getattr(pot, "aux", pot)
    if not hasattr(aux, "_g3_grid_bulk"):
        print("this params.json is not a 3D tabulated pmf run")
        return 2
    wgh = md.make_weigher(config_specs.kernel_spec(cfg), pot)
    st = ChainState.fresh(cfg.dimension, seed=cfg.seed)
    st.positions = runner._init_config(cfg, np.random.default_rng(cfg.seed))
    print(
        f"start: {len(st.positions)} molecules, k={cfg.n_trial_molecules} "
        f"L={cfg.trial_box_side}"
    )

    grabbed: list[tuple] = []
    orig = aux._g3_grid_bulk

    def spy(dists, coords, idxs, kdtree):
        if kdtree is not None and len(coords) >= a.min_pool:
            grabbed.append((dists.copy(), coords.copy(), idxs.copy(), kdtree))
        return orig(dists, coords, idxs, kdtree)

    aux._g3_grid_bulk = spy
    try:
        [md.advance(st, cfg, pot, wgh) for _ in range(a.windows)]
    finally:
        aux._g3_grid_bulk = orig

    if not grabbed:
        print(
            f"no level-1 pool reached {a.min_pool} admissible trials. Run "
            f"aux_pool_probe first: there is nothing here to rank."
        )
        return 1

    diameter = config_specs.kernel_spec(cfg).diameter
    rows: dict[str, list[float]] = {}
    var: dict[str, list[float]] = {}
    sizes = []
    for dists, coords, idxs, tree in grabbed:
        tgt = room(coords, cfg.box, diameter)
        if np.ptp(tgt) == 0:
            continue  # every choice leaves the same room: nothing to rank
        sizes.append(len(coords))
        for name, lb in variants(aux, dists, coords, idxs, tree).items():
            if not np.all(np.isfinite(lb)) or np.ptp(lb) == 0:
                continue
            rows.setdefault(name, []).append(float(spearmanr(lb, tgt).statistic))
            var.setdefault(name, []).append(float(np.var(lb)))

    if not rows:
        print(
            "no pool had both a spread in the bias and a spread in the "
            "target; nothing to compare"
        )
        return 1
    print(f"\n{len(sizes)} pools, median {np.median(sizes):.0f} admissible trials")
    print(
        "\nrho > 0 means the bias prefers positions that LEAVE ROOM for the "
        "molecules the\nmove still has to place. rho < 0 means it prefers "
        "positions that block them.\n"
    )
    print(
        f"{'variant':<20}{'rho vs room':>13}{'Var(log b)':>13}"
    )
    print("-" * 46)
    base = None
    for name in ORDER:
        if name not in rows:
            continue
        rho = float(np.median(rows[name]))
        v = float(np.median(var[name]))
        if base is None:
            base = rho
        flag = (
            ""
            if name == "sum"
            else (
                "  better"
                if rho > base + 0.02
                else "  worse"
                if rho < base - 0.02
                else "  same"
            )
        )
        print(f"{name:<20}{rho:>13.3f}{v:>13.3f}{flag}")
    print("-" * 46)
    print(
        "`sum` is the shipped bias. A variant is worth trying in production "
        "when its rho\nis clearly higher, and worth preferring when it also "
        "reaches the same rho at a\nlower variance: the same ranking with "
        "less greed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

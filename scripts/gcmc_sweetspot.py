#!/usr/bin/env python3
"""
Sweet-spot heatmaps over (trial box side L) x (trial pool size k), one row per
method, two columns (a third, t/step, can be switched back on in COLUMNS):

    acc / step     accepted exchanges per exchange move,  as a multiple of std GCMC
    acc / second   accepted exchanges per CPU-second,      as a multiple of std GCMC

acc/sec is acc/step divided by t/step, exactly, because all three are ratios
against the same reference run. A method buys a higher acceptance per move and
pays for it in seconds per move; only the quotient says whether the trade was
worth making.

ACCEPTANCE. Every run here is accept/reject: acceptance is
(add_acc + rm_acc) / (add_stp + rm_stp). An accepted single-particle exchange
moves N by exactly one, so this is also <dN^2> per move for these engines.
Ladder engines (Glauber, Suwa-Todo) have no accept/reject step and are skipped.

THE REFERENCE IS READ FROM DISK, not from a table pasted into this file. Set
STD_REF_DIRS below to the std-GCMC run directories. Reference and method runs
are matched on (dimension, potential, T) and interpolated in mu, so the
reference sweep does not have to sit on the same mu grid -- but it does have to
cover it. A record whose mu falls outside the reference range gets NaN and its
cell is left white, rather than being silently divided by the nearest endpoint.

Layout expected under each path: either a leaf directory holding params.json and
run.log, or a directory of such leaves.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, NullLocator, FuncFormatter
from matplotlib import gridspec

# ============================================================================
# CONFIG
# ============================================================================
RUNS_ROOT = "runs"

#: One row per entry: (directory under RUNS_ROOT, row label, colour key).
#: Colour keys are defined in CMAPS below. Order here is top-to-bottom order in
#: the figure.
#: (directory under RUNS_ROOT, row label, colour family, y axis).
#: The y axis is optional and defaults to Y_KEY -- give it when a row sweeps
#: something else. The aux-pot runs hold k fixed at 200 and sweep gamma, so
#: plotting them against k stacks the whole sweep into one row of cells.


ROWS = [
    ("SR_2d",     "SR 2D",              "green"),
    ("SR_aux_2d", "SR+aux 2D", "purple"),
    ("branch_2d", "BRANCH-MC 2D",       "blue"),
    ("SR_3d",     "SR 3D",              "green"),
    ("SR_aux_3d", "SR+aux 3D", "purple"),
    ("branch_3d", "BRANCH-MC 3D",       "blue"),
]

#: THE STANDARD GCMC REFERENCE. Every number in the figure is divided by one of
#: these runs. Give a group directory or a single leaf; several are fine and are
#: pooled, since a 2D and a 3D reference are two different state spaces and both
#: are needed when the rows mix dimensions. Paths are taken as given if absolute
#: or if they exist as written, otherwise looked up under RUNS_ROOT.
STD_REF_DIRS = [
    "std_2d",
    "std_3d",
]

#: Axis scales, "log" or "linear". These only set how the cells are SPACED and
#: where the edges fall; the colour of a cell never changes. Linear is the
#: default because these sweeps are even grids -- a log axis squeezes the top
#: of the sweep into a sliver and invents a resolution at the bottom that the
#: runs do not have. It also cannot show gamma at all, since gamma starts at 0.
X_SCALE = "linear"
Y_SCALE = "linear"

X_KEY = "trial_box_side"   # heatmap x
Y_KEY = "trials"           # heatmap y
SLICE_KEY = "depth"        # the third sampler axis: one slice of it is drawn

#: Which slice to draw. "best" takes, per row, the depth whose best cell has the
#: highest acceptance per second; a number forces that depth everywhere.
SLICE_PICK = "best"

BURN_IN_M = 0.0            # drop rows with MC_cycle(M) below this
BURN_IN_FRACTION = 0.5     # AND drop this leading fraction of what is left.
                           # These runs start from a lattice at a guessed N and
                           # the approach to the stationary density is itself a
                           # long correlation; leaving any of it in inflates the
                           # per-step cost and deflates the jump.
N_BLOCKS = 20              # blocks for the error on each ratio
MIN_MU, MAX_MU = -6, 15

#: Draw dashed contours on the efficiency column at these fractions of that
#: panel's maximum, so "the sweet spot" is a region and not a pixel. Empty to
#: switch off.
CONTOUR_FRACTIONS = (0.5, 0.9)

#: Second x axis on top of each panel: the mean number of particles inside a
#: trial box of side L, <N> = rho L^d with rho = eta / v_particle. That is the
#: quantity the sampler actually feels -- an L of 1.6 means nothing until you
#: know how many neighbours it encloses -- and it is a pure rescaling of the
#: same axis, so no cell moves.
SHOW_N_AXIS = True
PARTICLE_RADIUS = 0.5

#: Packing fraction per (dimension, mu), measured at the state point. Anything
#: not listed falls back to the mean of the run's own pckng_frctn column, which
#: is the same number as long as the run is equilibrated.
PACKING_FRACTION = {
    (2, 10): 0.647,     # 2D hard disks at mu = 10
    (3, 12): 0.4473,    # 3D hard spheres at mu = 12
}

#: Base point size. Everything on the figure is sized from this and nothing is
#: drawn below it, so the layout has to make room rather than the text shrink.
#: Enforced floor of 20 -- a smaller value here is treated as 20.
FONT_PT = 20

OUTDIR = "figs"
OUTNAME = "sweetspot_acc.png"
DPI = 300
#: Size of the PLOT BOX itself, in inches. Axis labels, ticks, titles and the
#: colour bar are added AROUND this, measured at the font actually in use, so
#: these two numbers stay honest whatever FONT_PT is set to.
PANEL_W, PANEL_H = 3.6, 2.9

#: Colour families. Each one gets its OWN colour scale per column (see
#: SHARED_NORM), so add a family whenever a set of rows lives on a different
#: order of magnitude from the rest. (low, high, accent, short name for the bar)
CMAPS = {
    "green":  ("#ffffff", "#0b6b4f", "#029E73", "SR"),
    "blue":   ("#ffffff", "#0a4a73", "#0173B2", "BRANCH-MC"),
    "purple": ("#ffffff", "#4B2A73", "#7A57B5", "SR+aux"),
    "orange": ("#ffffff", "#8a4200", "#D55E00", "other"),
}

#: How far a colour scale reaches.
#:   "row"    every panel is scaled to its own data and carries its own bar
#:            beside it. Best contrast, and no shade compares to anything
#:            outside its own panel.
#:   "family" one scale per colour family per column, bars stacked underneath.
#:   "all"    one scale per column for everybody: shades comparable across the
#:            whole figure, structure inside a row flattened.
#: Whatever this is, only the printed multiples compare across panels safely.
NORM_SCOPE = "row"

#: Split the rows into separate figures. "dim" gives one file per dimension,
#: None puts everything in one. The tag lands in the file name.
SPLIT_BY = "dim"

#: WHAT "ONE STEP" IS. This choice cancels out of the third column entirely --
#: acc/second is (acc/step)/(t/step) and the step unit divides away -- but it
#: sets the first two columns, and it is the only place a t/step below 1 can
#: come from other than the hardware.
#:
#:    "exchange"  add_stp + rm_stp, or glb_step. The proposal count.
#:    "all"       the above plus mv_stp, so the denominator covers every move
#:                the wall clock was spent on, displacement moves included.
#:    "cycle"     MC cycles. Nothing is left out and nothing engine-specific is
#:                counted, but a cycle is only comparable if both codes define
#:                it the same way.
STEP_DEFINITION = "exchange"

#: How many cells of the printed sweet-spot table per row. More than one
#: because the top cell is usually inside a plateau, and knowing the plateau is
#: worth more than knowing its argmax to three digits.
TOP_N = 3

#: Sampler parameters printed next to each sweet spot, when a run has them.
#: Anything numeric in the sampler block is captured, so a new knob (gamma,
#: window width, whatever) shows up here without touching the reader.
REPORT_KEYS = ["trial_box_side", "trials", "pool_M", "depth", "gamma", "eta",
               "trial_density", "channel_length", "channel_width"]

#: Print the absolute per-run numbers behind every ratio. Leave this on until
#: the figure stops surprising you: a ratio cannot tell you whether it moved
#: because the numerator rose or the denominator sank.
DIAGNOSTICS = True

#: Every metric that is COMPUTED and divided by the reference. t/step stays in
#: here whether or not it is drawn: it is the denominator the other two are
#: related through, and the diagnostics and the cheaper-than-reference warning
#: both need it.
METRICS = [
    ("acc_per_step",    r"acc/step"),
    ("t_per_step",      r"t/step"),
    ("acc_per_second",  r"acc/sec"),
]

#: Which of them become COLUMNS, in this order. Add "t_per_step" back for the
#: three-column version.
COLUMNS = ["acc_per_step", "acc_per_second"]

EFFICIENCY = "acc_per_second"   # the column the star and the contours follow


def columns():
    lab = dict(METRICS)
    return [(k, lab[k]) for k in COLUMNS]

AXIS_LABEL = {
    "trial_box_side": r"trial box $L$",
    "trials":         r"trials $k$",
    "trial_density":  r"trial density",
    "depth":          r"depth",
    "gamma":          r"$\gamma$",
    "pool_M":         r"pool $M$",
    "mu":             r"$\mu$",
    "box":            r"box $L_{\mathrm{box}}$",
}
# ============================================================================

CBAR_OFF = 0.045      # colour bar offset from the panel's right edge, in panel widths
CBAR_W = 0.05         # its width, same units

LADDER_ENGINES = ("glauber_dynamics", "suwa_todo_dynamics")
DEFAULT_ENGINE = "metropolis_dynamics"

LEGACY_POSITIONS = ["MC_cycle(M)", "pckng_frctn", "N", "add_acc", "add_att",
                    "add_stp", "rm_acc", "rm_stp", "mv_acc", "mv_stp", "wall_s"]
LEGACY_COLUMNS = {
    "wnd_stp": "glb_step", "wnd_id": "glb_same", "wnd_dN": "glb_dN",
    "wnd_ovk": "glb_cut_k", "wnd_ovf": "glb_cut_M", "wnd_inf": "glb_infeas",
    "wnd_maxk": "glb_kmax", "wnd_pkmax": "glb_p_kmax", "wnd_H": "glb_entropy",
    "nhat_sum": "glb_nhat_sum", "nhat_sq": "glb_nhat_sq",
    "nhat_wthn": "glb_nhat_wthn", "nhat_n": "glb_nhat_n",
}


# ----------------------------------------------------------------------------
# reading one run
# ----------------------------------------------------------------------------
def read_log(path):
    """One run.log as named float columns. The header is the authority; a
    headerless log falls back to the old positional layout, and a header that
    disagrees with its own rows is refused instead of read by position."""
    with open(path) as fh:
        head = fh.readline()
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.size == 0:
        return {}
    if head.startswith("#"):
        names = head.lstrip("#").split()
        if rows.shape[1] != len(names):
            raise ValueError(f"header names {len(names)} columns, rows have "
                             f"{rows.shape[1]}: log written by another version")
    else:
        names = LEGACY_POSITIONS[:rows.shape[1]]
        if rows.shape[1] > len(LEGACY_POSITIONS):
            raise ValueError("headerless log with more columns than the legacy layout")
    names = [LEGACY_COLUMNS.get(n, n) for n in names]
    return {n: rows[:, i] for i, n in enumerate(names)}


def resolve_engine(p):
    """The dynamics axis, from either layout: a block of its own, or the old key
    inside the sampler. Absent from both means the shipped dynamics -- every run
    predating the engine axis is Metropolis and says so by saying nothing."""
    eng = p.get("engine")
    if isinstance(eng, dict):
        eng = eng.get("type")
    elif not isinstance(eng, str):
        eng = None
    if not eng:
        eng = (p.get("sampler") or {}).get("engine")
    return eng or DEFAULT_ENGINE


def read_params(leaf):
    p = json.loads((leaf / "params.json").read_text())
    ens, pot, smp = p.get("ensemble", {}), p.get("potential", {}), p.get("sampler", {})
    aux = p.get("auxiliary", {}) or {}
    mu = ens.get("mu")
    if isinstance(mu, dict):
        mu = next(iter(mu.values()))
    dim = int(ens["dimension"])
    pool_M = np.nan
    stype = smp.get("type")
    if "trial_density" in smp:
        td = float(smp["trial_density"])
        if stype == "channel" and not isinstance(smp.get("channel_length"), str):
            # "full" has no configured length, so no pool size to report here:
            # its body, and therefore M, is drawn per move.
            pool_M = round(td * float(smp["channel_length"])
                           * float(smp["channel_width"]) ** (dim - 1))
        elif stype in ("branch", "rosenbluth", "reweight") and "trial_box_side" in smp:
            pool_M = round(td * float(smp["trial_box_side"]) ** dim)
    out = {
        "mu": float(mu),
        "dim": dim,
        "box": float(ens.get("box", np.nan)),
        "potential": pot.get("main", "hard_core"),
        "T": float(pot["T"]) if pot.get("T") is not None else None,
        "engine": resolve_engine(p),
        "sampler": smp.get("type", "metropolis"),
        "depth": float(smp.get("depth", np.nan)),
        "trials": float(smp.get("trials", np.nan)),
        "trial_box_side": float(smp.get("trial_box_side", np.nan)),
        "trial_density": float(smp.get("trial_density", np.nan)),
        "pool_M": pool_M,
    }
    out["aux"] = aux.get("type")
    # any other numeric knob rides along under its own name, so an axis or a
    # REPORT_KEYS entry can name it without a reader change. gamma lives in the
    # auxiliary block, not the sampler, which is why both are swept here.
    for block in (smp, aux):
        for k, v in block.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.setdefault(k, float(v))
    return out


def _blocked_ratio(num, den, n_blocks):
    """(sum num / sum den, standard error of that ratio).

    The estimator is the pooled ratio, NOT the mean of per-window ratios: a
    window with almost no moves in it would otherwise carry the same weight as a
    full one. The error is the spread of the same pooled ratio computed inside
    each block, which is the only honest error here because consecutive windows
    of one run are correlated and a naive per-window sem understates it by the
    autocorrelation time."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    tot_d = den.sum()
    if num.size == 0 or tot_d <= 0:
        return np.nan, np.nan
    point = float(num.sum() / tot_d)
    nb = min(n_blocks, num.size)
    if nb < 2:
        return point, np.nan
    edges = np.linspace(0, num.size, nb + 1).astype(int)
    vals = []
    for a, b in zip(edges[:-1], edges[1:]):
        d = den[a:b].sum()
        if d > 0:
            vals.append(num[a:b].sum() / d)
    if len(vals) < 2:
        return point, np.nan
    return point, float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def read_run(leaf):
    """One leaf -> a record holding the three metrics and their errors, or None.

    Accepted exchanges over proposed exchanges, per window:
        add_acc + rm_acc  over  add_stp + rm_stp
    Ladder engines have no accept/reject step, so they are skipped rather
    than given a number that means something else.
    """
    par = read_params(leaf)
    col = read_log(leaf / "run.log")
    if not col or "MC_cycle(M)" not in col:
        return None
    keep = col["MC_cycle(M)"] >= BURN_IN_M
    col = {k: v[keep] for k, v in col.items()}
    if len(col["MC_cycle(M)"]) < 2:
        return None
    cut = int(len(col["MC_cycle(M)"]) * BURN_IN_FRACTION)
    tail = {k: v[cut:] for k, v in col.items()}

    if "wall_s" not in tail:
        print(f"  ! {leaf.name}: no wall_s column -- cannot cost anything, skipped")
        return None
    wall = tail["wall_s"]

    if par["engine"] in LADDER_ENGINES:
        print(f"  ! {leaf.name}: engine={par['engine']} has no accept/reject step, "
              "so acceptance is undefined. Skipped.")
        return None
    need = ("add_acc", "rm_acc", "add_stp", "rm_stp")
    if not all(c in tail for c in need):
        print(f"  ! {leaf.name}: log is missing some of {need} -- written by "
              "another version? Skipped.")
        return None
    acc = tail["add_acc"] + tail["rm_acc"]
    steps = tail["add_stp"] + tail["rm_stp"]

    # ---- THE DENOMINATOR, and the one place a t/step below 1 can be
    # manufactured. `steps` above is the proposal count; the wall clock, in
    # contrast, covers EVERYTHING the run did in that window -- displacement
    # moves, neighbour rebuilds, logging. If a code spends most of its seconds
    # on work that is not in `steps`, its seconds-per-step is not the cost of
    # its proposal, and dividing two such numbers compares two different
    # mixtures of work rather than two proposals.
    mv = tail.get("mv_stp")
    exch_steps = steps
    if STEP_DEFINITION == "all":
        if mv is not None:
            steps = steps + mv
    elif STEP_DEFINITION == "cycle":
        d = np.diff(tail["MC_cycle(M)"])
        per = float(np.median(d)) * 1e6 if d.size and np.median(d) > 0 else np.nan
        if not np.isfinite(per) or per <= 0:
            print(f"  ! {leaf.name}: MC_cycle(M) does not advance -- cannot use "
                  "STEP_DEFINITION='cycle'")
            return None
        steps = np.full(wall.shape, per)
    elif STEP_DEFINITION != "exchange":
        raise ValueError(f"STEP_DEFINITION must be exchange|all|cycle, "
                         f"got {STEP_DEFINITION!r}")

    rec = dict(par)
    rec["leaf"] = leaf.name
    rec["n_windows"] = len(wall)
    # absolute numbers, kept so the diagnostics can say whether a ratio moved
    # because of the run or because of the reference
    rec["packing"] = (float(np.mean(tail["pckng_frctn"]))
                      if "pckng_frctn" in tail else np.nan)
    rec["raw_wall_per_window"] = float(wall.mean())
    rec["raw_steps_per_window"] = float(np.mean(steps))
    rec["raw_exch_per_window"] = float(np.mean(exch_steps))
    rec["raw_cycles_per_window"] = (float(np.median(np.diff(tail["MC_cycle(M)"]))) * 1e6
                                    if len(tail["MC_cycle(M)"]) > 1 else np.nan)
    # what fraction of the counted moves are displacement moves -- i.e. how much
    # of the wall clock the exchange denominator is silently absorbing
    rec["mv_frac"] = (float(mv.sum() / (mv.sum() + exch_steps.sum()))
                      if mv is not None and (mv.sum() + exch_steps.sum()) > 0 else 0.0)
    for key, (num, den) in {
        "acc_per_step":    (acc, steps),
        "t_per_step":      (wall, steps),
        "acc_per_second":  (acc, wall),
    }.items():
        v, e = _blocked_ratio(num, den, N_BLOCKS)
        rec[key], rec[key + "_err"] = v, e
    return rec


def find_leaves(path):
    path = Path(path)
    if (path / "run.log").exists():
        return [path]
    if not path.is_dir():
        return []
    return sorted(d for d in path.iterdir() if d.is_dir() and (d / "run.log").exists())


def resolve_path(p):
    """A config path, taken as written when it resolves, else under RUNS_ROOT."""
    q = Path(p)
    if q.is_absolute() or q.exists():
        return q
    return Path(RUNS_ROOT) / q


def collect(path, what):
    recs = []
    leaves = find_leaves(path)
    if not leaves:
        print(f"  ! {what}: nothing readable under {path}")
        return recs
    for leaf in leaves:
        try:
            r = read_run(leaf)
            if r and MIN_MU <= r["mu"] <= MAX_MU:
                recs.append(r)
        except Exception as e:
            print(f"  ! {leaf.name}: {e}")
    return recs


# ----------------------------------------------------------------------------
# the reference
# ----------------------------------------------------------------------------
def build_reference(paths):
    """({(dim, potential, T): sorted-by-mu arrays}, [every reference record]).

    Runs are keyed by state space, not by folder, so one call covers a 2D and a
    3D reference at once and a row picks whichever matches its own runs."""
    ref, flat = {}, []
    for p in paths:
        path = resolve_path(p)
        print(f"reading reference {path}")
        recs = collect(path, "reference")
        print(f"  {len(recs)} reference runs")
        flat += recs
        for r in recs:
            ref.setdefault((r["dim"], r["potential"], r["T"]), []).append(r)
    out = {}
    for key, recs in ref.items():
        recs.sort(key=lambda r: r["mu"])
        mu = np.array([r["mu"] for r in recs], float)
        entry = {"mu": mu}
        for m, _lab in METRICS:
            entry[m] = np.array([r[m] for r in recs], float)
            entry[m + "_err"] = np.array([r[m + "_err"] for r in recs], float)
        # duplicate mu values (repeats of one state point) are averaged
        uniq = np.unique(mu)
        if uniq.size != mu.size:
            ent = {"mu": uniq}
            for k in list(entry):
                if k == "mu":
                    continue
                ent[k] = np.array([np.nanmean(entry[k][mu == u]) for u in uniq])
            entry = ent
        out[key] = entry
    return out, flat


def ref_at(ref, rec, metric):
    """(reference value, its relative error) for this record's state point.

    Matched on (dim, potential, T) and interpolated in mu. OUTSIDE the reference
    mu range the answer is NaN, deliberately: clamping to the endpoint would
    divide a dense state point by a dilute reference and every number in the
    figure would come out flattering."""
    key = (rec["dim"], rec["potential"], rec["T"])
    entry = ref.get(key)
    if entry is None:                       # T is often None on hard cores
        cand = [v for k, v in ref.items() if k[0] == key[0] and k[1] == key[1]]
        entry = cand[0] if len(cand) == 1 else None
    if entry is None:
        return np.nan, np.nan
    mu, y = entry["mu"], entry[metric]
    if rec["mu"] < mu.min() or rec["mu"] > mu.max():
        return np.nan, np.nan
    v = float(np.interp(rec["mu"], mu, y))
    e = float(np.interp(rec["mu"], mu, entry[metric + "_err"]))
    return v, (e / v if v else np.nan)


def as_multiples(recs, ref):
    """Attach metric/reference and its relative error to every record."""
    unmatched = 0
    for r in recs:
        for m, _lab in METRICS:
            v, rel = ref_at(ref, r, m)
            if not np.isfinite(v) or v <= 0:
                r["x_" + m], r["x_" + m + "_rel"] = np.nan, np.nan
                continue
            r["x_" + m] = r[m] / v
            own = (r[m + "_err"] / r[m]) if r[m] else np.nan
            r["x_" + m + "_rel"] = float(np.hypot(*np.nan_to_num([own, rel])))
        if not np.isfinite(r.get("x_" + EFFICIENCY, np.nan)):
            unmatched += 1
    if unmatched:
        print(f"  ! {unmatched} runs had no reference at their state point "
              "(mu outside the reference sweep, or a dim/potential the reference "
              "does not cover) -- those cells are left white")
    cheap = [r for r in recs if np.isfinite(r.get("x_t_per_step", np.nan))
             and r["x_t_per_step"] < 1.0]
    if cheap:
        best = min(r["x_t_per_step"] for r in cheap)
        mvs = max(r["mv_frac"] for r in recs)
        print(f"  !! {len(cheap)}/{len(recs)} cells come out CHEAPER PER STEP than "
              f"the reference (down to {best:.2g}x). A window resample is not "
              "cheaper than a single-particle trial, so one of the two sides is "
              "not being measured the way you think:")
        print("     - the wall clock covers work the step count does not. "
              f"Displacement moves are {mvs:.0%} of the counted moves in this "
              "folder; if the reference's share is higher, its seconds-per-"
              "exchange is inflated and every cell here is flattered. "
              "Try STEP_DEFINITION='all' or 'cycle'.")
        print("     - wall_s means different things in the two logs "
              "(cumulative vs per-window), which rescales one side by ~n/2.")
        print("     - the two sets of runs were not timed on the same hardware.")
        print("     Column 3 is unaffected by the first of those and destroyed "
              "by the other two.")
    return recs


# ----------------------------------------------------------------------------
# grids
# ----------------------------------------------------------------------------
def val_of(r, key=None):
    """The record's value on an axis. For `trials` it falls back to the pool
    size sim_setup derives from trial_density, which is the same quantity under
    another name; every other key is read as written."""
    key = key or Y_KEY
    v = r.get(key, np.nan)
    if key == "trials" and (v is None or not np.isfinite(v)):
        v = r.get("pool_M", np.nan)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v


def grid_for(recs, metric, ykey=None):
    """(xvals, yvals, Z, Zrel) -- one metric on the L x k grid.

    Repeats of one cell are averaged and said so, because two runs at the same
    (L, k, depth) are replicates and silently keeping the last one throws half
    the statistics away."""
    xs = np.array(sorted({r[X_KEY] for r in recs if np.isfinite(r[X_KEY])}))
    ys = np.array(sorted({val_of(r, ykey) for r in recs
                          if np.isfinite(val_of(r, ykey))}))
    Z = np.full((ys.size, xs.size), np.nan)
    R = np.full((ys.size, xs.size), np.nan)
    if xs.size == 0 or ys.size == 0:
        return xs, ys, Z, R
    bucket = {}
    for r in recs:
        i = np.searchsorted(ys, val_of(r, ykey))
        j = np.searchsorted(xs, r[X_KEY])
        if i >= ys.size or j >= xs.size:
            continue
        bucket.setdefault((i, j), []).append(r)
    for (i, j), rs in bucket.items():
        vals = np.array([r.get("x_" + metric, np.nan) for r in rs], float)
        rels = np.array([r.get("x_" + metric + "_rel", np.nan) for r in rs], float)
        if not np.isfinite(vals).any():
            continue
        Z[i, j] = np.nanmean(vals)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R[i, j] = np.nanmean(rels) / np.sqrt(max(np.isfinite(vals).sum(), 1))
    return xs, ys, Z, R


def pick_slice(recs):
    """(slice value, records in it). The depth whose best cell is highest on the
    efficiency metric, or the forced value."""
    vals = sorted({r[SLICE_KEY] for r in recs if np.isfinite(r.get(SLICE_KEY, np.nan))})
    if not vals:
        return np.nan, recs
    if SLICE_PICK != "best":
        want = float(SLICE_PICK)
        near = min(vals, key=lambda v: abs(v - want))
        if abs(near - want) > 1e-9:
            print(f"  ! no runs at {SLICE_KEY}={want:g}; using {near:g}")
        return near, [r for r in recs if r[SLICE_KEY] == near]
    best, best_v = vals[0], -np.inf
    for v in vals:
        sub = [r for r in recs if r[SLICE_KEY] == v]
        top = np.nanmax([r.get("x_" + EFFICIENCY, np.nan) for r in sub]) \
            if sub else np.nan
        if np.isfinite(top) and top > best_v:
            best, best_v = v, top
    return best, [r for r in recs if r[SLICE_KEY] == best]


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
def out_name(tag):
    stem, _, ext = OUTNAME.rpartition(".")
    return f"{stem}_{tag}.{ext}" if tag else OUTNAME


def make_cmap(key):
    lo, hi = CMAPS[key][0], CMAPS[key][1]
    return LinearSegmentedColormap.from_list(f"sweetspot_{key}", [lo, hi])


def accent_of(key):
    return CMAPS[key][2]


def family_name(key):
    return CMAPS[key][3] if len(CMAPS[key]) > 3 else key


def cell_edges(v, scale):
    """Cell edges halfway between the centres, in whichever space the axis is
    drawn in, so the boundary a reader sees is the boundary the mesh used. The
    outer two are mirrored from the first and last gap. A single value gets an
    arbitrary width so one-column grids still draw.

    Log spacing on values <= 0 is undefined, so a non-positive value forces the
    linear branch rather than producing NaN edges and an empty panel."""
    v = np.asarray(v, float)
    use_log = (scale == "log") and bool(np.all(v > 0))
    if scale == "log" and not use_log:
        print("  ! non-positive axis values -- falling back to a linear axis")
    if v.size == 1:
        return (np.array([v[0] / 1.3, v[0] * 1.3]) if use_log
                else np.array([v[0] - 0.5, v[0] + 0.5]))
    t = np.log(v) if use_log else v
    mid = 0.5 * (t[1:] + t[:-1])
    e = np.concatenate(([t[0] - (mid[0] - t[0])], mid,
                        [t[-1] + (t[-1] - mid[-1])]))
    return np.exp(e) if use_log else e


def axis_scale(v, scale):
    """The scale actually usable for these values."""
    return "log" if scale == "log" and np.all(np.asarray(v, float) > 0) else "linear"


def sparse_ticks(v, n=4):
    v = np.asarray(v, float)
    if v.size <= n:
        return v
    idx = np.unique(np.linspace(0, v.size - 1, n).round().astype(int))
    return v[idx]


def _frac(v, lim, scale):
    """Where a value sits across an axis, 0 to 1, in the axis's own space."""
    a, b = lim
    if scale == "log" and v > 0 and a > 0 and b > 0:
        return (np.log(v) - np.log(a)) / (np.log(b) - np.log(a))
    return (v - a) / (b - a) if b != a else 0.5


def fmt_mult(v):
    if not np.isfinite(v):
        return "--"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.0f}"
    return f"{v:.2g}"


def n_of_L(dim, eta):
    """(L -> <N>, <N> -> L) for a trial box of side L at packing fraction eta.

    rho = eta / v_particle, so <N> = rho L^d. This is the average occupancy of
    ANY box of that volume; correlations change the fluctuations around it, not
    the mean, so the axis is exact and not an approximation."""
    v = (np.pi * PARTICLE_RADIUS ** 2 if dim == 2
         else (4.0 / 3.0) * np.pi * PARTICLE_RADIUS ** 3)
    rho = eta / v

    def fwd(L):
        return rho * np.power(np.clip(np.asarray(L, float), 1e-12, None), dim)

    def inv(N):
        return np.power(np.clip(np.asarray(N, float), 1e-12, None) / rho, 1.0 / dim)

    return fwd, inv


def packing_for(recs):
    """(dim, eta) for a row, from the table if the state point is listed, else
    from the runs' own pckng_frctn column."""
    dims = {r["dim"] for r in recs}
    mus = {r["mu"] for r in recs}
    if len(dims) != 1 or len(mus) != 1:
        return None, None      # mixed state points: no single second axis
    dim, mu = int(next(iter(dims))), float(next(iter(mus)))
    for (d, m), eta in PACKING_FRACTION.items():
        if d == dim and abs(m - mu) < 1e-9:
            return dim, eta
    seen = [r["packing"] for r in recs if np.isfinite(r.get("packing", np.nan))]
    return (dim, float(np.mean(seen))) if seen else (dim, None)


def apply_fonts():
    """One place where every size is set, with the floor applied."""
    f = max(20, FONT_PT)
    plt.rcParams.update({
        "font.size": f, "axes.labelsize": f, "axes.titlesize": f + 2,
        "xtick.labelsize": f, "ytick.labelsize": f, "legend.fontsize": f,
    })
    return f


def retile(fig, axes_grid, caxes, nr, nc, extra_bottom_in=0.0, secaxes=None):
    """Lay the panels out from MEASURED decoration sizes instead of guessed
    fractions, and return the new geometry.

    Every fraction-based margin in matplotlib assumes you know how wide a tick
    label or a colour bar's numbers will be. At one font size a guess looks
    fine and at another the row name lands on "trials k" and the bar numbers
    land on the next column's axis. So: draw once, ask the renderer how much
    room each decoration actually took, then give every panel exactly that much
    and resize the canvas around a plot box of fixed size. Inset colour bars
    are positioned against their parent axes, so they follow along."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    W, H = fig.get_size_inches()
    left_dec, right_dec = [0.0] * nc, [0.0] * nc
    top_dec, bot_dec = [0.0] * nr, [0.0] * nr
    for (i, j), ax in axes_grid.items():
        pos = ax.get_position()
        tb = ax.get_tightbbox(rend).transformed(inv)
        l, r = (pos.x0 - tb.x0) * W, (tb.x1 - pos.x1) * W
        t, b = (tb.y1 - pos.y1) * H, (pos.y0 - tb.y0) * H
        sec = (secaxes or {}).get((i, j))
        if sec is not None:
            sb = sec.get_tightbbox(rend)
            if sb is not None:
                sb = sb.transformed(inv)
                t = max(t, (sb.y1 - pos.y1) * H)
                l = max(l, (pos.x0 - sb.x0) * W)
                r = max(r, (sb.x1 - pos.x1) * W)
        cax = caxes.get((i, j))
        if cax is not None:
            cb_tb = cax.get_tightbbox(rend).transformed(inv)
            cpos = cax.get_position()
            # the bar's own offset and width scale with the panel, its numbers
            # do not, so the two are rebuilt separately for the new panel width
            r = max(r, CBAR_OFF * PANEL_W + (cb_tb.x1 - cpos.x1) * W)
            t = max(t, (cb_tb.y1 - pos.y1) * H)
            b = max(b, (pos.y0 - cb_tb.y0) * H)
        left_dec[j] = max(left_dec[j], l)
        right_dec[j] = max(right_dec[j], r)
        top_dec[i] = max(top_dec[i], t)
        bot_dec[i] = max(bot_dec[i], b)

    PAD_X, PAD_Y, EDGE = 0.30, 0.34, 0.06
    xs, x = [], EDGE
    for j in range(nc):
        x += left_dec[j]
        xs.append(x)
        x += PANEL_W + right_dec[j] + (PAD_X if j < nc - 1 else 0.0)
    W2 = x + EDGE
    ys, y = [], EDGE
    for i in range(nr):
        y += top_dec[i]
        ys.append(y)
        y += PANEL_H + bot_dec[i] + (PAD_Y if i < nr - 1 else 0.0)
    H2 = y + extra_bottom_in + EDGE
    fig.set_size_inches(W2, H2)
    for (i, j), ax in axes_grid.items():
        ax.set_position([xs[j] / W2, 1 - (ys[i] + PANEL_H) / H2,
                         PANEL_W / W2, PANEL_H / H2])
    return {"xs": xs, "ys": ys, "W": W2, "H": H2,
            "bot_dec": bot_dec, "left_dec": left_dec}


def draw(rows, out, name=None):
    """rows: [(label, colour family, slice value, records, y-axis key)]"""
    FS = apply_fonts()
    cols = columns()
    nr, nc = len(rows), len(cols)
    fams = list(dict.fromkeys(r[1] for r in rows))
    per_row = (NORM_SCOPE == "row")
    # Per-row scales put a bar beside every panel, so the bottom strip is not
    # needed at all; otherwise it grows with the number of families, one bar
    # and one tick row each, instead of overlapping itself.
    BAR_IN, LBL_IN = 0.16, FS / 72 + 0.18
    foot_in = (FS / 72 + 0.30) if per_row else 0.0
    strip_in = 0.0 if per_row else 0.35 + len(fams) * (BAR_IN + LBL_IN)
    # Rough first pass only. retile() measures what the decorations really take
    # and rebuilds the geometry, so none of these numbers has to be right.
    H = PANEL_H * nr * 1.6
    W = PANEL_W * nc * 1.6
    fig = plt.figure(figsize=(W, H))
    gs = gridspec.GridSpec(nr, nc, figure=fig, hspace=0.6, wspace=0.6,
                           left=0.12, right=0.92, top=0.94, bottom=0.10)
    col_box = {}          # column -> (left, width), for placing the colour bars
    panel_ax, panel_cax, panel_sec = {}, {}, {}

    # ---- colour scales, keyed (column, colour family). With SHARED_NORM off
    # each family is scaled to its own data so its structure is visible; the
    # price is that shades no longer compare between families, which is why the
    # bars are drawn one per family and the multiples are printed on the panels.
    keys = fams

    def make_norm(vals):
        vals = np.array([v for v in vals if np.isfinite(v) and v > 0], float)
        if vals.size == 0:
            return LogNorm(vmin=0.1, vmax=10)
        return LogNorm(vmin=max(vals.min(), vals.max() / 1e4), vmax=vals.max())

    norms = {}
    for m, _lab in cols:
        if per_row:
            for i, (_lbl, _ck, _sv, recs, _yk) in enumerate(rows):
                norms[(m, i)] = make_norm([r.get("x_" + m, np.nan) for r in recs])
        else:
            for ckey in keys:
                vals = []
                for _lbl, ck, _sv, recs, _yk in rows:
                    if NORM_SCOPE == "all" or ck == ckey:
                        vals += [r.get("x_" + m, np.nan) for r in recs]
                norms[(m, ckey)] = make_norm(vals)

    left_axes = []
    for i, (label, ckey, sval, recs, ykey) in enumerate(rows):
        cmap = make_cmap(ckey)
        accent = accent_of(ckey)
        dim_row, eta = packing_for(recs)
        if SHOW_N_AXIS and not eta:
            print(f"  ! {label}: no packing fraction for this state point and no "
                  "pckng_frctn column -- second x axis left off this row")
        # the star: the cell with the highest efficiency, marked on every column
        xs_e, ys_e, Ze, _Re = grid_for(recs, EFFICIENCY, ykey)
        star = None
        if np.isfinite(Ze).any():
            si, sj = np.unravel_index(np.nanargmax(Ze), Ze.shape)
            star = (xs_e[sj], ys_e[si])
        for j, (m, mlab) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            panel_ax[(i, j)] = ax
            pos = ax.get_position()
            col_box[j] = (pos.x0, pos.width)
            xs, ys, Z, R = grid_for(recs, m, ykey)
            if xs.size and ys.size and np.isfinite(Z).any():
                ax.pcolormesh(cell_edges(xs, X_SCALE), cell_edges(ys, Y_SCALE),
                              np.ma.masked_invalid(Z), cmap=cmap,
                              norm=norms[(m, i if per_row else ckey)],
                              shading="flat",
                              edgecolors="none")
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=FS)
            # Ticks sit ON the swept values when the axis is log, because the
            # default log locator would label decades the sweep never visited.
            # On a linear axis that rule backfires -- log-spaced values crowd
            # into each other at the bottom -- so there the locator decides.
            if xs.size:
                sc = axis_scale(xs, X_SCALE)
                ax.set_xscale(sc)
                if sc == "log":
                    ax.set_xticks(sparse_ticks(xs))
                    ax.set_xticklabels([f"{v:g}" for v in sparse_ticks(xs)])
                else:
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if ys.size:
                sc = axis_scale(ys, Y_SCALE)
                ax.set_yscale(sc)
                if sc == "log":
                    ax.set_yticks(sparse_ticks(ys))
                    ax.set_yticklabels([f"{v:g}" for v in sparse_ticks(ys)])
                else:
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.minorticks_off()
            ax.tick_params(direction="in", which="both", labelsize=FS)
            ax.grid(True, ls=":", lw=0.6, color="0.75", alpha=0.7)
            ax.set_axisbelow(False)

            if m == EFFICIENCY and CONTOUR_FRACTIONS and np.isfinite(Z).any() \
                    and xs.size > 2 and ys.size > 2:
                top = np.nanmax(Z)
                lv = sorted(f * top for f in CONTOUR_FRACTIONS)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ax.contour(xs, ys, np.nan_to_num(Z, nan=0.0), levels=lv,
                               colors="k", linewidths=0.8, linestyles="--")

            if star is not None:
                ax.plot(*star, marker="*", ms=FS + 6, color="#F5B301",
                        mec="k", mew=0.6, ls="none", zorder=5)
                ax.axvline(star[0], color="#F5B301", lw=0.8, ls="--", alpha=0.9, zorder=4)
                ax.axhline(star[1], color="#F5B301", lw=0.8, ls="--", alpha=0.9, zorder=4)
                v = _at(recs, m, star, ykey)
                rel = _at(recs, m + "_rel", star, ykey)
                txt = fmt_mult(v) + r"$\times$"
                if m == EFFICIENCY and np.isfinite(rel) and np.isfinite(v):
                    txt += "\n" + rf"$\pm${fmt_mult(v * rel)}"
                # Put the label in the corner FURTHEST from the star. A fixed
                # bottom-left corner sits on top of the star whenever the sweet
                # spot is at low L and low y, which is most of these rows.
                fx = _frac(star[0], ax.get_xlim(), ax.get_xscale())
                fy = _frac(star[1], ax.get_ylim(), ax.get_yscale())
                ha, x_a = ("left", 0.035) if fx > 0.5 else ("right", 0.965)
                va, y_a = ("bottom", 0.04) if fy > 0.5 else ("top", 0.96)
                ax.annotate(txt, xy=(x_a, y_a), xycoords="axes fraction",
                            fontsize=FS + 2, fontweight="bold", ha=ha, va=va,
                            zorder=6,
                            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))

            # ---- per-panel colour bar. Each panel has its own scale under
            # NORM_SCOPE="row", so the bar has to live next to the panel it
            # belongs to; a shared strip at the bottom would be a lie.
            if per_row and xs.size and ys.size and np.isfinite(Z).any():
                cax = ax.inset_axes([1.0 + CBAR_OFF, 0.0, CBAR_W, 1.0])
                panel_cax[(i, j)] = cax
                nrm = norms[(m, i)]
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cmap),
                                  cax=cax, orientation="vertical")
                cb.outline.set_linewidth(0.6)
                tk = np.geomspace(nrm.vmin, nrm.vmax, 4)
                cb.set_ticks(tk)
                cb.set_ticklabels([fmt_mult(t) for t in tk])
                cax.yaxis.set_minor_locator(NullLocator())
                cax.tick_params(labelsize=FS, direction="in", pad=2.5)
                if nrm.vmin < 1.0 < nrm.vmax:
                    cax.axhline(1.0, color="k", lw=0.9, ls=":")

            # ---- second x axis: mean occupancy of the trial box
            if SHOW_N_AXIS and eta:
                sec = ax.secondary_xaxis("top", functions=n_of_L(dim_row, eta))
                sec.tick_params(labelsize=FS, direction="in")
                sec.xaxis.set_major_locator(MaxNLocator(nbins=5))
                sec.xaxis.set_minor_locator(NullLocator())
                sec.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                panel_sec[(i, j)] = sec
                if i == 0:
                    sec.set_xlabel(r"$\langle N\rangle$ in trial box",
                                   fontsize=FS)

            if i == 0:
                ax.set_title(mlab, fontsize=FS + 2, pad=FS + 14
                             if (SHOW_N_AXIS and eta) else 6)
            ax.set_xlabel(AXIS_LABEL.get(X_KEY, X_KEY), fontsize=FS)
            if j == 0:
                ax.set_ylabel(AXIS_LABEL.get(ykey, ykey), fontsize=FS)
                # the row name is placed after layout, once the width of the
                # tick labels and the y label is actually known
                tag = (f"{label}\n({AXIS_LABEL.get(SLICE_KEY, SLICE_KEY)} {sval:g})"
                       if np.isfinite(sval) else label)
                left_axes.append((ax, tag, accent))

    # ---- lay everything out from measured sizes before anything else is
    # placed relative to the panels
    geo = retile(fig, panel_ax, panel_cax, nr, nc,
                 extra_bottom_in=strip_in + foot_in, secaxes=panel_sec)
    W2, H2 = geo["W"], geo["H"]
    col_box = {j: (geo["xs"][j] / W2, PANEL_W / W2) for j in range(nc)}
    base_in = (PANEL_H + geo["ys"][nr - 1] + geo["bot_dec"][nr - 1] + 0.30)

    # ---- shared colour bars: one per colour family per column. They carry
    # DIFFERENT ranges unless NORM_SCOPE is "all", so each gets its own tick
    # labels and its own name on the left; reading a shade off the wrong bar is
    # then the obvious mistake to make, and the labels are what stop it.
    for j, (m, mlab) in enumerate(cols if not per_row else []):
        left, width = col_box[j]
        for b, ckey in enumerate(keys):
            top_in = base_in + b * (BAR_IN + LBL_IN)
            cax = fig.add_axes([left, 1 - (top_in + BAR_IN) / H2,
                                width, BAR_IN / H2])
            sm = plt.cm.ScalarMappable(norm=norms[(m, ckey)], cmap=make_cmap(ckey))
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.outline.set_linewidth(0.6)
            # Explicit ticks in plain numbers. The default log formatter labels
            # every minor decade multiple, which on a bar spanning less than a
            # decade prints "3 x 10^2  4 x 10^2" straight through itself.
            nrm = norms[(m, ckey)]
            tk = np.geomspace(nrm.vmin, nrm.vmax, 4)
            cb.set_ticks(tk)
            cb.set_ticklabels([fmt_mult(t) for t in tk])
            cax.xaxis.set_minor_locator(NullLocator())
            cax.tick_params(labelsize=FS, direction="in", top=False)
            if nrm.vmin < 1.0 < nrm.vmax:
                cax.axvline(1.0, color="k", lw=0.9, ls=":")
            # only on the first column: the bars line up row by row, and a
            # repeat on column two would sit on column one's last tick label
            if j == 0:
                cax.text(-0.012, 0.5, family_name(ckey), transform=cax.transAxes,
                         ha="right", va="center", fontsize=FS,
                         color=accent_of(ckey), fontweight="bold")
            if b == 0:
                cax.set_title(mlab, fontsize=FS, pad=4)
    if per_row:
        pass
        #fig.text(0.5, 0.35 * foot_in / H2,
        #         "colour = multiple of std GCMC, scaled per panel; "
        #         "the dotted line on a bar marks 1",
        #         ha="center", va="bottom", fontsize=FS, color="0.35")
    else:
        fig.text(col_box[0][0] - 0.012,
                 1 - (base_in + len(keys) * (BAR_IN + LBL_IN)) / H2,
                 "1 = std GCMC", ha="right", va="center", fontsize=FS)

    # ---- row names, placed against the MEASURED left edge of each row.
    # A fixed offset in axes fractions cannot work here: how far the tick
    # labels and the y label stick out depends on the numbers in them and on
    # whatever font size is in force, so at a bigger font the name lands on top
    # of "trials k". Asking the renderer where things actually ended up is the
    # only version that survives a font change.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, label, accent in left_axes:
        bb = ax.get_tightbbox(rend).transformed(inv)
        pos = ax.get_position()
        fig.text(bb.x0 - 0.10 / W2, (pos.y0 + pos.y1) / 2, label, rotation=90,
                 ha="right", va="center", fontsize=FS + 2, fontweight="bold",
                 color=accent)

    out.mkdir(exist_ok=True, parents=True)
    fig.savefig(out / (name or OUTNAME), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / (name or OUTNAME)}")


def _at(recs, metric, star, ykey=None):
    """The value of `metric` at the starred cell, averaged over replicates."""
    x, y = star
    vals = [r.get("x_" + metric, np.nan) for r in recs
            if r[X_KEY] == x and val_of(r, ykey) == y]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


# ----------------------------------------------------------------------------
def audit(label, recs):
    """Shout when a row is not one experiment. A folder holding two samplers or
    two state points draws as a perfectly clean heatmap of two different things,
    and no colourbar can show you that."""
    for key in ("sampler", "engine", "potential", "dim", "box", "mu"):
        vals = {r.get(key) for r in recs}
        if len(vals) > 1:
            print(f"  !! {label}: {key} takes {len(vals)} values across this "
                  f"folder ({sorted(map(str, vals))}) -- these cells are not one "
                  "experiment and nothing in the row compares them fairly")


SHORT = {"trial_box_side": "L", "trials": "k", "trial_density": "rho_t",
         "gamma": "gamma", "depth": "depth", "pool_M": "M"}


def short_name(key):
    return SHORT.get(key, str(key)[:7])


def row_tag(row):
    """The figure a row belongs to. Dimension is the only split that makes
    sense by default: a 2D and a 3D run are different state spaces and share
    neither a reference nor a colour scale."""
    if SPLIT_BY != "dim":
        return ""
    dims = sorted({r["dim"] for r in row[3] if np.isfinite(r.get("dim", np.nan))})
    return f"{dims[0]:g}d" if len(dims) == 1 else "mixed"


def sweetspot(rows, top_n=TOP_N, heading=""):
    """The numbers behind the stars, so nothing has to be read off a colour.

    Top `top_n` cells per row ranked on the efficiency metric, with the sampler
    settings that produced them and both the multiples and the absolute values."""
    ynames = {short_name(r[4]) for r in rows}
    yhdr = "y" if len(ynames) > 1 else next(iter(ynames), "y")
    hdr = (f"{'row':<28}{'#':>2}{short_name(X_KEY):>8}{yhdr:>8}{'depth':>7}"
           f"{'acc x':>10}{'acc/sec x':>13}"
           f"{'acc abs':>11}{'acc/s abs':>12}{'s/step':>10}   other")
    print("\n" + "=" * len(hdr))
    print(f"SWEET SPOT{' ' + heading if heading else ''} -- best {top_n} cells "
          f"per row, ranked on {EFFICIENCY}")
    print(hdr)
    print("-" * len(hdr))
    for label, _ck, sval, recs, ykey in rows:
        # what is already in its own column for THIS row -- the y axis differs
        # per row, so the "other" column has to be worked out per row too
        shown = {X_KEY, ykey, "depth"}
        xs, ys, Ze, _R = grid_for(recs, EFFICIENCY, ykey)
        if not np.isfinite(Ze).any():
            print(f"{label[:28]:<28} no finite cells")
            continue
        flat = Ze.ravel()
        order = np.argsort(np.where(np.isfinite(flat), flat, -np.inf))[::-1]
        for rank, idx in enumerate(order[:top_n], start=1):
            if not np.isfinite(flat[idx]):
                break
            i, j = np.unravel_index(idx, Ze.shape)
            x, y = xs[j], ys[i]
            hit = [r for r in recs if r[X_KEY] == x and val_of(r, ykey) == y]
            r0 = hit[0] if hit else {}
            eff = _at(recs, EFFICIENCY, (x, y), ykey)
            rel = _at(recs, EFFICIENCY + "_rel", (x, y), ykey)
            acc = _at(recs, "acc_per_step", (x, y), ykey)
            err = f"+-{eff * rel:.2g}" if np.isfinite(rel) and np.isfinite(eff) else ""
            other = " ".join(f"{k}={r0[k]:g}" for k in REPORT_KEYS
                             if k not in shown and np.isfinite(r0.get(k, np.nan)))
            tag = f"{label} [y={short_name(ykey)}]" if rank == 1 else ""
            dep = f"{sval:g}" if np.isfinite(sval) else "-"
            print(f"{tag[:28]:<28}{rank:>2}"
                  f"{x:>8g}{y:>8g}{dep:>7}"
                  f"{acc:>10.3g}{eff:>8.3g} {err:<4}"
                  f"{r0.get('acc_per_step', np.nan):>11.3g}"
                  f"{r0.get('acc_per_second', np.nan):>12.3g}"
                  f"{r0.get('t_per_step', np.nan):>10.3g}   {other}")
        print("-" * len(hdr))
    print("=" * len(hdr))


def diagnostics(rows, ref_recs):
    """The absolute numbers behind the ratios, reference first.

    A ratio hides which side moved. Everything in the figure is a quotient of
    two of these lines, so this is where an implausible multiple gets traced
    back to a column that does not mean what it is named."""
    hdr = (f"{'run':<40}{'s/step':>11}{'steps/win':>12}{'s/win':>10}"
           f"{'cyc/win':>11}{'mv frac':>9}{'acc/step':>11}{'acc/s':>11}")
    print("\n" + "=" * len(hdr))
    print(f"ABSOLUTE numbers  (STEP_DEFINITION={STEP_DEFINITION!r})")
    print(hdr)
    print("-" * len(hdr))

    def line(name, r):
        print(f"{name[:40]:<40}{r['t_per_step']:>11.3g}"
              f"{r['raw_steps_per_window']:>12.4g}{r['raw_wall_per_window']:>10.3g}"
              f"{r['raw_cycles_per_window']:>11.4g}{r['mv_frac']:>9.2f}"
              f"{r['acc_per_step']:>11.3g}{r['acc_per_second']:>11.3g}")

    for r in sorted(ref_recs, key=lambda r: (r["dim"], r["mu"])):
        line(f"std ref  dim={r['dim']} mu={r['mu']:g}", r)
    print("-" * len(hdr))
    for label, _ck, _sv, recs, ykey in rows:
        xs_e, ys_e, Ze, _R = grid_for(recs, EFFICIENCY, ykey)
        if not np.isfinite(Ze).any():
            continue
        si, sj = np.unravel_index(np.nanargmax(Ze), Ze.shape)
        star = (xs_e[sj], ys_e[si])
        hit = [r for r in recs if r[X_KEY] == star[0]
               and val_of(r, ykey) == star[1]]
        if hit:
            line(f"{label}  {short_name(X_KEY)}={star[0]:g} "
                 f"{short_name(ykey)}={star[1]:g}", hit[0])
    print("=" * len(hdr) + "\n")


def main():
    ref, ref_recs = build_reference(STD_REF_DIRS)
    if not ref:
        print("no std GCMC reference found -- set STD_REF_DIRS. Nothing to divide by.")
        return
    print("reference state spaces: "
          + ", ".join(f"dim={d} {p} T={t}" for d, p, t in ref))

    rows = []
    for entry in ROWS:
        path, label, ckey = entry[:3]
        ykey = entry[3] if len(entry) > 3 else Y_KEY
        print(f"reading {path}")
        recs = collect(resolve_path(path), label)
        if not recs:
            print("  (no usable runs)")
            continue
        as_multiples(recs, ref)
        audit(label, recs)
        if not any(np.isfinite(val_of(r, ykey)) for r in recs):
            print(f"  ! nothing readable on the y axis {ykey!r} in this folder -- "
                  "check the key name against params.json; skipping the row")
            continue
        sval, sub = pick_slice(recs)
        if np.isfinite(sval):
            print(f"  {len(recs)} runs, y={ykey}, drawing {SLICE_KEY}={sval:g} "
                  f"({len(sub)} cells, {SLICE_PICK} slice)")
        else:
            print(f"  {len(recs)} runs, y={ykey}, no {SLICE_KEY} axis in this folder")
        rows.append((label, ckey, sval, sub, ykey))

    if not rows:
        print("nothing to plot.")
        return
    figs = {}
    for row in rows:
        figs.setdefault(row_tag(row), []).append(row)
    for tag, sub in sorted(figs.items()):
        sweetspot(sub, heading=tag.upper())
    if DIAGNOSTICS:
        diagnostics(rows, ref_recs)
    for tag, sub in sorted(figs.items()):
        draw(sub, Path(OUTDIR), out_name(tag))
    print("done.")


if __name__ == "__main__":
    main()

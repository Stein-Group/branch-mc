#!/usr/bin/env python3
"""Tufte-style bar charts of how n_cav and N are distributed, all methods.

Writes four figures to OUTDIR:

    ncav_hist_2d_mu12.png   n_cav, 2D groups at mu = 12
    ncav_hist_3d_mu15.png   n_cav, 3D groups at mu = 15
    N_hist_2d_mu12.png      N,     2D groups at mu = 12
    N_hist_3d_mu15.png      N,     3D groups at mu = 15

x is the value (one group of bars per integer, one bar per method), y is the
probability of that value: its row count over the method's total row count,
so each method's bars sum to 1 (tails trimmed by X_TRIM included in the
total, so a trimmed plot never inflates what is shown). N is drawn as N - N0,
with N0 (the smallest N shown) in the axis label. Every row of run.log is
used. mu, dimension, L (sampler.trial_box_side) and k (sampler.trials) come
from each run's params.json, columns from the log's `#` header. std GCMC has
no cavity, so it only appears in the N figures.

Broken y axis: when one bar towers over the rest, the axis is cut at the
largest empty band of bar heights. Both panels keep the same scale and tick
step, every bar top stays visible, and only empty space is removed. The cut
is marked on the spine and on every bar that crosses it. No band worth
cutting (BREAK_MIN_FRAC of the height) means no break.

Tufte conventions, as in plot_ESS_3.py: no top/right spines, range-framed
spines, white scale lines through the bars instead of a grid, no x tick
marks, a frameless legend above the axes.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.transforms import blended_transform_factory

# ============================================================================
# CONFIG
# ============================================================================
RUNS_ROOT = Path("runs")
OUTDIR = Path("figs_hist")
DPI = 300
FIGSIZE = (6, 6)
MU = {2: 12.0, 3: 15.0}  # state point to plot, per dimension
GROUPS = {  # legend name -> group dir, per dimension; order = bar order
    2: {"std GCMC": "std_2d", "SR": "SR_2d", "SR with aux-pot": "SR_aux_2d",
        "BRANCH": "branch_2d"},
    3: {"std GCMC": "std_3d", "SR": "SR_3d", "SR with aux-pot": "SR_aux_3d",
        "BRANCH": "branch_3d"},
}
SYMBOL = {"n_cav": r"n_{\mathrm{cav}}", "N": "N"}
XLABEL = {"n_cav": r"$n_{\mathrm{cav}}$", "N": r"$N - N_0$"}
MEAN_FMT = {"n_cav": "{:.2f}", "N": "{:.1f}"}
YLABEL = "Probability"

# red / green / blue for the three cavity methods, black for std GCMC
COLORS = {"std GCMC": "#222222", "SR": "#D7191C", "SR with aux-pot": "#1A9641",
          "BRANCH": "#2B6CC4"}

FS_LABEL, FS_TICK, FS_LEGEND = 20, 17, 12
GROUP_W = 0.84        # width of one integer's group of bars, in x units
GRID_LW = 1.0         # white scale lines drawn through the bars
Y_TICK_LEN = 5
X_TRIM = 0.002        # hide tails below this fraction of the tallest bar; 0 = off
BREAK_MIN_FRAC = 0.3  # break only if the cut removes >= this share of the height
TOP_MIN_RATIO = 0.18  # smallest top-panel height relative to the bottom one
INK = "0.3"           # spines and break marks
GAP = 0.12            # white gap between the two panels of a broken axis
MARGIN = {"left": 0.17, "right": 0.97, "bottom": 0.13}  # figure fractions
LEGEND_PAD = 0.035    # space between the legend and the axes' top
# ============================================================================


def read_log(path):
    """run.log as {column name: array}, names taken from the `#` header."""
    with open(path) as fh:
        head = fh.readline()
    if not head.startswith("#"):
        raise ValueError(f"{path}: no '#' header, cannot read columns by name")
    names = head.lstrip("#").split()
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.size == 0:
        return {}
    if rows.shape[1] != len(names):
        raise ValueError(f"{path}: header has {len(names)} columns, rows have "
                         f"{rows.shape[1]}")
    return {n: rows[:, i] for i, n in enumerate(names)}


def read_state_point(leaf):
    """(dimension, mu) from a run's params.json."""
    ens = json.loads((leaf / "params.json").read_text())["ensemble"]
    mu = ens["mu"]
    if isinstance(mu, dict):
        mu = next(iter(mu.values()))
    return int(ens["dimension"]), float(mu)


def collect(group, dim, mu):
    """(logs, sampler blocks) of every leaf in runs/<group> at this (dim, mu)."""
    logs, samplers = [], []
    root = RUNS_ROOT / group
    if not root.is_dir():
        print(f"  {group}: no directory {root}, skipped")
        return logs, samplers
    for leaf in sorted(p for p in root.iterdir() if (p / "params.json").exists()):
        d, m = read_state_point(leaf)
        if d == dim and math.isclose(m, mu) and (leaf / "run.log").exists():
            logs.append(read_log(leaf / "run.log"))
            params = json.loads((leaf / "params.json").read_text())
            samplers.append(params.get("sampler", {}))
    if len(logs) > 1:
        print(f"  {group}: {len(logs)} runs at mu = {mu:g}, rows pooled")
    return logs, samplers


def trial_knobs(samplers, group):
    """Legend text for L and k, or '' for a sampler without a trial box."""
    if not samplers or "trial_box_side" not in samplers[0]:
        return ""
    s0 = samplers[0]
    if any(s.get("trial_box_side") != s0.get("trial_box_side")
           or s.get("trials") != s0.get("trials") for s in samplers[1:]):
        print(f"  {group}: pooled runs differ in L or k, legend shows the first")
    txt = rf", $L$ = {float(s0['trial_box_side']):g}"
    if "trials" in s0:
        txt += rf", $k$ = {int(s0['trials'])}"
    elif "trial_density" in s0:
        txt += rf", $\rho_k$ = {float(s0['trial_density']):g}"
    return txt


def values(logs, col, group):
    """One column across logs as ints, nan rows dropped. None if empty."""
    have = [lg[col] for lg in logs if col in lg]
    if not have:
        why = "no run at this mu" if not logs else f"no '{col}' column"
        print(f"  {group}: {why}, left out of the {col} figure")
        return None
    v = np.concatenate(have)
    v = v[np.isfinite(v)]
    if v.size == 0:
        print(f"  {group}: '{col}' is nan in every row, left out")
        return None
    return np.rint(v).astype(int)


def x_range(series):
    """Integer span to draw, with near-empty tails trimmed (X_TRIM)."""
    lo = min(s[1].min() for s in series)
    hi = max(s[1].max() for s in series)
    if X_TRIM > 0:
        counts = [np.bincount(s[1] - lo, minlength=hi - lo + 1) for s in series]
        peak = max(c.max() for c in counts)
        seen = np.nonzero(np.max(counts, axis=0) >= X_TRIM * peak)[0]
        new_lo, new_hi = int(lo + seen[0]), int(lo + seen[-1])
        cut = sum(int(((s[1] < new_lo) | (s[1] > new_hi)).sum()) for s in series)
        if cut:
            print(f"  x range {lo}..{hi} trimmed to {new_lo}..{new_hi}: "
                  f"{cut} rows in the tails are not drawn")
        lo, hi = new_lo, new_hi
    return int(lo), int(hi)


def nice_step(span):
    """Tick step MaxNLocator would pick for an axis from 0 to `span`."""
    t = ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, span)
    return float(t[1] - t[0])


def plan_break(heights):
    """(y_low, a, b, step) for a broken axis, or (y_top, None, None, step).

    Every distinct bar top is a candidate place to cut above. The bottom
    panel covers the tops below the cut, rounded up to its tick step; the top
    panel covers the rest, from one step below its lowest top to the first
    tick at or above the tallest bar, on the same step. The candidate that
    removes the most height wins, if it removes BREAK_MIN_FRAC of it.
    """
    tops = np.unique(np.concatenate([h[h > 0] for h in heights]))
    tallest = float(tops[-1])
    step = nice_step(tallest)
    plain = (step * math.ceil(tallest / step - 1e-9), None, None, step)
    best, best_cut = plain, 0.0
    for i in range(len(tops) - 1):
        below, above = float(tops[i]), float(tops[i + 1])
        st = nice_step(below)
        y_low = st * math.ceil(below / st - 1e-9)
        b = st * math.ceil(tallest / st - 1e-9)
        a = min(st * math.floor(above / st) - st, b - st)
        cut = a - y_low
        if cut > best_cut:
            best, best_cut = (y_low, a, b, st), cut
    if best_cut < BREAK_MIN_FRAC * tallest:
        return plain
    return best


def paint_bars(ax, x, series, heights, w, ticks):
    """One bar per method per integer, with white scale lines through them."""
    n = len(series)
    for i, ((label, _v, _txt), h) in enumerate(zip(series, heights)):
        xi = x + (i - (n - 1) / 2) * w
        ax.bar(xi, h, w * 0.94, color=COLORS[label], linewidth=0, zorder=3)
        for t in ticks:
            m = h > t
            if m.any():
                ax.hlines(np.full(m.sum(), t), xi[m] - w * 0.47,
                          xi[m] + w * 0.47, color="white", lw=GRID_LW, zorder=4)


def decimals(step):
    """Places after the point that `step` needs, so every label matches."""
    txt = f"{step:g}"
    return len(txt.split(".")[1]) if "." in txt else 0


def style_y(ax, lo, hi, step):
    """Range-framed left spine from lo to hi, ticks every step."""
    ticks = np.arange(lo, hi + step / 2, step)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f"%.{decimals(step)}f"))
    ax.spines["left"].set_bounds(lo, hi)
    ax.tick_params(axis="y", direction="in", length=Y_TICK_LEN,
                   labelsize=FS_TICK, colors="0.15")
    return [t for t in ticks if t > 0]


def mark_break(ax_top, ax_bot, xs, w, ratio):
    """Slashes on the left spine and across every bar the cut runs through."""
    d, dy = 0.018, 0.016           # half-size in axes units of the bottom panel
    kw = dict(color=INK, lw=1.1, clip_on=False, zorder=6)
    for ax, y, s in ((ax_bot, 1.0, dy), (ax_top, 0.0, dy / ratio)):
        ax.plot([-d, d], [y - s, y + s], transform=ax.transAxes, **kw)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        for x in xs:
            ax.plot([x - w * 0.75, x + w * 0.75], [y - s, y + s], transform=tr,
                    **kw)


def draw(fig, series, col, top_edge):
    lo, hi = x_range(series)
    n0 = lo if col == "N" else 0
    x = np.arange(lo, hi + 1) - n0
    w = GROUP_W / len(series)
    # normalised by each method's full row count, trimmed tails included
    heights = [np.bincount(v[(v >= lo) & (v <= hi)] - lo, minlength=len(x))
               / v.size for _, v, _t in series]
    y_low, a, b, step = plan_break(heights)

    frame = dict(left=MARGIN["left"], right=MARGIN["right"],
                 bottom=MARGIN["bottom"], top=top_edge)
    if a is None:
        gs = fig.add_gridspec(1, 1, **frame)
        axes = [fig.add_subplot(gs[0])]
        spans = [(0.0, y_low)]
    else:
        ratio = max((b - a) / y_low, TOP_MIN_RATIO)
        gs = fig.add_gridspec(2, 1, height_ratios=[ratio, 1.0], hspace=GAP,
                              **frame)
        axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        spans = [(a, b), (0.0, y_low)]
        axes[1].sharex(axes[0])

    for k, (ax, (y0, y1)) in enumerate(zip(axes, spans)):
        ax.set_ylim(y0, y1 + step * 0.02)
        ticks = style_y(ax, y0 if y0 > 0 else 0.0, y1, step)
        if y0 > 0:
            # the top panel's first tick sits at the cut, right above the
            # bottom panel's last label: keep the tick, drop its label
            dp = decimals(step)
            ax.set_yticklabels(
                ["" if t == y0 else f"{t:.{dp}f}" for t in ax.get_yticks()]
            )
        paint_bars(ax, x, series, heights, w, ticks)
        ax.spines[["top", "right"]].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color(INK)
            ax.spines[sp].set_linewidth(0.9)
    bot = axes[-1]
    bot.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    bot.spines["bottom"].set_bounds(x[0] - 0.5, x[-1] + 0.5)
    bot.xaxis.set_major_locator(ticker.MaxNLocator(8, integer=True))
    bot.set_xticks([t for t in bot.get_xticks() if x[0] <= t <= x[-1]])
    bot.tick_params(axis="x", length=0, labelsize=FS_TICK, colors="0.15")
    xl = XLABEL[col] + (rf",   $N_0 = {n0}$" if col == "N" else "")
    bot.set_xlabel(xl, fontsize=FS_LABEL, color="0.1")

    if a is not None:
        top = axes[0]
        top.spines["bottom"].set_visible(False)
        top.tick_params(axis="x", bottom=False, labelbottom=False)
        # the bars that run through the cut, at their drawn x positions
        n = len(series)
        cross = [x[j] + (i - (n - 1) / 2) * w
                 for i, h in enumerate(heights) for j in np.nonzero(h > y_low)[0]]
        mark_break(top, bot, cross, w, ratio)
        fig.supylabel(YLABEL, fontsize=FS_LABEL, color="0.1", x=0.015,
                      y=(MARGIN["bottom"] + top_edge) / 2)
    else:
        bot.set_ylabel(YLABEL, fontsize=FS_LABEL, color="0.1")
    return axes


def legend(fig, series, col, dim):
    """Two-line entries in two columns, above the axes, no frame."""
    handles, labels = [], []
    for label, v, knobs in series:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=COLORS[label], lw=0))
        mean = MEAN_FMT[col].format(v.mean())
        labels.append(f"{label} {dim}D\n"
                      rf"$\langle {SYMBOL[col]} \rangle$ = {mean}{knobs}")
    return fig.legend(handles, labels, loc="upper center", ncol=2,
                      bbox_to_anchor=(0.5, 0.995), frameon=False,
                      fontsize=FS_LEGEND, handlelength=1.1, handleheight=1.1,
                      handletextpad=0.5, columnspacing=1.2, labelspacing=0.7)


def figure(dim, col):
    mu = MU[dim]
    series = []
    for label, group in GROUPS[dim].items():
        logs, samplers = collect(group, dim, mu)
        v = values(logs, col, group)
        if v is not None:
            series.append((label, v, trial_knobs(samplers, group)))
    if not series:
        print(f"  nothing to plot for {col} in {dim}D at mu = {mu:g}")
        return
    fig = plt.figure(figsize=FIGSIZE)
    # legend first: the axes then get exactly the height it leaves free
    leg = legend(fig, series, col, dim)
    fig.canvas.draw()
    leg_bottom = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
    draw(fig, series, col, top_edge=leg_bottom - LEGEND_PAD)
    name = f"{'ncav' if col == 'n_cav' else col}_hist_{dim}d_mu{mu:g}.png"
    fig.savefig(OUTDIR / name, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {OUTDIR / name}")


def main():
    OUTDIR.mkdir(exist_ok=True)
    for col in XLABEL:
        for dim in (2, 3):
            print(f"{col}, {dim}D, mu = {MU[dim]:g}")
            figure(dim, col)


if __name__ == "__main__":
    main()

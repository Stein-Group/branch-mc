#!/usr/bin/env python3

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ----------------------------------------------------------------- config
SUMMARY = Path("ess_summary.csv")
PER_RUN = Path("ess_per_run.csv")                  # optional; "" to disable
OUT = Path(".")
DPI = 200
FIGSIZE = (6, 6)                                   # square, as requested

# x order, left to right. Each entry: (candidate labels, plot label).
# Matching is on ESS_analysis.py's `label` column, case- and punctuation-
# insensitive, so "STD-GCMC", "std_gcmc" and "Std GCMC" all hit the same slot.
METHODS = [
    (("STD-GCMC", "std_gcmc", "metropolis"), "std\nGCMC"),
    # one group is ~110 pt wide, a 24 pt "SR with aux-pot" nearly 200, so the
    # long name is wrapped rather than shrunk; diagnose() unwraps it for stderr
    (("HD-AUX", "aux", "hd_aux"), "SR with\naux-pot"),
    (("Branch", "branch"), "BRANCH"),
]

DIMS = ["2D", "3D"]
DIM_COLORS = {"2D": "#4C72B0", "3D": "#DD8452"}    # blue, orange -- bar fill
# The direct series labels sit on white paper, where the bar fills read far
# paler than they do as a solid block (the orange especially). They get their
# own darker, more saturated shade, so the label is legible while the bars keep
# the seaborn-deep palette. Darken the bars too by editing DIM_COLORS.
DIM_TEXT = {"2D": "#2B4A7D", "3D": "#A9490D"}

# metric -> (summary mean col, summary sem col, per-run col, scale, label)
# y labels are wrapped by hand: at FS_LABEL = 24 a single line of 30 characters
# is ~430 pt and the axis is only ~380 pt tall, so one line would be clipped.
RATE = ("ess_per_sec_mean", "ess_per_sec_sem", "ess_per_second",
        3600.0, "Effective Sample Size\nper CPU-hour")
CYCLE = ("ess_per_step_mean", "ess_per_step_sem", "ess_per_cycle",
         1.0, "ESS / MC cycle")                    # fallback if wall_s absent
GEE = ("g_mean", "g_sem", "g_window",
       None, "statistical inefficiency\n" + r"$g = 1 + 2\tau$")  # None -> autoscale

CI_MULT = 1.0              # whisker = CI_MULT * s.e.m. over seeds
Y_SPINE = True             # keep the left spine, bounded to the data (range frame)
Y_TICK_LEN = 6             # inward tick marks, as in the density figure
SCALE_TAG = True           # small "x10^6" above the axis when values are scaled
SEED_DOTS = True           # overplot individual seeds from ess_per_run.csv
DRIFT_Z_MAX = 3.0          # |z| above this is called out on stderr

# Sized to sit beside the 12x6 density panel in a composite figure. What the
# eye compares there is rendered text height, which is fontsize x (displayed
# width / figure width): the density panel runs 26 pt on 12 in, so on a 6 in
# canvas the same apparent size is ~24 pt, not 16. Below 20 pt these two panels
# read as the small ones no matter what the file says.
FS_LABEL, FS_TICK, FS_SERIES, FS_ANNOT = 24, 22, 26, 22
ANNOT_ROTATION = None      # None = upright if it fits, else 90; force with 0/90
ANNOT_INSIDE = True        # rotated label goes inside its bar when it fits
BAR_W = 0.34
HEADROOM = 1.15            # floor only; draw() derives the real headroom from
                           # the measured height of the stacked labels
GRID_LW = 1.1              # white scale lines drawn through the bars
FADE_PARTIAL = True        # incomplete cells at reduced saturation
LOG_Y = False              # bars encode value by length; keep linear if it fits

REQUIRED = ["label", "dim", "n_seeds", "g_mean", "g_sem",
            "ess_per_sec_mean", "ess_per_step_mean", "max_abs_drift_z"]


# ------------------------------------------------------------------ data
def key(s):
    """'STD-GCMC', 'std_gcmc', 'Std GCMC' -> 'STDGCMC'."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def resolve_methods(present):
    """Per METHODS entry, the candidate label actually present in the CSV."""
    keys, labels = [], {}
    for cands, label in METHODS:
        k = next((key(c) for c in cands if key(c) in present), key(cands[0]))
        keys.append(k)
        labels[k] = label
    return keys, labels


def load(summary_csv=SUMMARY, per_run_csv=PER_RUN):
    """Read ess_summary.csv (+ ess_per_run.csv if present). No reshaping."""
    if not Path(summary_csv).exists():
        sys.exit(f"{summary_csv} not found. Produce it with:\n"
                 f"    python ESS_analysis.py --root runs --outdir .")
    m = pd.read_csv(summary_csv)
    missing = [c for c in REQUIRED if c not in m.columns]
    if missing:
        sys.exit(f"{summary_csv} is missing {missing}. This script expects the "
                 f"summary written by ESS_analysis.py, not the older "
                 f"ess_per_sec_summary.csv.")
    m["key"] = m["label"].map(key)

    runs = None
    if SEED_DOTS and per_run_csv and Path(per_run_csv).exists():
        runs = pd.read_csv(per_run_csv)
        runs["key"] = runs["label"].map(key)
        for c in ("reliable", "window_resolved"):
            if c in runs.columns:
                runs[c] = runs[c].astype(str).str.strip().str.lower() == "true"
    keys, labels = resolve_methods(set(m["key"]))
    return m, runs, keys, labels


def pick_metric(m):
    """RATE unless the wall clock never made it into the logs."""
    if m["ess_per_sec_mean"].notna().any():
        return RATE
    print("  ess_per_sec_mean is NaN in every row: run.log has no `wall_s` "
          "column, so a per-second rate does not exist. Plotting ESS per MC "
          "cycle instead -- the y label says so.", file=sys.stderr)
    return CYCLE


def diagnose(m, runs, keys, labels):
    """Say out loud what is thin and what is missing. Never guess."""
    print("\n--- cells ---------------------------------------------------",
          file=sys.stderr)
    for k in keys:
        for d in DIMS:
            row = m[(m.key == k) & (m.dim == d)]
            name = labels[k].replace("\n", " ")
            if row.empty:
                why = ("no rows at all in ess_per_run.csv either: the "
                       "method_dim directory was skipped (SKIP_PREFIXES / not "
                       "in ESS_analysis.METHODS), missing, or every run was "
                       "shorter than --min-rows")
                if runs is not None:
                    sub = runs[(runs.key == k) & (runs.dim == d)]
                    if len(sub):
                        bad = sub[~sub.get("reliable", True)]
                        why = (f"{len(sub)} seed(s) analysed, "
                               f"{len(bad)} dropped by the reliability filter "
                               f"(window_resolved and n_prod >= tol*g); "
                               f"notes: "
                               + "; ".join(sorted({str(n) for n in bad["note"]
                                                   if str(n).strip()})[:2]))
                print(f"  {name:24s} {d}  NO SUMMARY ROW -- {why}",
                      file=sys.stderr)
            else:
                r = row.iloc[0]
                print(f"  {name:24s} {d}  n_seeds={int(r.n_seeds):2d}  "
                      f"seeds={r.get('seeds', '?')}  "
                      f"max|z|={r.max_abs_drift_z:.1f}", file=sys.stderr)
    thin = m[m.n_seeds < m.n_seeds.max()]
    if len(thin):
        print(f"\n  {len(thin)} cell(s) have fewer seeds than the fullest "
              f"({int(m.n_seeds.max())}): drawn pale, no count printed on the "
              f"figure.", file=sys.stderr)
    drift = m[m.max_abs_drift_z > DRIFT_Z_MAX]
    if len(drift):
        print(f"  {len(drift)} cell(s) exceed |z| = {DRIFT_Z_MAX} on the "
              f"half-vs-half stationarity test (worst "
              f"{m.max_abs_drift_z.max():.1f}): their g is suspect.",
              file=sys.stderr)
    if "g_relspread_obs" in m and "g_relerr_theory" in m:
        loud = m[m.g_relspread_obs > 2 * m.g_relerr_theory]
        if len(loud):
            print(f"  {len(loud)} cell(s) scatter more between seeds than a "
                  f"single-chain IAT estimator should (g_relspread_obs > "
                  f"2*g_relerr_theory): the seeds may not share one "
                  f"stationary state.", file=sys.stderr)
    print(file=sys.stderr)


# ------------------------------------------------------------------ plot
def autoscale(vmax):
    """
    Engineering scale factor + the tag that must be shown somewhere, so
    2.3e6 reads as 2.29 with a corner "x10^6" rather than 2290000.
    """
    if not vmax or vmax <= 0:
        return 1.0, ""
    e = 0
    while vmax / 10.0 ** e >= 1000:
        e += 3
    return 10.0 ** -e, "" if e == 0 else rf"$\times 10^{{{e}}}$"


def collect(m, runs, keys, metric):
    """Geometry first: fix the y scale before any ink is laid."""
    mean_c, sem_c, run_c, scale, label = metric
    tag = ""
    if scale is None:
        scale, tag = autoscale(float(m[mean_c].max()))
    off = {d: (i - (len(DIMS) - 1) / 2) * BAR_W for i, d in enumerate(DIMS)}
    n_full = int(m["n_seeds"].max())

    bars, gaps, top = [], [], 0.0
    for i, k in enumerate(keys):
        for d in DIMS:
            row = m[(m.key == k) & (m.dim == d)]
            x = i + off[d]
            if row.empty or not pd.notna(row.iloc[0][mean_c]):
                gaps.append((x, d))
                continue
            r = row.iloc[0]
            v = float(r[mean_c]) * scale
            sem = float(r[sem_c]) * scale if sem_c in m and pd.notna(r[sem_c]) else 0.0
            err = CI_MULT * sem
            pts = []
            if runs is not None and run_c in runs.columns:
                sub = runs[(runs.key == k) & (runs.dim == d)]
                if "reliable" in sub:
                    sub = sub[sub.reliable]
                pts = [p * scale for p in sub[run_c].dropna().tolist()]
            bars.append(dict(x=x, dim=d, v=v, err=err, group=i, pts=pts,
                             thin=int(r.n_seeds) < n_full or int(r.n_seeds) < 2))
            top = max(top, v + err, max(pts, default=0.0))
    return bars, gaps, top, label, tag


def axes_points(ax):
    """(points per x data unit, axes height in points). Needs a laid-out canvas."""
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    lo, hi = ax.get_xlim()
    k = 72.0 / fig.dpi
    return bb.width / (hi - lo) * k, bb.height * k


def text_w(s, fs):
    """Rough width of a string in points. DejaVu Sans digits are ~0.60 em."""
    return 0.60 * fs * len(s)


def draw(ax, bars, gaps, top, keys, labels, ylabel, tag=""):
    ax.set_xlim(-0.62, len(keys) - 0.38)
    if LOG_Y:
        ax.set_yscale("log")

    # series names go on the bars of the tallest group that has BOTH dims --
    # direct labelling, no legend, and never a lone "2D" on a half-empty group
    full = [g for g in {b["group"] for b in bars}
            if len({b["dim"] for b in bars if b["group"] == g}) == len(DIMS)]
    pool = [b for b in bars if b["group"] in full] or bars
    anchor = max(pool, key=lambda b: b["v"])["group"] if pool else 0

    # --- how much room the text actually needs, measured, not guessed --------
    # Bar centres in a pair are only BAR_W apart: about 30 pt on a 6 in canvas.
    # At 22 pt a value label like "0.0566" is ~80 pt wide, so horizontal is
    # hopeless -- it is turned on its side instead, where it costs ~one em of
    # width and nothing collides. The series name ("2D") is short enough to
    # stay flat, but the pair is still split outward so the two never touch.
    pt_x, h_pt = axes_points(ax)
    annot_w = max((text_w(fmt_auto(b["v"]), FS_ANNOT) for b in bars), default=0.0)
    rot = ANNOT_ROTATION
    if rot is None:
        rot = 90 if annot_w > BAR_W * pt_x else 0
    split = len(DIMS) == 2 and text_w("2D", FS_SERIES) > BAR_W * pt_x
    if split and rot == 0:
        pad = BAR_W / 2 + max(annot_w - 5.0, 0.0) / pt_x + 0.06
        ax.set_xlim(min(-pad, -0.62), max(len(keys) - 1 + pad, len(keys) - 0.38))

    # Headroom is derived, not fixed. A rotated label is as tall as the number
    # is long, and stacking one under a 26 pt "2D" above the tallest bar would
    # eat a third of the axis, leaving the bars stubby. So a label that fits
    # inside its own bar is drawn there in white, and only what still sticks
    # out above a bar buys y range. Both directions feed back -- a shorter span
    # makes bars taller in points, which lets more labels move inside -- so the
    # pair is iterated to a fixed point, which it reaches in two or three passes.
    lab = {id(b): (text_w(fmt_auto(b["v"]), FS_ANNOT) if rot else 1.25 * FS_ANNOT)
           for b in bars}
    hi_of = {id(b): max([b["v"] + b["err"]] + list(b["pts"])) for b in bars}
    span, inside = top * HEADROOM, {}
    for _ in range(5):
        inside = {id(b): (ANNOT_INSIDE and rot and h_pt
                          and b["v"] / span * h_pt > lab[id(b)] + 18.0)
                  for b in bars}
        nxt = top
        for b in bars:
            need = 0.0 if inside[id(b)] else 6.0 + lab[id(b)]
            if b["group"] == anchor:
                need += 6.0 + 1.3 * FS_SERIES
            f = min(need / h_pt, 0.55) if h_pt else 0.0
            nxt = max(nxt, hi_of[id(b)] / (1.0 - f))
        nxt = max(nxt, top * HEADROOM)
        if abs(nxt - span) < 1e-9 * max(nxt, 1.0):
            break
        span = nxt
    ax.set_ylim(0, span)

    ax.yaxis.set_major_locator(ticker.MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
    ticks = [t for t in ax.get_yticks() if 0 < t <= ax.get_ylim()[1]]

    for b in bars:
        c = DIM_COLORS[b["dim"]]
        ax.bar(b["x"], b["v"], BAR_W * 0.94, color=c, linewidth=0, zorder=3,
               alpha=0.45 if (b["thin"] and FADE_PARTIAL) else 1.0)

        # Tufte: the scale lines are white, and they run THROUGH the bar --
        # except where a white number is already sitting there, or the two
        # whites cross and neither reads
        d = span / h_pt if h_pt else 0.0
        blank = None
        if inside[id(b)]:
            hi_b = b["v"] - 7.0 * d
            blank = (hi_b - (lab[id(b)] + 4.0) * d, hi_b + 4.0 * d)
        for t in ticks:
            if t < b["v"] and not (blank and blank[0] <= t <= blank[1]):
                ax.plot([b["x"] - BAR_W * 0.47, b["x"] + BAR_W * 0.47], [t, t],
                        color="white", lw=GRID_LW, solid_capstyle="butt",
                        zorder=4)

        hi = b["v"]
        if b["err"] > 0:
            ax.errorbar(b["x"], b["v"], yerr=b["err"], fmt="none", ecolor="0.3",
                        elinewidth=1.2, capsize=0, zorder=5)
            hi = b["v"] + b["err"]
        for p in b["pts"]:                       # the seeds, not the summary
            ax.plot(b["x"], p, "o", ms=3.2, mfc="none", mec="0.25", mew=0.9,
                    zorder=6)
            hi = max(hi, p)

        ha, dx = "center", 0.0
        if split and rot == 0:
            ha, dx = ("right", 5.0) if b["dim"] == DIMS[0] else ("left", -5.0)

        if inside[id(b)]:
            ax.annotate(fmt_auto(b["v"]), (b["x"], b["v"]),
                        textcoords="offset points", xytext=(0, -7),
                        ha="center", va="top", rotation=rot, fontsize=FS_ANNOT,
                        color="white", zorder=7)
        else:
            ax.annotate(fmt_auto(b["v"]), (b["x"], hi),
                        textcoords="offset points", xytext=(dx, 6),
                        ha="center" if rot else ha, va="bottom", rotation=rot,
                        fontsize=FS_ANNOT, color="0.2", zorder=7,
                        annotation_clip=False)
        if b["group"] == anchor:
            up = 6 + (0 if inside[id(b)] else lab[id(b)] + 6)
            sha, sdx = ("center", 0.0)
            if split:
                sha, sdx = ("right", 5.0) if b["dim"] == DIMS[0] else ("left", -5.0)
            ax.annotate(b["dim"], (b["x"], hi), textcoords="offset points",
                        xytext=(sdx, up), ha=sha, va="bottom",
                        fontsize=FS_SERIES, fontweight="bold",
                        color=DIM_TEXT[b["dim"]], zorder=7,
                        annotation_clip=False)

    for x, d in gaps:
        ax.text(x, top * 0.02, "no row in summary", rotation=90, ha="center",
                va="bottom", fontsize=FS_ANNOT, color="0.45")

    # -- Tufte style: no frame, no background gridlines, range-framed spines --
    # Both spines are range frames (Tufte, VDQI): each is drawn only across the
    # span the data occupy, so the axis line itself reports the data range.
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_bounds(-0.5, len(keys) - 0.5)
    ax.spines["left"].set_visible(Y_SPINE)
    if Y_SPINE:
        ax.spines["left"].set_bounds(0, ticks[-1] if ticks else top)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color("0.3")
        ax.spines[sp].set_linewidth(0.9)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([labels[k] for k in keys])
    ax.set_yticks(ticks)
    ax.tick_params(axis="y", direction="in", length=Y_TICK_LEN if Y_SPINE else 0,
                   labelsize=FS_TICK, colors="0.15")
    ax.tick_params(axis="x", length=0, labelsize=FS_TICK, colors="0.15")
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.yaxis.label.set_color("0.1")

    # the scale factor lives in the corner, never in the axis label
    if tag and SCALE_TAG:
        ax.text(0.0, 1.0, tag, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=FS_ANNOT, color="0.25")


def fmt_auto(v):
    """Value labels with as many digits as the magnitude deserves, no more."""
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def figure(m, runs, keys, labels, metric, name, out=OUT):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars, gaps, top, ylabel, tag = collect(m, runs, keys, metric)
    if not bars:
        plt.close(fig)
        print(f"  nothing to plot for {name}", file=sys.stderr)
        return
    draw(ax, bars, gaps, top, keys, labels, ylabel, tag)
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print(f"  wrote {p}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Tufte bar charts from ESS_analysis.py's ess_summary.csv.")
    ap.add_argument("summary", nargs="?", default=SUMMARY, type=Path)
    ap.add_argument("--per-run", default=None, type=Path,
                    help="ess_per_run.csv for seed dots "
                         "(default: alongside the summary)")
    ap.add_argument("-o", "--out", default=OUT, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    per_run = a.per_run if a.per_run is not None else a.summary.parent / PER_RUN

    m, runs, keys, labels = load(a.summary, per_run)
    diagnose(m, runs, keys, labels)
    figure(m, runs, keys, labels, pick_metric(m), "ESS_per_sec_2D_3D.png", a.out)
    figure(m, runs, keys, labels, GEE, "stat_inefficiency_2D_3D.png", a.out)

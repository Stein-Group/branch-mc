#!/usr/bin/env python3
"""Overlay your QA g(r) against the Lomba 1993 GCMC reference (Figs. 7 & 8).
Run after qa_rdf.py has produced the two averaged-RDF CSVs.

Set CSV_0303 / CSV_0601 / OUT below, then just:

    python qa_rdf_plot.py

Each rdf CSV has columns: r g11_mean g11_sem g01_mean g01_sem (from qa_rdf.py).
The *_sem columns are the standard error across matrix realizations, i.e. the
uncertainty on each mean g(r) curve. ERR_STYLE below controls how that gets
drawn: shaded band, capped error bars, or both.

Reference CSVs (ref_g11_*.csv, ref_g01_*.csv) sit in ./lomba_ref/ next to this
script. Produces a 2-row x 1-column comparison: one panel per state point, with
both g_solvent-solvent and g_matrix-solvent in each. The curve identity lives in
the legend, so the y axis carries no label.
"""
import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import blended_transform_factory

# ============================ DISPLAY TOGGLE =========================
ERR_STYLE = "both"       # "bars", "band", or "both"
ERR_EVERY = 1            # draw a capped error bar every Nth bin (bars/both)

CSV_0303 = "rdf_0.3_0.3.csv"     # rho0=0.3, rho1=0.3 (box 9.48)
CSV_0601 = "rdf_0.6_0.1.csv"     # rho0=0.6, rho1=0.1 (box 9.41)
OUT      = "lomba_compare.png"

FIGSIZE   = (8.0, 6.0)   # width x height, inches
FS        = 20           # ticks + legend + curve labels
ANN_FS    = 20           # the rho annotation sitting under the flat tail
ANN_SIDE  = ("above", "below")   # rho annotation placement per panel,
                                 # "above" or "below" the g=1 line
SRC_KEY   = ("line: BRANCH-MC "      # key above the legend entries;
             "square: Lomba 1993")           # one line each keeps it narrow

COL_SS    = "#5B54A8"    # g_solvent-solvent
COL_MS    = "#E07A3F"    # g_matrix-solvent
HEADROOM  = 1.14         # y padding above the tallest point so nothing clips
XPAD      = 0.06         # data-unit slack at x=1.0 so the "1.0" tick can't clip
YPAD_FRAC = 0.045        # fraction of the y range left under 0 for the "0" tick
# ====================================================================

REFDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lomba_ref")


def load_ref(name):
    #p = os.path.join(REFDIR, f"ref_{name}.csv")
    p = f"ref_{name}.csv"
    print("os.path.exists(p)", os.path.exists(p), p)
    return np.loadtxt(p, skiprows=1) if os.path.exists(p) else None


def draw_uncertainty(a, r, gm, ge, color):
    if ERR_STYLE in ("band", "both"):
        a.fill_between(r, gm - ge, gm + ge, color=color, alpha=0.25, lw=0)
    if ERR_STYLE in ("bars", "both"):
        s = slice(None, None, max(1, ERR_EVERY))
        a.errorbar(r[s], gm[s], yerr=ge[s], fmt="none",
                   ecolor=color, elinewidth=0.9, capsize=2, alpha=0.85)


def draw_pair(a, d, cm, ce, refname, color, label):
    """One g(r): your curve + its uncertainty + the Lomba squares.
    Returns (handles, label, top) with top = tallest drawn point."""
    r, gm, ge = d[:, 0], d[:, cm], d[:, ce]
    print(f"{label[:34]:36s} {refname:16s} SEM max={ge.max():.4g} median={np.median(ge):.4g}")
    line, = a.plot(r, gm, "-", color=color, lw=2.0, zorder=3)
    draw_uncertainty(a, r, gm, ge, color)
    top = float((gm + ge).max())

    ref = load_ref(refname)
    if ref is not None:
        mark, = a.plot(ref[:, 0], ref[:, 1], "s", color=color, ms=6.0,
                       mfc="none", mew=1.5, ls="none", zorder=4)
        top = max(top, float(ref[:, 1].max()))
    else:
        mark = a.plot([], [], "s", color=color, ms=6.0, mfc="none",
                      mew=1.5, ls="none")[0]
    return (line, mark), label, top


def main(csv_0303, csv_0601, out):
    d33 = np.loadtxt(csv_0303, skiprows=1)   # rho0=0.3 rho1=0.3 (box 9.48)
    d61 = np.loadtxt(csv_0601, skiprows=1)   # rho0=0.6 rho1=0.1 (box 9.41)

    L_SS = "RDF: solvent-solvent" # r"$g_{\text{solvent-solvent}}(r)$"
    L_MS = "RDF: matrix-solvent"  #r"$g_{\text{matrix-solvent}}(r)$"

    # (density annotation, data, reference suffix, x spine, y spine, y ticks, side)
    panels = [
            (r"$\rho:{\text{matrix}}=0.6,\ \rho:{\text{solvent}}=0.1$",
         d61, "0.6_0.1", (1, 4), (0.0, 6.0), [0, 2, 4, 6], ANN_SIDE[0]),
            (r"$\rho:{\text{matrix}}=0.3,\ \rho:{\text{solvent}}=0.3$",
         d33, "0.3_0.3", (1, 4), (0.0, 3.0), [0, 1, 2, 3], ANN_SIDE[1]),
    ]

    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE)
    for a, (dens, d, suf, xb, yb, yt, side) in zip(ax.flat, panels):
        h_ss, l_ss, t_ss = draw_pair(a, d, 1, 2, f"g11_{suf}", COL_SS, L_SS)
        h_ms, l_ms, t_ms = draw_pair(a, d, 3, 4, f"g01_{suf}", COL_MS, L_MS)
        top = max(t_ss, t_ms)

        a.axhline(1, color="gray", lw=0.6, ls=":", zorder=1)
        a.set_xlabel("r", fontsize=FS, labelpad=1)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(direction="in", labelsize=22, pad=4)

        # a little slack past the spine ends so the "1.0" and "0" ticks never clip
        a.set_xlim(xb[0] - XPAD, None)
        a.set_ylim(yb[0] - YPAD_FRAC * top * HEADROOM, top * HEADROOM)
        a.spines["bottom"].set_bounds(xb[0], xb[1])
        a.spines["left"].set_bounds(yb[0], yb[1])
        a.spines["left"].set_position(("outward", 2))
        a.set_yticks(yt)   # explicit: keeps ticks on the shortened left spine

        a.legend([h_ss, h_ms], [l_ss, l_ms],
                 handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
                 handlelength=3.4, handletextpad=0.7, columnspacing=1.2,
                 fontsize=FS, loc="upper right", framealpha=0.9, borderpad=0.4,
                 labelspacing=0.3,
                 title=SRC_KEY, alignment="left",
                 title_fontsize=20,
                 ncols=1)   # key line matches the entry font size

        # x in axes fraction, y in data units: constant gap from the g=1 tail,
        # mirrored above or below it depending on where the panel has room
        gap = 0.5 * (1.0 - yb[0])
        y_ann = 1.0 + gap if side == "above" else 1.0 - gap
        a.text(0.985, y_ann, dens, ha="right", va="center",
               fontsize=ANN_FS,
               transform=blended_transform_factory(a.transAxes, a.transData))

    fig.tight_layout(pad=0.6, h_pad=1.4)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    plt.show()


if __name__ == "__main__":
    main(CSV_0303, CSV_0601, OUT)

"""Collect framework runs and draw the benchmark figures.

Point it at the run roots your params files were launched into (any depth): it
reads every run folder holding a framework `params.json`, analyses it with the
C8 estimators (`benchmark_report.analyse_branch/analyse_raspa2`: one Sokal tau
and one particle-energy-evaluation cost for both codes) and writes
`results.csv`, one row per run and observable, loadings in mol/kg.

Which system, gas, state point and method a run is comes from its own
params.json, so nothing else has to be kept in step with the runs:
  system    mfm300_in (MFM-300-In.cif) or mgmof74_ch4 (MgMOF-74.cif)
  mixture   Xe, Kr, XeKr or CH4
  arm       branch_st / channel_st (suwa_todo), branch_gl / channel_gl (glauber),
            conv (metropolis_dynamics), cfcmc / cbmc / conv1 (raspa2_*)

Figures (PDF + PNG), drawn only for what the runs contain:
  fig_mfm300_in_isotherms   Xe, Kr, Xe/Kr 20:80 uptake and selectivity vs P
  fig_mgmof74_ch4           CH4 isotherm vs fugacity, cost per independent N
  fig_efficiency_mfm300_in  cost per independent sample: N (pure Xe), x_Xe (20:80)
  fig_robustness_mfm300_in  empty vs over-packed start at 10 bar, 20:80
  si_cost_control           conventional BRANCH_MC vs conventional RASPA2
  si_cfcmc_lambda           CFCMC lambda histograms and Wang-Landau weights
  si_henry_mfm300           with --henry-dir: Henry constants vs Table S4

Published curves are overlaid from `data/reference/*.csv` when present: `#`
provenance lines including `# x_unit:` (bar or Pa) and `# y_unit:` (mol/kg or
molecules/cif_cell), then `series,x,y` rows with series Xe, Kr, XeKr:Xe,
XeKr:Kr, S or CH4.

    python -m scripts.mof.benchmark_figures runs/mof_ch4 runs/mof_mfm --out figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from gcmc.framework.params import FrameworkRun, is_framework, parse
from gcmc.run.io import LOG_CYCLE, read_log
from scripts.mof.benchmark_report import (
    _n_total,
    analyse_branch,
    analyse_raspa2,
    branch_pee_per_row,
    mean_neighbours,
    pool_size,
)

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"
REFERENCES = {
    "mfm300_in": ("kancharlapalli2019_fig6c.csv", "Kancharlapalli 2019"),
    "mgmof74_ch4": ("torres_knoop2014_fig2a.csv", "Torres-Knoop 2014"),
}
# arm -> (legend label, colour slot, marker); categorical slots in fixed order,
# so an arm keeps its colour in every panel; markers carry identity as well
ARMS = {
    "branch_st": ("BRANCH (branch, ST)", "#2a78d6", "o"),
    "channel_st": ("BRANCH (channel, ST)", "#1baf7a", "s"),
    "cfcmc": ("RASPA2 CFCMC", "#eb6834", "^"),
    "cbmc": ("RASPA2 CBMC k=10", "#6250d6", "v"),
    "conv": ("BRANCH conventional", "#52514e", "D"),
    "conv1": ("RASPA2 conventional", "#a8a7a2", "d"),
    "branch_gl": ("BRANCH (branch, Glauber)", "#eda100", "P"),
    "channel_gl": ("BRANCH (channel, Glauber)", "#e87ba4", "X"),
}
# efficiency panels: the two production BRANCH bodies, the two RASPA2 advanced
# exchanges, and RASPA2's conventional swap as the baseline both try to beat
COST_ARMS = ["branch_st", "channel_st", "cfcmc", "cbmc", "conv1"]
REF_STYLE = {"color": "#8a8985", "lw": 1.6, "zorder": 1}


ARM_OF = {
    ("suwa_todo_dynamics", "branch"): "branch_st",
    ("suwa_todo_dynamics", "channel"): "channel_st",
    ("glauber_dynamics", "branch"): "branch_gl",
    ("glauber_dynamics", "channel"): "channel_gl",
    ("metropolis_dynamics", "metropolis"): "conv",
    ("raspa2_cfcmc", None): "cfcmc",
    ("raspa2_cbmc", None): "cbmc",
    ("raspa2_conventional", None): "conv1",
}
SYSTEM_OF = {"MFM-300-In": "mfm300_in", "MgMOF-74": "mgmof74_ch4"}


def _describe(fr: FrameworkRun, params: dict[str, Any]) -> dict[str, Any]:
    """System, gas, state point, arm and start of one run, from its params."""
    guests = [g for g in fr.guests if fr.y[g] > 0]
    if "pressure" in params["ensemble"]:
        kind, value, point = "p", fr.pressure_pa, f"{fr.pressure_pa / 1e5:g}bar"
    else:
        kind, value = "f", fr.pressure_pa
        point = f"{value:.0e}Pa".replace("+0", "")
    start = (
        "empty"
        if not fr.preload
        else ",".join(f"{k}={v}" for k, v in fr.preload.items())
    )
    return {
        "system": SYSTEM_OF.get(fr.cif.stem, fr.cif.stem),
        "mixture": "".join(guests),
        "point": point,
        "kind": kind,
        "value": value,
        "y": {g: fr.y[g] for g in guests},
        "arm": ARM_OF.get((fr.engine, fr.sampler.get("type")), fr.engine),
        "code": "raspa2" if fr.is_raspa else "branch_mc",
        "seed": fr.seed,
        "start": start,
    }


def _observables(guests: list[str]) -> list[str]:
    """N_total always; per-species N and x_<first> only for a mixture."""
    if len(guests) == 1:
        return ["N_total"]
    return ["N_total", *(f"N_{g}" for g in guests), f"x_{guests[0]}"]


def collect(roots: list[Path], burn: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pj in sorted(q for root in roots for q in Path(root).rglob("params.json")):
        params = json.loads(pj.read_text())
        log = pj.parent / "run.log"
        if not is_framework(params) or not log.exists():
            continue
        fr = parse(params)
        if fr.cif is None:  # an empty-box reference fluid, not a figure system
            continue
        run = _describe(fr, params)
        sp = json.loads((pj.parent / "state_point.json").read_text())
        cols = read_log(log)
        n_rows = len(cols[LOG_CYCLE])
        if n_rows < 20:
            continue
        obs = _observables(list(run["y"]))
        if fr.is_raspa:
            res = {
                o: analyse_raspa2(
                    run["arm"],
                    log,
                    pj.parent / "raspa2" / "simulation.input",
                    o,
                    burn,
                    pj.parent / "moves.json",
                )
                for o in obs
            }
            extra: dict[str, Any] = {"n_rows": n_rows}
        else:
            n_mean = float(np.mean(_n_total(cols)[n_rows // 2 :]))
            # pairs per PEE at this loading: framework atoms plus guests within rc
            ppp = mean_neighbours(
                fr.cif,
                fr.forcefield.path,
                fr.replicate,
                n_mean / float(np.prod(sp["box_angstrom"])),
            )
            res = {o: analyse_branch(run["arm"], pj.parent, o, burn, ppp) for o in obs}
            extra = {"n_rows": n_rows, "pairs_per_pee": ppp}
        to_molkg = 1000.0 / sp["framework_mass_amu"]
        for o, r in res.items():
            scale = 1.0 if o.startswith("x_") else to_molkg
            out.append(
                {
                    **{k: v for k, v in run.items() if k != "y"},
                    "observable": o,
                    "mean": r.mean * scale,
                    "sem": r.sem * scale,
                    "mean_raw": r.mean,
                    "tau_samples": r.tau_samples,
                    "n": r.n,
                    "reliable": r.reliable,
                    "pee_per_sample": r.pee_per_sample,
                    "tau_pee": r.tau_pee,
                    "molkg_per_molecule": to_molkg,
                    "framework_mass_amu": sp["framework_mass_amu"],
                    "cif_cells": sp["cif_cells"],
                    "run_dir": str(pj.parent),
                    **extra,
                }
            )
    return out


def write_results(rows: list[dict[str, Any]], path: Path) -> None:
    keys = list(rows[0]) if rows else []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------- figures
def _reference(system: str, mass: tuple[float, int] | None) -> dict[str, np.ndarray]:
    name, _ = REFERENCES[system]
    path = REFERENCE_DIR / name
    if not path.exists():
        return {}
    meta, rows = {}, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            meta[k.strip()] = v.strip()
        elif line.strip() and not line.startswith("series"):
            s, x, y = line.split(",")[:3]
            rows.append((s.strip(), float(x), float(y)))
    xs = 1.0 if meta.get("x_unit", "bar") in ("bar", "Pa") else math.nan
    ys = 1.0
    if meta.get("y_unit") == "molecules/cif_cell":
        if mass is None:
            raise ValueError(f"{path}: molecules per cell needs the framework mass")
        ys = 1000.0 / (mass[0] / mass[1])
    out: dict[str, list] = {}
    for s, x, y in rows:
        out.setdefault(s, []).append((x * xs, y * ys))
    return {k: np.asarray(sorted(v)) for k, v in out.items()}


def _sel(rows: list[dict], **kw) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def _xval(r: dict) -> float:
    return r["value"] / 1e5 if r["kind"] == "p" else r["value"]


def _isotherm(ax, rows: Iterable[dict], arms: list[str], obs: str, filled: bool = True):
    for arm in arms:
        pts = sorted(
            (_xval(r), r["mean"], r["sem"])
            for r in rows
            if r["arm"] == arm and r["observable"] == obs
        )
        if not pts:
            continue
        lab, col, mk = ARMS[arm]
        x, y, e = map(np.asarray, zip(*pts, strict=True))
        ax.errorbar(
            x,
            y,
            yerr=e,
            fmt=mk,
            ms=6,
            color=col,
            mfc=col if filled else "white",
            mew=1.4,
            lw=0,
            elinewidth=1.2,
            capsize=0,
            label=lab,
            zorder=3,
        )


def _style(ax, xlabel: str, ylabel: str, logx: bool = False):
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)


def fig_mfm(rows, out: Path, mass) -> None:
    import matplotlib.pyplot as plt

    rows = _sel(rows, system="mfm300_in")
    if not rows:
        return
    ref = _reference("mfm300_in", mass)
    arms = ["branch_st", "channel_st", "cfcmc", "conv1"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0))
    for ax, (mix, panel) in zip(
        axs.flat[:2], (("Xe", "(a) Xe"), ("Kr", "(b) Kr")), strict=True
    ):
        for s, xy in ref.items():
            if s == mix:
                ax.plot(
                    xy[:, 0], xy[:, 1], **REF_STYLE, label=REFERENCES["mfm300_in"][1]
                )
        _isotherm(ax, _sel(rows, mixture=mix), arms, "N_total")
        _style(ax, "pressure (bar)", "uptake (mol/kg)")
        ax.set_title(panel, loc="left", fontsize=10)
    ax = axs[1, 0]
    mixrows = _sel(rows, mixture="XeKr")
    for g, filled in (("Xe", True), ("Kr", False)):
        if f"XeKr:{g}" in ref:
            xy = ref[f"XeKr:{g}"]
            ax.plot(xy[:, 0], xy[:, 1], **REF_STYLE, ls="-" if filled else "--")
        _isotherm(ax, mixrows, arms, f"N_{g}", filled=filled)
    _style(ax, "pressure (bar)", "uptake (mol/kg)")
    ax.set_title("(c) Xe/Kr 20:80  (filled Xe, open Kr)", loc="left", fontsize=10)
    ax = axs[1, 1]
    for arm in arms:
        pts = []
        for r in _sel(mixrows, arm=arm, observable="x_Xe"):
            x = r["mean"]
            # S = (x_Xe / x_Kr) / (y_Xe / y_Kr), y = 0.2 / 0.8
            pts.append((_xval(r), (x / (1 - x)) / 0.25, r["sem"] / (1 - x) ** 2 / 0.25))
        if pts:
            lab, col, mk = ARMS[arm]
            x, y, e = map(np.asarray, zip(*sorted(pts), strict=True))
            ax.errorbar(
                x, y, yerr=e, fmt=mk, ms=6, color=col, lw=0, elinewidth=1.2, label=lab
            )
    if "S" in ref:
        ax.plot(ref["S"][:, 0], ref["S"][:, 1], **REF_STYLE)
    _style(ax, "pressure (bar)", "selectivity Xe/Kr")
    ax.set_title("(d) 20:80 selectivity", loc="left", fontsize=10)
    axs[0, 0].legend(frameon=False, fontsize=8)
    _save(fig, out, "fig_mfm300_in_isotherms")
    plt.close(fig)


RESOLVED_TAU = 1.0  # samples; below it the stored-sample spacing sets the cost


def _cost_panel(ax, rows, arms, obs: str, xobs: str | None) -> None:
    """Cost of one independent sample (tau x PEE per sample) against the loading
    `xobs` (or the state point when None). Open marker: tau unreliable (series
    shorter than 50 tau, a lower bound). Faded marker: tau below one stored sample,
    so the number is the output spacing, not the sampler (an upper bound)."""
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm and r["observable"] == obs]
        if not sub:
            continue
        lab, col, mk = ARMS[arm]
        load = {
            (r["point"], r["seed"]): r["mean"]
            for r in rows
            if r["arm"] == arm and r["observable"] == xobs
        }
        pts = sorted(
            (
                load[(r["point"], r["seed"])] if xobs else _xval(r),
                r["tau_pee"],
                r["reliable"],
                r["tau_samples"] >= RESOLVED_TAU,
            )
            for r in sub
        )
        x, y, ok, res = map(np.asarray, zip(*pts, strict=True))
        ax.plot(x, y, color=col, lw=1.2, alpha=0.5)
        good = ok & res
        ax.scatter(x[good], y[good], marker=mk, s=36, color=col, label=lab, zorder=3)
        ax.scatter(
            x[~ok], y[~ok], marker=mk, s=36, facecolor="white", edgecolor=col, zorder=3
        )
        ax.scatter(x[ok & ~res], y[ok & ~res], marker=mk, s=36, color=col, alpha=0.3)
    ax.set_yscale("log")


def fig_ch4(rows, out: Path, mass) -> None:
    import matplotlib.pyplot as plt

    rows = _sel(rows, system="mgmof74_ch4")
    if not rows:
        return
    ref = _reference("mgmof74_ch4", mass)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axs[0]
    if "CH4" in ref:
        ax.plot(
            ref["CH4"][:, 0],
            ref["CH4"][:, 1],
            **REF_STYLE,
            label=REFERENCES["mgmof74_ch4"][1],
        )
    _isotherm(ax, rows, ["branch_st", "channel_st", "cfcmc", "cbmc"], "N_total")
    _style(ax, "fugacity (Pa)", "CH$_4$ uptake (mol/kg)", logx=True)
    ax.set_title("(a) CH$_4$ in Mg-MOF-74, 313 K", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    _cost_panel(ax, rows, COST_ARMS, "N_total", "N_total")
    ax.legend(frameon=False, fontsize=7)
    _style(ax, "CH$_4$ uptake (mol/kg)", "PEE per independent sample")
    ax.set_title("(b) cost of one independent N", loc="left", fontsize=10)
    _save(fig, out, "fig_mgmof74_ch4")
    plt.close(fig)


def fig_efficiency_mfm(rows, out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _sel(rows, system="mfm300_in")
    if not rows:
        return
    arms = COST_ARMS
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    # against pressure: MFM-300(In) is near saturation from ~1 bar, so a loading
    # axis would stack every dense point into one column
    _cost_panel(axs[0], _sel(rows, mixture="Xe"), arms, "N_total", None)
    _style(axs[0], "pressure (bar)", "PEE per independent sample", logx=True)
    axs[0].set_title("(a) pure Xe, observable N", loc="left", fontsize=10)
    _cost_panel(axs[1], _sel(rows, mixture="XeKr"), arms, "x_Xe", None)
    _style(axs[1], "pressure (bar)", "PEE per independent sample", logx=True)
    axs[1].set_title("(b) Xe/Kr 20:80, observable x$_{Xe}$", loc="left", fontsize=10)
    axs[0].legend(frameon=False, fontsize=7)
    _save(fig, out, "fig_efficiency_mfm300_in")
    plt.close(fig)


def fig_robustness(rows, out: Path, point: str = "10bar") -> None:
    """Running loading against cumulative cost from an empty and an over-packed
    start at the densest mixture point (BRANCH arms; the start is the variable)."""
    import matplotlib.pyplot as plt

    runs = _sel(
        rows,
        system="mfm300_in",
        mixture="XeKr",
        point=point,
        code="branch_mc",
        observable="N_total",
    )
    if len({r["start"] for r in runs}) < 2:
        return
    # one curve per (method, start): the mean over its seeds, with a +-1 s.e.
    # band when there are several (the over-packed files run 16 seeds)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["arm"], r["start"]), []).append(r)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for (arm, start), members in groups.items():
        series = []
        for r in members:
            d = Path(r["run_dir"])
            stream = read_log(d / "run.log")
            cost = np.cumsum(
                branch_pee_per_row(stream, pool_size(d), r["pairs_per_pee"])
            )
            n = _n_total(stream)
            xe = stream["N_Xe"]
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(n > 0, xe / n, np.nan)
            series.append((cost, n * r["molkg_per_molecule"], frac))
        m = min(len(c) for c, _n, _f in series)
        cost = np.mean([c[:m] for c, _n, _f in series], axis=0)
        lab, col, _ = ARMS[arm]
        ls = "-" if start == "empty" else "--"
        tag = "empty" if start == "empty" else f"over-packed (n={len(series)})"
        for ax, k in ((axs[0], 1), (axs[1], 2)):
            ys = np.array([s_[k][:m] for s_ in series])
            with warnings.catch_warnings():  # rows where every seed has N = 0
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(ys, axis=0)
            ax.plot(
                cost,
                mean,
                color=col,
                ls=ls,
                lw=1.1,
                label=f"{lab}, {tag}" if k == 1 else None,
            )
            if len(series) > 1:
                se = np.nanstd(ys, axis=0) / np.sqrt(len(series))
                ax.fill_between(cost, mean - se, mean + se, color=col, alpha=0.15, lw=0)
    for ax, yl in zip(axs, ("total uptake (mol/kg)", "x$_{Xe}$ in pore"), strict=True):
        _style(ax, "cumulative PEE", yl)
        ax.set_xscale("log")
    axs[1].legend(
        *axs[0].get_legend_handles_labels(),
        frameon=False,
        fontsize=6,
        loc="lower right",
    )
    axs[0].set_title(f"(a) Xe/Kr 20:80, {point}", loc="left", fontsize=10)
    axs[1].set_title("(b) pore composition", loc="left", fontsize=10)
    _save(fig, out, "fig_robustness_mfm300_in")
    plt.close(fig)


def si_control(rows, out: Path) -> None:
    import matplotlib.pyplot as plt

    pairs = []
    for r in rows:
        if r["arm"] != "conv" or r["observable"] != "N_total":
            continue
        o = [
            q
            for q in rows
            if q["arm"] == "conv1"
            and q["observable"] == "N_total"
            and all(q[k] == r[k] for k in ("system", "mixture", "point"))
        ]
        if o:
            pairs.append((r, o[0]))
    if not pairs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for a, b in pairs:
        ok = a["reliable"] and b["reliable"]
        col = "#2a78d6" if a["system"] == "mfm300_in" else "#eb6834"
        kw = {"edgecolor": col, "facecolor": col if ok else "white", "s": 30}
        axs[0].scatter(b["mean"], a["mean"], **kw)
        axs[1].scatter(b["tau_pee"], a["tau_pee"], **kw)
    for ax, lab in zip(axs, ("uptake (mol/kg)", "PEE / independent N"), strict=True):
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color="#8a8985", lw=1)
        _style(ax, f"RASPA2 conventional: {lab}", f"BRANCH conventional: {lab}")
    axs[0].set_title("(a) blue MFM-300(In), orange Mg-MOF-74", loc="left", fontsize=9)
    axs[1].set_title("(b) open: tau unreliable", loc="left", fontsize=9)
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    _save(fig, out, "si_cost_control")
    plt.close(fig)


def si_lambda(rows, out: Path) -> None:
    import matplotlib.pyplot as plt

    runs = [
        r
        for r in rows
        if r["arm"] == "cfcmc"
        and r["observable"] == "N_total"
        and len(r["mixture"]) <= 3
    ]
    if not runs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(runs):
        d = Path(r["run_dir"])
        g = r["mixture"]
        stream = read_log(d / "run.log")
        lam = stream[f"lambda_{g}"][stream[f"frac_{g}"] > 0]
        bias = (
            json.loads((d / "moves.json").read_text()).get("cf_biasing_kT", {}).get(g)
        )
        col = cmap(i / max(1, len(runs) - 1))
        h, e = np.histogram(lam, bins=20, range=(0, 1), density=True)
        axs[0].step(e[:-1], h, where="post", color=col, lw=0.8, alpha=0.8)
        if bias:
            axs[1].plot(
                (np.arange(len(bias)) + 0.5) / len(bias),
                bias,
                color=col,
                lw=1.0,
                label=f"{r['system'].split('_')[0]} {r['mixture']} {r['point']}",
            )
    _style(axs[0], "$\\lambda$", "p($\\lambda$), production")
    _style(axs[1], "$\\lambda$", "Wang-Landau weight ($k_BT$)")
    axs[1].legend(frameon=False, fontsize=6, ncol=2)
    _save(fig, out, "si_cfcmc_lambda")
    plt.close(fig)


def si_henry(henry_dir: Path, out: Path) -> None:
    """Henry constants (v1_henry --json) against Kancharlapalli Table S4."""
    import matplotlib.pyplot as plt

    recs = [json.loads(f.read_text()) for f in sorted(henry_dir.glob("*.json"))]
    recs = [r for r in recs if r.get("anchor_table_s4")]
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for g, col, mk in (("Xe", "#2a78d6", "o"), ("Kr", "#eb6834", "s")):
        x = [r["anchor_table_s4"][g] * 1e5 for r in recs]
        y = [r["engine"][g]["k_henry"] * 1e5 for r in recs]
        ax.scatter(x, y, color=col, marker=mk, s=30, label=g, zorder=3)
        for r, xi, yi in zip(recs, x, y, strict=True):
            if g == "Xe":
                ax.annotate(
                    r["system"].replace("MFM-300-", ""),
                    (xi, yi),
                    fontsize=6,
                    xytext=(3, -8),
                    textcoords="offset points",
                )
    lo, hi = 0.8, 80
    ax.plot([lo, hi], [lo, hi], color="#8a8985", lw=1)
    _style(ax, "Table S4, 10$^{-5}$ mol kg$^{-1}$ Pa$^{-1}$", "this work")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "si_henry_mfm300")
    plt.close(fig)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "roots", type=Path, nargs="+", help="run roots (searched recursively)"
    )
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--burn", type=float, default=0.2, help="leading fraction dropped")
    p.add_argument("--henry-dir", type=Path, default=None, help="v1_henry --json files")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = collect(a.roots, a.burn)
    write_results(rows, a.out / "results.csv")
    print(f"{len(rows)} result rows -> {a.out / 'results.csv'}")
    import matplotlib

    matplotlib.use("Agg")
    masses = {}
    for r in rows:
        masses.setdefault(r["system"], (r["framework_mass_amu"], r["cif_cells"]))
    empty = [r for r in rows if r["start"] == "empty"]  # preloaded starts: Fig 3 only
    fig_mfm(empty, a.out, masses.get("mfm300_in"))
    fig_ch4(empty, a.out, masses.get("mgmof74_ch4"))
    fig_efficiency_mfm(empty, a.out)
    fig_robustness(rows, a.out)
    si_control(empty, a.out)
    si_lambda(empty, a.out)
    if a.henry_dir is not None:
        si_henry(a.henry_dir, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(_main())

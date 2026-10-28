"""C7: RASPA2 output -> a run.log in the format the BRANCH_MC runs write.

R1.8 makes RASPA2's PrintEvery stream the PRIMARY data source: record runs are
restart-chained legs killed by walltime, and RASPA2 writes its block averages
only when a leg completes. So this parser reads the instantaneous production
blocks of `Output/System_0/*.data`, one row per printed cycle, and treats the
end-of-run move statistics as a bonus that exists only for completed legs.

Every regex below is the C format string it reads, copied from RASPA2's own
source (iRASPA/RASPA2, src/statistics.c PrintIntervalStatusProduction and the
Print*Statistics functions in src/mc_moves.c), not inferred from an example
output. The parser fails loudly on a production block it cannot complete rather
than writing a row with a silently missing component.

Rows (`gcmc.run.io.RunLog`):
  MC_cycle (RASPA2 cycles, in millions, numbered continuously across legs),
  N_<comp>..., U_host_guest_K, U_guest_guest_K, and on a CFCMC run
  frac_<comp> (fractional molecules present) and lambda_<comp> (its lambda).

  A leg starts where the previous leg's last printed cycle ended, and its
  cycle-0 block, which is that same restart state, is dropped.

Moves (JSON, from completed legs only): per move kind and component, tried and
accepted, summed over legs.

    write_parsed([leg1.data, leg2.data], Path("run.log"), Path("moves.json"))
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# "Current cycle: %lld out of %lld\n" (production; init/equilibration carry a
# "[Init] " / "[Equilibration] " prefix and are skipped by the anchor)
_CYCLE = re.compile(r"^Current cycle: (\d+) out of (\d+)\s*$")
# "Component %d (%s), current number of integer/fractional/reaction molecules:
#  %d/%d/%d (avg. %9.5lf), density: ..."   (framework present)
# "... %d/%d/%d (average %9.5lf/%9.5lf), density: ..."     (no framework)
_COMP = re.compile(
    r"^Component (\d+) \((\S+)\), current number of integer/fractional/reaction "
    r"molecules: (\d+)/(\d+)/(\d+) \((?:avg\.|average)"
)
# "\tLambda factors: %4f %4f ..." -- first value is the fractional molecule's
# lambda for a single-site guest
_LAMBDA = re.compile(r"^\s*Lambda factors:\s+([-\d.eE+]+)")
# "\tBiasing Factors: %lf ..." , 10 per line, continued on indented lines of
# bare numbers: the CFCMC Wang-Landau weights (frozen during production)
_BIAS = re.compile(r"^\s*Biasing Factors:((?:\s+[-\d.eE+]+)+)\s*$")
_NUMS = re.compile(r"^((?:\s+[-\d.eE+]+)+)\s*$")
# "Current total potential energy:       % 22.10lf [K]  (avg. ...)"
_UTOT = re.compile(r"^Current total potential energy:\s+([-\d.eE+]+) \[K\]")
_UHA = re.compile(r"^\s*Current Host-Adsorbate energy:\s+([-\d.eE+]+) \[K\]")
_UAA = re.compile(r"^\s*Current Adsorbate-Adsorbate energy:\s+([-\d.eE+]+) \[K\]")

# --- end-of-run move statistics ---------------------------------------------
_PERF = re.compile(r"^Performance of the (.+) move:\s*$")
# translation: "Component %d [%s]" then "\ttotal        %f %f %f" and
# "\tsuccesfull   %f %f %f"
_TR_COMP = re.compile(r"^Component (\d+) \[(\S+)\]\s*$")
_TR_TOTAL = re.compile(r"^\s*total\s+([\d.]+) ([\d.]+) ([\d.]+)")
_TR_SUCC = re.compile(r"^\s*succesfull\s+([\d.]+) ([\d.]+) ([\d.]+)")
# swap addition/deletion, reinsertion: "Component [%s] total tried: %lf
# succesfull growth: %lf (%lf [%]) accepted: %lf (%lf [%])"
_GROW = re.compile(
    r"^Component \[(\S+)\] total tried: ([\d.]+) succesfull growth: [\d.]+ "
    r"\([\d.]+ \[%\]\) accepted: ([\d.]+)"
)
# identity change: "Component [%s]->[%s] total tried: %lf succesfull growth: ...
# accepted: %lf"
_IDENT = re.compile(
    r"^Component \[(\S+)\]->\[(\S+)\] total tried: ([\d.]+) succesfull growth: "
    r"[\d.]+ \([\d.]+ \[%\]\) accepted: ([\d.]+)"
)
# CFCMC swap lambda: "Component [%s] total tried: %lf constant-lambda accepted:
# %lf (...)" then two continuation lines "total tried: %lf insert-lambda
# accepted: %lf" and "... remove-lambda accepted: %lf"
_CF_CONST = re.compile(
    r"^Component \[(\S+)\] total tried: ([\d.]+) constant-lambda accepted: ([\d.]+)"
)
_CF_INS = re.compile(r"^\s*total tried: ([\d.]+) insert-lambda accepted: ([\d.]+)")
_CF_REM = re.compile(r"^\s*total tried: ([\d.]+) remove-lambda accepted: ([\d.]+)")

_MOVE_KIND = {
    "translation": "translation",
    "swap addition": "swap_add",
    "swap deletion": "swap_remove",
    "identity change": "identity",
    "CFCMC swap lambda": "cf_lambda",
}


@dataclass
class Leg:
    """One RASPA2 .data file: its component names, production rows, end-of-run
    move statistics (completed legs only) and the last Wang-Landau weights."""

    path: Path
    components: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    moves: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    completed: bool = False
    biasing: dict[str, list[float]] = field(default_factory=dict)  # last block


def _add(moves: dict, kind: str, comp: str, tried: float, acc: float) -> None:
    slot = moves.setdefault(kind, {}).setdefault(comp, [0.0, 0.0])
    slot[0] += tried
    slot[1] += acc


def parse_data_file(path: Path) -> Leg:
    """One RASPA2 .data file -> its production rows and (if present) move stats."""
    leg = Leg(path=Path(path))
    lines = Path(path).read_text(errors="replace").splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = _CYCLE.match(lines[i])
        if not m:
            pm = _PERF.match(lines[i])
            if pm:
                i = _parse_perf(lines, i, pm.group(1), leg)
                continue
            i += 1
            continue
        cycle = int(m.group(1))
        row: dict[str, Any] = {"cycle": cycle}
        comps: dict[int, tuple[str, int, int]] = {}
        lam: dict[str, float] = {}
        j = i + 1
        cur: str | None = None
        while j < n and not _CYCLE.match(lines[j]) and not _PERF.match(lines[j]):
            ln = lines[j]
            cm = _COMP.match(ln)
            if cm:
                idx, name = int(cm.group(1)), cm.group(2)
                comps[idx] = (name, int(cm.group(3)), int(cm.group(4)))
                cur = name
            else:
                lm = _LAMBDA.match(ln)
                if lm and cur is not None and cur not in lam:
                    lam[cur] = float(lm.group(1))
                bm = _BIAS.match(ln)
                if bm and cur is not None:
                    vals = [float(x) for x in bm.group(1).split()]
                    while j + 1 < n and (nm := _NUMS.match(lines[j + 1])):
                        vals += [float(x) for x in nm.group(1).split()]
                        j += 1
                    leg.biasing[cur] = vals
                for key, rx in (
                    ("u_total_K", _UTOT),
                    ("u_hg_K", _UHA),
                    ("u_gg_K", _UAA),
                ):
                    em = rx.match(ln)
                    if em:
                        row[key] = float(em.group(1))
            j += 1
        if not comps:
            raise ValueError(
                f"{path}: production block at cycle {cycle} has no components"
            )
        names = [comps[k][0] for k in sorted(comps)]
        if leg.components and names != leg.components:
            raise ValueError(f"{path}: component list changed mid-file: {names}")
        leg.components = names
        for k in sorted(comps):
            name, n_int, n_frac = comps[k]
            row[f"N_{name}"] = n_int
            row[f"frac_{name}"] = n_frac
            row[f"lambda_{name}"] = lam.get(name, "")
        row["N_total"] = sum(comps[k][1] for k in comps)
        leg.rows.append(row)
        i = j
    return leg


def _parse_perf(lines: list[str], i: int, title: str, leg: Leg) -> int:
    kind = _MOVE_KIND.get(title)
    j = i + 1
    n = len(lines)
    if kind is None:
        return j
    leg.completed = True
    if kind == "translation":
        comp = None
        while j < n and not _PERF.match(lines[j]):
            tc = _TR_COMP.match(lines[j])
            if tc:
                comp = tc.group(2)
            tt = _TR_TOTAL.match(lines[j])
            if tt and comp is not None:
                # RASPA2 counts translation attempts per Cartesian direction;
                # one attempt is one of x, y, z, so the move total is the sum
                tried = sum(float(x) for x in tt.groups())
                ts = _TR_SUCC.match(lines[j + 1]) if j + 1 < n else None
                acc = sum(float(x) for x in ts.groups()) if ts else 0.0
                _add(leg.moves, kind, comp, tried, acc)
                comp = None
            j += 1
        return j
    while j < n and not _PERF.match(lines[j]):
        ln = lines[j]
        if kind == "identity":
            im = _IDENT.match(ln)
            if im:
                _add(
                    leg.moves,
                    kind,
                    f"{im.group(1)}->{im.group(2)}",
                    float(im.group(3)),
                    float(im.group(4)),
                )
        elif kind == "cf_lambda":
            cm = _CF_CONST.match(ln)
            if cm:
                comp = cm.group(1)
                _add(
                    leg.moves,
                    "cf_lambda_constant",
                    comp,
                    float(cm.group(2)),
                    float(cm.group(3)),
                )
                if j + 2 < n:
                    a = _CF_INS.match(lines[j + 1])
                    b = _CF_REM.match(lines[j + 2])
                    if a:
                        _add(
                            leg.moves,
                            "cf_lambda_insert",
                            comp,
                            float(a.group(1)),
                            float(a.group(2)),
                        )
                    if b:
                        _add(
                            leg.moves,
                            "cf_lambda_remove",
                            comp,
                            float(b.group(1)),
                            float(b.group(2)),
                        )
        else:
            gm = _GROW.match(ln)
            if gm:
                _add(
                    leg.moves, kind, gm.group(1), float(gm.group(2)), float(gm.group(3))
                )
        j += 1
    return j


def stitch(legs: list[Leg]) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Legs in order -> (header, rows with leg + cycle_global, summed moves)."""
    if not legs:
        raise ValueError("no legs")
    comps = legs[0].components
    for lg in legs:
        if lg.components and lg.components != comps:
            raise ValueError(f"{lg.path}: components {lg.components} != {comps}")
    rows: list[dict[str, Any]] = []
    offset = 0
    for k, lg in enumerate(legs):
        last = 0
        for r in lg.rows:
            last = r["cycle"]
            if k > 0 and r["cycle"] == 0:
                # RASPA2 writes its restart file at the same PrintEvery cycles it
                # prints, so a resumed leg's cycle-0 block IS the previous leg's
                # last printed state. Keeping it would count one sample twice.
                continue
            rows.append({"leg": k, **r, "cycle_global": offset + r["cycle"]})
        offset += last
    header = [
        "leg",
        "cycle",
        "cycle_global",
        *[f"N_{c}" for c in comps],
        "N_total",
        *[f"frac_{c}" for c in comps],
        *[f"lambda_{c}" for c in comps],
        "u_total_K",
        "u_hg_K",
        "u_gg_K",
    ]
    moves: dict[str, Any] = {
        "completed_legs": sum(lg.completed for lg in legs),
        "moves": {},
        "cf_biasing_kT": legs[-1].biasing,
    }
    for lg in legs:
        for kind, per in lg.moves.items():
            for comp, (t, a) in per.items():
                slot = moves["moves"].setdefault(kind, {}).setdefault(comp, [0.0, 0.0])
                slot[0] += t
                slot[1] += a
    return header, rows, moves


def write_parsed(data: list[Path], log_path: Path, moves_path: Path | None) -> int:
    """Parse and stitch legs, write the run log (and the moves JSON); returns rows."""
    from ..run.io import LOG_CYCLE, RunLog

    header, rows, moves = stitch([parse_data_file(d) for d in data])
    comps = [h[2:] for h in header if h.startswith("N_") and h != "N_total"]
    cfcmc = any(r.get(f"lambda_{c}", "") != "" for r in rows for c in comps)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    log = RunLog(log_path)
    for r in rows:
        out: dict[str, Any] = {LOG_CYCLE: r["cycle_global"] / 1e6}
        for c in comps:
            out[f"N_{c}"] = int(r[f"N_{c}"])
        out["U_host_guest_K"] = float(r["u_hg_K"])
        out["U_guest_guest_K"] = float(r["u_gg_K"])
        if cfcmc:
            for c in comps:
                out[f"frac_{c}"] = int(r.get(f"frac_{c}") or 0)
                out[f"lambda_{c}"] = float(r.get(f"lambda_{c}") or 0.0)
        log.write(out)
    log.close()
    if moves_path is not None:
        moves_path.write_text(json.dumps(moves, indent=2, sort_keys=True) + "\n")
    return len(rows)

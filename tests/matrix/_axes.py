"""The feature axes of a run, and how to turn one point into a params dict.

WHY THIS FILE EXISTS (R13 of restructure_plan.md). A hand-written list of modes
goes stale the first time a feature is added: `tests/scripts/_harness.REF_MODES`
covers fourteen serial modes and silently misses parallel, reweight, pmf, NVT,
QA crossed with each sampler, the 2D channel, the cyclic rod, typed
multi-species and the whole framework path. Nobody notices, because a list that
is missing an entry looks exactly like a list that is complete.

So the matrix is not written. It is ENUMERATED, and the axis values come from
`examples/schema.json` -- the same machine-readable surface `from_params`
validates against and `test_schema_matches_code.py` already pins. Adding a
sampler to that enum puts it in the matrix automatically; there is no second
place to remember.

The division of labour, which is the whole design:

    this module        every axis and its values; how to BUILD a params dict
    _enumerate.py      the Cartesian product, partitioned by asking the
                       validator which points it accepts
    test_smoke.py      runs the accepted points and asserts invariants

Nothing here decides what is legal. `sim_setup.from_params` is the only
authority on that, so the matrix cannot drift away from the validator: if the
two disagree the disagreement IS the test failure.

UNITS. Everything in this module is in the engine's reduced units (sigma = 1,
epsilon = 1 for the untyped path); a typed point takes its units from the
species file it names. No quantity here is in Angstrom or Kelvin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = _ROOT / "examples" / "schema.json"
_FF = _ROOT / "ff"

# ---------------------------------------------------------------- tiny systems
#
# M14 of multispecies_plan.md sets the budget: every default-tier point runs on
# a box of 6 to 10, N of order 20 to 60, a few hundred cycles, one process. The
# numbers below are chosen so the WHOLE matrix fits the tier-2 budget of about
# five minutes, not so that any single point is fast.
#
# BOX is per dimension because a 2D box of 10 holds a sane number of disks while
# a 3D box of 10 at the same side holds far more; keeping the side fixed and
# letting N follow is what makes the two dimensions comparable at a glance.
BOX = {2: 10.0, 3: 8.0}

# A parallel run needs the box to hold at least two cells per axis with a border
# of `buffer >= rc` between them, so it cannot use the serial box. cell_plan
# derives the geometry from the count; this is the smallest box that admits
# 2 cells/axis at rc = 2.5.
PARALLEL_BOX = {2: 16.0, 3: 16.0}

SAMPLING_WINDOW = 40  # exchange steps per window
N_WINDOWS = 3  # enough to exercise accept AND reject on every branch
SEED = 20260824

RC = 2.5  # interaction cutoff, reduced units
TEMP = 1.0  # reduced temperature, LJ only


class Axis(NamedTuple):
    """One feature axis: its name, its values, and where it lives in params.

    `values` is every value the axis can take INCLUDING the "absent" case where
    that is meaningful (represented as None), because absence is itself a mode:
    a run with no `auxiliary` block is a different code path from one with
    `auxiliary.type = "pmf"`, and the matrix has to cover both.
    """

    name: str
    values: tuple[Any, ...]


def _schema() -> dict[str, Any]:
    """The parsed schema. Read once per process; it is the axis source of truth."""
    with _SCHEMA.open() as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _enum_at(path: str) -> tuple[Any, ...]:
    """The `enum` list at a dotted path in the schema, e.g. "sampler.type".

    Raises rather than defaulting if the path is missing: a renamed schema key
    must break the matrix loudly. Silently enumerating an empty axis would drop
    a whole feature from the suite and still report green, which is the exact
    failure this file exists to prevent (M15: fail at the earliest layer that
    can see the problem).
    """
    node: Any = _schema()
    for part in path.split("."):
        node = node["properties"][part]
    if "enum" not in node:
        raise KeyError(f"schema path {path!r} carries no enum; the matrix needs one")
    return tuple(node["enum"])


# ---------------------------------------------------------------------- axes
#
# Depth is an axis of its own rather than a property of the sampler, because the
# combination is exactly what the validator has opinions about: metropolis
# forbids it, channel requires >= 1, and `full_repacking` is its XOR opposite.
# Enumerating them independently and letting from_params reject the nonsense is
# what makes the rejected set (see _enumerate.rejected) worth reading.
#
# NOTE ON DEPTH RANGE. The schema currently caps depth at 3 while the code
# enforces only depth >= 0; that mismatch is a finding recorded in
# restructure_plan.md section 0 and fixed in S1. The axis deliberately probes
# depth 4 so the matrix REPORTS the disagreement instead of encoding one side of
# it -- parallelisation and the cavity algebra work at any depth, and the cap is
# a stale claim, not a constraint.
DEPTH_VALUES: tuple[Any, ...] = (0, 1, 2, 3, 4, "full_repacking")

AXES: tuple[Axis, ...] = (
    Axis("potential", _enum_at("potential.main")),
    Axis("sampler", _enum_at("sampler.type")),
    # THE DYNAMICS AXIS, and it is orthogonal to the sampler axis. `engine`
    # decides what happens to a proposed trial cloud -- a Metropolis
    # accept/reject test, or an exact conditional draw over the whole window --
    # while `sampler` decides how the cloud is proposed. Enumerating them
    # independently is what makes the REJECTED set worth reading: the Glauber engine
    # is legal only with `sampler = channel` at depth >= 1, so the product puts
    # every illegal pair in front of the validator and the matrix records which
    # message each one produced.
    Axis("engine", _enum_at("engine.type")),
    Axis("depth", DEPTH_VALUES),
    Axis("dimension", _enum_at("ensemble.dimension")),
    Axis("ensemble", _enum_at("ensemble.type")),
    Axis("kind", _enum_at("run.kind")),
    # None = no auxiliary block at all, which is the common case and its own path.
    Axis("auxiliary", (None, *_enum_at("auxiliary.type"))),
    # Rod length, numeric only. The wall-to-wall rod is NOT a length here any
    # more -- it is `sampler.type = "channel_cyclic"` (R16), which is why it
    # shows up on the sampler axis and gets its own points.
    #
    # This axis used to carry ("full",) as a second value, which silently made
    # 115 of 185 channel points cyclic under a label that read as a rod length.
    # Two rod lengths keep a genuine length axis without that confusion.
    Axis("channel_length", (6.0, 10.0)),
    # None = the untyped scalar path. A species file switches the whole typed
    # path on (P2, M9), so it is one axis with one switch, not a family of keys.
    # None = the untyped scalar path. A species file switches the whole typed
    # path on (P2, M9), so it is one axis with one switch, not a family of keys.
    # The shipped test file declares three species because three is the smallest
    # number that catches "the second species is special" bugs (M13).
    Axis("species_file", (None, "species_three_reduced.yaml")),
)

# Which main potential each species file describes. A file stating
# `interaction.form: lennard_jones` paired with `potential.main: hard_core` is
# rejected -- correctly, since the two disagree about the Hamiltonian -- so the
# matrix pairs them rather than enumerating the contradiction 1728 times.
SPECIES_FILE_POTENTIAL = {"species_three_reduced.yaml": "lennard_jones"}


def _needs_g3(potential: str, auxiliary: str | None) -> bool:
    """Whether this point needs the tabulated three-body g3 grid on disk.

    The 3D pmf auxiliary interpolates a g(r, cos theta) table shipped in
    `data/`; the 2D flavour is analytic and needs nothing. Kept as a predicate
    rather than a try/except at the call site so a missing table is a SKIP with
    a reason, never a silent pass.
    """
    return auxiliary == "pmf" and potential in ("hard_core", "lennard_jones")


class Point(NamedTuple):
    """One point of the matrix: the axis values, and a name to report it under."""

    potential: str
    sampler: str
    engine: str
    depth: Any
    dimension: int
    ensemble: str
    kind: str
    auxiliary: str | None
    channel_length: Any
    species_file: str | None

    @property
    def id(self) -> str:
        """A stable, greppable identifier. Only the axes that MATTER for this
        point appear, so `hard_core-branch-d1-3d-muVT-single` does not grow a
        `-len6.0` suffix it never used; two points that differ only in an
        irrelevant axis would otherwise look distinct and be run twice."""
        bits = [
            self.potential,
            self.sampler,
            f"d{self.depth}",
            f"{self.dimension}d",
            self.ensemble,
            self.kind,
        ]
        if self.engine != "metropolis_dynamics":
            bits.append(self.engine)
        if self.auxiliary:
            bits.append(f"aux_{self.auxiliary}")
        if self.sampler == "channel":
            bits.append(f"len_{self.channel_length}")
        if self.species_file:
            bits.append("typed")
        return "-".join(bits)


def build_params(point: Point, matrix_dir: Path | None = None) -> dict[str, Any]:
    """Turn one matrix point into a params dict for `sim_setup.from_params`.

    This function does NOT decide legality. It builds the most reasonable params
    for the point and hands them to the validator; an illegal combination is
    supposed to be rejected THERE, with a message naming the offending key, and
    that rejection is itself an assertion the matrix checks (test_coverage).
    Pre-filtering here would hide validator bugs behind a test helper.

    Parameters
    ----------
    point
        One point of the Cartesian product of `AXES`.
    matrix_dir
        Directory of generated QA obstacle files (the `matrix_dir` fixture).
        Enumeration passes None because it only needs to know WHETHER the
        validator accepts the shape of the params, and a QA point is accepted or
        rejected on the presence of the key, not the contents of the file. A
        point that is actually going to be advanced must pass the real directory.

    Returns
    -------
    dict
        A params.json-shaped dict. Every quantity is in reduced units.
    """
    dim = point.dimension
    box = PARALLEL_BOX[dim] if point.kind == "parallel" else BOX[dim]

    potential: dict[str, Any] = {"main": point.potential}
    if point.potential == "lennard_jones":
        potential["T"] = TEMP
        # The tail correction is meaningless in a pore and under a frozen
        # matrix, and sim_setup auto-disables it there; asking for it would be
        # rejected. R11 makes this derived rather than declared, but until then
        # the matrix has to respect the current rule.
        if point.ensemble in ("QA", "confined"):
            potential["tail"] = False
    if point.species_file is not None:
        potential["species_file"] = str(_FF / point.species_file)

    sampler: dict[str, Any] = {"type": point.sampler}
    # THE DYNAMICS AND ITS KNOBS NOW LIVE IN THEIR OWN BLOCK. `sampler` says
    # where the trial cloud comes from; `engine` says what is done with it, and
    # the group size and the repacking switch are the second question.
    #
    # The DEFAULT is written only when it is not the default, so a
    # `metropolis_dynamics` point produces exactly the params file it produced before
    # this axis existed -- which is what keeps every tier-2 digest of an
    # existing point where it was.
    engine: dict[str, Any] = {}
    if point.engine != "metropolis_dynamics":
        engine["type"] = point.engine
    # metropolis is standard GCMC: it is depth-0 by definition and the validator
    # forbids depth, trials, trial_density and trial_box_side outright. Depth is
    # a BRANCH-MC concept and does not apply, so the axis is simply not written
    # (and `_redundant` collapses the duplicate points that would result).
    if point.sampler != "metropolis":
        if point.depth == "full_repacking":
            engine["full_repacking"] = True
        else:
            engine["depth"] = point.depth
    if point.sampler in ("channel", "channel_cyclic"):
        # A channel draws its cloud by DENSITY, not by count: the rod's length
        # (and hence its trial count) changes per move for a full-span rod, so a
        # fixed `trials` is rejected. Using density for both rod kinds keeps the
        # two points comparable.
        sampler["trial_density"] = 1.5
        # The cyclic rod has no fixed length: it spans the box, so its extent is
        # set per move by the orientation. Writing one would be rejected.
        if point.sampler == "channel":
            sampler["channel_length"] = point.channel_length
        # The rod's cross-section must lie in [sigma, 2 sigma]: narrower than
        # sigma and no sphere fits, wider than 2 sigma and the "thin channel"
        # premise of the transfer-matrix sweep fails. The typed species file
        # has sigma up to 1.15, so 1.5 is the value legal on BOTH the untyped
        # (sigma = 1) and typed paths -- one number, both branches.
        sampler["channel_width"] = 1.5
    elif point.sampler != "metropolis":
        sampler["trials"] = 20
        sampler["trial_box_side"] = 3.0

    ensemble: dict[str, Any] = {"dimension": dim, "box": box, "type": point.ensemble}
    # Both fixed-N ensembles take a count. They differ in what else they take:
    # NVT refuses a state point (its composition is the start file's, and with a
    # species file it also demands one, so typed NVT stays a REJECTED point here
    # -- deliberately, since a matrix point has no initial_config), while
    # semigrand requires exactly one, because the activity RATIO is what moves
    # composition in it.
    if point.ensemble in ("NVT", "semigrand"):
        ensemble["N"] = 20
    if point.ensemble == "NVT":
        pass  # NVT states no state point at all, typed or not
    elif point.species_file is not None:
        # Typed mode states one activity per species (M10). The species file
        # shipped for the test suite declares three, and a missing or unknown
        # name is an error naming it, so the map is written out in full rather
        # than defaulted.
        ensemble["mu"] = {"A": 1.0, "B": 1.0, "C": 1.0}
    else:
        ensemble["mu"] = 1.0
    if point.ensemble == "confined":
        ensemble["pore"] = {"geometry": "cylinder", "R": 3.0}

    params: dict[str, Any] = {
        "potential": potential,
        "sampler": sampler,
        "ensemble": ensemble,
        "run": {
            "kind": point.kind,
            "sampling_window": SAMPLING_WINDOW,
            "seed": SEED,
        },
    }
    # ABSENT WHEN EMPTY, not `{}`: a metropolis point with no group size wrote
    # no such block before this move and must keep writing none, or every
    # metropolis digest shifts for a spelling change.
    if engine:
        params["engine"] = engine

    if point.auxiliary is not None:
        aux: dict[str, Any] = {"type": point.auxiliary}
        if point.auxiliary == "pmf" and dim == 3:
            aux["g3_csv"] = str(_ROOT / "data" / "g3_analysis_data.csv")
        elif point.auxiliary == "pmf":
            aux["eta"] = 0.69
        else:
            # short_range proposes into a radius SHORTER than the physics
            # cutoff (M12); equal or longer costs work and buys nothing. 2.0
            # against rc = 2.5 keeps it strictly shorter and stays outside the
            # first LJ shell (~1.5 sigma), below which acceptance collapses.
            aux["cutoff"] = 2.0
        params["auxiliary"] = aux

    if point.ensemble == "QA":
        # The obstacle file is generated by the `matrix_dir` fixture under a
        # name keyed on (dimension, box). During enumeration no directory
        # exists yet, so a sentinel is used: the validator's QA rule is about
        # the KEY being present, and resolving a real path there would make
        # enumeration depend on fixture setup order.
        base = matrix_dir if matrix_dir is not None else Path("/nonexistent")
        params["run"]["matrix_config"] = str(base / f"matrix_{dim}d_{box:g}.txt")

    if point.kind == "parallel":
        # One cell is one core (cell_plan AUTO mode), so the count IS the worker
        # count and is never stated twice. Two per axis is the smallest
        # decomposition that has a border at all.
        #
        # `buffer` is deliberately NOT set alongside: n_cell selects AUTO mode,
        # where the geometry is DERIVED (pitch = L/n, cell_length = pitch -
        # buffer), and stating both is how a run ends up with two competing
        # definitions of the same length. The validator rejects the pair, which
        # is the behaviour we want and which this matrix records.
        params["run"]["n_cell"] = 2**dim

    return params

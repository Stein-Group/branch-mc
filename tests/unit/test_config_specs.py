"""The sub-configs together lose nothing (R8, second half).

This is the gate for the projection. A decomposition that drops a field is worse
than no decomposition: modules would migrate onto sub-configs, the dropped field
would go unread, and the run would be subtly wrong in a way no type checker
sees. So the round trip is asserted on every mode the feature matrix accepts,
not on one hand-picked config.

WHAT THE ROUND TRIP CHECKS. `Config -> the five submission specs -> Config`
must recover every field the user SUBMITTED. `config_fields.DERIVED` is
deliberately outside that loop: those are computed from the submitted ones, so a
Config rebuilt from the five recomputes them, and round-tripping them would test
`Config.__post_init__` rather than the partition.

`KernelSpec` -- the sixth spec, carrying exactly that derived group -- is
therefore gated differently, and gated it is: coverage (the six together leave no
stored field behind), non-identity (no derived field is ever hashed), and
value-fidelity (projection changes neither the value nor its type). The three
together are what let a kernel take a `KernelSpec` in place of a `Config` and
still be provably bit-identical.
"""

from __future__ import annotations

import dataclasses

import pytest

from gcmc import config_specs as CS
from gcmc.run import sim_setup
from tests.matrix import _enumerate as E
from tests.matrix._axes import build_params


def _all_stored(cfg: object) -> set[str]:
    """Every stored field name on a Config."""
    return {f.name for f in dataclasses.fields(cfg)}  # type: ignore[arg-type]


def _reference_cfg():
    """One ordinary LJ/branch config, enough to enumerate field NAMES."""
    return sim_setup.from_params(
        {
            "potential": {"main": "lennard_jones", "T": 1.0},
            "sampler": {"type": "branch", "depth": 1, "trials": 20},
            "ensemble": {"mu": 1.0, "dimension": 3, "box": 8.0},
            "run": {"sampling_window": 20},
        }
    )


def test_the_six_cover_every_stored_field() -> None:
    """No stored field is missing from all six sub-configs.

    The failure this prevents: a field added to `Config` but to no group. It
    would vanish from the projection AND from the identity, so two runs
    differing in it would collide -- silently, because nothing reads a field
    that nobody projected.

    The `- set(CF.DERIVED)` escape hatch this assertion used to need is GONE.
    Derived fields were excluded because no spec carried them; `KernelSpec` now
    does, so the six specs cover the stored half of `Config` with no exemption
    and the test states that directly. An exemption is a place a field can hide,
    and this one no longer has to exist.
    """
    cfg = _reference_cfg()
    projected: set[str] = set()
    for spec in (
        CS.potential_spec(cfg),
        CS.sampler_spec(cfg),
        CS.ensemble_spec(cfg),
        CS.execution_spec(cfg),
        CS.io_spec(cfg),
        CS.kernel_spec(cfg),
    ):
        projected |= {f.name for f in dataclasses.fields(spec)}

    uncovered = _all_stored(cfg) - projected
    assert not uncovered, (
        f"stored Config fields covered by no sub-config: {sorted(uncovered)}\n"
        "Add each to the matching group in config_fields AND to the matching "
        "class in config_specs -- the import-time check requires both."
    )


def test_the_kernel_spec_never_reaches_the_identity() -> None:
    """Not one derived field is hashed.

    The same guard `test_execution_and_io_never_reach_the_identity` provides for
    the other two non-physics specs, and it matters more here: every KernelSpec
    field is a FUNCTION of a hashed field, so one leaking in would not merely
    split runs -- it would double-count its own inputs. That is the concrete
    defect `config_fields` was written to end, where `sampler` reached the old
    digest four separate ways (`potential_type`, `branch`, `channel`,
    `metropolis`) while being hashed itself.
    """
    from gcmc.run import hash_fingerprint

    derived = {f.name for f in dataclasses.fields(CS.KernelSpec)}
    leaked = sorted(derived & set(hash_fingerprint._IDENTITY_FIELDS))
    assert not leaked, (
        f"derived fields reached the identity: {leaked}. Each is a function of "
        "a field already hashed, so hashing it double-counts its inputs."
    )


def test_the_kernel_spec_reads_the_same_values_as_config() -> None:
    """Projection carries derived values through unchanged.

    Trivial by construction today (`_project` is a getattr loop), and asserted
    anyway because the migrations that follow depend on it: a kernel handed a
    `KernelSpec` instead of a `Config` must compute bit-identically, and the
    counter digests can only prove that if this holds. If `kernel_spec` ever
    grows a conversion -- a tuple() around `box_dims`, a float() around `beta` --
    this is what says so.
    """
    cfg = _reference_cfg()
    ks = CS.kernel_spec(cfg)
    for f in dataclasses.fields(CS.KernelSpec):
        want, got = getattr(cfg, f.name), getattr(ks, f.name)
        assert type(want) is type(got), f"{f.name}: {type(want)} became {type(got)}"
        assert want is got or want == got, f"{f.name} changed in projection"


def _covering_points() -> list:
    """The pairwise cover: every axis value and every legal pair of values.

    Round-tripping one config would prove almost nothing -- most fields are None
    on any single mode, and None survives any bug. The cover exercises pore
    geometry, matrix coordinates, typed species, auxiliaries, parallel cells and
    both rod kinds, which is where a dropped field would actually show.
    """
    return E.pairwise_cover(E.accepted())


@pytest.mark.parametrize("point", _covering_points(), ids=lambda p: p.id)
def test_round_trip_loses_nothing(point, matrix_dir) -> None:
    """Config -> five sub-configs -> Config reproduces every stored field."""
    cfg = sim_setup.from_params(build_params(point, matrix_dir=matrix_dir))

    rebuilt: dict[str, object] = {}
    for spec in (
        CS.potential_spec(cfg),
        CS.sampler_spec(cfg),
        CS.ensemble_spec(cfg),
        CS.execution_spec(cfg),
        CS.io_spec(cfg),
    ):
        for f in dataclasses.fields(spec):
            rebuilt[f.name] = getattr(spec, f.name)

    stored = _all_stored(cfg)
    mismatched = []
    for name, value in rebuilt.items():
        if name not in stored:
            continue  # a property (e.g. species_identity), not a stored field
        original = getattr(cfg, name)
        same = original is value or original == value
        # Arrays compare elementwise; reduce to one verdict.
        if hasattr(same, "all"):
            same = bool(same.all())
        if not same:
            mismatched.append(name)

    assert not mismatched, f"{point.id}: projection changed {mismatched}"


def test_identity_is_exactly_the_three_physics_specs() -> None:
    """The hashed field set equals the union of the three physics sub-configs.

    Stated against the CLASSES rather than the groups, so the fingerprint,
    `config_fields` and `config_specs` are pinned to each other in a triangle.
    Any two of them agreeing while the third drifts is caught here.
    """
    from gcmc.run import hash_fingerprint

    union: set[str] = set()
    for cls in (CS.PotentialSpec, CS.SamplerSpec, CS.EnsembleSpec):
        union |= {f.name for f in dataclasses.fields(cls)}

    assert set(hash_fingerprint._IDENTITY_FIELDS) == union


def test_execution_and_io_never_reach_the_identity() -> None:
    """Not one execution or IO field is hashed.

    The complement of the test above, written separately because it is the more
    dangerous direction: an execution field leaking into the identity would
    split one experiment across many directories and quietly break resume, and
    it would look like nothing more than a lot of run folders.
    """
    from gcmc.run import hash_fingerprint

    never: set[str] = set()
    for cls in (CS.ExecutionSpec, CS.IOSpec):
        never |= {f.name for f in dataclasses.fields(cls)}

    leaked = sorted(never & set(hash_fingerprint._IDENTITY_FIELDS))
    assert not leaked, f"execution/IO fields reached the identity: {leaked}"


def test_spec_annotations_match_config() -> None:
    """A spec field's declared type is the one `Config` declares.

    The companion to the import-time NAME check in `config_specs`, and it was
    missing. Names alone are what keep a field inside the identity; TYPES are
    what make a spec usable as a replacement for `Config` at a call site.
    Twenty-four had drifted by the time the first kernel migrated onto them --
    eighteen widened to `| None` against a `Config` that promises a value, and
    six narrowed against a `Config` that allows more, including `n_cell` losing
    the per-axis `list[int]` the launcher documents and `lj_tail_pref` -- a
    float coefficient multiplied into `delta_E` -- declared `bool`.

    The widened ones are why this is a gate and not a one-off cleanup. A
    spurious `| None` fails nothing; it makes the next migration add a dead
    branch or reach for `# type: ignore`, and once that habit exists a genuine
    optional slips through behind it.

    WHY IT LIVES HERE RATHER THAN BESIDE THE NAME CHECK. It needs `Config`, and
    `config_specs` is L0 vocabulary while `sim_setup` is L7 -- importing one
    from the other is the upward edge the ladder refuses, function-local import
    or not. A test sits outside the ladder and can see both. `Any` opts a field
    out, for the handful whose stored value is a numpy array or a species object
    that no annotation usefully describes.
    """
    import typing

    cfg_hints = typing.get_type_hints(sim_setup.Config)
    drift = []
    for cls, _group, _name in CS._PAIRS:
        spec_hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            want = cfg_hints.get(f.name)
            if want is None:
                continue  # a property, not a stored field
            got = spec_hints[f.name]
            if got is typing.Any or got == want:
                continue
            drift.append(f"{cls.__name__}.{f.name}: Config={want} spec={got}")

    assert not drift, (
        "spec annotations disagree with Config:\n  " + "\n  ".join(drift) + "\n"
        "Copy the annotation from Config, or use `Any` if no annotation "
        "usefully describes the stored value."
    )

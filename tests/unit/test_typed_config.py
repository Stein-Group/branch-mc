"""Gate P2 of multispecies_plan.md: the typed configuration surface.

Config only, no physics, every test sub-second. The rules under test are M9 (the
species file is the model, params.json is the experiment), M10 (three mutually
exclusive state-point routes), M11 (lambda is part of the physics identity) and
section 4.2.

The load-bearing test in here is the first one: a single-species params file must
produce a `Config` field for field identical to today's, and a fingerprint
byte-identical to today's. Everything else is new surface; that one is the
promise that the new surface costs the old path nothing.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from gcmc import config_fields
from gcmc.run.hash_fingerprint import identity_dict, run_id

#: Pinned digest of the canonical untyped run. See point 4 below for its history.
# Regenerated THREE times for one feature and its clean-up, and only the first
# was a change of content:
#
#   e864f492d3ba -> 36c18832f07b   `engine` and `kmax_margin` joined
#                                  config_fields.SAMPLER (design plan, 3.4).
#                                  A NEW FIELD reaching a hashed group, which
#                                  is the whole design of `config_fields`.
#   36c18832f07b -> e40cfc02d816   the two engine VALUES renamed:
#                                  `metropolis_mc` -> `metropolis_dynamics`,
#                                  `window_heat_bath` -> `glauber_dynamics`.
#                                  The default's SPELLING is hashed.
#   e40cfc02d816 -> 45254c9bfde0   the field `no_of_trial_molecules` renamed to
#                                  `n_trial_molecules`. `identity_dict` keys the
#                                  hash on `field.name`, so a field rename is a
#                                  key rename is a different digest.
#
# THE LAST TWO COST NOTHING BEYOND THE FIRST, which is why they were allowed to
# land here rather than being deferred: the first move had already invalidated
# every pre-`engine` run directory in the project, so renaming afterwards
# orphans nothing that was not already orphaned. Doing it later, once runs
# existed at e40cfc02d816, would have cost a real migration.
#
# The alternative in each case was worse: hashing a legacy spelling or a legacy
# field name that the code no longer uses anywhere puts a lie in the identity to
# protect a directory that is already gone. R9 sanctions a moved run id with a
# stated cause; all three causes are here and in
# docs/decisions/DIGEST_PROVENANCE.md, and
# docs/decisions/GLAUBER_DYNAMICS_REPORT.md says why the default is not
# excluded from the digest in the first place.
_UNTYPED_RUN_ID = "9c9ffdf52c84"
from gcmc.run.sim_setup import from_params

FF = Path(__file__).resolve().parents[2] / "ff"
BINARY = str(FF / "species_binary_reduced.yaml")
THREE = str(FF / "species_three_reduced.yaml")
XEKR = str(FF / "species_xekr_reduced.yaml")


def _typed(**over) -> dict:
    p = {
        "potential": {
            "main": "lennard_jones",
            "T": 1.2,
            "species_file": BINARY,
        },
        "ensemble": {"mu": {"A": -2.0, "B": -2.5}, "dimension": 3, "box": 8.0},
        # rosenbluth, not metropolis: this module tests typed CONFIG PARSING and
        # only needs some valid sampler, but `sim_setup` now refuses typed +
        # metropolis (that engine is species-blind in three separate ways -- see
        # docs/decisions/DIGEST_PROVENANCE.md, S2 commit 13). The untyped fixture
        # below keeps metropolis, which is exactly the contrast: the refusal is
        # about the pairing, not about either half of it.
        "sampler": {"type": "rosenbluth", "trials": 20},
        "run": {},
    }
    for block, keys in over.items():
        p.setdefault(block, {})
        if keys is None:
            p.pop(block)
        else:
            # A None VALUE deletes that key, which is how a test swaps one
            # state-point route for another: the base dict carries `mu`, and the
            # routes are mutually exclusive, so overriding without deleting would
            # only ever exercise the two-routes-at-once rejection.
            merged = {**p.get(block, {}), **keys}
            p[block] = {k: v for k, v in merged.items() if v is not None}
    return p


def _untyped() -> dict:
    return {
        "potential": {"main": "lennard_jones", "T": 1.2},
        "ensemble": {"mu": -2.0, "dimension": 3, "box": 8.0},
        "sampler": {"type": "metropolis"},
        "run": {},
    }


def _raises(p, message):
    with pytest.raises(ValueError) as exc:
        from_params(p)
    assert message in str(exc.value), f"message was: {exc.value}"


# ------------------------------------------------- the old path is untouched


def test_a_single_species_config_is_field_for_field_what_it_was():
    """No species file means no typed anything: the spec is None, n_species is 1,
    and every scalar field keeps its old default."""
    cfg = from_params(_untyped())
    assert cfg.species is None
    assert cfg.species_identity is None
    assert cfg.n_species == 1
    assert cfg.radius == 0.5
    assert cfg.lower_cutoff == 0.9
    assert cfg.upper_cutoff == 2.5
    assert cfg.volume_of_molecule == (4.0 / 3.0) * math.pi * 0.5**3
    assert cfg.chem_pot == -2.0


def test_the_untyped_fingerprint_does_not_gain_a_null_key():
    """species_identity is dropped, not hashed as None, so every run id produced
    before typed mode existed is still the same twelve hex digits.

    The load-bearing assertion is the key SET: a new identity key leaking into
    the untyped path is what would silently re-address every existing run
    directory, and that is caught here without reference to any stored number.
    The hash below is a regression pin measured on this tree, not evidence about
    what a pre-typed id was -- the id has moved once already for a good reason
    (`seed` became identity-bearing after tests/ref/engineA was frozen, so
    engineA computes a different digest for the same params). If the key set
    below is intact and this digest moves, the cause is a changed VALUE, and the
    diff of identity_dict says which one.
    """
    ident = identity_dict(from_params(_untyped()))

    # 1. An untyped run carries no typed model, so the key is ABSENT.
    assert "species_identity" not in ident

    # 2. NO key hashes as null. This is the property the test is named for, and
    #    it is now general rather than a list: `identity_dict` omits any field
    #    whose value is None, so an inactive feature contributes nothing at all.
    #    Before S2 eight of twenty-five keys hashed as literal null on an
    #    ordinary run, while four others had bespoke skip logic -- two
    #    conventions applied by accident of history.
    nulls = sorted(k for k, v in ident.items() if v is None)
    assert not nulls, f"identity gained null keys: {nulls}"

    # 3. The hashed set is exactly the physics half of the Config partition.
    #    Asserted by DERIVATION, not by a literal list: writing the names out
    #    here would be a third copy of the partition and a third thing to drift.
    #    `tests/unit/test_identity_partition.py` owns the grouping's contents;
    #    this asserts that the fingerprint uses it and nothing else.
    assert set(ident) <= set(config_fields.PHYSICS)
    assert set(ident) == {k for k in config_fields.PHYSICS if k in ident}

    # 4. Regression pin on the VALUE. Regenerated 2026-10-17, twice in one
    #    change: `channel_cyclic` LEFT `config_fields.SAMPLER` with the cyclic
    #    rod, and `channel_full` joined it for the full-span rod
    #    (`sampler.channel_length = "full"`). It is False on every run that does
    #    not ask for that body, so no chain changed; the digest moved because a
    #    new field in a physics group reaches it by virtue of the group, which
    #    is the whole design of `config_fields` and the same reason the previous
    #    two regenerations below happened. Before that, regenerated 2026-09-10,
    #    when `engine` and `kmax_margin` joined `config_fields.SAMPLER` for the
    #    Glauber engine.
    #    Previous values, in order: fbdc3125c74a (pre-S1), 0c683de8d49b (S1,
    #    after lj_tail_pref and pore_axis_len were added), e864f492d3ba (S2,
    #    when the identity set became `config_fields.PHYSICS`). Run ids moving
    #    is sanctioned by R9; what is not sanctioned is moving one without
    #    saying why, which is how the earlier pin 1a24fe1c17aa lost its
    #    provenance.
    #
    #    WHY THIS ONE MOVED, and why the default was not excluded to avoid it.
    #    A new field in a physics group reaches the digest because of the group
    #    it was declared in -- that is the whole design of `config_fields`, and
    #    special-casing `engine == "metropolis_dynamics"` out of the hash would mean
    #    re-introducing exactly the hand-maintained identity logic the partition
    #    replaced. Existing Metropolis params files still validate and run
    #    correctly; they start a fresh run directory rather than resuming an old
    #    one, and resuming a pre-`engine` directory is explicitly not a
    #    requirement. Precedent: `upper_cutoff` and `no_neigh`.
    assert run_id(from_params(_untyped())) == _UNTYPED_RUN_ID


def test_a_route_map_without_a_species_file_is_refused():
    p = _untyped()
    p["ensemble"]["mu"] = {"A": -2.0}
    _raises(p, "potential.species_file")
    p = _untyped()
    p["ensemble"]["z"] = 0.1
    _raises(p, "potential.species_file")


# ------------------------------------------------------------ M10 the routes


def test_the_mu_route_consults_lambda():
    """z_s = exp(beta mu_s) / lambda_s^d. The three-species file has lambdas
    1, 1.25 and 0.8, so a dropped lambda would show up here as a factor of
    1.25^3 = 1.953125 on B."""
    cfg = from_params(
        _typed(
            potential={"species_file": THREE},
            ensemble={"mu": {"A": -2.0, "B": -2.0, "C": -2.0}},
        )
    )
    s = cfg.species
    assert s is not None and s.route == "mu" and s.lam_consulted
    beta = 1.0 / 1.2
    expect = math.exp(beta * -2.0) / np.array([1.0, 1.25, 0.8]) ** 3
    assert s.z == pytest.approx(expect, rel=1e-15)
    assert s.mu.tolist() == [-2.0, -2.0, -2.0]  # stored as given, not round-tripped


def test_the_fugacity_route_is_beta_f_and_ignores_lambda():
    cfg = from_params(
        _typed(
            potential={"species_file": THREE},
            ensemble={"mu": None, "fugacity": {"A": 0.3, "B": 0.3, "C": 0.3}},
        )
    )
    s = cfg.species
    assert s is not None and s.route == "fugacity" and not s.lam_consulted
    assert s.z == pytest.approx(np.full(3, 0.3 / 1.2), rel=1e-15)
    # mu is derived, and inverts the mu route exactly: z = exp(beta mu)/lam^d
    assert np.exp((1.0 / 1.2) * s.mu) / s.lam**3 == pytest.approx(s.z, rel=1e-14)


def test_the_z_route_is_taken_verbatim():
    cfg = from_params(_typed(ensemble={"mu": None, "z": {"A": 0.25, "B": 0.125}}))
    s = cfg.species
    assert s is not None and s.route == "z" and not s.lam_consulted
    assert s.z.tolist() == [0.25, 0.125]


def test_two_routes_at_once_is_refused():
    _raises(
        _typed(ensemble={"mu": {"A": -2.0, "B": -2.0}, "z": {"A": 0.1, "B": 0.1}}),
        "mutually exclusive",
    )


def test_no_route_at_all_is_refused():
    p = _typed()
    p["ensemble"].pop("mu")
    _raises(p, "exactly one state-point route")


def test_a_missing_species_is_named():
    _raises(_typed(ensemble={"mu": {"A": -2.0}}), "no entry for ['B']")


def test_an_unknown_species_is_named_with_the_known_ones():
    p = _typed()
    p["ensemble"]["mu"] = {"A": -2.0, "B": -2.5, "Q": -1.0}
    _raises(p, "unknown species ['Q']")


def test_a_scalar_mu_in_typed_mode_says_what_it_should_have_been():
    _raises(_typed(ensemble={"mu": -2.0}), "one entry per species")


def test_an_activity_route_must_be_positive():
    _raises(_typed(ensemble={"mu": None, "z": {"A": 0.1, "B": 0.0}}), "<= 0 for ['B']")
    _raises(
        _typed(ensemble={"mu": None, "fugacity": {"A": -1.0, "B": 1.0}}),
        "<= 0 for ['A']",
    )


def test_a_map_of_lists_is_refused_with_the_sweep_shape_explained():
    p = _typed()
    p["ensemble"]["mu"] = {"A": [-2.0, -1.0], "B": -2.5}
    _raises(p, "list OF maps")


# ------------------------------------------------------- 4.2 forbidden keys


@pytest.mark.parametrize(
    "key,value",
    [("radius", 0.5), ("lower_cutoff", 0.9), ("upper_cutoff", 2.5)],
)
def test_the_species_file_owns_the_geometry_keys(key, value):
    _raises(_typed(run={key: value}), f"{key!r}")


def test_the_auxiliary_block_is_refused_in_typed_mode():
    p = _typed()
    p["auxiliary"] = {"type": "short_range", "cutoff": 1.6}
    _raises(p, "stated in the species file")


def test_typed_mode_takes_no_target_controller():
    _raises(_typed(ensemble={"target_eta": 0.4}), "never an input")


def test_typed_mode_takes_no_scalar_sweep():
    _raises(_typed(ensemble={"sweep": [-2.0, -1.0]}), "LIST OF MAPS")


def test_typed_mode_needs_lennard_jones():
    p = _typed()
    p["potential"] = {"main": "hard_core", "species_file": BINARY}
    _raises(p, "must be 'lennard_jones'")


def test_a_mof_force_field_is_not_a_species_file():
    _raises(
        _typed(potential={"species_file": str(FF / "forcefield_A.yaml")}),
        "not a species file",
    )


def test_an_angstrom_kelvin_species_file_is_refused_with_the_reason(tmp_path):
    """potential.T is already T*, so there is no kelvin temperature to reduce an
    Angstrom-Kelvin file with. Rather than invent a second temperature key, the
    bulk typed path takes reduced files only and says where to get one."""
    import yaml

    doc = yaml.safe_load(Path(BINARY).read_text())
    doc["units"] = "angstrom-kelvin"
    doc["reference"] = "A"
    path = tmp_path / "ak.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    _raises(_typed(potential={"species_file": str(path)}), "units: reduced")


# -------------------------------------------------- geometry from the model


def test_the_scalar_stand_ins_are_upper_bounds_over_species():
    """radius, lower_cutoff, upper_cutoff and volume_of_molecule survive in typed
    mode only as conservative stand-ins for the legacy geometric guards, so each
    must be the LARGEST value over species or pairs. A stand-in that is not an
    upper bound would let a guard pass for the small species and be violated by
    the large one."""
    cfg = from_params(
        _typed(
            potential={"species_file": THREE},
            ensemble={"mu": {"A": -2.0, "B": -2.0, "C": -2.0}},
        )
    )
    s = cfg.species
    assert s is not None
    assert cfg.radius == 0.5 * float(s.sigma_self.max()) == 0.5 * 1.15
    assert cfg.lower_cutoff == float(s.core_floor.max()) == pytest.approx(0.9 * 1.15)
    assert cfg.upper_cutoff == s.rc == 2.5
    assert cfg.volume_of_molecule == float(s.volume_of_molecule.max())


def test_the_packing_fraction_sums_over_species():
    cfg = from_params(
        _typed(
            potential={"species_file": THREE},
            ensemble={"mu": {"A": -2.0, "B": -2.0, "C": -2.0}},
        )
    )
    s = cfg.species
    assert s is not None
    counts = [7, 3, 11]
    expect = sum(n * v for n, v in zip(counts, s.volume_of_molecule))
    assert cfg.packing_fraction(counts) == pytest.approx(
        expect / cfg.total_volume, rel=1e-15
    )
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        cfg.packing_fraction([1, 2])


def test_the_single_species_packing_fraction_is_the_old_arithmetic():
    cfg = from_params(_untyped())
    assert cfg.packing_fraction(40) == 40 * cfg.volume_of_molecule / cfg.total_volume


def test_the_pair_table_reaches_the_config():
    cfg = from_params(_typed())
    s = cfg.species
    assert s is not None
    assert s.table.types == ("A", "B")
    assert s.table.eps[0, 0] == 1.0 and s.table.eps[1, 1] == 0.8
    assert s.list_cutoff == max(s.rc, s.max_core_floor)


def test_the_aux_cutoff_travels_from_the_species_file():
    cfg = from_params(
        _typed(
            potential={"species_file": THREE},
            ensemble={"mu": {"A": -2.0, "B": -2.0, "C": -2.0}},
        )
    )
    s = cfg.species
    assert s is not None and s.aux_cutoff is not None
    assert s.max_aux_cutoff == pytest.approx(1.6 * 1.15)
    # with the auxiliary on it is the binding margin for the sweep, being
    # strictly shorter than rc by construction (M12)
    assert s.sweep_reach == s.max_aux_cutoff < s.rc
    assert from_params(_typed()).species.sweep_reach == 2.5  # aux: none


# ---------------------------------------------- M11 / M10 in the fingerprint


def _id_of(**over) -> str:
    return run_id(from_params(_typed(**over)))


def test_the_fingerprint_moves_when_the_route_moves():
    """The same state point stated two ways is two experiments as far as the
    directory layout is concerned, because the recorded mu means different
    things: only the mu route consulted lambda."""
    beta = 1.0 / 1.2
    z = {s: math.exp(beta * m) for s, m in (("A", -2.0), ("B", -2.5))}
    assert _id_of() != _id_of(ensemble={"mu": None, "z": z})


def test_the_fingerprint_moves_when_lambda_moves(tmp_path):
    import yaml

    doc = yaml.safe_load(Path(BINARY).read_text())
    doc["species"]["B"]["lambda"] = 1.3
    path = tmp_path / "lam.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    assert _id_of() != _id_of(potential={"species_file": str(path)})


def test_the_fingerprint_moves_when_k_ij_moves(tmp_path):
    import yaml

    doc = yaml.safe_load(Path(BINARY).read_text())
    doc["k_ij"]["A"]["B"] = 0.3
    path = tmp_path / "k.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    assert _id_of() != _id_of(potential={"species_file": str(path)})


def test_the_fingerprint_moves_when_the_state_point_moves():
    assert _id_of() != _id_of(ensemble={"mu": {"A": -2.0, "B": -2.4}})


def test_the_fingerprint_ignores_operational_knobs():
    """Two runs differing only in how long they run and how often they write are
    the same experiment and must share a directory, or resume stops working."""
    # `windows` is not a run key; the schema calls the chain-length knobs
    # `sampling_window`, `burn_in` and `target_cycles`. Gate P2 in the plan names
    # "windows", which is the plan being loose, not the schema being wrong.
    assert _id_of() == _id_of(
        run={
            "sampling_window": 500,
            "burn_in": 200,
            "target_cycles": 5e6,
            "checkpoint_every": 7,
            "traj_every": 3,
            "log_flush_interval": 11,
        }
    )


def test_a_comment_key_does_not_change_the_identity():
    p = _typed()
    q = copy.deepcopy(p)
    q["_comment"] = "the demixing anchor, k_AB = 0"
    assert run_id(from_params(p)) == run_id(from_params(q))


# --------------------------------------------------------- the real guests


def test_the_xekr_species_file_builds_a_typed_config():
    """The fictional arm and the real arm go through one code path: only the
    numbers change."""
    cfg = from_params(
        _typed(
            potential={"species_file": XEKR, "T": 298.0 / 221.0},
            ensemble={"mu": {"Xe": -2.0, "Kr": -2.5}},
        )
    )
    s = cfg.species
    assert s is not None
    assert s.names == ("Xe", "Kr")
    assert s.symbol == ("Xe", "Kr")
    assert s.rc == pytest.approx(12.0 / 4.1, rel=1e-15)
    assert s.core_floor_c == 0.7155
    assert s.sigma_self[1] == pytest.approx(3.636 / 4.1, rel=1e-15)


def test_semigrand_is_accepted_and_still_needs_its_activity_ratio():
    """A mixture at fixed TOTAL N is the semi-grand ensemble, and it parses.

    It used to be neither supported nor refused: `build_species_spec` demands a
    state-point route for any typed run, so a typed NVT file died with "a typed
    run needs exactly one state-point route: ensemble.mu" -- and adding
    `ensemble.mu` produced "NVT takes no 'mu'". Two validators, each naming the
    other's refusal as the fix, with no configuration in between.

    It then spent a stage reachable but MISNAMED, as `"NVT"`, which is the more
    expensive of the two failures: three cluster campaigns were launched under a
    params file that read as canonical and sampled something else. It has its own
    name now.

    Both halves are asserted because the pair is the point. `mu` is MEANINGFUL
    here -- at fixed N the ratio of activities is what sets the composition, and
    only the ratio does any work -- so the route is required rather than ignored,
    and omitting it is under-specifying the run rather than choosing a default.
    """
    p = _typed(ensemble={"N": 100, "type": "semigrand"})
    cfg = from_params(p)
    assert cfg.ensemble_type == "semigrand" and cfg.n_species == 2

    q = _typed(ensemble={"N": 100, "type": "semigrand", "mu": None})
    _raises(q, "exactly one state-point route")

    # ...and the OTHER fixed-N ensemble refuses the same key, because there it
    # would be a knob that turns and does nothing. This assertion is the one that
    # would have caught the misnaming: the two ensembles must disagree about mu.
    c = _typed(ensemble={"N": 100, "type": "NVT"})
    _raises(c, "would do nothing")

    # a species file with more than one species makes NVT canonical, and a
    # canonical run reads no trial cloud, so the fixture's rosenbluth block is
    # refused before anything else about it is considered
    d = _typed(ensemble={"N": 100, "type": "NVT", "mu": None})
    _raises(d, "runs no trial cloud")

    # ...and once the sampler is the only legal one, the remaining demand is the
    # composition itself, which a canonical run takes from a file rather than a
    # draw
    e = _typed(ensemble={"N": 100, "type": "NVT", "mu": None})
    e["sampler"] = {"type": "metropolis"}
    _raises(e, "run.initial_config")

    # ...and the route error is still what an under-specified muVT run gets, so
    # the NVT path has not swallowed the older one.
    r = _typed()
    r["ensemble"].pop("mu")
    _raises(r, "exactly one state-point route")


@pytest.mark.parametrize(
    ("sampler", "foreign"),
    [
        ("metropolis", "channel_length"),
        ("metropolis", "channel_width"),
        ("rosenbluth", "channel_length"),
        ("rosenbluth", "channel_width"),
        ("branch", "channel_length"),
    ],
)
def test_a_sampler_refuses_the_keys_it_cannot_read(sampler, foreign):
    """Only the chosen sampler's keys are legal in the sampler block.

    These cells were ACCEPTED AND IGNORED. That is worse than it sounds:
    `channel_length` and `channel_width` are in `_IDENTITY_FIELDS`, so a
    rosenbluth run stating one got a different run directory for the identical
    chain -- measured, four ids for one trajectory. A sweep over a key the
    sampler cannot read yields N directories of one chain and nothing says so.

    The union-of-all-keys check that allowed this is gone; each sampler names
    what it reads in `_SAMPLER_KEYS`, once.
    """
    # the group size travels in the `engine` block: left in the sampler it
    # would trip the moved-key refusal and this test would assert that guard
    # instead of the per-sampler one it is about
    base, engine = {
        "metropolis": ({}, {}),
        "rosenbluth": ({"trials": 20, "trial_box_side": 3.0}, {}),
        "branch": ({"trials": 20, "trial_box_side": 3.0}, {"depth": 1}),
    }[sampler]
    smp = {"type": sampler, **base, foreign: 5.0}
    p = {
        "potential": {"main": "hard_core"},
        "sampler": smp,
        "engine": engine,
        "ensemble": {"type": "muVT", "dimension": 3, "box": 20.0, "mu": 1.0},
        "run": {"kind": "single", "radius": 0.5, "seed": 1},
    }
    _raises(p, "does not read")

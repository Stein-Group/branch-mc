"""Gate P1 of multispecies_plan.md: the species file, its cross terms and its
validation.

Unit only, no chain, every test sub-second. The rules under test are M1 (cross
terms), M3 (units), M9 (one file holds everything species-indexed), M11 (per
species lambda), M12 (the auxiliary multiplier) and the required checks of
section 4.1, which M15 puts in the loader because a malformed model file is a
property of the file.

The two shipped MOF force fields are schema 1 and are untouched by any of this;
test_mof_units.py and test_pair_table.py pin that, unmodified.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from gcmc.forcefield.lennard_jones import lj_pair_energy_sq
from gcmc.forcefield.pair_table import PairTable, table_pair_energy_sq
from gcmc.forcefield.species_file import load_forcefield, reduce_tables

FF = Path(__file__).resolve().parents[2] / "ff"
ONE = FF / "species_one_reduced.yaml"
THREE = FF / "species_three_reduced.yaml"
BINARY = FF / "species_binary_reduced.yaml"
XEKR = FF / "species_xekr_reduced.yaml"


def _base() -> dict:
    """A well-formed three-species reduced doc, mutated per test."""
    return {
        "schema": 2,
        "name": "base",
        "units": "reduced",
        "mixing": "lorentz-berthelot-kij",
        "interaction": {
            "form": "lennard-jones-12-6",
            "rc": 2.5,
            "shifted": False,
            "tail_correction": False,
        },
        "core_floor": {"c": 0.9},
        "aux": "none",
        "species": {
            "status": "confirmed",
            "A": {"sigma": 1.0, "epsilon": 1.0, "lambda": 1.0, "symbol": "Ne"},
            "B": {"sigma": 1.15, "epsilon": 0.8, "lambda": 1.25, "symbol": "Ar"},
            "C": {"sigma": 0.9, "epsilon": 1.3, "lambda": 0.8, "symbol": "Kr"},
        },
        "k_ij": {"A": {"B": 0.25}, "B": {"C": -0.15}},
    }


def _write(tmp_path: Path, doc: dict, stem: str = "s") -> Path:
    p = tmp_path / f"{stem}.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


def _table(path: Path, T: float = 1.2) -> PairTable:
    ff = load_forcefield(path)
    red = reduce_tables(ff, T)
    assert red.core_floor_c is not None
    return PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, red.core_floor_c
    )


# --------------------------------------------------------------- M1 cross terms


def test_k_zero_is_lorentz_berthelot_element_for_element(tmp_path):
    """At k = 0 the table must be the plain Berthelot outer product it has always
    been, off-diagonal included, so every existing force field keeps its meaning
    and RASPA2 still accepts the same statement."""
    doc = _base()
    doc.pop("k_ij")
    doc["mixing"] = "lorentz-berthelot"
    ff = load_forcefield(_write(tmp_path, doc))
    eps = ff.pair_epsilon()
    lb = np.sqrt(ff.epsilon[:, None] * ff.epsilon[None, :])
    assert eps.tobytes() == lb.tobytes()
    # and stating the kij mixing with an all-zero block changes nothing
    zero = _base()
    zero["k_ij"] = {"A": {"B": 0.0}, "B": {"C": 0.0}}
    assert load_forcefield(_write(tmp_path, zero, "z")).pair_epsilon().tobytes() == (
        eps.tobytes()
    )


def test_diagonal_is_the_stored_epsilon_bit_for_bit_at_any_k(tmp_path):
    """M1: the diagonal is copied, never computed. sqrt(e*e) == e holds for every
    double in the normal range, but the single-species reduction gate must not
    rest on a rounding theorem, so the copy is asserted directly."""
    for k in (0.0, 0.25, -0.9, 0.999):
        doc = _base()
        doc["k_ij"] = {"A": {"B": k, "C": k}, "B": {"C": k}}
        ff = load_forcefield(_write(tmp_path, doc, f"k{k}"))
        eps = ff.pair_epsilon()
        assert np.diag(eps).tobytes() == ff.epsilon.tobytes()


def test_hand_computed_cross_term(tmp_path):
    """eps_AB = (1 - 0.25) sqrt(1.0 * 0.8) = 0.75 * 0.894427190999916 =
    0.670820393249937, and eps_BC = (1 + 0.15) sqrt(0.8 * 1.3)
    = 1.15 * 1.019803902718557 = 1.172774488126341. Sigma stays Lorentz:
    sigma_AB = (1.0 + 1.15)/2 = 1.075."""
    ff = load_forcefield(THREE)
    a, b, c = (ff.index(t) for t in ("A", "B", "C"))
    eps, sig = ff.pair_epsilon(), ff.pair_sigma()
    assert eps[a, b] == pytest.approx(0.670820393249937, rel=1e-15)
    assert eps[b, c] == pytest.approx(1.172774488126341, rel=1e-15)
    assert eps[a, c] == pytest.approx(np.sqrt(1.0 * 1.3), rel=1e-15)  # k absent = 0
    assert sig[a, b] == pytest.approx(1.075, rel=1e-15)


def test_three_species_table_shape(tmp_path):
    """Two stated k values, one pair left at Lorentz-Berthelot: symmetric, the
    diagonal is the input, and the three off-diagonals are all different."""
    ff = load_forcefield(THREE)
    eps = ff.pair_epsilon()
    assert eps.shape == (3, 3)
    assert eps.tobytes() == eps.T.tobytes()
    assert np.diag(eps).tobytes() == ff.epsilon.tobytes()
    off = [eps[0, 1], eps[0, 2], eps[1, 2]]
    assert len(set(off)) == 3
    assert ff.k_ij.tobytes() == ff.k_ij.T.tobytes()
    assert not np.diag(ff.k_ij).any()


# ------------------------------------------------------- M3 units and reduction


def test_reduced_file_is_a_passthrough():
    """A reduced file is already in engine units: nothing is divided, no type is
    pinned to sigma* = 1, and T is read as T* rather than as kelvin."""
    ff = load_forcefield(THREE)
    red = reduce_tables(ff, 1.2)
    assert (red.sigma_ref, red.eps_ref) == (1.0, 1.0)
    assert red.sigma_star.tobytes() == ff.pair_sigma().tobytes()
    assert red.eps_star.tobytes() == ff.pair_epsilon().tobytes()
    assert red.T_star == 1.2 and red.rc_star == ff.rc
    assert not np.isclose(np.diag(red.sigma_star), 1.0).all()  # nothing pinned
    with pytest.raises(ValueError, match="reduced file"):
        reduce_tables(ff, 1.2, reference="A")
    with pytest.raises(ValueError, match="reference scale"):
        red.z_star(1.0e5)


def test_one_species_reduced_is_the_frozen_lj_bit_for_bit():
    """The reduction anchor of M3: one species at sigma = eps = 1 must make
    table_pair_energy_sq the locked lj_pair_energy_sq, same float, same bits."""
    tab = _table(ONE)
    assert tab.types == ("A",)
    d2 = np.array([0.81, 1.0, 1.2345, 2.0, 4.0, 6.25])
    assert (
        table_pair_energy_sq(d2, 0, 0, tab).tobytes() == lj_pair_energy_sq(d2).tobytes()
    )
    assert tab.core_floor[0, 0] == pytest.approx(0.9, rel=1e-15)


def test_angstrom_kelvin_still_reduces_and_schema_one_needs_no_reference_key():
    """The MOF files are schema 1 and keep taking their reference from the
    caller; a schema-2 Angstrom-Kelvin file must name its own."""
    ff = load_forcefield(FF / "forcefield_A.yaml")
    assert ff.schema == 1 and ff.units == "angstrom-kelvin" and ff.reference is None
    red = reduce_tables(ff, 298.0)
    assert red.reference == "Xe"
    assert red.sigma_star[ff.index("Xe"), ff.index("Xe")] == 1.0


# ---------------------------------------------------- M11 lambda, M6 p_trial


def test_lambda_and_p_trial_are_read_per_species():
    ff = load_forcefield(THREE)
    assert ff.lam.tolist() == [1.0, 1.25, 0.8]
    assert ff.p_trial.tolist() == [0.4, 0.35, 0.25]


def test_p_trial_defaults_to_uniform_and_lambda_to_one():
    ff = load_forcefield(BINARY)
    assert ff.p_trial.tolist() == [0.5, 0.5]
    assert ff.lam.tolist() == [1.0, 1.0]


def test_the_binary_production_model_is_one_size_two_well_depths():
    """The first production mixture is the easiest one to reason about: every
    pair has the same sigma, so the core floor, the rod geometry and the
    auxiliary cutoff are identical for both species and any effect on
    composition or density has to come from the energetics."""
    ff = load_forcefield(BINARY)
    sig = ff.pair_sigma()
    assert (sig == 1.0).all()
    assert ff.epsilon.tolist() == [1.0, 0.8]


def test_xekr_reduced_matches_forcefield_a():
    """ff/species_xekr_reduced.yaml is FF-A divided through by Xe-Xe, so it must
    reproduce FF-A's guest block after reduction rather than approximate it.
    Re-derived here so the two files cannot drift apart silently."""
    anchor = reduce_tables(load_forcefield(FF / "forcefield_A.yaml"), 298.0)
    i, j = anchor.types.index("Xe"), anchor.types.index("Kr")
    red = reduce_tables(load_forcefield(XEKR), 298.0 / anchor.eps_ref)
    assert red.types == ("Xe", "Kr")
    for a, b in ((0, 0), (0, 1), (1, 1)):
        src = (i, j)[a], (i, j)[b]
        assert red.sigma_star[a, b] == anchor.sigma_star[src]
        assert red.eps_star[a, b] == anchor.eps_star[src]
    assert red.rc_star == anchor.rc_star
    assert red.T_star == anchor.T_star
    assert red.core_floor_c == anchor.core_floor_c


def test_framework_types_carry_no_lambda_and_no_trial_weight():
    """p_trial is a property of the trial cloud, which never proposes a framework
    atom, and lambda is meaningless for something that cannot move."""
    ff = load_forcefield(FF / "forcefield_A.yaml")
    assert ff.p_trial[ff.n_guests :].tolist() == [0.0] * len(ff.framework_types)
    assert ff.lam.tolist() == [1.0] * len(ff.types)
    assert ff.p_trial[: ff.n_guests].sum() == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------- M12 auxiliary


def test_aux_cutoff_is_per_pair_and_shorter_for_smaller_pairs():
    """One multiplier, so the proposal radius tracks the pair it proposes into."""
    ff = load_forcefield(THREE)
    aux = ff.aux_cutoff()
    assert aux is not None
    assert aux.tobytes() == (1.6 * ff.pair_sigma()).tobytes()
    assert aux.max() < ff.rc
    assert aux[ff.index("C"), ff.index("C")] < aux[ff.index("B"), ff.index("B")]
    assert load_forcefield(BINARY).aux_cutoff() is None  # aux: none


def test_aux_reaching_rc_raises_and_names_the_binding_pair(tmp_path):
    """The inequality is strict. B-B is the widest pair at sigma_ij = 1.15, so
    a = 2.5/1.15 = 2.1739... is the first value that is refused."""
    doc = _base()
    doc["aux"] = {"multiplier": 2.5 / 1.15}
    with pytest.raises(ValueError, match="B-B"):
        load_forcefield(_write(tmp_path, doc))
    doc["aux"] = {"multiplier": 2.5 / 1.15 - 1e-9}  # just inside
    assert load_forcefield(_write(tmp_path, doc, "ok")).aux_multiplier is not None


def test_aux_inside_the_first_shell_warns_but_loads(tmp_path):
    doc = _base()
    doc["aux"] = {"multiplier": 1.2}
    with pytest.warns(UserWarning, match="first shell|1.4"):
        ff = load_forcefield(_write(tmp_path, doc))
    assert ff.aux_multiplier == 1.2


def test_the_control_arm_runs_the_machinery_at_rc_and_takes_no_multiplier(tmp_path):
    """`control: true` is the auxiliary ON at rc, not the auxiliary off.

    This assertion is inverted from what it used to say, and the inversion is the
    point. It read

        assert ff.aux_control and ff.aux_multiplier is None
        assert ff.aux_cutoff() is None

    which pinned `aux: {control: true}` to exactly the same value as `aux: none`:
    a run with no auxiliary machinery in it. The block is documented as "the
    exact no-op arm the tuning sweep uses as its own reference", and a reference
    arm that is byte-for-byte the arm it is compared against cannot fail. The old
    test locked that in and read as coverage.

    The cutoff is `rc` for every pair, so the correction executes and evaluates to
    exactly zero -- see `ForceField.aux_cutoff` for why the zero is exact and not
    merely small.
    """
    doc = _base()
    doc["aux"] = {"control": True}
    ff = load_forcefield(_write(tmp_path, doc))
    assert ff.aux_control and ff.aux_multiplier is None
    aux = ff.aux_cutoff()
    assert aux is not None, "the control arm must configure a radius, not None"
    assert aux.shape == ff.pair_sigma().shape
    assert (aux == ff.rc).all(), (
        "every pair's control radius must be rc exactly: the correction is a "
        "difference of two sums over the same pair list, and only equality makes "
        "it identically zero rather than merely tiny"
    )
    doc["aux"] = {"control": True, "multiplier": 1.6}
    with pytest.raises(ValueError, match="no-op arm"):
        load_forcefield(_write(tmp_path, doc, "bad"))


# ------------------------------------------------- section 4.1, every check


def _mutate(doc: dict, path: tuple, value) -> dict:
    d = copy.deepcopy(doc)
    node = d
    for key in path[:-1]:
        node = node[key]
    if value is _DELETE:
        node.pop(path[-1], None)
    else:
        node[path[-1]] = value
    return d


class _Delete:
    pass


_DELETE = _Delete()

# (label, path, value, expected substring). The substring is what the message
# must name: a check that cannot point at the offending species, pair or key is
# not finished (M15).
_MALFORMED = [
    ("sigma <= 0", ("species", "B", "sigma"), 0.0, "'B'"),
    ("sigma non-finite", ("species", "B", "sigma"), float("inf"), "finite"),
    ("epsilon <= 0", ("species", "C", "epsilon"), -1.0, "'C'"),
    ("lambda <= 0", ("species", "B", "lambda"), 0.0, "lambda for 'B'"),
    ("p_trial partial", ("species", "A", "p_trial"), 0.5, "'B'"),
    ("unknown species key", ("species", "A", "mu"), 1.0, "mu"),
    ("core_floor null", ("core_floor", "c"), _DELETE, "core_floor.c"),
    ("core_floor > 1", ("core_floor", "c"), 1.5, "(0, 1]"),
    ("core_floor 0", ("core_floor", "c"), 0.0, "(0, 1]"),
    ("rc <= 0", ("interaction", "rc"), 0.0, "rc must be positive"),
    ("units missing", ("units",), _DELETE, "must declare `units`"),
    ("units unknown", ("units",), "sigma-epsilon", "'sigma-epsilon'"),
    ("reference on reduced", ("reference",), "A", "reduced file carries no"),
    ("bad mixing", ("mixing",), "kong", "'kong'"),
    ("k_ij unknown species", ("k_ij", "A", "Z"), 0.1, "'Z'"),
    ("k_ij diagonal", ("k_ij", "A", "A"), 0.1, "diagonal"),
    ("k_ij >= 1", ("k_ij", "A", "B"), 1.0, "must be < 1"),
    ("symbol not an element", ("species", "A", "symbol"), "Xx", "'Xx'"),
    ("duplicate symbol", ("species", "A", "symbol"), "Ar", "'Ar'"),
    ("empty species block", ("species",), {"status": "confirmed"}, "empty"),
    ("no species block", ("species",), _DELETE, "no `species:`"),
]


@pytest.mark.parametrize(
    "label,path,value,message",
    _MALFORMED,
    ids=[m[0] for m in _MALFORMED],
)
def test_malformed_species_file_raises_and_names_the_offender(
    tmp_path, label, path, value, message
):
    doc = _mutate(_base(), path, value)
    with pytest.raises(ValueError) as exc:
        load_forcefield(_write(tmp_path, doc))
    assert message in str(exc.value), f"{label}: message was {exc.value}"


def test_p_trial_must_sum_to_one(tmp_path):
    doc = _base()
    for name, v in zip("ABC", (0.5, 0.3, 0.3)):
        doc["species"][name]["p_trial"] = v
    with pytest.raises(ValueError, match="sum to 1"):
        load_forcefield(_write(tmp_path, doc))


def test_p_trial_must_be_positive(tmp_path):
    doc = _base()
    for name, v in zip("ABC", (0.6, 0.4, 0.0)):
        doc["species"][name]["p_trial"] = v
    with pytest.raises(ValueError, match="p_trial for 'C'"):
        load_forcefield(_write(tmp_path, doc))


def test_a_k_ij_block_needs_the_kij_mixing_rule(tmp_path):
    doc = _base()
    doc["mixing"] = "lorentz-berthelot"
    with pytest.raises(ValueError, match="lorentz-berthelot-kij"):
        load_forcefield(_write(tmp_path, doc))


def test_a_pair_stated_in_both_orders_must_agree(tmp_path):
    doc = _base()
    doc["k_ij"]["B"]["A"] = 0.30  # A->B says 0.25
    with pytest.raises(ValueError, match="twice in both orders"):
        load_forcefield(_write(tmp_path, doc))
    doc["k_ij"]["B"]["A"] = 0.25  # agreeing restatement is fine
    ff = load_forcefield(_write(tmp_path, doc, "ok"))
    assert ff.k_ij[ff.index("A"), ff.index("B")] == 0.25


def test_species_and_guests_are_exclusive(tmp_path):
    doc = _base()
    doc["guests"] = {"status": "confirmed", "Xe": {"sigma": 4.1, "epsilon": 221.0}}
    with pytest.raises(ValueError, match="not both"):
        load_forcefield(_write(tmp_path, doc))


def test_schema_two_angstrom_kelvin_needs_a_reference(tmp_path):
    doc = _base()
    doc["units"] = "angstrom-kelvin"
    with pytest.raises(ValueError, match="needs a `reference`"):
        load_forcefield(_write(tmp_path, doc))
    doc["reference"] = "B"
    red = reduce_tables(load_forcefield(_write(tmp_path, doc, "ok")), 300.0)
    assert red.reference == "B"
    assert red.sigma_star[1, 1] == 1.0 and red.eps_star[1, 1] == 1.0


def test_a_species_name_may_clash_with_nothing_in_the_framework(tmp_path):
    """Species names must be disjoint from framework names; the guest/framework
    clash check now covers the species block too."""
    doc = _base()
    doc["framework"] = {
        "status": "confirmed",
        "A": {"sigma": 3.0, "epsilon": 50.0, "symbol": "O"},
    }
    with pytest.raises(ValueError, match="both blocks"):
        load_forcefield(_write(tmp_path, doc))


def test_the_provisional_gate_still_bites_on_a_species_block(tmp_path):
    doc = _base()
    doc["species"]["status"] = "provisional"
    with pytest.raises(ValueError, match="provisional"):
        load_forcefield(_write(tmp_path, doc))
    assert load_forcefield(_write(tmp_path, doc), allow_provisional=True).status == {
        "species": "provisional"
    }

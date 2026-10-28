"""The truncated-and-shifted LJ convention (FF-A-shift), end to end.

Kancharlapalli 2019 ran RASPA with a shifted potential: with u(r) - u(rc) inside
rc our Widom Henry constants land on their Table S4, without it they sit ~60 %
high. The shift lives in one place, `pair_table`, so these tests check that:

  1. switching it OFF changes nothing (the unshifted functions are the old ones);
  2. switching it ON gives u(r) - u(rc) strictly inside rc and exactly 0 at and
     beyond rc, including on the inf pads callers use to fill neighbour rows;
  3. the reduced engine core (FrameworkField + LennardJonesTable, i.e. what the
     MOF loop's acceptance reads) equals the angstrom/kelvin oracle
     ref_energy.py on a mixed Xe/Kr configuration in the real MFM-300(In)
     framework, to 1e-10;
  4. the loader and the two RASPA writers carry the flag, and refuse it where it
     is not supported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gcmc.engine.rigid_framework import mof_energy
from gcmc.forcefield.lj_table import FrameworkField, LennardJonesTable
from gcmc.forcefield.pair_table import (
    PairTable,
    table_pair_energy,
    table_pair_energy_sq,
    table_pair_energy_vec,
)
from gcmc.forcefield.species_file import load_forcefield, reduce_tables
from gcmc.run.io_framework import load_framework_typed
from scripts.mof.ref_energy import read_extxyz, total_energy

ROOT = Path(__file__).resolve().parents[2]
FF_A = ROOT / "ff" / "forcefield_A.yaml"
FF_S = ROOT / "ff" / "forcefield_A_shift.yaml"


def _tables(ff_path: Path) -> tuple[object, object, PairTable]:
    ff = load_forcefield(ff_path)
    red = reduce_tables(ff, 298.0)
    table = PairTable.from_reduced(
        red.types,
        red.n_guests,
        red.sigma_star,
        red.eps_star,
        ff.core_floor_c,
        rc=red.rc_star,
        shifted=red.shifted,
    )
    return ff, red, table


def test_the_two_yamls_differ_only_in_the_shift():
    a, b = load_forcefield(FF_A), load_forcefield(FF_S)
    assert not a.shifted and b.shifted
    assert a.types == b.types
    np.testing.assert_array_equal(a.sigma, b.sigma)
    np.testing.assert_array_equal(a.epsilon, b.epsilon)
    assert a.rc == b.rc
    # FF-A-shift pins the floor for all seven metals (Fe binds), never above FF-A
    assert b.core_floor_c is not None and b.core_floor_c <= a.core_floor_c


def test_unshifted_table_is_the_old_table():
    """from_reduced without the new keywords, and with shifted=False, build the
    same table and the same energies, bit for bit."""
    _, red, _ = _tables(FF_A)
    old = PairTable.from_reduced(
        red.types, red.n_guests, red.sigma_star, red.eps_star, 0.7155
    )
    new = PairTable.from_reduced(
        red.types,
        red.n_guests,
        red.sigma_star,
        red.eps_star,
        0.7155,
        rc=red.rc_star,
        shifted=False,
    )
    assert not old.shifted and not new.shifted and new.shift is None
    d = np.linspace(0.75, 4.0, 301)
    for ti, tj in ((0, 0), (0, 1), (1, 3)):
        a = table_pair_energy(d, ti, tj, old)
        b = table_pair_energy(d, ti, tj, new)
        assert np.array_equal(a, b)
        assert np.array_equal(
            table_pair_energy_sq(d * d, ti, tj, old),
            table_pair_energy_sq(d * d, ti, tj, new),
        )


def test_shifted_energy_is_u_minus_u_rc_inside_and_zero_outside():
    _, red, plain = _tables(FF_A)
    _, _, shifted = _tables(FF_S)
    rc = red.rc_star
    assert shifted.shifted and shifted.rc2 == pytest.approx(rc * rc)
    d = np.array([0.8, 1.0, 1.5, 2.5, rc * (1 - 1e-12), rc, rc * 1.01, 5.0, np.inf])
    for ti, tj in ((0, 0), (0, 1), (1, 1), (0, 3), (1, 5)):
        u_rc = float(table_pair_energy(np.array([rc]), ti, tj, plain)[0])
        assert u_rc < 0.0  # attractive tail at 12 A: the shift raises energies
        got = table_pair_energy(d, ti, tj, shifted)
        want = np.where(d < rc, table_pair_energy(d, ti, tj, plain) - u_rc, 0.0)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-14)
        assert got[-1] == 0.0 and got[-2] == 0.0 and got[-4] == 0.0  # inf, >rc, =rc
        assert abs(got[-5]) < 1e-9  # continuous at rc
        np.testing.assert_allclose(
            table_pair_energy_sq(d * d, ti, tj, shifted), got, rtol=1e-12, atol=1e-14
        )
        si = np.full(d.shape, ti)
        sj = np.full(d.shape, tj)
        np.testing.assert_allclose(
            table_pair_energy_vec(d, si, sj, shifted), got, rtol=1e-12, atol=1e-14
        )


def test_inf_pads_stay_neutral_in_the_potential_object():
    """Callers pad neighbour rows with inf. Shifted or not, a pad must weigh 1."""
    _, red, table = _tables(FF_S)
    pot = LennardJonesTable(
        beta=1.0, table=table, trial_type=0, upper_cutoff=red.rc_star
    )
    row = np.array([[1.2, np.inf, np.inf]])
    nt = np.array([[1, 0, 0]])
    w = pot.weight_cloud(row, np.array([0]), nt)
    u = float(table_pair_energy(np.array([1.2]), 0, 1, table)[0])
    assert w[0] == pytest.approx(np.exp(-u), rel=1e-12)
    g = pot.pair_u_grid(row, np.array([0]), nt)
    assert g[0, 1] == 0.0 and g[0, 2] == 0.0


@pytest.fixture(scope="module")
def mfm(tmp_path_factory):
    from gcmc.framework.structure import build, write_extxyz

    ff, red, table = _tables(FF_S)
    atoms, _ = build(ROOT / "structures" / "MFM-300-In.cif", ff)
    ex = tmp_path_factory.mktemp("shift") / "mfm.extxyz"
    write_extxyz(atoms, ex)
    frame_A, frame_t_ref, cell_A = read_extxyz(ex, ff)
    pos_star, tids = load_framework_typed(ex, table)
    sr = red.sigma_ref
    field = FrameworkField(
        pos_star / sr, tids, table, np.asarray(cell_A) / sr, red.rc_star
    )
    return ff, red, table, field, frame_A, frame_t_ref, np.asarray(cell_A)


def test_engine_core_matches_the_shifted_oracle_on_a_mixed_configuration(mfm):
    """The gate: a dozen Xe/Kr guests at admissible random sites; the engine's
    decomposition (what acceptance reads) times eps_ref equals ref_energy.py with
    the shifted force field, to 1e-10 relative, term by term in total."""
    ff, red, table, field, frame_A, frame_t, cell_A = mfm
    rng = np.random.default_rng(20261027)
    xe, kr = table.index("Xe"), table.index("Kr")
    pos, spc = [], []
    while len(pos) < 12:
        p = rng.uniform(0, 1, 3) * field.box
        s = xe if len(pos) % 3 else kr
        if field.overlaps(p, s) or field.energy(p, s) > 0:
            continue
        if any(
            np.linalg.norm((p - q) - field.box * np.round((p - q) / field.box))
            < table.core_floor[s, t]
            for q, t in zip(pos, spc)
        ):
            continue
        pos.append(p)
        spc.append(s)
    P, S = np.array(pos), np.array(spc)
    u_host, u_guest = mof_energy(
        P, S, field=field, table=table, beta=1.0, upper_cutoff=field.rc
    )
    ref = total_energy(P * red.sigma_ref, S, frame_A, frame_t, cell_A, ff)
    assert (u_host + u_guest) * red.eps_ref == pytest.approx(ref["total"], rel=1e-10)
    assert u_guest * red.eps_ref == pytest.approx(
        ref["guest_guest"], rel=1e-10, abs=1e-9
    )

    # and the shift is not a no-op on this configuration: the unshifted oracle
    # disagrees by far more than the tolerance
    ff_plain = load_forcefield(FF_A)
    ref_plain = total_energy(P * red.sigma_ref, S, frame_A, frame_t, cell_A, ff_plain)
    assert abs(ref_plain["total"] - ref["total"]) > 1e-3 * abs(ref["total"])


def test_batched_field_energy_agrees_with_pointwise_when_shifted(mfm):
    _, _, _, field, *_ = mfm
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, (64, 3)) * field.box
    many = field.energy_many(pts, 0)
    one = np.array([field.energy(p, 0) for p in pts])
    np.testing.assert_allclose(many, one, rtol=1e-12, atol=1e-12)


def test_loader_refuses_shift_on_species_files_and_non_bool(tmp_path):
    src = (ROOT / "ff" / "species_binary_reduced.yaml").read_text()
    assert "shifted: false" in src
    bad = tmp_path / "s.yaml"
    bad.write_text(src.replace("shifted: false", "shifted: true"))
    with pytest.raises(ValueError, match="schema-1 MOF force fields only"):
        load_forcefield(bad)
    weird = tmp_path / "w.yaml"
    weird.write_text(FF_S.read_text().replace("shifted: true ", "shifted: 1 "))
    with pytest.raises(ValueError, match="true or false"):
        load_forcefield(weird)


def test_raspa_writers_carry_the_shift():
    from gcmc.framework.raspa2 import force_field_mixing_rules_def
    from scripts.mof.raspa_input import force_field_json

    present = ("Xe", "Kr", "In", "C", "O", "H")
    for path, word in ((FF_A, "truncated"), (FF_S, "shifted")):
        ff = load_forcefield(path)
        lines = force_field_mixing_rules_def(ff, present).splitlines()
        assert lines[1] == word
        assert force_field_json(ff, present)["TruncationMethod"] == word


def test_the_shift_file_floor_is_safe_for_every_mfm300_metal():
    """The pinned floor must not exceed what any of the seven metals needs at the
    1e-10 Boltzmann target (P0.6), at the temperature the file states."""
    from gcmc.forcefield.species_file import core_floor_scan

    ff = load_forcefield(FF_S)
    for metal in ("Al", "Ga", "In", "Sc", "V", "Cr", "Fe"):
        need = core_floor_scan(ff, 298.0, present=("Xe", "Kr", metal, "C", "O", "H"))
        assert ff.core_floor_c <= need["c"] + 1e-12, metal

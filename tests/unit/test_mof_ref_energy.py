"""P0.3/P0.5/P0.7 against hand calculations and a toy framework.

ref_energy.py is the oracle every later phase is checked against, so it gets
checked here against arithmetic rather than against the engine. The toy framework
is a square lattice of carbon walls around one channel along c: small enough to
reason about, tetragonal like the real target, and enough to exercise the P0.3
sizing rule and the P0.7 channel finder end to end without a CIF in the repo.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write

from gcmc.potentials.lennard_jones import lj_pair_energy
from scripts.mof.build_framework import (
    build,
    orthogonal_lengths,
    replication_for,
    write_extxyz,
)
from scripts.mof.channel_scan import axial_profile, channel_centers, scan_field
from scripts.mof.ref_energy import (
    check_cell,
    minimum_image,
    one_body_field,
    pair_energy,
    read_extxyz,
    total_energy,
)
from scripts.mof.units import lj_energy_kelvin, load_forcefield

FF_A = Path(__file__).resolve().parents[2] / "ff" / "forcefield_A.yaml"


@pytest.fixture(scope="module")
def ff():
    return load_forcefield(FF_A)


# --------------------------------------------------------------- pair energy


def test_lj_zero_at_sigma_and_minimum_at_2_to_the_sixth(ff):
    """The 12-6 shape, at the two places it is known by hand:
      u(sigma)          = 4 eps (1 - 1) = 0
      u(2^(1/6) sigma)  = 4 eps (1/4 - 1/2) = -eps
    For Xe-Xe that is u(4.1 A) = 0 and u(4.6020944 A) = -221.0 K.
    """
    sig, eps = 4.1, 221.0
    assert lj_energy_kelvin(np.array(sig), sig, eps) == pytest.approx(0.0, abs=1e-9)
    rmin = 2.0 ** (1.0 / 6.0) * sig
    assert rmin == pytest.approx(4.6020944, abs=1e-6)
    assert lj_energy_kelvin(np.array(rmin), sig, eps) == pytest.approx(-eps, rel=1e-12)


def test_truncation_is_a_hard_zero(ff):
    """Truncated and unshifted: a pair at exactly rc contributes zero, a pair a
    hair inside contributes its full unshifted energy with no offset.

    Xe-Xe just inside 12 A: u = 4*221.0*((4.1/12)^12 - (4.1/12)^6) = -1.40403 K,
    which is what the pair just inside the cutoff must carry, not zero, and not
    that value minus itself.
    """
    i = ff.index("Xe")
    sig, eps = ff.pair_sigma(), ff.pair_epsilon()
    assert pair_energy(12.0, i, i, sig, eps, ff.rc) == 0.0
    assert pair_energy(12.5, i, i, sig, eps, ff.rc) == 0.0
    inside = pair_energy(11.999999, i, i, sig, eps, ff.rc)
    x = (4.1 / 11.999999) ** 6
    assert inside == pytest.approx(4.0 * 221.0 * x * (x - 1.0), rel=1e-12)
    assert inside == pytest.approx(-1.40403, abs=1e-4)


def test_reference_pair_reduces_to_the_engine_shape():
    """The permanent reduction (A1.2's spirit, at script level): at sigma = eps = 1
    the oracle's pair energy IS the engine's `lj_pair_energy`, bit for bit. If this
    ever parts, one of the two grew an offset or a different algebra."""
    r = np.array([0.9, 1.0, 1.1234, 2.5, 3.0])
    mine = lj_energy_kelvin(r, 1.0, 1.0)
    theirs = lj_pair_energy(r)
    assert np.array_equal(mine, theirs)


def test_physical_pair_scales_to_the_reduced_pair(ff):
    """u(r)/eps_ref evaluated at r = r* sigma_ref is the reduced Xe-Xe pair. This
    is the statement that the reduced tables mean what P0.4 says they mean."""
    r_star = np.array([1.0, 1.2, 2.0, 3.0])
    u_star = lj_energy_kelvin(r_star * 4.1, 4.1, 221.0) / 221.0
    assert u_star == pytest.approx(lj_pair_energy(r_star), rel=1e-12)


# ------------------------------------------------------------ minimum image


def test_minimum_image_per_axis():
    """Anisotropic cell, one pair straddling each boundary. In a 30 x 30 x 24 box a
    separation of (29, 0, 23) is really (-1, 0, -1)."""
    cell = np.array([30.0, 30.0, 24.0])
    assert minimum_image(np.array([29.0, 0.0, 23.0]), cell) == pytest.approx(
        [-1.0, 0.0, -1.0]
    )
    assert minimum_image(np.array([-29.0, 16.0, 0.0]), cell) == pytest.approx(
        [1.0, -14.0, 0.0]
    )
    # exactly half a box is the ambiguous case; numpy's round-half-to-even fixes it
    assert abs(minimum_image(np.array([15.0, 0.0, 12.0]), cell))[0] == 15.0


def test_check_cell_refuses_a_box_the_cutoff_does_not_fit_in(ff):
    """rc <= min(L)/2 or minimum image is a lie. P0.3's replication rule exists to
    keep this satisfied; the oracle refuses rather than silently double-counting."""
    check_cell(np.array([24.0, 24.0, 24.0]), ff.rc)
    with pytest.raises(ValueError, match="exceeds half"):
        check_cell(np.array([24.0, 24.0, 23.0]), ff.rc)
    with pytest.raises(ValueError, match="orthogonal"):
        check_cell(np.array([[24.0, 0.0], [0.0, 24.0]]), ff.rc)


# --------------------------------------------------------------- total energy


def test_total_energy_two_guests_by_hand(ff):
    """Two Xe atoms 5 A apart in a 40 A box, no framework:
      x = (4.1/5)^6 = 0.30400667
      u = 4 * 221.0 * x * (x - 1) = -187.0426 K
    and nothing else contributes.
    """
    cell = np.array([40.0, 40.0, 40.0])
    pos = np.array([[1.0, 1.0, 1.0], [6.0, 1.0, 1.0]])
    typ = np.array([ff.index("Xe")] * 2)
    u = total_energy(pos, typ, np.empty((0, 3)), np.empty(0, int), cell, ff)
    x = (4.1 / 5.0) ** 6
    assert u["guest_guest"] == pytest.approx(4.0 * 221.0 * x * (x - 1.0), rel=1e-12)
    assert u["guest_guest"] == pytest.approx(-187.0426, abs=1e-3)
    assert u["guest_framework"] == 0.0
    assert u["total"] == u["guest_guest"]


def test_total_energy_uses_the_minimum_image(ff):
    """The same two atoms, now placed across the periodic seam. The energy must be
    identical: a guest at x = 39.5 in a 40 A box is 1 A from a guest at x = 0.5."""
    cell = np.array([40.0, 40.0, 40.0])
    typ = np.array([ff.index("Xe")] * 2)
    near = total_energy(
        np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        typ,
        np.empty((0, 3)),
        np.empty(0, int),
        cell,
        ff,
    )
    seam = total_energy(
        np.array([[38.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        typ,
        np.empty((0, 3)),
        np.empty(0, int),
        cell,
        ff,
    )
    assert seam["total"] == pytest.approx(near["total"], rel=1e-12)


def test_guest_framework_term_is_per_pair_typed(ff):
    """One Xe over one C at 4 A, mixed by Lorentz-Berthelot:
      sigma = 3.786495, eps = 102.8408
      x = (3.786495/4)^6 = 0.71955563
      u = 4 * 102.84075 * x * (x - 1) = -83.0111 K
    Swapping the framework atom's type to O must change the number, which is the
    whole point of the table: types, not one global sigma.
    """
    cell = np.array([40.0, 40.0, 40.0])
    guest = np.array([[0.0, 0.0, 0.0]])
    gt = np.array([ff.index("Xe")])
    frame = np.array([[4.0, 0.0, 0.0]])
    u_c = total_energy(guest, gt, frame, np.array([ff.index("C")]), cell, ff)
    sigma = 0.5 * (4.1 + 3.47299)
    eps = math.sqrt(221.0 * 47.8562)
    x = (sigma / 4.0) ** 6
    assert u_c["guest_framework"] == pytest.approx(4.0 * eps * x * (x - 1.0), rel=1e-12)
    assert u_c["guest_guest"] == 0.0
    u_o = total_energy(guest, gt, frame, np.array([ff.index("O")]), cell, ff)
    assert u_o["guest_framework"] != u_c["guest_framework"]


def test_one_body_field_is_the_guest_framework_term(ff):
    """P0.7 and A5.1 both lean on the field being exactly the guest-framework term
    of the total energy, with no guest-guest contamination. Vectorised or not, the
    two routes must agree."""
    rng = np.random.default_rng(4)
    cell = np.array([30.0, 30.0, 30.0])
    frame = rng.uniform(0, 30, size=(60, 3))
    ft = np.array([ff.index("C")] * 60)
    pts = rng.uniform(0, 30, size=(5, 3))
    field = one_body_field(pts, ff.index("Xe"), frame, ft, cell, ff)
    for p, f in zip(pts, field):
        u = total_energy(p[None, :], np.array([ff.index("Xe")]), frame, ft, cell, ff)
        assert u["guest_framework"] == pytest.approx(f, rel=1e-12)


# ------------------------------------------------------- P0.3 supercell rule


def test_replication_rule(ff):
    """Every edge must reach 2 rc = 24 A. A 15 A edge needs 2 (30 A); a 12 A edge
    needs 2 (24 A) exactly, not 3; a 25 A edge needs 1."""
    reps = replication_for(np.array([15.0, 12.0, 25.0]), 12.0)
    assert list(reps) == [2, 2, 1]
    assert list(replication_for(np.array([6.7]), 12.0)) == [4]  # 4*6.7 = 26.8 >= 24


def test_build_refuses_a_non_orthogonal_cell(tmp_path, ff):
    """The fence is a refusal, not a rotation: a hexagonal cell would put the
    channel axis where the sweep's axis-aligned cuboid cannot follow."""
    hexy = Atoms(
        "C",
        positions=[[0, 0, 0]],
        cell=[[5, 0, 0], [2.5, 4.33, 0], [0, 0, 5]],
        pbc=True,
    )
    cif = tmp_path / "hex.cif"
    ase_write(str(cif), hexy)
    with pytest.raises(ValueError, match="90 degrees"):
        build(cif, ff)


# ------------------------------------------ toy framework: P0.3 -> P0.5 -> P0.7


@pytest.fixture(scope="module")
def ff_toy(ff):
    """FF-A with a cutoff the toy cell can hold.

    The toy is 12 A on a side and the production rc is 12 A, so minimum image
    would be a lie there (check_cell says so, correctly). Replicating the toy out
    to 24 A instead would make the P0.7 grid scan minutes long for no extra
    coverage: what these tests exercise is the finder, not the cutoff. The real
    scan runs at the production rc on a P0.3 supercell.
    """
    return dataclasses.replace(ff, rc=5.0)


#: the columns left out of the toy lattice, i.e. the channel
_TOY_HOLE = ((1, 1), (1, 2), (2, 1), (2, 2))


def _toy_channel(spacing: float = 3.0, nz: int = 4) -> Atoms:
    """Carbon walls on a square lattice with one channel along c.

    A 4 x 4 x nz grid of carbons at `spacing` with the central 2 x 2 block of
    columns left out: a tetragonal cell with a single through-going channel
    centered at (4.5, 4.5) whose field repeats along c with period `spacing`.

    The hole has to be 2 x 2. A single missing column leaves a 3 A gap, and
    sigma_XeC is 3.70 A, so Xe sees a solid: the field never goes below +4000 K
    and there is correctly no channel to find. Which is itself the reason the
    real channel centers come from this scan and not from looking at the
    structure.
    """
    pos, n = [], 4
    for i in range(n):
        for j in range(n):
            if (i, j) in _TOY_HOLE:
                continue
            for k in range(nz):
                pos.append([i * spacing, j * spacing, k * spacing])
    return Atoms(
        f"C{len(pos)}",
        positions=pos,
        cell=[n * spacing, n * spacing, nz * spacing],
        pbc=True,
    )


def test_toy_framework_roundtrips_through_extxyz(tmp_path, ff):
    """P0.3 writes what P0.5 reads: positions, force-field type ids and the cell
    survive the trip, and the sidecar's replication actually reached 2 rc."""
    cif = tmp_path / "toy.cif"
    ase_write(str(cif), _toy_channel())
    atoms, meta = build(cif, ff)
    assert meta["atoms_per_type"] == {"C": len(atoms)}
    assert min(meta["supercell_angstrom"]) >= 2 * ff.rc - 1e-9
    assert meta["min_edge_over_2rc"] >= 1.0

    out = tmp_path / "toy.extxyz"
    write_extxyz(atoms, out)
    pos, typ, cell = read_extxyz(out, ff)
    assert len(pos) == len(atoms)
    assert set(typ) == {ff.index("C")}
    assert cell == pytest.approx(orthogonal_lengths(atoms))
    assert np.all(pos >= -1e-9) and np.all(pos <= cell + 1e-9)  # wrapped


def test_channel_scan_finds_the_channel_it_was_given(ff_toy):
    """P0.7 on the toy: the field, not the eye, has to put the channel center at
    (6, 6) and read the corrugation period as the 3 A lattice repeat along c.

    Grid at 0.5 A here to keep the test quick; the real scan runs finer.
    """
    atoms = _toy_channel()
    cell = orthogonal_lengths(atoms)
    frame = atoms.get_positions()
    ft = np.full(len(frame), ff_toy.index("C"))
    field, axes = scan_field(frame, ft, cell, ff_toy, "Xe", spacing=0.5)

    chans = channel_centers(field, axes, cell, threshold=0.0)
    assert len(chans) == 1
    assert chans[0]["center_xy"] == pytest.approx([4.5, 4.5], abs=0.5)
    assert chans[0]["min_field_kelvin"] < 0.0

    prof = axial_profile(field, axes, cell, (4.5, 4.5))
    assert prof["corrugation_period_angstrom"] == pytest.approx(3.0, abs=0.5)
    assert prof["corrugation_amplitude_kelvin"] > 0.0


def test_channel_scan_reports_no_channel_in_a_solid(ff_toy):
    """The negative control: fill the channel back in and the accessible region
    must vanish. A finder that always finds something is not a measurement."""
    atoms = _toy_channel()
    atoms += Atoms(
        "C16",
        positions=[[i * 3.0, j * 3.0, k * 3.0] for i, j in _TOY_HOLE for k in range(4)],
    )
    cell = orthogonal_lengths(atoms)
    ft = np.full(len(atoms), ff_toy.index("C"))
    field, axes = scan_field(atoms.get_positions(), ft, cell, ff_toy, "Xe", spacing=0.5)
    assert channel_centers(field, axes, cell, threshold=0.0) == []

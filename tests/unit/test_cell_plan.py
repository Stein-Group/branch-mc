"""P1: the parallel cell planner (gcmc/cell_plan.py).

Pure-function tests: no Config, no engine, no process pool, no simulation.
Whole module runs in well under a second.
"""

from __future__ import annotations

import math

import pytest

from gcmc.cell_plan import (
    CellPlan,
    factor_cells,
    plan_auto,
    plan_manual,
    usable_cell_counts,
)

CUBIC = (30.0, 30.0, 30.0)
MFM = (31.322, 31.322, 24.994)  # MFM-300(In) 2x2x2 supercell
SKEW = (40.0, 12.0, 7.0)


def _all_factorisations(n: int, dim: int) -> list[tuple[int, ...]]:
    """Independent brute force, written differently from the module's own
    recursion so the two can disagree."""
    out = []
    if dim == 2:
        for a in range(1, n + 1):
            if n % a == 0:
                out.append((a, n // a))
        return out
    for a in range(1, n + 1):
        if n % a:
            continue
        for b in range(1, n // a + 1):
            if (n // a) % b == 0:
                out.append((a, b, n // a // b))
    return out


def _aniso(n_ax, sides):
    e = [s / n for s, n in zip(sides, n_ax)]
    return max(e) / min(e)


# ---------------------------------------------------------------- factor_cells


@pytest.mark.parametrize("sides", [CUBIC, MFM, SKEW, (30.0, 30.0)])
def test_factorisation_product_is_the_requested_count(sides):
    for n in range(1, 65):
        n_ax = factor_cells(n, sides)
        assert len(n_ax) == len(sides)
        assert math.prod(n_ax) == n, f"n={n} in {sides} gave {n_ax}"


@pytest.mark.parametrize("sides", [CUBIC, MFM, SKEW])
def test_factorisation_is_the_most_cubic_split(sides):
    """D2: the chosen split achieves the minimum anisotropy over ALL splits."""
    for n in range(1, 49):
        chosen = factor_cells(n, sides)
        best = min(_aniso(c, sides) for c in _all_factorisations(n, len(sides)))
        assert _aniso(chosen, sides) == pytest.approx(best, rel=1e-12), (
            f"n={n} sides={sides}: chose {chosen} (aniso "
            f"{_aniso(chosen, sides):.4f}) but {best:.4f} was available"
        )


def test_factorisation_is_deterministic():
    """A resumed run must decompose exactly as the run it resumes."""
    first = [factor_cells(n, MFM) for n in range(1, 33)]
    for _ in range(100):
        assert [factor_cells(n, MFM) for n in range(1, 33)] == first


def test_anisotropic_box_gets_anisotropic_counts():
    """On MFM-300 the short c axis should take fewer cells than a and b."""
    n_ax = factor_cells(12, MFM)
    assert math.prod(n_ax) == 12
    assert n_ax[2] <= n_ax[0] and n_ax[2] <= n_ax[1], n_ax


# ------------------------------------------------------- the round-trip pin


# The measured float64 failures: (side, n, buffer). Recovering n from a
# derived cell_length by floor() loses a cell in each of these.
ROUND_TRIP_TRAPS = [
    (31.322, 4, 0.7155),  # MFM-300 a-axis with the FF-A core floor
    (31.322, 6, 0.7155),
    (24.994, 5, 0.7155),  # MFM-300 c-axis
    (200.0, 11, 2.0),
    (200.0, 22, 2.5),
    (30.0, 29, 1.0),
]


@pytest.mark.parametrize("side,n,buf", ROUND_TRIP_TRAPS)
def test_planner_is_exact_where_the_floor_reconstruction_is_not(side, n, buf):
    """THE PIN (plan section 0 / D7).

    Two assertions, and the second is the point: the planner reports exactly
    the requested count, AND the floor() reconstruction that a future
    "simplification" would reach for is shown to be wrong on this input. If
    someone later replaces the explicit counts with a derived cell_length, this
    test fails loudly instead of the node quietly losing cells.
    """
    sides = (side, side, side)
    plan = plan_auto([n, 1, 1], sides, buf)
    assert plan.n_ax == (n, 1, 1)
    assert plan.n_cells == n

    cl = plan.cell_length[0]
    reconstructed = int(side // (cl + buf))
    assert reconstructed != n, (
        f"this input no longer round-trips wrongly (side={side}, n={n}, "
        f"buffer={buf}); the pin has lost its teeth and needs a new trap"
    )


def test_pitch_minus_buffer_leaves_exactly_the_buffer_as_the_gap():
    plan = plan_auto(8, MFM, 2.5)
    for s, n, cl in zip(MFM, plan.n_ax, plan.cell_length):
        assert (s / n) - cl == pytest.approx(2.5, abs=1e-12)


# ------------------------------------------------------------------ auto mode


def test_auto_total_and_explicit_list_agree_when_they_describe_the_same_split():
    a = plan_auto(8, CUBIC, 2.0)
    b = plan_auto([2, 2, 2], CUBIC, 2.0)
    assert a == b


def test_auto_rejects_a_list_of_the_wrong_length():
    with pytest.raises(ValueError, match="one count per axis"):
        plan_auto([2, 2], CUBIC, 2.0)


@pytest.mark.parametrize("bad", [0, -3])
def test_auto_rejects_a_nonpositive_count(bad):
    with pytest.raises(ValueError, match=">= 1"):
        plan_auto(bad, CUBIC, 2.0)


def test_auto_refuses_a_count_that_leaves_no_room_and_names_both_numbers():
    """D1: too many cells is an error, never a silent reduction. At 15 cells per
    axis in a 30-box the pitch is exactly the buffer, so nothing is left."""
    with pytest.raises(ValueError) as exc:
        plan_auto(3375, CUBIC, 2.0)
    msg = str(exc.value)
    assert "n_cell=3375" in msg
    assert "buffer" in msg
    assert "Nearest usable cell counts" in msg


def test_the_suggested_counts_actually_work():
    """A hint that suggests an unusable count is worse than no hint."""
    with pytest.raises(ValueError) as exc:
        plan_auto(3375, CUBIC, 2.0)
    suggested = [
        int(t.strip().rstrip("."))
        for t in str(exc.value).split("Nearest usable cell counts:")[1].split(",")
        if t.strip().rstrip(".").isdigit()
    ]
    assert suggested, "no usable counts were offered"
    for n in suggested:
        plan_auto(n, CUBIC, 2.0)  # must not raise


def test_auto_refuses_a_cell_smaller_than_the_cavity():
    """min_cell is 2*cavity_radius: the cell-torus proposal needs the cavity to
    fit inside the cell. 64 cells in a 30-box gives 5.5, under a 6.0 floor."""
    with pytest.raises(ValueError) as exc:
        plan_auto(64, CUBIC, 2.0, min_cell=6.0, min_cell_desc="twice the cavity radius")
    assert "shortest cell edge" in str(exc.value)
    # the message must name the constraint the CALLER says is binding, not a
    # hard-coded guess: the floor can come from the cavity or from the reach
    assert "twice the cavity radius" in str(exc.value)
    assert "5.5" in str(exc.value) and "6" in str(exc.value)
    # one fewer cell per axis clears it
    plan_auto(27, CUBIC, 2.0, min_cell=6.0)


def test_the_floor_is_what_stops_absurdly_thin_cells():
    """Pure geometry allows a cell of edge 1.0 with a buffer of 2.0 (1000 cells
    in a 30-box): legal, but the cell is thinner than the interaction reach, so
    it is all border and no independent interior. That is why sim_setup passes
    a floor of max(2*cavity_radius, buffer) rather than relying on
    cell_length > 0 -- see D8. Without a floor this submission is accepted."""
    plan_auto(1000, CUBIC, 2.0)  # no floor: accepted, cell edge 1.0
    with pytest.raises(ValueError, match="shortest cell edge"):
        plan_auto(1000, CUBIC, 2.0, min_cell=2.0)


# ---------------------------------------------------------------- manual mode


def test_manual_reproduces_the_historical_count_formula():
    """MANUAL mode must keep giving what grid_up_nd has always given."""
    plan = plan_manual(8.0, 2.0, CUBIC)
    assert plan.n_ax == (3, 3, 3)
    assert plan.n_cells == 27
    assert plan.cell_length == (8.0, 8.0, 8.0)  # submitted value is kept


def test_manual_accepts_a_per_axis_cell_length():
    plan = plan_manual([10.5, 10.5, 8.0], 2.0, CUBIC)
    assert plan.n_ax == (2, 2, 3)


def test_manual_refuses_a_cell_that_does_not_fit():
    with pytest.raises(ValueError, match="does not fit"):
        plan_manual(30.0, 2.0, CUBIC)  # the historical default: 32 > 30


def test_manual_rejects_a_cell_length_list_of_the_wrong_length():
    with pytest.raises(ValueError, match="one length per axis"):
        plan_manual([8.0, 8.0], 2.0, CUBIC)


# ------------------------------------------------------------------- 2D boxes


def test_two_dimensional_planning():
    plan = plan_auto(16, (75.0, 75.0), 2.0)
    assert plan.n_ax == (4, 4)
    assert plan.n_cells == 16
    assert len(plan.cell_length) == 2


def test_min_cell_property_is_the_shortest_edge():
    plan = plan_auto([2, 2, 1], MFM, 1.0)
    assert plan.min_cell == min(plan.cell_length)
    assert isinstance(plan, CellPlan)


def test_usable_counts_are_returned_sorted_and_are_all_valid():
    got = usable_cell_counts(MFM, 2.5, min_cell=4.0, near=200, k=4)
    assert got == sorted(got)
    for n in got:
        plan_auto(n, MFM, 2.5, min_cell=4.0)

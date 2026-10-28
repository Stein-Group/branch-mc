"""C7: the RASPA2 stream parser, on text generated from RASPA2's own C formats.

The fixture is written with the format strings of src/statistics.c and
src/mc_moves.c (printf-equivalent % formatting), so the test pins the parser to
what RASPA2 prints, not to what one example output happened to look like.
"""

# printf-style % formatting is deliberate throughout: the fixture is written with
# RASPA2's own C format strings, so it must use the same formatting language.
# ruff: noqa: UP031

from __future__ import annotations

import json

import pytest

from gcmc.framework.raspa2_parse import parse_data_file, stitch, write_parsed
from gcmc.run.io import read_log

COMP_FW = (
    "Component %d (%s), current number of integer/fractional/reaction molecules: "
    "%d/%d/%d (avg. %9.5f), density: %9.5f (avg. %9.5f) [kg/m^3]\n"
)


def _block(cycle, total, comps, lam=None, u=(-1000.0, -900.0, -100.0), header=""):
    out = [f"{header}Current cycle: {cycle} out of {total}\n", "=" * 104 + "\n\n"]
    out.append("Loadings per component:\n" + "-" * 148 + "\n")
    for k, (name, n_int, n_frac) in enumerate(comps):
        out.append(COMP_FW % (k, name, n_int, n_frac, 0, 1.0, 2.0, 3.0))
        if lam is not None and n_frac:
            out.append(
                "\tFractional molecule-id: %d, max. Lambda-change: %5f (CFMC) %5f (CB/CFMC)\n"
                % (7, 0.3, 0.3)
            )
            out.append("\tLambda factors: %4f \n" % lam[name])
            out.append("\tBiasing Factors: %4f %4f \n" % (0.0, 1.5))
        out.append(
            "\tabsolute adsorption: %9.5f (avg. %9.5f) [mol/uc], ...\n" % (1.0, 1.0)
        )
    out.append("-" * 148 + "\n")
    out.append(
        "Current total potential energy:       % 22.10f [K]  (avg. % 22.10f)\n"
        % (u[0], u[0])
    )
    out.append(
        "\tCurrent Host-Host energy:           % 22.10f [K]  (avg. % 22.10f)\n"
        % (0.0, 0.0)
    )
    out.append(
        "\tCurrent Host-Adsorbate energy:      % 22.10f [K]  (avg. % 22.10f)\n"
        % (u[1], u[1])
    )
    out.append(
        "\tCurrent Adsorbate-Adsorbate energy: % 22.10f [K]  (avg. % 22.10f)\n"
        % (u[2], u[2])
    )
    return "".join(out) + "\n"


def _perf_block():
    s = "Performance of the translation move:\n======================================\n"
    for k, name in enumerate(("Xe", "Kr")):
        s += f"Component {k} [{name}]\n"
        s += "\ttotal        %f %f %f\n" % (100.0, 110.0, 90.0)
        s += "\tsuccesfull   %f %f %f\n" % (50.0, 55.0, 45.0)
        s += "\taccepted   %f %f %f\n\n" % (0.5, 0.5, 0.5)
    s += "\nPerformance of the identity change move:\n=====\n"
    s += (
        "Component [%s]->[%s] total tried: %f succesfull growth: %f (%f [%%]) "
        "accepted: %f (%f [%%])\n" % ("Xe", "Kr", 40.0, 40.0, 100.0, 10.0, 25.0)
    )
    s += "\nPerformance of the CFCMC swap lambda move:\n=====\n"
    for name in ("Xe", "Kr"):
        s += (
            "Component [%s] total tried: %f constant-lambda accepted: %f (%f [%%])\n"
            % (name, 200.0, 120.0, 60.0)
        )
        s += "               total tried: %f insert-lambda accepted: %f (%f [%%])\n" % (
            20.0,
            5.0,
            25.0,
        )
        s += "               total tried: %f remove-lambda accepted: %f (%f [%%])\n" % (
            18.0,
            4.0,
            22.2,
        )
    s += "\nPerformance of the swap addition move:\n=====\n"
    s += (
        "Component [%s] total tried: %f succesfull growth: %f (%f [%%]) accepted: %f (%f [%%])\n"
        % ("Xe", 30.0, 25.0, 83.3, 6.0, 20.0)
    )
    return s


def _leg(path, cycles, start_n, completed=True):
    text = "[Init] Current cycle: 0 out of 10\nComponent 0 (Xe), current number of integer/fractional/reaction molecules: 99/0/0 (avg. 1.0)\n\n"
    text += "[Equilibration] Current cycle: 0 out of 10\n\n"
    for c in cycles:
        text += _block(
            c,
            cycles[-1],
            [("Xe", start_n + c, 1), ("Kr", 2 * c, 1)],
            lam={"Xe": 0.25, "Kr": 0.75},
        )
    if completed:
        text += _perf_block()
    path.write_text(text)
    return path


def test_production_rows_skip_init_and_equilibration(tmp_path):
    leg = parse_data_file(_leg(tmp_path / "a.data", [0, 10, 20], 5))
    assert leg.components == ["Xe", "Kr"]
    assert [r["cycle"] for r in leg.rows] == [0, 10, 20]
    r = leg.rows[2]
    assert (r["N_Xe"], r["N_Kr"], r["N_total"]) == (25, 40, 65)
    assert (r["frac_Xe"], r["lambda_Xe"], r["lambda_Kr"]) == (1, 0.25, 0.75)
    assert (r["u_total_K"], r["u_hg_K"], r["u_gg_K"]) == (-1000.0, -900.0, -100.0)


def test_move_statistics_from_a_completed_leg(tmp_path):
    leg = parse_data_file(_leg(tmp_path / "a.data", [0, 10], 1))
    assert leg.completed
    assert leg.moves["translation"]["Xe"] == [300.0, 150.0]
    assert leg.moves["identity"]["Xe->Kr"] == [40.0, 10.0]
    assert leg.moves["cf_lambda_constant"]["Kr"] == [200.0, 120.0]
    assert leg.moves["cf_lambda_insert"]["Xe"] == [20.0, 5.0]
    assert leg.moves["cf_lambda_remove"]["Xe"] == [18.0, 4.0]
    assert leg.moves["swap_add"]["Xe"] == [30.0, 6.0]


def test_killed_leg_still_streams_and_legs_stitch(tmp_path):
    a = _leg(tmp_path / "a.data", [0, 10, 20], 1, completed=True)
    b = _leg(tmp_path / "b.data", [0, 10], 3, completed=False)
    legs = [parse_data_file(a), parse_data_file(b)]
    assert not legs[1].completed and len(legs[1].rows) == 2
    header, rows, moves = stitch(legs)
    # leg 1's cycle-0 block is leg 0's restart state and is dropped
    assert [r["cycle_global"] for r in rows] == [0, 10, 20, 30]
    assert [r["leg"] for r in rows] == [0, 0, 0, 1]
    assert moves["completed_legs"] == 1
    assert header[:3] == ["leg", "cycle", "cycle_global"]

    out = tmp_path / "run.log"
    mv = tmp_path / "m.json"
    assert write_parsed([a, b], out, mv) == 4
    got = read_log(out)
    assert list(got)[:4] == ["MC_cycle", "N_Xe", "N_Kr", "U_host_guest_K"]
    assert len(got["N_Kr"]) == 4 and got["N_Kr"][-1] == 20
    assert json.loads(mv.read_text())["moves"]["translation"]["Kr"] == [300.0, 150.0]


def test_a_block_without_components_is_an_error(tmp_path):
    p = tmp_path / "bad.data"
    p.write_text(
        "Current cycle: 5 out of 10\n\nCurrent total potential energy:  -1.0 [K]\n"
    )
    with pytest.raises(ValueError, match="no components"):
        parse_data_file(p)


def test_cfcmc_biasing_factors_with_continuation_lines(tmp_path):
    """RASPA2 prints 10 weights per line and continues on bare-number lines; the
    production value (frozen after Wang-Landau) is the last block's."""
    p = _leg(tmp_path / "a.data", [0, 10], 3)
    first = parse_data_file(p).biasing
    assert first == {"Xe": [0.0, 1.5], "Kr": [0.0, 1.5]}
    text = p.read_text().replace(
        "\tBiasing Factors: %4f %4f \n" % (0.0, 1.5),
        "\tBiasing Factors: %4f %4f \n\t                 %4f %4f \n"
        % (0.0, 1.5, -0.5, -2.0),
    )
    p.write_text(text)
    leg = parse_data_file(p)
    assert leg.biasing["Xe"] == [0.0, 1.5, -0.5, -2.0]
    assert len(leg.rows) == 2  # the continuation lines did not disturb the stream
    _, _, moves = stitch([leg])
    assert moves["cf_biasing_kT"]["Kr"] == [0.0, 1.5, -0.5, -2.0]

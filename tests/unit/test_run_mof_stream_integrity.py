"""The record-run stream must be self-consistent, and aux legs must not cross-resume.

Two failures that were live in the driver and that nothing tested for:

1. SILENT ZERO IN A DECOMPOSITION COLUMN. mof_energy_decomposition keys pair terms
   by TYPE ID order and emits `u_gg_Xe_Kr`; run_mof built its columns from
   sorted(guests), i.e. ALPHABETICALLY, and asked for `u_gg_Kr_Xe`. The writer used
   decomp.get(col, 0.0), so every cross-species guest-guest energy was written as
   0.0 in every row while u_total still contained the term. A mixture stream with a
   silently missing cross term is unusable for the R3.1 comparison, and nothing in
   the suite looked at those columns. The test is the invariant, not the spelling:
   the printed components must sum to the printed total.

2. AUX-ON RESUMING AN AUX-OFF DIRECTORY. The auxiliary changes the acceptance test
   and the RNG consumption, so it changes the chain; if it is absent from the run
   identity the two configurations share a directory and the second silently
   splices itself onto the first. Same class of bug as the pre-fix
   sweep_moves_per_window omission (ledger 2026-07-19, BLOCKER-2).

A small bulk box (rc <= L/2), two species with different sigma: the cross term is only non-trivial
when both species are present and interacting.
"""

from __future__ import annotations

import csv

from scripts.mof.run_mof import MofRunSpec, mof_run_id, run_mof
from scripts.mof.system_setup import setup_system

FF = "ff/forcefield_A.yaml"


def _spec(root, **kw):
    base: dict[str, object] = {
        "forcefield": FF,
        "T": 298.0,
        "z_star_by_name": {"Xe": 0.08, "Kr": 0.08},
        "seed": 7,
        "windows": 6,
        "window_cycles": 60,
        "bulk_box": 30.0,
        "checkpoint_every": 3,
        "root": root,
    }
    base.update(kw)
    return MofRunSpec(**base)


def _rows(result):
    with result.paths.log.open() as fh:
        return list(csv.DictReader(fh))


def test_decomposition_columns_sum_to_the_reported_total(tmp_path):
    r = run_mof(_spec(tmp_path))
    rows = _rows(r)
    assert rows, "no windows streamed"
    parts = [c for c in rows[0] if c.startswith(("u_hg_", "u_gg_"))]
    # all three pair terms must be present as columns at two species
    assert sum(c.startswith("u_gg_") for c in parts) == 3, parts
    for row in rows:
        total = float(row["u_total"])
        summed = sum(float(row[c]) for c in parts)
        assert abs(summed - total) <= 1e-6 * max(1.0, abs(total)), (
            f"components {summed} do not close on u_total {total}; a column is "
            f"being written as a silent 0.0 (columns: {parts})"
        )


def test_the_cross_species_term_is_not_identically_zero(tmp_path):
    """Non-vacuity for the test above: if the two species never met, the closure
    check would pass with the cross column at a legitimate 0.0 and prove nothing."""
    r = run_mof(_spec(tmp_path))
    rows = _rows(r)
    cross = [c for c in rows[0] if c.startswith("u_gg_") and "Xe" in c and "Kr" in c]
    assert len(cross) == 1, cross
    values = [float(row[cross[0]]) for row in rows]
    assert any(v != 0.0 for v in values), (
        f"{cross[0]} is zero in every window, so the closure test is vacuous: "
        f"either the species never interact in this fixture or the column is "
        f"still being filled by decomp.get(col, 0.0)"
    )


def test_the_auxiliary_is_part_of_the_run_identity(tmp_path):
    off = _spec(tmp_path, sweep_moves_per_window=2, sweep_L_long=2.0)
    on = _spec(tmp_path, sweep_moves_per_window=2, sweep_L_long=2.0, sweep_rc_aux=1.5)
    noop = _spec(tmp_path, sweep_moves_per_window=2, sweep_L_long=2.0, sweep_rc_aux=2.5)
    assert mof_run_id(off) != mof_run_id(on), (
        "aux-on and aux-off share a run id, so an aux leg would resume an "
        "aux-off checkpoint and splice two different chains"
    )
    assert mof_run_id(on) != mof_run_id(noop), (
        "two different proposal radii share a run id"
    )


def test_sweep_counters_are_streamed_when_the_sweep_is_on(tmp_path):
    """The sweep firing rate and the captured window occupancy k are the two numbers
    that decide whether the DP (and therefore the auxiliary) can pay at all. They
    were not reported, so a sweep that declined every attempt looked identical to
    one that worked."""
    r = run_mof(_spec(tmp_path, sweep_moves_per_window=3, sweep_L_long=2.0))
    rows = _rows(r)
    for col in ("sweep_attempts", "sweep_accepts", "sweep_k_mean", "sweep_k_max"):
        assert col in rows[0], f"{col} missing from the stream"
    assert sum(int(row["sweep_attempts"]) for row in rows) == 3 * len(rows)


def test_the_mof_no_op_auxiliary_reproduces_the_auxiliary_off_chain(tmp_path):
    """`sweep_rc_aux == rc` must be an exact no-op, in the MOF driver too.

    The third place this gate was missing. `_channel_move_ms` guarded the
    correction on `rc_aux is not None and rc_sel < rc`, so the spec this file
    already calls `noop` never reached the correction it was meant to control --
    the same shape as the bulk engine's `aux_on = ljbranch and rc_sel < uc` and
    the typed `aux: {control: true}` returning no cutoff at all. The flag now
    reads `rc_aux is not None`, so the correction runs, its value is exactly zero
    (both splits are the same sum over the same pairs), and the chain must be the
    auxiliary-off chain.

    WHAT IS COMPARED, AND WHY IT IS THE STREAM. The driver's chain state is not
    returned, so the per-window CSV is the observable: it carries N per species,
    the energy decomposition and the sweep counters, which is a strictly richer
    record than the counter stream this file's other tests read. Every column of
    every row must match. Comparing floats for equality is correct here rather
    than sloppy: the claim is bit-identity, and the two runs execute the same
    arithmetic in the same order with the same seed, so a mismatch of one ulp is
    the finding, not noise.

    The two runs still have DIFFERENT run ids, which is what
    `test_the_auxiliary_is_part_of_the_run_identity` above requires -- the
    auxiliary is identity-bearing even when it is a no-op, because a leg that
    resumed across it would splice two chains. So they get separate directories
    and neither reads the other's checkpoint.
    """
    # rc comes from the SYSTEM, not the force-field file, and the difference is a
    # finding in its own right. `MofRunSpec.sweep_rc_aux` is compared against
    # `fieldobj.rc`, which is REDUCED (2.927 for this force field); the file's
    # `interaction.rc` is 12.0 Angstrom. The spec field carries no unit in its
    # name or its comment, and `test_the_auxiliary_is_part_of_the_run_identity`
    # above calls 2.5 the "noop" spec -- 2.5 reduced against 2.927 reduced is a
    # real short-range auxiliary, not a no-op. That test only compares run ids, so
    # the mislabel cost it nothing; a control arm built on the same reading would
    # have compared two auxiliaries and called their agreement a no-op.
    rc = setup_system(FF, 298.0, bulk_box=30.0, reference="Xe").field.rc

    off = run_mof(_spec(tmp_path / "off", sweep_moves_per_window=2, sweep_L_long=2.0))
    noop = run_mof(
        _spec(
            tmp_path / "noop",
            sweep_moves_per_window=2,
            sweep_L_long=2.0,
            sweep_rc_aux=rc,
        )
    )

    # `wall_s` and the cumulative evaluation counters are the clock, not the
    # chain. Everything else -- occupancies, the energy decomposition, the sweep
    # counters, the auxiliary's own moments -- is chain state and is compared.
    # `wall_s` is the clock. `aux_corr_n` is the diagnostic whose PRESENCE is the
    # difference between the two arms -- the off arm evaluates none by definition
    # -- so it is asserted directly above rather than compared. `aux_corr_mean`
    # and `aux_corr_sd` stay IN the comparison: both arms must report 0, which is
    # the exactness claim, and dropping them would remove the assertion that the
    # correction really was zero and not merely rounded to it in the log.
    volatile = {"wall_s", "aux_corr_n"}
    # NON-VACUITY FIRST. A correction that never ran reports mean 0 and sd 0,
    # exactly as a correct no-op does, so the count is what separates the two.
    # Asserted before the comparison, because otherwise the comparison below
    # passes hardest when the thing under test did not happen.
    ran = sum(int(r["aux_corr_n"]) for r in _rows(noop))
    assert ran > 0, (
        "the no-op auxiliary evaluated the correction 0 times, so the "
        "bit-identity below is a comparison of two auxiliary-off chains. Check "
        "that the correction is not guarded on rc_sel < rc."
    )
    assert sum(int(r["aux_corr_n"]) for r in _rows(off)) == 0, (
        "the auxiliary-OFF arm evaluated a correction; the two arms are not "
        "what they say they are"
    )
    rows_off = [{k: v for k, v in r.items() if k not in volatile} for r in _rows(off)]
    rows_noop = [{k: v for k, v in r.items() if k not in volatile} for r in _rows(noop)]
    assert rows_off, "no windows streamed"
    assert len(rows_off) == len(rows_noop), (
        f"{len(rows_off)} windows without the auxiliary, {len(rows_noop)} with "
        "the no-op one"
    )
    for i, (a, b) in enumerate(zip(rows_off, rows_noop)):
        differing = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
        assert not differing, (
            f"window {i} differs between the auxiliary-off run and the no-op "
            f"auxiliary: {differing}. At sweep_rc_aux == rc the correction is a "
            "difference of two identical sums, so any difference here means it is "
            "not being added where the algebra puts it."
        )

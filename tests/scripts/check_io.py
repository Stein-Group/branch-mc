"""IO checks: checkpoint round-trip and atomic fallback, RNG restore through the
JSON blob, trajectory snapshot vs full, log header-once and trim. No engine or
original needed.
"""

import os
import sys
import tempfile

import numpy as np
from ase.io import read as ase_read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from gcmc import io


def paths_in(d):
    # the real dataclass (str fields are coerced to Path in __post_init__),
    # so this harness exercises the exact object the runner hands to io
    from gcmc.hash_fingerprint import RunPaths

    return RunPaths(
        directory=d,
        checkpoint=os.path.join(d, "checkpoint.npz"),
        checkpoint_bak=os.path.join(d, "checkpoint.bak.npz"),
        trajectory=os.path.join(d, "trajectory.extxyz"),
        log=os.path.join(d, "run.log"),
        params=os.path.join(d, "params.json"),
        lock=os.path.join(d, "lock"),
    )


with tempfile.TemporaryDirectory() as root:
    P = paths_in(root)

    # 1. checkpoint round-trip
    gen = np.random.default_rng(0)
    _ = gen.random(3)
    st = io.rng_state(gen)
    pos = np.random.default_rng(1).uniform(0, 10, size=(17, 3))
    counters = {"add_acc": 5, "add_att": 9, "rm_acc": 2}
    params = {
        "potential": {"main": "lennard_jones", "T": 0.85},
        "ensemble": {"mu": -4.0},
    }
    io.save_checkpoint(
        P, positions=pos, step=4000, rng=st, counters=counters, params=params
    )
    ck = io.load_checkpoint(P)
    assert np.allclose(ck["positions"], pos), "positions round-trip"
    assert ck["step"] == 4000 and ck["n"] == 17, "step/n round-trip"
    assert ck["counters"] == counters, "counters round-trip"
    assert ck["params"] == params, "params round-trip"
    print("  ok  checkpoint round-trip")

    # 2. RNG restore reproduces the stream
    seq_ref = gen.random(5)  # gen continues from the saved state
    gen2 = io.set_rng_state(np.random.default_rng(), ck["rng"])
    seq_new = gen2.random(5)
    assert np.array_equal(seq_ref, seq_new), "RNG state restore must reproduce draws"
    print("  ok  RNG state restored through JSON blob")

    # 3. atomic fallback: second save rotates v1 -> bak; corrupt primary -> load bak
    pos2 = pos + 100.0
    io.save_checkpoint(
        P, positions=pos2, step=8000, rng=st, counters=counters, params=params
    )
    assert os.path.exists(P.checkpoint_bak), "bak should exist after second save"
    with open(P.checkpoint, "wb") as f:
        f.write(b"corrupt not an npz")
    ck_fb = io.load_checkpoint(P)
    assert ck_fb is not None and ck_fb["step"] == 4000, "must fall back to .bak (v1)"
    print("  ok  corrupt primary falls back to .bak")

    # 4. trajectory: snapshot overwrites (1 frame); full appends (grows)
    snap = paths_in(os.path.join(root, "snap"))
    os.makedirs(snap.directory)
    for s in (1000, 2000, 3000):
        io.write_frame(
            snap, pos[:5], two_dim=False, box=20.0, info={"step": s}, append=False
        )
    assert len(ase_read(snap.trajectory, index=":")) == 1, "snapshot keeps one frame"
    full = paths_in(os.path.join(root, "full"))
    os.makedirs(full.directory)
    for s in (1000, 2000, 3000):
        io.write_frame(
            full, pos[:5], two_dim=False, box=20.0, info={"step": s}, append=True
        )
    assert len(ase_read(full.trajectory, index=":")) == 3, "full trajectory appends"
    print("  ok  trajectory snapshot vs full")

    # 4b. 2D frame pads z and is reload-safe
    io.write_frame(snap, pos[:4, :2], two_dim=True, box=75.0, info={}, append=False)
    a = ase_read(snap.trajectory, index=-1)
    assert a.positions.shape == (4, 3) and np.allclose(a.positions[:, 2], 0.0), (
        "2D pads z=0"
    )
    print("  ok  2D frame pads z")

    # 5. log: header once across reopen, then trim
    hdr = "#MC_cycle(M)  eta  N"
    lg = io.Log(P.log, hdr)
    lg.write("0.001000 0.10 5")
    lg.write("0.002000 0.11 6")
    lg.close()
    lg2 = io.Log(P.log, hdr)
    lg2.write("0.003000 0.12 7")
    lg2.close()
    with open(P.log) as _f:
        body = _f.read()
    assert body.count("#MC_cycle") == 1, "header written once"
    assert body.count("\n") == 4, "header + 3 rows"
    io.trim_log(P.log, max_cycle_millions=0.002)  # drop the 0.003 row
    with open(P.log) as _f:
        rows = [l for l in _f.read().splitlines() if l and not l.startswith("#")]
    assert len(rows) == 2 and rows[-1].startswith("0.002000"), "trim past checkpoint"
    print("  ok  log header-once and trim")

print("\nALL IO TESTS PASSED")

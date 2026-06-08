"""Migration checks: a legacy .extxyz converts to a params.json plus a checkpoint
that `run` resumes from, with positions and cycle count preserved.
"""
import os
import sys
import json
import glob
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gcmc import migrate, identity, config, io, runner  # noqa: E402


def make_legacy(path, positions3d, side, *, mu, cav, k, mc_cycle, two_dim, aux):
    from ase import Atoms
    from ase.io import write
    at = Atoms("H" * len(positions3d), positions=positions3d,
               cell=[side]*3, pbc=True)
    at.info.update({"mc_cycle": mc_cycle, "chem_pot": mu, "cavity_radius": cav,
                    "trial_density": 0.5, "no_trials": k, "two_dim": two_dim,
                    "aux_pot": aux})
    write(path, at, append=False)


with tempfile.TemporaryDirectory() as tmp:
    # ---- a hard-sphere 3D legacy run -----------------------------------
    rng = np.random.default_rng(0)
    pos3d = rng.uniform(1, 19, size=(120, 3))
    name = "mu_3.0_r_1.5_rhok_0.5_k_30_2d_False_.extxyz"
    legacy = os.path.join(tmp, name)
    make_legacy(legacy, pos3d, 20.0, mu=3.0, cav=1.5, k=30, mc_cycle=5_000_000,
                two_dim=False, aux=False)

    params, positions, step = migrate.legacy_to_params(legacy, temp=1.0)
    assert params["potential"]["type"] == "hard_sphere"
    assert params["ensemble"] == {"mu": 3.0, "dimension": 3, "box": 20.0}
    assert params["trials"] == {"k": 30, "cavity_radius": 1.5}
    assert step == 5_000_000
    assert positions.shape == (120, 3) and np.allclose(positions, pos3d)
    print("  ok  hard-sphere 3D legacy parsed (params, positions, step)")

    run_dir = migrate.migrate(legacy, root=os.path.join(tmp, "runs"), temp=1.0)
    assert os.path.exists(os.path.join(run_dir, "checkpoint.npz"))
    assert os.path.exists(os.path.join(run_dir, "params.json"))
    # checkpoint round-trips the configuration and cycle count
    cfg = config.from_params(params)
    paths = identity.run_paths(cfg, os.path.join(tmp, "runs"))
    ck = io.load_checkpoint(paths)
    assert int(ck["step"]) == 5_000_000
    assert np.allclose(ck["positions"], pos3d)
    print(f"  ok  checkpoint written and resumable in {os.path.basename(run_dir)}")

    # ---- `run` resumes from the migrated checkpoint --------------------
    with open(os.path.join(run_dir, "params.json")) as f:
        p = json.load(f)
    p["execution"] = {"sampling_window": 500, "target_cycles": 5_000_000 + 1000,
                      "checkpoint_every": 1, "seed": 1}
    state, paths2 = runner.run(p, root=os.path.join(tmp, "runs"))
    assert paths2.directory == run_dir, "resume must land in the same directory"
    assert state.step >= 5_000_000 + 1000, f"did not advance past the legacy cycle: {state.step}"
    print(f"  ok  run resumed from cycle 5,000,000 and advanced to {state.step}")

    # ---- a 2D aux legacy name parses to the right schema ---------------
    pos2d = np.column_stack((rng.uniform(1, 74, size=(50, 2)), np.zeros(50)))
    name2 = ("mu_2.0_r_2.5_rhok_0.4_k_20_2d_True_aux_pot_True_"
             "gammamax_5.0_deltaeta_0.05_.extxyz")
    legacy2 = os.path.join(tmp, name2)
    make_legacy(legacy2, pos2d, 75.0, mu=2.0, cav=2.5, k=20, mc_cycle=10,
                two_dim=True, aux=True)
    pr, po, st = migrate.legacy_to_params(legacy2, temp=1.0)
    assert pr["potential"]["type"] == "aux"
    assert pr["ensemble"]["dimension"] == 2 and pr["ensemble"]["box"] == 75.0
    assert po.shape == (50, 2), "2D migration must drop the padded z column"
    config.from_params(pr)  # the emitted schema validates
    print("  ok  2D aux legacy parsed and validates as a tagged schema")

print("\nALL MIGRATION TESTS PASSED")

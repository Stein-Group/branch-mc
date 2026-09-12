#!/bin/bash
# The Glauber engine's production sweeps: one base config -> a Cartesian
# sweep of single-core runs, expanded and scheduled by gcmc.run.launcher, then
# read by scripts/tau_report.py so the numbers land beside the runs.
#
# THREE FILES, and the first two are one experiment in two halves:
#
#   ./scripts/submit_engine_sweep.sbatch run-files/tau_crossover.json    # 14 runs
#   ./scripts/submit_engine_sweep.sbatch run-files/tau_crossover_reference.json  #  7 runs
#   ./scripts/submit_engine_sweep.sbatch run-files/srsw_lj_rc3.json        # 10 runs
#
# READ THIS FIRST IF YOU WANT THE ONE NUMBER: it is tau_crossover.json. The
# Glauber engine's advantage is density-dependent -- clear at rho* = 0.12, gone by
# rho* = 0.59 -- and where between them it disappears is the number that decides
# whether the engine is worth running for a given problem. The reference file is
# its third arm (plain single-particle insertion, which is what most codes do)
# and is a separate file because `sampler.type` decides which keys the validator
# accepts, so it cannot be a swept axis. `sampler.engine` can be and is.
#
# The SRSW file is a different Hamiltonian, not a different setting: truncation
# at 3 sigma with the tail on, which is what those tables use, and the cuboid is
# 4.5 sigma long because the periodic guard caps the body diagonal at box - uc.
# It closes the <N>-against-mu half of a SRSW comparison. It does NOT produce
# ln Pi(N) -- flat-histogram biasing is not driven from a params file -- and its
# header says what that would need.
#
# The runs root defaults to runs/<base name>, so the three never share a
# directory:
#
#   ./scripts/submit_engine_sweep.sbatch run-files/tau_crossover.json runs/tau
#
# Same shape as submit_demixing.sbatch, which is the pattern that works: the
# launcher takes the Cartesian product, refuses to start if the product needs
# more cores than it was given, and keeps the node at its budget as runs finish.
# A run that is killed resumes from its checkpoint on resubmission -- runner.run
# is idempotent and a finished run re-entered is a no-op -- so re-running this
# script is always safe.
#
# LENGTH IS THE RESULT HERE, which is unusual and worth knowing before trimming
# target_cycles to fit a queue. `Tau.reliable` wants a series of about fifty
# tau, and at rho* = 0.6 tau is three to ten windows. At 40 measured windows
# this very comparison made the Glauber engine look 1.5x faster than both incumbents;
# at 400 it is at parity. scripts/tau_report.py refuses to print a ratio for a
# state point where either arm came back unresolved, so a short run cannot
# quietly become a result -- but it also cannot give you the answer.
#
#SBATCH --clusters=httc
#SBATCH --partition=httc_high_mem
#SBATCH --mail-user=dipesh.somvanshi@tum.de
#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --mem=8000
#SBATCH --output=%x.out
#SBATCH --qos=httc_mid_HM
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    BASE="${1:?usage: ./submit_engine_sweep.sbatch <params.json> [runs_root]}"
    exec sbatch --job-name="$(basename "$BASE" .json)" "$0" "$@"
fi

# --- environment ----------------------------------------------------------
# module load python
# source ~/venvs/gcmc/bin/activate
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BASE="${1:?usage: ./submit_engine_sweep.sbatch <params.json> [runs_root]}"
ROOT="${2:-runs/$(basename "$BASE" .json)}"

# The core budget comes from the BASE FILE's run.n_cores, so the params and the
# allocation cannot disagree. Keep --cpus-per-task >= the largest n_cores any
# base file asks for; tau_crossover.json is the largest at 14.
CORES=$(python - "$BASE" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1])).get("run", {}).get("n_cores", 14))
PYEOF
)

# ONE THREAD PER RUN, and this is not a performance hint. The metric is
# wall-seconds per independent value of N; an oversubscribed node makes wall
# clock depend on what else is running, which turns a measurement into a coin
# flip. The launcher's core budget and --threads 1 together are what keep the
# two arms comparable.
python -m gcmc.run.launcher "$BASE" --root "$ROOT" --cores "$CORES" --threads 1

# The reading half, run here so the table lands next to the runs rather than
# waiting for someone to remember. Cheap -- it reads run.log and nothing else --
# so it is serial and in-process rather than fanned out.
python -m scripts.tau_report "$ROOT" --out "$ROOT/tau.json"

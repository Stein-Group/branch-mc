#!/bin/bash
# Serial mu-sweep: one base config -> 16 single-core runs, one per mu.
#
#   ./submit_serial.sbatch params/lj_T0.85_eq.json
#   ./submit_serial.sbatch params/aux3d.json runs/aux3d
#
#SBATCH --clusters=httc
#SBATCH --partition=httc_high_mem
#SBATCH --mail-user=dipesh.somvanshi@tum.de
#SBATCH --cpus-per-task=22
#SBATCH --time=24:00:00
#SBATCH --mem=10000
#SBATCH --output=%x.out
#SBATCH --qos=httc_mid_HM
set -euo pipefail

# Run directly (not yet under SLURM): re-exec through sbatch with the
# job name set to the config basename. --output=%x.out then follows it, so
# lj_T0.85_eq.json -> job "lj_T0.85_eq" -> lj_T0.85_eq.out
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    BASE="${1:?usage: ./submit_serial.sbatch <params.json> [runs_root]}"
    exec sbatch --job-name="$(basename "$BASE" .json)" "$0" "$@"
fi

# --- environment ----------------------------------------------------------
# module load python
# source ~/venvs/gcmc/bin/activate
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BASE="${1:?usage: ./submit_serial.sbatch <params.json> [runs_root]}"
ROOT="${2:-runs/$(basename "$BASE" .json)}"

python -m gcmc.launcher "$BASE" --root "$ROOT" --threads 1

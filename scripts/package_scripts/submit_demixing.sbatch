#!/bin/bash
# The LJ A-B demixing series: one base config -> a Cartesian sweep of
# single-core runs, expanded and scheduled by gcmc.run.launcher.
#
# TWO ARMS, submitted as two jobs. They are the same experiment at one state
# point and they answer different halves of P8's first claim:
#
#   ./scripts/submit_demixing.sbatch run-files/demix_canonical.json    # THE ONE
#   ./scripts/submit_demixing.sbatch run-files/demix_kseries.json      # muVT
#   ./scripts/submit_demixing.sbatch run-files/demix_kseries_nvt.json  # semigrand
#
# READ THIS FIRST IF YOU WANT TWO PHASES IN A BOX: it is demix_canonical.json,
# at T* = 1.3, and neither of the other two can show you that. The first is
# grand canonical and the second is semi-grand, and in both the composition is
# free to move -- so both dissolve an interface rather than hold one, which is
# not a bug in them but the equilibrium they sample. They measure composition
# curves; the canonical file measures the two-phase state.
#
# TEMPERATURE. The canonical and semi-grand files run at T* = 1.3, because that
# is what separated on the cluster (Dipesh, 2026-08-27). Note that re-running the
# canonical file HERE separates at both 1.0 and 1.3, so the mechanism is not
# established -- demix_canonical.json's header has the numbers and the one run
# that would settle it. The muVT file alone stays at 1.0, because its chemical
# potentials are calibrated at that temperature and nowhere else.
#
# muVT (18 runs: 6 k x 3 starts) gives the transition curve -- composition
# against k at fixed chemical potentials on the asymmetric model, where a
# demixing mixture expels a species and the density falls with it. Fixed N
# (30 runs: 6 k x 5 Delta-mu, one equimolar start) is the semi-grand arm -- now
# spelled ensemble.type = "semigrand" rather than "NVT" -- on the
# SYMMETRIC model, where Delta-mu = 0 is exactly the coexistence condition and
# the sweep traces the isotherm x(Delta-mu) whose shape IS the transition. Each
# file says the rest in its own header.
#
# The runs root defaults to runs/<base name>, so the two arms never share a
# directory:
#
#   ./scripts/submit_demixing.sbatch run-files/demix_kseries.json runs/demix
#
# Same shape as submit_serial.sbatch, which is the pattern that works: the
# launcher takes the Cartesian product, refuses to start if the product needs
# more cores than it was given, and keeps the node at its budget as runs finish.
#
# SIZING, measured rather than guessed. muVT arm: one window of 2000 cycles at
# N = 2744 costs about 5 s on one core, so its 4e6 target is about 3 h per run.
# Fixed-N arm: one window of 1000 cycles at N = 5600 with the full-repacking
# sampler costs about 7.3 s, so its 5e6 target is about 10 h per run. Both fit
# the 24 h wall, and 30 runs on 30 cores run concurrently. A run killed anyway resumes from its checkpoint on
# resubmission -- `runner.run` is idempotent and a finished run re-entered is a
# no-op, so re-running this script is always safe.
#
#SBATCH --clusters=httc
#SBATCH --partition=httc_high_mem
#SBATCH --mail-user=dipesh.somvanshi@tum.de
#SBATCH --cpus-per-task=30
#SBATCH --time=24:00:00
#SBATCH --mem=16000
#SBATCH --output=%x.out
#SBATCH --qos=httc_mid_HM
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    BASE="${1:?usage: ./submit_demixing.sbatch <params.json> [runs_root]}"
    exec sbatch --job-name="$(basename "$BASE" .json)" "$0" "$@"
fi

# --- environment ----------------------------------------------------------
# module load python
# source ~/venvs/gcmc/bin/activate
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BASE="${1:?usage: ./submit_demixing.sbatch <params.json> [runs_root]}"
ROOT="${2:-runs/$(basename "$BASE" .json)}"

# The core budget, taken from the BASE FILE's run.n_cores so the launcher and
# the analysis fan-out cannot disagree with each other or with the params. Falls
# back to the allocation above. Keep --cpus-per-task >= the largest n_cores any
# base file asks for: the muVT arm is 18 runs, the fixed-N arm 30.
CORES=$(python - "$BASE" <<'PYEOF'
import json, sys
print(json.load(open(sys.argv[1])).get("run", {}).get("n_cores", 30))
PYEOF
)

# The models and the starts are DERIVED, not committed state, and the BASE FILE
# says what they are -- its "_models" and "_starts" blocks. Regenerated here so a
# job landing on a fresh checkout cannot run against a stale series or a start
# someone edited; idempotent, since the same anchor and seed give the same bytes.
#
# This was a hardcoded box, eta and composition loop. That could not serve two
# arms with different geometry without a second copy, and the wrong copy would
# have been found by a run starting from the wrong file rather than by anything
# failing. `prepare` also checks that every model and start the params NAME now
# exists, so a typo fails here in a second instead of eighteen runs later.
python -m scripts.demixing prepare "$BASE"

python -m gcmc.run.launcher "$BASE" --root "$ROOT" --cores "$CORES" --threads 1

# Analysis, run here so the numbers land next to the runs rather than waiting
# for someone to remember. Each directory gets its own JSON: composition per
# window, the order parameter and the composition profile per frame, the partial
# g_ab, and the permutation null the order parameter has to beat.
#
# In parallel across the same core budget, because it is not free: measured at
# N = 2744, one frame costs about 0.7 s across the three observables, so 200
# frames is ~3 minutes per run and 18 runs would be an hour of the wall spent
# serially at the very end of the job, after the physics is already done.
#
# `-P "$CORES" -n 1` and a subshell per directory rather than one long pipeline,
# so a directory that fails to analyse (a killed run with a truncated last frame,
# say) does not take the other seventeen with it -- the loop reports it and
# continues, since a missing JSON is recoverable by re-running this script and a
# lost hour is not.
export PYTHONPATH
printf '%s\n' "$ROOT"/*/ | while read -r d; do
    [[ -f "$d/run.log" ]] && printf '%s\n' "$d"
done | xargs -r -P "$CORES" -I{} bash -c \
    'python -m scripts.demixing analyse "$1" --out "$1/demixing.json" \
        || echo "ANALYSIS FAILED: $1" >&2' _ {}

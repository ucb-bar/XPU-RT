#!/usr/bin/env bash
# One GPU, strictly sequential. Waits for whatever flight job is running, then runs the
# campaign in the order that makes early results usable:
#   1 showdown campaign (arms x speeds x gains)      scripts/campaign_showdown.sh
#   2 envelope, calibrated gain, course A            scripts/gain_controlled_grid.sh
#   3 mechanism runs                                 scripts/run_energy_experiment.sh
#   4 envelope, calibrated gain, course B            WAREHOUSE_COURSE=b scripts/gain_controlled_grid.sh
# Every stage is cell-resumable; re-running this script after an interruption is safe.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
say() { echo "=== $(date +%H:%M:%S) $*"; }
while pgrep -f "g0_backlog_gate.sh|sweep_rate_demo.py" >/dev/null; do sleep 60; done
say "GPU free"
say "stage 1: showdown campaign";      scripts/campaign_showdown.sh
say "stage 2: envelope on-law, A";     [ -x scripts/gain_controlled_grid.sh ] && scripts/gain_controlled_grid.sh
say "stage 3: mechanism";              [ -x scripts/run_energy_experiment.sh ] && RES="$REPO/results/codesign_feedback" scripts/run_energy_experiment.sh
say "stage 4: envelope on-law, B";     [ -x scripts/gain_controlled_grid.sh ] && WAREHOUSE_COURSE=b scripts/gain_controlled_grid.sh
say "GPU_QUEUE_DONE"

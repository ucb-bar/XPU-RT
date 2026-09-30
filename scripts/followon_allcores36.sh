#!/usr/bin/env bash
# Everything the 36 Hz all-8-hart showdown still needs after its census: panel D's energy ladder,
# the display pairs on the separating seeds, and the renders. Each stage waits for the one before,
# so this can be launched while the census is still flying.
#
#   scripts/followon_allcores36.sh
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback; T=$PWD/$R/ctrl_traces
say(){ echo "=== $(date +%H:%M:%S) $*"; }

say "waiting for the census"
until grep -q "CAMPAIGN_ALLCORES36_DONE" "$R/campaign_allcores36_run.log" 2>/dev/null; do sleep 60; done

say "panel D's energy runs, cadence only"
CONDS="xpu_p36free:$T/xpu_p36free.csv ros_vanilla4x2_36:$T/ros_vanilla4x236.csv" \
  CRUISE=1.4 GAIN=0.0055 ER=$PWD/$R/energy_runs_allcores36 \
  OUTCSV=$PWD/$R/flight_energy_allcores36.csv MAX_SIMS=3 \
  bash scripts/run_energy_pair.sh 2>&1 | tail -4

say "display pairs"
bash scripts/display_pairs_allcores36.sh 2>&1 | tail -8

say "render"
bash scripts/render_allcores36.sh 2>&1 | tail -20
say "FOLLOWON_ALLCORES36_DONE"

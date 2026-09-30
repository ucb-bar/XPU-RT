#!/usr/bin/env bash
# The submitted-equivalent figure at 45 Hz, everything it still needs from the GPU, in order.
#
# Queued behind the 30 Hz chain because they share the simulator budget. Stage 3 flies FOUR arms, not
# two: the mechanism panel is a ladder over scheduling quality -- our solver, our greedy scheduler,
# the static 6-core ROS partition, and the deployment the submitted figure described -- so the reader
# sees a spectrum rather than a two-horse race, and two of the four rungs are ours.
set -u
cd "$(dirname "$0")/.."
R=results/codesign_feedback
say(){ echo "=== $(date +%H:%M:%S) $*"; }

until grep -qE "CAMPAIGN_STATIC6_DONE|FOLLOWON_STATIC6_DONE|no pair accepted" $R/campaign_static6.log $R/followon_static6.log 2>/dev/null; do sleep 120; done
say "the 30 Hz chain has released the simulators"

say "stage 0: the 45 Hz census (baseline only; the scheduled arm's rows carry over)"
MAX_SIMS=3 bash scripts/campaign_static6_45.sh 2>&1 | tail -8

say "stage 1: the displayed pair"
MAX_SIMS=3 CRUISES="${CRUISES:-1.4 1.6 1.2 1.0 1.8}" bash scripts/display_search_static6_45.sh 2>&1 | tail -40
D=$R/campaign_static6_45/display
SEED=$(grep -ho "PAIR_SEED=[0-9]*" $D/search_c*.log 2>/dev/null | head -1 | cut -d= -f2)
CRU=$(grep -l "PAIR_SEED" $D/search_c*.log 2>/dev/null | head -1 | sed 's/.*search_c//; s/\.log//')
if [ -z "${SEED:-}" ]; then echo "no pair accepted; stages 2 and 3 need one, stopping here"; exit 0; fi
say "pair: cruise $CRU seed $SEED"

say "stage 2: panel A's scene census, twelve seeds per arm on the scene the pair flies"
T=$PWD/$R/ctrl_traces
CELL=static6_45_l$SEED LAYOUT_SEED=$SEED CRUISE=$CRU MAX_SIMS=3 GAIN=0.0055 \
  ARMS="xpu:$T/xpu_a_cpsat_hard.csv:56.8:0 p3:$T/ros_cp345.csv:56.2:0" \
  bash scripts/scene_runs_pair.sh 2>&1 | tail -6

say "stage 3: panel D's ladder, four arms, cadence only"
CONDS="xpu_cpsat:$T/xpu_a_cpsat_hard.csv xpu_greedy:$T/xpu_a_greedy.csv ros_static6:$T/ros_cp345.csv ros_shipped:$T/ros_vanilla_c5045.csv" \
  CRUISE=$CRU GAIN=0.0055 ER=$PWD/$R/energy_runs_static6_45 OUTCSV=$PWD/$R/flight_energy_static6_45.csv \
  MAX_SIMS=3 bash scripts/run_energy_pair.sh 2>&1 | tail -6

say "FOLLOWON_STATIC6_45_DONE"

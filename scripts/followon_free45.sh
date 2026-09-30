#!/usr/bin/env bash
# The 45 Hz solver-derived comparison, everything it needs from the GPU, queued behind the census
# that is establishing why it was needed.
set -u
cd "$(dirname "$0")/.."
R=results/codesign_feedback
say(){ echo "=== $(date +%H:%M:%S) $*"; }

C=$R/campaign_static6_45/campaign.csv
until [ -f "$C" ] && [ "$(( $(wc -l < "$C") - 1 ))" -ge 60 ]; do sleep 120; done
say "the 45 Hz baseline census is complete ($(( $(wc -l < "$C") - 1 )) rows); the simulators are free"

say "stage 0: the scheduled arm's 60 flights (the baseline's carry over)"
MAX_SIMS=3 bash scripts/campaign_free45.sh 2>&1 | tail -8

say "stage 1: the displayed pair"
MAX_SIMS=3 CRUISES="${CRUISES:-1.4 1.6 1.2 1.0 1.8}" bash scripts/display_search_free45.sh 2>&1 | tail -40
D=$R/campaign_free45/display
SEED=$(grep -ho "PAIR_SEED=[0-9]*" $D/search_c*.log 2>/dev/null | head -1 | cut -d= -f2)
CRU=$(grep -l "PAIR_SEED" $D/search_c*.log 2>/dev/null | head -1 | sed 's/.*search_c//; s/\.log//')
if [ -z "${SEED:-}" ]; then echo "no pair accepted; stages 2 and 3 need one, stopping here"; exit 0; fi
say "pair: cruise $CRU seed $SEED"

say "stage 2: panel A's scene census, twelve seeds per arm"
T=$PWD/$R/ctrl_traces
CELL=free45_l$SEED LAYOUT_SEED=$SEED CRUISE=$CRU MAX_SIMS=3 GAIN=0.0055 \
  ARMS="xpu:$T/xpu_p45free.csv:28.3:0 p3:$T/ros_cp345.csv:56.2:0" \
  bash scripts/scene_runs_pair.sh 2>&1 | tail -6

say "stage 3: panel D's ladder, four arms, cadence only"
CONDS="xpu_p45free:$T/xpu_p45free.csv xpu_greedy:$T/xpu_a_greedy.csv ros_static6:$T/ros_cp345.csv ros_shipped:$T/ros_vanilla_c5045.csv" \
  CRUISE=$CRU GAIN=0.0055 ER=$PWD/$R/energy_runs_free45 OUTCSV=$PWD/$R/flight_energy_free45.csv \
  MAX_SIMS=3 bash scripts/run_energy_pair.sh 2>&1 | tail -6

say "FOLLOWON_FREE45_DONE"

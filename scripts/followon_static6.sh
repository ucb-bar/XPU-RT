#!/usr/bin/env bash
# Everything the per-node-pinning figure still needs from the GPU after its census, in order.
#
# The baseline here is the deployment the Tier A showdown's panel I actually drew: `cp3`, a static
# partition with a four-hart YOLO pool, nav and control each pinned to one further hart, and two harts
# never touched. Its cadence is the same 30 Hz as the 8-core arm (control gap 33.40 ms on both), so the
# flights differ from the free30 form only in which baseline deployment the schedule panel shows.
#
# Each stage waits for the one before because they share the simulator budget. Nothing here renders.
set -u
cd "$(dirname "$0")/.."
R=results/codesign_feedback
say(){ echo "=== $(date +%H:%M:%S) $*"; }

C=$R/campaign_static6/campaign.csv
until [ -f "$C" ] && [ "$(( $(wc -l < "$C") - 1 ))" -ge 96 ]; do sleep 120; done
say "census complete ($(( $(wc -l < "$C") - 1 )) rows)"

say "stage 1: the displayed pair"
MAX_SIMS=3 CRUISES="${CRUISES:-1.6 1.4 1.2 1.0}" bash scripts/display_search_static6.sh 2>&1 | tail -40
D=$R/campaign_static6/display
SEED=$(grep -ho "PAIR_SEED=[0-9]*" $D/search_c*.log 2>/dev/null | head -1 | cut -d= -f2)
CRU=$(grep -l "PAIR_SEED" $D/search_c*.log 2>/dev/null | head -1 | sed 's/.*search_c//; s/\.log//')
if [ -z "${SEED:-}" ]; then echo "no pair accepted; stages 2 and 3 need one, stopping here"; exit 0; fi
say "pair: cruise $CRU seed $SEED"

say "stage 2: panel A's scene census, twelve seeds per arm on the scene the pair flies"
T=$PWD/$R/ctrl_traces
CELL=static6_l$SEED LAYOUT_SEED=$SEED CRUISE=$CRU MAX_SIMS=3 GAIN=0.00500 \
  ARMS="xpu:$T/xpu_p30free.csv:26.8:33.3 p3:$T/ros_cp330.csv:30.7:33.3" \
  bash scripts/scene_runs_pair.sh 2>&1 | tail -6

say "stage 3: panel D's energy runs, cadence only"
CONDS="xpu_free30:$T/xpu_p30free.csv ros_cp3_30:$T/ros_cp330.csv" \
  CRUISE=$CRU GAIN=0.00500 ER=$PWD/$R/energy_runs_static6 OUTCSV=$PWD/$R/flight_energy_static6.csv \
  MAX_SIMS=3 bash scripts/run_energy_pair.sh 2>&1 | tail -4

say "FOLLOWON_STATIC6_DONE"

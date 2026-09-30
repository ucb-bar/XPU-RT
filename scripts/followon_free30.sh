#!/usr/bin/env bash
# Everything the figure still needs from the GPU after the rate-correct census, in order, unattended.
#
# Each stage waits for the one before because they share the simulator budget, and every flight still
# passes through the same admission (MAX_SIMS, NEED_MB) as the census. Nothing here renders a figure:
# that is a no-hardware step, run once these have landed.
#
# Two censuses stand behind the two forms of this figure. The rate-correct one (campaign_free30) gives
# each arm the gain its own command rate calls for; under it the baseline never reaches the first gate
# in any of its flights, so there is no crash-after-a-gate pair to draw and its result is a census, not
# a picture. The equal-gain one (campaign_free30_eq) flies both arms at 0.00500 -- the baseline's best
# gain of the four measured -- and is what the drawn pair, the scene census and the energy runs come
# from. Stage 0 flies the second; the first is already running when this starts.
set -u
cd "$(dirname "$0")/.."
R=results/codesign_feedback
say(){ echo "=== $(date +%H:%M:%S) $*"; }

C=$R/campaign_free30/campaign.csv
until [ -f "$C" ] && [ "$(( $(wc -l < "$C") - 1 ))" -ge 96 ]; do sleep 120; done
say "rate-correct census complete ($(( $(wc -l < "$C") - 1 )) rows)"

say "stage 0: the equal-gain census (baseline only; the scheduled arm's 48 flights carry over)"
MAX_SIMS=3 bash scripts/campaign_free30_eq.sh 2>&1 | tail -12

say "stage 1: the displayed pair, at the equal gain"
MAX_SIMS=3 CRUISES="${CRUISES:-1.4 1.2 1.6 1.0}" bash scripts/display_search_free30_eq.sh 2>&1 | tail -40
D=$R/campaign_free30_eq/display
SEED=$(grep -ho "PAIR_SEED=[0-9]*" $D/search_c*.log 2>/dev/null | head -1 | cut -d= -f2)
CRU=$(grep -l "PAIR_SEED" $D/search_c*.log 2>/dev/null | head -1 | sed 's/.*search_c//; s/\.log//')
if [ -z "${SEED:-}" ]; then echo "no pair accepted; stages 2 and 3 need one, stopping here"; exit 0; fi
say "pair: cruise $CRU seed $SEED"

say "stage 2: panel A's scene census, twelve seeds per arm on the scene the pair flies"
T=$PWD/$R/ctrl_traces
CELL=free30eq_l$SEED LAYOUT_SEED=$SEED CRUISE=$CRU MAX_SIMS=3 GAIN=0.00500 \
  ARMS="xpu:$T/xpu_p30free.csv:26.8:33.3 ros8:$T/ros_cp3n430.csv:30.1:33.3" \
  bash scripts/scene_runs_pair.sh 2>&1 | tail -6

say "stage 3: panel D's energy runs, cadence only"
CONDS="xpu_free30:$T/xpu_p30free.csv ros_cp3n4_30:$T/ros_cp3n430.csv" \
  CRUISE=$CRU GAIN=0.00500 ER=$PWD/$R/energy_runs_free30 OUTCSV=$PWD/$R/flight_energy_free30.csv \
  MAX_SIMS=3 bash scripts/run_energy_pair.sh 2>&1 | tail -4

say "FOLLOWON_FREE30_DONE"

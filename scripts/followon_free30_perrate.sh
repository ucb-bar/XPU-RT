#!/usr/bin/env bash
# The GPU stages for the per-rate-gain form of the 30 Hz figure, after the equal-gain form has its own.
#
# Both forms draw from censuses that are already flown by the time this starts: this one adds the
# displayed pair, panel A's scene census and panel D's energy runs for the form where each arm flies
# the gain its own command rate calls for. Its baseline does not reach the first gate -- in none of its
# census flights -- so the pair is accepted on that rather than on a crash after a gate, and the render
# declares it with --baseline-fails-before-first-gate.
#
# It waits for the equal-gain chain because the two share the simulator budget, not because it depends
# on its result.
set -u
cd "$(dirname "$0")/.."
R=results/codesign_feedback
say(){ echo "=== $(date +%H:%M:%S) $*"; }

until grep -q FOLLOWON_FREE30_DONE $R/followon_free30.log 2>/dev/null \
   || grep -q "no pair accepted" $R/followon_free30.log 2>/dev/null; do sleep 120; done
say "equal-gain chain finished; starting the per-rate form"

say "stage 1: the displayed pair, each arm at its own rate's gain"
MAX_SIMS=3 CRUISES="${CRUISES:-1.4 1.2 1.0 1.6}" bash scripts/display_search_free30.sh 2>&1 | tail -40
D=$R/campaign_free30/display
SEED=$(grep -ho "PAIR_SEED=[0-9]*" $D/search_c*.log 2>/dev/null | head -1 | cut -d= -f2)
CRU=$(grep -l "PAIR_SEED" $D/search_c*.log 2>/dev/null | head -1 | sed 's/.*search_c//; s/\.log//')
if [ -z "${SEED:-}" ]; then echo "no pair accepted; stages 2 and 3 need one, stopping here"; exit 0; fi
say "pair: cruise $CRU seed $SEED"

# Each arm keeps its own gain here, which scene_runs_pair.sh and run_energy_pair.sh take as one number
# for both arms, so each stage is run once per arm rather than once for the pair.
T=$PWD/$R/ctrl_traces
say "stage 2: panel A's scene census, twelve seeds per arm on the scene the pair flies"
for spec in "xpu:$T/xpu_p30free.csv:26.8:33.3:0.00500" "ros8:$T/ros_cp3n430.csv:30.1:33.3:0.01667"; do
  g=${spec##*:}; arm=${spec%:*}
  CELL=free30pr_l$SEED LAYOUT_SEED=$SEED CRUISE=$CRU MAX_SIMS=3 GAIN=$g \
    ARMS="$arm" bash scripts/scene_runs_pair.sh 2>&1 | tail -3
done

say "stage 3: panel D's energy runs, cadence only"
for spec in "xpu_free30:$T/xpu_p30free.csv:0.00500" "ros_cp3n4_30:$T/ros_cp3n430.csv:0.01667"; do
  g=${spec##*:}; cond=${spec%:*}
  CONDS="$cond" CRUISE=$CRU GAIN=$g ER=$PWD/$R/energy_runs_free30_perrate \
    OUTCSV=$PWD/$R/flight_energy_free30_perrate.csv MAX_SIMS=3 \
    bash scripts/run_energy_pair.sh 2>&1 | tail -3
done

say "FOLLOWON_FREE30_PERRATE_DONE"

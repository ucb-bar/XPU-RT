#!/usr/bin/env bash
# More attempts at the equal-gain pair, run only if the first search does not find one.
#
# The pair needs both arms to do their own thing in ONE scene: the scheduled arm to complete the course
# and the baseline to crash having entered it. Across the 48-flight equal-gain census the baseline
# crashes after one or two gates in 21, so the binding term is the scheduled arm, which completes in 3,
# and a seed is a pair only when the two land together.
#
# The census says where to spend the attempts. Per cruise speed, of twelve seeds each:
#
#     cruise   scheduled arm completes   baseline crashes after 1-2 gates
#      1.0            1                          2
#      1.2            0                          7
#      1.4            2                          7
#      1.6            0                          5
#
# 1.4 is where both are highest, so the first pass here is TWELVE MORE SEEDS at 1.4 and 1.2 rather than
# a new cruise speed: it draws more samples from the best cell instead of trading one arm's chances for
# the other's. Only after those does it try speeds outside the censused range -- 1.8, where the r30
# pair was drawn, and 0.8 below it, where the scheduled arm should complete more often and the baseline
# is likelier to run out of step budget rather than crash.
#
# It calls display_same_env.sh directly because the seed list here is not the census's twelve. It waits
# for both earlier chains: display_same_env.sh admits itself by polling the GPU rather than through
# gpu_admit.lock, so two searches running at once can each see room and start.
set -u
cd "$(dirname "$0")/.."
R=results/codesign_feedback; T=$PWD/$R/ctrl_traces
OD=$PWD/$R/campaign_free30_eq/display
say(){ echo "=== $(date +%H:%M:%S) $*"; }
done_(){ grep -qE "FOLLOWON_FREE30(_PERRATE)?_DONE|no pair accepted|stopping here" "$1" 2>/dev/null; }
found(){ grep -qho "PAIR_SEED=[0-9]*" $OD/search_c*.log 2>/dev/null; }

until done_ $R/followon_free30.log && done_ $R/followon_free30_perrate.log; do sleep 120; done
if found; then say "the equal-gain pair was already found; nothing to widen"; exit 0; fi

try() {   # try CRUISE SEEDS TAG
  local cru=$1 seeds=$2 tag=$3
  say "attempt: cruise $cru seeds $seeds"
  CRUISE=$cru SEEDS="$seeds" DENS=0.30 \
    XT=$T/xpu_p30free.csv RT=$T/ros_cp3n430.csv XLAT=26.8 RLAT=30.1 XHOLD=33.3 RHOLD=33.3 \
    XGAIN=0.00500 RGAIN=0.00500 \
    ROS_GATES='[12]' RENDER=0 MAX_SIMS=3 NEED_MB=10000 \
    OUTDIR="$OD/search_c$tag" bash scripts/display_same_env.sh > "$OD/search_c$tag.log" 2>&1
  grep -E "^  (xpu|ros) |PAIR" "$OD/search_c$tag.log" | tail -8
}

MORE="1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023"
BASE="1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011"
for a in "1.4:$MORE:1.4b" "1.2:$MORE:1.2b" "1.8:$BASE:1.8" "0.8:$BASE:0.8"; do
  IFS=: read -r cru seeds tag <<< "$a"
  try "$cru" "$seeds" "$tag"
  found && break
done

SEED=$(grep -ho "PAIR_SEED=[0-9]*" $OD/search_c*.log 2>/dev/null | head -1 | cut -d= -f2)
CRU=$(grep -l "PAIR_SEED" $OD/search_c*.log 2>/dev/null | head -1 | sed 's/.*search_c//; s/b\?\.log//')
if [ -z "${SEED:-}" ]; then echo "no pair over the censused seeds, twelve more at 1.4 and 1.2, or 1.8 and 0.8; stopping here"; exit 0; fi
say "pair: cruise $CRU seed $SEED"

say "panel A's scene census, twelve seeds per arm on the scene the pair flies"
CELL=free30eq_l$SEED LAYOUT_SEED=$SEED CRUISE=$CRU MAX_SIMS=3 GAIN=0.00500 \
  ARMS="xpu:$T/xpu_p30free.csv:26.8:33.3 ros8:$T/ros_cp3n430.csv:30.1:33.3" \
  bash scripts/scene_runs_pair.sh 2>&1 | tail -6

say "panel D's energy runs, cadence only"
CONDS="xpu_free30:$T/xpu_p30free.csv ros_cp3n4_30:$T/ros_cp3n430.csv" \
  CRUISE=$CRU GAIN=0.00500 ER=$PWD/$R/energy_runs_free30 OUTCSV=$PWD/$R/flight_energy_free30.csv \
  MAX_SIMS=3 bash scripts/run_energy_pair.sh 2>&1 | tail -4

say "FOLLOWON_FREE30_EQ_WIDER_DONE"

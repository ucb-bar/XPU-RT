#!/usr/bin/env bash
# The displayed pair for the 30 Hz camera figure (census: scripts/campaign_rate30.sh).
#
# The arms are the ones the census paired: XPU-RT's CP-SAT schedule at a 30 Hz camera (55.2 ms
# camera->control, control on its own 100 Hz slot) against ROS 2 running two model instances across
# both clusters (31.4 ms camera->goal, 0 frames late, control chained to the 30 Hz camera). The
# baseline is the wide one here -- it occupies seven harts and its pipeline is not backed up -- so
# the pair shows the command rate deciding the flight, not a starved chain.
#
# Like scripts/display_search_rate36.sh: display_same_env.sh flies ONE episode with layout_seed =
# seed, so a census cell is a different scene and a census outcome never carries over. A pair is
# kept only when, on the scene it flies, XPU-RT completes the course and the baseline crashes before
# the third gate, replaying the census hold of one 30 Hz camera period (33.3 ms) on both arms.
# ROS_GATES='[12]': the drawn baseline must have entered the course to be seen losing
# it -- scripts/verify_showdown_figure.py's _compare_display_pair accepts one or two gates.
#
# Seeds are ordered per cruise from the census (cells where the baseline crashed after one or two
# gates first, and where XPU-RT also completed ahead of those). That is a search order, not a
# selection: the pair is accepted on what the display flight itself does, and every attempt is logged.
#
#   scripts/display_search_rate30.sh          env CRUISES OD MAX_SIMS NEED_MB
#
# CRUISES and OD split the search over several GPU-admitted processes; each flight still waits for
# NEED_MB free and fewer than MAX_SIMS simulators before it starts.
set -u
cd "$(dirname "$0")/.."
OD=${OD:-$PWD/results/codesign_feedback/campaign_rate30/display30}; mkdir -p "$OD"
T=$PWD/results/codesign_feedback/ctrl_traces
ALL="1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011"
order() { local first="$1"; echo "$first $(for s in $ALL; do case " $first " in *" $s "*) ;; *) echo -n "$s ";; esac; done)"; }
# cells where the census saw the baseline crash after 1-2 gates; the two where XPU-RT also
# completed all four (1.4/1001 and 1.8/1001) lead their cruise.
declare -A FIRST=( [1.4]="1001 1000 1009 1010" [1.8]="1001 1006 1007 1008 1009 1011" \
                   [1.6]="1001 1004 1005 1008 1009 1010" [1.2]="1008" [1.0]="1001" )
for cru in ${CRUISES:-1.4 1.8 1.6 1.2 1.0}; do
  echo "=== $(date +%H:%M:%S) search cruise $cru"
  CRUISE=$cru SEEDS="$(order "${FIRST[$cru]:-}")" DENS=0.30 GAIN=0.0055 \
    XT=$T/xpu_a30_cpsat.csv RT=$T/ros_vanilla4x230.csv XLAT=55.2 RLAT=31.4 XHOLD=33.3 RHOLD=33.3 \
    ROS_GATES='[12]' RENDER=0 MAX_SIMS=${MAX_SIMS:-3} NEED_MB=${NEED_MB:-10000} \
    OUTDIR="$OD/search_c$cru" bash scripts/display_same_env.sh > "$OD/search_c$cru.log" 2>&1
  grep -E "^  (xpu|ros) |PAIR" "$OD/search_c$cru.log"
  grep -q PAIR_SEED "$OD/search_c$cru.log" && break
done
echo "DISPLAY_RATE30_SEARCH_DONE $(date +%H:%M:%S)"

#!/usr/bin/env bash
# The displayed pair for ROS 2 out of the box (scripts/campaign_submitted_config.sh).
#
# Like scripts/display_search_rate36.sh: the display script flies ONE episode with layout_seed = seed
# through record_sensor_demo.py, so a census cell is a different scene and a census outcome never
# carries over. A pair is kept only when, on the scene it flies, XPU-RT completes the course and the
# baseline crashes before the third gate. ROS_GATES='[12]': the pair the figure draws must show the
# baseline losing the course, so it has to have entered it -- scripts/verify_showdown_figure.py's
# _compare_display_pair accepts one or two gates, and a flight that never clears a gate is not a pair.
#
# Seeds are ordered per cruise with the census's completing seeds first: that is a search order, not a
# selection -- the pair is accepted on what the display flight itself does, and every attempt is logged.
#
#   scripts/display_search_submitted.sh        env CRUISES OD MAX_SIMS NEED_MB
#
# CRUISES and OD split the search over several GPU-admitted processes; each flight still waits for
# NEED_MB free and fewer than MAX_SIMS simulators before it starts.
set -u
cd "$(dirname "$0")/.."
OD=${OD:-$PWD/results/codesign_feedback/campaign_submitted/display}; mkdir -p "$OD"
T=$PWD/results/codesign_feedback/ctrl_traces
ALL="1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011"
order() { local first="$1"; echo "$first $(for s in $ALL; do case " $first " in *" $s "*) ;; *) echo -n "$s ";; esac; done)"; }
declare -A FIRST=( [1.4]="1001 1003 1009" [1.2]="1002 1011" [1.6]="1003 1007" [1.8]="1007" [1.0]="1006" )
for cru in ${CRUISES:-1.4 1.8 1.2 1.0 1.6}; do
  echo "=== $(date +%H:%M:%S) search cruise $cru"
  CRUISE=$cru SEEDS="$(order "${FIRST[$cru]:-}")" DENS=0.30 GAIN=0.0055 \
    XT=$T/xpu_a_cpsat_hard.csv RT=$T/ros_vanilla_c5045.csv XLAT=56.8 RLAT=265.9 XHOLD=0 RHOLD=0 \
    ROS_GATES='[12]' RENDER=0 MAX_SIMS=${MAX_SIMS:-3} NEED_MB=${NEED_MB:-10000} \
    OUTDIR="$OD/search_c$cru" bash scripts/display_same_env.sh > "$OD/search_c$cru.log" 2>&1
  grep -E "^  (xpu|ros) |PAIR" "$OD/search_c$cru.log"
  grep -q PAIR_SEED "$OD/search_c$cru.log" && break
done
echo "DISPLAY_SUBMITTED_SEARCH_DONE $(date +%H:%M:%S)"

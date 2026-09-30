#!/usr/bin/env bash
# The third-form figure's simulator work, one simulator (the third slot), GPU-room gated, nothing else touched:
#   1. the display pair in the tall scene at 1.2 m/s, both arms' latency replayed, the baseline clearing exactly two gates
#   2. twelve recorded runs per arm in that scene (panel K)
#   3. the calibrated-gain pair in the 1.7 m scene (people 1.7 m, gain 0.5 / control rate per arm) and its scene runs
#   4. the ROS 2 layouts that occupy all eight cores, flown with their latency (two YOLO nodes at 45 and 90 Hz, the
#      multi-threaded executor)
#   5. every render of the matrix (cells × tuned-ROS placements)
#   nohup bash scripts/queue_v3.sh > results/codesign_feedback/queue_v3.log 2>&1 &
set -u; cd "$(dirname "$0")/.."
WT=$PWD; R=$WT/results/codesign_feedback; T=$R/ctrl_traces
say(){ echo "=== $(date +%H:%M:%S) $*"; }
pair_seed(){ grep -h '^PAIR_SEED=' "$1" 2>/dev/null | tail -n 1 | cut -d= -f2; }

say "1. display pair, tall scene, 1.2 m/s, latency replayed, ROS 2 clears exactly two gates"
CRUISE=1.2 XLAT=56.8 RLAT=242 ROS_GATES=2 SEEDS="${SEEDS_TALL:-1008 1005 1003 1001 1002 1007 1009 1011 1000 1004 1006 1010}" \
  OUTDIR=$R/campaign_v2/display_v3 RENDER=0 bash scripts/display_same_env.sh 2>&1 | tee $R/campaign_v2/display_v3.log | grep -E 'xpu |ros |PAIR|no seed'
S=$(pair_seed $R/campaign_v2/display_v3.log)
if [ -z "$S" ]; then say "no tall pair at 1.2 m/s — trying 1.8 m/s (seed 1005 first)"
  CRUISE=1.8 XLAT=56.8 RLAT=242 ROS_GATES=2 SEEDS="1005 1000 1003 1009 1001 1002 1004 1006 1007 1008 1010 1011" \
    OUTDIR=$R/campaign_v2/display_v3 RENDER=0 bash scripts/display_same_env.sh 2>&1 | tee -a $R/campaign_v2/display_v3.log | grep -E 'xpu |ros |PAIR|no seed'
  S=$(pair_seed $R/campaign_v2/display_v3.log); TALL_CRU=1.8
else TALL_CRU=1.2; fi
if [ -n "$S" ]; then
  say "2. scene runs, tall scene seed $S at $TALL_CRU m/s"
  CELL=tall$S LAYOUT_SEED=$S CRUISE=$TALL_CRU bash scripts/scene_runs.sh
  say "render tall$S"
  CELL=tall1008 XPU_DIR=$R/campaign_v2/display_v3/xpu_s${S}_figdata ROS_DIR=$R/campaign_v2/display_v3/ros_s${S}_figdata SCENE_RECORDS=$R/campaign_scene/tall$S \
    DISPLAY_CRUISE=$TALL_CRU MAIN=1 bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|wrote|DONE'
fi

say "3. calibrated-gain pair, 1.7 m people, 1.2 m/s, gain 0.5 / control rate per arm"
WAREHOUSE_PERSON_H=1.7 CRUISE=1.2 XGAIN=0.0052 RGAIN=0.01277 XLAT=0 RLAT=0 SEEDS="${SEEDS_CAL:-1004 1001 1006 1010 1000 1002 1003 1005 1007 1008 1009 1011}" \
  OUTDIR=$R/campaign_v2/display_same_cal17 RENDER=0 bash scripts/display_same_env.sh 2>&1 | tee $R/campaign_v2/display_same_cal17.log | grep -E 'xpu |ros |PAIR|no seed'
C=$(pair_seed $R/campaign_v2/display_same_cal17.log)
if [ -n "$C" ]; then
  say "scene runs, calibrated scene seed $C"
  CELL=cal17 LAYOUT_SEED=$C CRUISE=1.2 XGAIN=0.0052 RGAIN=0.01277 XLAT=0 RLAT=0 GLAT=0 PERSON_H=1.7 bash scripts/scene_runs.sh
  say "render cal17"
  CELL=cal17 XPU_DIR=$R/campaign_v2/display_same_cal17/xpu_s${C}_figdata ROS_DIR=$R/campaign_v2/display_same_cal17/ros_s${C}_figdata SCENE_RECORDS=$R/campaign_scene/cal17 \
    bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|wrote|DONE'
fi

say "4. ROS 2 on all eight cores, flown with its latency (two YOLO nodes at 45 and 90 Hz, multi-threaded executor)"
# latency = e2e_goal_med_ms of the layout's runs; hold = 1000 / goals delivered per warm second (90 Hz two-YOLO 51/s, multi 19.9/s;
# the 45 Hz two-YOLO layout delivers 40.8/s, every goal refreshed as for the other 45 Hz arms)
OUT=$R/campaign_percep ARMS="ros_vanilla4x2:$T/ros_vanilla4x245.csv:37.0:0 ros_vanilla4x2_90:$T/ros_vanilla4x290.csv:254.3:${HOLD_4X2_90:-19.6} ros_multi:$T/ros_multi45.csv:264.7:${HOLD_MULTI:-50.1}" \
  SPEEDS="1.2 1.0 1.4 1.8 1.6" bash scripts/campaign_percep.sh 2>&1 | grep -E '^\[SWEEP\]|^=== ' | cut -c1-140

say "5. final renders"
[ -n "${S:-}" ] && CELL=tall1008 XPU_DIR=$R/campaign_v2/display_v3/xpu_s${S}_figdata ROS_DIR=$R/campaign_v2/display_v3/ros_s${S}_figdata SCENE_RECORDS=$R/campaign_scene/tall$S DISPLAY_CRUISE=$TALL_CRU MAIN=1 bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|DONE'
CELL=tall1005 bash scripts/render_showdown_v3.sh 2>&1 | grep -E 'FAIL|DONE'
echo QUEUE_V3_DONE

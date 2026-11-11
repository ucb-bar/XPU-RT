#!/usr/bin/env bash
# PHASE 2: google at the ~40 ms control grid (27 Hz), SIM-ONLY.
#
# Uses the ONLY structure in which refinement is neutral: a target-accumulating,
# PLANNER-FREE arm controller (measured drawer 25.0% at 3 Hz vs 29.2% at 27 Hz --
# a 9x control-rate change moves it within noise). This ABANDONS SIMPLER's real2sim
# calibration, which is acceptable because the comparison is sim-internal: every arm
# in this sweep uses the SAME controller, and the native-rate run is the control.
#
# Originals are untouched: this writes to runs_g40/, never runs/.
#
# Argument: "KEY:TASK:ARM:RATE" with RATE in {native,fine27}.
source /home/ubuntu/simpler/sim_eval/roselite/env.sh
cd /home/ubuntu/simpler/sim_eval/roselite/finegrain
IFS=: read -r KEY TASK ARM RATE <<< "$1"
case "$TASK" in
  coke)   T=google_robot_pick_coke_can ;;
  drawer) T=google_robot_close_drawer ;;
  *) echo "bad task $TASK (google only)"; exit 2 ;;
esac
case "$ARM" in
  lat0)       LAT=0;     PER=auto   ;;
  pipe110fix) LAT=385.1; PER=111.4  ;;
  p105w300)   LAT=258.7; PER=124.8  ;;
  p130w275)   LAT=272.3; PER=130.1  ;;
  p150w300)   LAT=281.6; PER=150.4  ;;
  pipe200fix) LAT=260.5; PER=219.2  ;;
  serial283)  LAT=283.4; PER=283.4  ;;
  fp32_555)   LAT=555.0; PER=555.0  ;;
  cpu685)     LAT=684.8; PER=684.8  ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
CM="arm_pd_ee_target_delta_pose_align_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
case "$RATE" in
  native) ACT="--actuation native --tick-hz 3" ;;
  fine27) ACT="--actuation fine   --tick-hz 27" ;;
  *) echo "bad rate $RATE"; exit 2 ;;
esac
EXTRA=""; [ "$PER" != auto ] && EXTRA="--issue-period-ms $PER"
OUT=/home/ubuntu/simpler/sim_eval/roselite/finegrain/runs_g40/${KEY}
mkdir -p "$(dirname "$OUT")"
# Idempotent: a relaunch must not redo completed work.
if [ -f "$OUT/summary.json" ]; then echo "skip ${KEY}"; exit 0; fi
python finegrain_eval.py --task "$T" --latency-ms "$LAT" $EXTRA $ACT \
       --control-mode "$CM" --init-rng "${RNG:-100}" --n 24 --out "$OUT" \
       > /home/ubuntu/simpler/logs/g40_${KEY}.log 2>&1
echo "done ${KEY} rc=$? : $(grep -h 'SUCCESS RATE' /home/ubuntu/simpler/logs/g40_${KEY}.log | tail -1)"

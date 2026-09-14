#!/usr/bin/env bash
# SIM-ONLY criterion: real2sim fidelity is not required, so a controller swap is allowed.
# The bar is now INTERNAL VALIDITY: the config must (a) give a workable success rate at
# its NATIVE rate, and (b) give the SAME rate at 27 Hz, so that refinement is neutral and
# arms can be compared within the config. Real2sim comparability is explicitly abandoned.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
source /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/env.sh
cd "$HERE"; source parity/configs.sh
N=${N:-12}; T=${T:-google_robot_pick_coke_can}
short=$(echo "$T" | sed 's/google_robot_//')
GRIP_NP=gripper_pd_joint_target_delta_pos
# arm variants, all target-accumulating (required for refinement to preserve displacement)
declare -A CM=(
  [lin_npg]="arm_pd_ee_target_delta_pose_align_interpolate_${GRIP_NP}"
  [lin_pg]="arm_pd_ee_target_delta_pose_align_interpolate_${GRIP}"
)
run () {
  local name=$1
  local act=$2
  local tick=$3
  local mode=$4
  local key="${short}_${name}_n${N}"
  [ -f "parity/runs/$key/summary.json" ] && { echo "skip $key"; return; }
  python finegrain_eval.py --task "$T" --latency-ms 0 --init-rng 80 --n "$N" \
      --actuation "$act" --tick-hz "$tick" --control-mode "${CM[$mode]}" \
      --out "parity/runs/$key" > "parity/logs/${key}.log" 2>&1
  echo "$key rc=$? : $(grep -h 'SUCCESS RATE' parity/logs/${key}.log | tail -1)"
}
run I_lin_npg_native native 3  lin_npg
run J_lin_pg_native  native 3  lin_pg

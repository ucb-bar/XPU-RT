#!/usr/bin/env bash
# Isolate PATH TRUNCATION as a mechanism, keeping SIMPLER's validated controller.
#
# The registered google arm controller plans a trajectory and executes only
# sim_freq/control_freq steps of it per period: 171 natively, but 19 at 27 Hz.
# Scaling sim_freq by the same 9x restores 171 steps per period, so the planner
# traverses the same FRACTION of its plan as it does natively.
#
#   F0  registered, native 3 Hz, sim_freq 4617  -- must reproduce A, else raising
#                                                  sim_freq is not itself neutral
#   F   registered, fine  27 Hz, sim_freq 4617  -- the actual test
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
source /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/env.sh
cd "$HERE"
source parity/configs.sh
N=${N:-12}; RNG=${RNG:-80}; T=${T:-google_robot_pick_coke_can}
short=$(echo "$T" | sed 's/google_robot_//')
run () {
  local name=$1
  local act=$2
  local tick=$3
  local extra=$4
  local key="${short}_${name}_n${N}"
  [ -f "parity/runs/$key/summary.json" ] && { echo "skip $key"; return; }
  # shellcheck disable=SC2086
  python finegrain_eval.py --task "$T" --latency-ms 0 --init-rng "$RNG" --n "$N" \
      --actuation "$act" --tick-hz "$tick" --control-mode "$CM_STOCK" --sim-freq 4617 \
      $extra --out "parity/runs/$key" > "parity/logs/${key}.log" 2>&1
  echo "$key rc=$? : $(grep -h 'SUCCESS RATE' parity/logs/${key}.log | tail -1)"
}
run F0_stock_native_sf9  native 3  ""
run F_stock_fine27_sf9   fine   27 "--allow-lag-discard"

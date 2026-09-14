#!/usr/bin/env bash
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
source /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/env.sh
cd "$HERE"; source parity/configs.sh
N=${N:-12}; T=google_robot_pick_coke_can
run () {
  local name=$1
  local act=$2
  local tick=$3
  local key="pick_coke_can_${name}_n${N}"
  [ -f "parity/runs/$key/summary.json" ] && { echo "skip $key"; return; }
  python finegrain_eval.py --task "$T" --latency-ms 0 --init-rng 80 --n "$N" \
      --actuation "$act" --tick-hz "$tick" --control-mode "$CM_TGT_NPG" \
      --out "parity/runs/$key" > "parity/logs/${key}.log" 2>&1
  echo "$key rc=$? : $(grep -h 'SUCCESS RATE' parity/logs/${key}.log | tail -1)"
}
run G_tgt_npgrip_native native 3
run H_tgt_npgrip_fine27 fine   27

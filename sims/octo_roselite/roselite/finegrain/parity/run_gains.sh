#!/usr/bin/env bash
# Planner-free + raised PD gains. Planner-free is the ONLY structure where refinement is
# neutral (drawer 25.0% -> 29.2%); it fails the GRASP because it undershoots. If higher
# gains recover coke at the NATIVE rate, the same config at 27 Hz is a valid 40 ms google.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
source /scratch2/dima/misc_sw/octo_work/sim_eval/roselite/env.sh
cd "$HERE"; source parity/configs.sh
N=${N:-12}; T=google_robot_pick_coke_can
for G in 3 10; do
  for spec in "native:3" "fine:27"; do
    act=${spec%%:*}; tick=${spec##*:}
    key="pick_coke_can_G${G}_${act}${tick}_n${N}"
    [ -f "parity/runs/$key/summary.json" ] && { echo "skip $key"; continue; }
    python finegrain_eval.py --task "$T" --latency-ms 0 --init-rng 80 --n "$N" \
        --actuation "$act" --tick-hz "$tick" --control-mode "$CM_TGT" \
        --planner-limit-scale 1 --arm-gain-scale "$G" \
        --out "parity/runs/$key" > "parity/logs/${key}.log" 2>&1
    echo "$key : $(grep -h 'SUCCESS RATE' parity/logs/${key}.log | tail -1)"
  done
done
echo GAINS_DONE

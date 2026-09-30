#!/usr/bin/env bash
# Board runs for the tables whose clamp step had failed: 60 / 90 / 30 Hz chain and the shard-aware spec.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/solver_v2
for pair in "a60 wh_chain60_solve_500" "a90 wh_chain90_solve_500" "a30 wh_chain30_solve_500" "ash wh_chain45_shard_solve"; do
  set -- $pair; echo "=== $(date +%H:%M:%S) board $1"; bash scripts/board_stage2.sh $1 $2 > $L/board_stage2_$1.log 2>&1; tail -n 2 $L/board_stage2_$1.log
done
echo BOARD_RERUN_DONE

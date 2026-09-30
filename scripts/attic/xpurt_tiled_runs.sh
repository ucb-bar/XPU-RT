#!/usr/bin/env bash
# XPU-RT on the tiled CP-SAT schedules, inserted between the chained ROS arms and the replicates.
set -u; cd "$(dirname "$0")/.."
say() { echo "=== $(date +%H:%M:%S) $*"; }
while ! grep -q "=== .* replicates" results/codesign_feedback/ros_traced/board_queue3.log 2>/dev/null; do sleep 10; done
pkill -f "^bash scripts/board_queue[3].sh"; sleep 1; pkill -f "^bash scripts/ros_traced_matri[x].sh"; sleep 1
ssh k1 'pkill -f cpu_sampler; pkill -f ros_mb_chain_traced; true'
say "tiled 25 Hz";  scripts/run_xpurt_long.sh schedules/tiled_coupled_25hz.json tiled25 3 2>&1 | grep -E "^===|trace rows|DONE|Error"
say "tiled 45 Hz";  scripts/run_xpurt_long.sh schedules/tiled_coupled_45hz.json tiled45 3 2>&1 | grep -E "^===|trace rows|DONE|Error"
say "tiled 25 Hz FIFO"; MODELBLASTER_K1_RT_PRIORITY=80 scripts/run_xpurt_long.sh schedules/tiled_coupled_25hz.json tiled25 1 2>&1 | grep -E "^===|trace rows|DONE"
say "resume replicates"
for rep in 2 3; do for arm in cship cspin cp3 spin nproc yproc p3 ship multi; do
  RATES="15 25 45" scripts/ros_traced_matrix.sh $arm $rep 2>&1 | grep -E "MATRIX_DONE|error|incomplete"
done; done
say "TILED_RUNS_DONE"

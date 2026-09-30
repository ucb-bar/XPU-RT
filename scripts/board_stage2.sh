#!/usr/bin/env bash
# Execute the solver tables of one spec on the K1, interleaved, three runs each: clamp widths to the
# codegen contract, check for double-booked harts, then run.  scripts/board_stage2.sh <tag> <spec>
set -u; cd "$(dirname "$0")/.."
TAG="${1:?tag}"; SPEC="${2:?spec}"
while ! grep -q "STAGE2_DONE" results/codesign_feedback/solver_v2/stage2_${TAG}.log 2>/dev/null; do sleep 60; done
say() { echo "=== $(date +%H:%M:%S) $*"; }
IRS="yolov8_nano_64x96:ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/graph.json fused_full:ModelBlaster/build/k1_xpurt/fused_full/int8/graph.json mlp_control:ModelBlaster/build/k1_xpurt/mlp_control/int8/graph.json ffn_block:ModelBlaster/build/k1_xpurt/ffn_block/int8/graph.json dronet:ModelBlaster/build/k1_xpurt/dronet/int8/graph.json"
ARMS=""
for solver in greedy cpsat_hard cpsat_soft; do
  f=schedules/fig_${TAG}_${solver}.json; [ -f $f ] || { say "no $f"; continue; }
  say "clamp + feasibility $solver"
  .venv/bin/python scripts/clamp_schedule_widths.py $f $IRS --out schedules/fig_${TAG}_${solver}_clamped.json 2>&1 | tail -n 2
  .venv/bin/python scripts/check_schedule_feasibility.py --schedule schedules/fig_${TAG}_${solver}_clamped.json 2>&1 | tail -n 2; echo "  feasibility rc=${PIPESTATUS[0]}"
  ARMS="$ARMS $solver"
done
# one board user at a time on the host side too: every board_stage2 holds this lock over its runs
exec 9>results/codesign_feedback/board.lock; flock 9
# wait for the board to be free of other runs (the per-core sampler is not a run; a sampler left behind is ended)
ssh k1 "pkill -x cpu_sampler" >/dev/null 2>&1 || true
while ssh k1 "ps aux | grep -E 'ros_mb_chain|xpurt' | grep -v -E 'grep|cpu_sampler' | wc -l" | grep -qv '^0$'; do sleep 60; done
for k in 1 2 3; do for solver in $ARMS; do
  say "board ${TAG}_${solver} run $k"
  scripts/run_xpurt_long.sh schedules/fig_${TAG}_${solver}_clamped.json ${TAG}${solver}r$k 1 2>&1 | grep -E "trace rows|re-pulled|FATAL|Error"
done; done
say "BOARD_STAGE2_DONE $TAG"

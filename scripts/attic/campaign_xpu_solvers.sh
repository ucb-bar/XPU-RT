#!/usr/bin/env bash
# The two solvers' tables in flight: once spec A's board runs are in, replay each solver's measured
# control cadence (first run) across cruise speeds.
set -u; cd "$(dirname "$0")/.."
while ! grep -q BOARD_STAGE2_DONE results/codesign_feedback/solver_v2/board_stage2_a.log 2>/dev/null; do sleep 120; done
ARMS=""
for solver in cpsat_soft greedy cpsat_hard; do
  t=results/codesign_feedback/xpurt_long/trace_a${solver}r1_other_run1.csv; [ -f $t ] || continue
  .venv/bin/python scripts/ctrl_trace_from_board.py $t --out results/codesign_feedback/ctrl_traces/xpu_a_${solver}.csv || continue
  ARMS="$ARMS xpu_${solver}:results/codesign_feedback/ctrl_traces/xpu_a_${solver}.csv"
done
ARMS="$ARMS" SPEEDS="${SPEEDS:-0.8 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.8 2.0}" bash scripts/campaign_v2.sh
echo XPU_SOLVERS_DONE

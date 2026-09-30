#!/usr/bin/env bash
# 120 and 150 Hz cameras on the board from the 200 ms hyperperiod tables (the half-second CP-SAT
# solve at 120 Hz returns no solution in 3000 s; the 200 ms certificate is feasible): both solvers'
# certificate tables, clamped and checked, three interleaved runs each.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/solver_v2
stage() { # <hz>
  local spec=wh_chain$1_solve_h200 tag=a$1h
  grep -qE "cpsat status=(FEASIBLE|OPTIMAL)" $L/${spec}_cpsat.log || { echo "$spec: no feasible certificate"; return 1; }
  cp schedules/scheduled_${spec}_cpsat_profiled.json schedules/fig_${tag}_cpsat_hard.json
  cp schedules/scheduled_${spec}_greedy_periodic_profiled.json schedules/fig_${tag}_greedy.json
  { echo "=== $(date +%H:%M:%S) tables from the certificate solves ($spec)"; echo "=== STAGE2_DONE $spec"; } > $L/stage2_${tag}.log
  bash scripts/board_stage2.sh $tag $spec > $L/board_stage2_${tag}.log 2>&1; tail -n 2 $L/board_stage2_${tag}.log
}
stage 120
until grep -q "cpsat status=" $L/wh_chain150_solve_h200_cpsat.log 2>/dev/null; do sleep 60; done
OLD=$(pgrep -f "chain_rates_high[.]sh" | head -n 1); [ -n "$OLD" ] && { pkill -f "wh_chain150_solve_500" ; kill $OLD; echo "stopped the half-second chain ($OLD)"; }
stage 150
echo CHAIN_RATES_HIGH_H200_DONE

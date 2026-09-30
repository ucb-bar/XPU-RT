#!/usr/bin/env bash
# Spec B (90 Hz camera + heavier stack): once its certificate is in and spec A's solves are done,
# solve the one-second table with both solvers and execute it after spec A's board runs.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/solver_v2
while ! grep -q CERT_DONE $L/certificates.log 2>/dev/null; do sleep 60; done
ST=$(grep -E 'cpsat status=' $L/wh_chain90_rich_solve_h100_cpsat.log | tail -n 1)
echo "=== $(date +%H:%M:%S) spec B certificate: $ST"
echo "$ST" | grep -qE "OPTIMAL|FEASIBLE" || { echo "spec B certificate not feasible; widen windows before solving 1 s"; echo STAGE2_B_SKIPPED; exit 0; }
while ! grep -q STAGE2_DONE $L/stage2_a.log 2>/dev/null; do sleep 60; done
bash scripts/solve_stage2.sh wh_chain90_rich_solve b > $L/stage2_b.log 2>&1
while ! grep -q BOARD_STAGE2_DONE $L/board_stage2_a.log 2>/dev/null; do sleep 60; done
bash scripts/board_stage2.sh b wh_chain90_rich_solve > $L/board_stage2_b.log 2>&1
echo "STAGE2_B_CHAIN_DONE"

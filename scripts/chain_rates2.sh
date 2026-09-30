#!/usr/bin/env bash
# Half-second tables for spec B and the chain spec at 60/90/30 Hz: certificate where missing,
# then both solvers' tables and their board runs.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/solver_v2
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1 XPURT_CPSAT_WORKERS=8
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json
run_pair() { # <spec500> <tag>
  bash scripts/solve_stage2_hard.sh $1 $2 3000 > $L/stage2_$2.log 2>&1; cat $L/stage2_$2.log
  bash scripts/board_stage2.sh $2 $1 > $L/board_stage2_$2.log 2>&1; tail -n 3 $L/board_stage2_$2.log
}
run_pair wh_chain90_rich_solve_500 b5
for hz in 60 90 30; do
  spec=wh_chain${hz}_solve
  if ! grep -qE "OPTIMAL|FEASIBLE" $L/${spec}_h200_cpsat.log 2>/dev/null; then
    echo "=== $(date +%H:%M:%S) $spec certificate"
    .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/${spec}_h200.json --solver greedy_periodic --max-periodic-iters 1 --use-profiled --board-calibration $CAL --random-seed 42 > $L/${spec}_h200_greedy.log 2>&1; echo "  greedy: $(grep -E 'periodic window misses' $L/${spec}_h200_greedy.log | tail -n 1)"
    CPSAT_LOG=1 .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/${spec}_h200.json --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 900 --use-profiled --board-calibration $CAL > $L/${spec}_h200_cpsat.log 2>&1; echo "  cpsat: $(grep -E 'cpsat status=' $L/${spec}_h200_cpsat.log | tail -n 1)"
  fi
  grep -qE "OPTIMAL|FEASIBLE" $L/${spec}_h200_cpsat.log || { echo "  $spec: certificate not feasible, skipped"; continue; }
  run_pair ${spec}_500 a${hz}
done
echo CHAIN_RATES2_DONE

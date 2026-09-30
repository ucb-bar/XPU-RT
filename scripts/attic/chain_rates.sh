#!/usr/bin/env bash
# The chain spec at other camera rates: certificate (both solvers, 200 ms), then the one-second
# tables and their board runs, after spec B's chain has finished with the board.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/solver_v2
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1 XPURT_CPSAT_WORKERS=8
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json
while ! grep -qE "STAGE2_B_CHAIN_DONE|STAGE2_B_SKIPPED" $L/stage2_b_chain.log 2>/dev/null; do sleep 120; done
for hz in 60 90 30; do
  spec=wh_chain${hz}_solve; tag=a${hz}
  echo "=== $(date +%H:%M:%S) $spec certificate"
  .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/${spec}_h200.json --solver greedy_periodic --max-periodic-iters 1 --use-profiled --board-calibration $CAL --random-seed 42 > $L/${spec}_h200_greedy.log 2>&1; echo "  greedy: $(grep -E 'periodic window misses' $L/${spec}_h200_greedy.log | tail -n 1)"
  CPSAT_LOG=1 .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/${spec}_h200.json --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 900 --use-profiled --board-calibration $CAL > $L/${spec}_h200_cpsat.log 2>&1; ST=$(grep -E 'cpsat status=' $L/${spec}_h200_cpsat.log | tail -n 1); echo "  cpsat: $ST"
  echo "$ST" | grep -qE "OPTIMAL|FEASIBLE" || { echo "  $spec: certificate not feasible, skipped"; continue; }
  bash scripts/solve_stage2.sh $spec $tag > $L/stage2_${tag}.log 2>&1
  bash scripts/board_stage2.sh $tag $spec > $L/board_stage2_${tag}.log 2>&1
done
echo CHAIN_RATES_DONE

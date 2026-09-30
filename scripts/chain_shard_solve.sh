#!/usr/bin/env bash
# Spec A with YOLO allowed to shard, costed from the board-measured shard profile: certificate,
# one-second tables from both solvers, then the board once the other chain has released it.
set -u; cd "$(dirname "$0")/.."
L=results/codesign_feedback/solver_v2
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1 XPURT_CPSAT_WORKERS=8
CAL=results/codesign_feedback/k1_board_calibration_yolo100.json      # the shard rows are board-measured already
spec=wh_chain45_shard_solve
echo "=== $(date +%H:%M:%S) $spec certificate"
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/${spec}_h200.json --solver greedy_periodic --max-periodic-iters 1 --use-profiled --board-calibration $CAL --random-seed 42 > $L/${spec}_h200_greedy.log 2>&1; echo "  greedy: $(grep -E 'periodic window misses' $L/${spec}_h200_greedy.log | tail -n 1)"
CPSAT_LOG=1 .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/${spec}_h200.json --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 900 --use-profiled --board-calibration $CAL > $L/${spec}_h200_cpsat.log 2>&1; echo "  cpsat: $(grep -E 'cpsat status=|periodic window misses' $L/${spec}_h200_cpsat.log | tail -n 2 | tr '\n' ' ')"
grep -qE "OPTIMAL|FEASIBLE" $L/${spec}_h200_cpsat.log || { echo "certificate not feasible"; echo SHARD_CHAIN_DONE; exit 0; }
sed -i "s|^CAL=.*|CAL=$CAL|" /dev/null
CALX=$CAL bash -c 'sed "s|CAL=results/codesign_feedback/k1_board_calibration_yolo110.json|CAL=$CALX|" scripts/solve_stage2_hard.sh > scripts/solve_stage2_hard_cal100.sh'; chmod +x scripts/solve_stage2_hard_cal100.sh
bash scripts/solve_stage2_hard_cal100.sh $spec ash 3000 > $L/stage2_ash.log 2>&1; cat $L/stage2_ash.log
while ! grep -q CHAIN_RATES2_DONE $L/chain_rates2.log 2>/dev/null; do sleep 300; done
bash scripts/board_stage2.sh ash $spec > $L/board_stage2_ash.log 2>&1; tail -n 3 $L/board_stage2_ash.log
echo SHARD_CHAIN_DONE

#!/usr/bin/env bash
# Hyperperiod certificates: the hard-window CP-SAT solve must be feasible on each spec, and greedy's
# predicted window misses on the same spec are recorded before any board run.
set -u; cd "$(dirname "$0")/.."
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1 XPURT_CPSAT_WORKERS=8
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json; OUT=results/codesign_feedback/solver_v2
for spec in wh_chain45_solve_h200 wh_chain90_rich_solve_h100; do
  echo "=== $(date +%H:%M:%S) $spec greedy"
  .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$spec.json --solver greedy_periodic --max-periodic-iters 1 --use-profiled --board-calibration $CAL --random-seed 42 > $OUT/${spec}_greedy.log 2>&1; echo "  rc=$? $(grep -E 'periodic window misses|makespan=' $OUT/${spec}_greedy.log | tail -n 1)"
  echo "=== $(date +%H:%M:%S) $spec cpsat (hard windows)"
  CPSAT_LOG=1 .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$spec.json --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 900 --use-profiled --board-calibration $CAL > $OUT/${spec}_cpsat.log 2>&1; echo "  rc=$? $(grep -E 'cpsat status=' $OUT/${spec}_cpsat.log | tail -n 1)"
done
echo CERT_DONE

#!/usr/bin/env bash
# The 45 Hz chain certificate under the two other YOLO costings (x1.00, x1.20): does the solver
# gap depend on the calibration table? Host only; each CP-SAT solve is bounded at 900 s.
set -u; cd "$(dirname "$0")/.."
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1 XPURT_CPSAT_WORKERS=6
OUT=results/codesign_feedback/solver_v2
for m in 100 120; do
  CAL=results/codesign_feedback/k1_board_calibration_yolo${m}.json; spec=wh_chain45_solve_h200
  echo "=== $(date +%H:%M:%S) yolo x$m greedy"
  .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$spec.json --solver greedy_periodic --max-periodic-iters 1 --use-profiled --board-calibration $CAL --random-seed 42 > $OUT/${spec}_yolo${m}_greedy.log 2>&1; echo "  $(grep -E 'periodic window misses' $OUT/${spec}_yolo${m}_greedy.log | tail -n 1)"
  echo "=== $(date +%H:%M:%S) yolo x$m cpsat"
  CPSAT_LOG=1 .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$spec.json --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 900 --use-profiled --board-calibration $CAL > $OUT/${spec}_yolo${m}_cpsat.log 2>&1; echo "  $(grep -E 'cpsat status=|periodic window misses' $OUT/${spec}_yolo${m}_cpsat.log | tail -n 2 | tr '\n' ' ')"
done
echo SENS_DONE

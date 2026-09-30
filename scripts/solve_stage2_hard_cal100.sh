#!/usr/bin/env bash
# The table both solvers produce for a spec, hard-window CP-SAT only. scripts/solve_stage2_hard.sh <spec> <tag> [limit_s]
set -u; cd "$(dirname "$0")/.."
SPEC="${1:?spec}"; TAG="${2:?tag}"; LIM="${3:-3000}"
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1 XPURT_CPSAT_WORKERS=8
CAL=results/codesign_feedback/k1_board_calibration_yolo100.json; OUT=results/codesign_feedback/solver_v2
echo "=== $(date +%H:%M:%S) $SPEC greedy"
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$SPEC.json --solver greedy_periodic --max-periodic-iters 1 --use-profiled --board-calibration $CAL --random-seed 42 > $OUT/${SPEC}_greedy.log 2>&1
cp schedules/scheduled_${SPEC}_greedy_periodic_profiled.json schedules/fig_${TAG}_greedy.json; cp schedules/scheduled_${SPEC}_greedy_periodic_profiled_metrics.json schedules/fig_${TAG}_greedy_metrics.json
echo "  $(grep -E 'periodic window misses' $OUT/${SPEC}_greedy.log | tail -n 1)"
echo "=== $(date +%H:%M:%S) $SPEC cpsat hard (limit $LIM s)"
CPSAT_LOG=1 .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$SPEC.json --solver cpsat --max-periodic-iters 1 --cpsat-time-limit $LIM --use-profiled --board-calibration $CAL > $OUT/${SPEC}_cpsat_hard.log 2>&1
echo "  hard rc=$? $(grep -E 'cpsat status=' $OUT/${SPEC}_cpsat_hard.log | tail -n 1)"
[ -f schedules/scheduled_${SPEC}_cpsat_profiled.json ] && cp schedules/scheduled_${SPEC}_cpsat_profiled.json schedules/fig_${TAG}_cpsat_hard.json && cp schedules/scheduled_${SPEC}_cpsat_profiled_metrics.json schedules/fig_${TAG}_cpsat_hard_metrics.json
echo "=== $(date +%H:%M:%S) STAGE2_DONE $SPEC"

#!/usr/bin/env bash
# The one-second tables both solvers produce for a spec; the greedy one first (seconds), then
# CP-SAT with hard windows and CP-SAT with soft windows (lexicographic misses -> lateness ->
# makespan, HEFT warm start) side by side. Outputs are copied to fig_<tag>_<solver>.json at once,
# because both CP-SAT paths write the same file name.
set -u; cd "$(dirname "$0")/.."
SPEC="${1:?spec name}"; TAG="${2:?tag}"
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json; OUT=results/codesign_feedback/solver_v2
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "$SPEC greedy"
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$SPEC.json --solver greedy_periodic --max-periodic-iters 1 --use-profiled --board-calibration $CAL --random-seed 42 > $OUT/${SPEC}_greedy.log 2>&1
cp schedules/scheduled_${SPEC}_greedy_periodic_profiled.json schedules/fig_${TAG}_greedy.json; cp schedules/scheduled_${SPEC}_greedy_periodic_profiled_metrics.json schedules/fig_${TAG}_greedy_metrics.json
echo "  $(grep -E 'periodic window misses' $OUT/${SPEC}_greedy.log | tail -n 1)"
say "$SPEC cpsat hard (background) + cpsat soft"
( XPURT_CPSAT_WORKERS=8 CPSAT_LOG=1 .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$SPEC.json --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 2400 --use-profiled --board-calibration $CAL > $OUT/${SPEC}_cpsat_hard.log 2>&1;
  echo "  hard rc=$? $(grep -E 'cpsat status=' $OUT/${SPEC}_cpsat_hard.log | tail -n 1)";
  [ -f schedules/scheduled_${SPEC}_cpsat_profiled.json ] && cp schedules/scheduled_${SPEC}_cpsat_profiled.json schedules/fig_${TAG}_cpsat_hard.json && cp schedules/scheduled_${SPEC}_cpsat_profiled_metrics.json schedules/fig_${TAG}_cpsat_hard_metrics.json; echo HARD_DONE ) &
sleep 20   # let the hard path claim its output name first; the soft path writes into a private copy of the spec
python3 -c "import json,sys; s=json.load(open('data/toplevel/$SPEC.json')); json.dump(s, open('data/toplevel/${SPEC}_soft.json','w'), indent=2)"
XPURT_CPSAT_WORKERS=8 .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/${SPEC}_soft.json --solver milp --scheduler cpsat --cpsat-time-limit 3600 --use-profiled --board-calibration $CAL --solver-verbosity 1 > $OUT/${SPEC}_cpsat_soft.log 2>&1
echo "  soft rc=$? $(grep -iE 'status|misses' $OUT/${SPEC}_cpsat_soft.log | tail -n 2 | tr '\n' ' ')"
[ -f schedules/scheduled_${SPEC}_soft_cpsat_profiled.json ] && cp schedules/scheduled_${SPEC}_soft_cpsat_profiled.json schedules/fig_${TAG}_cpsat_soft.json && cp schedules/scheduled_${SPEC}_soft_cpsat_profiled_metrics.json schedules/fig_${TAG}_cpsat_soft_metrics.json
wait
say "STAGE2_DONE $SPEC"

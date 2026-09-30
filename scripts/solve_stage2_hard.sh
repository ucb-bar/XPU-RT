#!/usr/bin/env bash
# The tables both solvers produce for a spec: greedy, then hard-window CP-SAT, with the soft-window
# CP-SAT (lexicographic misses -> lateness -> makespan, HEFT warm start) solved alongside as the
# stand-in for a round where the hard certificate is not found within the limit.
#   scripts/solve_stage2_hard.sh <spec> <tag> [limit_s]      env CAL=<table|none>  OUT=<log dir>  SOFT=0 (hard only)
# Outputs: schedules/fig_<tag>_{greedy,cpsat_hard,cpsat_soft}.json (+_metrics). The solver's own output
# files are removed before each solve, so a solve that returns no table leaves no table.
set -u; cd "$(dirname "$0")/.."
SPEC="${1:?spec}"; TAG="${2:?tag}"; LIM="${3:-3000}"
export XPURT_CPSAT_PYTHON=$PWD/.venv/bin/python XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1 XPURT_CPSAT_WORKERS=8
CAL="${CAL:-results/codesign_feedback/k1_board_calibration_yolo110.json}"; OUT="${OUT:-results/codesign_feedback/solver_v2}"   # CAL=none -> the isolated profile, no board calibration
CALARG="--board-calibration $CAL"; [ "$CAL" = none ] && CALARG=""
echo "=== $(date +%H:%M:%S) $SPEC greedy"
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$SPEC.json --solver greedy_periodic --max-periodic-iters 1 --use-profiled $CALARG --random-seed 42 > $OUT/${SPEC}_greedy.log 2>&1
cp schedules/scheduled_${SPEC}_greedy_periodic_profiled.json schedules/fig_${TAG}_greedy.json; cp schedules/scheduled_${SPEC}_greedy_periodic_profiled_metrics.json schedules/fig_${TAG}_greedy_metrics.json
echo "  $(grep -E 'periodic window misses' $OUT/${SPEC}_greedy.log | tail -n 1)"
rm -f schedules/scheduled_${SPEC}_cpsat_profiled.json schedules/scheduled_${SPEC}_cpsat_profiled_metrics.json schedules/scheduled_${SPEC}_soft_cpsat_profiled.json schedules/scheduled_${SPEC}_soft_cpsat_profiled_metrics.json
echo "=== $(date +%H:%M:%S) $SPEC cpsat hard (limit $LIM s)$([ "${SOFT:-1}" = 1 ] && echo ' + cpsat soft alongside')"
( CPSAT_LOG=1 .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/$SPEC.json --solver cpsat --max-periodic-iters 1 --cpsat-time-limit $LIM --use-profiled $CALARG > $OUT/${SPEC}_cpsat_hard.log 2>&1
  echo "  hard rc=$? $(grep -E 'cpsat status=' $OUT/${SPEC}_cpsat_hard.log | tail -n 1)"
  [ -f schedules/scheduled_${SPEC}_cpsat_profiled.json ] && cp schedules/scheduled_${SPEC}_cpsat_profiled.json schedules/fig_${TAG}_cpsat_hard.json && cp schedules/scheduled_${SPEC}_cpsat_profiled_metrics.json schedules/fig_${TAG}_cpsat_hard_metrics.json ) &
if [ "${SOFT:-1}" = 1 ]; then
  sleep 20   # the soft path solves a private copy of the spec, so its output name differs from the hard path's
  python3 -c "import json; s=json.load(open('data/toplevel/$SPEC.json')); json.dump(s, open('data/toplevel/${SPEC}_soft.json','w'), indent=2)"
  .venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/${SPEC}_soft.json --solver milp --scheduler cpsat --cpsat-time-limit $LIM --use-profiled $CALARG --solver-verbosity 1 > $OUT/${SPEC}_cpsat_soft.log 2>&1
  echo "  soft rc=$? $(grep -iE 'status|misses' $OUT/${SPEC}_cpsat_soft.log | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  [ -f schedules/scheduled_${SPEC}_soft_cpsat_profiled.json ] && cp schedules/scheduled_${SPEC}_soft_cpsat_profiled.json schedules/fig_${TAG}_cpsat_soft.json && cp schedules/scheduled_${SPEC}_soft_cpsat_profiled_metrics.json schedules/fig_${TAG}_cpsat_soft_metrics.json
fi
wait
for s in cpsat_hard cpsat_soft; do [ -f schedules/fig_${TAG}_${s}.json ] && echo "  table $s: schedules/fig_${TAG}_${s}.json" || echo "  table $s: none"; done
echo "=== $(date +%H:%M:%S) STAGE2_DONE $SPEC"

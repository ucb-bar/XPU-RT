#!/usr/bin/env bash
# Does the DETERMINISTIC 45 Hz table reach the board latency the figure draws?
#
# The figure's 27.45 ms (Gantt window) / 28.30 ms (run median) is measured on the K1, but the
# placement behind it came from an 8-worker CP-SAT solve, and xpu-rt/cpsat_scheduler.py says such a
# solve is not reproducible: with several search workers the answer depends on thread interleaving.
# scripts/solve45_determinism.sh re-solved the same unconstrained spec with XPURT_CPSAT_WORKERS=1
# under two different random seeds and got tables that are identical dispatch for dispatch, with the
# same makespan, the same critical path to 5 us, the same 1012 IME placements and the same zero
# deadline misses as the 8-worker table.
#
# That settles the solver. It does not settle the board: the two tables reach that makespan through
# slightly different width choices (YOLO 2-wide 389 vs 500 dispatches, nav 2-wide 25 vs 44), and
# width is what codegen builds. So the deterministic table is carried through the same codegen
# contract and run on the K1 under its own label, and its cadence trace is extracted the same way.
# If it lands at the same latency, the number in the figure is something the recipe produces rather
# than something one solve happened to find.
#
# Writes only under the p45det label; p45free's artifacts are not touched.
#
#   scripts/board_det45.sh                      env REPS=3 TABLE
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback
TABLE="${TABLE:-schedules/scheduled_wh_chain45_free_det1_cpsat_profiled.json}"
say(){ echo "=== $(date +%H:%M:%S) $*"; }
[ -f "$TABLE" ] || { echo "no table at $TABLE"; exit 1; }

say "placements the deterministic solve chose"
.venv/bin/python -c "
import json,collections,sys
ds=json.load(open('$TABLE'))['dispatches']
per=collections.defaultdict(set); wid=collections.defaultdict(collections.Counter)
for v in ds.values():
    k='control' if v['job_name'].startswith('mlp') else ('nav' if v['job_name'].startswith('fused') else 'YOLO')
    hs=v['hardware_target'].split('+'); per[k]|=set(hs); wid[k][len(hs)]+=1
for k in ('YOLO','nav','control'):
    print(f'    {k:8s} {len(per[k])} harts  widths ' + ' '.join(f'{a}x:{b}' for a,b in sorted(wid[k].items())))
print('  IME dispatches:', sum(1 for v in ds.values() if v.get('impl')=='ime'))
"

say "codegen contract + board, label p45det"
REPS="${REPS:-3}" bash scripts/board_partitioned30.sh "$TABLE" p45det gen/mb_shard_nav 2>&1 | tail -20

T=$R/xpurt_long/trace_p45detr1_other_run1.csv
if [ -f "$T" ]; then
  say "cadence trace"
  # an xpurt_long run is under a second, so the ROS arms' 3 s warm-up would discard all of it
  # (the tool exits 1 with "only 0 gaps after warm-up"); the default 100 ms is the one for this side
  .venv/bin/python scripts/ctrl_trace_from_board.py "$T" --out $R/ctrl_traces/xpu_p45det.csv \
    || { echo "cadence cut failed for p45det; stopping"; exit 1; }
else
  echo "no trace at $T; the board run did not land"
fi
say "BOARD_DET45_DONE"

#!/usr/bin/env bash
# The 45 Hz chain with the placement left to the solver, onto the board.
#
# Waits for the CP-SAT solve of data/toplevel/wh_chain45_free.json, then carries the table through
# the codegen contract and runs it three times on the K1, then pulls a cadence trace out of the
# first replicate. This is the 45 Hz sibling of p30free: the spec carries no allowed_machines and no
# machine_width, so where the kernels land is the solver's answer rather than an instruction.
#
# Why it exists: at a 45 Hz camera the scheduled arms already measured run 53-57 ms camera->control
# (xpu_a_cpsat_hard 56.8, xpu_shardcpsat45 53.2), which is the same latency as the hand-pinned ROS 2
# baseline cp3 (56.2 ms). With no latency advantage the comparison rests on command rate alone, and
# the rate-injected envelope is flat above the control-rate floor -- so the comparison at that camera
# rate was measuring nothing. p30free reaches 26.8 ms at 30 Hz with the same freedom; this asks what
# the solver does with a 22.2 ms camera period.
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback
S=schedules/scheduled_wh_chain45_free_cpsat_profiled.json
say(){ echo "=== $(date +%H:%M:%S) $*"; }

until [ -f "$S" ]; do
  grep -q "SOLVE45_EXIT=" $R/solve45_free.log 2>/dev/null && { [ -f "$S" ] || { echo "solve finished without a table; stopping"; exit 1; }; }
  sleep 60
done
say "table on disk: $S"
.venv/bin/python -c "
import json,collections,sys
d=json.load(open('$S')); ds=d['dispatches']
per=collections.Counter()
for v in ds.values():
    j=v['job_name']; k='control' if j.startswith('mlp') else ('nav' if j.startswith('fused') else 'YOLO')
    per[(k, v['hardware_target'])]+=1
print('  placements the solver chose:')
for k,n in sorted(per.items()): print(f'    {k[0]:8s} {k[1]:40s} {n}')
print('  IME dispatches:', sum(1 for v in ds.values() if v.get('impl')=='ime'))
"

say "board runs"
REPS=3 bash scripts/board_partitioned30.sh "$S" p45free gen/mb_shard_nav 2>&1 | tail -20

T=$R/xpurt_long/trace_p45freer1_other_run1.csv
if [ -f "$T" ]; then
  say "cadence trace"
  # an xpurt_long run is under a second, so the ROS arms' 3 s warm-up would discard all of it
  # (the tool exits 1 with "only 0 gaps after warm-up"); the default 100 ms is the one for this side
  .venv/bin/python scripts/ctrl_trace_from_board.py "$T" --out $R/ctrl_traces/xpu_p45free.csv \
    || { echo "cadence cut failed for p45free; stopping"; exit 1; }
else
  echo "no trace at $T; the board run did not land"
fi
say "BOARD_FREE45_DONE"

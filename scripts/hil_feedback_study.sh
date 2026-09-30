#!/usr/bin/env bash
# The runtime-feedback loop with the board in it, on the figure's own chain specs: round 0 solves
# both tables from the isolated profile (no board knowledge) and executes them on the K1; each
# later round fits a per-dispatch service-time calibration from the previous round's executed CP-SAT
# traces (scripts/calibration_from_executed.py, joined through the IR), re-solves both solvers on it,
# and executes again. Per round the hard-window CP-SAT table is the round's CP-SAT table; when the
# hard certificate is not found within the limit, the soft-window CP-SAT table solved alongside
# (HEFT warm start, misses -> lateness -> makespan) stands in, and the round says so.
# Three specs where the frame is tight enough for the gap to matter; three rounds each; every
# table run three times. Everything under results/codesign_feedback/hil_feedback/.
#   ROUNDS="0 1 2" PAIRS="a90:wh_chain90_solve_500 ..." LIMIT=3000 scripts/hil_feedback_study.sh
set -u; cd "$(dirname "$0")/.."
H=results/codesign_feedback/hil_feedback; L=results/codesign_feedback/solver_v2; X=results/codesign_feedback/xpurt_long
say(){ echo "=== $(date +%H:%M:%S) $*"; }
ROUNDS="${ROUNDS:-0 1 2}"; LIMIT="${LIMIT:-3000}"
PAIRS="${PAIRS:-a90:wh_chain90_solve_500 a120h:wh_chain120_solve_h200 b5:wh_chain90_rich_solve_500}"
WIN="yolov8_nano_64x96=66.6667,fused_full=77.7778,mlp_control=10"
cpsat_of(){ # the CP-SAT table a round executed: hard if it exists, else soft
  [ -f schedules/fig_$1_cpsat_hard_clamped.json ] && echo cpsat_hard || { [ -f schedules/fig_$1_cpsat_soft_clamped.json ] && echo cpsat_soft; }; }
for pair in $PAIRS; do
  tag=${pair%%:*}; spec=${pair#*:}; cal=none
  for r in $ROUNDS; do
    T=fb${tag}r${r}
    if [ "$r" -gt 0 ]; then
      prev=fb${tag}r$((r-1)); cal=$H/cal_${tag}_r${r}.json; ps=$(cpsat_of $prev)
      [ -n "$ps" ] || { say "$T: round $((r-1)) has no executed CP-SAT table, stopping this spec"; break; }
      say "$T: calibration from the executed round-$((r-1)) $ps traces"
      .venv/bin/python scripts/calibration_from_executed.py --trace-glob "$X/trace_${prev}${ps}r*_other_run1.csv" \
        --schedule schedules/fig_${prev}_${ps}_clamped.json --workload "$spec, round $((r-1)) executed ($ps)" --out $cal 2>&1 | tail -n 2 | cut -c1-400
      [ -f $cal ] || { say "$T: no calibration table, stopping this spec"; break; }
    fi
    if grep -q STAGE2_DONE $L/stage2_${T}.log 2>/dev/null; then say "$T: tables exist"; else
      say "$T: solving ($spec, calibration ${cal}, limit $LIMIT s)"; CAL=$cal bash scripts/solve_stage2_hard.sh $spec $T $LIMIT > $L/stage2_${T}.log 2>&1; cat $L/stage2_${T}.log
    fi
    if grep -q BOARD_STAGE2_DONE $L/board_stage2_${T}.log 2>/dev/null; then say "$T: board runs exist"; else
      say "$T: board"; bash scripts/board_stage2.sh $T $spec > $L/board_stage2_${T}.log 2>&1; grep -E "feasibility rc|no schedules|=== .* board" $L/board_stage2_${T}.log | tail -n 10
    fi
    for s in cpsat_hard cpsat_soft greedy; do for k in 1 2 3; do f=$X/trace_${T}${s}r${k}_other_run1.csv; [ -f $f ] || continue
      echo "  $T $s r$k: $(.venv/bin/python scripts/xpurt_trace_report.py --windows $WIN --skip-ms 100 $f 2>/dev/null | grep -E 'camera->control|window misses' | tr -s ' ' | tr '\n' ' ' | cut -c1-220)"; done; done
  done
done
say "HIL_FEEDBACK_STUDY_DONE"

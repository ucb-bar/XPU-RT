#!/usr/bin/env bash
# The whole OUTER loop for one rung, unattended: take a converged AOT spec, solve it,
# run THAT schedule on the K1, calibrate from the run's own trace, attribute the residual,
# then re-cost and re-solve against the measured costs. Five steps, each needing the
# previous one's output path -- which is why this is a script and not a runbook. Doing it
# by hand makes the --backends arity and the --staged-ir flags easy to get wrong.
#
# Produces the four-beat arc the study reports:
#   baseline -> AOT (inner) -> board re-cost (the reveal) -> board re-solve (the fix)
#
# Usage:
#   scripts/run_board_arc.sh --workload data/toplevel/scaling/s5_solvable_reveal.json \
#       --stem s5_arc --models mlp_control,fused_full,ffn_block,dronet,yolov8_nano_64x96
#
#   --spec        an already-converged spec to execute. Omit it and the script runs the
#                 inner loop first and reads the converged spec out of the loop report.
#   --calibration where to write the measured table (default results/codesign_feedback/
#                 k1_cal_<stem>_measured.json). ALWAYS per-rung: see the warning below.
#
# NEEDS THE K1. Board access is an ssh config entry named by MODELBLASTER_K1_HOST
# (default "k1"). No credentials are read or written here.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${XPURT_PY:-$REPO/.venv/bin/python}"

WORKLOAD=""; SPEC=""; STEM=""; MODELS=""; CAL=""; OUTDIR=""; ROUNDS=4
while [ $# -gt 0 ]; do
  case "$1" in
    --workload) WORKLOAD="$2"; shift 2;;
    --spec) SPEC="$2"; shift 2;;
    --stem) STEM="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --calibration) CAL="$2"; shift 2;;
    --out-dir) OUTDIR="$2"; shift 2;;
    --max-rounds) ROUNDS="$2"; shift 2;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$WORKLOAD" ] && [ -n "$STEM" ] && [ -n "$MODELS" ] || {
  echo "usage: $0 --workload SPEC --stem NAME --models a,b,c [--spec CONVERGED] [--calibration OUT]" >&2; exit 2; }
CAL="${CAL:-results/codesign_feedback/k1_cal_${STEM}_measured.json}"
OUTDIR="${OUTDIR:-results/codesign_feedback/${STEM}}"
WORK="${TMPDIR:-/tmp}/${STEM}"; mkdir -p "$WORK"
P="$WORK/progress.txt"; : > "$P"
say() { echo "$(date +%H:%M) $*" | tee -a "$P"; }

# ---- step 0: the inner loop, if a converged spec was not supplied ------------------------
if [ -z "$SPEC" ]; then
  say "step 0/5 inner loop (AOT co-design)"
  LOOPDIR="$WORK/inner"
  timeout 2400 "$PY" scripts/run_codesign_loop.py --workload "$WORKLOAD" \
    --solver greedy --max-rounds "$ROUNDS" --time-limit 120 \
    --out-dir "$LOOPDIR" > "$WORK/inner.log" 2>&1
  STEM_WL="$(basename "$WORKLOAD" .json)"
  REPORT="$LOOPDIR/$STEM_WL/loop_report.json"
  [ -f "$REPORT" ] || { say "inner loop produced no report; see $WORK/inner.log"; exit 1; }
  # THE CONVERGED SPEC IS THE ONE FROM THE LAST *ACCEPTED* ROUND, and that is NOT the newest
  # file on disk. The loop writes a candidate spec for every lever it TRIES, so the final
  # round -- the one that accepts nothing and ends the search -- leaves the most recent files
  # behind, and those are rejects. Picking by mtime once sent a REJECTED round-3 candidate
  # (shard:dronet, which the accept rule had just refused) to the board. Read the report.
  SPEC=$("$PY" - "$REPORT" "$LOOPDIR/$STEM_WL/specs" "$STEM_WL" <<'PY'
import json, os, sys
rep, specdir, stem = sys.argv[1], sys.argv[2], sys.argv[3]
acc = [r for r in (json.load(open(rep)).get("rounds") or []) if r.get("accepted")]
print(os.path.join(specdir, f"{stem}_r{acc[-1]['round']}_{acc[-1]['lever']}.json") if acc else "")
PY
)
  [ -n "$SPEC" ] && [ -f "$SPEC" ] || { say "no accepted round: the baseline was already best"; exit 1; }
  say "converged spec: $(basename "$SPEC")"
fi

# ---- step 1: solve it --------------------------------------------------------------------
say "step 1/5 solve the converged spec"
XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1 timeout 900 \
  "$PY" scripts/run_xpurt_schedule.py --networks-json "$SPEC" \
  --profiled --max-periodic-iters 1 --solver greedy --time-limit 60 \
  > "$WORK/solve.log" 2>&1
SCHED="schedules/scheduled_$(basename "$SPEC" .json)_greedy_profiled.json"
[ -f "$SCHED" ] || { say "solve produced no schedule; see $WORK/solve.log"; exit 1; }
cp "$SCHED" "$WORK/sched.json"
say "solved -> $(basename "$SCHED")"

# ---- step 2: execute on the K1 -----------------------------------------------------------
# --backends takes one entry per CORE KIND (cpu_p, cpu_e), NOT one per model. Passing one
# per model is the arity mistake this script exists to stop repeating.
say "step 2/5 execute on the K1"
eval "$(bash scripts/setup_spacemit_toolchain.sh 2>/dev/null)"
B=ModelBlaster/build/k1_xpurt
STAGED=""
for m in ${MODELS//,/ }; do STAGED="$STAGED --staged-ir $m:$PWD/$B/$m/int8"; done
timeout 2400 env CROSS="${CROSS:-}" bash ModelBlaster/scripts/run_xpurt_k1.sh \
  --schedule "$WORK/sched.json" --models "$MODELS" $STAGED \
  --backends rvv_x60,rvv_x60 --quant int8 --out-root "$WORK/board" \
  > "$WORK/board.log" 2>&1
say "board exit=$? verified=$(grep -c MODELBLASTER_VERIFY "$WORK/board.log")"
TRACE=$(find "ModelBlaster/tmp/$(basename "$WORK")_board" ModelBlaster/tmp -name "*_trace.csv" -newer "$WORK/sched.json" 2>/dev/null | head -1)
[ -n "$TRACE" ] || { say "no trace produced; see $WORK/board.log"; exit 1; }
say "trace: $(wc -l < "$TRACE") rows"

# ---- step 3: calibrate from THIS run's own trace ------------------------------------------
# PER-RUNG, ALWAYS. A calibration measured on a different rung is an extrapolation: reusing
# b5z's table for s5 mispredicted it by 1.66x, which is precisely the error the outer loop
# exists to catch, so committing it at the calibration step defeats the purpose.
# --schedule enables the dispatch-id alignment check: the runner renumbers dispatches around
# zero-cost ops, so without it a per-dispatch multiplier can be filed against a DIFFERENT
# dispatch and nothing about the emitted table looks wrong.
say "step 3/5 calibrate from that run's own trace"
mkdir -p "$(dirname "$CAL")"
"$PY" scripts/emit_board_calibration.py --trace "$TRACE" --schedule "$WORK/sched.json" \
  --workload "$STEM converged, measured on K1" --out "$CAL" > "$WORK/cal.log" 2>&1
say "calibration: $(grep -oE 'aggregate x[0-9.]+' "$WORK/cal.log" | tail -1)"

# ---- step 4: attribute the residual --------------------------------------------------------
# Answers the question that decides whether a leftover miss is the loop's fault: an
# EXECUTION-BOUND instance cannot fit its window however it is placed, so it is a compiler
# gap; a queueing-bound one is a scheduling miss and the loop is accountable for it.
say "step 4/5 attribute the residual"
"$PY" scripts/attribute_board_misses.py --trace "$TRACE" --spec "$WORKLOAD" \
  --json-out "results/codesign_feedback/${STEM}_miss_attribution.json" > "$WORK/attr.log" 2>&1
say "attribution: $(grep -oE 'execution-bound instances.*' "$WORK/attr.log" | head -1)"

# ---- step 5: the four beats, scored on the measured costs ----------------------------------
say "step 5/5 full four-beat arc against the measured costs"
timeout 2400 env XPURT_CPSAT_WORKERS=4 "$PY" scripts/run_codesign_loop.py \
  --workload "$WORKLOAD" --solver cpsat --max-rounds "$ROUNDS" --time-limit 120 \
  --board-calibration "$CAL" --board-solver cpsat --board-time-limit 300 \
  --out-dir "$OUTDIR" > "$WORK/outer.log" 2>&1
say "arc: $(grep -oE 'instance-miss arc.*' "$WORK/outer.log" | tail -1)"
say "ALL DONE -- artifacts in $OUTDIR, calibration $CAL, logs $WORK"

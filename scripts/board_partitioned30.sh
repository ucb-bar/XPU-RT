#!/usr/bin/env bash
# A spatially partitioned 30 Hz chain table, made buildable and executed on the K1.
#
# The partition is declared in the SPEC (data/toplevel/wh_chain30_part{,ime}.json), not here:
# `allowed_machines` per network -- yolo on the four P cores, nav on CPU_E#0, control on
# CPU_E#1 -- plus `shard_only_networks: "none"` to hold every dispatch at one hart. This script
# only carries a solved table through the codegen contract and onto the board, the way
# scripts/board_stage2.sh does for the unpartitioned arms.
#
#   scripts/board_partitioned30.sh <schedule.json> <label> [gen-root]
#     gen-root defaults to gen/mb; pass gen/mb_shard for a table with ime placements, which is
#     also where the ime_x60 kernel-availability rows live.
# An `impl: ime` in the table selects the IME backend per dispatch, so the board build needs the
# three-backend walker; every ime placement is on cluster 0 because smt.vmadot SIGILLs on harts 4-7.
set -u; cd "$(dirname "$0")/.."
SCHED="${1:?schedule json}"; LABEL="${2:?label}"; GEN="${3:-gen/mb}"; REPS="${REPS:-3}"
RES=results/codesign_feedback
say() { echo "=== $(date +%H:%M:%S) $*"; }
[ -f "$SCHED" ] || { echo "no table at $SCHED"; exit 1; }
IRS="yolov8_nano_64x96:ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/graph.json fused_full:ModelBlaster/build/k1_xpurt/fused_full/int8/graph.json mlp_control:ModelBlaster/build/k1_xpurt/mlp_control/int8/graph.json"
CL="${SCHED%.json}_clamped.json"
say "clamp + feasibility $SCHED"
.venv/bin/python scripts/clamp_schedule_widths.py "$SCHED" $IRS --out "$CL" 2>&1 | tail -n 2
.venv/bin/python scripts/check_schedule_feasibility.py --schedule "$CL" --gen-root "$GEN" 2>&1 | tail -n 2
rc=${PIPESTATUS[0]}; echo "  feasibility rc=$rc"; [ "$rc" = 0 ] || { echo "INFEASIBLE, not run"; exit 1; }
# an ime placement anywhere in the table means the three-backend walker
NIME=$(.venv/bin/python -c "import json,sys;d=json.load(open(sys.argv[1]))['dispatches'];print(sum(1 for v in d.values() if v.get('impl')=='ime'))" "$CL")
if [ "$NIME" -gt 0 ]; then
  export BACKENDS="rvv_x60,rvv_x60,ime_x60" CORE_KINDS="rvv,rvv_c1,ime"
  say "$NIME dispatch(es) on the IME: BACKENDS=$BACKENDS MB_IME_FORCE=${MB_IME_FORCE:-0}"
fi
# MB_IME_FORCE=1 in the caller's environment reaches generate_kernels through the harness and
# builds the blanket-IME kernels (every op that has one), which the table-guided default does not.
# It changes the BINARY, so a table solved against table-guided costs must not be run under it.
# one board user at a time, the same lock every other board script takes
( exec 9>"$RES/board.lock"; flock 9
  ssh k1 "pkill -x cpu_sampler" >/dev/null 2>&1 || true
  while ssh k1 "ps aux | grep -E 'ros_mb_chain|xpurt' | grep -v -E 'grep|cpu_sampler' | wc -l" | grep -qv '^0$'; do sleep 60; done
  for k in $(seq 1 "$REPS"); do
    say "board ${LABEL}r$k"
    scripts/run_xpurt_long.sh "$CL" "${LABEL}r$k" 1 2>&1 | grep -E "trace rows|re-pulled|FATAL|Error"
  done )
say "BOARD_PARTITIONED_DONE $LABEL"

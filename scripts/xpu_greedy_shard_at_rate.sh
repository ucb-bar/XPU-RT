#!/usr/bin/env bash
# XPU-RT's greedy placement with the board's per-width costs, at any camera rate, run on the K1.
#
# WHY. A flight comparison pairs two deployments at the SAME camera rate, each at its own measured
# cadence and latency. The two-instance ROS 2 graph was measured at 36 and 40 Hz; XPU-RT had no
# board run there. This builds the 45 Hz greedy+shard arm's spec (wh_chain45_w2p: gen/mb_shard costs,
# perception held to two camera periods, never under 40 ms) at another rate, solves it, makes it
# buildable, checks it, and executes it three times.
#
#   scripts/xpu_greedy_shard_at_rate.sh <hz>          -> xpurt_long/trace_w2pg<hz>r{1,2,3}_other_run1.csv
set -u
cd "$(dirname "$0")/.."
HZ="${1:?camera rate}"; REPS="${REPS:-3}"; RES=results/codesign_feedback; TAG="w2pg${HZ}"
SPEC="data/toplevel/wh_chain${HZ}_w2p.json"
python3 - "$HZ" "$SPEC" <<'PY'
import json, sys
hz, out = float(sys.argv[1]), sys.argv[2]
s = json.load(open("data/toplevel/wh_chain45_w2p.json"))
p = 1000.0 / hz
for net in ("yolov8_nano_64x96", "fused_full"):
    s["networks"][net]["period"] = p; s["networks"][net]["num_instances"] = int(round(hz))
s["networks"]["yolov8_nano_64x96"]["window_duration"] = max(2.0 * p, 40.0)
s["_comment"] = (f"wh_chain45_w2p at a {hz:g} Hz camera: gen/mb_shard per-width costs, perception held to two camera "
                 f"periods ({2*p:.1f} ms, floor 40), nav window and the 100 Hz control task unchanged.")
json.dump(s, open(out, "w"), indent=2)
print(f"  spec {out}: period {p:.2f} ms, yolo window {max(2*p,40):.2f} ms, {int(round(hz))} instances")
PY
B=$(basename "$SPEC" .json)
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json "$SPEC" --solver greedy_periodic --max-periodic-iters 1 \
  --use-profiled --random-seed 42 > "$RES/solver_v2/${B}_greedy.log" 2>&1 || { echo "greedy solve failed"; exit 1; }
cp "schedules/scheduled_${B}_greedy_periodic_profiled.json" "schedules/fig_${TAG}_greedy.json"
IRS="yolov8_nano_64x96:ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/graph.json fused_full:ModelBlaster/build/k1_xpurt/fused_full/int8/graph.json mlp_control:ModelBlaster/build/k1_xpurt/mlp_control/int8/graph.json"
.venv/bin/python scripts/clamp_schedule_widths.py "schedules/fig_${TAG}_greedy.json" $IRS --out "schedules/fig_${TAG}_greedy_clamped.json" | tail -n 1
.venv/bin/python scripts/check_schedule_feasibility.py --schedule "schedules/fig_${TAG}_greedy_clamped.json" > "$RES/solver_v2/${B}_feasibility.log" 2>&1 \
  || { echo "INFEASIBLE, not run (see $RES/solver_v2/${B}_feasibility.log)"; exit 1; }
( exec 9>"$RES/board.lock"; flock 9
  for k in $(seq 1 "$REPS"); do scripts/run_xpurt_long.sh "schedules/fig_${TAG}_greedy_clamped.json" "${TAG}r${k}" 1 2>&1 | grep -E "trace rows|FATAL|Error"; done )
echo "XPU_GREEDY_SHARD_DONE $HZ"

#!/usr/bin/env bash
# The warehouse showdown composite, assembled from artifacts only.
#
#   scripts/render_showdown_measured.sh [out-stem]
#
# Panel I  : both rows built from board traces by scripts/make_measured_gantt_pair.py --
#            XPU-RT = $XPU_ARM (xpurt_long/trace_<arm>_other_run1.csv + its schedule),
#            ROS 2  = $ROS_TAG (ros_traced/<tag>/trace.csv), same window, same code.
# Panel A  : the displayed flights. XPU-RT = the campaign cell's dump (first success by the
#            simulator's own rule); ROS 2 = the single-seed re-dump of the deepest crash in the
#            ROS cell (scripts/campaign_select.py names the seed). Rates in every label are derived
#            by the figure script from the measured control gaps through ceil(gap / 10 ms).
# Panels B/C/D: hil_ablation.csv (+ gain_controlled.csv when present), courseB, flight_energy.csv.
#
# Then scripts/verify_showdown_figure.py re-derives every panel from its artifact.
. "$(dirname "$0")/env.sh"
set -eu
cd "$(dirname "$0")/.."
OUT="${1:-results/codesign_feedback/refined/warehouse_showdown_measured}"
XPU_ARM="${XPU_ARM:-best45alt2}"; XPU_SCHED="${XPU_SCHED:-schedules/best45_alt2.json}"
ROS_TAG="${ROS_TAG:-45_spin_r1}"
PY="$ISAAC_PY"
XL=results/codesign_feedback/xpurt_long; RT=results/codesign_feedback/ros_traced

XPU2_ARM="${XPU2_ARM:-}"; XPU2_SCHED="${XPU2_SCHED:-}"      # optional second XPU-RT row (the other solver's table)
XPU2_ARGS=""; [ -n "$XPU2_ARM" ] && XPU2_ARGS="--arm xpu2:xpu:$XL/trace_${XPU2_ARM}_other_run1.csv:$XL/cpu_${XPU2_ARM}_other_run1.csv:$XL/manifest_${XPU2_ARM}_other_run1.json:$XPU2_SCHED"
python3 scripts/make_measured_gantt_pair.py \
  --arm "xpu:xpu:$XL/trace_${XPU_ARM}_other_run1.csv:$XL/cpu_${XPU_ARM}_other_run1.csv:$XL/manifest_${XPU_ARM}_other_run1.json:$XPU_SCHED" \
  $XPU2_ARGS \
  --arm "ros:ros:$RT/$ROS_TAG/trace.csv:$RT/$ROS_TAG/cpu.csv:$RT/$ROS_TAG/manifest.json" \
  --window-ms "${WINDOW_MS:-120}" --skip-ms 400 --spec "${SPEC:-data/toplevel/wh_coupled_chain_long25.json}"

# control gaps, read back from the sidecars the builder just wrote
LAT_XPU=$(python3 -c "import json;print(json.load(open('schedules/measured_gantt_xpu_metrics.json'))['ctrl_gap_mean_ms'])")
LAT_ROS=$(python3 -c "import json;print(json.load(open('schedules/measured_gantt_ros_metrics.json'))['ctrl_gap_mean_ms'])")
XPU_DIR="${XPU_DIR:-results/codesign_feedback/campaign/dumps/xpu_fixed_c1.4}"
ROS_DIR="${ROS_DIR:-$(ls -d results/codesign_feedback/campaign/display/ros_s*_figdata 2>/dev/null | head -1)}"
[ -d "$ROS_DIR" ] || ROS_DIR=results/codesign_feedback/mean_gap/ros_crash_figdata
echo "panel A: xpu=$XPU_DIR ros=$ROS_DIR   control gaps xpu=$LAT_XPU ros=$LAT_ROS"

MECH=""; [ -f results/codesign_feedback/flight_energy.csv ] && grep -q "ros33_s\|ros17_s" results/codesign_feedback/flight_energy.csv || MECH="--no-mechanism"
XPU2_SCHED_ARG=""; [ -n "$XPU2_ARM" ] && XPU2_SCHED_ARG="--sched-xpu2 schedules/measured_gantt_xpu2.json --label-xpu ${LABEL_XPU:-XPU-RT·CP-SAT} --label-xpu2 ${LABEL_XPU2:-XPU-RT·greedy}"
timeout 900 "$PY" sims/scripts/showdown_gatecourse.py --xpu-dir "$XPU_DIR" --ros-dir "$ROS_DIR" \
  --sched-xpu schedules/measured_gantt_xpu.json --sched-ros schedules/measured_gantt_ros.json $XPU2_SCHED_ARG --label-ros "${LABEL_ROS:-ROS}" \
  --lat-xpu "$LAT_XPU" --lat-ros "$LAT_ROS" --with-story $MECH --dpi "${DPI:-300}" --out "$OUT" 2>&1 | grep -vE "Warning|warn" | tail -1
python3 scripts/verify_showdown_figure.py --xpu-arm "$XPU_ARM" --ros-tag "$ROS_TAG" --xpu-dir "${XPU_DIR:-}" --ros-dir "${ROS_DIR:-}" | tail -1

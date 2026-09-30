#!/usr/bin/env bash
# The five showdown figures under audit, each from its own recorded inputs, in one command.
#
# Every argument below is the one its reproduction page carries; nothing here is reconstructed at run
# time. No board and no GPU: the display dumps, the scene censuses, the energy CSVs and the Gantt
# sidecars are all on disk (or in archive_v3/, see docs/Artifact/artifact_checklist.md §1).
#
#   scripts/render_audited_set.sh              # render all five, then verify each
#   ONLY=s1006 scripts/render_audited_set.sh   # just the stems whose name contains that
#
# Reproduction pages, one per figure:
#   showdown_36hz_solver_vs_rosallhart_s1006  docs/Evaluation/showdown_cam36_allcores_reproduction.md
#   showdown_45hz_pinned_vs_rosdefault_s1000  docs/Evaluation/showdown_cam45_ros_unpinned_reproduction.md
#   showdown_45hz_pinned_vs_rosdefault_s1003  artifact/05_render/README.md §B  (flags recovered from
#                                             the sidecar; this figure's flights have no producer)
#   showdown_45hz_pinned_vs_rospinned_s1011   docs/Evaluation/showdown_cam45_static6_reproduction.md
#   showdown_45hz_solver_vs_rospinned_s1007   docs/Evaluation/showdown_cam45_solver_placed_reproduction.md
set -u
cd "$(dirname "$0")/.."
R=results/codesign_feedback
PY=${HOST_PY:-.venv/bin/python}
ONLY=${ONLY:-}
rc=0
say(){ echo "=== $(date +%H:%M:%S) $*"; }

render() {  # render <stem> <energy-csv> <args...>
  local stem="$1" energy="$2"; shift 2
  case "$stem" in *${ONLY}*) ;; *) return 0 ;; esac
  say "$stem"
  ENERGY_CSV="$energy" $PY scripts/showdown_paper_figure.py "$@" --out "$R/refined/$stem" 2>&1 | tail -1
  $PY scripts/verify_showdown_figure.py --metrics "$R/refined/${stem}_metrics.json" 2>&1 | tail -1
  $PY scripts/verify_showdown_figure.py --metrics "$R/refined/${stem}_metrics.json" 2>&1 | grep -q "^0 FAIL" || rc=1
}

# 36 Hz -- solver-placed XPU-RT against the baseline that uses every hart, two YOLO pools + a nav pool
for suffix in "" "_ladder"; do
  e=$R/flight_energy_navpool36.csv; [ -n "$suffix" ] && e=$R/flight_energy_navpool36_ladder.csv
  render "showdown_36hz_solver_vs_rosallhart_s1006$suffix" "$e" \
    --xpu-dir $R/campaign_free36/display/pairs_ac36/xpu_s1006_figdata \
    --ros-dir $R/campaign_free36/display/pairs_ns4/ros_s1006_figdata \
    --scene-records $R/campaign_scene/tall1000_ac36 --display-cruise 1.4 \
    --xpu-trace xpu_p36free.csv --ros-trace ros_vanilla4x236ns4.csv \
    --xpu-label "XPU-RT · solver-placed CP-SAT schedule, conv on the IME" \
    --ros-label "ROS 2 · two YOLO pools and a nav pool over all eight harts, unpinned, control chained to the goal" \
    --camera-hz 36 --gantt-prefix $R/refined/navshard36/measured_gantt_navshard36 --gantt-rows xpu,ros
done

# 45 Hz -- hand-pinned XPU-RT against ROS 2 as it is normally written (unpinned), two episode seeds
for s in 1000:display_v3s_c1.4 1003:display_lat_c1.4; do
  seed=${s%%:*}; cell=${s##*:}
  render "showdown_45hz_pinned_vs_rosdefault_s$seed" "$PWD/$R/flight_energy_v2.csv" \
    --xpu-dir $R/campaign_v2/$cell/xpu_s${seed}_figdata \
    --ros-dir $R/campaign_v2/$cell/ros_s${seed}_figdata \
    --scene-records $R/campaign_scene/tall1000s --display-cruise 1.4 \
    --xpu-trace xpu_a_cpsat_hard.csv --ros-trace ros_vanilla445.csv \
    --xpu-label "XPU-RT · CP-SAT" --ros-label "ROS 2 vanilla" \
    --camera-hz 45 --gantt-prefix $PWD/schedules/measured_gantt_v3 --gantt-rows xpu,ros
done

# 45 Hz -- the statically partitioned baseline: hand-pinned XPU-RT (the falsification test, a tie)
render "showdown_45hz_pinned_vs_rospinned_s1011" "$R/flight_energy_static6_45.csv" \
  --xpu-dir $R/campaign_static6_45/display/search_c1.4/xpu_s1011_figdata \
  --ros-dir $R/campaign_static6_45/display/search_c1.4/ros_s1011_figdata \
  --scene-records $R/campaign_scene/static6_45_l1011 --display-cruise 1.4 \
  --xpu-trace xpu_a_cpsat_hard.csv --ros-trace ros_cp345.csv \
  --xpu-label "XPU-RT · CP-SAT, hand-pinned placement" \
  --ros-label "ROS 2 · static 6-core partition, two cores idle" \
  --camera-hz 45 --gantt-prefix $R/refined/static6_45b/measured_gantt_static6_45b --gantt-rows xpu,p3

# 45 Hz -- the same baseline against the solver-placed arm
render "showdown_45hz_solver_vs_rospinned_s1007" "$R/flight_energy_free45.csv" \
  --xpu-dir $R/campaign_free45/display/search_c1.4/xpu_s1007_figdata \
  --ros-dir $R/campaign_free45/display/search_c1.4/ros_s1007_figdata \
  --scene-records $R/campaign_scene/free45_l1007 --display-cruise 1.4 \
  --xpu-trace xpu_p45free.csv --ros-trace ros_cp345.csv \
  --xpu-label "XPU-RT · solver-placed CP-SAT schedule, conv on the IME" \
  --ros-label "ROS 2 · static 6-core partition, two cores idle" \
  --camera-hz 45 --gantt-prefix $R/refined/free45b/measured_gantt_free45b --gantt-rows xpu,p3

say "RENDER_AUDITED_SET_DONE rc=$rc"
exit $rc

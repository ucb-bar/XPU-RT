#!/usr/bin/env bash
# The showdown composite, third form, for one display cell and one or every tuned-ROS placement.
#   CELL=tall1005|tall1008|cal17  TUNED=board|inb|none|all  DPI=300  scripts/render_showdown_v3.sh
# Builds the 100 ms measured Gantt rows (CP-SAT, greedy, ROS 2 on all eight cores, ROS 2 vanilla) from the
# board traces, renders the composite (+ companion), verifies the sidecar and emits the caption numbers.
# Environment:
#   CELL          which display pair: tall1005 (default; 1.0 m/s, seed 1005) | tall1008 | cal17 (no display dump exists for cal17)
#   TUNED         where the hand-pinned ROS 2 arm appears: board | inb | none | all (default: all three renders)
#   PAPER=1       the submitted figure's skeleton (--paper-form) instead of the composite
#   MAIN=1        also copy this cell's `board` render to the unsuffixed warehouse_showdown_v3* when CELL = MAIN_CELL
#   MAIN_CELL     the cell that owns the unsuffixed name (default tall1005)
#   DPI, PY       raster resolution (300) and the host interpreter (.venv/bin/python)
#   XPU_DIR, ROS_DIR, SCENE_RECORDS   override the cell's display dumps / same-scene records
#   DISPLAY_CRUISE, ENERGY_CSV        the pair's cruise (from the dump when unset) and the energy table for the mechanism panel
set -eu; cd "$(dirname "$0")/.."
CELL=${CELL:-tall1005}; TUNED=${TUNED:-all}; DPI=${DPI:-300}; PY=${PY:-.venv/bin/python}
R=results/codesign_feedback; XL=$R/xpurt_long; RT=$R/ros_traced
export ENERGY_CSV=${ENERGY_CSV:-$PWD/$R/flight_energy_v2.csv}
case $CELL in
  tall1005) XD=$R/campaign_v2/display_same/xpu_s1005_figdata; RD=$R/campaign_v2/display_same/ros_s1005_figdata; SC=$R/campaign_scene/tall1005s;;
  tall1000) XD=$R/campaign_v2/display_v3s_c1.4/xpu_s1000_figdata; RD=$R/campaign_v2/display_v3s_c1.4/ros_s1000_figdata; SC=$R/campaign_scene/tall1000s; DISPLAY_CRUISE=${DISPLAY_CRUISE:-1.4};;
  tall1008) XD=$R/campaign_v2/display_v3/xpu_s1008_figdata;   RD=$R/campaign_v2/display_v3/ros_s1008_figdata;   SC=$R/campaign_scene/tall1008;;   # no same-scene records were flown for this pair
  cal17)    XD=$R/campaign_v2/display_same_cal17/xpu_figdata; RD=$R/campaign_v2/display_same_cal17/ros_figdata; SC=$R/campaign_scene/cal17;;
  *) echo "unknown CELL $CELL"; exit 2;;
esac
XD=${XPU_DIR:-$XD}; RD=${ROS_DIR:-$RD}; SC=${SCENE_RECORDS:-$SC}
for d in $XD $RD; do [ -f $d/figure_data.npz ] || { echo "display dump missing: $d (fly the pair first: scripts/queue_v3.sh)"; exit 2; }; done
$PY scripts/make_measured_gantt_pair.py \
  --arm xpu:xpu:$XL/trace_acpsat_hardr1_other_run1.csv:$XL/cpu_acpsat_hardr1_other_run1.csv:$XL/manifest_acpsat_hardr1_other_run1.json:schedules/fig_a_cpsat_hard_clamped.json \
  --arm xpu2:xpu:$XL/trace_agreedyr1_other_run1.csv:$XL/cpu_agreedyr1_other_run1.csv:$XL/manifest_agreedyr1_other_run1.json:schedules/fig_a_greedy_clamped.json \
  --arm ros8:ros:$RT/45_vanilla4x2_r1/trace.csv:$RT/45_vanilla4x2_r1/cpu.csv:$RT/45_vanilla4x2_r1/manifest.json \
  --arm ros:ros:$RT/45_vanilla4_r1/trace.csv:$RT/45_vanilla4_r1/cpu.csv:$RT/45_vanilla4_r1/manifest.json \
  --window-ms 100 --skip-ms 400 --spec data/toplevel/wh_chain45_solve.json --out-prefix schedules/measured_gantt_v3 | cut -c1-160
[ "$TUNED" = all ] && LIST="board inb none" || LIST=$TUNED
SUF=""; PF=""; [ "${PAPER:-0}" = 1 ] && { SUF="_paper"; PF="--paper-form"; }     # PAPER=1: the submitted figure's skeleton
for t in $LIST; do
  OUT=$R/refined/warehouse_showdown_v3${SUF}_${CELL}_${t}
  $PY scripts/showdown_v3_figure.py --cell $CELL --tuned $t --xpu-dir $XD --ros-dir $RD --scene-records $SC --dpi $DPI --out $OUT $PF ${DISPLAY_CRUISE:+--display-cruise $DISPLAY_CRUISE} $([ "$t" = board ] && [ -z "$PF" ] && echo --companion)
  $PY scripts/verify_showdown_figure.py --v3-metrics ${OUT}_metrics.json | tail -n 2
  $PY scripts/emit_figure_numbers.py --metrics ${OUT}_metrics.json --prefix showV --figure "warehouse_showdown_v3_${CELL}_${t}" --out ${OUT}_numbers.tex >/dev/null 2>&1 || true
done
if [ "$CELL" = "${MAIN_CELL:-tall1005}" ] || [ "${MAIN:-0}" = 1 ]; then
  for e in .png .pdf _metrics.json; do cp $R/refined/warehouse_showdown_v3${SUF}_${CELL}_board$e $R/refined/warehouse_showdown_v3${SUF}$e; done
  [ -z "$SUF" ] && { cp $R/refined/hil_envelope_story_v3_${CELL}_board.png $R/refined/hil_envelope_story_v3.png; cp $R/refined/hil_envelope_story_v3_${CELL}_board.pdf $R/refined/hil_envelope_story_v3.pdf; }
fi
echo RENDER_V3_DONE $CELL

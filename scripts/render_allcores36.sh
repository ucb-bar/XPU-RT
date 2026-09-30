#!/usr/bin/env bash
# The showdown against a baseline that uses every hart, at the camera rate where it flies into the
# course, rendered once its flights land.
#
# Gantt rows are on disk under refined/allcores36, built against data/toplevel/wh_chain36_free.json
# so the 27.78 ms perception period is the one drawn, and against the vanilla4x2 replicate whose
# SECOND YOLO pool records its per-shard detail (results/.../ros_traced/36_vanilla4x2d2_r1):
#
#   XPU-RT p36free    solver-placed, all 8 harts   25.82 ms camera->control   0 frames late   100 Hz
#   ROS 2 vanilla4x2  two pools, all 8 harts       32.56 ms                   0 frames late    36 Hz
#
# The baseline meets the deadline it was given and leaves no hart idle; what it cannot do is command
# faster than its camera, because ROS 2 chains control to the perception output.
#
# The flights replay the cadence of the ORIGINAL run (36_vanilla4x2_r1, p95 gap 31.4 ms), not the
# re-traced replicate (p95 36.2 ms): recording the second pool's shards costs the baseline a few ms
# of tail, and the arm should be flown as deployed rather than as instrumented. Mean cadence is the
# same to 0.1 ms in both (27.77 / 27.77), which is what the flight actually follows.
#
# Waits for the display pairs and the energy ladder, renders one figure per pair that came out, and
# verifies each. One figure per separating seed rather than a chosen one: the census tally panel A
# draws is the population, and every seed in it is equally reportable.
#
#   scripts/render_allcores36.sh
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback
D=$R/campaign_free36/display/pairs_ac36
say(){ echo "=== $(date +%H:%M:%S) $*"; }

say "waiting for the display pairs"
until grep -q "DISPLAY_PAIRS_AC36_DONE" "$D/pairs.log" 2>/dev/null; do sleep 60; done
say "waiting for panel D's energy ladder"
until [ -f "$R/flight_energy_allcores36.csv" ]; do sleep 60; done

n=0
for xd in "$D"/xpu_s*_figdata; do
  s=$(basename "$xd" | sed 's/xpu_s\(.*\)_figdata/\1/')
  rd="$D/ros_s${s}_figdata"
  [ -f "$xd/figure_data.npz" ] && [ -f "$rd/figure_data.npz" ] || { say "seed $s: pair incomplete, skipped"; continue; }
  # A census cell and a one-episode display run are different runs and disagree often enough to
  # matter. verify_showdown_figure.py rejects a showdown whose scheduled arm does not finish the
  # course, which is the right call, so a seed that came out that way is kept under a name that
  # says so rather than shown as the headline.
  oc=$(.venv/bin/python -c "
import numpy as np, sys
X = np.load(sys.argv[1] + '/figure_data.npz', allow_pickle=True)
print(str(X['outcome']), int(X['gates_passed']))" "$xd")
  set -- $oc; xout="$1"; xg="$2"
  if [ "$xout" = "success" ]; then
    out=$R/refined/warehouse_showdown_cam36_allcores_s${s}
  else
    out=$R/refined/warehouse_showdown_cam36_allcores_s${s}_xpu_${xg}of4
    say "seed $s: the scheduled arm reached $xg/4 here, not the course -- kept under its own name"
  fi
  say "render seed $s -> $(basename "$out")"
  ENERGY_CSV=$R/flight_energy_allcores36.csv .venv/bin/python scripts/showdown_paper_figure.py \
    --xpu-dir "$xd" --ros-dir "$rd" \
    --scene-records $R/campaign_scene/tall1000_ac36 --display-cruise 1.4 \
    --xpu-trace xpu_p36free.csv --ros-trace ros_vanilla4x236.csv \
    --xpu-label "XPU-RT · solver-placed CP-SAT schedule, conv on the IME" \
    --ros-label "ROS 2 · two YOLO pools over all eight harts, unpinned, control chained to the goal" \
    --camera-hz 36 --gantt-prefix $R/refined/allcores36/measured_gantt_allcores36 --gantt-rows xpu,ros \
    --out "$out" 2>&1 | tail -2
  .venv/bin/python scripts/verify_showdown_figure.py --metrics "${out}_metrics.json" 2>&1 | tail -1
  n=$((n+1))
done
say "rendered $n figure(s)"
say "RENDER_ALLCORES36_DONE"

#!/usr/bin/env bash
# The showdown against a baseline that uses every hart, rendered once its flights land.
#
# Gantt rows are already on disk (refined/allcores30b), built against data/toplevel/wh_chain30_free.json
# so the 33.3 ms perception period and the spec's 66.67 ms on-time window are the ones drawn:
#
#   XPU-RT p30free    solver-placed, 8 harts   26.75 ms camera->control   0 of 14 frames late   100 Hz
#   ROS 2 vanilla4x2  all 8 harts, unpinned    31.44 ms                   0 of 298 frames late   30 Hz
#
# The baseline meets the deadline it was given and leaves no hart idle; what it cannot do is command
# faster than its camera, because ROS 2 chains control to the perception output. That is the whole
# claim, with nothing conceded about the deployment.
#
# Waits for the display pairs and the energy ladder, renders one figure per pair that came out, and
# verifies each. One figure per separating seed rather than a chosen one: the census tally panel A
# draws (3 of 12 against 0 of 12) is the population, and every seed in it is equally reportable.
#
#   scripts/render_allcores30.sh
set -u; cd "$(dirname "$0")/.."
R=results/codesign_feedback
D=$R/campaign_free30/display/pairs_ac30
say(){ echo "=== $(date +%H:%M:%S) $*"; }

say "waiting for the display pairs"
until grep -q "DISPLAY_PAIRS_AC30_DONE" "$D/pairs.log" 2>/dev/null; do sleep 60; done
say "waiting for panel D's energy ladder"
until [ -f "$R/flight_energy_allcores30.csv" ]; do sleep 60; done

n=0
for xd in "$D"/xpu_s*_figdata; do
  s=$(basename "$xd" | sed 's/xpu_s\(.*\)_figdata/\1/')
  rd="$D/ros_s${s}_figdata"
  [ -f "$xd/figure_data.npz" ] && [ -f "$rd/figure_data.npz" ] || { say "seed $s: pair incomplete, skipped"; continue; }
  # A census cell and a one-episode display run are different runs, and they disagree often enough to
  # matter: of the four seeds the 45 Hz census named, two reproduced. verify_showdown_figure.py
  # rejects a showdown whose scheduled arm does not finish the course, which is the right call, so a
  # seed that came out that way is rendered under a name that says so rather than as the headline.
  oc=$(.venv/bin/python -c "
import numpy as np, sys
X = np.load(sys.argv[1] + '/figure_data.npz', allow_pickle=True)
print(str(X['outcome']), int(X['gates_passed']))" "$xd")
  set -- $oc; xout="$1"; xg="$2"
  if [ "$xout" = "success" ]; then
    out=$R/refined/warehouse_showdown_cam30_allcores_s${s}
  else
    out=$R/refined/warehouse_showdown_cam30_allcores_s${s}_xpu_${xg}of4
    say "seed $s: the scheduled arm reached $xg/4 here, not the course -- kept under its own name"
  fi
  say "render seed $s -> $(basename "$out")"
  ENERGY_CSV=$R/flight_energy_allcores30.csv .venv/bin/python scripts/showdown_paper_figure.py \
    --xpu-dir "$xd" --ros-dir "$rd" \
    --scene-records $R/campaign_scene/tall1000_ac30 --display-cruise 1.4 \
    --xpu-trace xpu_p30free.csv --ros-trace ros_vanilla4x230.csv \
    --xpu-label "XPU-RT · solver-placed CP-SAT schedule, conv on the IME" \
    --ros-label "ROS 2 · two YOLO pools over all eight harts, unpinned, control chained to the goal" \
    --camera-hz 30 --gantt-prefix $R/refined/allcores30b/measured_gantt_allcores30b --gantt-rows xpu,ros8 \
    --out "$out" 2>&1 | tail -2
  .venv/bin/python scripts/verify_showdown_figure.py --metrics "${out}_metrics.json" 2>&1 | tail -1
  n=$((n+1))
done
say "rendered $n figure(s)"
say "RENDER_ALLCORES30_DONE"

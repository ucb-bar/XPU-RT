#!/usr/bin/env bash
# The display pair for the all-8-hart showdown at a 36 Hz camera, flown on the seeds the census
# says separate the arms.
#
# scripts/campaign_allcores36.sh flies both arms twelve times on the display scene at the deployed
# gain and prints which episode seeds XPU-RT completes and the all-eight-hart baseline does not.
# This waits for that, takes the list, and flies each of those seeds as a recorded pair.
#
# Two reasons the display flights are separate runs rather than frames pulled out of the census:
# the census uses sweep_rate_demo.py, which writes no `seed` key into figure_data.npz and panel A
# reads one; and it flies twelve episodes in a single process, so an episode's outcome there does
# not reliably reproduce in a one-episode run. Flying every separating seed rather than one is how
# that is handled: the figure draws a seed whose pair actually came out, and panel A's legend
# carries the census tally it was drawn from, so the population is on the figure's face.
#
# Settings are the census's own: cruise 1.4, gain 0.0055, layout 1000, goal hold one camera period
# (27.8 ms at 36 Hz), latency 25.8 ms for p36free and 32.3 ms for vanilla4x2.
#
#   scripts/display_pairs_allcores36.sh        env CELL SEEDS MAX_SIMS
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$WT/results/codesign_feedback"
T="$R/ctrl_traces"; CELL="${CELL:-tall1000_ac36}"
D="$R/campaign_free36/display/pairs_ac36"; LOG="$D/pairs.log"
. "$WT/scripts/env.sh"
mkdir -p "$D/tmp"
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }
gpu_room(){ local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
            n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
            [ "$free" -ge "${NEED_MB:-10000}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ]; }

if [ -z "${SEEDS:-}" ]; then
  say "waiting for the census to name the separating seeds"
  until grep -q "CAMPAIGN_ALLCORES36_DONE" "$R/campaign_allcores36_run.log" 2>/dev/null; do sleep 60; done
  SEEDS=$(cd "$WT" && .venv/bin/python - "$R/campaign_scene/$CELL" <<'PY'
import collections, glob, os, sys
sys.path.insert(0, "scripts")
from flight_quarantine import flight_rows
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "*.csv"), recursive=True)):
    try: rows += flight_rows(f)
    except Exception: pass
by = collections.defaultdict(dict)
for r in rows:
    by[r["ctrl_trace"].replace(".csv", "")][int(r["seed"])] = r["outcome"]
x = by.get("xpu_p36free", {}); b = by.get("ros_vanilla4x236", {})
print(" ".join(str(s) for s in sorted(x) if x[s] == "success" and b.get(s) != "success"))
PY
)
fi
[ -n "${SEEDS// /}" ] || { say "the census found no seed where the arms differ; nothing to fly"; exit 0; }
say "separating seeds: $SEEDS"

W="$WT/sims/models/warehouse/nav_fused_v12_cnn.pt"
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 \
--prop_density 0.30 --cruise_speed 1.4 --episodes 1 --max_steps 1800 --keep_video --moment_scale 0.0055 \
--percep_hold_ms 27.8"

fly(){  # fly <tag> <trace> <latency_ms> <extra...>
  local tag="$1" tr="$2" lat="$3"; shift 3
  [ -f "$D/${tag}_figdata/figure_data.npz" ] && { say "$tag: already on disk"; return; }
  exec 8>"$R/gpu_admit.lock"; flock 8; until gpu_room; do sleep 30; done
  ( sleep "${ADMIT_SETTLE:-150}"; flock -u 8 ) & local settle=$!
  say "$tag"
  (cd "$SIM_TREE" && TMPDIR="$D/tmp" timeout 2400 "$ISAAC_PY" sims/scripts/record_sensor_demo.py $COMMON \
     --ctrl_trace "$T/$tr" --percep_latency_ms "$lat" "$@" \
     --gantt_schedule "$WT/schedules/measured_gantt_xpu.json" \
     --save_video "$D/$tag.mp4" --dump_figure_data "$D/${tag}_figdata" > "$D/$tag.log" 2>&1 8>&-)
  kill $settle 2>/dev/null; flock -u 8
  if grep -qE "PhysX Internal CUDA error|Simulation cannot continue|GPU is out of memory|CUDA error, code" "$D/$tag.log"; then
    say "GPU fault in $tag: dump removed"; rm -rf "$D/${tag}_figdata"
  fi
  grep -hE "^\[ep00\]" "$D/$tag.log" 2>/dev/null | sed "s|^|  $tag |" | tee -a "$LOG"
}

for s in $SEEDS; do
  fly "xpu_s${s}" xpu_p36free.csv      25.8 --seed "$s" --layout_seed 1000 --post_success_steps 100
  fly "ros_s${s}" ros_vanilla4x236.csv 32.3 --seed "$s" --layout_seed 1000
done
say "DISPLAY_PAIRS_AC36_DONE"

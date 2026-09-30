#!/usr/bin/env bash
# The matched display pairs for the unpinned-ROS showdown, at every seed where the arms differ.
#
# The plan for this figure was to fly both arms on episode seed 1000, the baseline's own draw. The
# scene census on layout 1000 says that seed does not separate them: p45free crashes there too, at
# gate 3 against the baseline's gate 2, and verify_showdown_figure.py rejects the render for exactly
# that reason ("displayed XPU-RT flight completes the course (3 gates)"). Seed 1000 is a real result
# and it is kept; it is not a showdown.
#
# The same census names the seeds that do separate, out of twelve flown per arm:
#
#   seed      1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011   completed
#   p45free     x3   x2   x2   OK   x2   x2   OK   OK   x2   OK   x3   x1     4/12
#   vanilla445  x2   x2   x0   x1   x1   x1   x2   x1   x0   x1   x1   x1     0/12
#
# Four seeds -- 1003, 1006, 1007, 1009 -- and at every one of them the baseline crashes. So all four
# are flown here as display pairs rather than one being picked, and the census tally that produced
# the list is what panel A draws in its legend: the figure states 4/12 against 0/12 on its face, so a
# reader sees the population the displayed episode was drawn from, not just the episode.
#
# Both arms take the settings the census flew them at (scene_runs_pair.sh, campaign_scene/tall1000s):
# cruise 1.4, gain 0.0055, hold 0, latency 28.3 ms for p45free and 242.0 ms for vanilla445.
#
#   scripts/display_pairs_unpinned45.sh        env SEEDS MAX_SIMS
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$WT/results/codesign_feedback"
T="$R/ctrl_traces"; D="$R/campaign_free45/display/pairs45"; LOG="$D/pairs.log"
. "$WT/scripts/env.sh"
mkdir -p "$D/tmp"
SEEDS="${SEEDS:-1003 1006 1007 1009}"
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }
gpu_room(){ local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
            n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
            [ "$free" -ge "${NEED_MB:-10000}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ]; }

W="$WT/sims/models/warehouse/nav_fused_v12_cnn.pt"
COMMON="--headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 \
--prop_density 0.30 --cruise_speed 1.4 --episodes 1 --max_steps 1800 --keep_video --moment_scale 0.0055"

fly(){  # fly <tag> <trace> <latency_ms> <extra...>
  local tag="$1" tr="$2" lat="$3"; shift 3
  [ -f "$D/${tag}_figdata/figure_data.npz" ] && { say "$tag: already on disk"; return; }
  exec 8>"$R/gpu_admit.lock"; flock 8; until gpu_room; do sleep 30; done
  ( sleep "${ADMIT_SETTLE:-150}"; flock -u 8 ) & local settle=$!
  say "$tag"
  (cd "$SIM_TREE" && TMPDIR="$D/tmp" timeout 2400 "$ISAAC_PY" sims/scripts/record_sensor_demo.py $COMMON \
     --ctrl_trace "$T/$tr" --percep_latency_ms "$lat" --percep_hold_ms 0 "$@" \
     --gantt_schedule "$WT/schedules/measured_gantt_xpu.json" \
     --save_video "$D/$tag.mp4" --dump_figure_data "$D/${tag}_figdata" \
     > "$D/$tag.log" 2>&1 8>&-)
  kill $settle 2>/dev/null; flock -u 8
  if grep -qE "PhysX Internal CUDA error|Simulation cannot continue|GPU is out of memory|CUDA error, code" "$D/$tag.log"; then
    say "GPU fault in $tag: dump removed"; rm -rf "$D/${tag}_figdata"
  fi
  grep -hE "^\[ep00\]" "$D/$tag.log" 2>/dev/null | sed "s/^/  $tag /" | tee -a "$LOG"
}

for s in $SEEDS; do
  fly "xpu_s${s}" xpu_p45free.csv   28.3  --seed "$s" --layout_seed 1000 --post_success_steps 100
  fly "ros_s${s}" ros_vanilla445.csv 242.0 --seed "$s" --layout_seed 1000
done
say "DISPLAY_PAIRS45_DONE"

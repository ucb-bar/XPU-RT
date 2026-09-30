#!/usr/bin/env bash
# Everything the unpinned-ROS showdown needs to be rebuilt on our best schedule.
#
# The existing form (refined/showdown_45hz_pinned_vs_rosdefault_s1000) draws xpu_a_cpsat_hard at 56.8 ms
# against the default ROS 2 deployment, and its two flights are different episode draws -- layout seed
# 1000 for both, but episode seed 1100 for the scheduled arm and 1000 for the baseline. We have
# p45free at 27.5 ms on the same board at the same camera rate. This flies what is needed to put
# p45free in every XPU-RT quantity of that figure, on the baseline's own episode seed.
#
# Three stages, each writing somewhere new so the existing figure and its dumps survive untouched:
#
#   1. the display flight, layout 1000 / episode seed 1000, cruise 1.4 -- the pair the baseline flew.
#      record_sensor_demo.py, not sweep_rate_demo.py: only the former stamps `seed` into
#      figure_data.npz, and panel A reads it to decide whether the two arms share a seed.
#      --episodes 1, because the dump stamps the BASE seed, not the kept episode's.
#   2. panel A's scene census: the same scene, twelve episode seeds, matching how the other arms in
#      campaign_scene/tall1000s were flown (scene_runs_pair.sh hard-codes --seed 1000, so 1000-1011).
#   3. panel D's ladder. The existing four-bar ladder (flight_energy_v2.csv) was flown at cruise 1.8;
#      the free45 ladder at 1.4. Mixing them would compare arms at different speeds, so p45free is
#      flown at 1.8 to join the v2 set rather than pasted across campaigns. The condition dirs of the
#      three arms we keep are linked into a fresh run dir so the CSV carries exactly four conditions
#      and p45free -- not xpu_cpsat -- is the 1x reference.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$WT/results/codesign_feedback"
T="$R/ctrl_traces"; LOG="$R/followon_unpinned45.log"
. "$WT/scripts/env.sh"
say(){ echo "=== $(date +%H:%M:%S) $* ===" | tee -a "$LOG"; }
gpu_room(){ local free n; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
            n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
            [ "$free" -ge "${NEED_MB:-10000}" ] && [ "$n" -lt "${MAX_SIMS:-3}" ]; }

# ---- 1. the matched display flight -------------------------------------------------------------
D="$R/campaign_free45/display/pin_l1000_e1000"
if [ -f "$D/xpu_s1000_figdata/figure_data.npz" ]; then
  say "stage 1: have the display dump already"
else
  mkdir -p "$D/tmp"
  exec 8>"$R/gpu_admit.lock"; flock 8; until gpu_room; do sleep 30; done
  ( sleep "${ADMIT_SETTLE:-150}"; flock -u 8 ) & settle=$!
  say "stage 1: display flight, p45free, layout 1000 / episode seed 1000, cruise 1.4"
  (cd "$SIM_TREE" && TMPDIR="$D/tmp" timeout 1800 "$ISAAC_PY" sims/scripts/record_sensor_demo.py \
     --headless --controller rl --weights "$WT/sims/models/warehouse/nav_fused_v12_cnn.pt" \
     --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --cruise_speed 1.4 \
     --episodes 1 --max_steps 1800 --keep_video --post_success_steps 100 \
     --ctrl_trace "$T/xpu_p45free.csv" --percep_latency_ms 28.3 --percep_hold_ms 0 \
     --moment_scale 0.0055 --seed 1000 --layout_seed 1000 \
     --gantt_schedule "$WT/schedules/measured_gantt_xpu.json" \
     --save_video "$D/xpu_s1000.mp4" --dump_figure_data "$D/xpu_s1000_figdata" \
     > "$D/xpu_s1000.log" 2>&1 8>&-)
  kill $settle 2>/dev/null; flock -u 8
  grep -qE "PhysX Internal CUDA error|Simulation cannot continue|GPU is out of memory|CUDA error, code" \
    "$D/xpu_s1000.log" && { say "GPU fault in the display flight: dump removed"; rm -rf "$D/xpu_s1000_figdata"; }
fi
grep -hE "^\[ep00\]" "$D/xpu_s1000.log" 2>/dev/null | tee -a "$LOG"

# ---- 2. panel A's scene census -----------------------------------------------------------------
say "stage 2: scene census, p45free on layout 1000, twelve episode seeds"
CELL=tall1000s LAYOUT_SEED=1000 CRUISE=1.4 GAIN=0.0055 \
  ARMS="xpu_p45free:$T/xpu_p45free.csv:28.3:0.0" \
  OUT="$R/campaign_scene/tall1000s" bash "$WT/scripts/scene_runs_pair.sh" 2>&1 | tail -3 | tee -a "$LOG"

# ---- 3. panel D's ladder, all four arms at one cruise ------------------------------------------
ER="$R/energy_runs_unpinned45"; mkdir -p "$ER"
for c in xpu_greedy ros_vanilla ros_shipped; do
  for s in 1000 1001 1002 1003 1004 1005; do
    src="$R/energy_runs_v2/${c}_s${s}"
    [ -d "$src" ] && [ ! -e "$ER/${c}_s${s}" ] && ln -s "$src" "$ER/${c}_s${s}"
  done
done
say "stage 3: panel D, p45free at cruise 1.8 to join the v2 ladder"
CONDS="xpu_p45free:$T/xpu_p45free.csv" CRUISE=1.8 ER="$ER" \
  OUTCSV="$R/flight_energy_unpinned45.csv" MAX_SIMS="${MAX_SIMS:-3}" \
  bash "$WT/scripts/run_energy_pair.sh" 2>&1 | tail -4 | tee -a "$LOG"

say "FOLLOWON_UNPINNED45_DONE"

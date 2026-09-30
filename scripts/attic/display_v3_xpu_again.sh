#!/usr/bin/env bash
# Re-record the XPU-RT display flight with the row label from the executed table (deterministic:
# the same episodes as before), then the composite again.
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/env.sh"   # SIM_TREE, ISAAC_PY, RES: see scripts/env.local.sh.example
set -u; cd "$(dirname "$0")/.."
WT=$PWD; D=$WT/results/codesign_feedback/campaign_v2/display; export TMPDIR=$D/tmp
XL=results/codesign_feedback/xpurt_long; RTD=results/codesign_feedback/ros_traced
python3 scripts/make_measured_gantt_pair.py --arm "xpu:xpu:$XL/trace_acpsat_hardr1_other_run1.csv:$XL/cpu_acpsat_hardr1_other_run1.csv:$XL/manifest_acpsat_hardr1_other_run1.json:schedules/fig_a_cpsat_hard_clamped.json" \
  --arm "xpu2:xpu:$XL/trace_agreedyr1_other_run1.csv:$XL/cpu_agreedyr1_other_run1.csv:$XL/manifest_agreedyr1_other_run1.json:schedules/fig_a_greedy_clamped.json" \
  --arm "ros:ros:$RTD/45_vanilla4_r1/trace.csv:$RTD/45_vanilla4_r1/cpu.csv:$RTD/45_vanilla4_r1/manifest.json" --window-ms 140 --skip-ms 400 --spec data/toplevel/wh_chain45_solve.json | tail -n 1
python3 -c "import json; print(json.load(open('schedules/measured_gantt_xpu_metrics.json'))['arm_label'])"
PY="$ISAAC_PY"; W=sims/models/warehouse/nav_fused_v12_cnn.pt
gpu_room() { local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -ge 7000 ]; }
until gpu_room; do sleep 60; done
mv $D/xpu_c1.8.mp4 $D/xpu_c1.8_prev.mp4 2>/dev/null; rm -rf $D/xpu_c1.8_figdata
(cd "$SIM_TREE" && timeout 7200 $PY sims/scripts/record_sensor_demo.py --headless --controller rl --weights $W --sim_dt 0.01 --decimation 1 --obstacle_level 8 --prop_density 0.30 --moment_scale 0.0055 --cruise_speed 1.8 --episodes 12 --seed 1000 --max_steps 1800 --ctrl_trace $WT/results/codesign_feedback/ctrl_traces/xpu_a_cpsat_hard.csv --post_success_steps 100 --gantt_schedule $WT/schedules/measured_gantt_xpu.json --save_video $D/xpu_c1.8.mp4 --dump_figure_data $D/xpu_c1.8_figdata > $D/xpu_c1.8.log 2>&1)
grep -h "outcome=\|captured" $D/xpu_c1.8.log | cut -c1-100
SPEC=data/toplevel/wh_chain45_solve.json XPU_ARM=acpsat_hardr1 XPU_SCHED=schedules/fig_a_cpsat_hard_clamped.json XPU2_ARM=agreedyr1 XPU2_SCHED=schedules/fig_a_greedy_clamped.json ROS_TAG=45_vanilla4_r1 \
  LABEL_XPU="XPU-RT·CP-SAT" LABEL_XPU2="XPU-RT·greedy" LABEL_ROS="ROS 2 vanilla" WINDOW_MS=140 XPU_DIR=$D/xpu_c1.8_figdata ROS_DIR=$D/ros_c1.8_figdata DPI=200 \
  scripts/render_showdown_measured.sh results/codesign_feedback/refined/warehouse_showdown_v2 2>&1 | tail -n 3
echo XPU_AGAIN_DONE

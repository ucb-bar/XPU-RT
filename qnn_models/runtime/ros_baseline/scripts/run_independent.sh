#!/usr/bin/env bash
# "Typical deployment" ROS baseline runner.
#
# Each node is spawned by an INDEPENDENT `ros2 run` invocation,
# backgrounded so they run as truly separate processes started by
# separate commands — the way production deployments actually launch
# multiple ROS nodes (one systemd service per node, or three separate
# operator terminals, etc.). No `ros2 launch` coordinating startup.
#
# Functionally identical to the launch-based test (each Node action in
# all_nodes.py also forks its own process), but eliminates any "but
# launch is coordinating!" objection. Run results should match the
# launch-based ones up to startup ordering.
#
# Usage on board:
#   ./run_independent.sh B1   # all on DSP
#   ./run_independent.sh B2   # dronet→HTA, mlp→CPU, yolov8→DSP
# Note: `set -u` would trip on Foxy's setup.bash referencing
# AMENT_TRACE_SETUP_FILES before defining it; keep -e and -o pipefail
# without -u.
set -eo pipefail
SCEN="${1:-B2}"
DUR="${2:-2.5}"     # seconds — overall barrier; nodes self-cap via max_iters
WARMUP="${WARMUP:-2}"

OUT=/tmp/ros_baseline
rm -f "$OUT"/B*_*.csv
mkdir -p "$OUT"

source /opt/ros/foxy/setup.bash
source /root/ros2_ws/install/setup.bash
export LD_LIBRARY_PATH=/root/qairt/lib/target:${LD_LIBRARY_PATH:-}
export ADSP_LIBRARY_PATH='/root/qairt/lib/hexagon-v66;/dsp/cdsp;/dsp'

CTX=/root/qnn_runtime_ctx
LIB=/root/qairt/lib/target

# Per-scenario backend pick.
case "$SCEN" in
    B1)
        DRONET_LIB=$LIB/libQnnDsp.so;     DRONET_CTX=$CTX/ctx_dronet_full_seg0__Dsp.bin
        MLP_LIB=$LIB/libQnnDsp.so;        MLP_CTX=$CTX/ctx_mlp_control_full_seg0__Dsp.bin
        YOLO_LIB=$LIB/libQnnDsp.so
        ;;
    B2)
        DRONET_LIB=$LIB/libQnnHta.so;     DRONET_CTX=$CTX/ctx_dronet_full_seg0__Hta.bin
        MLP_LIB=$LIB/libQnnCpu.so;        MLP_CTX=$CTX/ctx_mlp_control_full_seg0__Cpu.bin
        YOLO_LIB=$LIB/libQnnDsp.so
        ;;
    *) echo "unknown scenario $SCEN"; exit 1;;
esac

echo "==> independent-process ROS run, scenario=$SCEN warmup=$WARMUP"

# Start each node via ros2 run, backgrounded. Each gets its own params
# block via -p key:=value pairs (Foxy CLI form). The trace_csv encodes
# the scenario tag so plot_ros_vs_milp.py sees the same filenames.
ros2 run ros_qnn_baseline dronet_node --ros-args \
    -p "ctx_path:=$DRONET_CTX" \
    -p "backend_lib:=$DRONET_LIB" \
    -p "warmup_iters:=$WARMUP" \
    -p "period_ms:=5.0" \
    -p "max_iters:=400" \
    -p "trace_csv:=$OUT/${SCEN}_dronet.csv" \
    > "$OUT/${SCEN}_dronet.stdout" 2>&1 &
PID_D=$!

ros2 run ros_qnn_baseline mlp_control_node --ros-args \
    -p "ctx_path:=$MLP_CTX" \
    -p "backend_lib:=$MLP_LIB" \
    -p "warmup_iters:=$WARMUP" \
    -p "period_ms:=2.0" \
    -p "max_iters:=1000" \
    -p "trace_csv:=$OUT/${SCEN}_mlp.csv" \
    > "$OUT/${SCEN}_mlp.stdout" 2>&1 &
PID_M=$!

ros2 run ros_qnn_baseline yolov8n_node --ros-args \
    -p "backbone_ctx_path:=$CTX/ctx_yolov8n_HTA_split_seg100__Dsp.bin" \
    -p "head_ctx_path:=$CTX/ctx_yolov8n_HTA_split_seg101__Dsp.bin" \
    -p "backend_lib:=$YOLO_LIB" \
    -p "warmup_iters:=$WARMUP" \
    -p "period_ms:=33.33" \
    -p "max_iters:=60" \
    -p "trace_csv:=$OUT/${SCEN}_yolov8.csv" \
    > "$OUT/${SCEN}_yolov8.stdout" 2>&1 &
PID_Y=$!

echo "  pids: dronet=$PID_D mlp=$PID_M yolov8=$PID_Y"

# Wait for all three (each self-caps via max_iters and rclcpp::shutdown).
# A backstop: kill anything still alive after $DUR seconds + slack.
sleep "$DUR" &
SLEEP_PID=$!
wait $PID_D $PID_M $PID_Y 2>/dev/null || true
kill $SLEEP_PID 2>/dev/null || true
echo "==> done. CSV outputs:"
ls -lh "$OUT"/${SCEN}_*.csv 2>/dev/null

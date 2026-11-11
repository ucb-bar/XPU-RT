"""ROS launch for the QNN baseline. Two scenarios via the `scenario` arg:

  scenario:=B1   all three nodes on libQnnDsp.so   (naive — DSP-only)
  scenario:=B2   dronet→Hta, mlp→Cpu, yolov8→Dsp   (best-isolated per node)

Both scenarios run all three nodes in separate processes (each node is its
own ROS executable, started here as Nodes — same process tree as `ros2
launch` produces). Cross-process QNN backend coexistence was validated by
profile_segments POC; same-backend across processes is fine, the
"Context handle already exists!" warning is informational on HTA.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import LaunchConfigurationEquals
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


CTX_DIR = "/root/qnn_runtime_ctx"
LIB_DIR = "/root/qairt/lib/target"


def _node(name, exe, ctx_paths, lib, period_ms, trace_csv, warmup_iters, max_iters, scenario_tag, condition):
    params = [
        {"backend_lib": f"{LIB_DIR}/{lib}"},
        {"period_ms":   period_ms},
        {"warmup_iters": warmup_iters},
        {"max_iters":   max_iters},
        {"trace_csv":   trace_csv},
    ]
    if isinstance(ctx_paths, dict):
        for k, v in ctx_paths.items():
            params.append({k: f"{CTX_DIR}/{v}"})
    else:
        params.append({"ctx_path": f"{CTX_DIR}/{ctx_paths}"})
    return Node(
        package="ros_qnn_baseline", executable=exe, name=name,
        parameters=params,
        condition=condition,
    )


def generate_launch_description():
    scenario_arg = DeclareLaunchArgument("scenario", default_value="B2")
    warmup_arg   = DeclareLaunchArgument("warmup_iters", default_value="2")
    duration_arg = DeclareLaunchArgument("duration_s",   default_value="2.0")
    out_dir_arg  = DeclareLaunchArgument("trace_dir",    default_value="/tmp/ros_baseline")

    scen = LaunchConfiguration("scenario")
    warm = LaunchConfiguration("warmup_iters")
    out  = LaunchConfiguration("trace_dir")

    # max_iters per node = ceil(duration_s / period_s). We let the node
    # cancel its timer + shutdown after that many ticks so the run is
    # bounded. For a 2 s window: dronet 5ms→400, mlp 2ms→1000, yolov8
    # 33.33ms→60.
    DURATION_S = 2.0  # the launch arg drives nothing now (kept for future)
    DRONET_ITERS = 400
    MLP_ITERS    = 1000
    YOLO_ITERS   = 60

    nodes = []

    # ---- B1: all-DSP (naive) ----
    cond_b1 = LaunchConfigurationEquals("scenario", "B1")
    nodes.append(_node(
        "dronet_node", "dronet_node",
        "ctx_dronet_full_seg0__Dsp.bin",
        "libQnnDsp.so", 5.0, "/tmp/ros_baseline/B1_dronet.csv",
        warm, DRONET_ITERS, "B1", cond_b1))
    nodes.append(_node(
        "mlp_control_node", "mlp_control_node",
        "ctx_mlp_control_full_seg0__Dsp.bin",
        "libQnnDsp.so", 2.0, "/tmp/ros_baseline/B1_mlp.csv",
        warm, MLP_ITERS, "B1", cond_b1))
    nodes.append(_node(
        "yolov8n_node", "yolov8n_node",
        {"backbone_ctx_path": "ctx_yolov8n_HTA_split_seg100__Dsp.bin",
         "head_ctx_path":     "ctx_yolov8n_HTA_split_seg101__Dsp.bin"},
        "libQnnDsp.so", 33.33, "/tmp/ros_baseline/B1_yolov8.csv",
        warm, YOLO_ITERS, "B1", cond_b1))

    # ---- B2: best-isolated per node ----
    cond_b2 = LaunchConfigurationEquals("scenario", "B2")
    nodes.append(_node(
        "dronet_node", "dronet_node",
        "ctx_dronet_full_seg0__Hta.bin",
        "libQnnHta.so", 5.0, "/tmp/ros_baseline/B2_dronet.csv",
        warm, DRONET_ITERS, "B2", cond_b2))
    nodes.append(_node(
        "mlp_control_node", "mlp_control_node",
        "ctx_mlp_control_full_seg0__Cpu.bin",
        "libQnnCpu.so", 2.0, "/tmp/ros_baseline/B2_mlp.csv",
        warm, MLP_ITERS, "B2", cond_b2))
    nodes.append(_node(
        "yolov8n_node", "yolov8n_node",
        {"backbone_ctx_path": "ctx_yolov8n_HTA_split_seg100__Dsp.bin",
         "head_ctx_path":     "ctx_yolov8n_HTA_split_seg101__Dsp.bin"},
        "libQnnDsp.so", 33.33, "/tmp/ros_baseline/B2_yolov8.csv",
        warm, YOLO_ITERS, "B2", cond_b2))

    return LaunchDescription([scenario_arg, warmup_arg, duration_arg, out_dir_arg, *nodes])

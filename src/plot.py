import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import OrderedDict


def plot_schedule_from_json(schedule_data: dict, save_path: str = "schedule.png", plot_title: str = "Schedule"):
    """
    Plot a schedule from a resolved JSON schedule (as produced by the periodic scheduler).

    The JSON format expected:
    {
        "dispatches": {
            "<dispatch_name>": {
                "hardware_target": "GPU_0+GPU_1",   # or "CPU_0+CPU_1", etc.
                "start_time": 0.0,
                "duration": 4.8,
                "job_name": "dronet",
                "id": 0
            }, ...
        },
        "metadata": {
            "machine_combinations": [["CPU_0"], ["CPU_1"], ["CPU_0","CPU_1"], ...]
        }
    }
    """

    dispatches = schedule_data.get("dispatches", {})
    metadata = schedule_data.get("metadata", {})

    # ------------------------------------------------------------------ #
    # 1. Build individual core rows from metadata machines dict           #
    #    e.g. {"CPU": 2, "GPU": 2} -> ["CPU_0","CPU_1","GPU_0","GPU_1"]  #
    # ------------------------------------------------------------------ #
    machines_meta = metadata.get("machines", {})

    # Build ordered core list from metadata
    all_cores = []
    for machine_type, count in machines_meta.items():
        for i in range(count):
            all_cores.append(f"{machine_type}_{i}")

    # Fallback: infer cores from hardware_targets in dispatches
    if not all_cores:
        core_set = set()
        for d in dispatches.values():
            for core in d["hardware_target"].split("+"):
                core_set.add(core)
        # Sort: group by type, then index
        all_cores = sorted(core_set, key=lambda c: (c.rsplit("_", 1)[0], int(c.rsplit("_", 1)[1])))

    row_index = {core: i for i, core in enumerate(all_cores)}
    num_rows = len(all_cores)

    # ------------------------------------------------------------------ #
    # 2. Assign a color per job                                           #
    # ------------------------------------------------------------------ #
    job_names = sorted({d["job_name"] for d in dispatches.values()})
    highly_distinct_colors = [
        (0.2, 0.6, 1.0),   # Sky Blue
        (0.8, 0.35, 0.0),  # Orange-Brown
        (0.0, 0.72, 0.35), # Green
        (0.9, 0.1, 0.1),   # Red
        (0.6, 0.0, 0.85),  # Purple
        (0.95, 0.75, 0.0), # Yellow
        (0.0, 0.65, 0.85), # Cyan
        (0.85, 0.45, 0.65),# Pink
        (0.45, 0.28, 0.08),# Brown
        (0.4, 0.4, 0.4),   # Gray
    ]
    job_color = {}
    for i, jn in enumerate(job_names):
        job_color[jn] = highly_distinct_colors[i % len(highly_distinct_colors)]

    # ------------------------------------------------------------------ #
    # 3. Draw                                                             #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(14, max(4, num_rows * 1.2)))

    for dispatch_name, d in dispatches.items():
        hw = d["hardware_target"]
        start = d["start_time"]
        dur = d["duration"]
        job = d["job_name"]
        op_id = d.get("id", "")
        color = job_color[job]

        # Draw on every core in the hardware target
        cores = hw.split("+")
        valid_cores = [c for c in cores if c in row_index]
        if not valid_cores:
            print(f"Warning: no known cores in hardware_target '{hw}', skipping '{dispatch_name}'")
            continue

        is_multi = len(valid_cores) > 1

        for core in valid_cores:
            row = row_index[core]
            ax.broken_barh(
                [(start, dur)],
                (row - 0.4, 0.8),
                facecolors=color,
                edgecolor="black",
                linewidth=0.8,
            )

        # Draw label once, centred on the middle core
        mid_row = row_index[valid_cores[len(valid_cores) // 2]]
        if dur > 0:
            text_x = start + dur / 2
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "white" if brightness < 0.55 else "black"
            label = str(op_id)
            ax.text(
                text_x, mid_row, label,
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color=text_color,
            )

    # ------------------------------------------------------------------ #
    # 4. Axes, legend, save                                               #
    # ------------------------------------------------------------------ #
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(all_cores, fontsize=9)
    ax.set_ylim(-0.8, num_rows - 0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Hardware Target")
    ax.set_title(plot_title)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    legend_handles = [
        mpatches.Patch(facecolor=job_color[jn], edgecolor="black", label=jn)
        for jn in job_names
    ]
    ax.legend(
        handles=legend_handles,
        title="Jobs",
        loc="upper right",
        bbox_to_anchor=(1.18, 1),
        framealpha=0.9,
        fontsize=9,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close()


# ------------------------------------------------------------------ #
# Convenience: load from file and plot                               #
# ------------------------------------------------------------------ #
def plot_schedule_from_json_file(json_path: str, save_path: str = None, plot_title: str = None):
    with open(json_path) as f:
        data = json.load(f)

    if save_path is None:
        base = os.path.splitext(json_path)[0]
        save_path = base + "_plot.png"
    if plot_title is None:
        plot_title = os.path.basename(json_path)

    plot_schedule_from_json(data, save_path=save_path, plot_title=plot_title)


# ------------------------------------------------------------------ #
# Quick self-test using the embedded sample JSON                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    sample = {
        "dispatches": {
            "dronet_dispatch_0":  {"id": 0,  "hardware_target": "GPU_0+GPU_1", "start_time": 0.0,              "duration": 4.796, "job_name": "dronet"},
            "dronet_dispatch_1":  {"id": 1,  "hardware_target": "CPU_0+CPU_1", "start_time": 4.796,            "duration": 2.990, "job_name": "dronet"},
            "dronet_dispatch_2":  {"id": 2,  "hardware_target": "GPU_0+GPU_1", "start_time": 7.785,            "duration": 3.457, "job_name": "dronet"},
            "dronet_dispatch_3":  {"id": 3,  "hardware_target": "CPU_0+CPU_1", "start_time": 11.243,           "duration": 1.064, "job_name": "dronet"},
            "dronet_dispatch_4":  {"id": 4,  "hardware_target": "CPU_0+CPU_1", "start_time": 11.243,           "duration": 1.264, "job_name": "dronet"},
            "dronet_dispatch_5":  {"id": 5,  "hardware_target": "CPU_0+CPU_1", "start_time": 12.507,           "duration": 2.721, "job_name": "dronet"},
            "dronet_dispatch_6":  {"id": 6,  "hardware_target": "CPU_0+CPU_1", "start_time": 15.228,           "duration": 1.967, "job_name": "dronet"},
            "dronet_dispatch_7":  {"id": 7,  "hardware_target": "CPU_0+CPU_1", "start_time": 15.228,           "duration": 2.153, "job_name": "dronet"},
            "dronet_dispatch_8":  {"id": 8,  "hardware_target": "GPU_0+GPU_1", "start_time": 17.382,           "duration": 2.154, "job_name": "dronet"},
            "dronet_dispatch_9":  {"id": 9,  "hardware_target": "CPU_0+CPU_1", "start_time": 19.535,           "duration": 3.506, "job_name": "dronet"},
            "dronet_dispatch_10": {"id": 10, "hardware_target": "GPU_0+GPU_1", "start_time": 19.535,           "duration": 3.845, "job_name": "dronet"},
            "dronet_dispatch_11": {"id": 11, "hardware_target": "CPU_0+CPU_1", "start_time": 23.380,           "duration": 3.916, "job_name": "dronet"},
            "dronet_dispatch_12": {"id": 12, "hardware_target": "CPU_0+CPU_1", "start_time": 27.296,           "duration": 4.455, "job_name": "dronet"},
            "dronet_dispatch_13_1":{"id":13, "hardware_target": "GPU_0+GPU_1", "start_time": 31.751,           "duration": 4.560, "job_name": "dronet"},
            "dronet_dispatch_13_2":{"id":13, "hardware_target": "CPU_0+CPU_1", "start_time": 30.067,           "duration": 1.929, "job_name": "dronet"},
            "dronet_dispatch_14": {"id": 14, "hardware_target": "GPU_0+GPU_1", "start_time": 27.296,           "duration": 2.771, "job_name": "dronet"},
            "mlp0_dispatch_0":    {"id": 0,  "hardware_target": "GPU_0+GPU_1", "start_time": 0.0,              "duration": 1.798, "job_name": "mlp0"},
            "mlp0_dispatch_1":    {"id": 1,  "hardware_target": "CPU_0+CPU_1", "start_time": 1.798,            "duration": 1.422, "job_name": "mlp0"},
            "mlp0_dispatch_2":    {"id": 2,  "hardware_target": "GPU_0+GPU_1", "start_time": 22.000,           "duration": 3.000, "job_name": "mlp0"},
            "mlp1_dispatch_0":    {"id": 0,  "hardware_target": "GPU_0+GPU_1", "start_time": 25.0,             "duration": 1.798, "job_name": "mlp1"},
            "mlp1_dispatch_1":    {"id": 1,  "hardware_target": "CPU_0+CPU_1", "start_time": 26.798,           "duration": 1.422, "job_name": "mlp1"},
            "mlp1_dispatch_2":    {"id": 2,  "hardware_target": "GPU_0+GPU_1", "start_time": 47.000,           "duration": 3.000, "job_name": "mlp1"},
        },
        "metadata": {
            "machine_combinations": [
                ["CPU_0"], ["CPU_1"], ["CPU_0", "CPU_1"],
                ["GPU_0"], ["GPU_1"], ["GPU_0", "GPU_1"],
            ]
        }
    }
    plot_schedule_from_json(
        sample,
        save_path="/mnt/user-data/outputs/schedule_fixed.png",
        plot_title="Dronet + Mlp0 + Mlp1 Schedule (Fixed)",
    )

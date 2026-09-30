#!/usr/bin/env python3
"""Turn the CP-SAT single-frame coupled schedule into a periodic one by tiling it at the camera
period, with control as an independent periodic task.

    scripts/tile_schedule.py schedules/cmp_coupled_cpsat_board.json --period 40 --frames 25 \\
        --ctrl-period 10 --out schedules/tiled_coupled_25hz.json

The single-frame solve is the solver's placement for one camera frame: which harts each YOLO
dispatch shards across, and when nav follows. A periodic schedule is that placement repeated
every period, frame k offset by k * period, with the camera->YOLO->nav data edges kept inside
each frame. Control is placed as its own periodic task -- one instance per control period,
carrying only its internal dependencies -- because in the deployed pipeline control acts on the
newest goal available at its own release rather than waiting for a particular frame. Nothing in
the per-dispatch placement is changed: same harts, same widths, same durations.
"""
from __future__ import annotations
import argparse, copy, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "xpu-rt"))
from job_names import split_job_name   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schedule"); ap.add_argument("--period", type=float, default=40.0)
    ap.add_argument("--frames", type=int, default=25); ap.add_argument("--ctrl-period", type=float, default=10.0)
    ap.add_argument("--ctrl-net", default="mlp_control"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = json.load(open(a.schedule)); disp = src["dispatches"]
    known = {split_job_name(v["job_name"], {"yolov8_nano_64x96", "fused_full", "mlp_control"})[0] for v in disp.values()}
    # split the single frame into the perception chain and the control task
    chain = {k: v for k, v in disp.items() if split_job_name(v["job_name"], known)[0] != a.ctrl_net}
    ctrl = {k: v for k, v in disp.items() if split_job_name(v["job_name"], known)[0] == a.ctrl_net}
    ctrl_t0 = min(v["start_time"] for v in ctrl.values())
    horizon = a.period * a.frames
    n_ctrl = int(round(horizon / a.ctrl_period))

    def rekey(k, inst):
        net, _ = split_job_name(k.split("_dispatch_")[0], known)
        return f"{net}{inst}_dispatch_{k.split('_dispatch_')[1]}"

    out = {}
    for f in range(a.frames):
        off = f * a.period
        for k, v in chain.items():
            e = copy.deepcopy(v)
            net, _ = split_job_name(v["job_name"], known)
            e["job_name"] = f"{net}{f}"
            e["start_time"] = round(v["start_time"] + off, 6)
            e["dependencies"] = [rekey(d, f) for d in v.get("dependencies", []) if split_job_name(d.split("_dispatch_")[0], known)[0] != a.ctrl_net]
            if e.get("time_dependency"):
                e["time_dependency"] = rekey(e["time_dependency"], f)
            if "release_us" in e:
                e["release_us"] = round(v.get("release_us", 0.0) + off * 1000.0, 3)
            e.pop("deadline_miss", None); e.pop("deadline_overrun_us", None)
            out[rekey(k, f)] = e
    for j in range(n_ctrl):
        off = j * a.ctrl_period
        for k, v in ctrl.items():
            e = copy.deepcopy(v)
            e["job_name"] = f"{a.ctrl_net}{j}"
            e["start_time"] = round(v["start_time"] - ctrl_t0 + off, 6)
            # only the task's internal edges survive: control does not wait for a frame
            e["dependencies"] = [rekey(d, j) for d in v.get("dependencies", []) if split_job_name(d.split("_dispatch_")[0], known)[0] == a.ctrl_net]
            if e.get("time_dependency"):
                td = e["time_dependency"]
                e["time_dependency"] = rekey(td, j) if split_job_name(td.split("_dispatch_")[0], known)[0] == a.ctrl_net else None
                if e["time_dependency"] is None:
                    e.pop("time_dependency")
            e["release_policy"] = "phase_locked" if not e["dependencies"] else e.get("release_policy", "immediate")
            e["release_us"] = round(off * 1000.0, 3)
            e.pop("deadline_miss", None); e.pop("deadline_overrun_us", None)
            out[rekey(k, j)] = e
    mk = max(v["start_time"] + v["duration"] for v in out.values())
    meta = dict(src.get("metadata", {}))
    meta.update({"makespan": round(mk, 4), "tiled_from": a.schedule, "camera_period_ms": a.period, "frames": a.frames,
                 "ctrl_period_ms": a.ctrl_period, "ctrl_instances": n_ctrl,
                 "solver": "cpsat single-frame, tiled at the camera period; control independent periodic",
                 "chain_nodes": ["yolov8_nano_64x96", "fused_full"]})
    for k in ("deadline_miss_count", "per_net_deadline_miss", "total_lateness_ms", "end_to_end_latency_ms",
              "end_to_end_deadline_miss", "yolo_frame_latency_ms"):
        meta.pop(k, None)
    json.dump({"dot_file": src.get("dot_file"), "dispatches": out, "metadata": meta}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(out)} dispatches, {a.frames} frames at {a.period} ms, {n_ctrl} control instances, makespan {mk:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())

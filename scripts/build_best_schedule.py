#!/usr/bin/env python3
"""Explicit XPU-RT placements for the camera->YOLO->nav->control pipeline that use what the
runtime offers -- per-dispatch sharding, cross-frame pipelining across the two clusters, and a
dedicated hart for control -- written as ordinary schedule JSON so the same ingest, walker and
trace path execute them.

    scripts/build_best_schedule.py --layout p4 --camera-hz 25 --seconds 1 --out schedules/best25.json
    scripts/build_best_schedule.py --layout alt --camera-hz 45 --seconds 1 --out schedules/best45.json

Layouts:
  p4    every frame's YOLO sharded 4-way on CPU_P#0-3 (convs whose OC divides by 4; the rest on
        CPU_P#0); nav on CPU_E#0; control on CPU_E#1 alone, phase-locked every control period.
  alt   even frames on CPU_P#0-3, odd frames on CPU_E#0-3, so two frames are in flight; nav on
        the frame's cluster hart #3 after its YOLO; control on CPU_E#3 alone... which is inside
        the odd-frame pool, so the odd pool is CPU_E#0-2 at width 2 (CPU_E#0+CPU_E#1) for convs
        and CPU_E#2 for the rest -- control keeps a hart of its own in both layouts.
  alt4  as alt but the odd pool is the full CPU_E#0-3 and control shares CPU_P#3 with even-frame
        shards; measures what sharing costs control.
  alt2  both clusters at width 2 (P#0+P#1 / E#0+E#1, the rest of each frame on #2), nav on
        CPU_P#3, control alone on CPU_E#3 -- two frames in flight and a dedicated control hart.
  alt4s alt4 with each control instance placed on hart #3 of whichever cluster is not running a
        YOLO frame at its release, so control rarely queues behind a shard.
  alt1  six frames in flight, each whole on one hart (P#0-2, E#0-2 round-robin), nav on P#3,
        control alone on E#3: the throughput end of the width / frames-in-flight trade.

Per-dispatch structure (dependencies, op names, OC) comes from the CP-SAT single-frame solve,
which was itself built from the network's dispatch graph; durations for the schedule's
predicted fields come from the measured shard profiles when present (they gate nothing but the
frame release, which is phase-locked to the camera). Every dispatch of frame k carries the
frame's release time as its start, so within a frame execution is dependency-driven.
"""
from __future__ import annotations
import argparse, copy, csv, glob, json, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
from job_names import split_job_name   # noqa: E402

SRC = "schedules/cmp_coupled_cpsat_board.json"
EXTRA_SRC = "schedules/scheduled_w5_ffn_dronet_yolo_r2_shard_greedy_profiled.json"   # dispatch templates for the heavier stack
NETS = {"yolov8_nano_64x96", "fused_full", "mlp_control", "ffn_block", "dronet"}
PACKED = ("conv2d",)


def oc_of(module):
    m = re.search(r"[x_]OC(\d+)", module or "")
    return int(m.group(1)) if m else 0


def profile(topo):
    """dispatch_id -> ms from the shard profile tree for one hart set, if measured."""
    out = {}
    for p in glob.glob(f"ModelBlaster/gen/profile_shard/*/*/yolov8_nano_64x96/*/*/topo_{topo}/results.csv"):
        for r in csv.DictReader(open(p)):
            try:
                out[int(r["dispatch_id"])] = float(r["mean_time"]) if r.get("mean_unit", "ms") == "ms" else float(r["mean_time_ns"]) / 1e6
            except (KeyError, ValueError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["p4", "alt", "alt4", "alt2", "alt4s", "alt1"], required=True)
    ap.add_argument("--camera-hz", type=float, required=True); ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--ctrl-period", type=float, default=10.0); ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default="", help="heavier stack: net:hz:hart[,...] e.g. ffn_block:10:CPU_P#3,dronet:30:CPU_E#3")
    ap.add_argument("--ime-nets", default="", help="comma list of nets whose dispatches run on the matrix engine (impl=ime; P-cluster harts only)")
    ap.add_argument("--cameras", type=int, default=1, help="2 = two cameras at --camera-hz each: frames 2k and 2k+1 are released together")
    a = ap.parse_args()
    src = json.load(open(SRC))["dispatches"]
    by_net = {n: {k: v for k, v in src.items() if split_job_name(v["job_name"], NETS)[0] == n} for n in NETS}
    period = 1000.0 / a.camera_hz
    frames = int(round(a.seconds * 1000.0 / period)) * a.cameras; n_ctrl = int(round(a.seconds * 1000.0 / a.ctrl_period))
    release_of = (lambda f: (f // a.cameras) * period)      # paired releases when there are two cameras
    prof4, prof2, prof1 = profile("0_1_2_3"), profile("0_1"), profile("0")
    e_slow = 1.35   # predicted-field scale for the E cluster; the trace measures the truth

    busy_ms = 27.0    # a 4-way frame occupies its cluster for about this long (measured 24.6-25.6)

    def target(net, did, module, frame):
        if net == "mlp_control":
            if a.layout == "alt4s":
                # control instance `frame` fires at frame*ctrl_period: use hart #3 of whichever cluster
                # is not running a YOLO frame at that moment (even frames on P, odd on E)
                t = frame * a.ctrl_period; f = int(t // period)
                cluster_busy = "P" if f % 2 == 0 else "E"
                return "CPU_E#3" if (t < f * period + busy_ms and cluster_busy == "P") else "CPU_P#3"
            return {"p4": "CPU_E#1", "alt": "CPU_E#3", "alt4": "CPU_P#3", "alt2": "CPU_E#3", "alt1": "CPU_E#3"}[a.layout]
        if net == "fused_full":
            if a.layout == "p4":
                return "CPU_E#0"
            if a.layout in ("alt2", "alt1"):
                return "CPU_P#3"
            return "CPU_P#3" if (frame % 2 == 0 or a.layout == "alt") else "CPU_E#3"
        packed = any(module.split("_rvv_x60_")[-1].startswith(p) for p in PACKED) if "_rvv_x60_" in module else False
        oc = oc_of(module)
        if a.layout == "alt2":
            c = "P" if frame % 2 == 0 else "E"
            return f"CPU_{c}#0+CPU_{c}#1" if packed and oc % 2 == 0 else f"CPU_{c}#2"
        if a.layout == "alt1":
            # six frames in flight, one hart each: P#0, P#1, P#2, E#0, E#1, E#2 round-robin;
            # nav on P#3, control alone on E#3. Latency of a 1-hart frame, throughput of six.
            slot = ["CPU_P#0", "CPU_E#0", "CPU_P#1", "CPU_E#1", "CPU_P#2", "CPU_E#2"][frame % 6]
            return slot
        if a.layout == "p4" or frame % 2 == 0:
            return "CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3" if packed and oc % 4 == 0 else "CPU_P#0"
        if a.layout in ("alt4", "alt4s"):
            return "CPU_E#0+CPU_E#1+CPU_E#2+CPU_E#3" if packed and oc % 4 == 0 else "CPU_E#0"
        return "CPU_E#0+CPU_E#1" if packed and oc % 2 == 0 else "CPU_E#2"

    def dur(net, did, tgt, v):
        w = tgt.count("+") + 1; base = v["duration"]
        if net == "yolov8_nano_64x96":
            base = {4: prof4, 2: prof2, 1: prof1}[w].get(did, base) if w in (4, 2, 1) else base
        return base * (e_slow if "CPU_E" in tgt else 1.0)

    def rekey(k, inst):
        net, _ = split_job_name(k.split("_dispatch_")[0], NETS)
        return f"{net}{inst}_dispatch_{k.split('_dispatch_')[1]}"

    out = {}
    for f in range(frames):
        rel = release_of(f)
        for net in ("yolov8_nano_64x96", "fused_full"):
            for k, v in by_net[net].items():
                e = copy.deepcopy(v); did = int(v["id"]); tgt = target(net, did, v.get("module_name", ""), f)
                e.update({"job_name": f"{net}{f}", "hardware_target": tgt, "start_time": round(rel, 6),
                          "duration": round(dur(net, did, tgt, v), 6),
                          "dependencies": [rekey(d, f) for d in v.get("dependencies", []) if split_job_name(d.split("_dispatch_")[0], NETS)[0] != "mlp_control"],
                          "release_policy": "phase_locked" if not v.get("dependencies") else "immediate",
                          "release_us": round(rel * 1000.0, 3)})
                if e.get("time_dependency"):
                    e["time_dependency"] = rekey(e["time_dependency"], f)
                e.pop("deadline_miss", None); e.pop("deadline_overrun_us", None)
                out[rekey(k, f)] = e
    ctrl = by_net["mlp_control"]; c0 = min(v["start_time"] for v in ctrl.values())
    for j in range(n_ctrl):
        rel = j * a.ctrl_period
        for k, v in ctrl.items():
            e = copy.deepcopy(v); tgt = target("mlp_control", int(v["id"]), "", j)
            deps = [rekey(d, j) for d in v.get("dependencies", []) if split_job_name(d.split("_dispatch_")[0], NETS)[0] == "mlp_control"]
            e.update({"job_name": f"mlp_control{j}", "hardware_target": tgt, "start_time": round(rel + (v["start_time"] - c0), 6),
                      "dependencies": deps, "release_policy": "phase_locked" if not deps else "immediate",
                      "release_us": round(rel * 1000.0, 3)})
            td = e.get("time_dependency")
            if td:
                if split_job_name(td.split("_dispatch_")[0], NETS)[0] == "mlp_control":
                    e["time_dependency"] = rekey(td, j)
                else:
                    e.pop("time_dependency")
            e.pop("deadline_miss", None); e.pop("deadline_overrun_us", None)
            out[rekey(k, j)] = e
    # the heavier stack: independent periodic nets, whole on one hart each, their own internal edges
    extras = []
    if a.extra:
        ex_src = json.load(open(EXTRA_SRC))["dispatches"]
        for item in a.extra.split(","):
            net, hz, hart = item.split(":"); hz = float(hz); per = 1000.0 / hz
            tmpl = {k: v for k, v in ex_src.items() if split_job_name(v["job_name"], NETS)[0] == net and split_job_name(v["job_name"], NETS)[1] in (0, "0")}
            n_inst = int(round(a.seconds * 1000.0 / per)); extras.append((net, hz, hart, n_inst))
            t0 = min(v["start_time"] for v in tmpl.values())
            for j in range(n_inst):
                rel = j * per
                for k, v in tmpl.items():
                    e = copy.deepcopy(v)
                    deps = [rekey(d, j) for d in v.get("dependencies", []) if split_job_name(d.split("_dispatch_")[0], NETS)[0] == net]
                    e.update({"job_name": f"{net}{j}", "hardware_target": hart, "start_time": round(rel + (v["start_time"] - t0), 6),
                              "dependencies": deps, "release_policy": "phase_locked" if not deps else "immediate",
                              "release_us": round(rel * 1000.0, 3)})
                    td = e.get("time_dependency")
                    if td:
                        if split_job_name(td.split("_dispatch_")[0], NETS)[0] == net: e["time_dependency"] = rekey(td, j)
                        else: e.pop("time_dependency")
                    e.pop("deadline_miss", None); e.pop("deadline_overrun_us", None)
                    # the template schedule chose IME for some transformer dispatches; this build is
                    # RVV on both clusters, the same kernels the ROS node runs
                    for key in ("implementation", "impl", "chosen_impl", "backend"):
                        e.pop(key, None)
                    if net in [n for n in a.ime_nets.split(",") if n]:
                        e["impl"] = "ime"          # the matrix engine, a per-dispatch backend choice only the runtime exposes
                    out[rekey(k, j)] = e
    mk = max(v["start_time"] + v["duration"] for v in out.values())
    meta = {"makespan": round(mk, 4), "solver": f"explicit {a.layout} placement, camera {a.camera_hz:g} Hz" + (f" + {a.extra}" if a.extra else "") + (f" [ime: {a.ime_nets}]" if a.ime_nets else ""),
            "ime_nets": a.ime_nets,
            "extra": extras, "cameras": a.cameras,
            "layout": a.layout, "camera_period_ms": period, "frames": frames, "ctrl_period_ms": a.ctrl_period,
            "ctrl_instances": n_ctrl, "structure_from": SRC, "chain_nodes": ["yolov8_nano_64x96", "fused_full"],
            "predicted_fields_note": "durations from gen/profile_shard where measured; E-cluster scaled x1.35; gates nothing but the frame release"}
    json.dump({"dot_file": None, "dispatches": out, "metadata": meta}, open(a.out, "w"), indent=1)
    import collections
    hist = collections.Counter(v["hardware_target"] for v in out.values() if v["job_name"].startswith("yolov8"))
    print(f"wrote {a.out}: {len(out)} dispatches, {frames} frames @ {a.camera_hz:g} Hz, {n_ctrl} control; yolo targets {dict(hist)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

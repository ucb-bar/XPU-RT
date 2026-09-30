#!/usr/bin/env python3
"""The ANALYTICAL ROS 2 per-node-pinning baseline: the model, its inputs, and its numbers.

`measured_timing.py` is the single source of truth for numbers that came off the SpaceMiT K1.
This module is its counterpart for the analytical (Tier A, docs/Baselines/ros_baseline_tiers.md) ROS 2
arrangement the warehouse showdown was submitted with (`plots/fig_hil_showdown.pdf`, on disk as
`results/codesign_feedback/refined/warehouse_showdown_story.pdf`). Its panel I is a schedule
MODEL, not a board trace. The submitted render's panel I is titled "Onboard K1 schedule -- ..." with
the axis "onboard schedule time (ms) . K1 board" and carries no model marker; the current composer's
form of that panel prints the footer "Calibrated model, not a board trace"
(`sims/scripts/compose_warehouse_showdown.py:379`). Provenance: docs/Baselines/ros_baseline_tiers.md.
Everything that model claims is re-derived here from stated inputs, so the modelled arm is
reproducible on the same terms as the measured ones rather than surviving only as a committed
schedule JSON.

    from ros_pinning_model import DEPLOYED_NODES, model, zoh_rate_hz, provenance

Re-derive everything with:  scripts/ros_pinning_model.py --verify


THE MODEL
---------
One ROS 2 node per network. Each node owns a STATIC partition of harts and is released by its
own periodic timer at k*T. Inside the node the network's whole dispatch graph runs SEQUENTIALLY:
a node is a process with a single executor thread, so the graph is not spread across harts the
way the scheduler spreads it. Per-dispatch durations are the SAME measured K1 numbers the
XPU-RT arm pays, so the two arms differ only in placement policy, never in kernel cost.

For a node with one-hart serial compute C, period T and instance index k:

    release_k = k * T
    start_k   = k * T                                     (executor policy "release")
    start_k   = max(k * T, finish_{k-1})                  (executor policy "queued")
    finish_k  = start_k + C
    response_k = finish_k - release_k
    miss_k    = response_k > T

Two executor policies are offered: the submitted schedule places instances by release time and the
script that shares its name serialises them, and both are computed so the difference is visible:

  * "release" places every instance at its release time even when the partition is still busy
    with the previous instance. This is what `schedules/scheduled_ros_partition_deployed.json`
    -- the schedule the submitted figure drew -- actually contains: YOLO instances start at
    0, 22, 44, 66, 88 ms while each takes 24.353 ms, so consecutive frames overlap on the same
    partition. The per-frame overrun is real and drawn; the growing queue is not modelled.
  * "queued" serialises instances on the partition, which is what a single-threaded executor
    does and what `scripts/ros_pinning_periodic.py` writes. It is the stricter reading and it
    makes the baseline slower than the "release" reading the submitted figure drew.

The chain quantity the figures quote is camera -> control:

    chain = response(perception) + C(nav) + C(control)            control in the goal callback
    chain = response(perception) + C(nav) + wait + C(control)     control on its own timer,
                                                                  wait = the slack to the next tick

and the command cadence a chain of that length can sustain under the zero-order hold at the
simulator's control step dt (`draw_control_rate_panel`, compose_warehouse_showdown.py:270):

    rate_hz = 1000 / (dt * ceil(response / dt))


ASSUMPTIONS, EVERY ONE OF THEM
------------------------------
 1. Per-dispatch durations are the K1-calibrated numbers of the source schedule. The model adds
    no compute of its own and removes none.
 2. A node's graph is serial. No intra-node sharding, no cross-hart split of one network.
 3. A node's partition is static for the whole horizon. Nothing migrates.
 4. Releases are strictly periodic and jitter-free.
 5. MIDDLEWARE IS FREE. No DDS serialisation, no message copy, no executor wake-up, no callback
    jitter is charged. `scripts/ros2_middleware_tax_k1.py` exists to bound exactly this omission
    and measures the tax on the board; the omission makes the modelled arm FASTER than a real
    ROS 2 deployment, so every "the baseline misses" claim built on it is conservative.
 6. But the model also assumes one executor thread per node. A real deployment with a
    MultiThreadedExecutor, callback groups or composed nodes can overlap what this serialises
    and would be FASTER than the model. That direction is NOT bounded by the tax measurement,
    and it is the reason the measured ROS 2 arms exist at all
    (`docs/Baselines/ros_arms_catalog.md`, `measured_timing.ROS_VANILLA`).
 7. A wider perception partition helps only through a MEASURED whole-net speedup table, never
    through core count. Sharding this detector is inefficient (1.42x at eight harts, and four
    harts is slower than two), and that inefficiency is applied in the baseline's favour.
 8. Deadline = period. A frame is late when its response exceeds its own release period.


WHAT IS ANALYTICAL HERE AND WHAT IS MEASURED
--------------------------------------------
MEASURED   per-dispatch durations (K1, ModelBlaster kernels, `gen/mb` profiles); the perception
           sharding speedup table; the middleware tax, when it is quoted.
ANALYTICAL the placement (which net on which harts), the release/queue arithmetic, every
           response and makespan that follows from it, the chain latency, and the command
           cadence the ZOH rule turns a response into.
NEITHER    the two worst-case control responses the submitted figure's panel A and B labels rest
           on -- 4.89 ms for XPU-RT and 12.40 ms for the baseline, giving the "100 Hz vs 50 Hz"
           captions. They are inputs to the composer, not outputs of either schedule named in
           the same sidecar: a prediction of this policy on the four-network workload, in
           `results/codesign_feedback/microros_baseline_k1/microros_baseline_k1.json`. `verify()`
           reports them as inputs. See `docs/Evaluation/showdown_analytical_reproduction.md` §6.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results/codesign_feedback")
sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
from job_names import split_job_name            # noqa: E402

CONTROL_DT_MS = 10.0          # the simulator's control step; cadence = 1000 / (dt * ceil(resp/dt))
PLATFORM_HARTS = 8            # the K1: four P harts and four E harts
# the order a widening partition grows through, so a width the P cluster cannot hold spills into
# the E cluster rather than inventing a CPU_P#4 the board does not have
PLATFORM_HART_ORDER = ("CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3",
                       "CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3")

# --- the model's inputs ------------------------------------------------------------------------
# The deployed flight stack as three ROS 2 nodes. compute_ms is the node's whole graph run
# sequentially on one hart, in the K1-calibrated durations of the source schedule; period_ms is
# the node's timer; harts is its static partition; instances is the horizon.
#
# These are the inputs, not outputs: `derive()` re-reads each one out of
# schedules/scheduled_ros_partition_deployed.json and reports drift if the artifact moved.
DEPLOYED_NODES = {
    "mlp_control":       {"role": "control",    "compute_ms": 0.083000, "period_ms": 10.0,
                          "instances": 12, "harts": ("CPU_E#0",)},
    "fused_full":        {"role": "nav",        "compute_ms": 3.621874, "period_ms": 20.0,
                          "instances": 6,  "harts": ("CPU_E#1",)},
    "yolov8_nano_64x96": {"role": "perception", "compute_ms": 24.353374, "period_ms": 22.0,
                          "instances": 5,  "harts": ("CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3")},
}

# MEASURED standalone on this board with OC-sharding at codegen (gen/profile_shard), the same
# table `scripts/ros_pinning_generic.py` applies. Four harts are slower than two: that is the
# kernel's property and it is charged to the baseline in the baseline's favour.
PERCEPTION_SPEEDUP = {1: 1.00, 2: 55.870 / 50.384, 4: 55.870 / 52.294, 8: 55.870 / 39.386}

# --- what the submitted figure prints --------------------------------------------------------
# refined/warehouse_showdown_story.pdf, panel I: "XPU-RT done 40 ms", "ROS still backlogged
# 112 ms -> CRASH", "CP-SAT . 8 cores" against "static . 6 cores" with two lanes reading
# "idle -- core unused", and the 22 ms YOLO deadline rules the baseline crosses every frame.
SUBMITTED = {
    "ros_schedule": "schedules/scheduled_ros_partition_deployed.json",
    "xpu_schedule": "schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json",
    "ros_makespan_ms": 112.353374,      # drawn as "112 ms"
    "xpu_makespan_ms": 40.378875,       # drawn as "40 ms"
    "ros_perception_response_ms": 24.353374,
    "ros_perception_period_ms": 22.0,
    "ros_harts_used": 6,
    "ros_harts_idle": 2,
    "executor": "release",
    "source": "sims/scripts/compose_warehouse_showdown.py + results/codesign_feedback/"
              "refined_src/showdown_recovered.py, over the two schedules above",
}

# The standalone modelled schedule panel, drawn from the board-recost pair and carrying a
# sidecar: results/codesign_feedback/warehouse_schedule_board_metrics.json.
MATCHED_BOARD = {
    "ros_schedule": "schedules/scheduled_ros_partition_deployed_matched_board.json",
    "xpu_schedule": "schedules/scheduled__flight_deployed_matched_board_cpsat_profiled.json",
    "sidecars": ("results/codesign_feedback/warehouse_schedule_board_metrics.json",
                 "results/codesign_feedback/warehouse_showdown_board_metrics.json"),
    "frame_instance_zero_based": 2,
    "budget_ms": 23.0,
    "xpu_response_ms": 22.999777,
    "ros_response_ms": 49.764795,
    "ros_makespan_ms": 153.523670,
    "ros_perception_work_ms": 30.704700,   # the recost per-frame compute
    "source": "one interior perception frame, release to output, across both schedules",
}

# The submitted figure's "100 Hz vs 50 Hz" labels, recorded in
# results/codesign_feedback/refined_src/warehouse_regen_metrics.json. NOT derived from either
# schedule in that same sidecar -- see the module docstring, "NEITHER".
CONTROL_RATE_PANEL = {
    "sidecar": "results/codesign_feedback/refined_src/warehouse_regen_metrics.json",
    "xpu_worst_response_ms": 4.89,
    "ros_worst_response_ms": 12.40,
    "xpu_control_rate_hz": 100.0,
    "ros_control_rate_hz": 50.0,
    "control_period_ms": 10.0,
    "ros_response_provenance": "results/codesign_feedback/microros_baseline_k1/microros_baseline_k1.json",
}

_KNOWN = set(DEPLOYED_NODES)


def provenance() -> str:
    return ("analytical ROS 2 per-node-pinning baseline: one node per network, static partition, "
            "serial graph, periodic release, K1-calibrated per-dispatch durations, middleware "
            "free. Re-derive with scripts/ros_pinning_model.py --verify")


# --- the model ---------------------------------------------------------------------------------
def pin_timeline(compute_ms: float, period_ms: float, instances: int,
                 executor: str = "release") -> list[dict]:
    """One node's instances under the pinning policy. See the module docstring for the recurrence."""
    out, free = [], 0.0
    for k in range(int(instances)):
        release = k * period_ms
        start = release if executor == "release" else max(release, free)
        finish = start + compute_ms
        free = finish
        out.append({"instance": k, "release_ms": release, "start_ms": start, "finish_ms": finish,
                    "response_ms": finish - release, "late": (finish - release) > period_ms + 1e-9})
    return out


def _widen(base_hart: str, width: int) -> tuple:
    """`width` harts starting at `base_hart`, following the platform's own hart order."""
    order = list(PLATFORM_HART_ORDER)
    if base_hart not in order:
        return tuple(f"{base_hart}+{i}" for i in range(width)) if width > 1 else (base_hart,)
    i = order.index(base_hart)
    if i + width > len(order):
        raise SystemExit(f"a {width}-hart partition starting at {base_hart} does not fit on "
                         f"{len(order)} harts")
    return tuple(order[i:i + width])


def zoh_rate_hz(response_ms: float, control_dt_ms: float = CONTROL_DT_MS) -> float:
    """Command cadence a response of this length sustains under a zero-order hold at `control_dt_ms`."""
    return 1000.0 / (control_dt_ms * max(1, math.ceil(response_ms / control_dt_ms - 1e-9)))


def model(nodes: dict = None, executor: str = "release", control_dt_ms: float = CONTROL_DT_MS,
          platform_harts: int = PLATFORM_HARTS, perception_width: int = None,
          perception_speedup: float = None, camera_hz: float = None,
          control_on_timer: bool = False) -> dict:
    """Run the analytical baseline and return everything it claims.

    `camera_hz` overrides the perception node's period (the camera is what releases it).
    `perception_width` re-costs the perception node through the MEASURED speedup table; the
    partition is widened to match, so the hart accounting follows the same lever.
    """
    nodes = {k: dict(v) for k, v in (nodes or DEPLOYED_NODES).items()}
    perception = [n for n, v in nodes.items() if v.get("role") == "perception"]
    speedup_applied = 1.0
    if perception_width:
        speedup_applied = (perception_speedup if perception_speedup
                           else PERCEPTION_SPEEDUP.get(int(perception_width), 1.0))
        for n in perception:
            nodes[n]["compute_ms"] = nodes[n]["compute_ms"] / speedup_applied
            nodes[n]["harts"] = _widen(nodes[n]["harts"][0], int(perception_width))
    if camera_hz:
        for n in perception:
            nodes[n]["period_ms"] = 1000.0 / float(camera_hz)

    timelines = {n: pin_timeline(v["compute_ms"], v["period_ms"], v["instances"], executor)
                 for n, v in nodes.items()}
    counts = collections.Counter(h for v in nodes.values() for h in v["harts"])
    used = sorted(counts)
    overlap = sorted(h for h, c in counts.items() if c > 1)
    makespan = max((t[-1]["finish_ms"] for t in timelines.values() if t), default=0.0)

    per_node = {}
    for n, v in nodes.items():
        t = timelines[n]
        per_node[n] = {
            "role": v.get("role", "other"), "harts": list(v["harts"]),
            "compute_ms": v["compute_ms"], "period_ms": v["period_ms"], "instances": len(t),
            "response_worst_ms": max((x["response_ms"] for x in t), default=0.0),
            "response_first_ms": t[0]["response_ms"] if t else 0.0,
            "late_instances": sum(1 for x in t if x["late"]),
            "sustainable_rate_hz": (1000.0 / v["compute_ms"]) if v["compute_ms"] > 0 else None,
        }

    def _one(role):
        hits = [n for n, v in nodes.items() if v.get("role") == role]
        return hits[0] if hits else None

    p, nav, ctrl = _one("perception"), _one("nav"), _one("control")
    chain = None
    if p:
        chain = per_node[p]["response_worst_ms"]
        if nav:
            chain += nodes[nav]["compute_ms"]
        if ctrl:
            if control_on_timer:
                tick = nodes[ctrl]["period_ms"]
                chain = math.ceil(chain / tick - 1e-9) * tick
            chain += nodes[ctrl]["compute_ms"]

    return {
        "executor": executor, "control_dt_ms": control_dt_ms,
        "platform_harts": platform_harts, "harts_used": used,
        "harts_used_n": len(used), "harts_idle_n": max(0, platform_harts - len(used)),
        "harts_shared_by_two_nodes": overlap,
        "perception_speedup_applied": speedup_applied,
        "makespan_ms": makespan, "nodes": per_node, "timelines": timelines,
        "chain_camera_to_control_ms": chain,
        "control_on_timer": control_on_timer,
        "command_rate_hz": zoh_rate_hz(chain, control_dt_ms) if chain else None,
        "perception_sustainable_camera_hz": (per_node[p]["sustainable_rate_hz"] if p else None),
    }


# --- reading the artifacts the submitted figure drew --------------------------------------------
def _dispatches(path: str) -> dict:
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.exists(p):
        return {}
    return json.load(open(p)).get("dispatches", {})


def _by_instance(disp: dict) -> dict:
    g = collections.defaultdict(list)
    for v in disp.values():
        n, i = split_job_name(v["job_name"], _KNOWN)
        g[(n, i)].append(v)
    return g


def schedule_facts(path: str) -> dict:
    """Makespan, per-node one-hart work, per-instance response and hart usage, from a schedule JSON."""
    disp = _dispatches(path)
    if not disp:
        return {}
    g = _by_instance(disp)
    nets = collections.defaultdict(dict)
    for (n, i), vs in g.items():
        nets[n][i] = {"start_ms": min(v["start_time"] for v in vs),
                      "finish_ms": max(v["start_time"] + v["duration"] for v in vs),
                      "work_ms": sum(v["duration"] for v in vs),
                      "harts": sorted({h for v in vs for h in v["hardware_target"].split("+")})}
    harts = sorted({h for v in disp.values() for h in v["hardware_target"].split("+")})
    return {"path": path, "n_dispatches": len(disp),
            "makespan_ms": max(v["start_time"] + v["duration"] for v in disp.values()),
            "harts_used": harts, "harts_used_n": len(harts), "nets": {k: v for k, v in nets.items()}}


def frame_response(path: str, instance: int, period_ms: float, net="yolov8_nano_64x96"):
    """One perception frame's release-to-output response, the statistic the modelled panel prints.

    This is `_frame_window` + `_hart_blocks`'s `wall` of compose_warehouse_showdown.py: the frame's
    release is instance*period and its output is the last dispatch of that instance to finish, so
    the response includes whatever backlog and interfering work sits in between.
    """
    f = schedule_facts(path).get("nets", {}).get(net)
    if not f or instance not in f:
        return None
    return f[instance]["finish_ms"] - instance * period_ms


def derive() -> dict:
    """Recompute every recorded number, from the artifacts and from the model."""
    d: dict = {"missing": []}

    # (1) the submitted pair, straight off disk
    ros = schedule_facts(SUBMITTED["ros_schedule"])
    xpu = schedule_facts(SUBMITTED["xpu_schedule"])
    if ros:
        d["submitted_ros_artifact"] = ros
    else:
        d["missing"].append(SUBMITTED["ros_schedule"])
    if xpu:
        d["submitted_xpu_artifact"] = xpu
    else:
        d["missing"].append(SUBMITTED["xpu_schedule"])

    # (2) the same numbers from the model, run on the recorded inputs alone
    d["submitted_model"] = model(executor=SUBMITTED["executor"])
    d["submitted_model_queued"] = model(executor="queued")

    # (3) the model's inputs, re-read from the artifact they were taken from
    if ros:
        got = {}
        for n, insts in ros["nets"].items():
            works = {round(v["work_ms"], 9) for v in insts.values()}
            got[n] = {"compute_ms": sorted(works)[0], "uniform": len(works) == 1,
                      "instances": len(insts),
                      "harts": sorted({h for v in insts.values() for h in v["harts"]})}
        d["submitted_inputs_on_disk"] = got

    # (4) the board-recost pair behind the standalone modelled panel
    mb = {}
    for tag, key in (("xpu", "xpu_schedule"), ("ros", "ros_schedule")):
        r = frame_response(MATCHED_BOARD[key], MATCHED_BOARD["frame_instance_zero_based"],
                           MATCHED_BOARD["budget_ms"])
        if r is None:
            d["missing"].append(MATCHED_BOARD[key])
        else:
            mb[f"{tag}_response_ms"] = r
    rf = schedule_facts(MATCHED_BOARD["ros_schedule"])
    if rf:
        mb["ros_makespan_ms"] = rf["makespan_ms"]
        yolo = rf["nets"].get("yolov8_nano_64x96", {})
        if yolo:
            mb["ros_perception_work_ms"] = sorted({round(v["work_ms"], 6) for v in yolo.values()})[0]
            # what the pinning model predicts for the same frame, from that recost compute
            pred = pin_timeline(mb["ros_perception_work_ms"], MATCHED_BOARD["budget_ms"],
                                len(yolo), "queued")
            mb["ros_response_ms_model_queued"] = pred[MATCHED_BOARD["frame_instance_zero_based"]]["response_ms"]
            mb["ros_response_ms_model_release"] = mb["ros_perception_work_ms"]
    d["matched_board"] = mb
    for p in MATCHED_BOARD["sidecars"]:
        fp = os.path.join(REPO, p)
        if os.path.exists(fp):
            d.setdefault("matched_board_sidecars", {})[p] = json.load(open(fp))

    # (5) the ZOH rule that turns a worst-case response into the drawn command rate
    d["control_rate_panel"] = {
        "xpu_rate_hz": zoh_rate_hz(CONTROL_RATE_PANEL["xpu_worst_response_ms"],
                                   CONTROL_RATE_PANEL["control_period_ms"]),
        "ros_rate_hz": zoh_rate_hz(CONTROL_RATE_PANEL["ros_worst_response_ms"],
                                   CONTROL_RATE_PANEL["control_period_ms"]),
    }
    sc = os.path.join(REPO, CONTROL_RATE_PANEL["sidecar"])
    d["control_rate_sidecar"] = json.load(open(sc)) if os.path.exists(sc) else None
    if d["control_rate_sidecar"] is None:
        d["missing"].append(CONTROL_RATE_PANEL["sidecar"])
    d["control_rate_input_provenance_on_disk"] = os.path.isdir(
        os.path.join(REPO, "results", "codesign_feedback", "microros_baseline_k1"))
    return d


def verify() -> int:
    d = derive()
    ok = True

    def chk(label, got, rec, tol):
        nonlocal ok
        if got is None:
            print(f"{label:<56} derived     none   recorded {rec:9.3f}   (artifact missing)")
            return
        flag = "" if abs(got - rec) <= tol else "   <-- DRIFT"
        if flag:
            ok = False
        print(f"{label:<56} derived {got:9.3f}   recorded {rec:9.3f}{flag}")

    def info(msg):
        print("INFO " + msg)

    print("== the submitted figure's modelled baseline "
          "(refined/warehouse_showdown_story.pdf, panel I) ==")
    a = d.get("submitted_ros_artifact") or {}
    m = d["submitted_model"]
    chk("ROS pinning makespan, artifact (ms)", a.get("makespan_ms"),
        SUBMITTED["ros_makespan_ms"], 0.001)
    chk("ROS pinning makespan, model (ms)", m["makespan_ms"],
        SUBMITTED["ros_makespan_ms"], 0.001)
    chk("ROS perception response, artifact (ms)",
        (a.get("nets", {}).get("yolov8_nano_64x96", {}).get(0) or {}).get("work_ms"),
        SUBMITTED["ros_perception_response_ms"], 0.001)
    chk("ROS perception response, model (ms)",
        m["nodes"]["yolov8_nano_64x96"]["response_worst_ms"],
        SUBMITTED["ros_perception_response_ms"], 0.001)
    chk("ROS harts carrying work, artifact", a.get("harts_used_n"), SUBMITTED["ros_harts_used"], 0)
    chk("ROS harts carrying work, model", m["harts_used_n"], SUBMITTED["ros_harts_used"], 0)
    chk("ROS harts idle, model", m["harts_idle_n"], SUBMITTED["ros_harts_idle"], 0)
    x = d.get("submitted_xpu_artifact") or {}
    chk("XPU-RT makespan drawn beside it, artifact (ms)", x.get("makespan_ms"),
        SUBMITTED["xpu_makespan_ms"], 0.001)

    # the model's own inputs must still be what the artifact says they are
    disk = d.get("submitted_inputs_on_disk") or {}
    for n, v in DEPLOYED_NODES.items():
        got = disk.get(n)
        if not got:
            info(f"model input {n}: not on disk"); continue
        chk(f"model input, {n} one-hart compute (ms)", got["compute_ms"], v["compute_ms"], 0.001)
        chk(f"model input, {n} instances", got["instances"], v["instances"], 0)
        if tuple(got["harts"]) != tuple(v["harts"]):
            ok = False
            print(f"model input, {n} partition                    derived {got['harts']}   "
                  f"recorded {list(v['harts'])}   <-- DRIFT")

    q = d["submitted_model_queued"]
    info(f"the stricter reading: a single-threaded executor that QUEUES instances gives "
         f"makespan {q['makespan_ms']:.3f} ms and a worst perception response of "
         f"{q['nodes']['yolov8_nano_64x96']['response_worst_ms']:.3f} ms. The submitted artifact "
         f"places each instance at its release instead, so the drawn baseline is the faster of the two readings.")
    info(f"model chain camera->control {m['chain_camera_to_control_ms']:.3f} ms -> "
         f"{m['command_rate_hz']:.1f} Hz commands under the {m['control_dt_ms']:.0f} ms ZOH; "
         f"the perception node alone sustains "
         f"{m['perception_sustainable_camera_hz']:.1f} Hz against its "
         f"{DEPLOYED_NODES['yolov8_nano_64x96']['period_ms']:.0f} ms release period "
         f"({m['nodes']['yolov8_nano_64x96']['late_instances']}/"
         f"{m['nodes']['yolov8_nano_64x96']['instances']} frames late)")

    print("\n== the standalone modelled schedule panel (warehouse_schedule_board) ==")
    mb = d.get("matched_board") or {}
    chk("XPU-RT frame response, artifact (ms)", mb.get("xpu_response_ms"),
        MATCHED_BOARD["xpu_response_ms"], 0.001)
    chk("ROS frame response, artifact (ms)", mb.get("ros_response_ms"),
        MATCHED_BOARD["ros_response_ms"], 0.001)
    chk("ROS pinning makespan, board-recost artifact (ms)", mb.get("ros_makespan_ms"),
        MATCHED_BOARD["ros_makespan_ms"], 0.001)
    chk("ROS per-frame compute, board-recost artifact (ms)", mb.get("ros_perception_work_ms"),
        MATCHED_BOARD["ros_perception_work_ms"], 0.001)
    for p, s in (d.get("matched_board_sidecars") or {}).items():
        if "ros_response_ms" in s:
            chk(f"sidecar agrees: {os.path.basename(p)}", s["ros_response_ms"],
                MATCHED_BOARD["ros_response_ms"], 0.001)
    if "ros_response_ms_model_queued" in mb:
        info("the board-recost schedule is NOT re-derivable from the pinning recurrence: its "
             f"per-frame compute is {mb['ros_perception_work_ms']:.3f} ms, from which the queued "
             f"model predicts {mb['ros_response_ms_model_queued']:.3f} ms and the release model "
             f"{mb['ros_response_ms_model_release']:.3f} ms for frame "
             f"{MATCHED_BOARD['frame_instance_zero_based']}, against the "
             f"{mb['ros_response_ms']:.3f} ms the artifact carries. The recost re-times the "
             "schedule by per-dispatch multipliers rather than re-running the policy, so it "
             "opens intra-frame gaps the recurrence does not model. §6 of the reproduction page.")

    print("\n== the panel A / panel B command-rate labels ==")
    cr = d["control_rate_panel"]
    chk("XPU-RT command rate from 4.89 ms under ZOH (Hz)", cr["xpu_rate_hz"],
        CONTROL_RATE_PANEL["xpu_control_rate_hz"], 0.01)
    chk("ROS command rate from 12.40 ms under ZOH (Hz)", cr["ros_rate_hz"],
        CONTROL_RATE_PANEL["ros_control_rate_hz"], 0.01)
    s = d.get("control_rate_sidecar")
    if s:
        chk("sidecar agrees: ROS worst response (ms)", s.get("ros_worst_response_ms"),
            CONTROL_RATE_PANEL["ros_worst_response_ms"], 0.001)
        chk("sidecar agrees: ROS command rate (Hz)", s.get("ros_control_rate_hz"),
            CONTROL_RATE_PANEL["ros_control_rate_hz"], 0.01)
    if not d["control_rate_input_provenance_on_disk"]:
        info("the 12.40 ms worst-case control response is an input to the figure, not an output "
             "of this model; its source, results/codesign_feedback/microros_baseline_k1/, is not "
             "present. The ZOH arithmetic above is re-derived; the 12.40 ms itself is an input.")

    if d["missing"]:
        print("\nnot re-derived (artifact missing): " + ", ".join(sorted(set(d["missing"]))))
    print(f"\n{provenance()}")
    return 0 if ok else 1


# --- CLI ---------------------------------------------------------------------------------------
def _parse_node(spec: str):
    """ROLE:NAME:COMPUTE_MS:PERIOD_MS:INSTANCES[:HART+HART...]"""
    parts = spec.split(":")
    if len(parts) < 5:
        raise SystemExit(f"--node wants ROLE:NAME:COMPUTE_MS:PERIOD_MS:INSTANCES[:HARTS], got {spec!r}")
    role, name, c, t, n = parts[:5]
    harts = tuple(parts[5].split("+")) if len(parts) > 5 and parts[5] else ("CPU_P#0",)
    return name, {"role": role, "compute_ms": float(c), "period_ms": float(t),
                  "instances": int(n), "harts": harts}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="the analytical ROS 2 per-node-pinning baseline",
        epilog="with no --node the deployed three-net stack of DEPLOYED_NODES is modelled")
    ap.add_argument("--verify", action="store_true",
                    help="re-derive every recorded number and report drift")
    ap.add_argument("--json", action="store_true", help="print the model (or derive()) as JSON")
    ap.add_argument("--node", action="append", default=[],
                    help="ROLE:NAME:COMPUTE_MS:PERIOD_MS:INSTANCES[:HART+HART...], repeatable")
    ap.add_argument("--executor", choices=["release", "queued"], default="release",
                    help="release places each instance at k*T; queued serialises them on the partition")
    ap.add_argument("--harts", type=int, default=PLATFORM_HARTS, help="harts on the platform")
    ap.add_argument("--camera-hz", type=float, default=None,
                    help="override the perception node's release period")
    ap.add_argument("--perception-width", type=int, default=None,
                    help="widen the perception partition and re-cost it through the MEASURED table")
    ap.add_argument("--perception-speedup", type=float, default=None,
                    help="override that measured speedup")
    ap.add_argument("--control-dt-ms", type=float, default=CONTROL_DT_MS,
                    help="the zero-order-hold step the command rate is quantised to")
    ap.add_argument("--control-on-timer", action="store_true",
                    help="control fires on its own timer instead of in the goal callback")
    a = ap.parse_args()

    if a.verify:
        return verify()

    nodes = dict(_parse_node(s) for s in a.node) if a.node else None
    r = model(nodes, executor=a.executor, control_dt_ms=a.control_dt_ms, platform_harts=a.harts,
              perception_width=a.perception_width, perception_speedup=a.perception_speedup,
              camera_hz=a.camera_hz, control_on_timer=a.control_on_timer)
    if a.json:
        print(json.dumps(r, indent=1, default=str)); return 0
    print(f"executor {r['executor']}   harts {r['harts_used_n']}/{r['platform_harts']} used, "
          f"{r['harts_idle_n']} idle   makespan {r['makespan_ms']:.3f} ms")
    if r["perception_speedup_applied"] != 1.0:
        print(f"perception speedup applied {r['perception_speedup_applied']:.4f}")
    if r["harts_shared_by_two_nodes"]:
        print("partitions are no longer disjoint -- two nodes claim "
              + ", ".join(r["harts_shared_by_two_nodes"])
              + " (the model still runs each node's graph serially on its own partition)")
    print(f"{'node':<22}{'role':<12}{'C (ms)':>9}{'T (ms)':>9}{'worst resp':>12}{'late':>8}  harts")
    for n, v in r["nodes"].items():
        print(f"{n:<22}{v['role']:<12}{v['compute_ms']:>9.3f}{v['period_ms']:>9.2f}"
              f"{v['response_worst_ms']:>12.3f}{v['late_instances']:>4}/{v['instances']:<3}  "
              + "+".join(v["harts"]))
    if r["chain_camera_to_control_ms"] is not None:
        print(f"\ncamera->control {r['chain_camera_to_control_ms']:.3f} ms   "
              f"commands {r['command_rate_hz']:.1f} Hz under the {r['control_dt_ms']:.0f} ms ZOH   "
              f"perception sustains {r['perception_sustainable_camera_hz']:.1f} Hz")
    print(f"\n{provenance()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

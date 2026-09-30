#!/usr/bin/env python3
"""The one place the showdown figures take their board numbers from.

Every replayed arm in the flight campaigns was launched with the cadence trace and the camera->control
latency measured for it on the K1 (`ctrl_trace_from_board.py`, `campaign_percep.sh` ARMS). Those launch
values are the keys the figure scripts select campaign rows by, and the strings they print. This module
holds them once, next to a pointer into `measured_timing` so `check_registry()` can re-derive each one from
the raw traces, and it owns the fallback policy: a figure that would have to invent a number raises
`MissingMeasurement` unless XPURT_FIG_ALLOW_FALLBACK=1, in which case the substitution is recorded in the
sidecar under `fallbacks_used`.

    import figure_constants as FC
    FC.lat_ms("ros_vanilla445.csv")        # 242.0   -> the CSV selector
    FC.lat_label("ros_vanilla445.csv")     # "242 ms" -> the drawn string
    FC.ctrl_hz_label("ros_vanilla445.csv") # "38 Hz"  -> round(1000 / control gap)
    FC.check_registry()                    # [] when every key still matches its trace
"""
from __future__ import annotations
import glob, hashlib, json, os, re, statistics, sys
from typing import NamedTuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
sys.path.insert(0, os.path.join(REPO, "scripts"))
import measured_timing as MT   # noqa: E402


def _spec_window(name="wh_chain45_solve"):
    p = os.path.join(REPO, "data", "toplevel", name + ".json")
    d = json.load(open(p))["networks"]["yolov8_nano_64x96"]
    return float(d.get("window_duration") or d.get("period"))


def _spec_horizon(name):
    p = os.path.join(REPO, "data", "toplevel", name + ".json")
    return float(json.load(open(p)).get("horizon_ms") or 0.0) if os.path.exists(p) else None


SPEC = dict(camera_hz=(30, 45, 60, 90, 120), control_window_ms=10.0, nav_window_ms=22.0, gantt_window_ms=100.0,
            yolo_window_45_ms=_spec_window("wh_chain45_solve"), table_horizon_120_ms=_spec_horizon("wh_chain120_solve_h200"),
            table_horizon_500_ms=_spec_horizon("wh_chain90_solve_500"), horizon_s=18.0, crash_contact_n=1.0)


class MissingMeasurement(RuntimeError):
    """a figure asked for a number no artifact provides."""


FALLBACKS: list[dict] = []


def fallback(name, value, why):
    """Substitute `value` for a measurement the artifact lacks — only when the caller opted in."""
    if os.environ.get("XPURT_FIG_ALLOW_FALLBACK") != "1":
        raise MissingMeasurement(f"{name}: {why} (set XPURT_FIG_ALLOW_FALLBACK=1 to substitute {value})")
    FALLBACKS.append({"name": name, "value": value, "why": why})
    return value


class ReplayArm(NamedTuple):
    trace: str        # ctrl_traces/<trace>: the cadence replayed into the flights
    label: str        # the display label
    csv_lat: float    # percep_latency_ms the campaign was launched with (the CSV selector)
    csv_hold: float   # percep_hold_ms it was launched with
    cam_hz: int       # the camera rate of the board run behind it
    derive: tuple | None    # where measured_timing re-derives csv_lat: ("SOLVER_ARMS", key, field) | ("ROS_VANILLA", (arm, hz), field) | ("ROS_SENSITIVITY", tag, field) | ("CHAIN", field)
    ctrl_gap: tuple | None  # same shape, the control-output gap the label's Hz comes from
    family: str       # xpu | greedy | ros | p3 | ros8 | multi | rich | legacy


def _sa(key, f="chain_ms"):
    return ("SOLVER_ARMS", key, f)


def _rv(arm, hz, f="chain_goal_ms"):
    return ("ROS_VANILLA", (arm, hz), f)


_ARMS = [
    ReplayArm("xpu_a_cpsat_hard.csv", "XPU-RT · CP-SAT", 56.8, 0.0, 45, _sa("acpsat_hardr"), _sa("acpsat_hardr", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_a_cpsat_hard.csv", "XPU-RT · CP-SAT, 120 Hz camera", 59.9, 8.3, 120, _sa("a120hcpsat_hardr"), _sa("a120hcpsat_hardr", "ctrl_gap_mean_ms"), "xpu"),
    # the same chain costed against the board per-width tables: the solver can see what a wider
    # machine combination buys, and shards
    ReplayArm("xpu_shardcpsat45.csv", "XPU-RT · CP-SAT, shard costs measured", 53.2, 0.0, 45,
              _sa("sonlycpr"), _sa("sonlycpr", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_shard45.csv", "XPU-RT · perception in two camera periods", 36.0, 0.0, 45,
              _sa("w2pgOCr"), _sa("w2pgOCr", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_a120h_cpsat.csv", "XPU-RT · CP-SAT, 120 Hz camera", 59.9, 8.3, 120, _sa("a120hcpsat_hardr"), _sa("a120hcpsat_hardr", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_a_greedy.csv", "XPU-RT · greedy", 748.0, 0.0, 45, _sa("agreedyr"), _sa("agreedyr", "ctrl_gap_mean_ms"), "greedy"),
    ReplayArm("xpu_a90_cpsat.csv", "XPU-RT · CP-SAT, 90 Hz camera", 55.9, 11.1, 90, _sa("a90cpsat_hardr"), _sa("a90cpsat_hardr", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_fb30r1.csv", "XPU-RT · CP-SAT, per-rate feedback, 30 Hz camera", 30.1, 33.3, 30, _sa("fb30r1r"), _sa("fb30r1r", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_w2pg36.csv", "XPU-RT · greedy, shard costs measured, 36 Hz camera", 30.1, 27.8, 36, _sa("w2pg36r"), _sa("w2pg36r", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_w2pg40.csv", "XPU-RT · greedy, shard costs measured, 40 Hz camera", 39.8, 25.0, 40, _sa("w2pg40r"), _sa("w2pg40r", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_a30_cpsat.csv", "XPU-RT · CP-SAT, 30 Hz camera", 55.2, 33.3, 30, _sa("a30cpsat_hardr"), _sa("a30cpsat_hardr", "ctrl_gap_mean_ms"), "xpu"),
    # the same 30 Hz chain with YOLO confined to the four P cores (the only cluster where the matrix
    # engine is legal), nav on CPU_E#0 and control on CPU_E#1, conv dispatches on the IME where the
    # measured table prefers it: the partition is what makes the whole of YOLO engine-eligible
    ReplayArm("xpu_p30w4ime.csv", "XPU-RT · CP-SAT, YOLO on the P cores, conv on the IME, 30 Hz camera",
              26.8, 33.3, 30, _sa("p30w4imecpr"), _sa("p30w4imecpr", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_p30free.csv", "XPU-RT · CP-SAT, placement chosen by the solver, conv on the IME, 30 Hz camera",
              26.8, 33.3, 30, _sa("p30freer"), _sa("p30freer", "ctrl_gap_mean_ms"), "xpu"),
    # the same unconstrained solve at a 45 Hz camera; hold 0 so it pairs with the 45 Hz baselines,
    # which were all flown without a goal hold
    ReplayArm("xpu_p45free.csv", "XPU-RT · CP-SAT, placement chosen by the solver, conv on the IME, 45 Hz camera",
              28.3, 0.0, 45, _sa("p45freer"), _sa("p45freer", "ctrl_gap_mean_ms"), "xpu"),
    # the determinism check on that spec: the same solve with XPURT_CPSAT_WORKERS=1, so the table comes
    # back from a recipe rather than from a particular thread interleaving. Flown by no figure; it is
    # here because scripts/board_det45.sh cuts the cadence and the registry must name every trace.
    ReplayArm("xpu_p45det.csv", "XPU-RT · CP-SAT, single-worker solve, placement unconstrained, 45 Hz camera",
              30.07, 0.0, 45, _sa("p45detr"), _sa("p45detr", "ctrl_gap_mean_ms"), "xpu"),
    # the same unconstrained solve at the camera rate where the all-eight-hart baseline still enters
    # the gate course; hold one camera period, matching the baselines it is drawn against
    ReplayArm("xpu_p36free.csv", "XPU-RT · CP-SAT, placement chosen by the solver, conv on the IME, 36 Hz camera",
              25.8, 27.8, 36, _sa("p36freer"), _sa("p36freer", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_p30efull.csv", "XPU-RT · CP-SAT, both clusters whole, conv on the IME, 30 Hz camera",
              25.3, 33.3, 30, _sa("p30efullr"), _sa("p30efullr", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_a60_cpsat.csv", "XPU-RT · CP-SAT, 60 Hz camera", 58.5, 16.7, 60, _sa("a60cpsat_hardr"), _sa("a60cpsat_hardr", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_a90_greedy.csv", "XPU-RT · greedy, 90 Hz camera", 947.5, 0.0, 90, _sa("a90greedyr"), _sa("a90greedyr", "ctrl_gap_mean_ms"), "greedy"),
    ReplayArm("xpu_b5_cpsat.csv", "XPU-RT · CP-SAT, heavier stack", 57.6, 11.1, 90, _sa("b5cpsat_hardr"), _sa("b5cpsat_hardr", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_b5_greedy.csv", "XPU-RT · greedy, heavier stack", 149.0, 11.1, 90, _sa("b5greedyr"), _sa("b5greedyr", "ctrl_gap_mean_ms"), "greedy"),
    ReplayArm("ros_vanilla445.csv", "ROS 2 vanilla (4-hart YOLO)", 242.0, 0.0, 45, _rv("vanilla4", 45), _rv("vanilla4", 45, "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_vanilla4tm45.csv", "ROS 2 vanilla, control timer", 242.0, 0.0, 45, _rv("vanilla4tm", 45), _rv("vanilla4tm", 45, "ctrl_gap_mean_ms"), "ros"),
    # the same deployment asking for every hart: the pool is created on 0-7 (manifest yolo_pool 8)
    # and the four processes are unpinned, so nothing about the machine is withheld from it
    ReplayArm("ros_vanilla8tm45.csv", "ROS 2 timer-driven, all 8 harts", 241.9, 0.0, 45,
              _rv("vanilla8tm", 45), _rv("vanilla8tm", 45, "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_vanilla4tm90.csv", "ROS 2 vanilla, control timer, 90 Hz camera", 136.5, 30.3, 90, _rv("vanilla4tm", 90), _rv("vanilla4tm", 90, "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_vanilla4t45.csv", "ROS 2 vanilla, control timer in the node", 121.0, 0.0, 45, _rv("vanilla4t", 45), _rv("vanilla4t", 45, "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_vanilla45.csv", "ROS 2 vanilla, serial YOLO", 264.8, 0.0, 45, _rv("vanilla", 45), _rv("vanilla", 45, "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_vanilla4_q145.csv", "ROS 2 vanilla, QoS depth 1", 42.0, 0.0, 45, ("ROS_SENSITIVITY", "45_vanilla4_q1_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_vanilla4_q1_r1", "ctrl_gap_mean_ms"), "ros"),
    # the strongest measured ROS 2 arrangement at the headline camera rate: each network its own
    # core set AND control on its own 100 Hz timer, so it is not rate-limited by the camera at all
    ReplayArm("ros_p330.csv", "ROS 2 · per-network pinning, control on its own 100 Hz timer, 30 Hz camera",
              30.4, 33.3, 30, _rv("p3", 30), _rv("p3", 30, "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_p345.csv", "ROS 2 hand-tuned (pinned)", 55.8, 0.0, 45, _rv("p3", 45), _rv("p3", 45, "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_p390.csv", "ROS 2 hand-tuned (pinned), 90 Hz camera", 56.2, 30.3, 90, _rv("p3", 90), _rv("p3", 90, "ctrl_gap_mean_ms"), "p3"),
    # the heavier stack, the two-camera workloads and the 200 Hz control request: the regimes where
    # ROS 2's per-node cost and its executor are what the measurement is about
    ReplayArm("ros_rp345.csv", "ROS 2 hand-pinned + heavier stack", 57.4, 0.0, 45,
              ("ROS_SENSITIVITY", "45_rp3_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_rp3_r1", "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_rspin45.csv", "ROS 2 one process + heavier stack", 198.4, 0.0, 45,
              ("ROS_SENSITIVITY", "45_rspin_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_rspin_r1", "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_rmulti45.csv", "ROS 2 multi-threaded executor + heavier stack", 247.3, 0.0, 45,
              ("ROS_SENSITIVITY", "45_rmulti_r2", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_rmulti_r2", "ctrl_gap_mean_ms"), "multi"),
    ReplayArm("ros_rvanilla45.csv", "ROS 2 serial YOLO + heavier stack", 300.7, 0.0, 45,
              ("ROS_SENSITIVITY", "45_rvanilla_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_rvanilla_r1", "ctrl_gap_mean_ms"), "rich"),
    ReplayArm("ros_x2rp345.csv", "ROS 2 hand-pinned, two cameras + heavier stack", 62.7, 0.0, 45,
              ("ROS_SENSITIVITY", "45_x2rp3_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_x2rp3_r1", "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_x2rmulti45.csv", "ROS 2 multi-threaded, two cameras + heavier stack", 248.1, 0.0, 45,
              ("ROS_SENSITIVITY", "45_x2rmulti_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_x2rmulti_r1", "ctrl_gap_mean_ms"), "multi"),
    ReplayArm("ros_x2rspin45.csv", "ROS 2 one process, two cameras + heavier stack", 623.4, 0.0, 45,
              ("ROS_SENSITIVITY", "45_x2rspin_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_x2rspin_r1", "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_x2p45.csv", "ROS 2 one process per camera", 80.2, 0.0, 45,
              ("ROS_SENSITIVITY", "45_x2p_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_x2p_r1", "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_x2spin45.csv", "ROS 2 one executor, two cameras", 418.9, 0.0, 45,
              ("ROS_SENSITIVITY", "45_x2spin_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_x2spin_r1", "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_p3_c20045.csv", "ROS 2 hand-pinned, 200 Hz control asked", 56.0, 0.0, 45,
              ("ROS_SENSITIVITY", "45_p3_c200_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_p3_c200_r1", "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_spin_c20045.csv", "ROS 2 one process, 200 Hz control asked", 119.7, 0.0, 45,
              ("ROS_SENSITIVITY", "45_spin_c200_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_spin_c200_r1", "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_ship_c20045.csv", "ROS 2 as it ships, 200 Hz control asked", 213.3, 0.0, 45,
              ("ROS_SENSITIVITY", "45_ship_c200_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_ship_c200_r1", "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("xpu_rich45.csv", "XPU-RT + heavier stack", 41.2, 0.0, 45, None,
              ("XPURT_POINTS", "rich45alt2", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_rich45ime.csv", "XPU-RT + heavier stack, ffn on the IME", 40.9, 0.0, 45, None,
              ("XPURT_POINTS", "rich45alt2ime", "ctrl_gap_mean_ms"), "xpu"),
    # the 36 Hz greedy+shard chain with the fused conv kernels the picker places on the IME; measured
    # on the board against the same chain built all-RVV, and cadence-traced, but flown in no campaign
    ReplayArm("xpu_w2pg36ime.csv", "XPU-RT · greedy on measured costs, conv on the IME, 36 Hz camera",
              22.3, 27.8, 36, ("XPURT_POINTS", "w2pg36ime", "chain_ms"),
              ("XPURT_POINTS", "w2pg36ime", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_rich25.csv", "XPU-RT + heavier stack, 25 Hz camera", 30.1, 0.0, 45, None,
              ("XPURT_POINTS", "rich25p4", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_cam2.csv", "XPU-RT, two cameras", 71.8, 0.0, 45, None,
              ("XPURT_POINTS", "cam2alt1", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_cam2rich.csv", "XPU-RT, two cameras + heavier stack", 74.3, 0.0, 45, None,
              ("XPURT_POINTS", "cam2rich", "ctrl_gap_mean_ms"), "xpu"),
    ReplayArm("xpu_c200.csv", "XPU-RT, 200 Hz control", 37.9, 0.0, 45, None,
              ("XPURT_POINTS", "best45alt2c200", "ctrl_gap_mean_ms"), "xpu"),
    # the heavier stack with the whole machine available to ROS 2, as XPU-RT has it
    ReplayArm("ros_rvanilla845.csv", "ROS 2 + heavier stack, pool on all 8 harts", 243.8, 0.0, 45,
              ("ROS_SENSITIVITY", "45_rvanilla8_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_rvanilla8_r1", "ctrl_gap_mean_ms"), "rich"),
    ReplayArm("ros_rvanilla8tm45.csv", "ROS 2 + heavier stack, all 8 harts, timer control", 244.1, 0.0, 45,
              ("ROS_SENSITIVITY", "45_rvanilla8tm_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_rvanilla8tm_r1", "ctrl_gap_mean_ms"), "rich"),
    ReplayArm("ros_rvanilla4x2tm45.csv", "ROS 2 + heavier stack, two instances on all 8, timer", 93.7, 0.0, 45,
              ("ROS_SENSITIVITY", "45_rvanilla4x2tm_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_rvanilla4x2tm_r1", "ctrl_gap_mean_ms"), "ros8"),
    # the two ladder rungs completed with an all-eight-hart variant
    ReplayArm("ros_vanilla8_q145.csv", "ROS 2 QoS depth 1, pool on all 8 harts", 42.5, 0.0, 45,
              ("ROS_SENSITIVITY", "45_vanilla8_q1_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_vanilla8_q1_r1", "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_rp845.csv", "ROS 2 hand-pinned all 8 + heavier stack", 57.0, 0.0, 45,
              ("ROS_SENSITIVITY", "45_rp8_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_rp8_r1", "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_p3_q145.csv", "ROS 2 hand-tuned + QoS 1", 31.1, 0.0, 45, None, _rv("p3", 45, "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_cp345.csv", "ROS 2 hand-tuned, control chained to the goal", 56.2, 0.0, 45,
              ("ROS_SENSITIVITY", "45_cp3_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "45_cp3_r1", "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_vanilla4x245.csv", "ROS 2 on all 8 cores (two YOLO nodes)", 37.0, 0.0, 45, _rv("vanilla4x2", 45), _rv("vanilla4x2", 45, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x2tm45.csv", "ROS 2 on all 8 cores, control on its own timer", 37.8, 0.0, 45,
              _rv("vanilla4x2tm", 45), _rv("vanilla4x2tm", 45, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x290.csv", "ROS 2 on all 8 cores, 90 Hz camera", 252.8, 19.6, 90, _rv("vanilla4x2", 90), _rv("vanilla4x2", 90, "ctrl_gap_mean_ms"), "ros8"),
    # the camera sweep of the two-instance graph; the navigation goal is held one camera period in the flights
    ReplayArm("ros_vanilla_c5045.csv", "ROS 2, serial YOLO, 50 Hz control timer (starves to 20 Hz)", 265.9, 0.0, 45,
              _rv("vanilla_c50", 45), _rv("vanilla_c50", 45, "ctrl_gap_mean_ms"), "ros"),
    ReplayArm("ros_vanilla4x236.csv", "ROS 2 on all 8 cores, 36 Hz camera", 32.2, 27.8, 36, _rv("vanilla4x2", 36), _rv("vanilla4x2", 36, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x240.csv", "ROS 2 on all 8 cores, 40 Hz camera", 37.5, 25.0, 40, _rv("vanilla4x2", 40), _rv("vanilla4x2", 40, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x225.csv", "ROS 2 on all 8 cores, 25 Hz camera", 31.4, 40.0, 25, _rv("vanilla4x2", 25), _rv("vanilla4x2", 25, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x230.csv", "ROS 2 on all 8 cores, 30 Hz camera", 31.4, 33.3, 30, _rv("vanilla4x2", 30), _rv("vanilla4x2", 30, "ctrl_gap_mean_ms"), "ros8"),
    # the same arm with the second YOLO pool's per-shard detail recorded, which is the replicate the
    # measured Gantt rows are drawn from. The flights replay the un-instrumented cadence above; these
    # two are registered because the traces are on disk and every replayable trace names its arm.
    ReplayArm("ros_vanilla4x230d2.csv", "ROS 2 on all 8 cores, per-shard detail recorded, 30 Hz camera",
              31.6, 33.3, 30, _rv("vanilla4x2d2", 30), _rv("vanilla4x2d2", 30, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x236d2.csv", "ROS 2 on all 8 cores, per-shard detail recorded, 36 Hz camera",
              32.8, 27.8, 36, _rv("vanilla4x2d2", 36), _rv("vanilla4x2d2", 36, "ctrl_gap_mean_ms"), "ros8"),
    # all eight harts AND a four-way worker pool under the navigation network, so no stage of the
    # perception->navigation chain runs on one hart: the most capable ROS 2 arrangement measured
    # here. The pool is left unpinned (variant c), which is the fastest of the three placements.
    ReplayArm("ros_vanilla4x236ns4.csv", "ROS 2 on all 8 cores with a nav pool, 36 Hz camera",
              37.4, 27.8, 36, _rv("vanilla4x2ns4c", 36), _rv("vanilla4x2ns4c", 36, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_cp3n430.csv", "ROS 2 · partitioned, nav across the E cores, 30 Hz camera", 30.1, 33.3, 30, _rv("cp3n4", 30), _rv("cp3n4", 30, "ctrl_gap_mean_ms"), "ros8"),
    # the static partition the submitted figure drew: a four-hart YOLO pool, nav and control each
    # pinned to one more hart, and two harts left untouched
    ReplayArm("ros_cp315.csv", "ROS 2 · static 6-core partition, two cores idle, 15 Hz camera", 30.8, 66.7, 15,
              ("ROS_SENSITIVITY", "15_cp3_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "15_cp3_r1", "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_cp325.csv", "ROS 2 · static 6-core partition, two cores idle, 25 Hz camera", 30.7, 40.0, 25,
              ("ROS_SENSITIVITY", "25_cp3_r1", "chain_goal_ms"), ("ROS_SENSITIVITY", "25_cp3_r1", "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_cp330.csv", "ROS 2 · static 6-core partition, two cores idle, 30 Hz camera", 30.7, 33.3, 30, _rv("cp3", 30), _rv("cp3", 30, "ctrl_gap_mean_ms"), "p3"),
    ReplayArm("ros_vanilla4x260.csv", "ROS 2 on all 8 cores, 60 Hz camera", 68.4, 16.7, 60, _rv("vanilla4x2", 60), _rv("vanilla4x2", 60, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x275.csv", "ROS 2 on all 8 cores, 75 Hz camera", 291.9, 13.3, 75, _rv("vanilla4x2", 75), _rv("vanilla4x2", 75, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x2120.csv", "ROS 2 on all 8 cores, 120 Hz camera", 200.0, 8.3, 120, _rv("vanilla4x2", 120), _rv("vanilla4x2", 120, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x2tm25.csv", "ROS 2 on all 8 cores, control on its own timer, 25 Hz camera", 31.6, 40.0, 25,
              _rv("vanilla4x2tm", 25), _rv("vanilla4x2tm", 25, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x2tm30.csv", "ROS 2 on all 8 cores, control on its own timer, 30 Hz camera", 31.6, 33.3, 30,
              _rv("vanilla4x2tm", 30), _rv("vanilla4x2tm", 30, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x2tm75.csv", "ROS 2 on all 8 cores, control on its own timer, 75 Hz camera", 291.5, 13.3, 75,
              _rv("vanilla4x2tm", 75), _rv("vanilla4x2tm", 75, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_vanilla4x2tm120.csv", "ROS 2 on all 8 cores, control on its own timer, 120 Hz camera", 200.4, 8.3, 120,
              _rv("vanilla4x2tm", 120), _rv("vanilla4x2tm", 120, "ctrl_gap_mean_ms"), "ros8"),
    ReplayArm("ros_multi45.csv", "ROS 2 multi-threaded executor", 264.7, 50.1, 45, _rv("multi", 45), _rv("multi", 45, "ctrl_gap_mean_ms"), "multi"),
    ReplayArm("ros_rvanilla445.csv", "ROS 2 vanilla + heavier stack", 243.0, 26.7, 45, _rv("rvanilla4", 45), _rv("rvanilla4", 45, "ctrl_gap_mean_ms"), "rich"),
    ReplayArm("ros_rvanilla490.csv", "ROS 2 vanilla + heavier stack, 90 Hz camera", 137.0, 26.1, 90, _rv("rvanilla4", 90), _rv("rvanilla4", 90, "ctrl_gap_mean_ms"), "rich"),
    # earlier arms, kept so every trace file in ctrl_traces/ resolves
    ReplayArm("xpurt_best45alt2long.csv", "XPU-RT · sharded (best45alt2)", 40.1, 0.0, 45, ("CHAIN", "xpurt_ms"), None, "legacy"),
    ReplayArm("xpurt_greedy_long25.csv", "XPU-RT · greedy, 25 Hz camera", 0.0, 0.0, 25, None, None, "legacy"),
    ReplayArm("ros_spin45_r1.csv", "ROS 2 pool arm (spin)", 119.0, 0.0, 45, ("CHAIN", "ros_pool_cam45_ms"), None, "legacy"),
    ReplayArm("ros_vanilla45_smoke.csv", "ROS 2 vanilla, smoke run", 0.0, 0.0, 45, None, None, "legacy"),
]
REPLAY_ARMS: dict[tuple[str, float], ReplayArm] = {(a.trace, a.csv_lat): a for a in _ARMS}
ARM_BY_TRACE: dict[str, ReplayArm] = {}
for _a in _ARMS:   # the primary entry per trace: the 45 Hz one, else the lowest latency
    _cur = ARM_BY_TRACE.get(_a.trace)
    if _cur is None or (_a.cam_hz == 45 and _cur.cam_hz != 45) or (_cur.cam_hz != 45 and _a.csv_lat < _cur.csv_lat):
        ARM_BY_TRACE[_a.trace] = _a
# the goal hold used when the nav goal is refreshed at the deployment's own rate (one camera period at 45 Hz for
# the schedule; the baseline's measured goal cadence for the pinned ROS 2 arm) — the "goal at deployment rate" cells
GOAL_HOLD_DEPLOYMENT_MS = {"xpu_a_cpsat_hard.csv": 22.2, "ros_p345.csv": 30.3, "ros_vanilla4tm45.csv": 30.3}
FAMILY_LABEL = {"xpu": "XPU-RT · CP-SAT", "greedy": "XPU-RT · greedy", "ros": f"ROS 2 vanilla ({{lat}})", "p3": f"ROS 2 hand-tuned ({{lat}})",
                "ros8": "ROS 2 on all 8 cores", "multi": "ROS 2 multi-threaded executor", "rich": "ROS 2 + heavier stack"}
_FAMILY_TRACE = {"ros": "ros_vanilla445.csv", "p3": "ros_p345.csv", "xpu": "xpu_a_cpsat_hard.csv", "greedy": "xpu_a_greedy.csv", "ros8": "ros_vanilla4x245.csv", "multi": "ros_multi45.csv", "rich": "ros_rvanilla445.csv"}


def arm_for(trace, lat=None, hold=None) -> ReplayArm:
    t = os.path.basename(trace)
    if lat is not None and (t, float(lat)) in REPLAY_ARMS:
        return REPLAY_ARMS[(t, float(lat))]
    if t not in ARM_BY_TRACE:
        raise KeyError(f"{t} is not a registered replay arm (figure_constants.REPLAY_ARMS)")
    return ARM_BY_TRACE[t]


def lat_ms(trace, hold=None) -> float:
    return arm_for(trace).csv_lat


def hold_ms(trace) -> float:
    return arm_for(trace).csv_hold


def lat_label(trace, lat=None) -> str:
    return f"{round(arm_for(trace, lat).csv_lat):.0f} ms"


def _resolve(ptr):
    """the recorded constant a pointer names (None when the table has no such entry)."""
    if ptr is None:
        return None
    kind = ptr[0]
    if kind == "SOLVER_ARMS":
        return MT.SOLVER_ARMS.get(ptr[1], {}).get(ptr[2])
    if kind == "ROS_VANILLA":
        return MT.ROS_VANILLA.get(tuple(ptr[1]), {}).get(ptr[2])
    if kind == "ROS_SENSITIVITY":
        return MT.ROS_SENSITIVITY.get(ptr[1], {}).get(ptr[2])
    if kind == "XPURT_POINTS":
        # the board points beyond the design rate and with the heavier stack, one trace label each
        return MT.XPURT_POINTS.get(ptr[1], {}).get(ptr[2])
    if kind == "CHAIN":
        return MT.CHAIN.get(ptr[1])
    raise KeyError(kind)


def ctrl_gap_ms(trace) -> float:
    a = arm_for(trace); g = _resolve(a.ctrl_gap)
    if g is None:
        raise MissingMeasurement(f"{a.trace}: no recorded control gap")
    return float(g)


def ctrl_hz_label(trace) -> str:
    return f"{round(1000.0 / ctrl_gap_ms(trace)):.0f} Hz"


def arm_label(trace) -> str:
    return arm_for(trace).label


def forest_family_label(fam) -> str:
    tmpl = FAMILY_LABEL[fam]
    return tmpl.format(lat=lat_label(_FAMILY_TRACE[fam])) if "{lat}" in tmpl else tmpl


def goals_per_s(arm, hz=45) -> float:
    v = MT.ROS_VANILLA.get((arm, hz), {}).get("goals_per_s")
    if v is None:
        raise MissingMeasurement(f"ROS_VANILLA has no goals/s for {arm}@{hz}")
    return float(v)


# ------------------------------------------------------------------------------------------ re-derivation
def _xpu_chain_from_traces(label_prefix):
    from make_measured_gantt_pair import per_frame_chain, read_trace   # noqa: E402
    chain = []
    for t in sorted(glob.glob(os.path.join(RES, "xpurt_long", f"trace_{label_prefix}[0-9]_other_run1.csv"))):
        chain += [c for k, c, _ in per_frame_chain(read_trace(t)) if k >= 1]
    return statistics.median(chain) if chain else None


def _xpu_chain_from_point(label):
    """The chain median for an XPURT_POINTS label, whose traces carry the policy in the name.

    SOLVER_ARMS labels take a digit suffix per replicate (`trace_<pre><k>_other_run1.csv`); a point's
    label is the whole name and its replicates are the run number, so it needs its own glob.
    """
    from make_measured_gantt_pair import per_frame_chain, read_trace   # noqa: E402
    rec = MT.XPURT_POINTS.get(label) or {}
    chain = []
    for t in sorted(glob.glob(os.path.join(RES, "xpurt_long",
                                           f"trace_{label}_{rec.get('policy', 'other')}_run*.csv"))):
        chain += [c for k, c, _ in per_frame_chain(read_trace(t)) if k >= 1]
    return statistics.median(chain) if chain else None


def check_registry(tol_ms=1.5) -> list[str]:
    """Every replayed latency against the board: the launch value must still be what the traces say."""
    problems = []; derived = MT.derive(); ra = derived.get("ros_arms", {})
    summ = {}
    p = os.path.join(RES, "ros_traced", "summary.csv")
    if os.path.exists(p):
        import csv
        summ = {r["tag"]: r for r in csv.DictReader(open(p))}
    for a in _ARMS:
        if a.derive is None:
            continue
        rec = _resolve(a.derive)
        if rec is None:
            problems.append(f"{a.trace}@{a.csv_lat}: measured_timing has no entry for {a.derive}"); continue
        tol = tol_ms
        if abs(float(rec) - a.csv_lat) > tol:
            problems.append(f"{a.trace}@{a.csv_lat}: recorded constant {rec} differs from the launch value")
        kind = a.derive[0]
        if kind == "SOLVER_ARMS":
            got = _xpu_chain_from_traces(a.derive[1])
        elif kind == "XPURT_POINTS":
            got = _xpu_chain_from_point(a.derive[1]) if a.derive[2] == "chain_ms" else None
        elif kind == "ROS_VANILLA":
            got = (ra.get(f"{a.derive[1][0]}@{a.derive[1][1]}") or {}).get("e2e_goal_med_pooled")
        elif kind == "ROS_SENSITIVITY":
            r = summ.get(a.derive[1]); got = float(r["e2e_goal_med_ms"]) if r and r.get("e2e_goal_med_ms") else None
        elif kind == "CHAIN" and a.derive[1] == "xpurt_ms":
            got = (derived.get("xpurt_chain_long", {}).get(MT.XPURT_ARM) or {}).get("chain_median_ms")
        elif kind == "CHAIN" and a.derive[1] == "ros_pool_cam45_ms":
            got = (ra.get("spin@45") or {}).get("e2e_goal_med_pooled")
        else:
            got = None
        if got is None:
            problems.append(f"{a.trace}@{a.csv_lat}: no trace on disk to re-derive {a.derive}")
        elif abs(float(got) - a.csv_lat) > tol:
            problems.append(f"{a.trace}@{a.csv_lat}: traces give {got:.1f} ms")
    return problems


def allowed_display_literals() -> set[str]:
    """the ms / Hz strings a figure may print: every registry label plus the spec's windows and camera rates."""
    out = set()
    for a in _ARMS:
        if a.csv_lat > 0:
            out.add(f"{round(a.csv_lat):.0f} ms")
        if a.ctrl_gap is not None and _resolve(a.ctrl_gap):
            out.add(f"{round(1000.0 / float(_resolve(a.ctrl_gap))):.0f} Hz")
    for k, v in MT.SOLVER_ARMS.items():
        out.add(f"{round(v['chain_ms']):.0f} ms")
    for (arm, hz), v in MT.ROS_VANILLA.items():
        out.add(f"{round(v['chain_goal_ms']):.0f} ms"); out.add(f"{hz} Hz")
        # goals/s is quoted by one figure and is not re-derived by measured_timing --verify, so an
        # arm recorded without it contributes the two values that are re-derived and nothing else,
        # rather than a literal no artifact backs
        if v.get("goals_per_s") is not None:
            out.add(f"{round(v['goals_per_s']):.0f} goals/s"); out.add(f"{round(v['goals_per_s']):.0f}/s")
    out |= {f"{SPEC['control_window_ms']:.0f} ms", f"{SPEC['nav_window_ms']:.0f} ms", f"{SPEC['gantt_window_ms']:.0f} ms", "15 Hz", "25 Hz", "33 Hz", "50 Hz", "200 Hz"}
    out |= {f"{SPEC[k]:.0f} ms" for k in ("table_horizon_120_ms", "table_horizon_500_ms") if SPEC.get(k)}   # the solved tables' horizons
    out |= {f"{h} Hz" for h in SPEC["camera_hz"]}
    return out


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_common(script_name, input_paths) -> dict:
    """what every sidecar carries: the script, the sha256 of each input it read, the substitutions it made."""
    inputs = {}
    for p in sorted({os.path.abspath(x) for x in input_paths if x and os.path.isfile(x)}):
        inputs[os.path.relpath(p, REPO)] = sha256_of(p)
    return {"script": script_name, "inputs": inputs, "fallbacks_used": list(FALLBACKS)}


if __name__ == "__main__":
    probs = check_registry()
    for p in probs:
        print("DRIFT", p)
    print(f"{len(_ARMS)} arms, {len(probs)} problem(s)")
    sys.exit(1 if probs else 0)

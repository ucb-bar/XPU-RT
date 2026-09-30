#!/usr/bin/env python3
"""Single source of truth for the timing numbers the HIL figures and captions quote.

Every value is measured on the SpaceMiT K1 with the ModelBlaster-generated kernels, and every
value is RE-DERIVED from the raw artifact it came from by `derive()`. The recorded dictionaries
below exist so a caption can import a number; `--verify` recomputes each one from disk and
fails if they disagree.

    from measured_timing import XPURT, ROS, CHAIN, ROS_ARMS, YOLO_STANDALONE_MS, provenance

Artifacts (all under results/codesign_feedback/):
  xpurt_long/trace_<label>_<policy>_run<k>.csv   XPU-RT board traces, one second of the chain
  xpurt_long/cpu_*.csv, hart_acc_*.csv           per-core busy %, per-hart kernel accounting
  cmp_coupled_cpsat_board_trace.csv              the single-frame coupled schedule, executed
  ros_traced/<hz>_<arm>_r<k>/{trace,ctrl_gaps,released,goals,consumed,cpu}.csv + manifest.json
  ros_traced/yolo_standalone/*.txt               the YOLO kernel alone, 1 and 4 harts, vs golden

Re-derive everything with:  scripts/measured_timing.py --verify

The one baseline that is NOT measured -- the modelled ROS 2 per-node-pinning arm the warehouse
showdown was submitted with -- lives in `scripts/ros_pinning_model.py`, which states its
assumptions and re-derives its numbers the same way; nothing modelled belongs in this module.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results/codesign_feedback")
HZ = 24e6                      # rdtime on the K1, NOT the core clock
WARMUP_MS = 3000.0             # discarded from every ROS run, on the host
CONTROL_DT_MS = 10.0           # the simulator's control step; cadence = ceil(gap / 10 ms)
XPURT_ARM = "best45alt2"       # the XPU-RT arm: at the 45 Hz design camera rate, YOLO sharded 2-way with frames
                               # alternating between the two clusters, nav on P#3, control alone on E#3
XPURT_ARM_25 = "best25p4"      # the 25 Hz variant: YOLO 4-way on the P cluster, nav E#0, control E#1

# --- control-loop cadence -------------------------------------------------------------------
# Gap between successive control outputs. XPU-RT's is bounded by the schedule, which places
# mlp_control in a guaranteed 10 ms window every period. ROS 2's depends on the executor: the
# default single-threaded one serves the control timer only between perception callbacks, so
# the faster the camera, the rarer the control output.
XPURT = {
    "ctrl_gap_mean_ms": 10.00,     # pooled over xpurt_long/trace_best45alt2_other_run{1,2,3}, 98 gaps each
    "ctrl_gap_max_ms": 10.10,
    "ctrl_gap_p95_ms": 10.00,
    "ctrl_rate_hz": 100.0,         # ceil(10.00 / 10 ms) -> 1 tick
    "legacy_ctrl_gap_mean_ms": 6.30,   # the w4/b4/b5y short traces (11 gaps each), a different workload
    "source": "xpurt_long/trace_best45alt2_other_run*.csv: 45 Hz camera, control on its own hart",
}
ROS = {
    "ctrl_gap_mean_ms": 24.05,     # default SingleThreadedExecutor, unpinned, camera 15 Hz; pooled over 3 replicates
    "ctrl_gap_max_ms": 57.14,
    "ctrl_rate_hz": 33.3,          # ceil(22.21 / 10 ms) -> 3 ticks
    "ctrl_gap_mean_cam20_ms": 50.91,   # camera 20 Hz: one control output per perception callback
    "ctrl_gap_mean_cam25_ms": 53.23,   # pooled over 3 replicates
    "ctrl_gap_mean_cam45_ms": 53.25,
    "ctrl_gap_mean_pool_cam45_ms": 29.95,  # YOLO on a 4-hart pool, one executor thread, 45 Hz camera: the figure's ROS arm
    "ctrl_gap_max_pool_cam45_ms": 33.84,
    "ctrl_gap_mean_chained_pool_cam45_ms": 30.50,
    "ctrl_gap_mean_mt_ms": 10.00,  # MultiThreadedExecutor + spin(): holds the timer at every camera rate
    "ctrl_gap_max_mt_ms": 13.87,
    "source": "real ROS 2 Jazzy C++ nodes running the ModelBlaster kernels on the K1, ros_traced/",
}

# --- camera -> control chain ----------------------------------------------------------------
CHAIN = {
    "xpurt_ms": 40.1,              # camera release -> control output at 45 Hz, median over 126 warm frames (3 runs), best45alt2
    "xpurt_yolo_span_ms": 31.0,    # per-frame YOLO span at width 2, two frames in flight
    "xpurt_25hz_ms": 29.8,         # the 25 Hz p4 layout: camera -> control, control 10.00 +- 0.06
    "xpurt_25hz_yolo_span_ms": 24.2,
    "xpurt_single_frame_ms": 35.71,    # the cold single-frame coupled run: 31.21 yolo + warm nav/ctrl
    "xpurt_cold_ms": 43.24,
    "ros_ms": 53.9,                # camera -> goal, single-threaded executor, unpinned, camera <= 15 Hz; 3 replicates
    "ros_pool_cam45_ms": 119.0,    # camera -> goal, 4-hart pool, one executor thread, 45 Hz camera (backlogged)
    "ros_max_sustained_hz": 15.0,  # at 20 Hz the single-threaded chain queues (goal latency 211 ms)
    "ros_mt_max_sustained_hz": 20.0,   # multi-threaded: keeps up to 20 Hz, drops frames from 25 Hz
    "xpurt_max_sustained_hz": 28.0,
    "deadline_ms": 33.3,
    "source": "cmp_coupled_cpsat_board_trace.csv; ros_traced/*_ship_r*/, *_multi_r*/",
}

# --- yolov8_nano_64x96 alone, same IR and curated kernels, bit-exact against golden ------------
YOLO_STANDALONE_MS = {"1hart": 47.85, "4hart_pool": 24.58, "cold_1hart": 55.61, "cold_4hart": 41.69,
                      "4hart_E_pool": 25.57, "8hart_pool": 25.03}

# --- beyond the design rate, and the heavier stack (one XPU-RT trace label each) -----------------
# camera rate the layout was asked for, what it delivered: every value from xpurt_long/trace_<label>_other_run*.csv
XPURT_POINTS = {
    "best60alt2": {"camera_hz": 60, "on_time": True,  "chain_ms": 39.6, "ctrl_gap_mean_ms": 10.00},   # two frames in flight, width 2
    "best75alt2": {"camera_hz": 75, "on_time": False, "chain_ms": 119.6, "ctrl_gap_mean_ms": 10.00},  # the width-2 layout's ceiling is ~64 Hz
    "best90alt1": {"camera_hz": 90, "on_time": True,  "chain_ms": 57.4, "ctrl_gap_mean_ms": 10.00},   # six frames in flight, width 1
    "rich45alt2": {"camera_hz": 45, "on_time": True,  "chain_ms": 40.8, "ctrl_gap_mean_ms": 9.98,     # + ffn_block 10 Hz + dronet 30 Hz
                   "ctrl_gap_max_ms": 12.12},
    "rich25p4":   {"camera_hz": 25, "on_time": True,  "chain_ms": 29.6, "ctrl_gap_mean_ms": 10.00},
    # the same 36 Hz greedy+shard chain built twice from one source tree, run back to back on the
    # board: every conv dispatch on the RVV kernels, then the fused conv kernels the picker places on
    # the IME (smt.vmadot) where its measured table says the matrix engine is the faster of the two.
    # Both arms verify bit-identically (the 1.8e-4 line is the nav head's fp16 output, in both).
    "w2pg36base": {"camera_hz": 36, "on_time": True, "chain_ms": 30.1, "ctrl_gap_mean_ms": 9.98,
                   "ctrl_gap_max_ms": 19.58},
    "w2pg36ime":  {"camera_hz": 36, "on_time": True, "chain_ms": 22.3, "ctrl_gap_mean_ms": 9.94,
                   "ctrl_gap_max_ms": 21.45},
    # sensitivity rows on the headline layout: control asked for 200 Hz; two unpinned busy-loop
    # processes started before the run (HOGS=2); and the same table run for five seconds
    "best45alt2c200": {"camera_hz": 45, "on_time": True, "chain_ms": 37.9, "ctrl_gap_mean_ms": 5.00, "ctrl_hz": 200},
    "best45alt2":     {"camera_hz": 45, "on_time": True, "chain_ms": 45.6, "ctrl_gap_mean_ms": 9.97,
                       "ctrl_gap_max_ms": 15.14, "policy": "other_hog2", "late_frames": "8/88"},
    "best45alt2long": {"camera_hz": 45, "on_time": True, "chain_ms": 40.1, "ctrl_gap_mean_ms": 10.00,
                       "ctrl_gap_max_ms": 10.03, "seconds": 5, "n_gaps": 996},
    # two 45 Hz cameras (90 frames offered per second), the alt1 layout carrying both streams
    "cam2alt1":     {"camera_hz": 90, "on_time": True, "chain_ms": 71.8, "ctrl_gap_mean_ms": 10.00,
                     "ctrl_gap_max_ms": 10.03, "cameras": 2},
    # the heavier stack with ffn_block's two linears on the matrix engine (backends rvv_x60,ime_x60,rvv_x60)
    "rich45alt2ime": {"camera_hz": 45, "on_time": True, "chain_ms": 40.9, "ctrl_gap_mean_ms": 9.97,
                      "ctrl_gap_max_ms": 12.15, "ffn_fc1_ms": 7.2, "ffn_fc2_ms": 5.9},   # vs 13.7 / 10.5 on RVV
    # combined load: two 45 Hz cameras AND the heavier stack (ffn_block on the IME, dronet), alt1
    "cam2rich":     {"camera_hz": 90, "on_time": True, "chain_ms": 74.3, "ctrl_gap_mean_ms": 9.96,
                     "ctrl_gap_max_ms": 12.23, "cameras": 2},
}
ROS_SENSITIVITY = {   # ros_traced/<tag>: the pool arm and its siblings under the same three perturbations
    "45_spin_q1_r1":   {"ctrl_gap_mean_ms": 21.4, "chain_goal_ms": 60.1},    # QoS depth 1: the queue drops frames, so fewer YOLO callbacks share the thread
    "45_ship_q1_r1":   {"ctrl_gap_mean_ms": 35.8, "chain_goal_ms": 107.2},
    "45_spin_c200_r1": {"ctrl_gap_mean_ms": 30.1, "chain_goal_ms": 119.7},   # asking the timer for 200 Hz changes nothing: the thread is the limit
    "45_p3_c200_r1":   {"ctrl_gap_mean_ms": 5.00, "chain_goal_ms": 56.0},
    "45_spin_hog2_r1": {"ctrl_gap_mean_ms": 30.1, "chain_goal_ms": 119.5},   # two background hogs land on idle E cores; the pinned arm does not see them
    "45_p3_hog2_r1":   {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 55.9},
    # the fully pinned arm with control chained to the goal instead of on its own timer: it
    # reaches the same camera-to-goal latency and then commands at the goal rate, not 100 Hz
    # the pinned six-core arm across camera rates: its control rate follows the camera until the
    # pipeline saturates, after which more frames buy no more commands (45/60/75/90 all land at ~39 Hz)
    "15_cp3_r1":       {"ctrl_gap_mean_ms": 66.67, "chain_goal_ms": 30.8},
    "25_cp3_r1":       {"ctrl_gap_mean_ms": 40.00, "chain_goal_ms": 30.7},
    "30_cp3_r1":       {"ctrl_gap_mean_ms": 33.40, "chain_goal_ms": 30.7},
    "45_cp3_r1":       {"ctrl_gap_mean_ms": 25.99, "chain_goal_ms": 56.2},
    "60_cp3_r1":       {"ctrl_gap_mean_ms": 25.63, "chain_goal_ms": 56.2},
    "75_cp3_r1":       {"ctrl_gap_mean_ms": 25.60, "chain_goal_ms": 56.1},
    "90_cp3_r1":       {"ctrl_gap_mean_ms": 25.57, "chain_goal_ms": 56.0},
    # the heavier stack (+ ffn_block 10 Hz, dronet 30 Hz): where ROS 2's per-node cost shows up. Adding
    # two networks costs the single-executor arm +79 ms and 44% of its frames; hand-pinning absorbs them
    "45_rp3_r1":       {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 57.4},
    "45_rspin_r1":     {"ctrl_gap_mean_ms": 53.11, "chain_goal_ms": 198.4},
    "45_rmulti_r2":    {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 247.3},   # r1 of the same config produced no frames at all
    "45_rvanilla_r1":  {"ctrl_gap_mean_ms": 84.01, "chain_goal_ms": 300.7},
    "45_x2rp3_r1":     {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 62.7},
    "45_x2rmulti_r1":  {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 248.1},
    "45_x2rspin_r1":   {"ctrl_gap_mean_ms": 92.57, "chain_goal_ms": 623.4},
    "45_x2p_r1":       {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 80.2},
    "45_x2spin_r1":    {"ctrl_gap_mean_ms": 60.01, "chain_goal_ms": 418.9},
    # the two rungs that had no all-eight-hart variant: QoS depth 1 with the pool on 0-7, and the
    # hand-pinned loaded stack with nav and control on their own hart pairs
    "45_vanilla8_q1_r1": {"ctrl_gap_mean_ms": 25.95, "chain_goal_ms": 42.5},
    "45_rp8_r1": {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 57.0},
    # the heavier stack with the whole machine available: the pool asked for harts 0-7 buys nothing
    # (the kernel stops scaling past four), and two instances cost more here than hand-pinning does
    "45_rvanilla8_r1": {"ctrl_gap_mean_ms": 26.74, "chain_goal_ms": 243.8},
    "45_rvanilla8tm_r1": {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 244.1},
    "45_rvanilla4x2tm_r1": {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 93.7},
    # asked for 200 Hz control: the multi-process arms deliver it, the single-executor ones cannot
    "45_p3_c200_r1":   {"ctrl_gap_mean_ms": 5.00, "chain_goal_ms": 56.0},     # 200 Hz as asked
    "45_spin_c200_r1": {"ctrl_gap_mean_ms": 30.11, "chain_goal_ms": 119.7},   # 33 Hz: the timer cannot preempt the YOLO callback
    "45_ship_c200_r1": {"ctrl_gap_mean_ms": 53.44, "chain_goal_ms": 213.3},   # 19 Hz
    # two 45 Hz cameras: one executor (x2spin) and one process per camera plus nav and control (x2p)
    "45_x2spin_r1":    {"ctrl_gap_mean_ms": 60.0, "chain_goal_ms": 418.9},
    "45_x2spin_r2":    {"ctrl_gap_mean_ms": 59.7, "chain_goal_ms": 416.5},
    "45_x2p_r1":       {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 80.2},
    "45_x2p_r2":       {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 45.2},
    # two cameras AND the heavier stack, one process per camera plus nav, control and the extras
    "45_x2rp3_r1":     {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 62.7},
    "45_x2rp3_r2":     {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 63.9},
    # the multi-threaded executor under the same combined load (both cameras' YOLO in one callback group)
    "45_x2rmulti_r1":  {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 248.1},
    "45_x2rmulti_r2":  {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 247.7},
    # the vanilla 4-hart graph with keep-last-1 QoS: the stale-frame queue is gone, the cadence is not changed
    "45_vanilla4_q1_r1": {"ctrl_gap_mean_ms": 25.6, "chain_goal_ms": 42.0},
    "90_vanilla4_q1_r1": {"ctrl_gap_mean_ms": 25.7, "chain_goal_ms": 36.9},
}
SOLVER_ARMS = {   # the two solvers on the same periodic spec (data/toplevel/wh_chain45_solve.json), executed on the K1
    # three runs each, pooled; frames late = YOLO instances ending after release + 66.7 ms window, after 100 ms
    "acpsat_hardr": {"solver": "CP-SAT (hard windows)", "chain_ms": 56.8, "chain_p95_ms": 72.3, "ctrl_gap_mean_ms": 10.00,
                     "ctrl_gap_max_ms": 19.36, "ctrl_gaps_over_15ms": "12/294", "frames_late": "0/120",
                     "predicted_window_misses_200ms": 0},
    "agreedyr":     {"solver": "greedy (list scheduling)", "chain_ms": 748.0, "chain_p95_ms": 1064.7, "ctrl_gap_mean_ms": 13.25,
                     "ctrl_gap_max_ms": 1036.96, "ctrl_gaps_over_15ms": "3/294", "frames_late": "120/120",
                     "predicted_window_misses_200ms": 336},
    # the same spec costed against the board's per-width tables (gen/mb_shard), where a wider
    # machine combination is no longer priced like one hart: data/toplevel/wh_chain45_shardonly.json
    # keeps the three-period window, wh_chain45_w2p.json holds perception to two periods
    "sonlycpr":     {"solver": "CP-SAT (hard windows), shard costs measured", "chain_ms": 53.2, "chain_p95_ms": 62.6,
                     "ctrl_gap_mean_ms": 10.03, "ctrl_gap_max_ms": 20.52, "frames_late": "0/120",
                     "predicted_window_misses_200ms": 0},
    "w2pgOCr":      {"solver": "greedy, shard costs measured, perception in two camera periods", "chain_ms": 36.0,
                     "chain_p95_ms": 44.2, "ctrl_gap_mean_ms": 9.99, "ctrl_gap_max_ms": 17.02, "frames_late": "0/120",
                     "predicted_window_misses_200ms": 0},
    # spec B: one camera at 90 Hz + ffn_block 10 Hz + dronet 30 Hz (data/toplevel/wh_chain90_rich_solve_500.json, half-second table)
    "b5cpsat_hardr": {"solver": "CP-SAT (hard windows), 90 Hz + heavier stack", "chain_ms": 58.4, "chain_p95_ms": 68.8,
                      "ctrl_gap_mean_ms": 9.95, "ctrl_gap_max_ms": 16.35, "frames_late": "0/108", "predicted_window_misses_100ms": 0},
    "b5greedyr":     {"solver": "greedy, 90 Hz + heavier stack", "chain_ms": 150.3, "chain_p95_ms": 172.1,
                      "ctrl_gap_mean_ms": 11.56, "ctrl_gap_max_ms": 47.88, "frames_late": "108/108", "predicted_window_misses_100ms": 18},
    # the chain spec at other camera rates, half-second tables (wh_chain{30,60,90}_solve_500.json)
    "a30cpsat_hardr": {"solver": "CP-SAT (hard windows), 30 Hz camera", "chain_ms": 55.2, "ctrl_gap_mean_ms": 9.98, "ctrl_gap_max_ms": 15.44, "frames_late": "0/36"},
    "a30greedyr": {"solver": "greedy, 30 Hz camera", "chain_ms": 70.1, "ctrl_gap_mean_ms": 10.0, "ctrl_gap_max_ms": 15.52, "frames_late": "0/36"},
    # THE SPATIALLY PARTITIONED 30 Hz ARMS (docs/K1/partitioned_schedule.md). Same spec, same windows
    # and the same half-second table as a30; the networks are pinned to disjoint harts in the spec
    # (`allowed_machines`), yolo to the four P cores because smt.vmadot is legal only on cluster 0.
    # `machine_width` says how many of those harts one yolo dispatch takes: 1 (a frame per hart) or
    # 4 (the whole cluster per frame). The IME arms turn `enable_impls` on, which the P-core pin is
    # what makes reachable for the whole of yolo.
    "p30greedyr":      {"solver": "greedy, partitioned, one hart per yolo frame", "chain_ms": 60.09,
                        "ctrl_gap_mean_ms": 10.00, "ctrl_gap_max_ms": 10.03, "frames_late": "0/36"},
    "p30cpsat_hardr":  {"solver": "CP-SAT (hard windows), partitioned, one hart per yolo frame", "chain_ms": 60.08,
                        "ctrl_gap_mean_ms": 10.00, "ctrl_gap_max_ms": 10.08, "frames_late": "0/36"},
    "p30imer":         {"solver": "greedy, partitioned, one hart per yolo frame, conv on the IME", "chain_ms": 43.42,
                        "ctrl_gap_mean_ms": 10.00, "ctrl_gap_max_ms": 10.05, "frames_late": "0/36"},
    "p30w4r":          {"solver": "greedy, partitioned, yolo on the whole P cluster", "chain_ms": 36.75,
                        "ctrl_gap_mean_ms": 10.00, "ctrl_gap_max_ms": 10.03, "frames_late": "0/36"},
    "p30w4cpr":        {"solver": "CP-SAT (hard windows), partitioned, yolo on the whole P cluster", "chain_ms": 36.75,
                        "ctrl_gap_mean_ms": 10.00, "ctrl_gap_max_ms": 10.02, "frames_late": "0/36"},
    "p30w4imer":       {"solver": "greedy, partitioned, yolo on the whole P cluster, conv on the IME", "chain_ms": 26.75,
                        "ctrl_gap_mean_ms": 10.00, "ctrl_gap_max_ms": 10.02, "frames_late": "0/36"},
    "p30w4imecpr":     {"solver": "CP-SAT (hard windows), partitioned, yolo on the whole P cluster, conv on the IME",
                        "chain_ms": 26.75, "ctrl_gap_mean_ms": 10.00, "ctrl_gap_max_ms": 10.34, "frames_late": "0/36"},
    # both clusters taken whole: yolo 4-wide on the P cores with conv on the IME, nav 4-wide on the E
    # cores, control distributed across them. The shorter chain is not the wider nav -- that stage's
    # span is 0.78 ms LONGER -- it is that nav now finishes closer to control's fixed 100 Hz tick, so
    # the wait from nav ending to the next control fire falls from 4.43 ms to 1.30 (docs/K1/nav_sharding.md).
    # NOTHING pinned: no allowed_machines, no machine_width. The solver is given the hardware's own
    # capability statement -- smt.vmadot on cluster 0 only -- and the measured per-width tables, and
    # places the three networks itself. It puts 808 of yolo's dispatches four-wide on the P cluster and
    # takes 679 IME dispatches, reaching the same 26.75 ms as the hand-partitioned table. The pinning
    # was not what bought the number; the capability description and the per-width costs were.
    "p30freer":        {"solver": "CP-SAT (hard windows), placement unconstrained, conv on the IME",
                        "chain_ms": 26.75, "chain_p95_ms": 30.1, "ctrl_gap_mean_ms": 9.99,
                        "ctrl_gap_max_ms": 18.06, "ctrl_gaps_over_15ms": "3/144", "frames_late": "0/36"},
    # the same freedom at a 22.2 ms camera period. The solver takes 1012 IME dispatches -- half again
    # as many as at 30 Hz -- picks a width per dispatch rather than per network, and spreads control
    # over all eight harts. It is the arm the 45 Hz comparison should have been flying: the scheduled
    # arms measured before it run 53-58 ms there, which is the hand-pinned baseline's own latency.
    # chain_ms is the pooled median over the three replicates, the same statistic as every other arm
    # here: r1/r2/r3 give 27.454 / 27.464 / 27.455 over 21 frames each, 27.455 pooled over 63.
    "p45freer":        {"solver": "CP-SAT (hard windows), placement unconstrained, conv on the IME, 45 Hz camera",
                        "chain_ms": 27.46, "chain_p95_ms": 37.2, "ctrl_gap_mean_ms": 9.92,
                        "ctrl_gap_max_ms": 20.19, "ctrl_gaps_over_15ms": "6/147", "frames_late": "0/66"},
    # the determinism check on the same 45 Hz unconstrained spec: re-solved with XPURT_CPSAT_WORKERS=1
    # (data/toplevel/wh_chain45_free_det1.json) and carried through the same codegen contract, so the
    # table is reproducible by recipe rather than by luck of the solver's thread interleaving.
    # Pooled over three replicates, 63 frames.
    "p45detr":         {"solver": "CP-SAT (hard windows), placement unconstrained, single-worker solve, 45 Hz camera",
                        "chain_ms": 30.07, "chain_p95_ms": 43.6, "ctrl_gap_mean_ms": 9.83,
                        "ctrl_gap_max_ms": 24.27, "frames_late": "0/63"},
    # the same unconstrained solve at a 27.78 ms camera period (data/toplevel/wh_chain36_free.json),
    # the rate at which the all-eight-hart ROS 2 baseline still enters the gate course. Three
    # replicates, pooled.
    "p36freer":        {"solver": "CP-SAT (hard windows), placement unconstrained, conv on the IME, 36 Hz camera",
                        "chain_ms": 25.82, "chain_p95_ms": 34.3, "ctrl_gap_mean_ms": 9.84,
                        "ctrl_gap_max_ms": 19.69, "ctrl_gaps_over_15ms": "9/144", "frames_late": "0/42"},
    "p30efullr":       {"solver": "CP-SAT (hard windows), both clusters whole, conv on the IME",
                        "chain_ms": 25.3, "chain_p95_ms": 30.2, "ctrl_gap_mean_ms": 10.00,
                        "ctrl_gap_max_ms": 29.88, "ctrl_gaps_over_15ms": "3/144", "frames_late": "0/36"},
    # the blanket-IME ablation: the same partitioned table with MB_IME_FORCE=1, so every one of
    # yolo's 63 IME-capable dispatches is on the engine instead of the 46 the measured table calls
    # wins. Costed from gen/mb_force, whose cells are the forced build's own per-dispatch numbers.
    "p30w4forcer":     {"solver": "CP-SAT (hard windows), partitioned, yolo on the whole P cluster, blanket IME",
                        "chain_ms": 30.08, "ctrl_gap_mean_ms": 10.00, "ctrl_gap_max_ms": 10.03, "frames_late": "0/36"},
    "a60cpsat_hardr": {"solver": "CP-SAT (hard windows), 60 Hz camera", "chain_ms": 58.5, "ctrl_gap_mean_ms": 10.06, "ctrl_gap_max_ms": 15.3, "frames_late": "0/72"},
    "a60greedyr": {"solver": "greedy, 60 Hz camera", "chain_ms": 583.6, "ctrl_gap_mean_ms": 17.48, "ctrl_gap_max_ms": 667.9, "frames_late": "72/72"},
    "a90cpsat_hardr": {"solver": "CP-SAT (hard windows), 90 Hz camera", "chain_ms": 55.9, "ctrl_gap_mean_ms": 10.0, "ctrl_gap_max_ms": 15.04, "frames_late": "0/108"},
    # greedy placement on the board per-width costs at the cameras where the two-instance ROS 2 graph was measured
    "fb30r1r": {"solver": "CP-SAT, per-rate feedback round 1, 30 Hz camera", "chain_ms": 30.1, "ctrl_gap_mean_ms": 9.99, "ctrl_gap_max_ms": 19.70, "frames_late": "0/81"},
    "w2pg36r": {"solver": "greedy + measured shard costs, 36 Hz camera", "chain_ms": 30.1, "ctrl_gap_mean_ms": 9.98, "ctrl_gap_max_ms": 17.23, "frames_late": "0/96"},
    "w2pg40r": {"solver": "greedy + measured shard costs, 40 Hz camera", "chain_ms": 39.8, "ctrl_gap_mean_ms": 10.06, "ctrl_gap_max_ms": 16.68, "frames_late": "0/108"},
    "a90greedyr": {"solver": "greedy, 90 Hz camera", "chain_ms": 947.5, "ctrl_gap_mean_ms": 25.67, "ctrl_gap_max_ms": 991.65, "frames_late": "111/111"},
    "a120hcpsat_hardr": {"solver": "CP-SAT (hard windows), 120 Hz camera, 200 ms table", "chain_ms": 59.9, "ctrl_gap_mean_ms": 10.0, "ctrl_gap_max_ms": 15.37, "frames_late": "0/36", "on_time": True},
    "a120hgreedyr": {"solver": "greedy, 120 Hz camera, 200 ms table", "chain_ms": 566.3, "ctrl_gap_mean_ms": 35.77, "ctrl_gap_max_ms": 513.81, "frames_late": "36/36", "on_time": False},
}
ROS_VANILLA = {   # ROS 2 as one would write it: one process per node, unpinned, default executor, default QoS
    # (arm, camera Hz): control-output gap mean, camera->goal median, goals reaching control per second (pooled over 3 runs)
    # per-network pinning with control on its own 100 Hz timer -- the strongest measured ROS 2
    # arrangement, and the measured stand-in for the modelled per-network-pinning baseline
    ("p3", 30):        {"ctrl_gap_mean_ms": 10.00, "chain_goal_ms": 30.4, "goals_per_s": 28.6},
    ("vanilla", 45):   {"ctrl_gap_mean_ms": 48.2, "chain_goal_ms": 265, "goals_per_s": 20.4},   # serial YOLO, control in the goal callback
    ("vanilla", 90):   {"ctrl_gap_mean_ms": 48.4, "chain_goal_ms": 159, "goals_per_s": 20.5},
    ("vanilla4", 45):  {"ctrl_gap_mean_ms": 26.0, "chain_goal_ms": 242, "goals_per_s": 38.4},   # the model's 4-hart build
    ("vanilla4", 90):  {"ctrl_gap_mean_ms": 25.6, "chain_goal_ms": 137, "goals_per_s": 38.7},
    ("vanilla4t", 45): {"ctrl_gap_mean_ms": 30.4, "chain_goal_ms": 121, "goals_per_s": 32.6},   # 4-hart YOLO, control on its 100 Hz timer, one process
    ("vanilla4tm", 45): {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 242, "goals_per_s": 32.6},   # 4-hart YOLO, control on its own 100 Hz timer in its own unpinned process
    ("vanilla4tm", 90): {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 137, "goals_per_s": 33.1},
    ("vanilla8", 45):  {"ctrl_gap_mean_ms": 26.0, "chain_goal_ms": 242.5, "goals_per_s": 38.4},   # the same one instance with an 8-hart pool: the width changes nothing
    ("vanilla8tm", 45): {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 241.9, "goals_per_s": 32.6},  # 8-hart pool, control on its own timer; goals per run match the 4-hart arm
    ("rvanilla", 45):  {"ctrl_gap_mean_ms": 83.8, "chain_goal_ms": 301, "goals_per_s": 11.7},   # + ffn_block and dronet as their own processes
    ("rvanilla4", 45): {"ctrl_gap_mean_ms": 26.4, "chain_goal_ms": 243, "goals_per_s": 37.3},   # 4-hart YOLO + the heavier stack: every core carries a node
    ("rvanilla4", 90): {"ctrl_gap_mean_ms": 26.3, "chain_goal_ms": 138, "goals_per_s": 37.7},
    ("vanilla4", 120): {"ctrl_gap_mean_ms": 26.1, "chain_goal_ms": 111, "goals_per_s": 37.9},   # the camera outruns the arm: 30% of frames reach a goal
    ("p3", 120):       {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 56, "goals_per_s": 38.6},   # unchanged from 45 Hz; 93% of frames reach a goal
    ("vanilla4x2", 45):  {"ctrl_gap_mean_ms": 22.2, "chain_goal_ms": 37, "goals_per_s": 44.9},    # pipelining by hand: two perception processes, all 8 cores busy
    # the whole machine and a real controller: two model instances across both clusters, control
    # on its own 100 Hz timer. The strongest arrangement of ROS 2 we have been able to build.
    ("vanilla4x2tm", 45): {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 37.8, "goals_per_s": 44.9},
    ("vanilla4x2", 60):  {"ctrl_gap_mean_ms": 16.6, "chain_goal_ms": 68, "goals_per_s": 59.7},
    ("vanilla4x2", 90):  {"ctrl_gap_mean_ms": 16.2, "chain_goal_ms": 253, "goals_per_s": 60.8},
    # the same two-instance graph across the camera sweep: control follows the pipeline's throughput,
    # min(camera, ~62 Hz), and the timer variant holds 100 Hz at every rate
    ("vanilla4x2", 25):    {"ctrl_gap_mean_ms": 40.0, "chain_goal_ms": 31.4, "goals_per_s": 23.6},
    ("vanilla4x2", 36):    {"ctrl_gap_mean_ms": 27.8, "chain_goal_ms": 32.2, "goals_per_s": 33.8},
    ("vanilla4x2", 40):    {"ctrl_gap_mean_ms": 25.0, "chain_goal_ms": 37.5, "goals_per_s": 38.1},
    ("vanilla4x2", 30):    {"ctrl_gap_mean_ms": 33.3, "chain_goal_ms": 31.4, "goals_per_s": 27.0},
    # the partitioned baselines: yolo on a 4-hart pool over the P cores, nav and control on the E
    # cluster, control still in the goal callback. cp3 gives each of the two a hart of its own;
    # cp3n4 shards nav four ways over the cluster and lets control share those harts.
    ("cp3", 30):           {"ctrl_gap_mean_ms": 33.4, "chain_goal_ms": 30.7, "goals_per_s": 25.4},
    # The all-eight-hart arm again with the SECOND YOLO pool's per-shard detail recorded, which is
    # the replicate the measured Gantt rows are drawn from. Instrumenting the second pool leaves the
    # mean cadence unchanged (27.77 ms, as r1-r3) and costs a little tail: p95 gap 36.2 ms against
    # 31.4 / 31.2 / 30.3 ms un-instrumented. The flights replay the un-instrumented cadence.
    ("vanilla4x2d2", 30):  {"ctrl_gap_mean_ms": 33.3, "chain_goal_ms": 31.6, "goals_per_s": None},
    ("vanilla4x2d2", 36):  {"ctrl_gap_mean_ms": 27.8, "chain_goal_ms": 32.8, "goals_per_s": None},
    # The same arm with a four-way worker pool under the NAVIGATION network as well, so no stage of
    # the perception->navigation chain is left on one hart -- the most capable ROS 2 arrangement
    # measured here. It is slower end to end than nav-on-one-hart because the board has no spare
    # harts: the pool takes YOLO's. a/b pin the pool to a cluster, c leaves it to the OS and is the
    # fastest of the three, so c is the variant the figures draw.
    ("vanilla4x2ns4a", 36): {"ctrl_gap_mean_ms": 27.8, "chain_goal_ms": 38.7, "goals_per_s": None},
    ("vanilla4x2ns4b", 36): {"ctrl_gap_mean_ms": 27.8, "chain_goal_ms": 39.2, "goals_per_s": None},
    ("vanilla4x2ns4c", 36): {"ctrl_gap_mean_ms": 27.8, "chain_goal_ms": 37.8, "goals_per_s": None},
    ("cp3n4", 30):         {"ctrl_gap_mean_ms": 33.4, "chain_goal_ms": 30.1, "goals_per_s": 25.4},
    ("vanilla4x2", 75):    {"ctrl_gap_mean_ms": 16.2, "chain_goal_ms": 291.9, "goals_per_s": 57.9},
    ("vanilla4x2", 120):   {"ctrl_gap_mean_ms": 16.4, "chain_goal_ms": 200.0, "goals_per_s": 57.0},
    ("vanilla4x2tm", 25):  {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 31.6, "goals_per_s": 23.4},
    ("vanilla4x2tm", 30):  {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 31.6, "goals_per_s": 28.4},
    ("vanilla4x2tm", 75):  {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 291.5, "goals_per_s": 57.7},
    ("vanilla4x2tm", 120): {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 200.4, "goals_per_s": 56.7},
    # the hand-placed layout (control in its own pinned process) and the multi-threaded executor, the arms the flight
    # campaigns replay as ros_p345 / ros_p390 / ros_multi45
    ("p3", 45):    {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 56.4, "goals_per_s": 38.9},   # camera->control is 61.6: control waits for its own timer
    ("p3", 90):    {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 56.2, "goals_per_s": 38.8},
    ("multi", 45): {"ctrl_gap_mean_ms": 10.0, "chain_goal_ms": 264.7, "goals_per_s": 19.9},
    # the submitted figure's baseline, measured: a 50 Hz control TIMER over serial YOLO. The timer is
    # configured at 50 Hz and starves to 20 Hz because the single-threaded executor cannot fire it
    # while a frame is in YOLO -- "serial on one hart backs up -> control starves", as its panel I says.
    ("vanilla_c50", 45): {"ctrl_gap_mean_ms": 49.2, "chain_goal_ms": 265.9, "goals_per_s": 8.4},
}
VITFLY_FRONTEND_MS = 0.35   # the VitFly convolutional front end (9 ops) alone on one hart, ModelBlaster/build/k1_vitfly
ROS_RICH = {   # the same heavier stack on ROS 2, 45 Hz camera, ros_traced/45_r*_r{1,2}
    "rspin_ctrl_gap_mean_ms": 53.2, "rspin_chain_ms": 198.2,      # YOLO 4-hart pool, one executor thread
    "rp3_ctrl_gap_mean_ms": 10.00, "rp3_chain_ms": 57.4,          # control in its own process
    "rmulti_ctrl_gap_mean_ms": 10.00, "rmulti_chain_ms": 247.3, "rmulti_drop_frac": 0.32,
}

# --- warm per-network execution --------------------------------------------------------------
WARM_MS = {"fused_full": 4.43, "mlp_control": 0.08, "dronet": 9.30, "ffn_block": 8.68}
COLD_OVER_WARM = {"fused_full": 2.75, "mlp_control": 1.29, "dronet": 1.27, "ffn_block": 1.41}

ROS_ARMS = {}   # filled by derive(): {(arm, rate_hz): {...}} from ros_traced/summary.csv


def provenance() -> str:
    return (f"XPU-RT control gap {XPURT['ctrl_gap_mean_ms']:.2f} ms mean / "
            f"{XPURT['ctrl_gap_max_ms']:.2f} max ({XPURT['source']}); "
            f"ROS control gap {ROS['ctrl_gap_mean_ms']:.2f} ms mean / "
            f"{ROS['ctrl_gap_max_ms']:.2f} max, default executor; "
            f"chain {CHAIN['xpurt_ms']:.2f} vs {CHAIN['ros_ms']:.2f} ms")


# --------------------------------------------------------------------------------------------
def _warm_gaps(net: str, trace: str):
    ends: dict[int, int] = {}
    for r in csv.DictReader(open(trace)):
        if r["network"] != net:
            continue
        try:
            e = int(r["actual_end_cycles"])
        except (ValueError, KeyError):
            continue
        if e <= 0:
            continue
        i = int(r["instance"])
        ends[i] = max(ends.get(i, 0), e)
    ks = sorted(ends)
    return [(ends[ks[j + 1]] - ends[ks[j]]) / HZ * 1000.0 for j in range(len(ks) - 1)]


def _spans(trace: str):
    span = {}
    for r in csv.DictReader(open(trace)):
        try:
            a, b = int(r["actual_start_cycles"]), int(r["actual_end_cycles"])
        except (ValueError, KeyError):
            continue
        if b <= 0:
            continue
        k = (r["network"], int(r["instance"]))
        lo, hi = span.get(k, (a, b)); span[k] = (min(lo, a), max(hi, b))
    return span


def _pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def derive() -> dict:
    """Recompute every constant that has a raw artifact on disk. Missing artifacts are reported,
    never silently replaced."""
    os.chdir(REPO)
    d: dict = {"missing": []}

    # XPU-RT control gaps: long traces first (hundreds of gaps), legacy short traces otherwise
    long_traces = sorted(glob.glob(f"{RES}/xpurt_long/trace_{XPURT_ARM}_other_run*.csv"))
    all_labels = sorted({re.search(r"trace_(\w+?)_other_run", os.path.basename(t)).group(1)
                         for t in glob.glob(f"{RES}/xpurt_long/trace_*_other_run*.csv")})
    d["xpurt_labels_on_disk"] = all_labels
    gaps = []
    for t in long_traces:
        g = _warm_gaps("mlp_control", t)
        gaps += g[1:]                         # instance 0 is cold
    if gaps:
        d["xpurt_ctrl_gap"] = {"mean": statistics.mean(gaps), "max": max(gaps), "p95": _pct(gaps, 0.95),
                               "n": len(gaps), "source": [os.path.relpath(t, REPO) for t in long_traces]}
    else:
        legacy = []
        for pat in ("w4", "b4", "b5y"):
            for t in glob.glob(f"ModelBlaster/tmp/{pat}_board_run/_gen/*/*_trace.csv"):
                legacy += _warm_gaps("mlp_control", t)
        if legacy:
            d["xpurt_ctrl_gap"] = {"mean": statistics.mean(legacy), "max": max(legacy), "n": len(legacy),
                                   "source": "ModelBlaster/tmp/{w4,b4,b5y}_board_run (legacy, 11 gaps each)"}
        else:
            d["missing"].append("xpurt control gaps")

    # XPU-RT chain: single-frame coupled trace (cold + warm-scaled) and, when present, the
    # per-frame chain from the long traces
    tr = f"{RES}/cmp_coupled_cpsat_board_trace.csv"
    if os.path.exists(tr):
        sp = _spans(tr)
        t0 = min(a for a, _ in sp.values()); cold = (max(b for _, b in sp.values()) - t0) / HZ * 1000.0
        y = sp.get(("yolov8_nano_64x96", 0))
        warm = (y[1] - y[0]) / HZ * 1000.0 + WARM_MS["fused_full"] + WARM_MS["mlp_control"] if y else None
        d["xpurt_chain_single_frame"] = {"cold_ms": cold, "warm_ms": warm}
    else:
        d["missing"].append("coupled trace")
    if long_traces:
        sys.path.insert(0, os.path.join(REPO, "scripts"))
        from make_measured_gantt_pair import per_frame_chain, read_trace   # noqa: E402
        per_label = {}
        for t in long_traces:
            lab = re.search(r"trace_(\w+?)_other_run", os.path.basename(t)).group(1)
            ch = per_frame_chain(read_trace(t))
            per_label.setdefault(lab, {"chain": [], "yolo": []})
            per_label[lab]["chain"] += [c for k, c, _ in ch if k >= 1]
            per_label[lab]["yolo"] += [yv for k, _, yv in ch if k >= 1]
        d["xpurt_chain_long"] = {lab: {"chain_median_ms": statistics.median(v["chain"]) if v["chain"] else None,
                                       "chain_p95_ms": _pct(v["chain"], 0.95) if v["chain"] else None,
                                       "yolo_span_median_ms": statistics.median(v["yolo"]) if v["yolo"] else None,
                                       "n_frames": len(v["chain"])} for lab, v in per_label.items()}
        # warm per-net medians from the same workload, instances >= 1
        wm = {}
        for t in long_traces:
            for (n, i), (a, b) in _spans(t).items():
                if i >= 1 and n in ("fused_full", "mlp_control"):
                    wm.setdefault(n, []).append((b - a) / HZ * 1000.0)
        d["warm_ms_long"] = {n: statistics.median(v) for n, v in wm.items()}

    # ROS arms from the pulled runs
    summ = f"{RES}/ros_traced/summary.csv"
    arms = {}
    if os.path.exists(summ):
        for r in csv.DictReader(open(summ)):
            tag = r["tag"]; m = re.match(r"(\d+)_(.+?)_r(\d+)$", tag)   # arm names may carry underscores (p3_q1, vanilla4_q1)
            if not m:
                continue
            hz, arm, rep = int(m.group(1)), m.group(2), int(m.group(3))
            a = arms.setdefault((arm, hz), {"gap_mean": [], "gap_max": [], "e2e_goal_med": [], "e2e_med": [],
                                            "n_frames": [], "n_goals": [], "reps": [], "executor": r["executor"],
                                            "busy": []})
            for k, col in (("gap_mean", "gap_mean_ms"), ("gap_max", "gap_max_ms"), ("e2e_goal_med", "e2e_goal_med_ms"),
                           ("e2e_med", "e2e_med_ms"), ("n_frames", "n_frames"), ("n_goals", "n_goals")):
                if r.get(col):
                    a[k].append(float(r[col]))
            a["reps"].append(rep)
            a["busy"].append({c: float(r[f"busy_c{c}"]) for c in range(8) if r.get(f"busy_c{c}")})
        for key, a in arms.items():
            a["gap_mean_pooled"] = statistics.mean(a["gap_mean"]) if a["gap_mean"] else None
            a["gap_max_pooled"] = max(a["gap_max"]) if a["gap_max"] else None
            a["e2e_goal_med_pooled"] = statistics.median(a["e2e_goal_med"]) if a["e2e_goal_med"] else None
            a["drop_frac"] = (1 - sum(a["n_goals"]) / sum(a["n_frames"])) if sum(a["n_frames"]) else None
            a["busiest_core_pct"] = max((max(b.values()) for b in a["busy"] if b), default=None)
        # sustained cap per arm: highest camera rate whose camera->goal median < 1.5 x period
        caps = {}
        for (arm, hz), a in arms.items():
            if a["e2e_goal_med_pooled"] is not None and a["e2e_goal_med_pooled"] < 1.5 * 1000.0 / hz:
                caps[arm] = max(caps.get(arm, 0), hz)
        d["ros_arms"] = {f"{arm}@{hz}": {k: v for k, v in a.items() if k not in ("busy",)} for (arm, hz), a in sorted(arms.items())}
        d["ros_caps_hz"] = caps
    else:
        d["missing"].append("ros_traced/summary.csv (run scripts/pull_ros_traced.py)")
    ROS_ARMS.update(arms)

    # YOLO standalone
    ys = {}
    for name in ("1core", "4core", "4core_E", "8core"):
        p = f"{RES}/ros_traced/yolo_standalone/{name}.txt"
        if os.path.exists(p):
            w = [int(x) / 24000.0 for x in re.findall(r"ITER_WALL \[\d+\] === (\d+)", open(p).read())]
            ver = re.search(r"max_abs_err=(\S+)", open(p).read())
            if len(w) > 2:
                ys[name] = {"warm_median_ms": statistics.median(w[1:]), "cold_ms": w[0],
                            "max_abs_err": float(ver.group(1)) if ver else None}
    if ys:
        d["yolo_standalone"] = ys
    else:
        d["missing"].append("yolo_standalone")
    return d


def verify() -> int:
    d = derive()
    ok = True
    def chk(label, got, rec, tol):
        nonlocal ok
        flag = "" if got is None or abs(got - rec) <= tol else "   <-- DRIFT"
        if got is not None and flag:
            ok = False
        print(f"{label:<44} derived {got if got is None else f'{got:8.2f}'}   recorded {rec:8.2f}{flag}")
    g = d.get("xpurt_ctrl_gap")
    if g:
        print(f"[XPU-RT control gap: n={g['n']} from {g['source'] if isinstance(g['source'], str) else len(g['source'])} trace(s)]")
        chk("XPU-RT control gap mean (ms)", g["mean"], XPURT["ctrl_gap_mean_ms"], 0.15)
        chk("XPU-RT control gap max (ms)", g["max"], XPURT["ctrl_gap_max_ms"], 0.15)
    c = d.get("xpurt_chain_single_frame")
    if c:
        chk("XPU-RT chain, single frame warm (ms)", c["warm_ms"], CHAIN["xpurt_single_frame_ms"], 0.1)
        chk("XPU-RT chain, single frame cold (ms)", c["cold_ms"], CHAIN["xpurt_cold_ms"], 0.1)
    for lab, v in d.get("xpurt_chain_long", {}).items():
        print(f"XPU-RT {lab}: chain median {v['chain_median_ms']}  p95 {v['chain_p95_ms']}  "
              f"yolo span median {v['yolo_span_median_ms']}  frames {v['n_frames']}")
        if lab == XPURT_ARM:
            chk(f"XPU-RT chain camera->control, {lab} (ms)", v["chain_median_ms"], CHAIN["xpurt_ms"], 0.5)
            chk(f"XPU-RT YOLO span per frame, {lab} (ms)", v["yolo_span_median_ms"], CHAIN["xpurt_yolo_span_ms"], 0.5)
    if d.get("xpurt_labels_on_disk"):
        print(f"XPU-RT traces on disk: {d['xpurt_labels_on_disk']}  (arm = {XPURT_ARM})")
    # every recorded operating point re-derives from its own traces
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from make_measured_gantt_pair import per_frame_chain, read_trace   # noqa: E402
    for lab, rec in XPURT_POINTS.items():
        ts = sorted(glob.glob(f"{RES}/xpurt_long/trace_{lab}_{rec.get('policy', 'other')}_run*.csv"))
        if not ts:
            print(f"XPU-RT point {lab}: no trace on disk"); continue
        chain, gaps, lags = [], [], []
        for t in ts:
            rows = read_trace(t)
            if len(rows) < 1000:
                continue
            chain += [c for k, c, _ in per_frame_chain(rows) if k >= 1]
            gaps += _warm_gaps("mlp_control", t)[1:]
            for k in sorted({r["inst"] for r in rows if r["net"] == "yolov8_nano_64x96"}):
                fr = [r for r in rows if (r["net"], r["inst"]) == ("yolov8_nano_64x96", k)]
                rel = min((r["rel"] for r in fr if r.get("rel") is not None), default=None)
                if rel is not None and k >= 1:
                    lags.append(min(r["s"] for r in fr) - rel)
        if chain and gaps:
            on_time = statistics.median(lags) < 1.0 if lags else None
            lab_p = lab if rec.get("policy", "other") == "other" else f"{lab} ({rec['policy']})"
            chk(f"XPU-RT {lab_p} camera->control (ms)", statistics.median(chain), rec["chain_ms"], 1.5)
            chk(f"XPU-RT {lab_p} control gap mean (ms)", statistics.mean(gaps), rec["ctrl_gap_mean_ms"], 0.1)
            if "ctrl_gap_max_ms" in rec:
                chk(f"XPU-RT {lab_p} control gap max (ms)", max(gaps), rec["ctrl_gap_max_ms"], 0.1)
            if "n_gaps" in rec:
                chk(f"XPU-RT {lab_p} control gaps counted", len(gaps), rec["n_gaps"], 0)
            flag = "" if on_time == rec["on_time"] else "   <-- DRIFT"
            if flag: ok = False
            print(f"{'XPU-RT ' + lab_p + ' frames on time':<44} derived {str(on_time):>8}   recorded {str(rec['on_time']):>8}{flag}")
    if d.get("warm_ms_long"):
        print(f"warm medians from the long traces: {d['warm_ms_long']}")
    ra = d.get("ros_arms", {})
    if ra:
        print(f"{'ROS arm@camera':<16} {'gap mean':>9} {'gap max':>8} {'cam->goal':>10} {'drop':>6} {'busiest':>8}  reps")
        for k, a in ra.items():
            print(f"{k:<16} {a['gap_mean_pooled'] or 0:9.2f} {a['gap_max_pooled'] or 0:8.2f} "
                  f"{a['e2e_goal_med_pooled'] or 0:10.2f} {100*(a['drop_frac'] or 0):5.0f}% {a['busiest_core_pct'] or 0:7.0f}%  {a['reps']}")
        print(f"sustained camera rate per arm (cam->goal median < 1.5 x period): {d.get('ros_caps_hz')}")
        s15 = ra.get("ship@15"); s20 = ra.get("ship@20"); s25 = ra.get("ship@25"); m = ra.get("multi@15")
        if s15:
            chk("ROS default executor gap mean @15 Hz cam", s15["gap_mean_pooled"], ROS["ctrl_gap_mean_ms"], 1.0)
            chk("ROS chain camera->goal @15 Hz cam", s15["e2e_goal_med_pooled"], CHAIN["ros_ms"], 1.5)
            chk("ROS default executor gap max @15 Hz cam", s15["gap_max_pooled"], ROS["ctrl_gap_max_ms"], 0.5)
        if s20:
            chk("ROS default executor gap mean @20 Hz cam", s20["gap_mean_pooled"], ROS["ctrl_gap_mean_cam20_ms"], 1.5)
        if s25:
            chk("ROS default executor gap mean @25 Hz cam", s25["gap_mean_pooled"], ROS["ctrl_gap_mean_cam25_ms"], 1.5)
        sp45 = ra.get("spin@45"); cs45 = ra.get("cspin@45")
        if sp45:
            chk("ROS pool arm gap mean @45 Hz cam (figure)", sp45["gap_mean_pooled"], ROS["ctrl_gap_mean_pool_cam45_ms"], 0.5)
            chk("ROS pool arm camera->goal @45 Hz cam", sp45["e2e_goal_med_pooled"], CHAIN["ros_pool_cam45_ms"], 3.0)
        if cs45:
            chk("ROS chained pool arm gap mean @45 Hz cam", cs45["gap_mean_pooled"], ROS["ctrl_gap_mean_chained_pool_cam45_ms"], 0.5)
        if m:
            chk("ROS multi executor gap mean @15 Hz cam", m["gap_mean_pooled"], ROS["ctrl_gap_mean_mt_ms"], 0.2)
    for pre, rec in SOLVER_ARMS.items():
        ts = sorted(glob.glob(f"{RES}/xpurt_long/trace_{pre}[0-9]_other_run1.csv"))
        if not ts:
            print(f"solver arm {pre}: no traces"); continue
        chain, gaps, late, n = [], [], 0, 0
        for t in ts:
            rows = read_trace(t); chain += [c for k, c, _ in per_frame_chain(rows) if k >= 1]; gaps += _warm_gaps("mlp_control", t)[1:]
            for k in sorted({r["inst"] for r in rows if r["net"] == "yolov8_nano_64x96"}):
                fr = [r for r in rows if (r["net"], r["inst"]) == ("yolov8_nano_64x96", k)]
                rel = min((r["rel"] for r in fr if r.get("rel") is not None), default=None)
                if rel is not None and k >= 1 and rel >= 100:
                    n += 1; late += (max(r["e"] for r in fr) - rel > 66.67)
        chk(f"{rec['solver']} camera->control median (ms)", statistics.median(chain), rec["chain_ms"], 1.0)
        chk(f"{rec['solver']} control gap mean (ms)", statistics.mean(gaps), rec["ctrl_gap_mean_ms"], 0.1)
        chk(f"{rec['solver']} control gap max (ms)", max(gaps), rec["ctrl_gap_max_ms"], 0.1)
        chk(f"{rec['solver']} frames late (of {n})", late, int(rec["frames_late"].split("/")[0]), 0)
    for (arm, hz), rec in ROS_VANILLA.items():
        a = ra.get(f"{arm}@{hz}")
        if not a:
            print(f"ROS vanilla {arm}@{hz}: no runs"); continue
        chk(f"ROS {arm} @{hz} Hz cam control gap mean (ms)", a["gap_mean_pooled"], rec["ctrl_gap_mean_ms"], 0.15)
        chk(f"ROS {arm} @{hz} Hz cam camera->goal (ms)", a["e2e_goal_med_pooled"], rec["chain_goal_ms"], 2.0)
    summ = f"{RES}/ros_traced/summary.csv"
    if os.path.exists(summ):
        by_tag = {r["tag"]: r for r in csv.DictReader(open(summ))}
        for tag, rec in ROS_SENSITIVITY.items():
            r = by_tag.get(tag)
            if not r:
                print(f"ROS sensitivity {tag}: not in summary.csv"); continue
            chk(f"ROS {tag} control gap mean (ms)", float(r["gap_mean_ms"]), rec["ctrl_gap_mean_ms"], 0.1)
            chk(f"ROS {tag} camera->goal (ms)", float(r["e2e_goal_med_ms"]), rec["chain_goal_ms"], 0.5)
    y = d.get("yolo_standalone", {})
    if y:
        chk("YOLO alone, 1 hart warm (ms)", y["1core"]["warm_median_ms"], YOLO_STANDALONE_MS["1hart"], 0.5)
        chk("YOLO alone, 4-hart pool warm (ms)", y["4core"]["warm_median_ms"], YOLO_STANDALONE_MS["4hart_pool"], 0.5)
        if "4core_E" in y:
            chk("YOLO alone, 4-hart E-cluster pool (ms)", y["4core_E"]["warm_median_ms"], YOLO_STANDALONE_MS["4hart_E_pool"], 0.5)
        if "8core" in y:   # the kernel's own scaling ceiling: eight harts buy nothing over four
            chk("YOLO alone, 8-hart pool warm (ms)", y["8core"]["warm_median_ms"], YOLO_STANDALONE_MS["8hart_pool"], 0.5)
        print(f"YOLO standalone verify vs golden: max_abs_err {[v['max_abs_err'] for v in y.values()]}")
    if d["missing"]:
        print("not re-derived (artifact missing):", ", ".join(d["missing"]))
    print(f"\n{provenance()}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--json", action="store_true", help="print derive() as JSON")
    a = ap.parse_args()
    if a.json:
        print(json.dumps(derive(), indent=1, default=str)); sys.exit(0)
    sys.exit(verify() if a.verify else (print(provenance()) or 0))

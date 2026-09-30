#!/usr/bin/env python3
"""One table of every run in the study, with what each configuration is, in words, and how the
runs relate to one another.

    results/codesign_feedback/master_runs.csv     one row per run
    results/codesign_feedback/master_configs.csv  one row per configuration named in the runs table

Kinds of row: `board_ros` (a ROS 2 layout executed on the K1), `board_xpurt` (an XPU-RT table
executed on the K1), `solve` (a solver certificate or table solve on the host), `flight` (one
simulated flight), `energy` (an energy flight). Every row carries `run_id`, `kind`, `config`,
`config_description`, `source`, `family` (the campaign / study the run belongs to), `cell` (the
runs it is directly comparable with: same scene, speed, gain, seed) and `related_runs`; the
run's own columns follow, blank where a column does not apply to that kind. Flights carry the
scene (course, gate positions, density, people height, walking speed, layout seed, nav weights),
the replayed deployment (cadence trace, its board run, latency, goal hold), the outcome (gates,
steps, flight time, crash type, crash position and time, progress, path length, the recorded
path file), the rotor-model power and energy (whole flight, and up to the moment the comparable
run crashed), the navigation command statistics, and for XPU-RT arms the schedule the board
executed, the solver and its configuration, the spec's windows, the predicted metrics, the board
metrics, and the counterpart tables (greedy / CP-SAT / feedback rounds) for the same spec.
Re-run any time; it reads only artifacts on disk.

    scripts/build_master_csv.py [--out results/codesign_feedback/master_runs.csv]
"""
from __future__ import annotations
import argparse, csv, glob, json, os, random, re, statistics, sys
import numpy as np
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
sys.path.insert(0, os.path.join(REPO, "scripts"))
CONTROL_DT_MS = 10.0

# ---------------------------------------------------------------- configurations, in words
ROS_ARMS = {
    "ship":     "ROS 2, one process, single-threaded executor, serial YOLO on one core, control on its 100 Hz timer, unpinned",
    "cship":    "ROS 2, one process, single-threaded executor, serial YOLO, control chained to the goal topic (once per perception result)",
    "spin":     "ROS 2, one process pinned to P0-3, single-threaded executor, YOLO on a 4-hart pool (P0-3), main thread on P0, control timer",
    "cspin":    "as spin, control chained to the goal topic",
    "multi":    "ROS 2, one process, MultiThreadedExecutor, serial YOLO, control timer, unpinned",
    "smte":     "ROS 2, one process pinned to P0-3, MultiThreadedExecutor, YOLO on a 4-hart pool, control timer",
    "p3":       "ROS 2, three processes: perception with the 4-hart YOLO pool pinned to P0-3, nav on E4, control on its 100 Hz timer on E5 (hand-tuned layout)",
    "cp3":      "as p3, control chained to the goal topic",
    "p8":       "ROS 2, three processes: perception pool on P0-3, nav on E4-5, control on E6-7",
    "yproc":    "ROS 2, two processes: perception alone (pool P0-3); nav and control together on E4",
    "nproc":    "ROS 2, two processes: nav alone on E4; perception (pool P0-3) and control together",
    "rspin":    "as spin plus ffn_block 10 Hz and dronet 30 Hz as two more nodes in the same process (heavier stack)",
    "rp3":      "as p3 plus ffn_block 10 Hz and dronet 30 Hz in a fourth process on E6-7 (heavier stack)",
    "rmulti":   "as multi plus ffn_block 10 Hz and dronet 30 Hz (heavier stack)",
    "vanilla":  "ROS 2 as written out of the box: one node per stage, one process each, unpinned, default executor and QoS (depth 10), serial YOLO, control chained to the goal topic",
    "rvanilla": "as vanilla plus ffn_block 10 Hz and dronet 30 Hz, each its own unpinned process",
    "vanilla4": "as vanilla, the perception node using the model's 4-hart YOLO build (the pool pins its own workers to P0-3; ROS itself unpinned)",
    "rvanilla4": "as vanilla4 plus ffn_block and dronet as their own unpinned processes: every core carries a node",
    "vanilla4t": "ROS 2, ONE process, default single-threaded executor, 4-hart YOLO pool, control on its 100 Hz timer (shares the executor with perception)",
    "vanilla4tm": "as vanilla4 (one process per node, unpinned, 4-hart YOLO) with control on its own 100 Hz timer in its own process, acting on the held goal",
    "vanilla4x2": "pipelining by hand: the camera alternates frames between two perception processes with 4-hart pools on P0-3 and E4-7; nav and control unpinned",
    "x2spin":   "two cameras (not in the figures): as spin with a second camera/perception pair",
    "x2p":      "two cameras (not in the figures): two pinned perception processes (P0-3, E4-7), nav and control pinned",
    "x2rspin":  "two cameras plus the heavier stack, spin layout (not in the figures)",
    "x2rmulti": "two cameras plus the heavier stack, MultiThreadedExecutor (not in the figures)",
    "x2rp3":    "two cameras plus the heavier stack, p3 layout (not in the figures)",
}
ROS_CTRL_MODE = {a: ("chained (once per goal)" if a.startswith("c") and a not in ("cship",) or a in ("cship", "cspin", "cp3", "vanilla", "rvanilla", "vanilla4", "rvanilla4", "vanilla4x2") else "100 Hz timer on the held goal") for a in ROS_ARMS}
ROS_SUFFIX = {"q1": "QoS history depth 1 instead of 10", "c200": "control timer at 200 Hz", "hog2": "two CPU-hog processes added", "smoke": "smoke run"}
SPEC_OF_TAG = {"a": "wh_chain45_solve", "a30": "wh_chain30_solve_500", "a60": "wh_chain60_solve_500", "a90": "wh_chain90_solve_500", "a120": "wh_chain120_solve_500",
               "a120h": "wh_chain120_solve_h200", "a150h": "wh_chain150_solve_h200", "b5": "wh_chain90_rich_solve_500", "ash": "wh_chain45_shard_solve",
               "fba90": "wh_chain90_solve_500", "fba120h": "wh_chain120_solve_h200", "fbb5": "wh_chain90_rich_solve_500"}
CAL_OF_TAG = {"a": "k1_board_calibration_yolo110.json (stock per-op table, YOLO dispatches x1.10)", "ash": "gen_mb_cal (YOLO multi-hart rows from executed shard traces)"}
FLIGHT_TRACES = {
    "xpu_a_cpsat_hard.csv": "XPU-RT CP-SAT, 45 Hz camera: control outputs at 10.0 ms measured on the K1 (96/s over the looped table)",
    "xpu_a_greedy.csv": "XPU-RT greedy, 45 Hz camera: measured bursty control outputs (a burst then silence)",
    "xpu_a90_cpsat.csv": "XPU-RT CP-SAT, 90 Hz camera: measured 10 ms control outputs",
    "xpu_a90_greedy.csv": "XPU-RT greedy, 90 Hz camera: measured bursty control outputs",
    "xpu_a120h_cpsat.csv": "XPU-RT CP-SAT, 120 Hz camera (200 ms table): measured control outputs",
    "ros_vanilla45.csv": "ROS 2 vanilla (serial YOLO, control chained), 45 Hz camera: 21 control outputs/s measured",
    "ros_vanilla445.csv": "ROS 2 vanilla with the 4-hart YOLO build, control chained, 45 Hz camera: 39 control outputs/s measured",
    "ros_vanilla4t45.csv": "ROS 2 vanilla4t (one process, control timer sharing the executor), 45 Hz: 33 outputs/s measured",
    "ros_vanilla4tm45.csv": "ROS 2 vanilla, control on its own timer in its own process, 45 Hz: 100 outputs/s measured",
    "ros_vanilla4tm90.csv": "ROS 2 vanilla, control on its own timer in its own process, 90 Hz camera: 100 outputs/s measured",
    "ros_vanilla4_q145.csv": "ROS 2 vanilla4 with QoS depth 1, 45 Hz: 39 outputs/s measured",
    "ros_vanilla4x245.csv": "ROS 2 vanilla, two perception processes alternating frames, 45 Hz: 60 outputs/s measured",
    "ros_p345.csv": "ROS 2 hand-tuned p3 layout, 45 Hz: 100 outputs/s measured",
    "ros_p3_q145.csv": "ROS 2 hand-tuned p3 layout with QoS depth 1, 45 Hz: 100 outputs/s measured",
    "ros_p390.csv": "ROS 2 hand-tuned p3 layout, 90 Hz camera: 100 outputs/s measured",
    "ros_spin45_r1.csv": "ROS 2 spin layout (one pinned process, pool), 45 Hz: measured control outputs",
    "xpurt_best45alt2long.csv": "XPU-RT hand placement best45alt2, 45 Hz: measured control outputs",
    "xpurt_greedy_long25.csv": "XPU-RT greedy, 25 Hz coupled chain: measured bursty control outputs",
}
CAMPAIGNS = {
    "campaign_v2/campaign_v2.csv": ("showdown_v2", "flight campaign v2: replayed K1 control cadences, course A, density 0.30, people 1.7 m — fixed and calibrated gains", "nav_fused_v12_cnn.pt"),
    "campaign_v2_courseB/campaign_v2.csv": ("showdown_v2_courseB", "flight campaign v2 on the unseen-gate course B, replayed cadences, people 1.7 m", "nav_fused_v12_cnn.pt"),
    "campaign_env/env_sweep.csv": ("env_sweep", "environment sweep: course x obstacle density x speed, replayed cadences, people 2.4 m, every flight recorded", "nav_fused_v12_cnn.pt"),
    "campaign_percep/campaign.csv": ("latency_replay", "replay of cadence + measured camera-to-control latency (+ goal-rate hold where set), people 2.4 m; also the tuned ROS layouts and camera-rate arms", "nav_fused_v12_cnn.pt"),
    "campaign_qos1/campaign.csv": ("qos1", "QoS-depth-1 ROS deployment, cadence + latency and cadence-only, people 2.4 m", "nav_fused_v12_cnn.pt"),
    "campaign_tallcal/campaign.csv": ("tall_calibrated_gain", "calibrated gain (0.5 / replayed rate) for both main arms in the 2.4 m-people scene", "nav_fused_v12_cnn.pt"),
    "campaign_courseC/campaign.csv": ("courseC", "gate course C (seeded generator), replayed cadences, people 2.4 m", "nav_fused_v12_cnn.pt"),
    "campaign_walk/campaign.csv": ("walking_1.5", "people walking at 1.5 m/s instead of 0.8 m/s, replayed cadences, people 2.4 m", "nav_fused_v12_cnn.pt"),
    "campaign_tallnet/campaign.csv": ("tall_retrained_net", "the guidance net re-trained for 2.4 m people, replayed cadences", "nav_fused_v20_tall_cnn.pt"),
    "campaign_rich/campaign.csv": ("heavier_stack", "the heavier stack (ffn_block 10 Hz + dronet 30 Hz added): XPU-RT's 90 Hz rich table vs the six-process vanilla graph, cadence + latency + goal rate replayed, people 2.4 m", "nav_fused_v12_cnn.pt"),
    "campaign_seeds24/campaign.csv": ("display_cell_seeds_13_24", "twelve more seeds (1012-1023) on the displayed cell, both main arms", "nav_fused_v12_cnn.pt"),
    "campaign_cross/campaign.csv": ("crossing_people", "people crossing the aisle (east-west) instead of patrolling along it, replayed cadences + latency, people 2.4 m", "nav_fused_v12_cnn.pt"),
    "campaign/campaign.csv": ("showdown_v1", "first recreation campaign: fixed control-gap latencies (ceil(gap/10 ms) hold), people 1.7 m", "nav_fused_v12_cnn.pt"),
    "hil_ablation.csv": ("envelope_fixed_gain", "flight envelope: speed x control latency (ZOH hold), fixed gain, people 1.7 m", "nav_fused_v12_cnn.pt"),
    "gain_controlled/gain_controlled.csv": ("envelope_calibrated_gain", "flight envelope, calibrated gain per rate (0.5/rate), people 1.7 m", "nav_fused_v12_cnn.pt"),
    "gain_controlled/envelope_20hz.csv": ("envelope_20hz", "flight envelope, 20 Hz cells, both gain policies", "nav_fused_v12_cnn.pt"),
}
PERSON_H_CHANGE_EPOCH = 1789119420   # 2026-09-11 10:37 local: people 1.7 m before, 2.4 m after (rows without a person_h column)


def course_gates(course, seed=8):
    if course == "b":
        return [(-7.75, 8.5), (-8.30, 12.5), (-7.80, 16.5), (-8.25, 20.5)]
    if course == "c":
        rng = random.Random(seed); y0 = rng.uniform(8.0, 9.5)
        return [(round(-8.0 + rng.uniform(-0.35, 0.35), 2), round(y0 + 4.0 * k, 2)) for k in range(4)]
    return [(-8.05, 9.0), (-8.30, 13.0), (-7.75, 17.0), (-8.05, 21.0)]


def describe_ros(tag):
    m = re.match(r"(\d+)_(.+?)(?:_(q1|c200|hog2|smoke))?(?:_r(\d+))?$", tag)
    if not m:
        return tag, tag, "", "", "", ""
    hz, arm, suf, rep = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    d = ROS_ARMS.get(arm, f"ROS 2 layout {arm}")
    if suf:
        d += f"; {ROS_SUFFIX[suf]}"
    return f"ros_{arm}{'_' + suf if suf else ''}@{hz}Hz", f"{d}; camera at {hz} Hz; 20 s run", hz, arm, rep, suf or ""


def xpu_label_parts(label):
    """label like 'acpsat_hardr1', 'fba90r1greedyr2', 'best45alt2', 'a120hcpsat_hardr3' -> (tag, solver, round, rep)."""
    m = re.match(r"^fb(a90|a120h|b5)r(\d)(cpsat_hard|greedy)r(\d)$", label)
    if m:
        return "fb" + m.group(1), m.group(3), int(m.group(2)), int(m.group(4))
    m = re.match(r"^(a120h|a150h|a120|a30|a60|a90|ash|b5|a)(cpsat_hard|cpsat_soft|greedy)r(\d)$", label)
    if m:
        return m.group(1), m.group(2), None, int(m.group(3))
    return None, None, None, None


def describe_xpu(label):
    tag, solver, rnd, rep = xpu_label_parts(label)
    if tag:
        spec = SPEC_OF_TAG.get(tag, "?"); sol = "CP-SAT (hard windows)" if solver == "cpsat_hard" else ("CP-SAT (soft)" if solver == "cpsat_soft" else "greedy (list scheduling)")
        base = {"a": "the 45 Hz chain spec (1 s table)", "a30": "the chain at a 30 Hz camera (half-second table)", "a60": "the chain at a 60 Hz camera (half-second table)",
                "a90": "the chain at a 90 Hz camera (half-second table)", "a120": "the chain at a 120 Hz camera (half-second table)", "a120h": "the chain at a 120 Hz camera (200 ms certificate table)",
                "a150h": "the chain at a 150 Hz camera (200 ms table)", "b5": "the 90 Hz camera + heavier stack spec (half-second table)", "ash": "the 45 Hz chain with YOLO free to shard (gen_mb_cal costing)",
                "fba90": "the chain at a 90 Hz camera", "fba120h": "the chain at a 120 Hz camera (200 ms table)", "fbb5": "the 90 Hz camera + heavier stack"}[tag]
        d = f"XPU-RT, {sol} table of {base} ({spec}), executed on the K1"
        if rnd is not None:
            d += f"; HIL feedback study round {rnd}: " + ("solved from the isolated profile, no board knowledge" if rnd == 0 else f"re-solved on the per-dispatch calibration fitted from round {rnd - 1}'s executed CP-SAT traces")
        elif tag in CAL_OF_TAG:
            d += f"; costed with {CAL_OF_TAG[tag]}"
        else:
            d += "; costed with the yolo x1.10 board calibration"
        return d
    if label.startswith("best45alt2"):
        return "XPU-RT, hand placement best45alt2: 45 Hz camera, control on its own hart (the first recreation's arm)"
    m = re.match(r"^best(\d+)", label)
    if m:
        return f"XPU-RT, hand placement at a {m.group(1)} Hz camera"
    if label.startswith("rich45"):
        return "XPU-RT, hand placement, 45 Hz camera plus ffn_block and dronet"
    return f"XPU-RT table {label}"


def trace_metrics(path):
    from make_measured_gantt_pair import per_frame_chain, read_trace
    from measured_timing import _warm_gaps
    rows = read_trace(path)
    if len(rows) < 200:
        return {}
    chain = [c for k, c, _ in per_frame_chain(rows) if k >= 1]
    gaps = _warm_gaps("mlp_control", path)[1:]
    late = n = 0; worst = 0.0
    for k in sorted({r["inst"] for r in rows if r["net"] == "yolov8_nano_64x96"}):
        fr = [r for r in rows if (r["net"], r["inst"]) == ("yolov8_nano_64x96", k)]
        rel = min((r["rel"] for r in fr if r.get("rel") is not None), default=None)
        if rel is not None and k >= 1 and rel >= 100:
            n += 1; l = max(r["e"] for r in fr) - rel - 66.67; late += l > 0; worst = max(worst, l)
    return {"board_chain_median_ms": round(statistics.median(chain), 2) if chain else "", "board_chain_p95_ms": round(sorted(chain)[int(0.95 * (len(chain) - 1))], 2) if chain else "",
            "board_ctrl_gap_mean_ms": round(statistics.mean(gaps), 3) if gaps else "", "board_ctrl_gap_max_ms": round(max(gaps), 2) if gaps else "",
            "board_n_frames": len({r["inst"] for r in rows if r["net"] == "yolov8_nano_64x96"}), "board_frames_late": late, "board_frames_counted": n,
            "board_worst_frame_lateness_ms": round(worst, 2), "board_run_ms": round(max(r["e"] for r in rows), 1),
            "board_dispatches_per_hart": json.dumps({h: sum(1 for r in rows if r["hart"] == h) for h in sorted({r["hart"] for r in rows})})}


def schedule_info(sched_path):
    """Predicted metrics of a table (its _metrics.json), the spec it came from, the solver configuration."""
    out = {}
    if not sched_path:
        return out
    p = os.path.join(REPO, sched_path) if not os.path.isabs(sched_path) else sched_path
    base = re.sub(r"_clamped\.json$", ".json", p)
    met = base.replace(".json", "_metrics.json")
    if os.path.exists(met):
        m = json.load(open(met))
        out.update({"sched_solver": m.get("scheduler", ""), "sched_num_operations": m.get("num_operations", ""), "sched_predicted_makespan_ms": round(float(m.get("makespan_ms", 0) or 0), 2),
                    "sched_predicted_deadline_misses": m.get("deadline_miss_count", ""), "sched_predicted_op_misses": m.get("op_deadline_miss_count", ""),
                    "sched_predicted_total_lateness_ms": m.get("total_lateness_ms", ""), "sched_predicted_max_lateness_ms": m.get("max_lateness_ms", ""),
                    "sched_solver_wall_s": round(float(m.get("solver_wall_time_s", 0) or 0), 1), "sched_critical_path_ms": m.get("critical_path_ms", ""),
                    "sched_per_machine_utilization": json.dumps(m.get("per_machine_utilization", "")) if isinstance(m.get("per_machine_utilization"), dict) else m.get("per_machine_utilization", "")})
    m = re.search(r"fig_([a-z0-9]+)_(cpsat_hard|cpsat_soft|greedy)", os.path.basename(p))
    if m:
        tag, solver = m.group(1), m.group(2)
        rnd = None
        mm = re.match(r"^(fb(?:a90|a120h|b5))r(\d)$", tag)
        if mm:
            tag, rnd = mm.group(1), int(mm.group(2))
        spec = SPEC_OF_TAG.get(tag, "")
        out.update({"sched_tag": tag, "sched_solver_kind": solver, "sched_spec": spec, "sched_feedback_round": "" if rnd is None else rnd})
        sp = os.path.join(REPO, "data", "toplevel", spec + ".json")
        if spec and os.path.exists(sp):
            s = json.load(open(sp))
            out["sched_solver_config"] = json.dumps(s.get("scheduler", {}))
            out["sched_spec_networks"] = json.dumps({n: {"period_ms": round(float(v.get("period", 0)), 3), "window_ms": round(float(v.get("window_duration", 0)), 3), "instances": v.get("num_instances"), "stateful": v.get("stateful", False)} for n, v in s["networks"].items()})
            out["sched_spec_edges"] = json.dumps(s.get("edges", [])); out["sched_horizon_ms"] = s.get("horizon_ms", "")
            hz = re.search(r"chain(\d+)", spec); out["camera_hz"] = hz.group(1) if hz else ""
        cal = CAL_OF_TAG.get(tag, "k1_board_calibration_yolo110.json (stock per-op table, YOLO dispatches x1.10)")
        if rnd is not None:
            cal = "none (isolated profile)" if rnd == 0 else f"hil_feedback/cal_{tag[2:]}_r{rnd}.json (per-dispatch, fitted from round {rnd - 1}'s executed traces)"
        out["sched_calibration"] = cal
        # the counterpart tables for the same spec: before optimisation (greedy), after (CP-SAT), after feedback rounds
        pre = f"fig_{tag}" if rnd is None else f"fig_{tag}r0"
        out["schedule_greedy_same_spec"] = f"schedules/{pre}_greedy_clamped.json" if os.path.exists(os.path.join(REPO, "schedules", f"{pre}_greedy_clamped.json")) else ""
        out["schedule_cpsat_same_spec"] = f"schedules/{pre}_cpsat_hard_clamped.json" if os.path.exists(os.path.join(REPO, "schedules", f"{pre}_cpsat_hard_clamped.json")) else ""
        fbt = tag if tag.startswith("fb") else "fb" + tag
        fb = [f"schedules/fig_{fbt}r{k}_cpsat_hard_clamped.json" for k in (1, 2) if os.path.exists(os.path.join(REPO, "schedules", f"fig_{fbt}r{k}_cpsat_hard_clamped.json"))]
        out["schedule_after_hil_feedback"] = ";".join(fb)
    return out


def trace_header(path):
    h = {}
    for line in open(path):
        if not line.startswith("#"):
            break
        for k, v in re.findall(r"(\w+)=(\S+)", line):
            h[k] = v
    return h


def flight_record(npz, pair_crash_t=None):
    from flight_energy_model import rotor_thrusts
    d = np.load(npz, allow_pickle=True); w = d["wrench"].astype(float).copy(); t = d["t_s"].astype(float); p = d["poses"]
    out = {"record": os.path.relpath(npz, REPO), "crash_x": round(float(p[-1, 0]), 3), "crash_y": round(float(p[-1, 1]), 3), "crash_z": round(float(p[-1, 2]), 3),
           "end_time_s": round(float(t[-1]), 2), "start_x": round(float(p[0, 0]), 3), "start_y": round(float(p[0, 1]), 3),
           "path_len_m": round(float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum()), 2), "max_y_m": round(float(p[:, 1].max()), 2),
           "mean_speed_mps": round(float(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1).sum() / max(t[-1] - t[0], 1e-6)), 3),
           "gates_xy": json.dumps([[round(float(x), 2), round(float(y), 2)] for x, y in d["gates_world"]])}
    g = d["goal_cmd"].astype(float); iw = d["imu_w"].astype(float)
    out.update({"mean_cmd_fwd_mps": round(float(g[:, 1].mean()), 3) if g.shape[1] > 1 else "", "mean_abs_cmd_yaw": round(float(np.abs(g[:, 0]).mean()), 3),
                "mean_abs_body_rate": round(float(np.abs(iw).mean()), 3), "max_abs_body_rate": round(float(np.abs(iw).max()), 3)})
    if str(d["outcome"]) != "crash":
        out.update({"crash_x": "", "crash_y": "", "crash_z": ""})
    if len(t) >= 2 and not np.allclose(w, 0):
        w[:, 0] = np.clip(1.0 + w[:, 0], 0.0, None)
        P = (rotor_thrusts(w, 0.09, 0.016) ** 1.5).sum(axis=1); dt = np.gradient(t); E = np.cumsum(P * dt)
        out.update({"energy_model_total": round(float(E[-1]), 3), "mean_power_model": round(float(E[-1] / max(t[-1] - t[0], 1e-6)), 4), "mean_abs_moment": round(float(np.abs(w[:, 1:4]).mean()), 5),
                    "peak_power_model": round(float(P.max()), 3)})
        if pair_crash_t is not None and t[-1] >= pair_crash_t:
            i = int(np.searchsorted(t, pair_crash_t)); i = min(max(i, 1), len(E) - 1)
            out.update({"energy_model_until_pair_crash": round(float(E[i]), 3), "mean_power_until_pair_crash": round(float(E[i] / max(t[i] - t[0], 1e-6)), 4)})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(RES, "master_runs.csv")); a = ap.parse_args()
    rows, configs = [], {}
    # ---- 1. ROS 2 board runs
    ros_by_tag = {}
    summ = os.path.join(RES, "ros_traced", "summary.csv")
    if os.path.exists(summ):
        for r in csv.DictReader(open(summ)):
            cfg, desc, hz, arm, rep, suf = describe_ros(r["tag"]); configs[cfg] = desc
            row = {"run_id": f"board_ros:{r['tag']}", "kind": "board_ros", "config": cfg, "config_description": desc, "source": f"results/codesign_feedback/ros_traced/{r['tag']}/",
                   "family": "ros_board_matrix", "cell": f"ros:{arm}{'_' + suf if suf else ''}@{hz}", "replicate": rep, "camera_hz": hz, "ros_arm": arm, "ros_variant": suf,
                   "ros_control_mode": ROS_CTRL_MODE.get(arm, ""), "ros_qos_depth": 1 if suf == "q1" else 10, "ros_control_timer_hz": 200 if suf == "c200" else 100,
                   **{("ros_" + k if k in ("executor", "nodes", "affinity_mask", "sched_policy", "yolo_pool", "rmw") else k): v for k, v in r.items()}}
            rows.append(row); ros_by_tag[r["tag"]] = row
    # ---- 2. XPU-RT tables executed on the board
    xpu_by_tag = {}
    for man in sorted(glob.glob(os.path.join(RES, "xpurt_long", "manifest_*.json"))):
        m = json.load(open(man)); tag = m.get("tag", os.path.basename(man)[9:-5]); label = re.sub(r"_(other|fifo|rr)_run\d+$", "", tag)
        desc = describe_xpu(label); base = re.sub(r"r\d+$", "", label); cfg = f"xpurt_{base}"; configs[cfg] = desc
        trace = man.replace("manifest_", "trace_").replace(".json", ".csv")
        try:
            met = trace_metrics(trace) if os.path.exists(trace) else {}
        except Exception as e:
            met = {"note": f"trace not parsed: {type(e).__name__}"}
        stag, solver, rnd, rep = xpu_label_parts(label)
        sched = os.path.relpath(m.get("schedule", ""), REPO) if m.get("schedule") else ""
        fam = "hil_feedback_study" if stag and stag.startswith("fb") else ("solver_rate_sweep" if stag else "xpurt_hand_placements")
        row = {"run_id": f"board_xpurt:{tag}", "kind": "board_xpurt", "config": cfg, "config_description": desc, "source": os.path.relpath(trace, REPO), "family": fam,
               "cell": f"xpurt:{stag or base}" + (f":round{rnd}" if rnd is not None else ""), "replicate": rep or "", "tag": tag, "schedule_executed": sched,
               "schedule_sha256": m.get("schedule_sha256", ""), "sched_policy": m.get("sched_policy", ""), "exit_code": m.get("exit_code", ""), "verify": str(m.get("verify_lines", ""))[:120],
               **schedule_info(sched), **met}
        if rnd is not None and rnd > 0:
            row["related_runs"] = ";".join(f"board_xpurt:{stag}r{rnd - 1}{solver}r{k}_other_run1" for k in (1, 2, 3))
        rows.append(row); xpu_by_tag[tag] = row
    # counterpart on the board for the same spec (the other solver's runs)
    for row in rows:
        if row["kind"] == "board_xpurt" and row.get("sched_tag"):
            other = "greedy" if row["sched_solver_kind"] == "cpsat_hard" else "cpsat_hard"
            pre = row["sched_tag"] + (f"r{row['sched_feedback_round']}" if row.get("sched_feedback_round") != "" else "")
            row["counterpart_board_runs"] = ";".join(t for t in xpu_by_tag if t.startswith(pre + other))
    # ---- 3. solver certificates and tables
    for log in sorted(glob.glob(os.path.join(RES, "solver_v2", "*.log"))):
        base = os.path.basename(log)[:-4]
        if not base.startswith("wh_"):
            continue
        txt = open(log, errors="replace").read()
        st = re.findall(r"cpsat status=(\w+)", txt); mis = re.findall(r"periodic window misses=(\d+)", txt); mk = re.findall(r"makespan=([\d.]+) ms", txt); wall = re.findall(r"wall=([\d.]+)s", txt)
        solver = "cpsat" if "cpsat" in base else "greedy"; spec = re.sub(r"_(cpsat_hard|cpsat|greedy|cpsat_3600)$", "", base)
        cfg = f"solve_{base}"; configs[cfg] = f"solver run: {solver} on {spec}" + (" (200 ms hyperperiod certificate)" if "h200" in spec else "")
        sp = os.path.join(REPO, "data", "toplevel", spec + ".json"); extra = {}
        if os.path.exists(sp):
            s = json.load(open(sp)); extra = {"sched_solver_config": json.dumps(s.get("scheduler", {})), "sched_horizon_ms": s.get("horizon_ms", ""),
                                             "sched_spec_networks": json.dumps({n: {"period_ms": round(float(v.get("period", 0)), 3), "window_ms": round(float(v.get("window_duration", 0)), 3), "instances": v.get("num_instances")} for n, v in s["networks"].items()})}
        hz = re.search(r"chain(\d+)", spec)
        rows.append({"run_id": f"solve:{base}", "kind": "solve", "config": cfg, "config_description": configs[cfg], "source": os.path.relpath(log, REPO), "family": "solver_certificates",
                     "cell": f"solve:{spec}", "sched_spec": spec, "sched_solver_kind": solver, "camera_hz": hz.group(1) if hz else "", "cpsat_status": st[-1] if st else "",
                     "sched_predicted_deadline_misses": mis[-1] if mis else "", "sched_predicted_makespan_ms": mk[-1] if mk else "", "sched_solver_wall_s": wall[-1] if wall else "", **extra})
    # ---- 4. flights
    records = {}
    for npz in glob.glob(os.path.join(RES, "*", "records", "*", "*.npz")):
        d = np.load(npz, allow_pickle=True); tagdir = os.path.basename(os.path.dirname(npz))
        lat = re.search(r"_lat([\d.]+)", tagdir); hold = re.search(r"_h([\d.]+)_", tagdir)
        key = (os.path.basename(str(d["ctrl_trace"])), round(float(d["cruise_speed"]), 3), round(float(d["moment_scale"]), 6), round(float(d["prop_density"]), 3), str(d["course"]),
               round(float(d["walk_speed"]), 2), float(lat.group(1)) if lat else 0.0, float(hold.group(1)) if hold else 0.0, int(d["seed"]))
        records[key] = npz
    headers = {os.path.basename(p): trace_header(p) for p in glob.glob(os.path.join(RES, "ctrl_traces", "*.csv"))}
    flights = []
    for rel, (fam, cdesc, weights) in CAMPAIGNS.items():
        p = os.path.join(RES, rel)
        if not os.path.exists(p):
            continue
        mtime = os.path.getmtime(p)
        for r in flight_rows(p):
            tr = os.path.basename(r.get("ctrl_trace", "") or "")
            lat = float(r.get("percep_latency_ms", 0) or 0); hold = float(r.get("percep_hold_ms", 0) or 0)
            if tr:
                cfg = f"flight_{tr[:-4]}"; d = FLIGHT_TRACES.get(tr, f"replayed cadence trace {tr}")
            else:
                cfg = f"flight_lat{r.get('sched_latency_ms', '')}"; d = f"fixed control latency {r.get('sched_latency_ms', '')} ms (ZOH hold of ceil(latency/10 ms) steps)"
            if lat:
                cfg += f"_lat{lat:g}"; d += f"; navigation decision delayed by the measured {lat:g} ms camera-to-control latency"
            if hold:
                cfg += f"_hold{hold:g}"; d += f"; navigation goal refreshed every {hold:g} ms (measured goal rate)"
            configs[cfg] = d
            course = r.get("course", "a") or "a"; dens = float(r.get("prop_density", 0.3) or 0.3); walk = float(r.get("walk_speed", 0) or 0); gain = float(r["moment_scale"]); cru = float(r["cruise_speed"])
            person_h = r.get("person_h") or ("2.4" if mtime > PERSON_H_CHANGE_EPOCH and fam not in ("showdown_v2", "showdown_v2_courseB", "showdown_v1", "envelope_fixed_gain", "envelope_calibrated_gain", "envelope_20hz") else "1.7")
            key = (tr, round(cru, 3), round(gain, 6), round(dens, 3), course, round(walk, 2), lat, hold, int(r["seed"]))
            cell = f"{fam}:{course}:d{dens:.2f}:w{walk:g}:c{cru:g}:g{gain:g}:lat{lat:g}:h{hold:g}:seed{r['seed']}"
            steps = int(float(r.get("steps", 0) or 0)); cdt = float(r.get("control_dt_ms", CONTROL_DT_MS) or CONTROL_DT_MS)
            h = headers.get(tr, {})
            src = h.get("source", "")
            row = {"run_id": f"flight:{rel}:{cfg}:{course}:d{dens:.2f}:w{walk:g}:c{cru:g}:g{gain:g}:s{r['seed']}", "kind": "flight", "config": cfg, "config_description": d, "source": rel,
                   "family": fam, "campaign": cdesc, "cell": cell, "nav_weights": weights, "yolo_weights": "yolov8n_gate_person_64x96.pt (perception on the board only; the flight's guidance is the nav net)",
                   "course": course, "gates_xy": json.dumps(course_gates(course)), "obstacle_level": 8, "person_h": person_h, "walk_speed_effective": walk if walk > 0 else 0.8,
                   "flight_time_s": round(steps * cdt / 1000.0, 2), "board_run_of_trace": src, "trace_gap_mean_ms": h.get("gap_mean_ms", ""), "trace_gap_p95_ms": h.get("gap_p95_ms", ""),
                   "trace_gap_max_ms": h.get("gap_max_ms", ""), "trace_eff_hz": h.get("eff_hz", ""), "trace_outputs": h.get("outputs", ""), "trace_span_ms": h.get("span_ms", ""),
                   **{k: v for k, v in r.items()}}
            # link the deployment behind the trace: the XPU-RT table on the board, or the ROS layout run
            if src:
                mm = re.search(r"xpurt_long/trace_(.+)\.csv", src)
                if mm and mm.group(1) in xpu_by_tag:
                    b = xpu_by_tag[mm.group(1)]; row["related_runs"] = b["run_id"]
                    row.update({k: b[k] for k in b if k.startswith(("sched_", "board_", "schedule_")) and k not in row})
                    row["deployment"] = "XPU-RT"; row["scheduler"] = b.get("sched_solver_kind", "")
                mm = re.search(r"ros_traced/([^/]+)/", src)
                if mm and mm.group(1) in ros_by_tag:
                    b = ros_by_tag[mm.group(1)]; row["related_runs"] = b["run_id"]; row["deployment"] = "ROS 2"; row["scheduler"] = "none (OS scheduler; ROS 2 executor)"
                    row.update({k: b[k] for k in b if k.startswith("ros_") or k in ("gap_mean_ms", "gap_p95_ms", "gap_max_ms", "e2e_goal_med_ms", "e2e_goal_p95_ms", "n_goals") or k.startswith("busy_c")})
            flights.append((key, row))
    # comparable run: the other main deployment in the same cell and seed (XPU-RT CP-SAT <-> ROS vanilla4 when present, else any other arm)
    by_cell = {}
    for key, row in flights:
        by_cell.setdefault((row["family"], row["cell"].rsplit(":seed", 1)[0], key[8]), []).append((key, row))
    for key, row in flights:
        grp = [(k, r) for k, r in by_cell[(row["family"], row["cell"].rsplit(":seed", 1)[0], key[8])] if r is not row]
        row["cell_runs"] = ";".join(r["run_id"] for _, r in grp)
        want = "ros_vanilla445" if row["config"].startswith("flight_xpu") else "xpu_a_cpsat_hard"
        pick = next((r for _, r in grp if want in r["config"]), None) or next((r for _, r in grp if (r["config"].startswith("flight_xpu") != row["config"].startswith("flight_xpu"))), None)
        if pick:
            row.update({"pair_run_id": pick["run_id"], "pair_config": pick["config"], "pair_outcome": pick.get("outcome", ""), "pair_gates_passed": pick.get("gates_passed", ""), "pair_flight_time_s": pick.get("flight_time_s", "")})
    for key, row in flights:
        npz = records.get(key)
        if npz:
            pair_t = None
            if row.get("pair_outcome") == "crash":
                pair_t = float(row.get("pair_flight_time_s") or 0) or None
            try:
                row.update(flight_record(npz, pair_t))
            except Exception as e:
                row["note"] = f"record not parsed: {type(e).__name__}"
        rows.append(row)
    # ---- 5. energy flights
    for ecsv in ("flight_energy.csv", "flight_energy_v2.csv"):
        p = os.path.join(RES, ecsv)
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                cfg = f"energy_{r.get('flight', '').rsplit('_s', 1)[0]}"; configs.setdefault(cfg, f"energy run {cfg[7:]}: commanded wrench logged, rotor-model power (arms as in the envelope story's panel)")
                rows.append({"run_id": f"energy:{ecsv}:{r.get('flight', '')}", "kind": "energy", "config": cfg, "config_description": configs[cfg], "source": ecsv, "family": "energy_" + ecsv[:-4], "cell": f"energy:{r.get('flight', '').rsplit('_s', 1)[-1]}", **r})
    cols = ["run_id", "kind", "config", "config_description", "source", "family", "cell", "cell_runs", "related_runs", "pair_run_id"]
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    cfg_out = a.out.replace("master_runs", "master_configs")
    counts = {}
    for r in rows:
        counts[r["config"]] = counts.get(r["config"], 0) + 1
    with open(cfg_out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "kind", "description", "n_runs"])
        kind_of = {r["config"]: r["kind"] for r in rows}
        for c in sorted(configs):
            w.writerow([c, kind_of.get(c, ""), configs[c], counts.get(c, 0)])
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {a.out}: {len(rows)} runs {kinds}, {len(cols)} columns; {cfg_out}: {len(configs)} configurations")
    return 0


if __name__ == "__main__":
    sys.exit(main())

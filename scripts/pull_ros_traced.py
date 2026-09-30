#!/usr/bin/env python3
"""Turn the traced ROS 2 runs into the per-run tables the figures and constants read.

For every run directory under results/codesign_feedback/ros_traced/<tag>/ (as pulled by
scripts/ros_traced_matrix.sh) this writes, next to the raw files:

  chain.csv     one row per released frame: when it was released, when its goal reached the
                control node, when a control fire first acted on it, and the two latencies
                (camera->goal, camera->control). Frames that never produced a goal (dropped by the
                depth-10 queue) or whose goal was superseded before a fire keep blank cells, so
                queue saturation is visible rather than averaged away.
  summary.json  the numbers a caption would quote, all after the warm-up window: control-gap
                mean/max/p95, chain medians, released/goal/consumed counts, per-core busy % over
                the run window.

and one table across all runs, results/codesign_feedback/ros_traced/summary.csv.

Three-process runs (p3/p8) hold their files in <tag>/{percep,nav,control}/; they are merged here:
the trace rows are concatenated and renumbered, the stamps joined on the shared --t0 origin.

Warm-up: the first WARMUP_MS of every run is discarded on the host, never on the board, and the
value used is written into summary.json.
"""
from __future__ import annotations
import csv, glob, json, os, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "results/codesign_feedback/ros_traced")
TICKS_PER_MS = 24000.0
WARMUP_MS = 3000.0


def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def merge_multiproc(d):
    """p3/p8: pull the sub-process files up into <tag>/ as if one process had written them."""
    parts = [p for p in ("camera", "percep", "percep2", "nav", "control", "extra", "extra2") if os.path.isdir(os.path.join(d, p))]
    if not parts:
        return
    tr = []
    for p in parts:
        tr += rows(os.path.join(d, p, "trace.csv"))
    tr.sort(key=lambda r: int(r["actual_start_cycles"]))
    with open(os.path.join(d, "trace.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry_id", "network", "instance", "dispatch_id", "op", "name", "core_kind", "hart",
                    "predicted_start_ms", "predicted_duration_ms", "worker_kind_idx", "worker_hart",
                    "actual_start_cycles", "actual_end_cycles"])
        for i, r in enumerate(tr):
            w.writerow([i] + [r[k] for k in ("network", "instance", "dispatch_id", "op", "name", "core_kind",
                                               "hart", "predicted_start_ms", "predicted_duration_ms",
                                               "worker_kind_idx", "worker_hart", "actual_start_cycles",
                                               "actual_end_cycles")])
    if os.path.isdir(os.path.join(d, "percep2")) and os.path.exists(os.path.join(d, "percep2", "released.csv")):
        # the second camera's releases join the first's
        with open(os.path.join(d, "percep", "released.csv"), "a") as f:
            f.writelines(list(open(os.path.join(d, "percep2", "released.csv")))[1:])
    rel_src = "camera" if os.path.isdir(os.path.join(d, "camera")) else "percep"
    for name, src in (("released.csv", rel_src), ("goals.csv", "control"), ("consumed.csv", "control"),
                      ("ctrl_gaps.csv", "control")):
        s = os.path.join(d, src, name)
        if os.path.exists(s):
            os.replace(s, os.path.join(d, name)) if not os.path.exists(os.path.join(d, name)) else None
    man = {}
    for p in parts:
        mp = os.path.join(d, p, "manifest.json")
        if os.path.exists(mp):
            m = json.load(open(mp)); man.setdefault("processes", {})[p] = m
            for k in ("tag", "rate_hz", "ctrl_hz", "seconds", "executor", "qos_depth", "rmw", "t0_rdtime",
                      "kernels_sha", "hostname", "kernel"):
                man.setdefault(k, m.get(k))
    man["nodes"] = " | ".join(parts) + f" ({len(parts)} processes" + (", unpinned" if "camera" in parts else "") + ")"
    man["n_released"] = man.get("processes", {}).get(rel_src, {}).get("n_released", 0)
    for k in ("n_goals", "n_ctrl_fires", "n_consumed", "wall_start_epoch_ms", "wall_end_epoch_ms",
              "run_t0_ticks", "run_end_ticks"):
        man[k] = man.get("processes", {}).get("control", {}).get(k)
    man["affinity_mask"] = {p: man["processes"][p].get("affinity_mask") for p in man.get("processes", {})}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=2)


def build_chain(d):
    rel = {int(r["frame_seq"]): r for r in rows(os.path.join(d, "released.csv"))}
    goal = {int(r["frame_seq"]): int(r["t_goal_ticks"]) for r in rows(os.path.join(d, "goals.csv"))}
    cons = {int(r["frame_seq"]): int(r["t_ctrl_ticks"]) for r in rows(os.path.join(d, "consumed.csv"))}
    out = []
    for seq in sorted(rel):
        r = rel[seq]; tr = int(r["t_release_ticks"]); tn = int(r["t_nominal_ticks"])
        tg = goal.get(seq); tc = cons.get(seq)
        out.append({"frame_seq": seq, "t_nominal_ticks": tn, "t_release_ticks": tr,
                    "t_goal_ticks": tg if tg is not None else "",
                    "t_ctrl_ticks": tc if tc is not None else "",
                    "e2e_goal_ms": f"{(tg - tr) / TICKS_PER_MS:.3f}" if tg is not None else "",
                    "e2e_ms": f"{(tc - tr) / TICKS_PER_MS:.3f}" if tc is not None else ""})
    with open(os.path.join(d, "chain.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()) if out else ["frame_seq"])
        w.writeheader(); w.writerows(out)
    return out


def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")


def summarise(d):
    man = json.load(open(os.path.join(d, "manifest.json")))
    chain = build_chain(d)
    t_run0 = int(man.get("run_t0_ticks") or 0)
    warm = t_run0 + WARMUP_MS * TICKS_PER_MS
    gaps = [float(r["gap_ms"]) for r in rows(os.path.join(d, "ctrl_gaps.csv")) if int(r["t_fire_ticks"]) >= warm]
    e2e_goal = [float(r["e2e_goal_ms"]) for r in chain if r["e2e_goal_ms"] != "" and int(r["t_release_ticks"]) >= warm]
    e2e = [float(r["e2e_ms"]) for r in chain if r["e2e_ms"] != "" and int(r["t_release_ticks"]) >= warm]
    n_rel = sum(1 for r in chain if int(r["t_release_ticks"]) >= warm)
    busy = {}
    cpu = rows(os.path.join(d, "cpu.csv"))
    if cpu and man.get("wall_start_epoch_ms"):
        lo = int(man["wall_start_epoch_ms"]) + WARMUP_MS; hi = int(man["wall_end_epoch_ms"])
        per = {}
        for r in cpu:
            e = int(r["epoch_ms"])
            if lo <= e <= hi:
                per.setdefault(int(r["cpu"]), []).append(float(r["busy_pct"]))
        busy = {c: round(statistics.mean(v), 1) for c, v in sorted(per.items())}
    s = {"tag": man.get("tag"), "rate_hz": man.get("rate_hz"), "executor": man.get("executor"),
         "nodes": man.get("nodes"), "affinity_mask": man.get("affinity_mask"), "sched_policy": man.get("sched_policy"),
         "yolo_pool": man.get("yolo_pool"), "warmup_ms": WARMUP_MS,
         "n_gaps": len(gaps), "gap_mean_ms": statistics.mean(gaps) if gaps else None,
         "gap_p95_ms": pct(gaps, 0.95) if gaps else None, "gap_max_ms": max(gaps) if gaps else None,
         "n_frames": n_rel, "n_goals": len(e2e_goal), "n_consumed": len(e2e),
         "e2e_goal_med_ms": statistics.median(e2e_goal) if e2e_goal else None,
         "e2e_goal_p95_ms": pct(e2e_goal, 0.95) if e2e_goal else None,
         "e2e_med_ms": statistics.median(e2e) if e2e else None,
         "e2e_p95_ms": pct(e2e, 0.95) if e2e else None,
         "busy_pct": busy, "rmw": man.get("rmw"), "kernels_sha": man.get("kernels_sha")}
    json.dump(s, open(os.path.join(d, "summary.json"), "w"), indent=2)
    return s


def main():
    tags = sys.argv[1:] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(p))
    out = []
    for t in tags:
        d = os.path.join(ROOT, t)
        if not os.path.isdir(d) or not (os.path.exists(os.path.join(d, "manifest.json")) or os.path.isdir(os.path.join(d, "control"))):
            continue
        merge_multiproc(d)
        s = summarise(d); out.append(s)
        b = " ".join(f"c{c}:{v:.0f}" for c, v in s["busy_pct"].items())
        print(f"{s['tag']:<18} {s['executor']:<6} rate {s['rate_hz']:>4}  gaps n={s['n_gaps']:<5} "
              f"mean {s['gap_mean_ms'] or 0:6.2f} max {s['gap_max_ms'] or 0:6.2f}   "
              f"chain goal-med {s['e2e_goal_med_ms'] or 0:7.2f}  ctrl-med {s['e2e_med_ms'] or 0:7.2f}  "
              f"frames {s['n_frames']} goals {s['n_goals']}   busy {b}")
    if out:
        # one row per run across every pull: rows for tags not processed this time are kept
        keys = [k for k in out[0] if k != "busy_pct"] + [f"busy_c{c}" for c in range(8)]
        path = os.path.join(ROOT, "summary.csv")
        kept = {r["tag"]: r for r in csv.DictReader(open(path))} if os.path.exists(path) else {}
        for s in out:
            row = {k: s.get(k) for k in keys if not k.startswith("busy_c")}
            row.update({f"busy_c{c}": s["busy_pct"].get(c) for c in range(8)})
            kept[s["tag"]] = row
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
            for tag in sorted(kept):
                w.writerow({k: kept[tag].get(k) for k in keys})
    return 0


if __name__ == "__main__":
    sys.exit(main())

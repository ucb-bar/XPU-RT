#!/usr/bin/env python3
"""The ROS 2 baseline, one entry point: every arm selectable by name, laid out as an effort ladder.

The baseline exists in many deployment configurations ("arms"). This tool does not define any of
them. It reads them from the places that already say what they are:

  * data/ros_arms.json -- what `scripts/ros_traced_matrix.sh` launches for each arm (processes,
    taskset, node flags) and what the runs' manifests recorded; generated and drift-checked by
    `scripts/gen_ros_arms.py`;
  * `ros_effort_ladder.LADDER` -- the rung order and labels the ladder figure uses;
  * `measured_timing.derive()` -- the board's control rate and camera->goal latency per arm and
    camera rate, pooled over replicates the way docs/Baselines/ros_arm_ranking.md §0 pools them;
  * `ros_ladder_paired.compare()` -- each rung against XPU-RT over the same flights;
  * `figure_constants` -- the cadence traces each arm is replayed from and the latency it is
    replayed with; the `refined/*_metrics.json` sidecars -- which figures draw each trace;
  * `ros_pinning_model` (Tier A) and `ros_pinning_profiled` (Tier B) -- the two models.

    scripts/ros_baseline.py ladder [--camera-hz 45] [--json out.json]
    scripts/ros_baseline.py show <arm>
    scripts/ros_baseline.py deploy <arm> --camera-hz N [--reps K] [--first-rep R]
    scripts/ros_baseline.py model <arm> --tier A|B --camera-hz N [--force]
    scripts/ros_baseline.py list

`deploy` prints commands; it never runs them (they need the K1 board and, for the flights, a GPU).
See docs/Baselines/ros_baseline_ladder.md.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import gen_ros_arms as GEN   # noqa: E402

RES = os.path.join(REPO, "results", "codesign_feedback")
REL_RES = "results/codesign_feedback"

# Rungs the ladder figure does not draw but the baseline has, each placed after an existing rung.
# The rung order and labels otherwise come from ros_effort_ladder.LADDER; what a rung CHANGES and
# what that COSTS are derived from the configurations, never written here.
EXTRA_RUNGS = [
    # (arm, insert after, label)
    ("vanilla_c50", "vanilla", "out of the box, launched with --ctrl-hz 50 (the paper figure's configuration)"),
    ("cp3", "vanilla4tm", "hand-pinned: 3 processes over 6 cores, control chained"),
    ("vanilla4x2ns4c", "vanilla4x2tm", "two instances + a 4-way nav pool, pool left to the OS"),
    ("vanilla4x2ns4a", "vanilla4x2ns4c", "two instances + a 4-way nav pool pinned to harts 0-3"),
    ("vanilla4x2ns4b", "vanilla4x2ns4a", "two instances + a 4-way nav pool pinned to harts 4-7"),
    ("cp3n4", "vanilla4x2ns4b", "hand-pinned, nav and control sharing the E cluster, control chained"),
    ("cp3n4_d", "cp3n4", "hand-pinned, nav sharded 4 ways over the E cluster (--nav-pool 4)"),
]
COMPUTE_NODES = ("perception", "perception2", "nav", "control")


# ------------------------------------------------------------------------------------------------
# arm table
# ------------------------------------------------------------------------------------------------
def arms() -> dict:
    return GEN.load()["arms"]


def resolve(name: str) -> dict:
    a = arms()
    if name not in a:
        close = [k for k in a if k.startswith(name) or name.startswith(k)]
        raise SystemExit(f"unknown arm {name!r}" + (f"; did you mean {', '.join(sorted(close)[:8])}?"
                                                     if close else "; `ros_baseline.py list` shows them all"))
    return a[name]


def node_view(e: dict) -> dict:
    """{node: its placement and knobs}, the unit two configurations are compared in."""
    out = {}
    for p in e.get("processes") or []:
        eff = p["effective"]
        nodes = [n for n in str(eff["nodes"]).split(",") if n and n != "none"]
        if eff.get("extra"):
            nodes += [f"extra:{x}" for x in eff["extra"].split(",")]
        for n in nodes:
            v = {"process": p["name"], "shares_process_with": ",".join(sorted(set(nodes) - {n})) or "-",
                 "cpus": eff.get("taskset") or "any", "main_thread_hart": eff.get("pin_main"),
                 "executor": eff["executor"], "qos_depth": eff["qos_depth"], "binary": eff["binary"]}
            if n.startswith("perception"):
                v.update(yolo_pool=eff["yolo_pool"], pool_harts=eff["pool_harts"] or "-")
            if n == "nav":
                v.update(nav_pool=eff["nav_pool"], nav_harts=eff["nav_harts"] or "-")
            if n == "control":
                v.update(ctrl_mode=eff["ctrl_mode"], ctrl_hz=eff["ctrl_hz"], ctrl_pool=eff["ctrl_pool"])
            if n.startswith("camera"):
                v.update(alternate=eff["alternate"])
            out[n] = v
    return out


# What each kind of change takes. A changed field maps to a category; the category says what an
# engineer has to do to make that change. Ordered by effort, cheapest first.
CATEGORY = {
    "qos_depth": "qos", "ctrl_mode": "ctrl_mode", "ctrl_hz": "ctrl_hz", "executor": "executor",
    "yolo_pool": "pool", "pool_harts": "pool", "ctrl_pool": "ctrl_pool",
    "cpus": "pinning", "main_thread_hart": "pinning", "nav_harts": "pinning",
    "shares_process_with": "layout", "-node": "layout",
    "+node": "instance", "alternate": "instance",
    "nav_pool": "nav_pool", "binary": "binary",
}
EFFORT = {   # category -> (rank, what it takes)
    "qos": (1, "one QoS setting (--qos-depth on every node)"),
    "ctrl_mode": (1, "one flag on the control node (--ctrl-mode)"),
    "ctrl_hz": (1, "one flag (--ctrl-hz)"),
    "executor": (1, "one executor choice (--executor)"),
    "ctrl_pool": (2, "one flag (--ctrl-pool/--ctrl-harts)"),
    "pool": (2, "one flag (--yolo-pool/--pool-harts) and the pool-enabled node build (MB_WITH_POOL) "
                "linked with the 4-wide YOLO kernels"),
    "binary": (2, "a differently linked node binary"),
    "layout": (3, "a launch-layout change (which nodes share a process)"),
    "pinning": (3, "expert per-network core choice (a taskset / pinned hart per process)"),
    "nav_pool": (4, "a nav build sharded at codegen (BINSUF=_nav4) and one flag (--nav-pool)"),
    "instance": (4, "a second model instance: another perception process, the camera alternating "
                    "frames (--alternate)"),
}


def diff(parent: dict, child: dict) -> list[tuple]:
    """(node, field, old, new) for every per-node difference between two configurations."""
    a, b = node_view(parent), node_view(child)
    out = []
    for n in sorted(set(a) | set(b)):
        if n not in a:
            out.append((n, "+node", None, b[n]["process"]))
            continue
        if n not in b:
            out.append((n, "-node", a[n]["process"], None))
            continue
        for k in sorted(set(a[n]) | set(b[n])):
            if k == "process":
                continue          # a rename of the out-dir is not a change; shares_process_with is
            if str(a[n].get(k)) != str(b[n].get(k)):
                out.append((n, k, a[n].get(k), b[n].get(k)))
    return out


def describe_diff(d: list[tuple], child: dict) -> tuple[str, str]:
    """(what changes, what it takes) as text, from a diff."""
    if not d:
        return "no change in any launched flag", "nothing"
    by = collections.defaultdict(list)
    for n, k, o, v in d:
        by[(k, str(o), str(v))].append(n)
    parts = []
    ctl = node_view(child).get("control", {})
    for (k, o, v), ns in sorted(by.items()):
        who = ",".join(ns) if len(ns) < 4 else "every node"
        if k == "+node":
            parts.append(f"+{who} ({v} process)")
        elif k == "-node":
            parts.append(f"-{who}")
        else:
            s = f"{who}: {k} {o}->{v}"
            if k == "ctrl_hz" and ctl.get("ctrl_mode") == "chained":
                s += " (inert: a chained control node runs no timer)"
            parts.append(s)
    cats = {CATEGORY.get(k, k) for _, k, _, _ in d}
    if cats & {"pool", "nav_pool"}:
        cats.discard("binary")          # the relink is part of adopting the pool build
    kinds = sorted(EFFORT.get(c, (2, c)) for c in cats)
    return "; ".join(parts), " + ".join(t for _, t in kinds)


def defaults_text(e: dict) -> str:
    nv = node_view(e)
    procs = {v["process"] for v in nv.values()}
    one_per = all(v["shares_process_with"] == "-" for v in nv.values())
    ctl = nv.get("control", {})
    pools = [v.get("yolo_pool") for n, v in nv.items() if n.startswith("perception")]
    return (f"{len(procs)} processes{', one node each' if one_per else ''}; executor "
            f"{','.join(sorted({v['executor'] for v in nv.values()}))}; keep-last-"
            f"{','.join(sorted({str(v['qos_depth']) for v in nv.values()}))}; "
            f"{'unpinned' if all(v['cpus'] == 'any' for v in nv.values()) else 'pinned'}; control "
            f"{ctl.get('ctrl_mode')}; YOLO {'on the callback thread' if not any(pools) else f'pool {pools}'}")


def ladder_rungs() -> list[dict]:
    """The rungs in order: LADDER's arms (deduplicated over camera rate) with EXTRA_RUNGS inserted."""
    import ros_effort_ladder as LAD
    order, label = [], {}
    for key, lab, _trace in LAD.LADDER:
        arm = key.split("@")[0]
        if arm not in label:
            order.append(arm); label[arm] = lab
    for arm, after, lab in EXTRA_RUNGS:
        order.insert(order.index(after) + 1, arm); label[arm] = lab
    A = arms()
    rungs = []
    for i, arm in enumerate(order):
        e = A[arm]
        if i == 0:
            parent, d = None, []
            change, takes = "none: " + defaults_text(e), "nothing (ROS 2 defaults)"
        else:
            # the earlier rung this one differs from least: the change it makes is then one step
            cands = [(len(diff(A[p], e)), -j, p) for j, p in enumerate(order[:i])]
            _, _, parent = min(cands)
            d = diff(A[parent], e)
            change, takes = describe_diff(d, e)
            if not d and A[parent].get("manifest_mismatch"):
                change = (f"the same launch as {parent} as the script stands today; {parent}'s recorded "
                          f"runs predate part of that launch (its note), this arm's runs do not")
        rungs.append({"rung": i, "arm": arm, "label": label[arm], "parent": parent,
                      "changes": change, "takes": takes,
                      "from_ladder_figure": arm not in {x[0] for x in EXTRA_RUNGS}})
    return rungs


# ------------------------------------------------------------------------------------------------
# measurements
# ------------------------------------------------------------------------------------------------
_MEAS = None


def measured() -> dict:
    """{arm@hz: pooled record} from measured_timing.derive(), which re-derives from summary.csv."""
    global _MEAS
    if _MEAS is None:
        import measured_timing as MT
        cwd = os.getcwd()
        try:
            _MEAS = MT.derive()["ros_arms"]
        finally:
            os.chdir(cwd)
    return _MEAS


def at_rate(arm: str, hz: int) -> dict | None:
    r = measured().get(f"{arm}@{hz}")
    if not r or not r.get("gap_mean_pooled"):
        return None
    return {"ctrl_hz": 1000.0 / r["gap_mean_pooled"], "camera_to_goal_ms": r["e2e_goal_med_pooled"],
            "worst_gap_ms": r["gap_max_pooled"], "drop_frac": r["drop_frac"], "reps": len(r["reps"])}


def measured_rates(arm: str) -> list[int]:
    return sorted(int(k.split("@")[1]) for k, v in measured().items()
                  if k.split("@")[0] == arm and v.get("gap_mean_pooled"))


def paired() -> dict:
    import ros_ladder_paired as PAIR
    rows, _ = PAIR.compare()
    return {r["arm"]: r for r in rows}


def trace_sources() -> dict:
    """{ctrl trace basename: the ros_traced tag it was cut from}, from each trace's own header."""
    out = {}
    for f in glob.glob(os.path.join(RES, "ctrl_traces", "ros_*.csv")):
        with open(f) as fh:
            m = re.search(r"ros_traced/([^/]+)/ctrl_gaps\.csv", fh.readline())
        if m:
            out[os.path.basename(f)] = m.group(1)
    return out


def replay_arms(cfg: str) -> list[dict]:
    """Every cadence trace replaying this arm: from the trace header, else the registry pointer."""
    import figure_constants as FC
    src = trace_sources()
    out = {}
    for t, tag in src.items():
        m = re.match(r"^(\d+)_(.+)_r\d+$", tag)
        if m and m.group(2) == cfg:
            out[t] = {"trace": t, "cut_from": tag, "camera_hz": int(m.group(1)), "on_disk": True}
    for a in FC._ARMS:
        if not a.trace.startswith("ros_"):
            continue
        ptr = a.derive
        who = None
        if ptr and ptr[0] == "ROS_VANILLA":
            who = ptr[1][0]
        elif ptr and ptr[0] == "ROS_SENSITIVITY":
            mm = re.match(r"^\d+_(.+)_r\d+$", ptr[1]); who = mm.group(1) if mm else None
        if a.trace in src:
            mm = re.match(r"^\d+_(.+)_r\d+$", src[a.trace]); who = mm.group(1) if mm else who
        if who != cfg:
            continue
        rec = out.setdefault(a.trace, {"trace": a.trace, "cut_from": src.get(a.trace), "camera_hz": a.cam_hz,
                                       "on_disk": a.trace in src})
        rec.setdefault("replayed_with", []).append({"label": a.label, "percep_latency_ms": a.csv_lat,
                                                    "percep_hold_ms": a.csv_hold})
    return sorted(out.values(), key=lambda r: (r["camera_hz"], r["trace"]))


def figures_drawing(traces: list[str]) -> dict:
    """{trace: [refined figure stems whose sidecar names it]}."""
    want = set(traces)
    hits = collections.defaultdict(set)

    def walk(o, stem):
        if isinstance(o, dict):
            for v in o.values():
                walk(v, stem)
        elif isinstance(o, list):
            for v in o:
                walk(v, stem)
        elif isinstance(o, str):
            for t in re.findall(r"(ros_[A-Za-z0-9_]+\.csv)", o):
                if t in want:
                    hits[t].add(stem)
    for f in sorted(glob.glob(os.path.join(RES, "refined", "*_metrics.json"))):
        try:
            walk(json.load(open(f)), os.path.basename(f)[:-len("_metrics.json")])
        except Exception:
            pass
    return {t: sorted(v) for t, v in hits.items()}


# ------------------------------------------------------------------------------------------------
# the models
# ------------------------------------------------------------------------------------------------
def model_shape(e: dict) -> tuple[list[str], list[str]]:
    """(reasons the arm is outside the pinning model, caveats that leave it inside)."""
    nv = node_view(e)
    out, cav = [], []
    if not nv:
        return ["no launch configuration is recoverable for this arm"], cav
    if "perception2" in nv or any(v.get("alternate") for v in nv.values()) or "camera2" in nv:
        out.append("two perception instances: the model has one perception node per chain")
    if any(n.startswith("extra:") for n in nv):
        out.append("the heavier stack's extra nodes: the model has no cost input for them")
    if nv.get("nav", {}).get("nav_pool"):
        out.append("a nav worker pool: the model runs nav serially on its partition")
    if any(v["executor"] != "single" for v in nv.values()):
        out.append("a multi-threaded executor: the model assumes one executor thread per node "
                   "(ros_pinning_model.py assumption 6)")
    ctl = nv.get("control", {})
    if "perception" in ctl.get("shares_process_with", "").split(","):
        out.append("control shares a single-threaded executor with perception: the model gives each "
                   "node its own thread, so it cannot see the timer starve")
    if any(nv[n]["cpus"] == "any" and nv[n].get("main_thread_hart") in (None, -1)
           for n in COMPUTE_NODES if n in nv):
        out.append("unpinned processes: the model assumes each node owns a static partition "
                   "(assumption 3); pass --force to evaluate it as if pinned")
    if ctl.get("ctrl_pool"):
        cav.append("the control worker pool is not modelled (control costs < 0.1 ms)")
    if "control" in nv.get("nav", {}).get("shares_process_with", "").split(","):
        cav.append("nav and control share a process; the model charges nothing for it")
    return out, cav


def run_model(name: str, tier: str, hz: float, force: bool = False) -> int:
    e = resolve(name)
    out, cav = model_shape(e)
    hard = [r for r in out if not r.startswith("unpinned")]
    if hard or (out and not force):
        print(f"{name}: outside what the Tier {tier} model supports:")
        for r in out:
            print(f"  - {r}")
        return 2
    if out:
        cav = [r.replace("; pass --force to evaluate it as if pinned", "") + " -- EVALUATED ANYWAY (--force)"
               for r in out] + cav
    nv = node_view(e)
    pool = int(nv["perception"].get("yolo_pool") or 0)
    ctl = nv["control"]
    timer = ctl["ctrl_mode"] == "timer"
    meas = at_rate(name, int(hz)) if float(hz).is_integer() else None
    at = f" @ {ctl['ctrl_hz']:g} Hz" if timer else ""
    print(f"arm {name}, camera {hz:g} Hz, Tier {tier}  (YOLO pool {pool or 'none'}, control {ctl['ctrl_mode']}"
          f"{at}, keep-last-{nv['perception']['qos_depth']})")
    for c in cav:
        print(f"  caveat: {c}")
    if tier == "A":
        import ros_pinning_model as RPM
        nodes = {k: dict(v) for k, v in RPM.DEPLOYED_NODES.items()}
        for v in nodes.values():
            if v["role"] == "control":
                v["period_ms"] = 1000.0 / float(ctl["ctrl_hz"])
        width = pool or 1
        if width not in RPM.PERCEPTION_SPEEDUP:
            print(f"  outside the model: no measured speedup for a {width}-hart perception partition")
            return 2
        print("  inputs: XPU-RT's K1-calibrated per-dispatch durations (ros_pinning_model.DEPLOYED_NODES), "
              f"perception re-costed through the measured {width}-hart speedup; QoS depth is not modelled")
        for ex in ("release", "queued"):
            r = RPM.model(nodes, executor=ex, perception_width=width, camera_hz=hz, control_on_timer=timer)
            p = next(v for v in r["nodes"].values() if v["role"] == "perception")
            print(f"  executor {ex:8s} camera->control {r['chain_camera_to_control_ms']:8.2f} ms   "
                  f"commands {r['command_rate_hz']:6.1f} Hz (10 ms ZOH)   perception late "
                  f"{p['late_instances']}/{p['instances']}, sustains {r['perception_sustainable_camera_hz']:.1f} Hz")
        print("  ('queued' serialises instances on the partition, what a single-threaded executor does)")
        if timer:
            print("  (Tier A's command rate is the rate its chain sustains under the ZOH rule; a timer arm on "
                  "the board re-sends the held goal at its timer rate instead)")
    else:
        import ros_pinning_profiled as RPP
        print("  inputs: per-node callback costs from the board's own ROS 2 traces (ros_pinning_profiled."
              "pooled_inputs; ~25 s to read)")
        costs = RPP.pooled_inputs()
        if ("perception", pool) not in costs:
            print(f"  outside the model: no traced perception cost at pool width {pool}")
            return 2
        c = {"perception": costs[("perception", pool)]["med"], "nav": costs[("nav", 0)]["med"],
             "control": costs[("control", 0)]["med"], "camera": costs[("camera", 0)]["med"]}
        coloc = "camera" in nv["perception"]["shares_process_with"].split(",")
        p = RPP.predict(c, hz, ctl["ctrl_mode"], float(ctl["ctrl_hz"]), int(nv["perception"]["qos_depth"]), coloc)
        print(f"  costs (ms): perception {c['perception']:.3f}  nav {c['nav']:.3f}  control {c['control']:.3f}  "
              f"camera {c['camera']:.3f} (derived residual)")
        print(f"  camera->goal model {p['chain_model_ms']:.2f} ms, + queue term {p['chain_model_queue_ms']:.2f} ms"
              f"   control {p['ctrl_pred_hz']:.1f} Hz   saturated: {'yes' if p['saturated'] else 'no'}"
              + (f"   (the recurrence alone drifts {p['drift_ms']:.0f} ms over 17 s)" if p["drift_ms"] else ""))
        print("  the model charges nothing for the three topic hops; docs/Baselines/ros_baseline_tiers.md gives the "
              "residual (-1.74 ms mean over its unsaturated rows)")
    if meas:
        print(f"  measured (Tier C, {meas['reps']} run(s)): camera->goal {meas['camera_to_goal_ms']:.2f} ms, "
              f"control {meas['ctrl_hz']:.1f} Hz")
    else:
        rs = measured_rates(name)
        print(f"  measured (Tier C): not measured at {hz:g} Hz" + (f" (measured at {', '.join(map(str, rs))} Hz)"
                                                                     if rs else " (never measured)"))
    return 0


# ------------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------------
def cmd_ladder(hz: int, as_json: str | None) -> int:
    rungs = ladder_rungs()
    pr = paired()
    import ros_effort_ladder as LAD
    harts = LAD.hart_counts()
    rows = []
    print(f"ROS 2 baseline effort ladder at a {hz} Hz camera (board: ros_traced/summary.csv pooled over "
          f"replicates; flights: ros_ladder_paired.py)\n")
    for r in rungs:
        m = at_rate(r["arm"], hz)
        p = pr.get(f"{r['arm']}@{hz}")
        h = harts.get(f"{r['arm']}@{hz}")
        rec = dict(r, camera_hz=hz, measured=m, cores_busy=(sorted(h)[len(h) // 2] if h else None),
                   paired_vs_xpurt=({k: p[k] for k in ("flights_per_arm", "ros_completed", "xpu_completed",
                                                       "fisher_p_two_sided", "cells")} if p else None))
        rows.append(rec)
        busy = f", {rec['cores_busy']:.0f} cores busy" if rec["cores_busy"] is not None else ""
        meas = (f"control {m['ctrl_hz']:5.1f} Hz, camera->goal {m['camera_to_goal_ms']:6.1f} ms "
                f"({m['reps']} run{'s' if m['reps'] > 1 else ''}{busy})"
                if m else f"not measured at {hz} Hz"
                          + (f" (measured at {', '.join(map(str, measured_rates(r['arm'])))} Hz)"
                             if measured_rates(r["arm"]) else ""))
        fl = (f"ROS {p['ros_completed']}/{p['flights_per_arm']} vs XPU-RT {p['xpu_completed']}/{p['flights_per_arm']}"
              f" (p={p['fisher_p_two_sided']:.3f})" if p else "no paired flights at this rate")
        print(f"{r['rung']:2d}. {r['arm']:<16s} {r['label']}")
        print(f"      changes: {r['changes']}" + (f"   [vs {r['parent']}]" if r["parent"] else ""))
        print(f"      takes:   {r['takes']}")
        print(f"      board:   {meas}")
        print(f"      flights: {fl}")
        for mm in arms()[r["arm"]].get("manifest_mismatch") or []:
            print(f"      note:    today's launch differs from the recorded runs -- {mm}")
    if as_json:
        json.dump({"generated_by": "scripts/ros_baseline.py ladder", "camera_hz": hz, "rungs": rows},
                  open(as_json, "w"), indent=1, default=str)
        print(f"\nwrote {as_json}")
    return 0


def cmd_list() -> int:
    A = arms()
    lad = {r["arm"]: r["rung"] for r in ladder_rungs()}
    print(f"{'arm':<18s}{'rung':>5s}  {'base arm + suffix':<24s}{'rates measured (runs)':<40s}")
    for k, v in sorted(A.items()):
        rates = ", ".join(f"{h}({n})" for h, n in v["rates"].items()) or "never run"
        base = v["base_arm"] + (f" + {v['suffix']}" if v["suffix"] else "")
        print(f"{k:<18s}{str(lad.get(k, '')):>5s}  {base:<24s}{rates}")
    return 0


def cmd_show(name: str) -> int:
    e = resolve(name)
    print(f"# {name}\n")
    if e.get("script_note"):
        print(f"ros_traced_matrix.sh: {e['script_note']}\n")
    print(f"base arm `{e['base_arm']}`" + (f", suffix `{e['suffix']}`" if e["suffix"] else "")
          + (f"; env {' '.join(f'{k}={v}' for k, v in e['env'].items())}" if e.get("env") else ""))
    rung = next((r for r in ladder_rungs() if r["arm"] == name), None)
    if rung:
        print(f"ladder rung {rung['rung']}: {rung['label']}\n  changes: {rung['changes']}\n  takes: {rung['takes']}")
    print("\n## Configuration (what ros_traced_matrix.sh launches, per process)\n")
    if not e.get("processes"):
        print(f"  not recoverable: {e.get('why_no_command')}")
    else:
        print(f"  {e['matrix_command']}\n")
        for p in e["processes"]:
            print(f"  [{p['name']}] {p['command']}")
        print("\n  effective per node (binary defaults from ros_mb_chain_traced.cpp overlaid with the flags):")
        for n, v in node_view(e).items():
            print(f"    {n:<12s} " + "  ".join(f"{k}={v2}" for k, v2 in v.items()))
    if e.get("manifest_mismatch"):
        print("\n  the latest run's manifest disagrees with the script's expansion:")
        for m in e["manifest_mismatch"]:
            print(f"    - {m}")
    elif e.get("latest_run") and e.get("processes"):
        print(f"\n  matches the manifest of its latest run, {e['latest_run']}")
    print("\n## Tiers\n")
    out, cav = model_shape(e)
    for t in ("A", "B"):
        print(f"  Tier {t}: " + ("supported" if not out else
                                 ("evaluable with --force (" + "; ".join(out) + ")") if all(
                                     r.startswith("unpinned") for r in out) else "not supported: " + "; ".join(
                                     r.split("; pass --force")[0] for r in out)))
    rates = e["rates"]
    print(f"  Tier C: " + (f"{sum(rates.values())} board run(s) at " + ", ".join(f"{h} Hz x{n}" for h, n in rates.items())
                           if rates else "never measured on the board"))
    for h in rates:
        m = at_rate(name, int(h))
        tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join(RES, "ros_traced", f"{h}_{name}_r*")))
        print(f"    {h:>4s} Hz: " + (f"control {m['ctrl_hz']:.1f} Hz, camera->goal {m['camera_to_goal_ms']:.2f} ms, "
                                     f"worst gap {m['worst_gap_ms']:.1f} ms" if m else "no goals recorded")
              + f"   [{', '.join(tags)}]")
    print("\n## Replay into flights (ctrl_traces/ + figure_constants)\n")
    rep = replay_arms(name)
    if not rep:
        print("  no cadence trace is cut from this arm")
    figs = figures_drawing([r["trace"] for r in rep])
    for r in rep:
        print(f"  {r['trace']}: cut from {r['cut_from'] or '(not on disk)'}, camera {r['camera_hz']} Hz"
              + ("" if r["on_disk"] else "  [trace file not on disk]"))
        for w in r.get("replayed_with", []):
            print(f"      registry: \"{w['label']}\" latency {w['percep_latency_ms']} ms, hold {w['percep_hold_ms']} ms")
        if not r.get("replayed_with"):
            print("      not registered in figure_constants._ARMS")
        fs = figs.get(r["trace"], [])
        print(f"      drawn in {len(fs)} refined figure(s)" + (": " + ", ".join(fs) if fs else ""))
    page = f"artifact/implementations/ros_deployments/{name}/README.md"
    if os.path.exists(os.path.join(REPO, page)):
        print(f"\nFull generated page: {page}")
    return 0


def cmd_deploy(name: str, hz: int, reps: int, first: int | None) -> int:
    e = resolve(name)
    if not e.get("matrix_command"):
        print(f"{name}: no command can be derived -- {e.get('why_no_command')}.")
        sib = sorted(k for k, v in arms().items() if k.startswith(name) and k != name and v.get("matrix_command"))
        if sib:
            print(f"  sibling arms whose command form is recorded: {', '.join(sib)}")
        return 2
    have = [int(m.group(1)) for d in glob.glob(os.path.join(RES, "ros_traced", f"{hz}_{name}_r*"))
            if (m := re.search(r"_r(\d+)$", d))]
    first = first or (max(have) + 1 if have else 1)
    tags = [f"{hz}_{name}_r{k}" for k in range(first, first + reps)]
    cmd = e["matrix_command"].replace("{hz}", str(hz))
    print(f"# {name} @ {hz} Hz, {reps} replicate(s). PRINTED, NOT RUN: these need the K1 "
          f"(MODELBLASTER_K1_HOST, default k1).")
    if have:
        print(f"# replicates r{','.join(map(str, sorted(have)))} already exist at {hz} Hz; starting at r{first} "
              f"so none is overwritten (a successful pull replaces its tag's directory)")
    bins = sorted({p["binary"] for p in e["processes"]})
    print(f"\n# 0. once per board: build the node (docs/Baselines/ros_baseline_reproduction.md §1-2). Needs {', '.join(bins)}")
    print("scripts/board_deploy_ros_node.sh")
    print("\n# 1. measure (one invocation per replicate; the script archives, pulls and digest-checks each run)")
    for k in range(first, first + reps):
        print(f"{cmd} {k}")
    print("\n#    what each invocation launches on the board, per process:")
    for p in e["processes"]:
        print("#    " + p["command"].replace("{hz}", str(hz)).replace("{rep}", str(first)))
    print("\n# 2. roll the pulled runs up (chain.csv, summary.json, ros_traced/summary.csv)")
    print("scripts/pull_ros_traced.py " + " ".join(tags))
    print("scripts/measured_timing.py --verify")
    trace = f"ros_{name}{hz}.csv"
    print("\n# 3. cut the cadence the flights replay")
    print(f"scripts/ctrl_trace_from_board.py {REL_RES}/ros_traced/{tags[0]}/ctrl_gaps.csv "
          f"--out {REL_RES}/ctrl_traces/{trace} --warmup-ms 3000")
    reg = [w for r in replay_arms(name) if r["trace"] == trace for w in r.get("replayed_with", [])]
    if reg:
        lat, hold = reg[0]["percep_latency_ms"], reg[0]["percep_hold_ms"]
        print(f"#    {trace} is registered in scripts/figure_constants.py (latency {lat} ms, hold {hold} ms)")
    else:
        m = at_rate(name, hz)
        lat = round(m["camera_to_goal_ms"], 1) if m else "<camera->goal ms from step 2>"
        hold = "<ms>"
        print(f"#    {trace} is not registered: add a ReplayArm for it to scripts/figure_constants.py._ARMS "
              f"(csv_lat = the measured camera->goal, csv_hold = the goal hold), then re-run step 2's --verify")
    print("\n# 4. fly it (GPU; docs/Artifact/reproduction_full.md 'One flight' has every flag)")
    print(f". scripts/env.sh; cd \"$SIM_TREE\"; $ISAAC_PY sims/scripts/sweep_rate_demo.py --headless ... "
          f"--ctrl_trace <abs path>/{REL_RES}/ctrl_traces/{trace} --percep_latency_ms {lat} --percep_hold_ms {hold} "
          f"--sweep-csv <out>/campaign.csv")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ladder"); s.add_argument("--camera-hz", type=int, default=45); s.add_argument("--json")
    sub.add_parser("list")
    s = sub.add_parser("show"); s.add_argument("arm")
    s = sub.add_parser("deploy"); s.add_argument("arm"); s.add_argument("--camera-hz", type=int, required=True)
    s.add_argument("--reps", type=int, default=3); s.add_argument("--first-rep", type=int, default=None)
    s = sub.add_parser("model"); s.add_argument("arm"); s.add_argument("--tier", choices=["A", "B"], required=True)
    s.add_argument("--camera-hz", type=float, required=True); s.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "ladder":
        return cmd_ladder(a.camera_hz, a.json)
    if a.cmd == "list":
        return cmd_list()
    if a.cmd == "show":
        return cmd_show(a.arm)
    if a.cmd == "deploy":
        return cmd_deploy(a.arm, a.camera_hz, a.reps, a.first_rep)
    return run_model(a.arm, a.tier, a.camera_hz, a.force)


if __name__ == "__main__":
    raise SystemExit(main())

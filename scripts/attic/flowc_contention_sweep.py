#!/usr/bin/env python3
"""Multi-model contention sweep for QRB5165: does the feedback knob still pay
once the board is busy?

The single-network ladder (`flowc_feedback_stages.py`) measures a knob in
isolation. The interesting question is whether the same knob survives
contention: ViNT is the heavy model, and periodic yolov8n / dronet /
mlp_control compete for the same lanes.

Each point varies one axis, so a regression is attributable:

  --lanes      3 (HTA,DSP,CPU) or 4 (+GPU)         -- how much silicon
  --heavy      vint | vint_par                     -- the slicing knob
  --periodic   which contenders, at which periods  -- how much pressure

Instance counts are PINNED and periodic pruning disabled, because the horizon
is derived from the heavy model's completion: any improvement shrinks the
instance count, and compare_candidates then refuses the pair as "two amounts
of work". Pinning is what makes two points comparable at all.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

GRAPH = ("gen/qnn_vmfb/{m}/qrb5165_flowc/{b}/{m}.int8/{m}.int8_dispatch_graph.json")

LANES3 = ({"cpu_p": 1, "cpu_e": 1, "cpu_x": 1},
          {"cpu_p": "HTA", "cpu_e": "DSP", "cpu_x": "CPU"})
LANES4 = ({"cpu_p": 1, "cpu_e": 1, "cpu_x": 1, "cpu_g": 1},
          {"cpu_p": "HTA", "cpu_e": "DSP", "cpu_x": "CPU", "cpu_g": "GPU"})


PROFILE = ("gen/profile/{b}/qrb5165_flowc/{m}/{m}.int8/"
           "{m}_qrb5165_flowc_{b}_{m}.int8/topo_0/results.csv")


def coverage_gap(models: list[str], lanes: int) -> list[str]:
    """Which (model, lane) profile cells are missing for this point.

    A lane is only usable if EVERY model in the workload has a measured cell
    on it -- `load_profile` is strict, and rightly so: the alternative is a
    synthetic fallback silently standing in for a measurement. This is a hard
    constraint on the placement knob, not a bug: adding the GPU lane to a
    workload containing dronet is not possible until dronet is profiled on
    the GPU.
    """
    kinds = ["HTA", "DSP", "CPU"] + (["GPU"] if lanes == 4 else [])
    missing = []
    for m in models:
        for b in kinds:
            if not os.path.exists(os.path.join(REPO, PROFILE.format(m=m, b=b))):
                missing.append(f"{m}@{b}")
    return missing


def build_spec(heavy: str, periodic: dict, lanes: int, out: str) -> str:
    machines, profile_hw = LANES4 if lanes == 4 else LANES3
    nets = {heavy: {"id": 0, "identifier": heavy,
                    "dispatch_deps_path": GRAPH.format(m=heavy, b="CPU")}}
    for i, (name, (period, inst)) in enumerate(periodic.items(), start=1):
        nets[name] = {"id": i, "identifier": name,
                      "dispatch_deps_path": GRAPH.format(m=name, b="CPU"),
                      "period": period, "window_duration": period,
                      "num_instances": inst}
    spec = {
        "_comment": (f"Contention sweep point: heavy={heavy}, lanes={lanes}, "
                     f"periodic={ {k: v[0] for k, v in periodic.items()} }. "
                     "Instances pinned and prune_periodic off so points are "
                     "comparable across stages."),
        "hardware": {"machines": machines, "profile_hw": profile_hw,
                     "profile": {"target": "qrb5165_flowc", "topo_tag": "topo_0",
                                 "topo_tag_override": True, "gen_root": "gen"},
                     "p_core_speedup": 1.0},
        "scheduler": {"prune_periodic": False},
        "networks": nets,
    }
    with open(out, "w") as f:
        json.dump(spec, f, indent=1)
    return out


def run(spec: str, py: str) -> dict:
    base = os.path.basename(spec)[:-5]
    r = subprocess.run(
        [py, os.path.join(HERE, "run_xpurt_schedule.py"),
         "--networks-json", spec, "--solver", "greedy", "--profiled",
         "--max-periodic-iters", "1"],
        cwd=REPO, capture_output=True, text=True, timeout=1800)
    sched = os.path.join(REPO, "schedules", f"scheduled_{base}_greedy_profiled.json")
    if r.returncode != 0 or not os.path.exists(sched):
        return {"ok": False, "err": (r.stderr or r.stdout)[-300:]}
    return {"ok": True, "schedule": sched}


def metrics(sched_path: str, heavy: str, critical: list[str]) -> dict:
    sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
    import schedule_scoring, job_names  # noqa: E402
    with open(sched_path) as f:
        s = json.load(f)
    known = job_names.known_from_schedule(s)
    disp = s["dispatches"]
    hw = s["metadata"]["profile_hw"]
    iv, busy, heavy_end = [], {}, 0.0
    for v in disp.values():
        st, du = float(v["start_time"]), float(v["duration"])
        iv.append((st, st + du))
        lane = hw.get(v["hardware_target"].split("#")[0], "?")
        busy[lane] = busy.get(lane, 0.0) + du
        if job_names.model_of(v.get("job_name") or "", known) == heavy:
            heavy_end = max(heavy_end, st + du)
    iv.sort()
    span = max(e for _, e in iv) if iv else 0.0
    overlap, cur = 0.0, 0.0
    for st, e in iv:
        if st < cur:
            overlap += min(e, cur) - st
        cur = max(cur, e)
    return {
        "span_ms": round(span, 3),
        "heavy_max_ms": round(heavy_end, 3),
        "sum_busy_ms": round(sum(busy.values()), 3),
        "overlap_ms": round(overlap, 3),
        "concurrency": round(sum(busy.values()) / span, 3) if span else None,
        "lane_busy": {k: round(v, 2) for k, v in sorted(busy.items())},
        "instances": schedule_scoring.instances_per_model(s),
        "n_dispatches": len(disp),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-dir", default="/tmp/flowc_sweep")
    ap.add_argument("--json", default=None)
    ap.add_argument("--heavy", default="vint,vint_par")
    ap.add_argument("--lanes", default="3,4")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # contention ladder: none -> light -> heavy -> mixed
    pressure = {
        "none":  {},
        "mlp":   {"mlp_control": (2.0, 20)},
        "mlp+dr": {"mlp_control": (2.0, 20), "dronet": (10.0, 4)},
        "mlp+dr+yolo": {"mlp_control": (2.0, 20), "dronet": (10.0, 4),
                        "yolov8n": (50.0, 1)},
    }
    rows = []
    for heavy, lanes, (pname, per) in itertools.product(
            a.heavy.split(","), [int(x) for x in a.lanes.split(",")], pressure.items()):
        tag = f"sweep_{heavy}_L{lanes}_{pname.replace('+','_')}"
        gap = coverage_gap([heavy] + list(per), lanes)
        if gap:
            print(f"  {tag:38s} SKIP: no profile for {', '.join(gap)}")
            rows.append({"tag": tag, "heavy": heavy, "lanes": lanes,
                         "pressure": pname, "ok": False,
                         "blocked_by": "profile-coverage", "missing": gap})
            continue
        spec = build_spec(heavy, per, lanes,
                          os.path.join(REPO, "data", "toplevel", f"{tag}.json"))
        got = run(spec, a.python)
        if not got["ok"]:
            print(f"  {tag:38s} FAILED: {got['err'][:90]}")
            rows.append({"tag": tag, "heavy": heavy, "lanes": lanes,
                         "pressure": pname, "ok": False})
            continue
        m = metrics(got["schedule"], heavy, list(per))
        m.update({"tag": tag, "heavy": heavy, "lanes": lanes,
                  "pressure": pname, "ok": True})
        rows.append(m)
        print(f"  {tag:38s} span={m['span_ms']:8.2f}  heavy_max={m['heavy_max_ms']:8.2f}"
              f"  conc={m['concurrency']}  lanes={m['lane_busy']}")
        os.remove(spec)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

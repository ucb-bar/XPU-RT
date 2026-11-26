"""How much a schedule moves a network between cores, and how much core time it burns.

Schedule-level metrics, computed from the schedule JSON alone (the `dispatches`
dict: `job_name`, `id`, `dependencies`, `hardware_target` such as
"CPU_E#0+CPU_E#1", `start_time`, `duration`). One row per network INSTANCE
(`job_name`), plus totals:

  core_changes       number of consecutive dispatch pairs, in start order, whose
                     MASTER core differs. The master core is the first machine of
                     `hardware_target`: it is the core the runtime's worker for
                     that dispatch runs on, and the one every solver uses for the
                     transfer-time lookup.
  kind_changes       the same count, restricted to pairs whose master cores are of
                     different kinds (CPU_P -> CPU_E).
  distinct_cores     cores the instance touches at all, helpers included.
  distinct_master_cores  cores it uses as a master.
  composite          dispatches with width > 1.
  core_ms            sum over dispatches of width * duration.
  dep_core_changes   dependency edges (predecessor -> successor inside the
                     instance) whose two master cores differ. This is the count
                     the transfer matrix actually prices: a solver charges a
                     hand-off per dependency edge, not per pair adjacent in time.
  dep_kind_changes   those edges that also change kind.

Start order is (start_time, id), so zero-duration dispatches that share a start
time are ordered by dispatch id.

Zero-duration dispatches count like any other by default: the runtime still
dispatches them on the core the schedule names. `skip_zero_duration=True`
(`--skip-zero-duration`) leaves them out of every count, which separates the
hops a solver chose from the ones an unprofiled, zero-costed dispatch forces
(a dispatch the profile database prices at 0 on one machine kind and marks
unavailable on the other drags its chain to that kind and back).

CLI:  python xpu-rt/handoff_metrics.py <schedule.json> [--json] [--network REGEX]
                                       [--skip-zero-duration]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys


def master_core(hardware_target: str) -> str:
    return str(hardware_target).split("+")[0].strip()


def core_kind(core: str) -> str:
    return core.split("#")[0] if "#" in core else core


def target_cores(hardware_target: str) -> list[str]:
    return [c.strip() for c in str(hardware_target).split("+") if c.strip()]


def _order_key(item):
    name, d = item
    did = d.get("id")
    return (float(d.get("start_time", 0.0)),
            did if isinstance(did, (int, float)) else 0, str(name))


def instance_metrics(dispatches: dict,
                     skip_zero_duration: bool = False) -> dict[str, dict]:
    """Per network instance (`job_name`) metrics; see the module docstring."""
    by_job: dict[str, list] = {}
    for name, d in dispatches.items():
        if skip_zero_duration and float(d.get("duration", 0.0)) == 0.0:
            continue
        by_job.setdefault(str(d.get("job_name", "")), []).append((name, d))
    out = {}
    for job, items in by_job.items():
        items.sort(key=_order_key)
        masters = [master_core(d["hardware_target"]) for _n, d in items]
        widths = [len(target_cores(d["hardware_target"])) for _n, d in items]
        core_changes = sum(1 for a, b in zip(masters, masters[1:]) if a != b)
        kind_changes = sum(1 for a, b in zip(masters, masters[1:])
                           if core_kind(a) != core_kind(b))
        touched = set()
        for _n, d in items:
            touched.update(target_cores(d["hardware_target"]))
        master_of = {n: m for (n, _d), m in zip(items, masters)}
        dep_core = dep_kind = dep_edges = 0
        for (n, d), m in zip(items, masters):
            for dep in d.get("dependencies") or []:
                pm = master_of.get(dep)
                if pm is None:             # dependency outside this instance
                    continue
                dep_edges += 1
                if pm != m:
                    dep_core += 1
                    if core_kind(pm) != core_kind(m):
                        dep_kind += 1
        out[job] = {
            "dispatches": len(items),
            "core_changes": core_changes,
            "kind_changes": kind_changes,
            "distinct_cores": len(touched),
            "distinct_master_cores": len(set(masters)),
            "composite": sum(1 for w in widths if w > 1),
            "core_ms": float(sum(w * float(d.get("duration", 0.0))
                                 for w, (_n, d) in zip(widths, items))),
            "dep_edges": dep_edges,
            "dep_core_changes": dep_core,
            "dep_kind_changes": dep_kind,
        }
    return out


_SUMMED = ("dispatches", "core_changes", "kind_changes", "composite", "core_ms",
           "dep_edges", "dep_core_changes", "dep_kind_changes")


def schedule_metrics(schedule: dict, skip_zero_duration: bool = False) -> dict:
    """`{"instances": {job_name: {...}}, "totals": {...}}` for a schedule JSON
    (or for a bare `dispatches` dict)."""
    dispatches = schedule.get("dispatches", schedule)
    if skip_zero_duration:
        dispatches = {k: d for k, d in dispatches.items()
                      if float(d.get("duration", 0.0)) != 0.0}
    inst = instance_metrics(dispatches)
    totals = {k: sum(m[k] for m in inst.values()) for k in _SUMMED}
    totals["core_ms"] = float(totals["core_ms"])
    totals["instances"] = len(inst)
    cores = set()
    for d in dispatches.values():
        cores.update(target_cores(d["hardware_target"]))
    totals["distinct_cores"] = len(cores)
    return {"instances": inst, "totals": totals}


def median_over(metrics: dict, pattern: str, key: str = "core_changes"):
    """Median of one per-instance metric over the instances whose job name
    matches `pattern` (a regex, `re.search`). None when nothing matches."""
    rx = re.compile(pattern)
    vals = [m[key] for job, m in metrics["instances"].items() if rx.search(job)]
    return statistics.median(vals) if vals else None


def solution_dispatches(workload, t, alpha) -> dict:
    """The minimal `dispatches` dict for a solved (workload, t, alpha), so the
    solver can report the same metrics before a schedule file exists."""
    import numpy as np
    combos = workload.get_machine_combinations()
    ops = workload.operations
    names = [getattr(op, "operation_name", None) or f"op_{i}"
             for i, op in enumerate(ops)]
    name_of = {id(op): names[i] for i, op in enumerate(ops)}
    job_names = getattr(workload, "job_names", None) or []
    out = {}
    for i, op in enumerate(ops):
        c = int(np.argmax(alpha[i]))
        job_id = op.job_id if getattr(op, "job_id", None) is not None else 0
        job = (job_names[job_id] if isinstance(job_id, int) and 0 <= job_id < len(job_names)
               else f"Job {job_id}")
        did = getattr(op, "operation_id", None)
        out[names[i]] = {
            "id": did if did is not None else i,
            "job_name": job,
            "dependencies": [name_of[id(p)] for p in op.predecessors
                             if id(p) in name_of],
            "hardware_target": "+".join(combos[c]),
            "start_time": float(t[i]),
            "duration": float(op.get_duration_for_combination(
                c, combos, workload.machines)),
        }
    return out


def solution_metrics(workload, t, alpha) -> dict:
    return schedule_metrics({"dispatches": solution_dispatches(workload, t, alpha)})


def format_table(metrics: dict, pattern: str | None = None) -> str:
    cols = ("dispatches", "core_changes", "kind_changes", "distinct_cores",
            "distinct_master_cores", "composite", "core_ms",
            "dep_core_changes", "dep_kind_changes")
    head = ("disp", "core_chg", "kind_chg", "cores", "masters", "composite",
            "core_ms", "dep_core_chg", "dep_kind_chg")
    rx = re.compile(pattern) if pattern else None
    rows = [(job, m) for job, m in sorted(metrics["instances"].items())
            if rx is None or rx.search(job)]
    w = max([len("instance")] + [len(j) for j, _m in rows] + [len("TOTAL")])
    lines = [f"{'instance':<{w}}  " + "  ".join(f"{h:>12}" for h in head)]
    for job, m in rows:
        lines.append(f"{job:<{w}}  " + "  ".join(
            f"{m[c]:>12.3f}" if c == "core_ms" else f"{m[c]:>12}" for c in cols))
    tot = metrics["totals"]
    lines.append(f"{'TOTAL':<{w}}  " + "  ".join(
        f"{tot[c]:>12.3f}" if c == "core_ms"
        else f"{tot.get(c, ''):>12}" for c in cols))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("schedule", help="schedule JSON written by run_xpurt_schedule.py")
    ap.add_argument("--json", action="store_true", help="print the metrics as JSON")
    ap.add_argument("--network", default=None, metavar="REGEX",
                    help="only list instances whose job name matches; also print "
                         "the median over them")
    ap.add_argument("--skip-zero-duration", action="store_true",
                    help="leave zero-duration dispatches out of every count")
    args = ap.parse_args(argv)
    with open(args.schedule) as f:
        sched = json.load(f)
    m = schedule_metrics(sched, skip_zero_duration=args.skip_zero_duration)
    if args.json:
        print(json.dumps(m, indent=2, sort_keys=True))
        return 0
    print(format_table(m, args.network))
    if args.network:
        for key in ("core_changes", "kind_changes", "composite", "core_ms"):
            print(f"median {key} over /{args.network}/: "
                  f"{median_over(m, args.network, key)}")
    md = (sched.get("metadata") or {}) if isinstance(sched, dict) else {}
    for key in ("handoff", "core_time_price"):
        if md.get(key) is not None:
            print(f"metadata.{key}: {json.dumps(md[key], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

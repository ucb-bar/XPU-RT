#!/usr/bin/env python3
"""Re-cost a solved schedule under the measured board costs — "how the RUN differs from the GANTT".

Keeps the schedule's assignment (which hart, per-hart order) and precedence FIXED, replaces each
dispatch's isolated-profile duration with its board-faithful duration (× the per-op / per-dispatch
calibration multiplier), and re-times the whole thing as-soon-as-possible under those longer costs.
Deadlines that the optimistic Gantt met can now be missed — that is the runtime-feedback signal that
the loop re-schedules against. Emits a schedule JSON in the SAME format (+ _metrics.json) so it renders
as an ordinary evolution panel, with the newly-late dispatches ringed by the renderer's deadline check.

This is the software twin of the board measurement (docs/board_calibration_codesign.md); it is NOT a
board run. The multiplier lookup mirrors xpu-rt/profile_loader._board_calibration_mult exactly:
per-dispatch "net/dispatch_id" (exact) → per-op (extrapolated) → aggregate.
"""
import argparse, json, os, re
from collections import defaultdict


def _net_inst(job):
    m = re.match(r"^(.*?)(\d*)$", job)
    return (m.group(1), int(m.group(2) or 0)) if m else (job, 0)


def _op_of(module_name):
    # e.g. "mlp_control$dispatch_0_rvv_x60_linear_s8_M1xK16xN256" -> "linear_s8"
    m = re.search(r"_x60_(.+?)_[A-Z0-9]+x[A-Z0-9]", module_name or "")
    if m:
        return m.group(1)
    m = re.search(r"_x60_([a-z0-9]+_[a-z0-9]+)", module_name or "")
    return m.group(1) if m else None


def _disp_idx(module_name):
    m = re.search(r"dispatch_(\d+)", module_name or "")
    return int(m.group(1)) if m else None


def mult(cal, net, module_name, stats):
    pd = cal.get("per_dispatch_multiplier") or {}
    di = _disp_idx(module_name)
    if di is not None and f"{net}/{di}" in pd:
        stats["per_dispatch"] += 1
        return float(pd[f"{net}/{di}"])
    op = _op_of(module_name)
    po = cal.get("per_op_multiplier") or {}
    if op and op in po:
        stats["per_op"] += 1
        return float(po[op])
    stats["aggregate"] += 1
    return float(cal.get("aggregate_multiplier", 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--spec", required=True)
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--calibration", default=os.path.join(_repo, "results/codesign_feedback/k1_board_calibration.json"))
    ap.add_argument("--out", required=True, help="output schedule .json (metrics go to <out>_metrics.json)")
    a = ap.parse_args()

    sched = json.load(open(a.schedule))
    disp = sched["dispatches"]
    cal = json.load(open(a.calibration))
    nets = json.load(open(a.spec))["networks"]
    period = {n: float(v.get("period", 0) or 0) for n, v in nets.items()}
    window = {n: float(v.get("window_duration", 0) or 0) for n, v in nets.items()}

    stats = defaultdict(int); applied = []
    order = sorted(disp.values(), key=lambda d: float(d["start_time"]))
    hart_free = defaultdict(float)          # per-hart next-free time under board costs
    new_end = {}                            # dispatch id -> board end time
    miss = 0

    for d in order:
        net, inst = _net_inst(d["job_name"])
        m = mult(cal, net, d.get("module_name", ""), stats); applied.append(m)
        bdur = float(d["duration"]) * m
        harts = d["hardware_target"].split("+")
        release = inst * period.get(net, 0.0)
        dep_end = max((new_end.get(dep, 0.0) for dep in d.get("dependencies", [])), default=0.0)
        start = max(release, dep_end, max((hart_free[h] for h in harts), default=0.0))
        end = start + bdur
        for h in harts:
            hart_free[h] = end
        new_end[d["id"]] = end
        d["start_time"] = start; d["duration"] = bdur
        dl = inst * period.get(net, 0.0) + window.get(net, 0.0)
        late = window.get(net, 0.0) > 0 and end > dl + 1e-6
        d["deadline_miss"] = bool(late)
        d["deadline_overrun_us"] = max(0.0, (end - dl)) * 1000.0
        if late:
            miss += 1

    makespan = max(new_end.values(), default=0.0)
    json.dump(sched, open(a.out, "w"), indent=1)
    metrics = {"makespan_ms": makespan, "deadline_miss_count": miss,
               "total_lateness_ms": sum(max(0.0, new_end[d["id"]] -
                   (_net_inst(d["job_name"])[1] * period.get(_net_inst(d["job_name"])[0], 0.0)
                    + window.get(_net_inst(d["job_name"])[0], 0.0))) for d in order if d["deadline_miss"]),
               "board_recost": True,
               "multiplier_source_counts": dict(stats),
               "multiplier_min": min(applied), "multiplier_max": max(applied),
               "multiplier_mean": sum(applied) / len(applied)}
    json.dump(metrics, open(a.out.replace(".json", "_metrics.json"), "w"), indent=1)
    print(f"re-cost {os.path.basename(a.schedule)} -> makespan {makespan:.2f} ms, {miss} miss "
          f"| mult src={dict(stats)} range {min(applied):.2f}-{max(applied):.2f} mean {sum(applied)/len(applied):.2f}")


if __name__ == "__main__":
    main()

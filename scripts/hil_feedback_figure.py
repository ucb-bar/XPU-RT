#!/usr/bin/env python3
"""The runtime feedback loop with the board in it, drawn: one row per round of the study on one spec.
Left: the table as solved (predicted Gantt, eight harts, one window of the run); middle: the same
window as the K1 executed it (every record of the board trace joined to the schedule through the IR slot
map); right: the timeline drift — executed start minus planned start of every dispatch against planned
time, one line per hart — with the round's frames late, control-window misses and camera→control median
in the corner. Round 0 is solved from the isolated profile; each later round is re-solved on the
service-time calibration fitted from the previous round's executed traces (its aggregate is printed).
    scripts/hil_feedback_figure.py --tag a90 --spec wh_chain90_solve_500 [--rounds 0 1 2] [--solver cpsat_hard]
                                   [--window 100 240] [--run 1] [--out results/codesign_feedback/refined/hil_feedback_a90.png]
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, statistics, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xpu-rt")); sys.path.insert(0, os.path.join(REPO, "scripts"))
from k1_trace import ir_slot_map, K1_RDTIME_HZ   # noqa: E402
from job_names import split_job_name             # noqa: E402
from xpurt_trace_report import window_misses      # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt   # noqa: E402
RES = os.path.join(REPO, "results", "codesign_feedback")
IRS = {"yolov8_nano_64x96": "ModelBlaster/build/k1_xpurt/yolov8_nano_64x96/int8/graph.json", "fused_full": "ModelBlaster/build/k1_xpurt/fused_full/int8/graph.json",
       "mlp_control": "ModelBlaster/build/k1_xpurt/mlp_control/int8/graph.json", "ffn_block": "ModelBlaster/build/k1_xpurt/ffn_block/int8/graph.json", "dronet": "ModelBlaster/build/k1_xpurt/dronet/int8/graph.json"}
COL = {"yolov8_nano_64x96": "#1f77b4", "fused_full": "#2ca02c", "mlp_control": "#d62728", "ffn_block": "#9467bd", "dronet": "#ff7f0e"}
HART = {f"CPU_P#{i}": i for i in range(4)}; HART.update({f"CPU_E#{i}": 4 + i for i in range(4)})


def planned(sched_path, nets):
    s = json.load(open(sched_path)); out = []
    for v in s["dispatches"].values():
        net, inst = split_job_name(v["job_name"], nets)
        h = HART.get(v["hardware_target"].split(",")[0].strip(), 0)
        out.append((h, float(v["start_time"]), float(v["start_time"]) + float(v["duration"]), net, int(inst or 0), int(v["id"])))
    return out


def executed(trace_path, slotmap):
    out = []
    for r in csv.DictReader(open(trace_path)):
        try:
            net = r["network"]; inst = int(r["instance"]); slot = int(r["dispatch_id"]); s0 = int(r["actual_start_cycles"]); e0 = int(r["actual_end_cycles"]); h = int(r["worker_hart"])
        except (KeyError, ValueError, TypeError):
            continue
        if slot < 0 or e0 <= s0:
            continue
        did = slotmap.get(net, {}).get(slot, slot if net not in slotmap else None)
        if did is None:
            continue
        out.append((h, s0 / K1_RDTIME_HZ * 1e3, e0 / K1_RDTIME_HZ * 1e3, net, inst, did))
    return out


def metrics(recs, periods, windows, skip_ms):
    span = {}
    for h, s0, e0, net, inst, did in recs:
        a, b, rel = span.get((net, inst), (s0, e0, inst * periods[net]))
        span[(net, inst)] = (min(a, s0), max(b, e0), rel)
    wm = window_misses(span, windows, skip_ms)
    ctrl = sorted(b for (n, k), (a, b, rel) in span.items() if n == "mlp_control" and rel >= skip_ms)
    gaps = [y - x for x, y in zip(ctrl, ctrl[1:])]
    # camera->control: frame release -> first control output after that frame's nav ends
    nav_end = {k: b for (n, k), (a, b, rel) in span.items() if n == "fused_full"}
    e2e = []
    for (n, k), (a, b, rel) in span.items():
        if n != "yolov8_nano_64x96" or rel < skip_ms or k not in nav_end:
            continue
        c = next((t for t in ctrl if t >= nav_end[k]), None)
        if c is not None:
            e2e.append(c - rel)
    return wm, gaps, e2e


def gantt(ax, recs, w0, w1, title):
    for h, s0, e0, net, inst, did in recs:
        if e0 < w0 or s0 > w1:
            continue
        ax.broken_barh([(s0, e0 - s0)], (7 - h - 0.4, 0.8), color=COL.get(net, "#888"), lw=0)
    ax.set_xlim(w0, w1); ax.set_ylim(-0.6, 7.6); ax.set_yticks(range(8)); ax.set_yticklabels([f"E{7-i-4}" if 7 - i >= 4 else f"P{7-i}" for i in range(8)], fontsize=7)
    ax.set_title(title, fontsize=9, loc="left"); ax.grid(axis="x", ls=":", lw=0.5); ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a90"); ap.add_argument("--spec", default="wh_chain90_solve_500"); ap.add_argument("--rounds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--solver", default="auto", help="cpsat_hard | cpsat_soft | greedy | auto (the CP-SAT table the round executed)")
    ap.add_argument("--window", nargs=2, type=float, default=[100.0, 240.0]); ap.add_argument("--run", type=int, default=1); ap.add_argument("--skip-ms", type=float, default=100.0)
    ap.add_argument("--out", default=None); ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    spec = json.load(open(os.path.join(REPO, "data/toplevel", a.spec + ".json")))
    periods = {n: float(v["period"]) for n, v in spec["networks"].items()}; windows = {n: float(v["window_duration"]) for n, v in spec["networks"].items()}
    nets = set(periods); slotmap = {n: ir_slot_map(json.load(open(os.path.join(REPO, IRS[n])))) for n in nets if n in IRS and os.path.exists(os.path.join(REPO, IRS[n]))}
    rows = []
    for r in a.rounds:
        T = f"fb{a.tag}r{r}"
        solver = a.solver
        if solver == "auto":
            solver = next((s for s in ("cpsat_hard", "cpsat_soft") if os.path.exists(os.path.join(REPO, f"schedules/fig_{T}_{s}_clamped.json"))), None)
        sp = os.path.join(REPO, f"schedules/fig_{T}_{solver}_clamped.json") if solver else None
        tp = os.path.join(RES, "xpurt_long", f"trace_{T}{solver}r{a.run}_other_run1.csv") if solver else None
        if not (sp and os.path.exists(sp) and tp and os.path.exists(tp)):
            print(f"round {r}: no executed {solver or 'CP-SAT'} table yet"); continue
        cal = os.path.join(RES, "hil_feedback", f"cal_{a.tag}_r{r}.json")
        agg = json.load(open(cal))["aggregate_multiplier"] if os.path.exists(cal) else None
        P = planned(sp, nets); E = executed(tp, slotmap)
        wm, gaps, e2e = metrics(E, periods, windows, a.skip_ms)
        pm = json.load(open(sp.replace("_clamped.json", "_metrics.json"))) if os.path.exists(sp.replace("_clamped.json", "_metrics.json")) else {}
        rows.append((r, solver, agg, P, E, wm, gaps, e2e, pm))
        print(f"round {r} [{solver}] cal {agg}: yolo late {wm.get('yolov8_nano_64x96', {}).get('miss')}/{wm.get('yolov8_nano_64x96', {}).get('n')}, "
              f"control misses {wm.get('mlp_control', {}).get('miss')}/{wm.get('mlp_control', {}).get('n')}, gap mean {statistics.mean(gaps):.2f} p95 {sorted(gaps)[int(0.95 * (len(gaps) - 1))]:.2f}, "
              f"camera->control median {statistics.median(e2e):.1f} ms, predicted misses {pm.get('op_deadline_miss_count')}")
    if not rows:
        return 1
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 2.6 * len(rows) + 0.8), squeeze=False, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    w0, w1 = a.window
    for i, (r, solver, agg, P, E, wm, gaps, e2e, pm) in enumerate(rows):
        lab = "isolated profile" if r == 0 else f"service-time calibration from round {r-1} (×{agg:.3f})"
        gantt(axes[i][0], P, w0, w1, f"round {r} · {solver.replace('_', ' ')} · solved on the {lab}")
        gantt(axes[i][1], E, w0, w1, f"round {r} · executed on the K1 (run {a.run})")
        ax = axes[i][2]
        plan = {(net, inst, did): (h, s0) for h, s0, e0, net, inst, did in P}
        drift = collections.defaultdict(list)
        for h, s0, e0, net, inst, did in E:
            if (net, inst, did) in plan:
                drift[h].append((plan[(net, inst, did)][1], s0 - plan[(net, inst, did)][1]))
        for h in sorted(drift):
            pts = sorted(drift[h]); ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=0.8, label=f"{'P' if h < 4 else 'E'}{h % 4}")
        ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("executed − planned start (ms)", fontsize=7); ax.tick_params(labelsize=7); ax.grid(ls=":", lw=0.5)
        ax.set_title("timeline drift per hart", fontsize=9, loc="left")
        y = wm.get("yolov8_nano_64x96", {}); c = wm.get("mlp_control", {})
        ax.text(0.99, 0.03, f"frames late {y.get('miss', 0)}/{y.get('n', 0)} (worst +{y.get('worst', 0):.1f} ms)\ncontrol windows missed {c.get('miss', 0)}/{c.get('n', 0)} (worst +{c.get('worst', 0):.1f} ms)\n"
                f"control gap {statistics.mean(gaps):.1f} ms mean · camera→control {statistics.median(e2e):.0f} ms\npredicted misses {pm.get('op_deadline_miss_count', '?')}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, bbox=dict(fc="white", ec="0.7", lw=0.5))
        if i == 0:
            ax.legend(fontsize=6, ncol=4, loc="upper left", frameon=False)
    for ax in axes[-1][:2]:
        ax.set_xlabel("time in the table (ms)", fontsize=8)
    axes[-1][2].set_xlabel("planned start (ms)", fontsize=8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COL[n], label=n) for n in sorted(nets)], loc="upper right", fontsize=7, ncol=len(nets), frameon=False)
    fig.suptitle(f"Runtime feedback with the board in it — {a.spec}: the table as solved, as executed, and how far the board slid from the plan, round by round", fontsize=10, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = a.out or os.path.join(RES, "refined", f"hil_feedback_{a.tag}.png")
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight"); print("wrote", out)
    json.dump([{"round": r, "solver": s, "calibration_aggregate": agg, "window_misses": wm, "gap_mean_ms": statistics.mean(g), "e2e_median_ms": statistics.median(e), "predicted_misses": pm.get("op_deadline_miss_count")}
               for r, s, agg, P, E, wm, g, e, pm in rows], open(out.replace(".png", "_metrics.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

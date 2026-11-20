#!/usr/bin/env python3
"""What a plan predicts per model, and what changed against another plan.

Per model, from the schedule file and the workload config (periods):
  planned frame time  = planned end of the frame's last layer - planned start of its first
  finish after release, against the deadline (release + window): planned late frames
  layers per (core kind, cores); multi-core layers
Against --old: how many layers changed option. A layer can have different options in different
frames of a mix, so two counts: layers whose MOST USED option (core kind, kernel type, cores)
changed, and the share of all (layer, frame) dispatches whose option changed.
With --tree (the corrected tree the new plan was solved with): how the entries the plan uses were
corrected (measured or fall-back run time; own gap or the core-kind mean), counted per dispatch.

usage: plan_summary.py --new <scheduled.json> --config <networks.json> [--old <scheduled.json>] [--tree <corrected tree>]
                       [--solve-log <solve log>] --out <dir> --tag plan_r1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

KIND = {"CPU_P": "x100", "CPU_E": "a100"}
NETS = tuple(x for x in os.environ.get("K3TL_NETS", "mlp_control,dronet,yolov8_nano").split(",") if x)
TREE_ENTRIES = "t023_tree_entries.csv"      # written by apply.py (loop_lib.TREE_ENTRIES)


def plan(path: str) -> dict:
    d = json.load(open(path))["dispatches"]
    out: dict = defaultdict(dict)       # (net, inst) -> irid -> entry
    for k, v in d.items():
        m = re.match(r"^(.*?)(\d+)_dispatch_(\d+)$", k)
        net, inst, irid = m.group(1), int(m.group(2)), int(m.group(3))
        t = v["hardware_target"].split("+")
        out[(net, inst)][irid] = dict(kind=KIND[t[0].split("#")[0]], cores=len(t), impl=v.get("impl", ""),
                                      harts=tuple(int(x.split("#")[1]) for x in t), start=float(v["start_time"]),
                                      dur=float(v["duration"]))
    return out


def opt(e: dict) -> tuple:
    return (e["kind"], e["impl"], e["cores"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--old")
    ap.add_argument("--tree")
    ap.add_argument("--solve-log")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))["networks"]
    pn = plan(a.new)
    po = plan(a.old) if a.old else None
    src = {}
    if a.tree:
        for r in csv.DictReader(open(os.path.join(a.tree, TREE_ENTRIES))):
            src[(r["network"], int(r["layer"]), r["core_kind"], r["kernel_type"], int(r["cores"]))] = (r["source"].split()[0], r["gap_source"].split()[0])
    out: dict = dict(new=a.new, old=a.old, tree=a.tree, models={})
    if a.solve_log and os.path.exists(a.solve_log):
        text = open(a.solve_log).read()
        m = re.findall(r"decomposed: (makespan=[^\n]*)", text)
        out["solver_says"] = m[-1].strip() if m else None
        m = re.search(r"exit=(\d+) seconds=(\d+)", text)
        out["solve_exit"], out["solve_seconds"] = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    for net in NETS:
        period, window = float(cfg[net]["period"]), float(cfg[net].get("window_duration", cfg[net]["period"]))
        frames = sorted(i for (n, i) in pn if n == net)
        ft, fin, late = [], [], 0
        mix: Counter = Counter()
        run_src: Counter = Counter()
        gap_src: Counter = Counter()
        for i in frames:
            real = {k: e for k, e in pn[(net, i)].items() if e["dur"] > 0}
            gate = min(e["start"] for e in real.values())
            end = max(e["start"] + e["dur"] for e in real.values())
            ft.append(end - gate)
            fin.append(end - i * period)
            late += (end - i * period) > window + 1e-6
            for irid, e in real.items():
                mix[f"{e['kind']} x{e['cores']}"] += 1
                if src:
                    s = src[(net, irid) + opt(e)]
                    run_src[s[0]] += 1
                    gap_src[s[1]] += 1
        steady = ft[1:] or ft
        b = dict(frames=len(frames), period_ms=period, window_ms=window,
                 planned_frame_ms=dict(median=statistics.median(steady), min=min(steady), max=max(steady), frame0=ft[0]),
                 finish_after_release_ms=dict(median=statistics.median(fin), max=max(fin)),
                 planned_late_frames=late, slack_of_worst_frame_ms=window - max(fin),
                 dispatches_by_kind_cores={k: mix[k] for k in sorted(mix)},
                 multi_core_share=sum(v for k, v in mix.items() if not k.endswith("x1")) / sum(mix.values()))
        if src:
            n = sum(run_src.values())
            b.update(run_time_source_share={k: v / n for k, v in sorted(run_src.items())},
                     gap_source_share={k: v / n for k, v in sorted(gap_src.items())})
        if po is not None:
            layers = sorted({k for (n, i) in pn if n == net for k, e in pn[(n, i)].items() if e["dur"] > 0})
            major_changed = moved = total = 0
            for irid in layers:
                on = Counter(opt(pn[(net, i)][irid]) for i in frames if irid in pn[(net, i)])
                oo = Counter(opt(po[(net, i)][irid]) for i in frames if (net, i) in po and irid in po[(net, i)])
                major_changed += on.most_common(1)[0][0] != oo.most_common(1)[0][0]
                for i in frames:
                    if (net, i) in po and irid in po[(net, i)]:
                        total += 1
                        moved += opt(pn[(net, i)][irid]) != opt(po[(net, i)][irid])
            b.update(layers=len(layers), layers_whose_main_option_changed=major_changed,
                     dispatches_whose_option_changed_share=moved / total if total else float("nan"))
        out["models"][net] = b
    out["plans_identical"] = (po is not None and set(po) == set(pn)
                              and all(po[k] == pn[k] for k in pn)) if po is not None else None
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    if "solver_says" in out:
        print(f"solver: {out['solver_says']}  (exit {out['solve_exit']}, {out['solve_seconds']} s)")
    for net, b in out["models"].items():
        p, fr = b["planned_frame_ms"], b["finish_after_release_ms"]
        line = (f"{net:12s} planned frame {p['median']:.3f} ms (min {p['min']:.3f}, max {p['max']:.3f}); finishes {fr['median']:.3f} ms after release "
                f"(worst {fr['max']:.3f} of {b['window_ms']:.3f}); planned late frames {b['planned_late_frames']} of {b['frames']}; "
                f"multi-core {100 * b['multi_core_share']:.0f}%")
        if po is not None:
            line += (f"; main option changed for {b['layers_whose_main_option_changed']} of {b['layers']} layers, "
                     f"{100 * b['dispatches_whose_option_changed_share']:.1f}% of dispatches")
        print(line)
        if src:
            print(f"{'':12s} run time: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in b["run_time_source_share"].items())
                  + " | gap: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in b["gap_source_share"].items()))
    if po is not None:
        print("plans identical (every dispatch):", out["plans_identical"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

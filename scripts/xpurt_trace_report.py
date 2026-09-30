#!/usr/bin/env python3
"""One-command reading of an XPU-RT board trace: what the run actually delivered.

    scripts/xpurt_trace_report.py results/codesign_feedback/xpurt_long/trace_best25p4_other_run1.csv [...]

Per trace: per-frame YOLO span, nav span, camera->control chain (frame release -> first control
output after that frame's nav), control-output gaps (mean / p95 / max, plus how many exceed
1.5x the control period), per-hart kernel fraction from the harness's own accounting and the
sampler's per-core busy %, when those files sit next to the trace.
"""
from __future__ import annotations
import collections, csv, json, os, re, statistics, sys

HZ = 24e6


def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")


def window_misses(span, windows, skip_ms):
    """Per network: instances whose last dispatch ended after release + window (release = the
    instance's scheduled start, i.e. k * period), ignoring the first skip_ms of the run."""
    out = {}
    for (n, k), (a, b, rel) in span.items():
        if n not in windows or rel < skip_ms:
            continue
        d = out.setdefault(n, {"n": 0, "miss": 0, "worst": 0.0})
        d["n"] += 1; late = b - (rel + windows[n])
        if late > 0:
            d["miss"] += 1; d["worst"] = max(d["worst"], late)
    return out


def report(path, windows=None, skip_ms=100.0):
    rows = [r for r in csv.DictReader(open(path)) if (r.get("actual_end_cycles") or "0").lstrip("-").isdigit() and int(r.get("actual_end_cycles") or 0) > 0 and (r.get("worker_hart") or "-1").lstrip("-").isdigit() and int(r.get("worker_hart") or -1) >= 0]
    if not rows:
        print(f"{path}: no executed rows"); return
    t0 = min(int(r["actual_start_cycles"]) for r in rows)
    span = collections.defaultdict(lambda: [1e18, 0.0, 1e18])
    for r in rows:
        k = (r["network"], int(r["instance"]))
        a = (int(r["actual_start_cycles"]) - t0) / HZ * 1e3; b = (int(r["actual_end_cycles"]) - t0) / HZ * 1e3
        s = span[k]; s[0] = min(s[0], a); s[1] = max(s[1], b); s[2] = min(s[2], float(r.get("predicted_start_ms") or 0))
    nets = sorted({k[0] for k in span})
    ctrl_ends = sorted(v[1] for (n, _), v in span.items() if n == "mlp_control")
    frames = sorted(i for (n, i) in span if n == "yolov8_nano_64x96")
    chain, yolo, nav, late = [], [], [], []
    for i in frames:
        y = span[("yolov8_nano_64x96", i)]; f = span.get(("fused_full", i))
        if i == 0 or not f:
            continue
        yolo.append(y[1] - y[0]); nav.append(f[1] - f[0])
        e = next((t for t in ctrl_ends if t >= f[1]), None)
        if e is not None:
            chain.append(e - y[2])           # from the frame's release (predicted start) to the control output
        late.append(y[0] - y[2])            # how long after its release the frame actually started
    g = [ctrl_ends[i + 1] - ctrl_ends[i] for i in range(1, len(ctrl_ends) - 1)]
    name = os.path.basename(path)
    print(f"== {name}: {len(rows)} rows, {len(frames)} frames, {len(ctrl_ends)} control outputs, run {max(v[1] for v in span.values()):.0f} ms")
    if yolo:
        print(f"   YOLO span/frame  median {statistics.median(yolo):6.2f}  p95 {pct(yolo, .95):6.2f}   nav {statistics.median(nav):5.2f}"
              f"   frame start lag vs release: median {statistics.median(late):6.2f} max {max(late):6.2f} ms")
    if chain:
        print(f"   camera->control  median {statistics.median(chain):6.2f}  p95 {pct(chain, .95):6.2f}  max {max(chain):6.2f} ms  (n={len(chain)})")
    if g:
        print(f"   control gaps     mean {statistics.mean(g):6.2f}  p50 {pct(g, .5):6.2f}  p95 {pct(g, .95):6.2f}  max {max(g):6.2f}  min {min(g):5.2f}"
              f"   >15 ms: {sum(x > 15 for x in g)}/{len(g)}   <5 ms: {sum(x < 5 for x in g)}")
    harts = collections.Counter(int(r["worker_hart"]) for r in rows if r["network"] == "yolov8_nano_64x96")
    print(f"   YOLO dispatches per hart {dict(sorted(harts.items()))}")
    acc = path.replace("trace_", "hart_acc_")
    if os.path.exists(acc):
        ks = [f"h{r['hart']}:{100*float(r['kernel_us'])/max(1,float(r['wall_total_us'])):.0f}%" for r in csv.DictReader(open(acc))]
        print(f"   kernel fraction per hart (HART_ACC) {' '.join(ks)}")
    cpu = path.replace("trace_", "cpu_"); man = path.replace("trace_", "manifest_").replace(".csv", ".json")
    if os.path.exists(cpu) and os.path.exists(man):
        m = json.load(open(man)); w0 = m.get("wall_start_epoch_ms")
        if w0:
            run_ms = max(v[1] for v in span.values())
            per = collections.defaultdict(list)
            for r in csv.DictReader(open(cpu)):
                e = int(r["epoch_ms"])
                if w0 + 500 <= e <= w0 + 500 + run_ms + 3000:      # the run starts ~0.5-3 s after the wall stamp
                    per[int(r["cpu"])].append(float(r["busy_pct"]))
            if per:
                print("   sampler busy % (coarse, run window) " + " ".join(f"c{c}:{statistics.mean(v):.0f}" for c, v in sorted(per.items())))
        print(f"   policy {m.get('sched_policy')}  solver {m.get('solver')}")
    if windows:
        wm = window_misses(span, windows, skip_ms)
        print("   window misses    " + "  ".join(f"{n}: {d['miss']}/{d['n']} (worst +{d['worst']:.1f} ms)" for n, d in sorted(wm.items()))
              + f"   [after {skip_ms:.0f} ms]")


if __name__ == "__main__":
    # scripts/xpurt_trace_report.py [--windows net=ms,...] [--skip-ms 100] trace.csv ...
    args = sys.argv[1:]; windows = None; skip = 100.0
    while args and args[0].startswith("--"):
        if args[0] == "--windows":
            windows = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args[1].split(",")}; args = args[2:]
        elif args[0] == "--skip-ms":
            skip = float(args[1]); args = args[2:]
        else:
            raise SystemExit(f"unknown option {args[0]}")
    for p in args:
        report(p, windows, skip)

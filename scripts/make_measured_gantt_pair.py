#!/usr/bin/env python3
"""Build the panel-I schedule pair from two board traces -- one per runtime -- in the same
format, from the same kind of file, by the same code.

    scripts/make_measured_gantt_pair.py \\
        --xpu-trace results/codesign_feedback/xpurt_long/trace_long25_other_run1.csv \\
        --xpu-cpu   results/codesign_feedback/xpurt_long/cpu_long25_other_run1.csv \\
        --xpu-manifest results/codesign_feedback/xpurt_long/manifest_long25_other_run1.json \\
        --ros-trace results/codesign_feedback/ros_traced/25_spin_r1/trace.csv \\
        --ros-cpu   results/codesign_feedback/ros_traced/25_spin_r1/cpu.csv \\
        --ros-manifest results/codesign_feedback/ros_traced/25_spin_r1/manifest.json \\
        --window-ms 160

Both traces carry one row per executed unit with the hart it ran on and rdtime at entry and
exit (24 MHz): for XPU-RT a row is a dispatch, for ROS a row is a node callback. Each row
becomes one bar on the hart's lane. Nothing is scaled, normalised or laid out by hand.

The window: `--skip-ms` into the run (past cold start), `--window-ms` long, the same length on
both arms, starting at the first camera release after the skip so each arm's window opens on
a frame boundary. Every lane's mean busy % over the whole run (past warm-up) comes from the
per-core sampler and is written into the schedule's metadata; lanes with no bars in the window
are labelled with that number rather than with a word.

Outputs schedules/measured_gantt_{xpu,ros}.json plus *_metrics.json sidecars that carry the
window, the source files, the per-lane busy % and (for XPU-RT) the per-hart kernel fraction
from the harness's own accounting when a hart_acc file sits next to the trace.
"""
from __future__ import annotations
import argparse, bisect, collections, csv, hashlib, json, os, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_busy_source(cpu_exists, has_rdtime, acc_exists, has_epoch):
    """Which instrument fills the per-lane busy %. The sampler (`cpu_*.csv`, windowed to the run by rdtime or by wall
    clock) comes first so that every row of one figure — XPU-RT and ROS 2 alike — is measured by the same instrument;
    the harness's own per-hart kernel accounting (`hart_acc_*.csv`, XPU-RT only) is the fallback when a run has no
    sampler window, and is always recorded alongside as `kernel_frac_pct` when present."""
    if cpu_exists and has_rdtime:
        return "sampler_rdtime"
    if cpu_exists and has_epoch:
        return "sampler_epoch"
    if acc_exists:
        return "hart_acc"
    return None

HZ = 24e6
WARMUP_MS = 3000.0
NET_ORDER = ["yolov8_nano_64x96", "fused_full", "mlp_control"]


def lane(hart: int) -> str:
    return f"CPU_{'P' if hart < 4 else 'E'}#{hart % 4}"


def executed_units(rows):
    """what to draw as bars: every measured unit, without drawing the same work twice.

    A pooled op is recorded twice over -- once as the op on the calling hart (its span
    apportioned from the model's per-op cycle counts) and once per worker slice with absolute
    stamps. The slices are the measurement, so they win, and a per-op row that a slice already
    covers on the same hart is dropped. Sequential ops keep their per-op row; a trace with no
    detail rows is returned unchanged, and a callback whose own ops were never traced keeps its
    row so that node's work is still drawn."""
    det = [r for r in rows if r.get("op") not in (None, "", "node_callback")]
    if not det:
        return rows
    # Detail rows stand in for the callback they came from, not for every callback in the file: a
    # node that traces only its boundary (nav, control) keeps its row, or its work leaves the chart
    # as soon as some other node starts reporting its ops.
    detailed = {(r["net"], r["inst"]) for r in det}
    rest = [r for r in rows if r.get("op") in (None, "", "node_callback")
            and (r["net"], r["inst"]) not in detailed]
    shards = [r for r in det if r.get("op") == "pool_shard"]
    if not shards:
        return det + rest
    # Per hart, the slices sorted by start, plus a running maximum of their ends: a row is covered
    # when some slice starting at or before it ends at or after it. Both are found by bisection --
    # a run carries tens of thousands of slices per hart and as many per-op rows, so a linear scan
    # per row is quadratic and does not finish.
    by_hart = {}
    for r in shards:
        by_hart.setdefault(r["hart"], []).append((r["s"], r["e"]))
    index = {}
    for h, v in by_hart.items():
        v.sort()
        index[h] = (v, [a for a, _ in v], max(b - a for a, b in v))

    def covered(r):
        """Is this per-op row the same work some slice on its hart already measured?

        The slice carries absolute stamps; the per-op row's span is apportioned from the model's
        per-op cycle counts over the whole callback, so the row is routinely wider than the slice
        at both ends -- the apportionment charges the op with a share of the pool's dispatch and
        barrier overhead, which the slice does not contain. Asking the slice to CONTAIN the row
        therefore keeps both, and the calling hart is drawn twice for work it did once: a frame
        whose slices span 29.2 ms was drawn with 32.4 ms of bar on that hart, a serial pass over
        every layer laid under the parallel one.

        A row counts as covered when the slices on its own hart account for at least
        COVER_FRAC of it. A pooled op always has a slice on the calling hart -- worker 0 is the
        calling thread -- so the test finds them all; a sequential op has none under it and keeps
        its row, which is what puts the residual adds and concatenations back on the calling
        hart between bursts."""
        ent = index.get(r["hart"])
        if not ent:
            return False
        sl, starts, longest = ent
        a, b = r["s"], r["e"]
        if b <= a:
            return False
        acc = 0.0
        i = bisect.bisect_left(starts, a - longest)      # first slice that can still reach a
        while i < len(sl) and sl[i][0] < b:
            acc += max(0.0, min(b, sl[i][1]) - max(a, sl[i][0]))
            i += 1
        return acc >= COVER_FRAC * (b - a)

    return coalesce_shards(shards) + [r for r in det if r.get("op") != "pool_shard" and not covered(r)] + rest


def coalesce_shards(shards):
    """One bar per pooled dispatch, spanning the harts that ran it.

    The pool records a slice per worker, so a YOLO frame arrives as sixty dispatches times four
    workers: two hundred and forty slivers under a millisecond each, which draw as a hatch rather
    than as a schedule and say nothing about how wide each dispatch was. The XPU-RT row already
    draws a sharded dispatch as one bar across its lanes (`width_of`), and this is the same thing
    measured rather than declared: the workers of one dispatch are joined into a single bar from the
    first slice's start to the last slice's end, carrying the lanes they ran on. A dispatch with one
    worker is unchanged."""
    by = {}
    for r in shards:
        by.setdefault((r["net"], r["inst"], r.get("did")), []).append(r)
    out = []
    for g in by.values():
        if len(g) == 1:
            out.append(g[0]); continue
        first = min(g, key=lambda r: r["s"])
        out.append(dict(first, s=first["s"], e=max(r["e"] for r in g),
                        lanes=sorted({lane(r["hart"]) for r in g})))
    return out


def read_trace(path):
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            a, b = int(r.get("actual_start_cycles") or 0), int(r.get("actual_end_cycles") or 0)
            hart = int(r.get("worker_hart") or -1)
        except (KeyError, ValueError, TypeError):
            continue
        if b <= a or hart < 0:
            continue
        rows.append({"net": r["network"], "inst": int(r["instance"]), "hart": int(r["worker_hart"]), "op": (r.get("op") or ""),
                     "did": int(r["dispatch_id"]) if (r.get("dispatch_id") or "").isdigit() else None,
                     "s": a / HZ * 1000.0, "e": b / HZ * 1000.0, "name": r.get("name") or r["network"],
                     "rel": float(r["predicted_start_ms"]) if (r.get("predicted_start_ms") or "") not in ("", "0", "0.0") else None})
    # a ROS trace has no scheduled start; its frame releases are in released.csv next to it
    rel_csv = os.path.join(os.path.dirname(path), "released.csv")
    if os.path.exists(rel_csv):
        rel = {int(r["frame_seq"]): int(r["t_release_ticks"]) / HZ * 1000.0 for r in csv.DictReader(open(rel_csv))}
        for r in rows:
            if r["net"] == NET_ORDER[0] and r["inst"] in rel:
                r["rel"] = rel[r["inst"]]
    return rows


def representative_skip_ms(rows, window_ms, floor_ms=100.0):
    """The offset whose own window behaves like the whole run, in ms from the first release.

    WHY. `--skip-ms` was a fixed 400 ms past cold start. That is fine for a schedule -- a periodic
    table is at its steady state from the first frame -- and wrong for a queue. On the timer-driven
    ROS arm the backlog is still filling at 400 ms: the drawn window showed frames completing in
    116 ms while the caption, correctly, gave the run median as 245.6, so measuring the bars and
    reading the text gave different answers. Rather than guess when a queue has converged (arms that
    drift slightly defeat any convergence test), slide the window and take the one whose median
    frame latency is closest to the run's -- the window is then representative by construction, and
    the sidecar records both medians so the claim can be checked.
    """
    net = NET_ORDER[0]
    spans = []
    for k in sorted({x["inst"] for x in rows if x["net"] == net}):
        fr = [x for x in rows if (x["net"], x["inst"]) == (net, k)]
        rel = min((x["rel"] for x in fr if x.get("rel") is not None), default=None)
        if rel is not None:
            spans.append((rel, max(x["e"] for x in fr) - rel))
    if len(spans) < 8:
        return floor_ms, None, None
    spans.sort()
    t0 = spans[0][0]
    med = statistics.median(lat for _, lat in spans)
    best = (float("inf"), floor_ms, None)
    step = max(window_ms / 4.0, 25.0)
    t = floor_ms
    while t0 + t + window_ms <= spans[-1][0]:
        inside = [lat for rel, lat in spans if t0 + t <= rel < t0 + t + window_ms]
        if len(inside) >= 3:
            m = statistics.median(inside)
            if abs(m - med) < best[0]:
                best = (abs(m - med), t, m)
        t += step
    return best[1], med, best[2]


def window_rows(rows, skip_ms, window_ms):
    """Rows inside [t_open, t_open + window), t_open = first yolo start at or after skip_ms."""
    t0 = min(r["s"] for r in rows)
    cand = [r["s"] for r in rows if r["net"] == NET_ORDER[0] and r["s"] - t0 >= skip_ms]
    t_open = min(cand) if cand else t0 + skip_ms
    sel = [r for r in executed_units(rows) if r["s"] >= t_open and r["s"] < t_open + window_ms]
    return t_open, sel


def busy_from_cpu(path, lo_ms, hi_ms, key="epoch_ms"):
    if not path or not os.path.exists(path):
        return {}
    per = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        t = float(r[key])
        if lo_ms <= t <= hi_ms:
            per[int(r["cpu"])].append(float(r["busy_pct"]))
    return {lane(c): round(statistics.mean(v), 1) for c, v in sorted(per.items()) if v}


COVER_FRAC = 0.5        # a per-op row whose hart's slices account for this much of it is that pooled op,
                        # drawn once as the slices; below it the op ran sequentially and keeps its row
POOL_BUSY_PCT = 50.0   # a declared pool hart counts as having run the kernel only if the sampler saw it busy
# The navigation pool's harts carry one 3.6 ms kernel per 33 ms camera period, so their duty cycle is
# around a tenth even when every shard runs; the perception pool's threshold would reject all four.
NAV_BUSY_PCT = 5.0


def to_schedule(sel, t_open, meta, pool_lanes=None, width_of=None, busy=None, nav_lanes=None):
    # A node built with per-slice tracing records the hart each pool worker used; then nothing is
    # credited for that network, because the work was measured where it ran. It is per network rather
    # than per run: the two pools are separate builds, and one of them tracing its slices says nothing
    # about whether the other did.
    traced_nets = {r["net"] for r in sel if r.get("op") == "pool_shard"}
    """pool_lanes: for a ROS perception node that runs YOLO on a worker pool, the trace row carries
    ONE hart -- the callback thread's. The pool's helpers ran the kernel's shards on other harts over
    the same interval (the callback takes the 4-hart standalone time) but the trace does not record
    them, so the bar also spans the pool's lanes, with the traced hart named in `traced_target` so a
    reader can tell a measured placement from a credited one.

    The pool is credited from the SAMPLER, not from the manifest's declared `pool_harts`: the nodes run
    unpinned, so the operating system may place the pool's threads elsewhere, and a declared hart the
    sampler saw idle did not run this kernel."""
    disp = {}
    for i, r in enumerate(sorted(sel, key=lambda r: r["s"]), start=1):
        tgt = lane(r["hart"]); measured = False
        pl = pool_lanes.get(r["name"], pool_lanes.get("*")) if isinstance(pool_lanes, dict) else pool_lanes   # per-node pools when several YOLO nodes run
        if pl and r["net"] == NET_ORDER[0] and NET_ORDER[0] not in traced_nets:
            # No per-slice rows in this trace: the callback is all the trace records, so the pool's
            # lanes are credited from the sampler (a declared hart the sampler saw idle did not run it).
            ran = [l for l in pl if (busy or {}).get(l, 100.0) >= POOL_BUSY_PCT]
            tgt = "+".join(sorted(set(ran) | {tgt}))
        elif nav_lanes and r["net"] == NET_ORDER[1] and NET_ORDER[1] not in traced_nets:
            # The navigation network shards at codegen time rather than through the runtime pool, so its
            # kernel spans several harts while the trace row carries only the callback's. Credited the
            # same way and on the same evidence: a declared hart the sampler saw idle did not run it.
            ran = [l for l in nav_lanes if (busy or {}).get(l, 100.0) >= NAV_BUSY_PCT]
            tgt = "+".join(sorted(set(ran) | {tgt}))
        elif r.get("lanes"):                # a pooled dispatch, measured: its workers' own harts
            tgt = "+".join(r["lanes"]); measured = True
        elif width_of:                      # XPU-RT: a sharded dispatch ran on every hart of its target
            w = width_of.get((r["net"], r["inst"], r.get("did")))
            if w and "+" in w:
                tgt = w
        disp[str(i)] = {"id": i, "ordinal": 1, "total": 1, "dependencies": [],
                        "hardware_target": tgt, "traced_target": lane(r["hart"]),
                        # every lane of this bar carries its own worker's stamps, so none of it is
                        # credited and the drawing has no reason to hatch any of it
                        **({"lanes_measured": True} if measured else {}),
                        "start_time": round(r["s"] - t_open, 6), "duration": round(r["e"] - r["s"], 6),
                        "job_name": f"{r['net']}{r['inst']}", "module_name": r["name"],
                        "release_policy": "periodic", "time_dep_mode": "hard"}
    mk = max(v["start_time"] + v["duration"] for v in disp.values()) if disp else 0.0
    meta = dict(meta); meta["makespan"] = round(mk, 4)
    return {"metadata": meta, "dispatches": disp}, mk


def callback_rows(rows):
    """the node-callback rows of a ROS trace, or every row of an XPU-RT one.

    A traced ROS node now also writes what ran *inside* each callback: the network's ops, and one
    row per pool worker that took a slice. Those carry the same (network, instance) as their
    callback, so latency measured from them is identical -- verified on the board, 195 frames,
    60.082 ms either way -- but the chain is defined at the callback boundary and is read there."""
    det = [r for r in rows if r.get("op") not in (None, "", "node_callback")]
    if not det:
        return rows
    cb = [r for r in rows if r.get("op") == "node_callback"]
    return cb if cb else rows


def per_frame_chain(rows):
    """camera-frame -> control latency per instance where the trace lets us pair them.

    Measured from the frame's RELEASE (the camera timer for ROS, the scheduled release for
    XPU-RT) to the first control output after that frame's nav result, so queueing before YOLO
    starts is inside the number on both arms.
    """
    by = collections.defaultdict(list)
    for r in callback_rows(rows):
        by[(r["net"], r["inst"])].append(r)
    ctrl_ends = sorted(max(x["e"] for x in v) for (n, _), v in by.items() if n == "mlp_control")
    out = []
    for (n, k), v in by.items():
        if n != NET_ORDER[0]:
            continue
        nav = by.get(("fused_full", k))
        if not nav:
            continue
        s = min(x["s"] for x in v); e_nav = max(x["e"] for x in nav)
        rel = min((x["rel"] for x in v if x.get("rel") is not None), default=s)
        e_ctrl = next((t for t in ctrl_ends if t >= e_nav), None)
        if e_ctrl is not None:
            out.append((k, e_ctrl - rel, max(x["e"] for x in v) - s))
    return out


def queue_depth(rows, t_open, window_ms, step_ms=0.5):
    """How many camera frames are released but not yet answered, sampled across the drawn window.

    WHY THIS IS IN THE SIDECAR. The bars alone mislead once the baseline uses the whole machine: an
    eight-lane ROS row at 55 % on every hart looks healthier than a four-lane one at 88 %, while its
    chain is seven times longer. What separates them is not how busy the harts are but how many
    frames are waiting -- the depth is the queue the default QoS lets build, and it is the mechanism
    the figure is about. A frame counts from its release until its own perception finishes.
    """
    spans = []
    for k in sorted({x["inst"] for x in rows if x["net"] == NET_ORDER[0]}):
        fr = [x for x in rows if (x["net"], x["inst"]) == (NET_ORDER[0], k)]
        rel = min((x["rel"] for x in fr if x.get("rel") is not None), default=None)
        if rel is None:
            continue
        spans.append((rel, max(x["e"] for x in fr)))
    out, t = [], t_open
    while t <= t_open + window_ms:
        out.append((round(t - t_open, 3), sum(1 for a, b in spans if a <= t < b)))
        t += step_ms
    return out


def late_frames_in_window(rows, t_open, window_ms, yolo_win):
    """The frames drawn in the window that missed their deadline, so the panel can mark them."""
    out = []
    for k in sorted({x["inst"] for x in rows if x["net"] == NET_ORDER[0]}):
        fr = [x for x in rows if (x["net"], x["inst"]) == (NET_ORDER[0], k)]
        rel = min((x["rel"] for x in fr if x.get("rel") is not None), default=None)
        if rel is None or not (t_open <= rel <= t_open + window_ms):
            continue
        end = max(x["e"] for x in fr)
        if end - rel > yolo_win:
            out.append({"inst": int(k), "release_ms": round(rel - t_open, 3),
                        "end_ms": round(end - t_open, 3), "over_ms": round(end - rel - yolo_win, 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    for arm in ("xpu", "ros"):
        ap.add_argument(f"--{arm}-trace"); ap.add_argument(f"--{arm}-cpu")
        ap.add_argument(f"--{arm}-manifest")
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME:KIND:trace[:cpu[:manifest[:schedule]]] (KIND xpu|ros); repeat for every row of the Gantt. "
                         "Without it the two --xpu-*/--ros-* arms are built as before.")
    ap.add_argument("--window-ms", type=float, default=160.0)
    ap.add_argument("--skip-ms", type=float, default=None,
                    help="offset into the run to open the window at; by default each row picks the offset whose own "
                         "frames have the run's median latency, so the bars and the caption describe the same thing")
    ap.add_argument("--out-prefix", default="schedules/measured_gantt")
    ap.add_argument("--xpu-schedule", help="the schedule JSON the XPU-RT trace executed; supplies each dispatch's hart set")
    ap.add_argument("--spec", default=None,
                    help="workload spec whose periods label the window bands, and whose yolo "
                         "window_duration decides when a frame is late (required unless --yolo-window-ms)")
    ap.add_argument("--yolo-window-ms", type=float, default=None,
                    help="a frame is on time when its last dispatch ends within this of its release (default: the spec's yolo window_duration, else its period)")
    a = ap.parse_args()
    # No default workload: a default (e.g. 25 Hz, 40 ms perception period) would label any other
    # camera rate's pair with the wrong period and judge "frame on time" against the wrong window --
    # and a scheduled arm's manifest carries no rate_hz, so the guard below cannot see it.
    # Requiring the spec covers BOTH arms rather than only the one that happens to state a rate.
    if not a.spec and a.yolo_window_ms is None:
        sys.exit("--spec is required (e.g. data/toplevel/wh_chain45_free.json): it supplies the period "
                 "the window bands are labelled with and the window a frame must finish inside to count "
                 "as on time. Pass --yolo-window-ms instead only when no spec describes the run.")
    spec = json.load(open(a.spec))["networks"] if a.spec and os.path.exists(a.spec) else {}
    yolo_win = a.yolo_window_ms or spec.get("yolov8_nano_64x96", {}).get("window_duration") or spec.get("yolov8_nano_64x96", {}).get("period") or 22.22
    periods = {"yolo_period_ms": spec.get("yolov8_nano_64x96", {}).get("period"),
               "nav_period_ms": spec.get("fused_full", {}).get("period"),
               "ctrl_period_ms": spec.get("mlp_control", {}).get("period")}

    first_sched = a.xpu_schedule or next(((sp.split(":") + [None] * 6)[5] for sp in a.arm if len(sp.split(":")) > 5 and sp.split(":")[5]), None)
    if first_sched and os.path.exists(first_sched):
        sm = json.load(open(first_sched)).get("metadata", {})
        if sm.get("camera_period_ms"):
            periods.update({"yolo_period_ms": sm["camera_period_ms"], "nav_period_ms": sm["camera_period_ms"],
                            "ctrl_period_ms": sm.get("ctrl_period_ms", 10.0)})
    arms = []
    for spec_ in a.arm:
        f = (spec_.split(":") + [None] * 6)[:6]
        arms.append({"name": f[0], "kind": f[1], "trace": f[2], "cpu": f[3] or None, "manifest": f[4] or None, "schedule": f[5] or None})
    if not arms:
        arms = [{"name": arm, "kind": arm, "trace": getattr(a, f"{arm}_trace"), "cpu": getattr(a, f"{arm}_cpu"),
                 "manifest": getattr(a, f"{arm}_manifest"), "schedule": a.xpu_schedule if arm == "xpu" else None}
                for arm in ("xpu", "ros") if getattr(a, f"{arm}_trace")]
    for A in arms:
        arm, kind = A["name"], A["kind"]; trace, cpu, manp = A["trace"], A["cpu"], A["manifest"]; xpu_schedule = A["schedule"]
        rows = read_trace(trace)
        man = json.load(open(manp)) if manp and os.path.exists(manp) else {}
        if a.skip_ms is None:
            skip_ms, run_med, win_med = representative_skip_ms(rows, a.window_ms)
            window_pick = "representative"
        else:
            skip_ms, run_med, win_med, window_pick = a.skip_ms, None, None, "fixed"
        t_open, sel = window_rows(rows, skip_ms, a.window_ms)
        # busy % over the whole run past warm-up: the per-core sampler windowed to the run (by rdtime for XPU-RT, whose
        # manifest carries the run's absolute origin; by wall clock for ROS 2, whose manifest carries epoch start/end),
        # the same instrument for every row; the harness's own per-hart kernel accounting is recorded alongside
        busy = {}; kernel_frac = {}
        acc = trace.replace("trace_", "hart_acc_")
        acc_exists = os.path.exists(acc) and "hart_acc_" in acc
        if acc_exists:
            for r in csv.DictReader(open(acc)):
                kernel_frac[lane(int(r["hart"]))] = round(100.0 * float(r["kernel_us"]) / max(1.0, float(r["wall_total_us"])), 1)
        busy_source = pick_busy_source(bool(cpu and os.path.exists(cpu)), bool(man.get("run_t0_rdtime")), acc_exists, "wall_start_epoch_ms" in man)
        if busy_source == "sampler_rdtime":
            run_ticks = (max(r["e"] for r in rows) - min(r["s"] for r in rows)) * HZ / 1000.0
            busy = busy_from_cpu(cpu, int(man["run_t0_rdtime"]), int(man["run_t0_rdtime"]) + run_ticks, key="rdtime_ticks")
            meta_busy_note = "sampler busy % over the run (rdtime-aligned)"
        elif busy_source == "sampler_epoch":
            if "wall_end_epoch_ms" in man:
                busy = busy_from_cpu(cpu, man["wall_start_epoch_ms"] + WARMUP_MS, man["wall_end_epoch_ms"])
            else:
                run_ms = (max(r["e"] for r in rows) - min(r["s"] for r in rows))
                busy = busy_from_cpu(cpu, man["wall_start_epoch_ms"], man["wall_start_epoch_ms"] + run_ms + 4000)
            meta_busy_note = "sampler busy % over the run window"
        elif busy_source == "hart_acc":
            busy = dict(kernel_frac); meta_busy_note = "kernel fraction per hart over the run (harness accounting)"
        else:
            meta_busy_note = None
        chain = per_frame_chain(rows)
        ctrl_ends = sorted(max(x["e"] for x in v) for (n, _), v in
                           collections.defaultdict(list, {(r["net"], r["inst"]): [x for x in rows if (x["net"], x["inst"]) == (r["net"], r["inst"])]
                                                          for r in rows if r["net"] == "mlp_control"}).items())
        gaps = [ctrl_ends[i + 1] - ctrl_ends[i] for i in range(1, len(ctrl_ends) - 1)]
        lags = []
        for k in sorted({x["inst"] for x in rows if x["net"] == NET_ORDER[0]}):
            fr = [x for x in rows if (x["net"], x["inst"]) == (NET_ORDER[0], k)]
            rel = min((x["rel"] for x in fr if x.get("rel") is not None), default=None)
            if k >= 1 and rel is not None:          # a frame whose release the trace did not record carries no lag
                lags.append(min(x["s"] for x in fr) - rel)
        warm = [c for (k, c, _) in chain if k >= 1]
        yolo = [y for (k, _, y) in chain if k >= 1]
        # frames late by the window: the frame's last dispatch ended after release + window
        late = []
        for k in sorted({x["inst"] for x in rows if x["net"] == NET_ORDER[0]}):
            fr = [x for x in rows if (x["net"], x["inst"]) == (NET_ORDER[0], k)]
            rel = min((x["rel"] for x in fr if x.get("rel") is not None), default=None)
            if rel is not None and k >= 1:
                late.append(max(x["e"] for x in fr) - rel > yolo_win)
        meta = {"source": trace, "cpu_source": cpu, "manifest": manp,
                "source_sha256": sha256_of(trace), "cpu_source_sha256": sha256_of(cpu) if cpu and os.path.exists(cpu) else None,
                "manifest_sha256": sha256_of(manp) if manp and os.path.exists(manp) else None,
                "window_open_ms": round(t_open - min(r["s"] for r in rows), 3), "window_ms": a.window_ms,
                "window_pick": window_pick,
                "run_latency_median_ms": round(run_med, 3) if run_med is not None else None,
                "window_latency_median_ms": round(win_med, 3) if win_med is not None else None,
                "busy_pct": busy, "busy_note": meta_busy_note, "busy_source": busy_source, "kernel_frac_pct": kernel_frac or None,
                "chain_ms_median": round(statistics.median(warm), 3) if warm else None,
                "chain_ms_p95": round(sorted(warm)[int(0.95 * (len(warm) - 1))], 3) if warm else None,
                "yolo_span_ms_median": round(statistics.median(yolo), 3) if yolo else None,
                "n_frames_paired": len(warm),
                "ctrl_gap_mean_ms": round(statistics.mean(gaps), 3) if gaps else None,
                "ctrl_gap_max_ms": round(max(gaps), 3) if gaps else None,
                "frame_start_lag_median_ms": round(statistics.median(lags), 3) if lags else None,
                "queue_depth": queue_depth(rows, t_open, a.window_ms),
                "queue_depth_max": max((d for _, d in queue_depth(rows, t_open, a.window_ms)), default=0),
                "late_frames_in_window": late_frames_in_window(rows, t_open, a.window_ms, yolo_win),
                "yolo_window_ms": yolo_win, "frames_late": int(sum(late)), "frames_checked": len(late),
                "frames_late_frac": round(sum(late) / len(late), 4) if late else None,
                "frame_start_lag_max_ms": round(max(lags), 3) if lags else None,
                "end_to_end_latency_ms": round(statistics.median(warm), 3) if warm else None,
                "end_to_end_deadline_ms": __import__("measured_timing").CHAIN["deadline_ms"],
                "executor": man.get("executor"), "affinity_mask": man.get("affinity_mask"),
                "sched_policy": man.get("sched_policy"), "solver": man.get("solver"),
                "rate_hz": man.get("rate_hz"), **{k: v for k, v in periods.items() if v}}
        # A spec for one camera rate applied to a pair flown at another labels the wrong period and
        # judges "frame on time" against the wrong window.
        # The ROS manifest states the rate it ran at, so the contradiction is detectable: refuse it
        # rather than emit a sidecar whose lateness count belongs to a different workload.
        if man.get("rate_hz") and periods.get("yolo_period_ms"):
            want = 1000.0 / float(man["rate_hz"])
            if abs(want - float(periods["yolo_period_ms"])) > 0.5:
                sys.exit(f"spec/rate mismatch: {arm}'s manifest ran at {man['rate_hz']} Hz "
                         f"({want:.2f} ms period) but --spec {a.spec} declares a "
                         f"{float(periods['yolo_period_ms']):.2f} ms perception period. Pass the spec for "
                         f"this camera rate (e.g. data/toplevel/wh_chain{int(man['rate_hz'])}_free.json), "
                         f"or --yolo-window-ms explicitly.")
        pool_lanes = None
        if man.get("yolo_pool") and man.get("pool_harts"):
            pool_lanes = [lane(int(h)) for h in str(man["pool_harts"]).split(",") if h.strip()]
        elif isinstance(man.get("processes"), dict):
            # one pool per perception node (a graph may run several YOLO nodes, each on its own harts)
            pools = {}
            for pp in man["processes"].values():
                if pp.get("yolo_pool") and pp.get("pool_harts"):
                    for node in str(pp.get("nodes", "")).split(","):
                        pools[node.strip()] = [lane(int(h)) for h in str(pp["pool_harts"]).split(",") if h.strip()]
            if pools:
                pool_lanes = pools if len(pools) > 1 else next(iter(pools.values()))
        nav_lanes = None
        if man.get("nav_pool") and man.get("nav_harts"):
            nav_lanes = [lane(int(h)) for h in str(man["nav_harts"]).split(",") if h.strip()]
        elif isinstance(man.get("processes"), dict):
            for pp in man["processes"].values():
                if pp.get("nav_pool") and pp.get("nav_harts"):
                    nav_lanes = [lane(int(h)) for h in str(pp["nav_harts"]).split(",") if h.strip()]
                    break
        meta["yolo_pool_lanes"] = pool_lanes; meta["nav_pool_lanes"] = nav_lanes
        meta["kind"] = kind; meta["arm"] = arm
        meta["solver"] = (man.get("solver") or "XPU-RT") if kind == "xpu" else None
        # what each row's placement is, from the manifests rather than from the bars in the window
        procs = man.get("processes") if isinstance(man.get("processes"), dict) else None
        masks = ([str(v.get("affinity_mask", "")) for v in procs.values()] if procs else [str(man.get("affinity_mask", ""))])
        n_procs = len(procs) if procs else 1
        if kind == "xpu":
            # the executed table names its solver (clamp step); the run manifest only knows the file
            sm_ = json.load(open(man["schedule"])).get("metadata", {}) if man.get("schedule") and os.path.exists(man["schedule"]) else {}
            solver_name = {"cpsat": "CP-SAT", "greedy_periodic": "greedy", "greedy": "greedy"}.get(sm_.get("solver") or "", sm_.get("solver") or "")
            meta["arm_label"] = f"XPU-RT · {solver_name}" if solver_name else (man.get("solver") or "XPU-RT")
            meta["solver_name"] = sm_.get("solver"); meta["placement_note"] = "8 harts · scheduled"
        else:
            unpinned = all(m.lower().endswith("ff") for m in masks if m)
            ex = man.get("executor", "single")
            meta["arm_label"] = (f"ROS 2 · {n_procs} process{'es' if n_procs > 1 else ''} · {ex} executor"
                                 + (" · chained control" if man.get("ctrl_mode") == "chained" or (procs and any(v.get("ctrl_mode") == "chained" for v in procs.values())) else "")
                                 + (" · unpinned" if unpinned else " · pinned"))
            n_pool = len({l for v in pool_lanes.values() for l in v}) if isinstance(pool_lanes, dict) else (len(pool_lanes) if pool_lanes else 0)
            meta["placement_note"] = (("8 cores · OS-scheduled" if unpinned else f"{n_procs} process{'es' if n_procs > 1 else ''} pinned")
                                      + (f" · YOLO on {n_pool} harts" if n_pool else ""))
        width_of = None
        if kind == "xpu" and xpu_schedule and os.path.exists(xpu_schedule):
            sys.path.insert(0, os.path.join(REPO, "xpu-rt")); from job_names import split_job_name   # noqa: E402
            known = {"yolov8_nano_64x96", "fused_full", "mlp_control", "ffn_block", "dronet"}; width_of = {}
            for k, v in json.load(open(xpu_schedule))["dispatches"].items():
                net, inst = split_job_name(v["job_name"], known)
                width_of[(net, int(inst or 0), int(v["id"]))] = v["hardware_target"]
            meta["xpu_schedule"] = xpu_schedule
        sched, mk = to_schedule(sel, t_open, meta, pool_lanes, width_of, busy, nav_lanes)
        p = f"{a.out_prefix}_{arm}.json"
        json.dump(sched, open(p, "w"), indent=1)
        json.dump(meta, open(p.replace(".json", "_metrics.json"), "w"), indent=1)
        lanes_used = sorted({v["hardware_target"] for v in sched["dispatches"].values()})
        print(f"{arm}: {len(sched['dispatches'])} bars in {a.window_ms:.0f} ms window from t={meta['window_open_ms']:.1f} ms; "
              f"lanes {lanes_used}; chain median {meta['chain_ms_median']} ms over {len(warm)} frames; busy {busy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

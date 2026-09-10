#!/usr/bin/env python3
"""Side-by-side gantts for cells where XPU-RT scheduling loses to pinning.

Both runtimes emit a machine-readable trace block, so these are drawn from the
measurements rather than from either side's prediction:

    ROS      logs/<cell>__a<N>.log   between ROS_PINSWEEP_TRACE_BEGIN/END
    XPU-RT   ../<sweep10>/runs/<cell>__<solver>/rep<k>/run.log
             between MODELBLASTER_XPURT_TRACE_BEGIN/END

Two things this has to get right, and both are easy to get wrong:

  * THE PLACEMENT MUST BE THE ONE THE NUMBER CAME FROM. The headline ratio is
    computed on `ros_np_best_ms`, and for several cells the np-best placement
    is NOT `ros_best_id` (which is ranked feasible-first on the wall clock).
    On control_mix_hd the ranked-best puts yolov8 on HTA and finishes the
    aperiodic work in 7.53 ms; the np-best puts it on DSP and finishes in
    4.65 ms. Drawing the ranked-best gantt beside a ratio computed from the
    np-best one would be showing a different experiment. The placement is
    resolved by matching each candidate log's own header against
    `ros_np_best_label`.
  * THE OBJECTIVE IS NOT THE WALL CLOCK. Both panels run to a similar wall
    time because both must honour the same periodic horizon. What is being
    compared is when the APERIODIC work finishes, so that is what the marker
    and the annotation show. Drawing only the wall clock would make these
    cells look like ties.

    python3 plot_gantt_compare.py [--cells control_mix_hd,bimodal_hd,...]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.dirname(HERE)
SWEEPS = os.path.dirname(SWEEP)
XRT = os.path.join(SWEEPS, "qrb5165_sched_algo_sweep10_20260908-210226")

SURFACE = "#fcfcfb"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]        # categorical 1-3, fixed order
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
MARK = "#d03b3b"                                  # status: critical -- the objective

NODE_RE = re.compile(r"node\s+(\S+)\s+backend=(\S+)\s+inst=\s*(\d+)\s+period=\s*(\S+)")


def block(text: str, begin: str, end: str) -> str:
    i = text.index(begin) + len(begin)
    return text[i:text.index(end, i)]


def ros_header(path: str):
    """-> ({net: backend}, {aperiodic nets})"""
    place, aper = {}, set()
    for m in NODE_RE.finditer(open(path).read()):
        net, be, _, period = m.group(1), m.group(2), m.group(3), float(m.group(4))
        place[net] = be
        if period < 0:
            aper.add(net)
    return place, aper


# Both network names and the label separator use "_", so a label like
# `mlp_control_sd@dsp_dronet_se@dsp` cannot be split on "_" or scanned with a
# lazy name pattern -- the latter reads the backend as part of the next name
# ("dsp_dronet_se"). Anchor on the known backend tokens instead, and compare
# placements as SETS, since the label's ordering carries no information.
PAIR_RE = re.compile(r"([A-Za-z0-9_]+?)@(dsp|hta|cpu|gpu)(?=_[A-Za-z]|$)")


def parse_label(label: str) -> set:
    # The lookahead leaves the separating "_" on the front of each name after
    # the first; it is a separator, not part of the network name.
    return {(n.lstrip("_"), b) for n, b in PAIR_RE.findall(label)}


def find_ros_log(cell: str, want_label: str):
    """The log whose own header reproduces `want_label` -- see module docstring."""
    want = parse_label(want_label)
    if not want:
        return None, None, None
    for path in sorted(_glob_logs(cell)):
        place, aper = ros_header(path)
        if set(place.items()) == want:
            return path, place, aper
    return None, None, None


def _glob_logs(cell: str):
    import glob
    return glob.glob(os.path.join(SWEEP, "logs", f"{cell}__a*.log"))


def ros_trace(path: str):
    txt = open(path).read()
    rows = list(csv.DictReader(io.StringIO(
        block(txt, "ROS_PINSWEEP_TRACE_BEGIN ===", "=== ROS_PINSWEEP_TRACE_END").strip())))
    reps = {}
    for m in re.finditer(r"\[summary\] rep=(\d+).*?np_makespan=([\d.]+)", txt):
        reps[int(m.group(1))] = float(m.group(2))
    med = st.median(reps.values())
    rep = min(reps, key=lambda r: abs(reps[r] - med))          # the median rep
    out = [r for r in rows if int(r["rep"]) == rep and int(r["warm"]) == 0]
    return out, reps[rep]


def xrt_trace(tag: str, aper: set[str]):
    import glob
    for d in sorted(glob.glob(os.path.join(XRT, "runs", tag, "rep*"))):
        p = os.path.join(d, "run.log")
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        try:
            body = block(txt, "MODELBLASTER_XPURT_TRACE_BEGIN ===",
                         "=== MODELBLASTER_XPURT_TRACE_END").strip()
        except ValueError:
            continue
        rows = list(csv.DictReader(io.StringIO(body)))
        for r in rows:                       # cycles are microseconds (time_unit)
            r["_s"] = float(r["actual_start_cycles"]) / 1000.0
            r["_e"] = float(r["actual_end_cycles"]) / 1000.0
        np_end = max((r["_e"] for r in rows if r["network"] in aper), default=0.0)
        yield rows, np_end


def xrt_np_best_run(cell: str):
    """(run-dir tag, solver, recorded np median) for the solver the headline
    ratio actually uses.

    `xrt_best_solver` in analysis.json is the WALL-CLOCK winner, and the ratio
    is computed on the non-periodic makespan -- on depth_contended_quad those
    are different solvers (greedy 5.698 ms vs greedy_periodic 4.263 ms), so
    keying the gantt off `xrt_best_solver` drew a panel whose marker
    contradicted the number in the title. Pick the np-best solver that has a
    run directory, exactly as the ROS side picks the np-best placement.
    """
    p4 = json.load(open(os.path.join(XRT, "results", "phase4_results.json")))
    short = cell[len("networks_"):]
    best = None
    for r in p4:
        if r["workload"] != cell or r.get("measured_np_median_ms") is None:
            continue
        tag = r.get("duplicate_of") or f"{short}__{r['solver']}"
        if not os.path.isdir(os.path.join(XRT, "runs", tag)):
            continue
        if best is None or r["measured_np_median_ms"] < best[2]:
            best = (tag, r["solver"], r["measured_np_median_ms"])
    return best


def pick_median_rep(gen):
    runs = list(gen)
    if not runs:
        return None, None
    med = st.median([n for _, n in runs])
    return min(runs, key=lambda x: abs(x[1] - med))


def draw(ax, spans, rows, colors, aper, np_end, title, xmax):
    """`rows` is a list of (lane, network) -- ONE SUB-ROW PER NETWORK.

    A single row per lane hides real overlap: in the ROS trace `end_ms -
    start_ms` includes time the instance spent blocked on the lane, so two
    networks pinned to one backend produce spans that genuinely overlap, and
    whichever is drawn second covers the first. On bimodal_hd that painted the
    mlp instances over the yolov8 span and left the objective marker floating
    at 7.05 ms with no visible bar under it.
    """
    for yi, (lane, net) in enumerate(rows):
        for s, e, n in spans.get(lane, []):
            if n != net:
                continue
            ax.barh(yi, max(e - s, 0.02), left=s, height=0.5,
                    color=colors[net], edgecolor=SURFACE, linewidth=0.7,
                    zorder=3, hatch="///" if net in aper else None)
    ax.axvline(np_end, color=MARK, lw=1.6, ls="--", zorder=4)
    ax.annotate(f"aperiodic work done  {np_end:.2f} ms",
                xy=(np_end, len(rows) - 0.45), xytext=(6, 0),
                textcoords="offset points", fontsize=8.5, color=MARK,
                va="center", ha="left", zorder=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{l.upper()}  ·  {n}" for l, n in rows],
                       fontsize=8, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(-xmax * 0.012, xmax * 1.02)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6)


def one_cell(cell_short: str, rec: dict, out_dir: str):
    cell = rec["cell"]
    log, place, aper = find_ros_log(cell, rec["ros_np_best_label"])
    if log is None:
        print(f"  {cell_short}: no ROS log matches {rec['ros_np_best_label']}")
        return
    rrows, ros_np = ros_trace(log)
    pick = xrt_np_best_run(cell)
    if pick is None:
        print(f"  {cell_short}: no XPU-RT run directory with a measured np makespan")
        return
    tag, solver, recorded_np = pick
    xr = pick_median_rep(xrt_trace(tag, aper))
    if xr[0] is None:
        print(f"  {cell_short}: no parseable XPU-RT trace in runs/{tag}")
        return
    xrows, xrt_np = xr
    if abs(xrt_np - recorded_np) > 0.05:
        print(f"  {cell_short}: WARNING trace np {xrt_np:.3f} != recorded "
              f"{recorded_np:.3f} for {solver}")

    nets = []
    for r in rrows:
        if r["network"] not in nets:
            nets.append(r["network"])
    if len(nets) > len(SLOTS):
        print(f"  {cell_short}: {len(nets)} networks exceeds the validated "
              f"3-slot all-pairs cap; skipping")
        return
    colors = {n: SLOTS[i] for i, n in enumerate(nets)}

    rspan, xspan = {}, {}
    for r in rrows:
        rspan.setdefault(r["backend"], []).append(
            (float(r["start_ms"]), float(r["end_ms"]), r["network"]))
    for r in xrows:
        xspan.setdefault(r["backend"].lower(), []).append((r["_s"], r["_e"], r["network"]))
    # One sub-row per (lane, network) actually used by either side, so the two
    # panels share a row layout and overlap on a lane stays visible.
    used = {(lane, n) for span in (rspan, xspan)
            for lane, v in span.items() for _, _, n in v}
    rows_yx = sorted(used, key=lambda t: (t[0], nets.index(t[1])))
    xmax = max([e for v in list(rspan.values()) + list(xspan.values()) for _, e, _ in v])

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.4, 1.35 + 0.42 * len(rows_yx) * 2),
                                 sharex=True, gridspec_kw=dict(hspace=0.34))
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)

    draw(ax, xspan, rows_yx, colors, aper, xrt_np,
         f"XPU-RT — per-op scheduling, solver `{solver}`", xmax)
    # Render the placement from the parsed pairs. Replacing "_" with a space in
    # the raw label also split the network names ("mlp  control  sd@dsp").
    pretty = ",  ".join(f"{n}→{b}" for n, b in
                        sorted(parse_label(rec["ros_np_best_label"]),
                               key=lambda t: nets.index(t[0]) if t[0] in nets else 99))
    draw(bx, rspan, rows_yx, colors, aper, ros_np,
         f"ROS — whole-network pinning:  {pretty}", xmax)
    bx.set_xlabel("ms from the start of the run", fontsize=9.5, color=INK2)

    ratio = rec["ros_over_xrt_np_best"]
    fig.suptitle(f"{cell_short} — pinning finishes the aperiodic work in "
                 f"{1/ratio:.2f}× less time ({ros_np:.2f} vs {xrt_np:.2f} ms)",
                 fontsize=12.5, color=INK, x=0.006, ha="left", y=0.985)
    fig.legend(handles=[Patch(facecolor=colors[n], label=n) for n in nets]
               + [Patch(facecolor="#ffffff", edgecolor=MUTED, hatch="///",
                        label="aperiodic (the timed work)")],
               fontsize=8.5, frameon=False, labelcolor=INK2, ncol=len(nets) + 1,
               loc="upper left", bbox_to_anchor=(0.006, 0.945))
    fig.subplots_adjust(left=0.175, right=0.995, top=0.80, bottom=0.115)

    p = os.path.join(out_dir, f"gantt_{cell_short}.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> {p}   ros_np={ros_np:.2f}  xrt_np={xrt_np:.2f}  ratio={ratio:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="control_mix_hd,bimodal_hd,depth_contended_quad")
    ap.add_argument("--out", default=os.path.join(SWEEP, "plots"))
    a = ap.parse_args()
    d = json.load(open(os.path.join(SWEEP, "results", "analysis.json")))
    by = {c["cell"]: c for c in d["cells"]}
    os.makedirs(a.out, exist_ok=True)
    for short_name in a.cells.split(","):
        rec = by.get(f"networks_{short_name}")
        if rec is None:
            print(f"  {short_name}: not in analysis.json")
            continue
        one_cell(short_name, rec, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

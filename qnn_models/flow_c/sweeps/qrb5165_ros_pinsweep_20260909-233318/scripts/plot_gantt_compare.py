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

WHY SOME CELLS DO NOT GET THREE COLOURS
---------------------------------------
`SLOTS` is a CATEGORICAL palette of three, and three is where it stops because
that is how far the reference palette is validated on the all-pairs list. A
gantt places colours at arbitrary spatial positions, so the adjacent-pair rule
does not rescue a fourth slot, and a cell with more networks than slots used to
be skipped outright.

`scale_ladder_{hd,dc}` has six -- `dronet_sb` .. `dronet_sg` -- and skipping it
threw away the interesting part of the cell. Those six are NOT six unrelated
networks: they are ONE architecture at six sizes (23 IR ops each; 2.64, 3.98,
6.86, 13.06, 15.94, 21.26 M MACs), so they are ORDINAL, not categorical, and a
categorical palette was the wrong tool rather than a palette one slot too
short. They get a single-hue light-to-dark ramp keyed to size, which encodes
the ladder truthfully and needs no cap.

The ramp is not applied on faith. `ladder()` requires that every network in the
cell share one base name, that each carry a MAC count in its binding manifest,
and that **sorting by MACs reproduce sorting by rung letter**; if the family is
not monotone the ramp would assert an order the data does not have, so it falls
back to one flat colour and lets the y labels carry identity alone (they always
do -- see below). A cell with >3 networks that is not a ladder at all is still
skipped.

Identity is NEVER carried by colour alone, on any path: every bar sits in a
sub-row labelled `<LANE> · <network>`, and the legend spells out each rung with
its size. The ramp is redundant encoding, not the encoding.
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
# Sequential, one hue, light -> dark. For an ORDINAL family only (see the
# module docstring): six rungs of one architecture, ordered by size. Starts
# light enough to read as a series and dark enough that the first step is not
# lost against SURFACE.
RAMP = ["#d3e3f8", "#a9c7ee", "#7fa9e2", "#5589d4", "#2f68bd", "#123f8f"]
FLAT = "#3d7ab8"                                  # ordinal family, unordered
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
MARK = "#d03b3b"                                  # status: critical -- the objective
DEAD = "#f0efec"   # pre-first-dispatch dead time, the neutral midpoint

NODE_RE = re.compile(r"node\s+(\S+)\s+backend=(\S+)\s+inst=\s*(\d+)\s+period=\s*(\S+)")

RUNG_RE = re.compile(r"^(?P<base>.+)_s(?P<rung>[a-z])$")
MACS_RE = re.compile(r"([\d,]+)\s+MACs")


def macs(net):
    """The network's MAC count, from the binding manifest phase1 wrote it into.

    That manifest is generated from the board's own compose verdict, so the
    number is a property of the model that actually ran -- not a guess from the
    name, which is what a ramp keyed to `sb..sg` alone would be.
    """
    p = os.path.join(XRT, "bindings", f"{net}.json")
    if not os.path.exists(p):
        return None
    m = MACS_RE.search(json.load(open(p)).get("_comment") or "")
    return int(m.group(1).replace(",", "")) if m else None


def ladder(nets):
    """-> {net: rank} over an ORDINAL family, or None.

    Returns a rank per network when all of `nets` are rungs of one ladder
    (same base, distinct rung letters, every rung carrying a MAC count) AND the
    MAC order agrees with the rung order. The agreement check is the point: the
    ramp asserts "darker is bigger", and asserting that over a family whose
    sizes do not follow its own names would be a figure that lies. When the
    family is recognised but not monotone the caller is told (rank None), and
    falls back to one flat colour with the labels carrying identity.
    """
    ms = [RUNG_RE.match(n) for n in nets]
    if not all(ms) or len({m.group("base") for m in ms}) != 1:
        return None
    rungs = [m.group("rung") for m in ms]
    if len(set(rungs)) != len(rungs) or len(nets) > len(RAMP):
        return None
    sizes = {n: macs(n) for n in nets}
    if any(v is None for v in sizes.values()):
        return None
    by_rung = [n for _, n in sorted(zip(rungs, nets))]
    by_size = sorted(nets, key=lambda n: sizes[n])
    if by_rung != by_size:
        return {"order": by_rung, "sizes": sizes, "monotone": False}
    return {"order": by_size, "sizes": sizes, "monotone": True}


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


def xrt_np_best_run(cell: str, prefer: str | None = None):
    """(run-dir tag, solver, recorded np median, is_preferred).

    `prefer` names the solver the study actually recommends (cpsat:warmbest).
    Comparing pinning against `greedy` understates XPU-RT badly -- on the
    all-solver cells greedy is up to 3.5x worse than cpsat:warmbest
    (saturation_dc 22.48 vs 6.38 ms) -- so a gantt drawn against greedy is not
    the comparison anyone should draw a conclusion from. Only 12 of 42 cells
    were measured with all twelve solvers, so on the rest the preferred solver
    simply does not exist as a measurement; the caller is told which it got.

    `xrt_best_solver` in analysis.json is the WALL-CLOCK winner, and the ratio
    is computed on the non-periodic makespan -- on depth_contended_quad those
    are different solvers (greedy 5.698 ms vs greedy_periodic 4.263 ms), so
    keying the gantt off `xrt_best_solver` drew a panel whose marker
    contradicted the number in the title. Pick the np-best solver that has a
    run directory, exactly as the ROS side picks the np-best placement.
    """
    p4 = json.load(open(os.path.join(XRT, "results", "phase4_results.json")))
    short = cell[len("networks_"):]

    # `duplicate_of` CHAINS, and a tag renders ":" as "-". On
    # depth_contended_hd, cpsat:warmbest -> best-of-fast -> cpsat-warm ->
    # greedy_periodic, and only the last is a real run directory. Following a
    # single hop found no directory and silently dropped the preferred solver,
    # which then fell back to whichever solver happened to be fastest -- the
    # exact substitution this function exists to prevent.
    rows = {r["solver"]: r for r in p4
            if r["workload"] == cell and r.get("measured_np_median_ms") is not None}
    tag_of = {f"{short}__{sv.replace(':', '-')}": sv for sv in rows}

    def resolve(sv, seen=()):
        r = rows.get(sv)
        if r is None or sv in seen:
            return None
        own = f"{short}__{sv.replace(':', '-')}"
        if os.path.isdir(os.path.join(XRT, "runs", own)):
            return own
        nxt = r.get("duplicate_of")
        if not nxt:
            return None
        if os.path.isdir(os.path.join(XRT, "runs", nxt)):
            return nxt
        return resolve(tag_of.get(nxt), seen + (sv,))

    cands = []
    for sv, r in rows.items():
        tag = resolve(sv)
        if tag:
            cands.append((tag, sv, r["measured_np_median_ms"]))
    if not cands:
        return None
    for tag, sv, np_ms in cands:
        if sv == prefer:
            return (tag, sv, np_ms, True)
    tag, sv, np_ms = min(cands, key=lambda c: c[2])
    return (tag, sv, np_ms, False)


def pick_median_rep(gen):
    runs = list(gen)
    if not runs:
        return None, None
    med = st.median([n for _, n in runs])
    return min(runs, key=lambda x: abs(x[1] - med))


def draw(ax, spans, rows, colors, hatch_nets, np_end, title, xmax,
         first=None):
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
                    zorder=3, hatch="///" if net in hatch_nets else None)
    # THE START BARRIER, drawn rather than described. Both runtimes are timed
    # from their own t0, and on some runs the first dispatch is over a
    # millisecond later than that -- on the XPU-RT side the wait sits in
    # `gate_ms`, not in `dep_wait_ms`, so nothing was blocked on a dependency.
    # Shading it makes the interval the corrected number measures visible as
    # the part of the panel with work in it, and stops this figure from
    # illustrating a startup artifact as though it were a scheduling
    # difference.
    if first is not None and first > 0.02:
        ax.axvspan(0.0, first, color=DEAD, zorder=1)
        ax.axvline(first, color=MUTED, lw=1.0, ls=":", zorder=4)
        ax.annotate(f"first dispatch {first:.2f} ms", xy=(first, -0.52),
                    xytext=(4, 0), textcoords="offset points", fontsize=7.8,
                    color=MUTED, va="center", ha="left", zorder=5)
    ax.axvline(np_end, color=MARK, lw=1.6, ls="--", zorder=4)
    # The two panels share an x range, so on a cell where pinning is much
    # slower its own marker sits at the right edge and a right-flowing label
    # runs off the figure (vint_multi_cg printed "ape"). Flip the label to the
    # inside of the marker when it is near the end of the axis.
    late = np_end > 0.72 * xmax
    corr = (f"  ({np_end - first:.2f} ms from first dispatch)"
            if first is not None and first > 0.02 else "")
    # Keep the label INSIDE the axes: at the old y it sat on the axis edge and
    # ran into the next panel's title on any cell with only two sub-rows.
    ax.annotate(f"aperiodic work done  {np_end:.2f} ms" + corr,
                xy=(np_end, len(rows) - 0.72), xytext=(-6 if late else 6, 0),
                textcoords="offset points", fontsize=8.5, color=MARK,
                va="center", ha="right" if late else "left", zorder=6,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0,
                          alpha=0.85))
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


def one_cell(cell_short: str, rec: dict, out_dir: str, prefer: str,
             baseline: str = "isolation"):
    cell = rec["cell"]
    # WHICH PINNING PLACEMENT THE PANEL SHOWS has to be the one the headline
    # ratio is computed on. `isolation` is the primary baseline -- each
    # network on the lane it is fastest on alone, which is what a ROS user
    # deploys; `oracle` is the best of every legal placement measured, which
    # is an upper bound nobody has. Drawing one beside a ratio computed from
    # the other would be showing a different experiment.
    LKEY = "iso_label" if baseline == "isolation" else "ros_np_best_label"
    label = rec.get(LKEY) or rec["ros_np_best_label"]
    log, place, aper = find_ros_log(cell, label)
    if log is None:
        print(f"  {cell_short}: no ROS log matches {label}")
        return
    rrows, ros_np = ros_trace(log)
    pick = xrt_np_best_run(cell, prefer)
    if pick is None:
        print(f"  {cell_short}: no XPU-RT run directory with a measured np makespan")
        return
    tag, solver, recorded_np, preferred = pick
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
    lad = None
    if len(nets) > len(SLOTS):
        lad = ladder(nets)
        if lad is None:
            print(f"  {cell_short}: {len(nets)} networks exceeds the validated "
                  f"3-slot all-pairs cap and they are not one ordinal family; "
                  f"skipping")
            return
    if lad and lad["monotone"]:
        nets = lad["order"]                       # draw the ladder in size order
        colors = {n: RAMP[i] for i, n in enumerate(nets)}
    elif lad:
        nets = lad["order"]
        colors = {n: FLAT for n in nets}
        print(f"  {cell_short}: rung order and size order disagree; one flat "
              f"colour, labels carry identity")
    else:
        colors = {n: SLOTS[i] for i, n in enumerate(nets)}

    rspan, xspan = {}, {}
    for r in rrows:
        rspan.setdefault(r["backend"], []).append(
            (float(r["start_ms"]), float(r["end_ms"]), r["network"]))
    for r in xrows:
        xspan.setdefault(r["backend"].lower(), []).append((r["_s"], r["_e"], r["network"]))
    # The first dispatch anywhere in each panel's run: the origin the corrected
    # numbers are measured from.
    ros_first = min((float(r["start_ms"]) for r in rrows), default=0.0)
    xrt_first = min((r["_s"] for r in xrows), default=0.0)
    # One sub-row per (lane, network) actually used by either side, so the two
    # panels share a row layout and overlap on a lane stays visible.
    used = {(lane, n) for span in (rspan, xspan)
            for lane, v in span.items() for _, _, n in v}
    rows_yx = sorted(used, key=lambda t: (t[0], nets.index(t[1])))
    xmax = max([e for v in list(rspan.values()) + list(xspan.values()) for _, e, _ in v])

    # The header is three stacked text blocks -- title, the raw-vs-corrected
    # line, and the legend -- so its height is fixed in INCHES and the axes
    # get whatever is left. Expressing it as a fraction of the figure (the
    # earlier `top=0.80`) made the header shrink with the panel and the title
    # ran into the legend on any cell with only two sub-rows.
    HEAD_IN = 1.60
    fig_h = HEAD_IN + 0.30 + 0.42 * len(rows_yx) * 2
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12.4, fig_h),
                                 sharex=True, gridspec_kw=dict(hspace=0.42))
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)

    # The hatch separates the timed work from the rest. On a cell where EVERY
    # network is aperiodic it separates nothing, and six hatched bars fight the
    # ramp for the eye, so it is dropped and the caption says so instead.
    all_aper = set(nets) <= set(aper)
    hatch_nets = set() if all_aper else set(aper)
    draw(ax, xspan, rows_yx, colors, hatch_nets, xrt_np,
         f"XPU-RT — per-op scheduling, solver `{solver}`"
         + ("" if preferred else
            f"   ({prefer} was not measured on this cell — winner+greedy tier)"),
         xmax, first=xrt_first)
    # Render the placement from the parsed pairs. Replacing "_" with a space in
    # the raw label also split the network names ("mlp  control  sd@dsp").
    pretty = ",  ".join(f"{n}→{b}" for n, b in
                        sorted(parse_label(label),
                               key=lambda t: nets.index(t[0]) if t[0] in nets else 99))
    draw(bx, rspan, rows_yx, colors, hatch_nets, ros_np,
         ("ROS — whole-network pinning, each network on its "
          "isolation-best lane:  " if baseline == "isolation"
          else "ROS — whole-network pinning, best measured placement "
               "(oracle):  ") + pretty, xmax, first=ros_first)
    bx.set_xlabel("ms from the start of the run", fontsize=9.5, color=INK2)

    # Compute the headline from the two numbers actually drawn. Using the
    # recorded `ros_over_xrt_np_best` is wrong here: that ratio is against the
    # BEST solver over all twelve, while these panels show a chosen one, so on
    # depth_contended_hd the title claimed 1.02 over a panel showing 3.36 vs
    # 3.82. A title must not be able to disagree with its own figure.
    ratio = ros_np / xrt_np if xrt_np else float("nan")
    # The headline is the CORRECTED ratio: both panels re-timed from their own
    # first dispatch, so neither runtime is charged for the other's startup.
    # The raw ratio is printed beside it and never replaced -- it is what the
    # earlier write-ups quote, and a reader has to be able to see what the
    # correction did.
    cros, cxrt = ros_np - ros_first, xrt_np - xrt_first
    cratio = cros / cxrt if cxrt else float("nan")
    verb = ("pinning finishes the aperiodic work in "
            f"{1/cratio:.2f}× less time" if cratio < 1 else
            f"scheduling finishes the aperiodic work in {cratio:.2f}× less time")
    moved = abs(cratio / ratio - 1) * 100 if ratio else 0.0
    fig.suptitle(f"{cell_short} — {verb}\n"
                 f"{cros:.2f} vs {cxrt:.2f} ms, each from its own first dispatch",
                 fontsize=11.5, color=INK, x=0.006, ha="left",
                 y=1 - 0.16 / fig_h, va="top", linespacing=1.35)
    fig.text(0.006, 1 - 0.78 / fig_h,
             f"Raw, from each runtime's t0: {ros_np:.2f} vs {xrt_np:.2f} ms = "
             f"{ratio:.2f}×.  Corrected: {cratio:.2f}×"
             + (f" — the start barrier was worth {moved:.0f} % of this cell's "
                f"ratio." if moved >= 1 else "; the start barrier moves it by "
                                             "less than 1 %."),
             fontsize=8.8, color=MUTED, ha="left", va="top")
    # A ramp has to be decodable, so each rung's legend entry carries its size:
    # the colour says "bigger", the label says how much bigger.
    if lad:
        labels = [f'{n}  ({lad["sizes"][n] / 1e6:.1f} M MACs)' for n in nets]
    else:
        labels = list(nets)
    handles = [Patch(facecolor=colors[n], label=l) for n, l in zip(nets, labels)]
    if not all_aper:
        handles.append(Patch(facecolor="#ffffff", edgecolor=MUTED, hatch="///",
                             label="aperiodic (the timed work)"))
    else:
        handles.append(Patch(facecolor=SURFACE, edgecolor=SURFACE,
                             label="every network here is aperiodic — all of "
                                   "it is the timed work"))
    fig.legend(handles=handles, fontsize=8.5, frameon=False, labelcolor=INK2,
               ncol=min(len(handles), 4), loc="upper left",
               bbox_to_anchor=(0.006, 1 - 1.10 / fig_h))
    fig.subplots_adjust(left=0.175, right=0.995,
                        top=1 - HEAD_IN / fig_h, bottom=0.62 / fig_h)

    # The primary baseline keeps the plain name; the oracle reading gets its
    # own file rather than overwriting it, because the two panels show
    # different placements and a reader must not get one while quoting the
    # other's number.
    p = os.path.join(out_dir, f"gantt_{cell_short}"
                     + ("" if baseline == "isolation" else "_oracle") + ".png")
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> {p}   raw {ros_np:.2f}/{xrt_np:.2f} = {ratio:.3f}   "
          f"corrected {cros:.2f}/{cxrt:.2f} = {cratio:.3f}   "
          f"(first dispatch ros {ros_first:.3f} / xrt {xrt_first:.3f} ms)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="control_mix_hd,bimodal_hd,depth_contended_quad")
    ap.add_argument("--out", default=os.path.join(SWEEP, "plots"))
    ap.add_argument("--prefer", default="cpsat:warmbest",
                    help="solver to compare against when it was measured")
    ap.add_argument("--baseline", default="isolation",
                    choices=["isolation", "oracle"],
                    help="which pinning placement the ROS panel draws: the "
                         "isolation-best one a user deploys (the default and "
                         "the headline), or the best measured placement (an "
                         "oracle, the secondary reading).")
    a = ap.parse_args()
    d = json.load(open(os.path.join(SWEEP, "results", "analysis.json")))
    by = {c["cell"]: c for c in d["cells"]}
    os.makedirs(a.out, exist_ok=True)
    for short_name in a.cells.split(","):
        rec = by.get(f"networks_{short_name}")
        if rec is None:
            print(f"  {short_name}: not in analysis.json")
            continue
        one_cell(short_name, rec, a.out, a.prefer, a.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

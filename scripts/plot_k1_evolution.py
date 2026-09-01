#!/usr/bin/env python3
"""Figures for the K1 closed loop: how the schedule changes when the scheduler
feeds evidence back to the compiler.

Everything plotted here is MEASURED on the board. Predicted values appear only
where they are labelled as such and drawn in a muted style, because the whole
point of the ladder is that a rung which helps in the model may not help on the
hardware -- two of the three feedback actions this project tried were rejected
exactly there.

Panels are built at final print size (Nature single/double column) so nothing is
rescaled afterwards.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle  # noqa: E402

# The print rcParams and the palette live in `figstyle` because they were
# copy-pasted into five renderers and drifted: DroNet was blue in one figure
# and orange in another, and yolov8_nano was blue in that one. Colour is an
# identity claim, so it is made once.
figstyle.use()
MM = figstyle.MM
SINGLE_COL = figstyle.SINGLE_COL
DOUBLE_COL = figstyle.DOUBLE_COL

# Okabe-Ito. DroNet is the model under study, so it takes the strong colour;
# MLP is the well-behaved co-runner and is muted.
C_DRONET = figstyle.model_color("dronet")
C_MLP = figstyle.model_color("mlp_control")
C_DEADLINE = figstyle.C_DEADLINE
C_MUTED = "#999999"
C_DARK = "#333333"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERIODS = {"mlp": 10.0, "dronet": 33.3}
CORES = [f"CPU_P#{i}" for i in range(4)] + [f"CPU_E#{i}" for i in range(4)]


#: Which toolchain produced a trace. This matters because the two are not
#: comparable and the figures do not say which they are showing.
#:
#: Every trace under artifacts/k1_run/baselines/ is IREE-era: module names read
#: `mlp$async_dispatch_0_embedded_elf_riscv_64_...` and each row carries a
#: `vmfb_path`. The project has since dropped the IREE path entirely and moved
#: to ModelBlaster, whose module names read
#: `mlp_control$dispatch_0_rvv_x60_linear_s8_M1xK16xN256` and whose traces
#: carry `worker_hart` / `actual_start_cycles` instead.
#:
#: The historical figure this file produced was captioned "B4 + sharding,
#: after the scheduler fed evidence back" and drawn from IREE measurements.
#: At that time ModelBlaster's `parallel_conv2d_s8` sliced IHWOC-packed weights
#: with an OIHW offset formula, so the claimed ModelBlaster sharding path had
#: not actually run. ModelBlaster now repacks conv weights for the exact width
#: selected by the schedule, and the exact-cycle experiment exercises that path
#: on K1. Keep the historical provenance warning: the old IREE trace still
#: cannot be relabelled as ModelBlaster evidence.
#:
#: Detecting provenance and stamping it on the output is the cheap half of the
#: fix. The other half -- a ModelBlaster rung ladder -- needs a schema adapter,
#: because these two traces share no column names; `plot_k1_trace_gantt.py`
#: already has `_normalise_modelblaster` for that.
IREE_ERA = "iree"
MODELBLASTER = "modelblaster"

#: Set by main() before any figure is built; read by the save sites so every
#: figure carries its own provenance rather than relying on the caller.
_PROVENANCE = "unknown"


def trace_provenance(rows) -> str:
    """`iree` or `modelblaster`, from the row schema itself."""
    if not rows:
        return "empty"
    r = rows[0]
    if "vmfb_path" in r or "$async_dispatch" in (r.get("module_name") or ""):
        return IREE_ERA
    if "worker_hart" in r or "actual_start_cycles" in r:
        return MODELBLASTER
    return "unknown"


def load_trace(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_schedule(path):
    with open(path) as f:
        return json.load(f)["dispatches"]


def model_of(job_name, known=None):
    """`job_names` owns this split; see that module for why it is not a
    trailing-digit strip."""
    return figstyle.model_of(job_name, known)


def instance_stats(rows):
    """Per periodic instance: span, lateness, response time.

    Response time is measured from the nominal release k*T, not from whenever
    the instance happened to start, because a model does not meet its frequency
    by running several invocations back to back.
    """
    inst = collections.defaultdict(lambda: [1e18, -1e18])
    for r in rows:
        j = r["job_name"]
        inst[j][0] = min(inst[j][0], float(r["start_us"]) / 1000.0)
        inst[j][1] = max(inst[j][1], float(r["end_us"]) / 1000.0)
    per = collections.defaultdict(
        lambda: {"n": 0, "miss": 0, "late": [], "resp": []})
    for job, (st, en) in inst.items():
        m = model_of(job)
        T = PERIODS.get(m)
        if T is None:
            continue
        i = int(job[len(m):] or 0)
        d = per[m]
        d["n"] += 1
        d["resp"].append(en - i * T)
        late = en - (i * T + T)
        if late > 0:
            d["miss"] += 1
            d["late"].append(late)
    return per


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


# ------------------------------------------------------- reusable Gantt render
# One renderer, parameterised. This block used to be the body of figure1() with
# the rungs, the two model colours, the eight K1 cores and the 140 ms window all
# hardcoded, so anything else that needed a core-lane Gantt wrote its own -- and
# by now `xpu-rt/plot_gantt.py` (terminal) and the retired merlin
# `analysis/plot_dispatch_trace.py` already disagree with it about the schema.
# The policy sweep needs exactly this picture per solver, so the picture became
# a function and figure1 became its first caller. Nothing about the published
# figure changed: same lanes, same window, same styling.

#: Okabe-Ito, minus the two already spoken for below.
PALETTE = [figstyle.GREEN, figstyle.PURPLE, figstyle.SKY,
           figstyle.YELLOW, figstyle.VERMILLION, figstyle.BLACK]


def model_colours(models):
    """Stable model -> colour map. DroNet and MLP keep the published colours."""
    fixed = {"dronet": C_DRONET, "mlp": C_MLP}
    out, spare = {}, list(PALETTE)
    for m in sorted(models):
        base = m
        if base in fixed:
            out[m] = fixed[base]
        else:
            out[m] = spare[len(out) % len(spare)]
    return out


def cores_from_schedule(sched):
    """Lane order from a schedule's own hardware targets, cluster-major.

    Sorting is (cluster, numeric core) rather than lexicographic so CPU_P#10
    lands after CPU_P#9 instead of after CPU_P#1.
    """
    seen = set()
    for ent in sched.values():
        for c in str(ent.get("hardware_target", "")).split("+"):
            if c:
                seen.add(c)

    def key(c):
        cluster, _, idx = c.partition("#")
        return (cluster, int(idx) if idx.isdigit() else 0)

    return sorted(seen, key=key)


def held_lane_span(hardware_target, cores):
    """Return the inclusive physical-lane span held by one dispatch.

    Scheduler combinations are contiguous and within one cluster, so a single
    rectangle across this span faithfully represents the runtime core lock.
    """
    lanes = [cores.index(c) for c in str(hardware_target).split("+")
             if c in cores]
    return (min(lanes), max(lanes)) if lanes else None


def draw_gantt_axis(ax, rows, sched, *, cores, window_ms, colours,
                    periods=None, deadline_model=None, known=None,
                    impl_hatch=False, cluster_boundary=True,
                    repeat_frame=False):
    """Draw one schedule on an existing axis using physical K1 core lanes."""
    periods = periods or {}
    seen_models = set()
    for r in rows:
        ent = sched.get(r["dispatch_key"])
        if ent is None:
            continue
        start = float(r["start_us"]) / 1000.0
        dur = float(r["run_us"]) / 1000.0
        if start >= window_ms:
            continue
        span = held_lane_span(ent.get("hardware_target", ""), cores)
        if span is None:
            continue
        m = model_of(r["job_name"], known)
        seen_models.add(m)
        lo, hi = span
        is_ime = ent.get("impl") == "ime" and any(
            op in (ent.get("module_name") or "")
            for op in ("linear_s8", "matmul_s8"))
        ax.broken_barh(
            [(start, max(dur, 0.15))], (lo - 0.42, (hi - lo) + 0.84),
            facecolors=colours.get(m, C_MUTED),
            edgecolors=("black" if is_ime and impl_hatch else "white"),
            linewidth=(0.35 if is_ime and impl_hatch else 0.15),
            hatch=("///" if is_ime and impl_hatch else None))
    if deadline_model and periods.get(deadline_model):
        period = periods[deadline_model]
        k = 0
        while k * period <= window_ms:
            ax.axvline(k * period, color=C_DEADLINE, lw=0.4,
                       ls=(0, (2, 2)), zorder=0)
            k += 1
    ax.set_yticks(range(len(cores)))
    ax.set_yticklabels(cores, fontsize=4.2)
    ax.set_ylim(-0.7, len(cores) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, window_ms)
    if repeat_frame:
        ax.axvline(window_ms, color=C_DARK, lw=0.9, ls=(0, (3, 2)),
                   zorder=8, clip_on=False)
        ax.text(0.995, 0.97, f"\u21bb repeat every {window_ms:g} ms",
                transform=ax.transAxes, ha="right", va="top", fontsize=4.6,
                color=C_DARK,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.82, "pad": 1.0})
    ax.spines[["top", "right"]].set_visible(False)
    if cluster_boundary:
        for i in range(1, len(cores)):
            if cores[i].split("#")[0] != cores[i - 1].split("#")[0]:
                ax.axhline(i - 0.5, color="0.75", lw=0.4)
    return seen_models


def render_gantt_panels(panels, out, *, periods, cores=None, window_ms=140.0,
                        deadline_model=None, colours=None,
                        xlabel="Time on the K1 (ms)", width=DOUBLE_COL,
                        model_labels=None,
                        panel_height_mm=26.0, panel_labels=True,
                        cluster_boundary=True, legend=True,
                        repeat_frame=False):
    """Gantt with physical cores as lanes, one panel per entry in `panels`.

    `panels` is a sequence of dicts with keys ``title``, ``rows`` (trace rows,
    measured or rendered from a schedule by `xpu-rt/schedule_trace.py`) and
    ``sched`` (that run's ``dispatches`` map, for the core assignment).

    A sharded dispatch is drawn as ONE bar spanning every core it holds, which
    is what the runtime actually does -- the core lock in scheduler_runner.cc
    keeps the other cores idle for its duration. Drawing it as four independent
    bars would show a machine that does not exist.

    Writes ``out + '.png'`` and ``out + '.pdf'`` and returns both paths.
    """
    panels = list(panels)
    if not panels:
        raise ValueError("render_gantt_panels: no panels")
    if cores is None:
        cores = cores_from_schedule(panels[0]["sched"])
    if not cores:
        raise ValueError("render_gantt_panels: no core lanes could be resolved")
    if colours is None:
        colours = model_colours({model_of(r["job_name"])
                                 for p in panels for r in p["rows"]})

    height = max(panel_height_mm * len(panels), 28.0) * MM
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, height),
                             sharex=True, squeeze=False)
    axes = list(axes[:, 0])
    seen_models = set()
    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        sched = panel["sched"]
        seen_models.update(draw_gantt_axis(
            ax, panel["rows"], sched, cores=cores, window_ms=window_ms,
            colours=colours, periods=periods,
            deadline_model=deadline_model,
            cluster_boundary=cluster_boundary,
            repeat_frame=repeat_frame))
        ax.set_title(panel["title"], loc="left", pad=2)
        if panel_labels:
            lab = chr(ord("a") + idx) if isinstance(panel_labels, bool) \
                else panel_labels[idx]
            ax.text(-0.062, 1.16, lab, transform=ax.transAxes, fontsize=8,
                    fontweight="bold", va="top", ha="right")
    axes[-1].set_xlabel(xlabel)
    if legend:
        names = model_labels or {}
        handles = [Patch(facecolor=colours.get(m, C_MUTED),
                         label=(f"{names.get(m, m)} ({periods[m]:g} ms period)"
                                if periods.get(m) else names.get(m, m)))
                   for m in sorted(seen_models)]
        if deadline_model and periods.get(deadline_model):
            handles.append(plt.Line2D([], [], color=C_DEADLINE, lw=0.6,
                                      ls=(0, (2, 2)),
                                      label=f"{names.get(deadline_model, deadline_model)}"
                                            f" deadlines"))
        axes[0].legend(handles=handles, ncol=max(1, len(handles)), frameon=False,
                       loc="lower left", bbox_to_anchor=(0, 1.18))
    fig.tight_layout(rect=(0.01, 0, 1, 0.97))
    _stamp(fig, _PROVENANCE)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", out + ".pdf/.png")
    return out + ".png", out + ".pdf"


# ---------------------------------------------------------------- figure 1
def figure1(rungs, out):
    """Measured Gantt, one panel per rung, lanes = physical cores."""
    panels = [{"title": title,
               "rows": load_trace(trace_p),
               "sched": load_schedule(sched_p)}
              for _tag, title, trace_p, sched_p in rungs]
    return render_gantt_panels(panels, out, periods=PERIODS, cores=CORES,
                               window_ms=140.0, deadline_model="dronet",
                               colours={"dronet": C_DRONET, "mlp": C_MLP},
                               model_labels={"dronet": "DroNet", "mlp": "MLP"})


# ---------------------------------------------------------------- figure 2
def figure2(rungs, out):
    """What each feedback action actually bought, measured.

    Worst-case lateness is on a log axis because the ladder spans three orders
    of magnitude; a linear axis would collapse everything after B0 into one
    indistinguishable band.
    """
    tags, worst, resp50, queue, miss_rate = [], [], [], [], []
    for tag, title, trace_p, _ in rungs:
        rows = load_trace(trace_p)
        per = instance_stats(rows)
        d = per["dronet"]
        tags.append(tag)
        worst.append(max(d["late"]) if d["late"] else 0.0)
        resp50.append(pct(d["resp"], 50))
        miss_rate.append(100.0 * d["miss"] / d["n"] if d["n"] else 0.0)
        s = sum(float(r["run_us"]) for r in rows)
        q = sum(float(r["queue_delay_us"]) for r in rows)
        queue.append(100.0 * q / (s + q) if (s + q) else 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 52 * MM))
    x = range(len(tags))

    # Dots with stems rather than bars: these two axes are logarithmic, and a
    # bar drawn on a log axis implies an area that has no meaning because the
    # baseline is arbitrary. The dot marks the value; the stem only guides
    # the eye down to the category.
    def stem(ax, vals, fmt="{:.0f}"):
        cols = [C_MUTED] * (len(vals) - 1) + [C_DRONET]
        lo = min(vals) / 3.0
        for i, v in enumerate(vals):
            ax.vlines(i, lo, v, color=cols[i], lw=0.8, alpha=0.55)
            ax.plot(i, v, "o", color=cols[i], ms=4.5)
            ax.text(i, v * 1.22, fmt.format(v), ha="center", fontsize=5)
        ax.set_ylim(lo, max(vals) * 2.4)

    ax = axes[0]
    stem(ax, worst)
    ax.set_yscale("log")
    ax.set_ylabel("DroNet worst-case lateness (ms)")
    ax.set_title("Lateness falls 32x", loc="left", pad=2)

    ax = axes[1]
    stem(ax, resp50)
    ax.axhline(PERIODS["dronet"], color=C_DEADLINE, lw=0.8, ls=(0, (2, 2)))
    ax.text(len(tags) - 0.45, PERIODS["dronet"] * 0.72, "33.3 ms deadline",
            color=C_DEADLINE, fontsize=4.6, ha="right", va="top")
    ax.set_yscale("log")
    ax.set_ylabel("DroNet median response time (ms)")
    ax.set_title("but never crosses the deadline", loc="left", pad=2)

    ax = axes[2]
    ax.bar(x, queue, color=[C_MUTED] * (len(tags) - 1) + [C_DRONET], width=0.62)
    ax.set_ylabel("Time spent queueing (%)")
    for i, v in enumerate(queue):
        ax.text(i, v + 1.6, f"{v:.1f}", ha="center", fontsize=5)
    ax.set_title("and queueing returns", loc="left", pad=2)

    for lab, ax in zip("abc", axes):
        ax.set_xticks(list(x))
        ax.set_xticklabels([t for t, _, _, _ in rungs], fontsize=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.22, 1.06, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")
    fig.tight_layout()
    _stamp(fig, _PROVENANCE)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", out + ".pdf/.png")


# ---------------------------------------------------------------- figure 3
def figure3(out):
    """The evidence the advisor acted on, and the limit it ran into.

    Left: measured per-dispatch scaling. This is what makes 'shard' a legal
    recommendation rather than a guess -- IREE really does distribute these
    convolutions, and the advisor has no business recommending it otherwise.
    Right: DroNet's whole-instance service time against its own period.
    """
    def totals(hw, model, topo):
        p = os.path.join(REPO, "gen", "profile", hw, "spacemit_x60", model,
                         f"{model}.q.int8", topo, "results.csv")
        with open(p) as f:
            rows = list(csv.DictReader(f))
        return {int(r["dispatch_id"]): float(r["mean_time_ns"]) / 1e6
                for r in rows}

    topos = [("topo_0", 1), ("topo_0_1", 2), ("topo_0_1_2_3", 4),
             ("topo_0_1_2_3_4_5_6_7", 8)]
    dronet = {n: totals("RVV", "dronet", t) for t, n in topos}
    mlp = {n: totals("RVV", "mlp", t) for t, n in topos}

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 55 * MM))

    ax = axes[0]
    heavy = sorted(dronet[1], key=lambda k: -dronet[1][k])[:5]
    ns = [n for _, n in topos]
    # Label only the extremes of the bundle; five labels at 8 cores land within
    # 1.5 ms of each other and collide into an unreadable stack.
    labelled = {heavy[0], heavy[-1]}
    for did in heavy:
        ax.plot(ns, [dronet[n][did] for n in ns], marker="o",
                color=C_DRONET, alpha=0.85)
        if did in labelled:
            ax.annotate(f"dispatch {did}", (ns[-1], dronet[ns[-1]][did]),
                        textcoords="offset points", xytext=(4, 0),
                        fontsize=4.6, color=C_DRONET, va="center")
    # Park the bundle label in the open wedge under the DroNet lines. Anchoring
    # it to a data point put it straight through them.
    ax.text(1.15, 1.9, "5 heaviest\nDroNet dispatches", fontsize=4.6,
            color=C_DRONET, va="center")
    mlp_tot = [sum(mlp[n].values()) for n in ns]
    ax.plot(ns, mlp_tot, marker="s", color=C_MLP, label="MLP (whole model)")
    ax.annotate("MLP", (ns[-1], mlp_tot[-1]), textcoords="offset points",
                xytext=(3, 0), fontsize=4.6, color=C_MLP, va="center")
    ideal = [dronet[1][heavy[0]] / n for n in ns]
    ax.plot(ns, ideal, color="0.55", lw=0.7, ls=(0, (3, 2)))
    ax.annotate("ideal 1/N", (ns[1], ideal[1]), textcoords="offset points",
                xytext=(0, 5), fontsize=4.6, color="0.45", ha="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Cores given to one dispatch")
    ax.set_ylabel("Measured service time (ms)")
    ax.set_title("DroNet's convolutions shard; the MLP does not",
                 loc="left", pad=6)
    ax.set_xlim(0.9, 12.0)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    tot = [sum(dronet[n].values()) for n in ns]
    cols = [C_MUTED if t > PERIODS["dronet"] else C_DRONET for t in tot]
    ax.bar(range(len(ns)), tot, color=cols, width=0.6)
    ax.axhline(PERIODS["dronet"], color=C_DEADLINE, lw=0.9, ls=(0, (2, 2)))
    ax.text(len(ns) - 0.4, PERIODS["dronet"] * 1.3, "33.3 ms period",
            color=C_DEADLINE, fontsize=5, ha="right")
    for i, t in enumerate(tot):
        ax.text(i, t * 1.06, f"{t:.1f}", ha="center", fontsize=5)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"{n} core{'s' if n > 1 else ''}" for n in ns],
                       fontsize=5)
    ax.set_ylabel("DroNet service time per instance (ms)")
    ax.set_ylim(0, max(tot) * 1.18)
    ax.set_title("4 cores is the first that fits \u2014 by 2.7%",
                 loc="left", pad=2)
    ax.spines[["top", "right"]].set_visible(False)

    for lab, ax in zip("ab", axes):
        ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")
    fig.tight_layout()
    _stamp(fig, _PROVENANCE)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", out + ".pdf/.png")


def _stamp(fig, provenance):
    """Say on the figure which toolchain produced the numbers.

    A figure that does not name its own provenance is how an IREE-era result
    ends up being read as a current one.
    """
    if provenance == IREE_ERA:
        fig.text(0.005, 0.002,
                 "Measured on the IREE path, which this project has since "
                 "dropped. NOT the current ModelBlaster toolchain; the two are "
                 "not comparable.", fontsize=4.5, color=figstyle.C_DEADLINE, va="bottom")
    elif provenance == MODELBLASTER:
        fig.text(0.005, 0.002, "Measured on the ModelBlaster path.",
                 fontsize=4.5, color="#666666", va="bottom")


def main():
    b = os.path.join(REPO, "artifacts", "k1_run", "baselines")
    s = os.path.join(REPO, "schedules")
    rungs = [
        ("B0", "B0  static per-model placement",
         f"{b}/trace_B0.csv", f"{s}/scheduled_k1_B0_static.json"),
        ("B1", "B1  XPU-RT schedules the fixed dispatch graph",
         f"{b}/trace_B1.csv",
         f"{s}/scheduled_networks_k1_mlp_dronet_greedy_profiled.json"),
        ("B4", "B4  + sharding, after the scheduler fed evidence back",
         f"{b}/trace_B4.csv",
         f"{s}/scheduled_networks_k1_B4_shard_greedy_profiled.json"),
    ]
    # Generated output, so it lands in the gitignored /out/ tree rather than
    # in the repo proper. There is no paper/ directory here any more; the
    # LaTeX sources live outside this checkout.
    out = os.environ.get("XPURT_FIGURE_DIR") or os.path.join(REPO, "out", "figures")
    os.makedirs(out, exist_ok=True)

    # Establish provenance BEFORE plotting, and refuse to draw a figure that
    # mixes toolchains -- a mixed ladder would be meaningless and would look
    # exactly like a meaningful one.
    provs = {}
    for tag, _title, trace_p, _sched_p in rungs:
        if not os.path.exists(trace_p):
            print(f"missing trace for {tag}: {trace_p}", file=sys.stderr)
            return 2
        provs[tag] = trace_provenance(load_trace(trace_p))
    distinct = set(provs.values())
    if len(distinct) > 1:
        print(f"refusing to plot a ladder that mixes toolchains: {provs}",
              file=sys.stderr)
        return 3
    provenance = distinct.pop()
    print(f"trace provenance: {provenance}")

    # The prefix is part of the fix. A file called
    # `k1_schedule_evolution.png` says nothing about which toolchain it came
    # from; `iree_era_k1_schedule_evolution.png` cannot be mistaken.
    prefix = "iree_era_" if provenance == IREE_ERA else ""
    if provenance == IREE_ERA:
        print("NOTE: these rungs are IREE-era. The project has moved to "
              "ModelBlaster and the two are not comparable. In particular the "
              "B4 sharding rung has NO ModelBlaster equivalent -- sharding has "
              "never run on that path. Figures are stamped and prefixed "
              "accordingly.", file=sys.stderr)

    global _PROVENANCE
    _PROVENANCE = provenance
    figure1(rungs, os.path.join(out, prefix + "k1_schedule_evolution"))
    figure2(rungs, os.path.join(out, prefix + "k1_feedback_ladder"))

    # figure3 reads gen/profile -- the retired IREE tree -- and its whole
    # premise is that IREE distributes these convolutions. It is an IREE
    # figure by construction, so it is always prefixed.
    figure3(os.path.join(out, "iree_era_k1_shard_evidence"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

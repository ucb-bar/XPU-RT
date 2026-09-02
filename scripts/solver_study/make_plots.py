"""Comparison figures for the spike 3-model workload.

Palette: dataviz reference instance. Makespan is a single measure across
methods, so bars take one hue (categorical slot 1) rather than a cycled
rainbow; schedules that miss a periodic deadline are drawn in status-critical
AND labelled, so validity is never carried by colour alone.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.environ.get("XPURT_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "plots")
S = os.environ.get("XPURT_STUDY_DATA") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data")

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
BLUE, CRITICAL, GOOD = "#2a78d6", "#d03b3b", "#0ca30c"
GRID = "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def _style(ax, xlabel):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.tick_params(length=0)


def rows_677():
    d = json.load(open(os.path.join(S, "bench_spike3.json")))
    r = {x["method"]: x for x in d["rows"] if x.get("objective") is not None}
    r["heft_edf"] = {"method": "heft_edf", "objective": 1241.61, "misses": 0, "wall_s": 0.04}
    order = ["greedy_reserved", "pso", "sa", "heft", "cpsat", "greedy_periodic",
             "heft_edf", "greedy", "decomposed"]
    return [r[k] for k in order if k in r]


def fig1():
    rows = sorted(rows_677(), key=lambda x: x["objective"])
    names = [x["method"] for x in rows]
    vals = [x["objective"] for x in rows]
    bad = [int(x["misses"] or 0) > 0 for x in rows]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = range(len(rows))
    bars = ax.barh(list(y), vals, height=0.62,
                   color=[CRITICAL if b else BLUE for b in bad])
    for b in bars:
        b.set_joinstyle("round")
    for i, (v, x) in enumerate(zip(vals, rows)):
        miss = int(x["misses"] or 0)
        lab = f"{v:,.0f} ms" + (f"   ✗ {miss} missed windows" if miss else "")
        ax.text(v + max(vals) * 0.012, i, lab, va="center", fontsize=9,
                color=CRITICAL if miss else INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.34)
    _style(ax, "non-periodic makespan (ms) — lower is better")
    fig.suptitle("Scheduler comparison — spike mlp_control + dronet + yolov8_nano",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.012, 0.905,
             "677 operations, 2 lanes, fixed instance (mlp_control ×63, dronet ×1). "
             "Red = schedule misses periodic deadlines.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = os.path.join(OUT, "spike3_makespan_comparison.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    return p


def fig2():
    rows = rows_677()
    # Several methods land on nearly identical (time, makespan); nudge their
    # labels apart by hand rather than letting them stack illegibly.
    nudge = {"greedy": (9, -13), "heft_edf": (9, 5), "decomposed": (9, 5),
             "heft": (-4, -16), "greedy_reserved": (9, -13),
             "greedy_periodic": (9, 5), "pso": (9, 5), "sa": (9, 5),
             "cpsat": (-16, 12)}
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for x in rows:
        miss = int(x["misses"] or 0)
        w = max(float(x["wall_s"]), 0.02)
        ax.scatter(w, x["objective"], s=110, zorder=3,
                   color=CRITICAL if miss else BLUE,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(x["method"] + (" ✗" if miss else ""),
                    (w, x["objective"]), textcoords="offset points",
                    xytext=nudge.get(x["method"], (9, 5)), fontsize=9,
                    color=CRITICAL if miss else INK, zorder=4)
    ax.set_xscale("log")
    _style(ax, "wall-clock to produce the schedule (s, log scale)")
    ax.set_ylabel("non-periodic makespan (ms)", fontsize=9, color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_ylim(560, 1360)
    ax.axhline(628.94, color=GOOD, linewidth=1.5, linestyle="--", zorder=1)
    ax.text(0.97, 0.045, "best valid: 628.94 ms", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=GOOD)
    fig.suptitle("Quality against cost — exact methods buy little for 1000× the time",
                 x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.012, 0.905,
             "Same 677-operation instance. ✗ = misses periodic deadlines.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = os.path.join(OUT, "spike3_quality_vs_cost.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    return p


def fig3():
    panels = [("backend_yolo212.json", "yolov8_only_spike\n212 ops"),
              ("backend_fsim242.json", "fsim dronet+yolov8\n242 ops"),
              ("backend_spike277.json", "spike 3-model\n271 ops")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    order = ["cpsat", "milp:MOSEK", "milp:HIGHS", "milp:SCIPY",
             "heft", "heft_edf", "greedy_reserved"]
    for ax, (fn, title) in zip(axes, panels):
        d = json.load(open(os.path.join(S, fn)))
        r = {x["method"]: x for x in d["rows"] if x.get("objective") is not None}
        names = [m for m in order if m in r]
        vals = [r[m]["objective"] for m in names]
        bad = [int(r[m]["misses"] or 0) > 0 for m in names]
        y = range(len(names))
        ax.barh(list(y), vals, height=0.6,
                color=[CRITICAL if b else BLUE for b in bad])
        for i, v in enumerate(vals):
            ax.text(v * 1.08, i, f"{v:,.0f}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=9.5, color=INK)
        ax.invert_yaxis(); ax.set_xscale("log")
        ax.set_xlim(min(vals) * 0.55, max(vals) * 4.0)
        _style(ax, "makespan (ms, log) — lower is better")
        ax.set_title(title, loc="left", fontsize=10.5, pad=8)
    fig.suptitle("Solver backends across instance sizes — CP-SAT wins at every size",
                 x=0.008, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.008, 0.90,
             "Matched 120 s budget for cpsat and every milp backend. "
             "HiGHS/SCIPY at 242 ops return the big-M constant itself, not a schedule.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    p = os.path.join(OUT, "solver_backends_by_size.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    return p


if __name__ == "__main__":
    pass


def fig4():
    """Lane occupancy over time for the three most interesting schedules.

    The pipeline's own Gantt draws one labelled box per dispatch, which at 677
    dispatches is unreadable. This collapses each lane to coloured spans by
    network, which is the thing the comparison is actually about: who owns RVV,
    and when.
    """
    import json as _json
    import re as _re
    SLOT = {"mlp_control": "#2a78d6", "dronet": "#eb6834", "yolov8_nano": "#1baf7a"}
    picks = [("auto", "auto  →  greedy_reserved     629 ms, valid"),
             ("greedy_periodic", "greedy_periodic     665 ms, valid"),
             ("heft", "heft     631 ms, 284 missed windows")]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 6.4), sharex=True)
    lanes = ["CPU_P#0", "CPU_E#0"]
    lane_label = {"CPU_P#0": "RVV", "CPU_E#0": "scalar"}
    for ax, (tag, title) in zip(axes, picks):
        f = (fos.path.join(REPO, "schedules") + "/"
             f"scheduled_networks_mlp_dronet_yolo_spike_{tag}_profiled.json")
        d = _json.load(open(f))
        for e in d["dispatches"].values():
            hw = e["hardware_target"]
            if hw not in lanes:
                continue
            # Only the trailing instance index is an index: stripping every
            # digit turns "yolov8_nano" into "yolov_nano" and loses its colour.
            net = _re.sub(r"\d+$", "", e["job_name"])
            ax.barh(lanes.index(hw), float(e["duration"]), left=float(e["start_time"]),
                    height=0.55, color=SLOT.get(net, INK3), linewidth=0)
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels([lane_label[l] for l in lanes], fontsize=9.5, color=INK)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10.5, pad=6, color=INK)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[-1].set_xlabel("time (ms)", fontsize=9, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SLOT.values()]
    axes[0].legend(handles, list(SLOT), loc="upper right", ncol=3, frameon=False,
                   fontsize=9, bbox_to_anchor=(1.0, 1.42))
    fig.suptitle("Where each schedule puts the work — spike 3-model workload",
                 x=0.008, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.008, 0.925,
             "yolov8_nano can only run well on RVV; the difference between these "
             "schedules is what else is allowed onto that lane.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(OUT, "spike3_lane_occupancy.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    return p


def fig5():
    """Validity by *lane composition*, not by which sweep produced the row.

    The two sweeps overlap: eight FireSim workloads appear in both, so
    plotting "spike sweep vs gemmini sweep" would compare two heavily
    overlapping populations. The real question is what the accelerator lane
    does to the ranking, so partition on that: does the workload's profile_hw
    name a gemmini lane or not.
    """
    import csv as _csv, json as _json, glob as _glob
    RVV_ONLY = "#2a78d6"; GEM = "#eb6834"
    order = ["auto", "heft_edf", "decomposed", "greedy", "heft",
             "greedy_reserved", "greedy_periodic"]

    best = {}          # spec -> {method: valid?}, later rows win
    for f in ("results_final_scored.csv", "results_gemmini_scored.csv"):
        for r in _csv.DictReader(open(os.path.join(S, f))):
            ok = (r["coverage"] == "OK"
                  and str(r["misses"]) not in ("", "None")
                  and int(r["misses"]) == 0)
            best.setdefault(r["spec"], {})[r["solver"]] = ok

    def has_gemmini(spec):
        d = _json.load(open(f"/scratch2/dima/misc_sw/XPU-RT/data/toplevel/{spec}.json"))
        lanes = (d.get("hardware", {}).get("profile_hw", {}) or {}).values()
        return any("gemmini" in str(v).lower() for v in lanes)

    gem_specs = [sp for sp in best if has_gemmini(sp)]
    rvv_specs = [sp for sp in best if not has_gemmini(sp)]

    def pct(specs, m):
        vals = [best[sp].get(m) for sp in specs if m in best[sp]]
        return 100.0 * sum(1 for v in vals if v) / max(1, len(vals))

    fig, ax = plt.subplots(figsize=(10, 5.4))
    y = list(range(len(order))); h = 0.36
    ax.barh([v + h / 2 for v in y], [pct(rvv_specs, m) for m in order], height=h,
            color=RVV_ONLY, label=f"CPU lanes only — RVV+scalar  ({len(rvv_specs)} workloads)")
    ax.barh([v - h / 2 for v in y], [pct(gem_specs, m) for m in order], height=h,
            color=GEM, label=f"with a Gemmini lane  ({len(gem_specs)} workloads)")
    for i, m in enumerate(order):
        ax.text(pct(rvv_specs, m) + 1.5, i + h / 2, f"{pct(rvv_specs, m):.0f}%",
                va="center", fontsize=8.5, color=INK)
        ax.text(pct(gem_specs, m) + 1.5, i - h / 2, f"{pct(gem_specs, m):.0f}%",
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=10, color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, 116)
    _style(ax, "share of workloads with a valid schedule (%)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Adding an accelerator lane reshuffles the ranking — the case for `auto`",
                 x=0.008, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.008, 0.915,
             "Partitioned by whether the workload's profile_hw names a Gemmini lane. "
             "Only 3 workloads have CPU-only lanes, so read the blue bars as 3/3, "
             "2/3, 1/3 — the Gemmini side (12) carries the weight.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    p = os.path.join(OUT, "solver_validity_by_target.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    return p


def fig6():
    """Makespan vs solve time for every solver, with the Pareto frontier.

    Two panels on a shared x: the left shows every point on a log makespan
    axis (HiGHS at 5000 ms is a real result and belongs on the chart, not in a
    caption), the right zooms on the band where the frontier actually lives.
    Colour is the solver family — three slots, the all-pairs validated limit
    for a scatter — and frontier points are ringed and labelled so identity
    never rests on colour alone.
    """
    import json as _json
    CONSTRUCTIVE, META, EXACT = "#2a78d6", "#eb6834", "#1baf7a"
    FAM = {"constructive": CONSTRUCTIVE, "metaheuristic": META, "exact": EXACT}

    pts = []
    ab = _json.load(open(os.path.join(S, "ablation_q31.json")))
    for r in ab["rows"]:
        if r.get("objective") is None:
            continue
        fam = "constructive" if r["status"] == "heuristic" else "exact"
        name = r["solver"] if fam == "constructive" else f"{r['solver']}@{r['budget']:.0f}s"
        pts.append((max(float(r["wall_s"]), 0.01), float(r["objective"]), name, fam))
    for r in _json.load(open(os.path.join(S, "pareto_fill.json")))["rows"]:
        pts.append((max(float(r["wall_s"]), 0.01), float(r["objective"]),
                    r["solver"], r["family"]))

    front = []
    for w_, o_, n_, f_ in pts:
        if not any((w2 <= w_ and o2 <= o_) and (w2 < w_ or o2 < o_)
                   for w2, o2, _n, _f in pts):
            front.append((w_, o_, n_, f_))
    front.sort()
    # PSO at 15 s and at 60 s are literally the same point — the swarm exits on
    # its own before either budget — so they would print two labels on top of
    # each other. Collapse coincident frontier points into one.
    merged = []
    for w_, o_, n_, f_ in front:
        if merged and abs(merged[-1][0] - w_) < 1e-9 and abs(merged[-1][1] - o_) < 1e-9:
            merged[-1] = (w_, o_, merged[-1][2] + ", " + n_.split("@")[1]
                          if "@" in n_ else merged[-1][2], f_)
        else:
            merged.append((w_, o_, n_, f_))
    front = merged

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, (lo, hi, logy, title) in zip(axes, [
            (None, None, True, "every result, log makespan"),
            (43.5, 54, False, "zoom: where the frontier lives")]):
        for w_, o_, n_, f_ in pts:
            if lo is not None and not (lo <= o_ <= hi):
                continue
            ax.scatter(w_, o_, s=64, color=FAM[f_], zorder=3,
                       edgecolor=SURFACE, linewidth=1.4)
        fw = [p[0] for p in front]; fo = [p[1] for p in front]
        ax.step(fw + [fw[-1] * 8], fo + [fo[-1]], where="post",
                color=INK3, linewidth=1.6, linestyle="--", zorder=2)
        for k, (w_, o_, n_, f_) in enumerate(front):
            ax.scatter(w_, o_, s=190, facecolor="none", edgecolor=INK,
                       linewidth=1.8, zorder=4)
            if lo is None:
                continue
            # Hand-placed: the frontier points sit close together on the log
            # x-axis and an alternating offset still overprints the markers.
            off = [(10, 14), (-46, -30), (14, 10), (10, -30)][k % 4]
            ax.annotate(f"{n_}\n{o_:.2f} ms", (w_, o_), textcoords="offset points",
                        xytext=off, fontsize=8.5, color=INK, zorder=5)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(lo, hi)
        _style(ax, "solve time (s, log) — less is better")
        ax.set_ylabel("makespan (ms) — less is better", fontsize=9, color=INK2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=10.5, pad=8, color=INK)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c,
                          label=l)
               for l, c in (("constructive heuristic", CONSTRUCTIVE),
                            ("metaheuristic (PSO/SA)", META),
                            ("exact (CP-SAT / MILP)", EXACT))]
    handles.append(plt.Line2D([], [], color=INK3, linestyle="--", label="Pareto frontier"))
    axes[0].legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Makespan against solve time — RVV+Gemmini q31, 242 operations",
                 x=0.008, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.008, 0.915,
             "One fixed instance, every solver and budget measured on it. The "
             "frontier is entirely heuristic and warm-started CP-SAT; no cold "
             "exact solve is on it.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    p = os.path.join(OUT, "pareto_makespan_vs_time.png")
    fig.savefig(p, dpi=170); plt.close(fig)
    return p, front

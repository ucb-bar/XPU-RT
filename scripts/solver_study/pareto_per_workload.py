"""One Pareto plot per workload, in the two-panel style of make_plots fig6.

fig6 plots a single hand-picked instance with a hardcoded zoom window
(43.5-54 ms). Here the same layout is produced for every workload in a sweep,
so the zoom band has to be derived per workload: it is the span the frontier
actually occupies, padded, with an absolute floor for the degenerate case
where every solver returns the same schedule (the scale_ladder family does
exactly that -- 97.16 ms for all ten methods -- and a proportional band would
collapse to nothing).

Two things differ from fig6 beyond that, both because this data has a trap
fig6's instance did not. Points are split by whether the schedule met every
periodic window, and the frontier is computed over the valid ones only: on
these workloads the lowest makespan is routinely invalid (heft posts the best
number in the study on 2/10 validity), so a frontier over all points would
recommend a schedule that drops control deadlines.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, INK3, GRID, SURFACE = "#1b1b1b", "#666666", "#c1440e", "#dcdcdc", "#ffffff"
CONSTRUCTIVE, META, EXACT = "#2a78d6", "#eb6834", "#1baf7a"
FAMILY = {
    "greedy": "constructive", "greedy_periodic": "constructive",
    "greedy_reserved": "constructive", "decomposed": "constructive",
    "heft": "constructive", "heft_edf": "constructive",
    "pso": "metaheuristic", "sa": "metaheuristic",
    "cpsat": "exact", "cpsat:warm": "exact", "milp:MOSEK": "exact",
}
FAM_COLOR = {"constructive": CONSTRUCTIVE, "metaheuristic": META, "exact": EXACT}


def frontier(points):
    """Pareto-minimal in (wall, objective) over VALID points only."""
    out = []
    for w, o, n in points:
        if not any((w2 <= w and o2 <= o) and (w2 < w or o2 < o)
                   for w2, o2, _ in points):
            out.append((w, o, n))
    out.sort()
    merged = []                        # collapse coincident points into one label
    for w, o, n in out:
        if merged and abs(merged[-1][0] - w) < 1e-9 and abs(merged[-1][1] - o) < 1e-9:
            merged[-1] = (w, o, merged[-1][2] + ", " + n)
        else:
            merged.append((w, o, n))
    return merged


def one(spec, rec, outdir):
    rows = [r for r in rec["rows"] if r.get("objective") is not None]
    if not rows:
        return None
    pts = [(max(float(r.get("wall_s", 0.0)), 0.01), float(r["objective"]),
            r["method"], r.get("misses") == 0) for r in rows]
    valid = [(w, o, n) for w, o, n, v in pts if v]
    if not valid:
        return None
    front = frontier(valid)

    # Zoom band: what the frontier spans, padded. The absolute floor keeps a
    # workload where every solver ties from collapsing to a zero-height axis.
    fo = [p[1] for p in front]
    lo_raw, hi_raw = min(fo), max(o for _, o, _, v in pts if v)
    pad = max((hi_raw - lo_raw) * 0.35, lo_raw * 0.02, 0.05)
    lo, hi = lo_raw - pad, hi_raw + pad

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, (zoom, title) in zip(axes, [
            (False, "every result, log makespan"),
            (True, "zoom: where the frontier lives")]):
        for w, o, n, v in pts:
            if zoom and not (lo <= o <= hi):
                continue
            c = FAM_COLOR[FAMILY.get(n, "constructive")]
            ax.scatter(w, o, s=64, zorder=3, linewidth=1.4,
                       color=c if v else "none",
                       edgecolor=SURFACE if v else c,
                       marker="o" if v else "X")
        fw = [p[0] for p in front]; fov = [p[1] for p in front]
        ax.step(fw + [fw[-1] * 8], fov + [fov[-1]], where="post",
                color=INK3, linewidth=1.6, linestyle="--", zorder=2)
        for k, (w, o, n) in enumerate(front):
            ax.scatter(w, o, s=190, facecolor="none", edgecolor=INK,
                       linewidth=1.8, zorder=4)
            if not zoom:
                continue
            off = [(10, 14), (-52, -32), (14, 10), (10, -32)][k % 4]
            ax.annotate(f"{n}\n{o:.2f} ms", (w, o), textcoords="offset points",
                        xytext=off, fontsize=8.5, color=INK, zorder=5)
        ax.set_xscale("log")
        if zoom:
            ax.set_ylim(lo, hi)
        else:
            ax.set_yscale("log")
        ax.set_xlabel("solve time (s, log) — less is better", fontsize=9, color=INK2)
        ax.set_ylabel("makespan (ms) — less is better", fontsize=9, color=INK2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=10.5, pad=8, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c, label=l)
               for l, c in (("constructive heuristic", CONSTRUCTIVE),
                            ("metaheuristic (PSO/SA)", META),
                            ("exact (CP-SAT)", EXACT))]
    handles += [plt.Line2D([], [], marker="X", linestyle="", markersize=8,
                           color="none", markeredgecolor=INK2,
                           label="missed a periodic window (never on frontier)"),
                plt.Line2D([], [], color=INK3, linestyle="--", label="Pareto frontier")]
    axes[0].legend(handles=handles, frameon=False, fontsize=8.5, loc="best")
    nice = spec.replace("networks_", "")
    fig.suptitle(f"Makespan against solve time — {nice}", x=0.008, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.008, 0.915,
             f"{rec.get('ops', '?')} operations, lanes {rec.get('lanes', '?')}. "
             f"One fixed instance; every solver measured on it. Frontier over "
             f"schedules that met every periodic window.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    p = os.path.join(outdir, f"pareto_{nice}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", default="plots/pareto_per_workload")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    data = json.load(open(a.results))
    made, skipped = [], []
    for spec, rec in sorted(data.items()):
        if "rows" not in rec:
            skipped.append((spec, "build failed")); continue
        p = one(spec, rec, a.outdir)
        (made.append(p) if p else skipped.append((spec, "no valid schedule")))
    for p in made:
        print("wrote", p)
    for s, why in skipped:
        print(f"skipped {s.replace('networks_','')}: {why}")
    print(f"\n{len(made)} plots, {len(skipped)} skipped -> {a.outdir}")


if __name__ == "__main__":
    main()

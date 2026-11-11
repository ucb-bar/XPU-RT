"""Pareto plot across every workload in a sweep, not just one instance.

`make_plots.py` fig6 plots one workload, so makespan in ms is a fair y-axis.
Across a sweep it is not: control_mix_gempair finishes in 60 ms and
vint_intro_hetero in 4160 ms, so raw makespan separates *workloads* and says
almost nothing about *solvers*. Three choices make the pooled plot mean
something.

1. y is a RATIO, not a time: objective / best valid objective on that same
   workload. 1.0 means "matched the best anyone achieved here". Dimensionless,
   so workloads of wildly different size can share an axis.

2. Invalid schedules never define the frontier. A schedule that misses a
   periodic window is not a cheaper point on a trade-off curve, it is a
   dropped control deadline -- and on these workloads the lowest makespan is
   routinely invalid (heft is fastest and misses 14 windows on
   control_mix_gempair). Invalid runs are drawn, hollow, so the cost of the
   good-looking number is visible, but the frontier is computed over valid
   runs only.

3. Same-kind pairs and heterogeneous pairs are separate panels. Pooling them
   hides the finding: the solver ranking inverts between them, so a single
   frontier would average away the effect it exists to show.

Per-(solver, workload) points are drawn faintly and the frontier runs through
per-solver medians -- median rather than mean because the diverging cases are
extreme enough to drag a mean somewhere no run actually sits.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, INK3 = "#1b1b1b", "#666666", "#c1440e"
ORDER = ["greedy", "greedy_periodic", "greedy_reserved", "decomposed",
         "heft", "heft_edf", "pso", "sa", "cpsat", "cpsat:warm", "milp:MOSEK"]
CMAP = plt.get_cmap("tab10")
COLOR = {m: CMAP(i % 10) for i, m in enumerate(ORDER)}


QUAD = "quad  (2+2 harts, 6 combinations)"
PAIR = "same-kind pair  (2 sibling harts, 3 combinations)"
HET  = "heterogeneous pair  (2 kinds, 2 combinations)"
PANELS = (HET, PAIR, QUAD)          # ordered by size of the assignment space


def family(spec):
    # Order matters: a quad spec is 2 gemmini + 2 rvv, so it is heterogeneous
    # too -- but what separates it is the number of combinations the search can
    # choose among, which is what the panels are really about.
    if "quad" in spec:
        return QUAD
    if "gempair" in spec or "rvvpair" in spec:
        return PAIR
    if "hetero" in spec:
        return HET
    return "other"


def load(paths):
    pts = []                      # (family, spec, method, ratio, wall, valid)
    for p in paths:
        data = json.load(open(p))
        for spec, d in data.items():
            if "rows" not in d:
                continue
            rows = [r for r in d["rows"] if r.get("objective") is not None]
            valid = [r["objective"] for r in rows if r.get("misses") == 0]
            if not valid:
                continue          # nothing valid: no reference to normalise to
            best = min(valid)
            if best <= 0:
                continue
            for r in rows:
                pts.append((family(spec), spec, r["method"],
                            r["objective"] / best, max(r.get("wall_s", 0.0), 1e-3),
                            r.get("misses") == 0))
    return pts


def frontier(points):
    """Pareto-minimal in (wall, ratio): keep a point if nothing is both
    at least as fast and at least as good."""
    out, best = [], float("inf")
    for w, rat, m in sorted(points, key=lambda z: (z[0], z[1])):
        if rat < best - 1e-12:
            out.append((w, rat, m)); best = rat
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="plots/pareto_across_workloads.png")
    a = ap.parse_args()
    pts = load(a.results)
    fams = [f for f in PANELS if any(p[0] == f for p in pts)]
    if not fams:
        raise SystemExit("no usable points")

    fig, axes = plt.subplots(1, len(fams), figsize=(7.4 * len(fams), 6.0),
                             squeeze=False)
    for ax, fam in zip(axes[0], fams):
        sub = [p for p in pts if p[0] == fam]
        specs = sorted({p[1] for p in sub})
        agg, labels = [], []
        for m in ORDER:
            mine = [p for p in sub if p[2] == m]
            if not mine:
                continue
            ok = [p for p in mine if p[5]]
            for _, _, _, rat, w, v in mine:      # faint per-workload points
                ax.scatter(w, rat, s=26, color=COLOR[m], alpha=0.30 if v else 0.30,
                           marker="o" if v else "x",
                           facecolors=COLOR[m] if v else "none",
                           linewidths=0.9, zorder=2)
            rate = len(ok) / len(mine)
            # A solver that was never valid still gets a marker -- hollow, and
            # at its median over ALL runs. Dropping it would quietly remove the
            # most interesting failures from the figure: heft and
            # greedy_reserved post the lowest makespans on these workloads and
            # miss windows on every one of them, which is the whole point.
            src = ok if ok else mine
            mr = float(np.median([p[3] for p in src]))
            mw = float(np.median([p[4] for p in src]))
            agg.append((mw, mr, m, rate))
            ax.scatter(mw, mr, s=90 + 220 * rate,
                       color=COLOR[m] if ok else "none",
                       edgecolor=INK if ok else COLOR[m],
                       linewidth=1.1 if ok else 2.0,
                       hatch=None if ok else "///", zorder=5)
            labels.append((mw, mr, m, rate))

        # Labels last, alternating above/below: several solvers land within a
        # few milliseconds of each other on the log axis and their text
        # otherwise lands on top of the markers it is naming.
        # Several solvers finish within a millisecond of each other, so on a
        # log x-axis their markers sit almost on top of one another. Bucket by
        # decade and stack the labels inside each bucket instead of nudging
        # every label by the same amount, which just moved the pile.
        buckets = {}
        for mw, mr, m, rate in sorted(labels, key=lambda z: (z[0], z[1])):
            buckets.setdefault(round(np.log10(mw) * 2), []).append((mw, mr, m, rate))
        for grp in buckets.values():
            grp.sort(key=lambda z: z[1])
            for k, (mw, mr, m, rate) in enumerate(grp):
                dy = 13 + 15 * k if len(grp) > 1 else 11
                tag = m if rate >= 0.999 else f"{m} ({rate:.0%} valid)"
                ax.annotate(tag, (mw, mr), textcoords="offset points",
                            xytext=(11, dy), fontsize=8.2, color=INK, zorder=8,
                            arrowprops=dict(arrowstyle="-", color=INK2,
                                            linewidth=0.6, shrinkA=0, shrinkB=3)
                            if len(grp) > 1 else None,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="none", alpha=0.85))

        # Frontier over solvers that were valid on EVERY workload in the panel:
        # a solver that is only sometimes valid has not earned a place on a
        # curve people will read as "pick the leftmost point you can afford".
        elig = [(w, r, m) for w, r, m, rate in agg if rate >= 0.999]
        fr = frontier(elig)
        if len(fr) > 1:
            ax.plot([f[0] for f in fr], [f[1] for f in fr], "--", color=INK3,
                    linewidth=1.6, zorder=4)
        for w, r, m in fr:
            ax.scatter(w, r, s=330, facecolors="none", edgecolors=INK3,
                       linewidth=1.8, zorder=7)

        ys = [p[3] for p in sub]
        from collections import Counter
        tallest = max(Counter(round(np.log10(w) * 2)
                              for w, _, _, _ in labels).values(), default=1)
        span = max(ys) - min(min(ys), 0.97)
        ax.set_ylim(min(0.97, min(ys) - 0.03),
                    max(ys) + 0.06 + 0.055 * span * tallest)
        ax.set_xscale("log")
        ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xlabel("solve wall time (s, log)")
        ax.set_ylabel("makespan / best valid makespan on that workload")
        ax.set_title(f"{fam}\n{len(specs)} workloads", fontsize=10.5)
        ax.grid(alpha=0.25, linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Solver Pareto frontier across workloads "
                 "(y normalised per workload; frontier over valid runs only)",
                 fontsize=12.5, y=0.995)
    fig.text(0.5, 0.005,
             "Filled = every periodic window met.  x = missed at least one; drawn but never on the frontier.  "
             "Large marker = per-solver median (size ∝ share of workloads it solved validly).  "
             "Ringed = Pareto-optimal among solvers valid on every workload in the panel.",
             ha="center", fontsize=8.4, color=INK2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170)
    print("wrote", a.out, f"({len(pts)} points, {len(fams)} panels)")


if __name__ == "__main__":
    main()

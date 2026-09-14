#!/usr/bin/env python3
"""Shared frontier machinery for the fig_pareto_{compute,missiontime,funnel} figures.

Three numbers describe every frontier in PARETO_SEARCH.md, and they are deliberately not
the same number:

  RAW           arms that nothing else dominates, using the 20-seed mean of each axis.
                Flattered by noise: a point that is only non-dominated because its mean
                landed high on 480 episodes still counts.

  RESOLVED      walk the raw frontier from one end and keep a point only when it differs
                from the last kept point by more than the 95% band on BOTH axes. This is
                the "well-separated points" count -- how many operating levels the sweep
                can actually tell apart.

  BOOTSTRAP     resample the 20 seeds with replacement 2,000x, recompute the arm means and
                the frontier each time, and report the median frontier size plus the arms
                that stay on it in at least half the resamples. This is the count that
                survives the harness's own nondeterminism.

The band is 1.96 x the standard error over the 20 seeds of that arm, averaged over arms,
i.e. a 95% interval on the seed mean. On success it comes out at 4.0 (egg) / 3.7 (spoon) /
3.5 (coke) / 3.0 (drawer) points, close to and slightly wider than the 3.33/2.29/2.08/2.08
bands quoted in earlier figures; both are drawn where a figure shows success.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE = json.load(open(HERE / "pareto_search_cache.json"))
TASKS = [("egg", "eggplant in basket", "widowx  ·  GRASP+PLACE"),
         ("spoon", "spoon on towel", "widowx  ·  GRASP+PLACE"),
         ("coke", "pick coke can", "google  ·  GRASP+LIFT"),
         ("drawer", "close drawer", "google  ·  PUSH")]
ARMS = sorted({k.split("|")[1] for k in CACHE},
              key=lambda a: (int(a.split("_")[0][1:]), int(a.split("_")[1])))
# the house fast->slow ramp, taken verbatim from fig_pareto_energy.py
RAMP = ["#0e6655", "#148f77", "#17a589", "#45b39d", "#d68910", "#e67e22", "#c0392b",
        "#7b241c"]
INK, MUTE, EDGE, CAPTION = "#16161C", "#6B6B78", "#9a9aa4", "#555"
# bands quoted by the earlier figures, kept for cross-reference
HOUSE_BAND = {"egg": 3.33, "spoon": 2.29, "coke": 2.08, "drawer": 2.08}


def cadence_ms(arm):
    """Release cadence the schedule asks for, from the arm name g<period>_<window>."""
    return int(arm.split("_")[0][1:])


CADENCES = sorted({int(a.split("_")[0][1:]) for a in ARMS})


def arm_colour(arm):
    """Fast cadence green -> slow cadence red. The 11 cadences are SNAPPED to the eight
    ramp entries rather than interpolated between them: blending #45b39d into #d68910
    invents a khaki that appears in no other figure, and 8 steps over 11 cadences is
    enough to read the ordering off the fill."""
    r = CADENCES.index(cadence_ms(arm)) / (len(CADENCES) - 1)
    return RAMP[int(round(r * (len(RAMP) - 1)))]


def seed_values(task, arm, metric):
    """The 20 per-seed values of one metric. None (a cell that won no episode) drops."""
    c = CACHE[f"{task}|{arm}"]
    v = [c[s].get(metric) for s in sorted(c, key=int)]
    return np.array([x for x in v if x is not None], float)


def axis(task, metric):
    """(mean, standard error) per arm, plus the raw 20-seed matrix for the bootstrap."""
    rows = [seed_values(task, a, metric) for a in ARMS]
    n = min(len(r) for r in rows)
    assert n >= 15, f"{task}/{metric}: an arm has only {n} usable seeds"
    mean = np.array([r.mean() for r in rows])
    sem = np.array([r.std(ddof=1) / np.sqrt(len(r)) for r in rows])
    return mean, sem, rows


def frontier(x, y, sx, sy):
    """Indices of points nothing else dominates. sx/sy = +1 if larger is better.
    Vectorised because the seed bootstrap calls this 2,000 times per pair and the
    full sweep is 689 pairs."""
    gx, gy = sx * np.asarray(x, float), sy * np.asarray(y, float)
    ge = (gx[None, :] >= gx[:, None]) & (gy[None, :] >= gy[:, None])
    gt = (gx[None, :] > gx[:, None]) | (gy[None, :] > gy[:, None])
    return list(np.flatnonzero(~(ge & gt).any(axis=1)))


def resolved(x, y, idx, sx, bx, by):
    """Frontier points separated from their kept neighbour by more than BOTH bands."""
    P = sorted([(x[i], y[i], i) for i in idx], key=lambda p: sx * p[0])
    keep = [P[0]]
    for p in P[1:]:
        if abs(p[0] - keep[-1][0]) > bx and abs(p[1] - keep[-1][1]) > by:
            keep.append(p)
    return [p[2] for p in keep]


def bootstrap(rx, ry, sx, sy, n=2000, seed=20260908):
    """Seed bootstrap of frontier membership. Returns (median size, membership rate)."""
    rng = np.random.default_rng(seed)
    ns = min(min(len(r) for r in rx), min(len(r) for r in ry))
    X = np.array([r[:ns] for r in rx]); Y = np.array([r[:ns] for r in ry])
    hit = np.zeros(len(X)); sizes = np.empty(n)
    for k in range(n):
        j = rng.integers(0, ns, ns)
        f = frontier(X[:, j].mean(1), Y[:, j].mean(1), sx, sy)
        hit[f] += 1
        sizes[k] = len(f)
    return float(np.median(sizes)), hit / n


def report(task, mx, sx, my, sy):
    """Everything the figures and PARETO_SEARCH.md quote, for one (task, pair)."""
    x, ex, rx = axis(task, mx)
    y, ey, ry = axis(task, my)
    bx, by = 1.96 * ex.mean(), 1.96 * ey.mean()
    idx = frontier(x, y, sx, sy)
    res = resolved(x, y, idx, sx, bx, by)
    med, hit = bootstrap(rx, ry, sx, sy)
    return dict(x=x, y=y, ex=ex, ey=ey, bx=bx, by=by, idx=idx, res=res,
                boot_median=med, hit=hit,
                stable=[ARMS[i] for i in np.argsort(-hit) if hit[i] >= 0.5],
                corr=float(np.corrcoef(sx * x, sy * y)[0, 1]))


# ---------------------------------------------------------------------------
# label placement
# ---------------------------------------------------------------------------
# Frontier arms cluster in one corner of a panel, so a fixed offset collides as soon as
# two of them land within a label's width of each other -- which happened on coke
# (g180_290 over g140_305) and drawer (g250_260 over g180_400) with a two-way stagger.
# This tries eight offsets in order and takes the first that overlaps neither an
# already-placed label nor the panel edge, measuring everything in POINTS so the test
# does not depend on the data units of the axis.
CANDIDATES = [(13, 7), (13, -16), (-13, 7), (-13, -16),
              (13, 25), (13, -34), (-13, 25), (-13, -34)]


def place_labels(ax, xs, ys, names, ax_w_pt, ax_h_pt, bold=None, fontsize=7.8,
                 ink="#16161C", char_w=6.2, line_h=9.5, obstacles=(), marker_pt=11.0):
    """Annotate (xs, ys) with names, skipping offsets that collide. Every labelled marker
    is itself an obstacle, as is anything in `obstacles` -- given in DATA units as
    (x0, y0, x1, y1) -- which is how the error crosses stay clear of the text. Returns
    the count of labels forced onto the last candidate (i.e. possibly still tight)."""
    xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
    sx = ax_w_pt / (xhi - xlo); sy = ax_h_pt / (yhi - ylo)
    placed, forced = [], 0
    for i in range(len(names)):
        px, py = (xs[i] - xlo) * sx, (ys[i] - ylo) * sy
        placed.append((px - marker_pt, py - marker_pt, px + marker_pt, py + marker_pt))
    for x0, y0, x1, y1 in obstacles:
        placed.append(((x0 - xlo) * sx, (y0 - ylo) * sy,
                       (x1 - xlo) * sx, (y1 - ylo) * sy))
    fixed = len(placed)
    order = sorted(range(len(names)), key=lambda i: -ys[i])
    for i in order:
        px, py = (xs[i] - xlo) * sx, (ys[i] - ylo) * sy
        w, h = char_w * len(names[i]), line_h
        for k, (dx, dy) in enumerate(CANDIDATES):
            x0 = px + dx if dx > 0 else px + dx - w
            y0 = py + dy - h / 2 if abs(dy) < 20 else py + dy - h / 2
            box = (x0, y0, x0 + w, y0 + h)
            if box[0] < 2 or box[2] > ax_w_pt - 2 or box[1] < 2 or box[3] > ax_h_pt - 2:
                continue
            if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                   for b in placed):
                continue
            break
        else:
            dx, dy = CANDIDATES[-1]; forced += 1
            x0 = px + dx if dx > 0 else px + dx - w
            box = (x0, py + dy - h / 2, x0 + w, py + dy + h / 2)
        placed.append(box)
        # a hairline leader removes the ambiguity when the offset had to be large: with
        # 44 markers a label 60 pt away otherwise reads as belonging to its neighbour
        ax.annotate(names[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(dx, dy), ha="left" if dx > 0 else "right", va="center",
                    fontsize=fontsize, color=ink, zorder=6,
                    fontweight="bold" if (bold is None or bold[i]) else "normal",
                    arrowprops=dict(arrowstyle="-", color="#9a9aa4", lw=0.6,
                                    shrinkA=1.5, shrinkB=9.0))
    return forced

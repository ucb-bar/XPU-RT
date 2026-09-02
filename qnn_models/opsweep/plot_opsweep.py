#!/usr/bin/env python3
"""Heatmaps and decomposition plots from results.jsonl.

Three views, because the sweep answers three different questions:

  heatmap    op x (backend,precision), coloured by speedup against the best CPU
             cell at the same size. This is the placement map -- green means an
             accelerator is worth using for that op, and most of it is not.
  floor      measured dispatch overhead per backend, warm vs cold. The gap is
             the DVFS/power-collapse component, which only bites dispatches
             shorter than roughly a millisecond.
  ladder     time against arithmetic per op, log-log, with the fitted
             `overhead + macs/throughput` line. A flat line means the op is
             overhead-bound at that size and no amount of tuning helps.

Compose failures are drawn as hatched cells rather than dropped, so "HTA cannot
do this at all" stays visible next to "HTA is slow at this".

    python3 plot_opsweep.py --out ../../plots
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# gpu/int8 is not a lane: the Adreno backend rejects quantized tensors outright,
# so it was 100% compose failures and only ever drew an empty hatched panel.
LANES = [("cpu", "int8"), ("cpu", "fp32"), ("dsp", "int8"),
         ("hta", "int8"), ("gpu", "fp16"), ("gpu", "fp32")]


def load():
    p = os.path.join(HERE, "results.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def heatmap(rows, out):
    """Per op, the median speedup of each lane over the best CPU time at the
    same size point. Ratios, not absolutes, so ops spanning six decades of work
    can share one colour scale."""
    best_cpu = {}
    for r in rows:
        if r.get("status") == "ok" and r["backend"] == "cpu":
            k = (r["op"], tuple(r["params"]))
            best_cpu[k] = min(best_cpu.get(k, 1e18), r["warm_us"])
    acc = defaultdict(list); fail = defaultdict(int); tot = defaultdict(int)
    for r in rows:
        lane = (r["backend"], r.get("precision"))
        if lane not in LANES:
            continue
        k = (r["op"], tuple(r["params"]))
        tot[(r["op"], lane)] += 1
        if r.get("status") != "ok":
            fail[(r["op"], lane)] += 1
            continue
        if k in best_cpu and r.get("warm_us"):
            acc[(r["op"], lane)].append(best_cpu[k] / r["warm_us"])
    ops = sorted({r["op"] for r in rows})
    if not ops:
        print("  no rows yet"); return
    M = np.full((len(ops), len(LANES)), np.nan)
    for i, op in enumerate(ops):
        for j, lane in enumerate(LANES):
            v = acc.get((op, lane))
            if v: M[i, j] = float(np.median(v))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(ops) + 2.4))
    L = np.log2(np.clip(M, 1/64, 64))
    im = ax.imshow(L, cmap="RdYlGn", vmin=-4, vmax=4, aspect="auto")
    for i, op in enumerate(ops):
        for j, lane in enumerate(LANES):
            if np.isnan(M[i, j]):
                miss = fail.get((op, lane), 0); n = tot.get((op, lane), 0)
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False,
                                           hatch="///", edgecolor="#999999", lw=0))
                if n: ax.text(j, i, "x", ha="center", va="center", fontsize=7, color="#666")
            else:
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black" if abs(L[i, j]) < 2.6 else "white")
    ax.set_xticks(range(len(LANES)))
    ax.set_xticklabels([f"{b}\n{p}" for b, p in LANES], fontsize=8.5)
    ax.set_yticks(range(len(ops))); ax.set_yticklabels(ops, fontsize=8.5)
    ax.set_title("Operator placement map — median speedup vs best CPU at the same size\n"
                 "green = accelerator wins, red = CPU wins, hatched = will not compose",
                 fontsize=10.5, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_ticks([-4, -2, 0, 2, 4]); cb.set_ticklabels(["1/16", "1/4", "1x", "4x", "16x"])
    fig.tight_layout()
    p = os.path.join(out, "opsweep_placement_heatmap.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p}")


def floors(rows, out):
    fits = {}
    fp = os.path.join(HERE, "fits.json")
    if os.path.exists(fp):
        for f in json.load(open(fp)):
            fits.setdefault((f["backend"], f["precision"], f["phase"]), []).append(f["overhead_us"])
    if not fits:
        print("  no fits yet"); return
    lanes = [l for l in LANES if (l[0], l[1], "warm_us") in fits]
    w = [np.median(fits[(b, p, "warm_us")]) for b, p in lanes]
    c = [np.median(fits.get((b, p, "cold_us"), [np.nan])) for b, p in lanes]
    x = np.arange(len(lanes))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - .2, w, .4, label="warm (gap 0)", color="#4C72B0")
    ax.bar(x + .2, c, .4, label="cold (gap 3000 us)", color="#C44E52")
    ax.set_yscale("log"); ax.set_ylabel("fitted dispatch overhead (us, log)")
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n{p}" for b, p in lanes], fontsize=9)
    ax.set_title("Per-dispatch system overhead, median of the per-op fits\n"
                 "the intercept of `t = overhead + macs/throughput`; the warm/cold gap "
                 "is power collapse", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False); ax.grid(axis="y", alpha=.3, ls=":")
    fig.tight_layout()
    p = os.path.join(out, "opsweep_dispatch_overhead.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p}")


def ladders(rows, out, max_ops=12):
    by = defaultdict(list)
    for r in rows:
        if r.get("status") == "ok" and r.get("macs"):
            by[r["op"]].append(r)
    ops = [o for o in sorted(by) if len(by[o]) >= 6][:max_ops]
    if not ops:
        print("  no ladders yet"); return
    nc = 4; nr = int(np.ceil(len(ops)/nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 3*nr), squeeze=False)
    col = {"cpu": "#B0B7C3", "dsp": "#4C72B0", "hta": "#DD8452", "gpu": "#55A868"}
    for i, op in enumerate(ops):
        ax = axes[i//nc][i % nc]
        for (b, p), rs in sorted({(r["backend"], r["precision"]): None for r in by[op]}.items()):
            pts = sorted([(r["macs"]/1e6, r["warm_us"]) for r in by[op]
                          if r["backend"] == b and r["precision"] == p])
            if len(pts) < 3: continue
            ax.plot([x for x, _ in pts], [y for _, y in pts], "o-", ms=3, lw=1.1,
                    color=col.get(b, "k"), alpha=.85 if p == "int8" else .45,
                    label=f"{b}/{p}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(op, fontsize=9); ax.grid(alpha=.25, ls=":")
        if i % nc == 0: ax.set_ylabel("us")
        if i//nc == nr-1: ax.set_xlabel("MMAC")
        ax.legend(fontsize=6, frameon=False)
    for j in range(len(ops), nr*nc): axes[j//nc][j % nc].axis("off")
    fig.suptitle("Time vs arithmetic per operator (warm). A flat line is overhead-bound.",
                 fontsize=11, x=.02, ha="left")
    fig.tight_layout()
    p = os.path.join(out, "opsweep_size_ladders.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p}")


def crossover(rows, out, compute_only=False):
    """The size axis the aggregate heatmap collapses.

    With compute_only, the fitted per-lane dispatch intercept is subtracted
    from every measurement first, so the map answers a different question: not
    "is this worth offloading" but "if dispatch were free, which side is
    actually faster at the arithmetic".  That is the ceiling you could reach by
    batching or fusing enough work to amortise the launch, and the gap between
    the two maps is exactly what the dispatch floor is costing.

    Subtracting an intercept is only meaningful while it is a modest part of
    the measurement.  Cells where the overhead is more than 80% of the measured
    time are stippled: the remainder there is mostly noise, and at the small end
    it can even go negative.

    One panel per accelerator lane: ops down, size rank across, coloured by
    speedup over the best CPU cell at that same size. Accelerators on this
    board lose at small sizes and only ever win past their dispatch floor, so
    the crossover column is the whole placement decision -- a median over sizes
    averages it away."""
    over = {}
    if compute_only:
        fp = os.path.join(HERE, "fits.json")
        if not os.path.exists(fp):
            print("  no fits yet (compute-only map needs them)"); return
        for f in json.load(open(fp)):
            if f["phase"] == "warm_us":
                over[(f["op"], f["backend"], f["precision"])] = max(f["overhead_us"], 0.0)

    def val(r):
        """Measured warm time, minus the fitted dispatch cost if asked.
        Returns (time, reliable)."""
        t = r.get("warm_us")
        if t is None:
            return None, False
        if not compute_only:
            return t, True
        o = over.get((r["op"], r["backend"], r["precision"]))
        if o is None:
            return None, False
        return max(t - o, 1e-3), (o <= 0.8 * t)

    best_cpu, cpu_ok = {}, {}
    for r in rows:
        if r.get("status") == "ok" and r["backend"] == "cpu":
            t, good = val(r)
            if t is None:
                continue
            k = (r["op"], tuple(r["params"]))
            if t < best_cpu.get(k, 1e18):
                best_cpu[k] = t
                cpu_ok[k] = good
    lanes = [l for l in LANES if l[0] != "cpu"]
    ops = sorted({r["op"] for r in rows if r.get("macs")})
    if not ops:
        print("  no rows yet"); return
    # rank each op's own size points, so a column means "nth smallest", which
    # is comparable across ops whose absolute sizes differ by decades
    order = {}
    for op in ops:
        sizes = sorted({r["macs"] for r in rows if r["op"] == op and r.get("macs")})
        order[op] = {m: i for i, m in enumerate(sizes)}
    ncol = max((len(v) for v in order.values()), default=0)
    if not ncol:
        print("  no sizes yet"); return
    figh = 0.34*len(ops)+2.6
    fig, axes = plt.subplots(1, len(lanes), figsize=(3.1*len(lanes)+1.6, figh),
                             squeeze=False)
    im = None
    for j, lane in enumerate(lanes):
        M = np.full((len(ops), ncol), np.nan)
        F = np.zeros((len(ops), ncol), bool)
        U = np.zeros((len(ops), ncol), bool)   # overhead-dominated
        for r in rows:
            if (r["backend"], r.get("precision")) != lane or not r.get("macs"):
                continue
            i = ops.index(r["op"]); c = order[r["op"]].get(r["macs"])
            if c is None:
                continue
            if r.get("status") != "ok":
                F[i, c] = True; continue
            k = (r["op"], tuple(r["params"]))
            b = best_cpu.get(k)
            t, good = val(r)
            if b and t:
                if good and cpu_ok.get(k, True):
                    M[i, c] = b / t
                else:
                    # Leave it uncoloured.  Subtracting a 1.3 ms intercept from
                    # a 1.4 ms measurement leaves noise, and colouring that
                    # saturated green reads as a result when it is a rounding
                    # error.  Stipple says "measured, but not decomposable".
                    U[i, c] = True
        ax = axes[0][j]
        im = ax.imshow(np.log2(np.clip(M, 1/64, 64)), cmap="RdYlGn",
                       vmin=-4, vmax=4, aspect="auto")
        for i in range(len(ops)):
            for c in range(ncol):
                if F[i, c]:
                    ax.add_patch(plt.Rectangle((c-.5, i-.5), 1, 1, fill=False,
                                               hatch="///", edgecolor="#999", lw=0))
                elif U[i, c]:
                    ax.add_patch(plt.Rectangle((c-.5, i-.5), 1, 1, fill=False,
                                               hatch="....", edgecolor="#444", lw=0))
        ax.set_title(f"{lane[0]}/{lane[1]}", fontsize=9.5)
        ax.set_xlabel("size rank (small to large)", fontsize=8)
        ax.set_xticks(range(0, ncol, max(1, ncol//6)))
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_yticks(range(len(ops))); ax.set_yticklabels(ops, fontsize=8)
        else:
            ax.set_yticks([])
    if compute_only:
        fig.suptitle("Compute only, dispatch overhead removed — the ceiling if "
                     "launches were free\ngreen = accelerator's arithmetic is "
                     "faster, hatched = will not compose, stippled = overhead was "
                     ">80% of the measurement so compute is not separable there",
                     fontsize=10.5, x=.02, y=.995, ha="left", va="top")
    else:
        fig.suptitle("Where each accelerator overtakes the CPU — speedup vs best CPU "
                     "at the same size\ngreen = accelerator wins, hatched = will not "
                     "compose, blank = not measured",
                     fontsize=10.5, x=.02, y=.995, ha="left", va="top")
    fig.subplots_adjust(top=1 - 0.85/figh)   # constant header, any row count
    cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
    cb.set_ticks([-4, -2, 0, 2, 4]); cb.set_ticklabels(["1/16", "1/4", "1x", "4x", "16x"])
    p = os.path.join(out, "opsweep_compute_only_heatmap.png" if compute_only
                     else "opsweep_crossover_heatmap.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p}")


def decomposition(rows, out):
    """Compute vs system overhead, the split the goal actually asks for.

    Uses the fitted intercept as the system term and `macs/throughput` as the
    compute term, evaluated at each op's largest measured size -- the most
    favourable case for the accelerators. Anything where the blue bar dominates
    is a dispatch-bound op that no kernel work will rescue."""
    fp = os.path.join(HERE, "fits.json")
    if not os.path.exists(fp):
        print("  no fits yet"); return
    fits = {(f["op"], f["backend"], f["precision"]): f
            for f in json.load(open(fp)) if f["phase"] == "warm_us"}
    biggest = {}
    for r in rows:
        if r.get("status") == "ok" and r.get("macs"):
            k = (r["op"], r["backend"], r["precision"])
            if r["macs"] > biggest.get(k, (0,))[0]:
                biggest[k] = (r["macs"], r["warm_us"])
    keys = [k for k in sorted(biggest) if k in fits]
    if not keys:
        print("  no decomposition yet"); return
    ov, cp, lab = [], [], []
    for k in keys:
        f = fits[k]; macs, _ = biggest[k]
        g = f.get("gmac_per_s") or 0
        ov.append(max(f["overhead_us"], 0.0))
        cp.append((macs / 1e3 / g) if g > 0 else 0.0)
        lab.append(f"{k[0]}\n{k[1]}/{k[2]}" + (" *" if f.get("origin_fit") else ""))
    y = np.arange(len(keys))
    ov = np.array(ov); cp = np.array(cp)
    tot = np.maximum(ov + cp, 1e-9)
    fig, ax = plt.subplots(figsize=(9.6, 0.26*len(keys)+2.4))
    # Fractions, not a log-scaled stack: these totals span five decades, and on
    # a log axis a stacked segment's length is not proportional to its value.
    ax.barh(y, ov/tot, .74, label="system (dispatch overhead)", color="#4C72B0")
    ax.barh(y, cp/tot, .74, left=ov/tot, label="compute (macs / throughput)",
            color="#DD8452")
    for i, t in enumerate(tot):
        ax.text(1.012, i, f"{t:,.0f} us", va="center", fontsize=5.8, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=5.6)
    ax.set_xlim(0, 1); ax.set_xlabel("fraction of time at each op's largest measured size")
    ax.invert_yaxis()
    ax.set_title("Compute vs system overhead at the most favourable size\n"
                 "blue-dominated rows are dispatch-bound: the arithmetic is not "
                 "the cost.  Absolute total at right.\n"
                 "* = fit through the origin; the free intercept went negative, "
                 "so overhead is below the noise (all CPU).",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left",
              bbox_to_anchor=(0, -0.14/max(len(keys)/28, 1)), ncol=2)
    ax.grid(axis="x", alpha=.3, ls=":")
    fig.tight_layout()
    p = os.path.join(out, "opsweep_compute_vs_overhead.png")
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "plots"))
    a = ap.parse_args()
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    rows = load()
    print(f"  {len(rows)} rows")
    heatmap(rows, out); crossover(rows, out)
    crossover(rows, out, compute_only=True)
    floors(rows, out); decomposition(rows, out); ladders(rows, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Figures for the two findings that had none.

  finding1_knob_inert.png   why the K1 feedback knob does nothing on QRB5165:
                            it reads per-core-width profiles, and QRB5165 has
                            exactly one width.
  finding4_contention.png   contention measured on a concurrent multi-model
                            workload, and what conditioning on it buys
                            out-of-sample.

Palette: validated categorical slots (CVD dE >= 9, normal-vision >= 22); text in
ink tokens, every mark direct-labelled.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID, SURF = "#1a1a19", "#5c5c5a", "#d8d8d5", "#fcfcfb"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _clean(ax, left=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if left:
        ax.spines["left"].set_color(GRID)
    else:
        ax.spines["left"].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_axisbelow(True)


def finding1(out):
    """Width coverage is the whole story: the knob has nothing to choose from."""
    def widths(pat):
        c = {}
        for p in glob.glob(os.path.join(REPO, pat), recursive=True):
            for part in p.split("/"):
                if part.startswith("topo_"):
                    n = len(part.replace("topo_", "").split("_"))
                    c[n] = c.get(n, 0) + 1
        return c
    k1 = widths("gen/profile_mb/*/spacemit_x60/**/topo_*/results.csv")
    qc = widths("gen/profile/*/qrb5165*/**/topo_*/results.csv")

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    slots = [1, 2, 4, 8]
    w = 0.36
    for i, (lab, data, col) in enumerate((("K1 / spacemit_x60", k1, BLUE),
                                          ("QRB5165", qc, ORANGE))):
        xs = [j + (i - 0.5) * w for j in range(len(slots))]
        vs = [data.get(s, 0) for s in slots]
        ax.bar(xs, vs, w * 0.92, color=col, edgecolor=SURF, lw=1.4,
               label=lab, zorder=3)
        for x, v in zip(xs, vs):
            ax.annotate(str(v) if v else "0", (x, v), xytext=(0, 5),
                        textcoords="offset points", ha="center",
                        fontsize=9, color=INK if v else MUTED,
                        weight="bold" if v else "normal")
    ax.set_xticks(range(len(slots)))
    ax.set_xticklabels([f"{s} hart{'s' if s > 1 else ''}\n(topo_{'_'.join(str(i) for i in range(s))})"
                        for s in slots], fontsize=8.4, color=INK)
    ax.set_ylabel("measured profile files at that width", fontsize=9, color=MUTED)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    _clean(ax)
    ax.set_title("Finding 1 — the K1 feedback knob has nothing to read on QRB5165\n"
                 "it selects a per-dispatch core WIDTH; QRB5165 was only ever "
                 "profiled at one",
                 fontsize=11.5, loc="left", color=INK, pad=14)
    top = max(list(k1.values()) + list(qc.values()) + [1])
    ax.annotate("flipping the two feedback switches on a QRB5165 workload\n"
                "leaves the makespan bit-identical at 120.11 ms",
                (1.5, top * 0.60), ha="center", fontsize=8.8,
                color=MUTED, style="italic")
    fig.tight_layout()
    p = os.path.join(out, "finding1_knob_inert.png")
    fig.savefig(p, dpi=160, facecolor=SURF)
    print(f"  wrote {p}")


def finding4(out):
    d = json.load(open(os.path.join(REPO, "results/flowc_contention/k1_tune.json")))
    bm, bc = d["bucket_medians"], d["bucket_counts"]
    err = d["leave_one_run_out_logerr"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.1),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    # left: the signal
    order = [("solo", "0"), ("1-2", "1–2"), ("3+", "3 or more")]
    xs = range(len(order))
    vals = [bm[k] for k, _ in order]
    cols = [MUTED, ORANGE, ORANGE]
    axL.bar(xs, vals, 0.56, color=cols, edgecolor=SURF, lw=1.4, zorder=3)
    for x, (k, _), v in zip(xs, order, vals):
        axL.annotate(f"{v:.4f}", (x, v), xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=10, color=INK, weight="bold")
    axL.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=4)
    axL.annotate("1.0 = the prediction was right", (2.45, 1.0), xytext=(0, 5),
                 textcoords="offset points", ha="right", fontsize=8.2, color=MUTED)
    axL.annotate(f"co-running costs {bm['1-2']/bm['solo']:.3f}x",
                 (0.5, 1.012), ha="center", fontsize=9.2,
                 color=INK, weight="bold")
    axL.set_xticks(list(xs))
    axL.set_xticklabels([f"{lab}\nn={bc[k]}" for k, lab in order],
                        fontsize=9.5, color=INK)
    axL.set_xlabel("other dispatches in flight", fontsize=9, color=MUTED)
    axL.set_ylabel("median actual / predicted", fontsize=9, color=MUTED)
    axL.set_ylim(0.98, max(vals) * 1.055)
    axL.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    _clean(axL)
    axL.set_title("the signal", fontsize=10.5, loc="left", color=INK)

    # right: what it buys, out of sample
    labs = [("none", "no correction"), ("kind", "per core-kind"),
            ("kind+co", "per core-kind\n× co-runners")]
    ys = list(range(len(labs)))[::-1]
    base = err["none"]
    for y, (k, lab) in zip(ys, labs):
        v = err[k]
        col = MUTED if k == "none" else (BLUE if k == "kind" else AQUA)
        axR.barh(y, v, 0.5, color=col, edgecolor=SURF, lw=1.4, zorder=3)
        tag = f"{v:.4f}" + ("" if k == "none" else f"   −{(1-v/base)*100:.1f}%")
        axR.annotate(tag, (v, y), xytext=(8, 0), textcoords="offset points",
                     va="center", fontsize=9.5, color=INK,
                     weight="bold" if k == "kind+co" else "normal")
    axR.set_yticks(ys)
    axR.set_yticklabels([lab for _, lab in labs], fontsize=9.5, color=INK)
    axR.set_xlabel("held-out median |ln(actual/predicted)| — lower is better",
                   fontsize=9, color=MUTED)
    axR.set_xlim(0, base * 1.42)
    axR.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    _clean(axR, left=False)
    axR.set_title("what conditioning on it buys, out of sample",
                  fontsize=10.5, loc="left", color=INK)

    fig.suptitle(f"Finding 4 — contention is real and learnable: "
                 f"{d['dispatches']} dispatches, {d['runs']} runs, "
                 f"{d['concurrent_fraction']*100:.0f}% overlapping\n"
                 "whole runs held out, never individual dispatches",
                 x=0.012, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = os.path.join(out, "finding4_contention.png")
    fig.savefig(p, dpi=160, facecolor=SURF)
    print(f"  wrote {p}")


def finding5(out):
    """Whether the correction transfers -- and why finding 4's did.

    Finding 5's own figures cover the bias, the within-configuration gain and
    the stalls. What they do not show is its central negative claim: fitted on
    one runtime configuration the correction does NOT carry to another. Shown
    beside finding 4's leave-one-run-out result, which does carry, the pair
    says what the correction has to be keyed on.
    """
    import statistics as st
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import flowc_residual_feedback as F

    per = {}
    for q in sorted(glob.glob(os.path.join(REPO, "runs/*/trace.csv"))):
        rows = F.read_trace(q)
        if rows:
            per[rows[0]["trace"]] = rows

    self_fit, held_out, names = [], [], []
    for name, rows in per.items():
        base = F.error(rows, None)["logerr_median"]
        own = F.error(rows, F.fit(rows))["logerr_median"]
        train = [r for k, v in per.items() if k != name for r in v]
        out_of = F.error(rows, F.fit(train))["logerr_median"]
        names.append(name.replace("v3_bundles", "v3"))
        self_fit.append((base, own))
        held_out.append((base, out_of))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    ys = list(range(len(names)))[::-1]
    for y, n, (b, own), (_, out_of) in zip(ys, names, self_fit, held_out):
        axL.plot([b, own], [y + 0.16, y + 0.16], color=GRID, lw=2.6,
                 solid_capstyle="round", zorder=1)
        axL.plot([b, out_of], [y - 0.16, y - 0.16], color=GRID, lw=2.6,
                 solid_capstyle="round", zorder=1)
        axL.scatter([b], [y + 0.16], s=52, color=MUTED, zorder=3,
                    edgecolors="white", lw=1.1)
        axL.scatter([own], [y + 0.16], s=72, color=AQUA, zorder=4,
                    edgecolors="white", lw=1.1)
        axL.scatter([b], [y - 0.16], s=52, color=MUTED, zorder=3,
                    edgecolors="white", lw=1.1)
        worse = out_of > b
        axL.scatter([out_of], [y - 0.16], s=72,
                    color=ORANGE if worse else BLUE, zorder=4,
                    marker="X" if worse else "o", edgecolors="white", lw=1.1)
        axL.annotate(f"{(1-own/b)*100:+.0f}%", (max(b, own), y + 0.16),
                     xytext=(9, 0), textcoords="offset points", va="center",
                     fontsize=8.4, color=INK)
        axL.annotate(f"{(1-out_of/b)*100:+.0f}%", (max(b, out_of), y - 0.16),
                     xytext=(9, 0), textcoords="offset points", va="center",
                     fontsize=8.4, color=ORANGE if worse else INK,
                     weight="bold" if worse else "normal")
    axL.set_yticks(ys); axL.set_yticklabels(names, fontsize=9.5, color=INK)
    axL.set_xlabel("median |ln(actual/predicted)| — lower is better",
                   fontsize=9, color=MUTED)
    axL.set_xlim(0, 0.20)
    axL.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    _clean(axL, left=False)
    axL.scatter([], [], s=52, color=MUTED, label="no correction")
    axL.scatter([], [], s=72, color=AQUA, label="fitted on ITSELF")
    axL.scatter([], [], s=72, color=BLUE, label="fitted ELSEWHERE, helps")
    axL.scatter([], [], s=72, color=ORANGE, marker="X",
                label="fitted ELSEWHERE, HURTS")
    axL.legend(fontsize=8, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 1.0))
    axL.set_title("QRB5165 — four traces, four runtime configurations",
                  fontsize=10.5, loc="left", color=INK, pad=34)

    # right: the contrast that explains it
    k1 = json.load(open(os.path.join(REPO,
                   "results/flowc_contention/k1_tune.json")))
    e = k1["leave_one_run_out_logerr"]
    qc_mean_self = st.mean([(1 - o / b) for b, o in self_fit]) * 100
    qc_mean_out = st.mean([(1 - o / b) for b, o in held_out]) * 100
    k1_out = (1 - e["kind+co"] / e["none"]) * 100
    bars = [("QRB5165\nfitted on itself", qc_mean_self, AQUA),
            ("QRB5165\nfitted elsewhere", qc_mean_out, ORANGE),
            ("K1\nheld-out runs", k1_out, BLUE)]
    xs = range(len(bars))
    axR.bar(xs, [v for _, v, _ in bars], 0.56,
            color=[c for _, _, c in bars], edgecolor=SURF, lw=1.4, zorder=3)
    for x, (_, v, _) in zip(xs, bars):
        # a negative bar's label was landing on its x tick label
        axR.annotate(f"{v:+.1f}%", (x, v),
                     xytext=(0, 6) if v >= 0 else (34, -4),
                     textcoords="offset points",
                     ha="center" if v >= 0 else "left",
                     fontsize=10.5, color=INK if v >= 0 else ORANGE,
                     weight="bold")
    axR.axhline(0, color=INK, lw=1.0)
    axR.set_xticks(list(xs))
    axR.set_xticklabels([n for n, _, _ in bars], fontsize=9, color=INK)
    axR.set_ylabel("mean error reduction (%)", fontsize=9, color=MUTED)
    axR.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    _clean(axR)
    axR.set_title("what the correction is keyed on decides it",
                  fontsize=10.5, loc="left", color=INK, pad=34)

    fig.suptitle("Finding 5 — the bias belongs to the RUN, not the silicon\n"
                 "a correction fitted on one runtime configuration does not "
                 "carry to another; one keyed on co-runners does",
                 x=0.012, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = os.path.join(out, "finding5_transfer.png")
    fig.savefig(p, dpi=160, facecolor=SURF)
    print(f"  wrote {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/Qualcomm/experiments")
    a = ap.parse_args()
    out = os.path.join(REPO, a.out)
    os.makedirs(out, exist_ok=True)
    finding1(out)
    finding4(out)
    finding5(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

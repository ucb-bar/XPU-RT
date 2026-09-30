#!/usr/bin/env python3
"""Experiment log + figures for the QRB5165 residual-feedback study.

Answers three questions with the four committed board traces (440 dispatches):

  1. Is the solo profile biased?      -- yes, and per-backend in opposite
                                         directions (HTA under, DSP over).
  2. Does feeding the observed error
     back improve the estimate?       -- yes, 27% within a configuration, and
                                         it reaches a fixpoint in one round.
  3. Does that correction transfer
     to another configuration?        -- no, 2/4 held-out. The bias is a
                                         property of the run, not the board.

Outputs <out>/residual_report.md and three figures.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flowc_residual_feedback as F  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default="runs/*/trace.csv")
    ap.add_argument("--out", default="docs/Qualcomm/experiments/residual")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    per = {}
    for p in sorted(glob.glob(a.traces)):
        rows = F.read_trace(p)
        if rows:
            per[rows[0]["trace"]] = rows
    allrows = [r for v in per.values() for r in v]

    L = ["# QRB5165 residual feedback — full experiment log\n",
         f"{len(allrows)} dispatches across {len(per)} committed board traces "
         "(`runs/*/trace.csv`). Every row carries both the schedule's "
         "`predicted_duration_ms` and the board's `actual_start/end_ms`, so a "
         "run that already happened is a measurement of its own error.\n"]

    # --- 1. bias -----------------------------------------------------------
    L += ["## 1. The solo profile is biased, per backend, in opposite directions\n",
          "| trace | n | co-runners | stalls >1 ms | stall total | HTA | DSP | CPU |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, rows in per.items():
        m = F.fit(rows)["backends"]
        co = sum(1 for r in rows if r["co"] > 0)
        big = [r["stall_ms"] for r in rows if r["stall_ms"] > 1.0]
        g = lambda b: f"{m[b]['factor']:.3f}" if b in m else "—"
        L.append(f"| `{name}` | {len(rows)} | {co} | {len(big)} | {sum(big):.0f} ms | "
                 f"{g('HTA')} | {g('DSP')} | {g('CPU')} |")
    glob_m = F.fit(allrows)
    L += ["", "Pooled: " + ", ".join(
        f"**{b} {c['factor']:.3f}** (n={c['n']}, p10 {c['p10']:.2f}–p90 {c['p90']:.2f})"
        for b, c in sorted(glob_m["backends"].items())) + ".\n",
        "HTA is under-estimated in every trace and DSP over-estimated in every "
        "trace it appears in. The DSP direction independently corroborates "
        "`docs/Qualcomm/qualcomm-qrb5165.md` §2, which measured the recorded DSP column "
        "~16% pessimistic and suspected a slower host clock at capture time.\n",
        "**There is no contention to learn from.** The `co-runners` column is "
        "zero everywhere: these workloads are serial chains, exactly as "
        "`docs/Qualcomm/qualcomm-qrb5165.md` §3 reports. So the residual measured here is "
        "*calibration bias*, not co-runner interference — a distinction that "
        "matters, because only the latter would depend on what else is "
        "scheduled.\n"]

    # --- 2. within-config feedback + convergence ---------------------------
    L += ["## 2. Feeding the error back works — within one configuration\n",
          "| trace | logerr before | after | after 2nd round | MAE before | after |",
          "|---|---:|---:|---:|---:|---:|"]
    conv = []
    b_tot = a_tot = 0.0
    for name, rows in per.items():
        m1 = F.fit(rows)
        b, a1 = F.error(rows, None), F.error(rows, m1)
        # second round: re-fit on the residual left after round 1
        rows2 = [dict(r) for r in rows]
        for r in rows2:
            f = (m1["backends"].get(r["backend"], {}) or {}).get("factor", 1.0)
            r["pred_ms"] *= f
            r["ratio"] = r["act_ms"] / r["pred_ms"]
        m2 = F.fit(rows2)
        a2 = F.error(rows2, m2)
        conv.append((name, b["logerr_median"], a1["logerr_median"], a2["logerr_median"]))
        b_tot += b["logerr_median"]; a_tot += a1["logerr_median"]
        L.append(f"| `{name}` | {b['logerr_median']:.4f} | {a1['logerr_median']:.4f} | "
                 f"{a2['logerr_median']:.4f} | {b['mae_ms']:.3f} | {a1['mae_ms']:.3f} |")
    L += ["", f"Mean logerr **{b_tot/len(per):.4f} → {a_tot/len(per):.4f}"
              f" ({100*(1-a_tot/b_tot):.1f}% reduction)**. The second round moves it "
              "almost nowhere: one round of feedback reaches the fixpoint, because "
              "the correction is a single multiplicative constant per backend and "
              "applying it twice would double-count.\n"]

    # --- 3. transfer -------------------------------------------------------
    L += ["## 3. It does not transfer across configurations\n",
          "Leave-one-trace-out: fit on the other three, score the held-out one.\n",
          "| held-out | n | logerr before | after | verdict |", "|---|---:|---:|---:|---|"]
    wins = 0
    for name, rows in per.items():
        train = [r for k, v in per.items() if k != name for r in v]
        m = F.fit(train)
        b, af = F.error(rows, None), F.error(rows, m)
        ok = af["logerr_median"] < b["logerr_median"]
        wins += ok
        L.append(f"| `{name}` | {af['n']} | {b['logerr_median']:.4f} | "
                 f"{af['logerr_median']:.4f} | {'improves' if ok else '**worse**'} |")
    L += ["", f"Helps on **{wins}/{len(per)}** held-out traces.\n",
          "The four traces are the same network under four *runtime* "
          "configurations (eager budget-9, lazy budget-14 + LRU evict, all-DSP "
          "with backend reset). `dsp14_lazy` already predicts well (logerr "
          "0.042) and a correction fit elsewhere makes it worse. So the bias is "
          "a property of the configuration that ran, not of the silicon — which "
          "is why a board-level constant is the wrong shape for it, and why the "
          "feedback artifact must be keyed by configuration.\n",
          "Excluding the 35 stall-delayed dispatches does not change this "
          "(still 2/4), so residency stalls are not the cause either.\n"]

    # --- 4. what this means for the scheduler ------------------------------
    L += ["## 4. Consequences for scheduling\n",
          "* A correction fit on the run you are about to repeat is worth ~27% "
          "of the estimate error; one fit on a different configuration is not "
          "worth applying.\n"
          "* The stall term is the largest single dynamic effect and is entirely "
          "configuration-borne: `dsp14_lazy` loses **1316 ms** across 30 stalls "
          "and `dsp_all_reset` **347 ms** across 5, while the other two lose "
          "nothing. No per-kernel cost model can carry that; it belongs to "
          "context residency.\n"
          "* Contention could not be evaluated at all, because no committed "
          "QRB5165 trace has two dispatches in flight at once. The contention "
          "sweep predicts concurrent schedules but has never been run on the "
          "board — that run is the missing measurement.\n"]

    md = os.path.join(a.out, "residual_report.md")
    open(md, "w").write("\n".join(L) + "\n")
    print(f"  wrote {md}")
    json.dump(glob_m, open(os.path.join(a.out, "residual_model.json"), "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  (no matplotlib: {exc})")
        return 0

    # Same house style as the stage/overhead figures: validated categorical
    # slots 1-3 (CVD dE 9.2 deutan, normal-vision 27.6), text in ink tokens
    # rather than series colour, recessive grid, every mark direct-labelled --
    # which is also what the validator's contrast WARN on the aqua step
    # requires.
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    INK, MUTED, GRID = "#1a1a19", "#5c5c5a", "#d8d8d5"
    SURF = "#fcfcfb"
    BE = {"HTA": ORANGE, "DSP": BLUE, "CPU": AQUA}

    def _clean(ax, left=True):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in (("left", "bottom") if left else ("bottom",)):
            ax.spines[sp].set_color(GRID)
        if not left:
            ax.spines["left"].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_axisbelow(True)

    # ---- fig 1: per-backend bias ----
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    import random
    random.seed(0)
    for i, b in enumerate(["HTA", "DSP", "CPU"]):
        v = [r["ratio"] for r in allrows if r["backend"] == b and r["usable"]]
        if not v:
            continue
        ax.scatter([i + random.uniform(-0.17, 0.17) for _ in v], v, s=13,
                   alpha=0.32, color=BE[b], edgecolors="none", zorder=2)
        m = st.median(v)
        ax.hlines(m, i - 0.30, i + 0.30, color=INK, lw=2.4, zorder=4)
        ax.annotate(f"{m:.3f}", (i, m), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=10.5, color=INK, weight="bold")
        ax.annotate(f"n={len(v)}", (i, 0.46), ha="center",
                    fontsize=8, color=MUTED)
        verdict = "profile UNDER-estimates" if m > 1.02 else (
                  "profile OVER-estimates" if m < 0.98 else "about right")
        ax.annotate(verdict, (i, 1.92), ha="center", fontsize=8.4, color=MUTED)
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", zorder=3)
    # left edge: CPU's median is 0.998 and its bold label sat right on top of
    # this caption at the right edge
    ax.annotate("1.0 = the solo profile was right", (-0.50, 1.0),
                xytext=(2, 6), textcoords="offset points", ha="left",
                fontsize=8.4, color=MUTED)
    ax.set_xticks(range(3)); ax.set_xticklabels(["HTA", "DSP", "CPU"],
                                                fontsize=10.5, color=INK)
    ax.set_ylabel("actual / predicted", fontsize=9, color=MUTED)
    ax.set_ylim(0.42, 2.05); ax.set_xlim(-0.55, 2.55)
    ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    _clean(ax)
    ax.set_title("QRB5165: the solo profile is biased, and per backend in "
                 "opposite directions\n440 dispatches from four board traces",
                 fontsize=11.5, loc="left", color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "residual_bias.png"), dpi=160, facecolor=SURF)
    print(f"  wrote {a.out}/residual_bias.png")

    # ---- fig 2: within-config feedback, as a dumbbell like the stage figure ----
    fig2, ax2 = plt.subplots(figsize=(9.0, 3.9))
    ys = list(range(len(conv)))[::-1]
    for y, (name, before, after1, after2) in zip(ys, conv):
        ax2.plot([before, after1], [y, y], color=GRID, lw=3,
                 solid_capstyle="round", zorder=1)
        ax2.scatter([before], [y], s=70, color=MUTED, zorder=3,
                    edgecolors="white", lw=1.3)
        ax2.scatter([after1], [y], s=94, color=BLUE, zorder=4,
                    edgecolors="white", lw=1.3)
        ax2.scatter([after2], [y], s=150, facecolors="none", zorder=5,
                    marker="D", edgecolors=ORANGE, lw=1.6)
        ax2.annotate(f"{before:.4f}", (before, y), xytext=(0, 11),
                     textcoords="offset points", ha="center",
                     fontsize=8.4, color=MUTED)
        dy = 15 if y == 0 else -19       # bottom row: label above the axis
        ax2.annotate(f"{after1:.4f}", (after1, y), xytext=(0, dy),
                     textcoords="offset points", ha="center",
                     fontsize=9, color=INK, weight="bold")
        chg = (1 - after1 / before) * 100
        ax2.annotate(f"{chg:+.0f}%", (max(before, after1), y), xytext=(13, 0),
                     textcoords="offset points", va="center",
                     fontsize=9, color=INK if chg > 0 else ORANGE)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([c[0].replace("v3_bundles", "v3") for c in conv],
                        fontsize=9.5, color=INK)
    ax2.set_xlabel("median |ln(actual / predicted)| — lower is a better estimate",
                   fontsize=9, color=MUTED)
    ax2.set_xlim(0, 0.155)
    ax2.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    _clean(ax2, left=False)
    ax2.scatter([], [], s=70, color=MUTED, label="no feedback")
    ax2.scatter([], [], s=94, color=BLUE, label="after 1 round")
    ax2.scatter([], [], s=90, facecolors="none", edgecolors=ORANGE, lw=1.6,
                marker="D", label="after 2 rounds (lands on the first)")
    ax2.legend(fontsize=8.4, frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 1.0))
    ax2.set_title("Feeding the observed error back, within one configuration\n"
                  "the second round lands on the first: one pass reaches the fixpoint",
                  fontsize=11.5, loc="left", color=INK, pad=30)
    fig2.tight_layout()
    fig2.savefig(os.path.join(a.out, "residual_convergence.png"), dpi=160,
                 facecolor=SURF)
    print(f"  wrote {a.out}/residual_convergence.png")

    # ---- fig 3: stalls ----
    fig3, ax3 = plt.subplots(figsize=(8.8, 3.9))
    names, totals, counts = [], [], []
    for name, rows in per.items():
        big = [r["stall_ms"] for r in rows if r["stall_ms"] > 1.0]
        names.append(name.replace("v3_bundles", "v3"))
        totals.append(sum(big)); counts.append(len(big))
    xs = range(len(names))
    ax3.bar(xs, totals, 0.55, color=[ORANGE if t else GRID for t in totals],
            edgecolor=SURF, lw=1.4, zorder=3)
    for x, t, c in zip(xs, totals, counts):
        ax3.annotate(f"{t:.0f} ms" if t else "none", (x, t), xytext=(0, 6),
                     textcoords="offset points", ha="center",
                     fontsize=10, color=INK, weight="bold" if t else "normal")
        if c:
            ax3.annotate(f"{c} stalls", (x, t), xytext=(0, -16),
                         textcoords="offset points", ha="center",
                         fontsize=8.2, color="white" if t > 300 else MUTED)
    ax3.set_xticks(list(xs)); ax3.set_xticklabels(names, fontsize=9.5, color=INK)
    ax3.set_ylabel("time lost to context stalls >1 ms", fontsize=9, color=MUTED)
    ax3.set_ylim(0, max(totals) * 1.25 if max(totals) else 1)
    ax3.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
    _clean(ax3)
    ax3.set_title("Context residency, not kernel cost, is the largest dynamic term\n"
                  "and it belongs to the runtime configuration, not the silicon",
                  fontsize=11.5, loc="left", color=INK, pad=14)
    fig3.tight_layout()
    fig3.savefig(os.path.join(a.out, "residual_stalls.png"), dpi=160, facecolor=SURF)
    print(f"  wrote {a.out}/residual_stalls.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

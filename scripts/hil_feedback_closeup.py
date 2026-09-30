#!/usr/bin/env python3
"""The feedback loop closing, on one spec, as two executed schedules side by side: the table solved
from the isolated profile (no board knowledge) and the table solved on the board-measured per-dispatch
cost, both executed on the K1. Left of each: the executed Gantt over one window; right: the executed
minus planned start of every dispatch against planned time, per hart. The isolated schedule slides
behind its plan (harts left idle waiting on hand-offs the plan timed at zero); the board-calibrated
schedule holds its windows. A third panel overlays the two drift curves.
    scripts/hil_feedback_closeup.py --spec wh_chain120_solve_h200 --isolated fba120hr0 --calibrated a120h
                                    --window 40 200 --out results/codesign_feedback/refined/hil_feedback_close_120.png
"""
from __future__ import annotations
import argparse, collections, json, os, statistics, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "xpu-rt"))
from hil_feedback_figure import planned, executed, IRS, COL, gantt   # noqa: E402
from k1_trace import ir_slot_map   # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt   # noqa: E402
RES = os.path.join(REPO, "results", "codesign_feedback")


def load(tag, nets, slotmap):
    sp = os.path.join(REPO, f"schedules/fig_{tag}_cpsat_hard_clamped.json")
    tp = os.path.join(RES, "xpurt_long", f"trace_{tag}cpsat_hardr1_other_run1.csv")
    P = {(n, i, d): (h, s0, e0) for h, s0, e0, n, i, d in planned(sp, nets)}
    E = sorted(executed(tp, slotmap), key=lambda r: r[1]); t0 = E[0][1]
    E = [(h, s0 - t0, e0 - t0, n, i, d) for h, s0, e0, n, i, d in E]
    drift = collections.defaultdict(list); busy = collections.defaultdict(float); span = E[-1][2]
    for h, s0, e0, n, i, d in E:
        busy[h] += e0 - s0
        if (n, i, d) in P:
            drift[(h)].append((P[(n, i, d)][1], s0 - P[(n, i, d)][1]))
    return E, drift, span, sum(busy.values()) / (8 * span)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="wh_chain120_solve_h200"); ap.add_argument("--isolated", default="fba120hr0"); ap.add_argument("--calibrated", default="a120h")
    ap.add_argument("--window", nargs=2, type=float, default=[40.0, 200.0]); ap.add_argument("--out", default=None); ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    spec = json.load(open(os.path.join(REPO, "data/toplevel", a.spec + ".json"))); nets = set(spec["networks"])
    slotmap = {n: ir_slot_map(json.load(open(os.path.join(REPO, IRS[n])))) for n in nets if n in IRS and os.path.exists(os.path.join(REPO, IRS[n]))}
    Ei, Di, spi, bi = load(a.isolated, nets, slotmap)
    Ec, Dc, spc, bc = load(a.calibrated, nets, slotmap)
    print(f"isolated {a.isolated}: span {spi:.0f} ms, harts {bi:.0%} busy, final drift {statistics.median([d for h in Di for _, d in Di[h] if _ > 180]):.0f} ms")
    print(f"calibrated {a.calibrated}: span {spc:.0f} ms, harts {bc:.0%} busy, final drift {statistics.median([d for h in Dc for _, d in Dc[h] if _ > 180]):.0f} ms")
    fig, ax = plt.subplots(2, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.4, 1]})
    w0, w1 = a.window
    gantt(ax[0][0], Ei, w0, w1, f"isolated profile ({a.isolated}) executed on the K1 — harts {bi:.0%} busy, span {spi:.0f} ms")
    gantt(ax[1][0], Ec, w0, w1, f"board-calibrated (per-dispatch cost, {a.calibrated}) executed on the K1 — harts {bc:.0%} busy, span {spc:.0f} ms")
    for lab, D, c in [("isolated profile", Di, "#d62728"), ("board-calibrated", Dc, "#1f9e5a")]:
        allpts = collections.defaultdict(list)
        for h in D:
            for x, y in D[h]:
                allpts[round(x / 10) * 10].append(y)
        xs = sorted(allpts); ax[0][1].plot(xs, [statistics.median(allpts[x]) for x in xs], "-o", ms=3, color=c, label=lab)
    ax[0][1].axhline(0, color="k", lw=0.5); ax[0][1].set_title("timeline drift (median over harts)", fontsize=9, loc="left")
    ax[0][1].set_ylabel("executed − planned start (ms)", fontsize=8); ax[0][1].legend(fontsize=8, frameon=False); ax[0][1].grid(ls=":", lw=0.5); ax[0][1].tick_params(labelsize=7)
    ax[1][1].axis("off")
    ax[1][1].text(0.0, 0.9, "Same spec (%s), same board, same solver.\n\nThe isolated-profile schedule assumes YOLO is\ncheap; on the board it runs long, downstream\nharts wait, and the timeline slides ~%d ms\nbehind a 200 ms plan with harts only %d%% busy.\n\nThe board-calibrated schedule (per-dispatch\nYOLO cost measured under load) keeps harts\n%d%% busy and holds every window.\n\nA global service-time multiplier fitted from the\nisolated run does not bridge them: the bad\nschedule under-contends YOLO, so the fit\nunder-weights it and re-solves the same shape." % (a.spec, statistics.median([d for h in Di for x, d in Di[h] if x > 180]), bi * 100, bc * 100),
                  transform=ax[1][1].transAxes, va="top", ha="left", fontsize=8.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COL[n], label=n) for n in sorted(nets) if n in COL], loc="upper right", fontsize=7, ncol=len(nets), frameon=False)
    fig.suptitle("Runtime feedback closes the loop: the schedule solved on measured board costs executes on time where the isolated-profile one slides", fontsize=10.5, weight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = a.out or os.path.join(RES, "refined", "hil_feedback_close_120.png")
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight"); fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight"); print("wrote", out)
    json.dump({"spec": a.spec, "isolated": {"tag": a.isolated, "span_ms": spi, "busy": bi}, "calibrated": {"tag": a.calibrated, "span_ms": spc, "busy": bc}}, open(out.replace(".png", "_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

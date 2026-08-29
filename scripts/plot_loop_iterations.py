#!/usr/bin/env python3
"""The schedule, iteration by iteration, with the verdict that ended each one.

This is the figure the loop exists to produce. Each panel is one rung: the
scheduled graph after a rewrite, on the real machine lanes, drawn from the
schedule the solver actually emitted. Underneath each panel is the verdict
`candidate_objective` returned against the baseline, and the TERM it was
decided on -- because "rejected" without the term is the thing this project
kept doing by eye, on a service-time percentage that ranks ninth of nine.

EACH RUNG NAMES THE BASELINE IT WAS JUDGED AGAINST, and it is not always the
first panel. The yolo unfuse rung was measured against a REBUILD of the
detector (`ctrl`) rather than against the shipping baseline, because those two
came off different toolchains -- judging it against panel a would credit the
rewrite with a compiler difference. `--judge-against` says so on the panel
instead of leaving it to a caption nobody reads. Panels whose instance counts
disagree are refused outright: that is two amounts of work, not two graphs.

Read the panels as a sequence: a rung that adds dispatches should visibly
change the weave, and if it does not, the rewrite did not do what the hint
asked for. `diff_dispatch_graph` proves the graph changed; this shows whether
the SCHEDULE did, which is a different question and the one that decides.

Colour is per network and shared with every other figure via `figstyle`.
Lanes are physical `hardware_target`s, so a schedule that uses more of the
machine looks like it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "xpu-rt"))
sys.path.insert(0, _HERE)

import candidate_objective as objective  # noqa: E402
import schedule_scoring as scoring  # noqa: E402
from schedule_scoring import score  # noqa: E402
import figstyle  # noqa: E402
import job_names  # noqa: E402
import schedule_trace  # noqa: E402
import trace_metrics  # noqa: E402
import workload_spec  # noqa: E402

figstyle.use()


def load(spec):
    """`LABEL=path.json` -> `(label, schedule_dict, path)`."""
    if "=" not in spec:
        raise SystemExit(f"--iteration needs LABEL=path.json, got {spec!r}")
    # rsplit, not split: a label naturally contains '=' ("dronet OC=32"), and
    # a path does not, so the LAST field is the file.
    label, path = spec.rsplit("=", 1)
    with open(path) as f:
        return label, json.load(f), path


def panel(ax, schedule, known, window_ms, title, subtitle):
    disp = schedule["dispatches"]
    lanes = sorted({d["hardware_target"] for d in disp.values()})
    y = {lane: i for i, lane in enumerate(lanes)}
    seen = {}
    for d in disp.values():
        st = float(d.get("start_time", 0.0))
        dur = float(d.get("duration", 0.0))
        if window_ms and st > window_ms:
            continue
        net = job_names.model_of(d.get("job_name", ""), known)
        c = figstyle.model_color(net)
        seen[net] = c
        ax.broken_barh([(st, max(dur, 1e-3))], (y[d["hardware_target"]] - 0.38, 0.76),
                       facecolors=c, edgecolors="none")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=4.0)
    ax.tick_params(axis="y", pad=1, length=0)
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_xlim(0, window_ms)
    ax.invert_yaxis()
    # Stats sit in the title block, verdict on its own line beneath it, so
    # neither can run into the axis or off the panel.
    ax.set_title(f"{title}\n{subtitle}", fontsize=5.4, loc="left", pad=3,
                 linespacing=1.5)
    figstyle.despine(ax)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", action="append", required=True,
                    help="LABEL=scheduled_*.json, repeatable; panel 1 is the "
                         "default baseline for every later one")
    ap.add_argument("--control", action="append", default=[], metavar="PANEL",
                    help="1-based panel index that is a CONTROL BUILD, not a "
                         "rewrite -- a rebuild of the same graph. It gets no "
                         "verdict: the objective would happily 'accept' it and "
                         "credit a toolchain difference to a rewrite that did "
                         "not happen. Later rungs may still be judged against it.")
    ap.add_argument("--judge-against", action="append", default=[],
                    metavar="PANEL=BASELINE",
                    help="1-based panel indices: '5=4' judges the fifth rung "
                         "against the fourth rather than against panel 1. Use "
                         "it whenever a rung has its own control build.")
    ap.add_argument("--windows-from", default=None)
    ap.add_argument("--critical-models", default="")
    ap.add_argument("--heavy-model", default=None)
    ap.add_argument("--window-ms", type=float, default=200.0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stem", default="k1_loop_evolution")
    ap.add_argument("--title", default="The loop, iteration by iteration")
    a = ap.parse_args()

    windows, known = {}, None
    if a.windows_from:
        windows, known = workload_spec.windows_and_names(
            json.load(open(a.windows_from)))
    critical = tuple(m.strip() for m in a.critical_models.split(",") if m.strip())

    iters = [load(s) for s in a.iteration]
    outcomes = [score(lbl, sch, windows, critical, a.heavy_model, known)[1]
                for lbl, sch, _ in iters]

    # Which panel each rung is judged against; panel 1 unless told otherwise.
    against = {i: 0 for i in range(len(iters))}
    for spec in a.judge_against:
        try:
            k, b = (int(x) - 1 for x in spec.split("=", 1))
        except ValueError:
            raise SystemExit(f"--judge-against needs PANEL=BASELINE, got {spec!r}")
        if not (0 <= b < k < len(iters)):
            raise SystemExit(
                f"--judge-against {spec}: a rung is judged against an EARLIER "
                f"panel, and both must be in 1..{len(iters)}")
        against[k] = b

    letters = [chr(ord("a") + i) for i in range(len(iters))]
    controls = set()
    for spec in a.control:
        k = int(spec) - 1
        if not (0 <= k < len(iters)):
            raise SystemExit(f"--control {spec}: no such panel")
        controls.add(k)

    verdicts = ["baseline"]
    for k in range(1, len(iters)):
        b = against[k]
        if k in controls:
            verdicts.append("control build of the same graph — no verdict")
            continue
        # Two amounts of work are not two graphs. Same check the verdict CLI
        # makes, from the same implementation.
        bi = scoring.instances_per_model(iters[b][1], known)
        ci = scoring.instances_per_model(iters[k][1], known)
        if bi != ci:
            raise SystemExit(
                f"panel {letters[k]} holds {ci} instances and its baseline "
                f"{letters[b]} holds {bi}: they were solved over different "
                f"amounts of work and no verdict between them is meaningful. "
                f"Re-solve both with --max-periodic-iters 1.")
        ok, why = objective.accept(outcomes[k], outcomes[b])
        # `why` names the deciding term and both values; the term alone is
        # what the panel needs -- the numbers are already in the stats line.
        tail = why.split("--", 1)[-1].strip()
        head = f"{'ACCEPT' if ok else 'REJECT'} vs {letters[b]}"
        # A tie names no term -- it is a rejection precisely BECAUSE no term
        # separated them -- so it reads as a clause, not as "on <term>".
        verdicts.append(f"{head} on {tail.split(':', 1)[0].strip()}"
                        if ":" in tail else f"{head} \u2014 {tail}")

    n = len(iters)
    # Built at final size: panels shrink to stay inside the page rather than
    # the figure growing past it and being scaled down later, which is what
    # turns 6 pt type into 4 pt type.
    per = min(34.0, (170.0 - 6.0) / n) * figstyle.MM
    fig, axes = plt.subplots(n, 1, figsize=(figstyle.DOUBLE_COL, per * n + 6 * figstyle.MM),
                             squeeze=False)
    seen = {}
    for i, ((lbl, sch, path), o, v) in enumerate(zip(iters, outcomes, verdicts)):
        nd = len(sch["dispatches"])
        sub = (f"{nd} dispatches · misses={o.total_misses()} · "
               f"p99={o.worst_p99():.2f} ms · makespan={o.makespan_ms:.1f} ms"
               f"\n{v}")
        seen.update(panel(axes[i][0], sch, known, a.window_ms,
                          f"{chr(97+i)}  {lbl}", sub))
        if i < n - 1:
            axes[i][0].set_xticklabels([])
    axes[-1][0].set_xlabel("Time on the K1 (ms)")
    handles = [Patch(facecolor=c, label=k) for k, c in sorted(seen.items())]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles),
               frameon=False, fontsize=5, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle(a.title, fontsize=7, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.4)
    png = figstyle.save(fig, a.stem, a.out_dir)
    print(f"wrote {png}")

    # One standalone PNG per iteration as well, so a single rung can be shown
    # on its own without cropping the composite.
    out_dir = a.out_dir or figstyle.FIGURE_DIR
    for i, ((lbl, sch, _), o, v) in enumerate(zip(iters, outcomes, verdicts)):
        f2, ax2 = plt.subplots(figsize=(figstyle.DOUBLE_COL, 38 * figstyle.MM))
        nd = len(sch["dispatches"])
        panel(ax2, sch, known, a.window_ms, lbl,
              f"{nd} dispatches · misses={o.total_misses()} · "
              f"p99={o.worst_p99():.2f} ms\n{v}")
        ax2.set_xlabel("Time on the K1 (ms)")
        f2.tight_layout(pad=0.4)
        stem = f"{a.stem}_{i}_{lbl.replace(' ', '_').replace('/', '_')}"
        print(f"wrote {figstyle.save(f2, stem, out_dir)}")
        plt.close(f2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

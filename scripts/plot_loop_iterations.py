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


#: Roles for the implementation axis. The accelerator gets vermillion because
#: it is the exception being pointed at; everything still on the vector unit is
#: the muted default. Deliberately NOT the model palette -- this figure is
#: about where a dispatch ran, not which network it belongs to, and reusing
#: model colours here would make two figures disagree about what blue means.
IMPL_COLOR = {"ime": figstyle.VERMILLION, "rvv": figstyle.SKY, None: figstyle.C_MUTED}

#: Ops the K1's MAC unit can actually execute. An "ime" COMBINATION is costed
#: from the ime_x60 profile whatever the op is, so a layernorm scheduled there
#: did not touch the MAC unit -- it fell through to the same RVV kernel. Only
#: these ops on an ime combination are genuine accelerator work, and conflating
#: the two would overstate how much of the schedule the NPU is carrying.
IME_CAPABLE_OPS = ("linear_s8", "matmul_s8")

#: Roles for the core-WIDTH axis, for a schedule solved in shard mode. A
#: sequential ramp, not the qualitative palette: the quantity being shown is
#: ordered (1 < 2 < 4 harts), and using three unrelated hues for it would say
#: the widths are categories rather than more and less of one thing.
#:
#: WHY THIS AXIS IS WORTH ITS OWN COLOURING. The per-dispatch measurement says
#: sharding gain varies 4.8x WITHIN a single model (4.02x on a wide-OC conv
#: down to 0.83x on a 1x1), so no single core width is right for a model. That
#: is a statement about the PROFILES. Whether a solver can act on it is a
#: different claim, and this is the figure that shows it: the widths the
#: scheduler actually chose, dispatch by dispatch.
WIDTH_COLOR = {1: "#BDD7E7", 2: figstyle.SKY, 4: figstyle.BLUE,
               8: figstyle.PURPLE}


def _width_of(d) -> int:
    """How many harts this dispatch was given.

    `hardware_target` is a '+'-joined core list, so the width is a count and
    not a field -- 'CPU_P#0' is one hart, 'CPU_P#0+CPU_P#1' is two. Reading it
    this way means the figure cannot disagree with the feasibility checker,
    which splits the same string the same way.
    """
    return max(1, len([p for p in str(d.get("hardware_target", "")).split("+")
                       if p]))


def _default_impl(schedule):
    """The implementation a single-impl schedule used, from its profile_hw.

    A solve with `enable_impls` off writes no per-dispatch `impl`, because
    there was nothing to choose. That is not "unknown" -- it is whichever
    backend the whole solve was costed from, and rendering it as grey
    "unspecified" beside an impl-aware panel invents an ambiguity the baseline
    does not have.
    """
    hw = (schedule.get("metadata") or {}).get("profile_hw") or {}
    names = {str(v).split("_")[0] for v in hw.values() if v}
    return names.pop() if len(names) == 1 else None


def _impl_of(d, default=None):
    """`'ime'` only when this dispatch really ran on the MAC unit."""
    impl = d.get("impl") or default
    if impl != "ime":
        return impl
    mod = d.get("module_name") or ""
    return "ime" if any(op in mod for op in IME_CAPABLE_OPS) else "rvv"


def panel(ax, schedule, known, window_ms, title, subtitle, color_by="network"):
    disp = schedule["dispatches"]
    default_impl = _default_impl(schedule) if color_by == "impl" else None
    lanes = sorted({d["hardware_target"] for d in disp.values()})
    y = {lane: i for i, lane in enumerate(lanes)}
    seen = {}
    for d in disp.values():
        st = float(d.get("start_time", 0.0))
        dur = float(d.get("duration", 0.0))
        if window_ms and st > window_ms:
            continue
        if color_by == "width":
            w = _width_of(d)
            c = WIDTH_COLOR.get(w, figstyle.C_MUTED)
            key = f"{w} hart" + ("" if w == 1 else "s")
        elif color_by == "impl":
            key = _impl_of(d, default_impl)
            c = IMPL_COLOR.get(key, figstyle.C_MUTED)
            key = {"ime": "IME (smt.vmadot, cluster 0)",
                   "rvv": "RVV"}.get(key, "unspecified")
        else:
            key = job_names.model_of(d.get("job_name", ""), known)
            c = figstyle.model_color(key)
        seen[key] = c
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
    ap.add_argument("--color-by", choices=("network", "impl", "width"),
                    default="network",
                    help="'impl' colours each bar by the implementation that "
                         "ran it rather than by which network it belongs to. "
                         "Use it for a heterogeneous schedule, where the "
                         "question is WHERE a dispatch ran, not whose it is. "
                         "'width' colours by how many harts the dispatch was "
                         "given -- for a shard-mode solve, where the question "
                         "is HOW WIDE the solver went, per dispatch.")
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

    # "baseline" names a role in a COMPARISON. With one panel there is no
    # comparison, so the word would assert a second schedule that does not
    # exist -- a single-panel figure is a picture of one solve, and saying so
    # is a blank rather than a label.
    verdicts = ["" if len(iters) == 1 else "baseline"]
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
                          f"{chr(97+i)}  {lbl}", sub, color_by=a.color_by))
        if i < n - 1:
            axes[i][0].set_xticklabels([])
    axes[-1][0].set_xlabel("Time on the K1 (ms)")
    handles = [Patch(facecolor=c, label=k) for k, c in sorted(seen.items())]
    # Legend under the title, not beside it: a long title and a wide legend
    # collided on the same line and the title lost.
    fig.legend(handles=handles, loc="upper left", ncol=len(handles),
               frameon=False, fontsize=5, bbox_to_anchor=(0.01, 0.965))
    fig.suptitle(a.title, fontsize=7, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.4)
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

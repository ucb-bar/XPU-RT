"""Plots 4-6: the candidate ladder, the switching headroom, and why it failed.

    python -m benchmarks.freshness_eval.plot_adaptive \
        --rows "results/freshness_cand/*/aggregate.csv" \
        --adaptive results/freshness_adaptive \
        --output figures/freshness_adaptive

Plot 4  the ladder: validity and soft utility against contention, one panel each.
Plot 5  the headroom bound: what switching could buy, against the validity target.
Plot 6  the selector on `step`: the risk signal it sees, and the epoch it overruns.

Two conventions are load-bearing here rather than decorative:

  * NO DUAL AXES. Plot 4 shows a rate and an instance count, which are different
    scales; they get two stacked panels sharing the x axis. A twin y axis would
    let the reader infer a crossing point that does not exist.
  * An epoch-overrunning cell is never plotted as an ordinary point. Its rate is
    measured over a longer trace with a different denominator, so drawing it on
    the same line would invite exactly the comparison the sweep spent effort
    establishing is invalid. Such points are hollow, ringed in the critical
    colour, and annotated.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

from benchmarks.freshness_eval.headroom import (  # noqa: E402
    DEFAULT_TARGETS,
    LADDER_RUNGS,
    bound,
    build_table,
    load_rows,
)
from benchmarks.freshness_eval.plot import (  # noqa: E402
    GRID,
    INK,
    INK_MUTED,
    MARKERS,
    NEUTRAL,
    SERIES,
    STATUS_CRITICAL,
    SURFACE,
    _save,
    _style,
)

# Colour follows the RUNG, not its rank in any filtered subset, so a plot that
# drops a rung does not repaint the survivors.
RUNG_COLOR = {lab: SERIES[i] for i, (lab, _) in enumerate(LADDER_RUNGS)}
RUNG_MARKER = {lab: MARKERS[i] for i, (lab, _) in enumerate(LADDER_RUNGS)}
RUNG_LABEL = {
    "C0": "C0 nominal",
    "C1": "C1 defer 12 ms",
    "C2": "C2 defer + admit 2",
    "C3": "C3 defer + admit 1",
}


def _label_right(ax, x, y, text, color, dy: float = 0.0):
    ax.annotate(text, xy=(x, y), xytext=(6, dy), textcoords="offset points",
                color=color, fontsize=8.5, va="center", fontweight="medium")


def _stagger(items, min_gap: float):
    """Nudge near-coincident label anchors apart, preserving order.

    Direct labels are the relief the palette's contrast WARN obligates, so two
    of them landing on top of each other is not cosmetic -- it removes the
    encoding that makes the series identifiable without relying on colour.
    Returns [(y, dy_points)].
    """
    order = sorted(range(len(items)), key=lambda i: items[i])
    dy = [0.0] * len(items)
    for pos in range(1, len(order)):
        lo, hi = order[pos - 1], order[pos]
        gap = (items[hi] + dy[hi] / 100.0) - (items[lo] + dy[lo] / 100.0)
        if gap < min_gap:
            dy[hi] += (min_gap - gap) * 100.0
    return dy


def plot_ladder(table, out_dir: str, *, delta: float, bursts: Sequence[int]) -> List[str]:
    """Plot 4 -- validity (top) and retained soft utility (bottom) vs contention."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.4, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.14})
    fig.patch.set_facecolor(SURFACE)

    # The offered-work reference goes down first and stays behind everything: it
    # is context, not a series, and where a rung retains all offered work its
    # line should visibly sit on top of this one.
    ob = [b for b in bursts if ("C0", b) in table]
    offered = [table[("C0", b)].soft_offered for b in ob]
    if offered:
        ax2.plot(ob, offered, color=INK_MUTED, linestyle=(0, (4, 3)),
                 linewidth=1.3, zorder=1)
        # Cornered rather than inline. Every point on this line coincides with
        # either a rung that retains all offered work or an overrun marker, so an
        # inline label has nowhere to sit without touching something else.
        ax2.annotate("dashed = offered work (y = B); a rung on it sheds nothing",
                     xy=(0.015, 0.94), xycoords="axes fraction",
                     color=INK_MUTED, fontsize=8.2, va="top")

    any_overrun = False
    label_at, label_y, label_col = [], [], []
    for lab, _cid in LADDER_RUNGS:
        cells = [(b, table[(lab, b)]) for b in bursts if (lab, b) in table]
        if not cells:
            continue
        col, mk = RUNG_COLOR[lab], RUNG_MARKER[lab]
        fit = [(b, c) for b, c in cells if c.admissible]
        # The line spans only the epoch-respecting points; a gap is honest where
        # the schedule stops being deployable.
        if fit:
            ax1.plot([b for b, _ in fit], [c.validity for _, c in fit],
                     color=col, marker=mk, markersize=6, linewidth=2, zorder=3,
                     markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax2.plot([b for b, _ in fit], [c.soft_completed for _, c in fit],
                     color=col, marker=mk, markersize=6, linewidth=2, zorder=3,
                     markeredgecolor=SURFACE, markeredgewidth=1.2)
            lb, lc = fit[-1]
            label_at.append(lb)
            label_y.append(lc.validity)
            label_col.append((RUNG_LABEL[lab], col))
        for b, c in cells:
            if c.admissible:
                continue
            any_overrun = True
            for ax, val in ((ax1, c.validity), (ax2, c.soft_completed)):
                ax.plot([b], [val], marker=mk, markersize=6.5,
                        markerfacecolor=SURFACE, markeredgecolor=STATUS_CRITICAL,
                        markeredgewidth=1.6, linestyle="none", zorder=4)

    for dy, x, y, (txt, col) in zip(_stagger(label_y, 0.055), label_at, label_y,
                                    label_col):
        _label_right(ax1, x, y, txt, col, dy=dy)

    for ax in (ax1, ax2):
        _style(ax)
        ax.set_xticks(list(bursts))
    ax1.set_ylim(-0.04, 1.04)
    ax1.set_ylabel("output-valid rate", color=INK, fontsize=10)
    ax2.set_ylabel("soft instances completed", color=INK, fontsize=10)
    ax2.set_xlabel("offered soft burst B  (YOLO instances per 300 ms epoch)",
                   color=INK, fontsize=10)

    ax1.set_title(
        f"The candidate ladder at $\\varphi$ = A0 + {delta:g} ms: validity rises "
        f"and utility falls,\nmonotonically, at every contention level",
        color=INK, fontsize=11.5, loc="left", pad=10)
    if any_overrun:
        ax1.annotate(
            "hollow red = schedule overruns the 300 ms epoch, so its rate is over\n"
            "a longer trace and is not comparable. Marker shape identifies the rung.",
            xy=(0.015, 0.045), xycoords="axes fraction", fontsize=8.2,
            color=STATUS_CRITICAL, va="bottom")
    fig.subplots_adjust(right=0.80)
    return _save(fig, out_dir, "plot4_candidate_ladder")


def plot_headroom(table, out_dir: str, *, delta: float) -> List[str]:
    """Plot 5 -- soft instances switching could gain, against the validity target."""
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    fig.patch.set_facecolor(SURFACE)

    ranges = [((0, 1, 2, 3), "bursts 0-3", SERIES[0], MARKERS[0]),
              ((0, 1, 2, 3, 4), "bursts 0-4", SERIES[1], MARKERS[1])]
    targets = [t for t in DEFAULT_TARGETS if t <= 0.95]
    for bursts, label, col, mk in ranges:
        xs, ys = [], []
        for t in targets:
            r = bound(table, target=t, bursts=bursts)
            if r.gain is None:
                continue
            xs.append(t)
            ys.append(r.gain)
        if not xs:
            continue
        ax.step(xs, ys, where="post", color=col, linewidth=2, zorder=3)
        ax.plot(xs, ys, marker=mk, markersize=5.5, linestyle="none", color=col,
                markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=4)
        # Anchored where the two series DIFFER, not at the right edge -- both end
        # at gain 0, so right-edge labels would print on top of each other.
        i = ys.index(max(ys))
        ax.annotate(label, xy=(xs[i], ys[i]), xytext=(4, 13 if ys[i] > 0 else -17),
                    textcoords="offset points", color=col, fontsize=8.5,
                    fontweight="medium", va="center")

    _style(ax)
    ax.axhline(0, color=INK_MUTED, linewidth=1.1, zorder=2)
    ax.set_ylim(-0.35, 1.6)
    ax.set_yticks([0, 1])
    ax.set_xlabel("required output-valid rate (the validity target)", color=INK,
                  fontsize=10)
    ax.set_ylabel("soft instances gained\nby switching", color=INK, fontsize=10)
    ax.set_title(
        "Plot 5 — the ceiling on adaptation: one soft instance, and only for\n"
        "loose targets. Over bursts 0-3 a single static rung is optimal everywhere.",
        color=INK, fontsize=11.5, loc="left", pad=10)
    ax.annotate("assumes perfect observation and free instant switching,\n"
                "so no real selector can exceed this",
                xy=(0.015, 0.90), xycoords="axes fraction", fontsize=8.2,
                color=INK_MUTED, va="top")
    fig.subplots_adjust(right=0.82)
    return _save(fig, out_dir, "plot5_switching_headroom")


def plot_selector_timeline(per_epoch: List[Dict[str, str]], out_dir: str, *,
                           phi: float, trajectory: str, epoch_ms: float) -> List[str]:
    """Plot 6 -- the observable the selector sees, and the epoch it overruns."""
    rows = [r for r in per_epoch
            if r["strategy"] == "adaptive" and r["trajectory"] == trajectory]
    if not rows:
        return []
    rows.sort(key=lambda r: int(r["epoch"]))
    eps = [int(r["epoch"]) for r in rows]
    risk = [float(r["max_input_age"]) / phi for r in rows]
    mk = [float(r["makespan_ms"]) for r in rows]
    bursts = [int(float(r["offered_burst"])) for r in rows]
    rung = [r["candidate_id"] for r in rows]
    short = {cid: lab for lab, cid in LADDER_RUNGS}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.8, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.16})
    fig.patch.set_facecolor(SURFACE)

    # -- panel 1: the risk signal, one epoch late, against its thresholds
    # Drawn as the observation the DECISION used: risk from epoch e is what the
    # selector reads when choosing for epoch e+1. Plotting it against e without
    # saying so would make the selector look like it ignored a visible warning.
    ax1.step(eps, risk, where="mid", color=NEUTRAL, linewidth=2, zorder=3)
    ax1.plot(eps, risk, marker="o", markersize=5, linestyle="none",
             color=NEUTRAL, markeredgecolor=SURFACE, markeredgewidth=1.1, zorder=4)
    for y, lab, col in ((0.85, "entry 0.85", SERIES[0]),
                        (1.10, "entry 1.10", SERIES[1])):
        ax1.axhline(y, color=col, linestyle=(0, (5, 3)), linewidth=1.3, zorder=2)
        _label_right(ax1, eps[-1], y, lab, col, dy=9)
    _style(ax1)
    ax1.set_yscale("log")
    ticks = sorted({round(r, 3) for r in risk} | {1.0})
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f"{t:g}" for t in ticks])
    ax1.minorticks_off()
    ax1.set_ylabel("risk = observed max input age / $\\varphi$", color=INK,
                   fontsize=10)
    # Upper-right: the left half holds the spike's vertical rise, and any wide
    # text anchored there is crossed by it.
    ax1.annotate(
        "risk is plotted at the epoch it was MEASURED in;\n"
        "the selector reads it for the NEXT epoch",
        xy=(0.42, 0.93), xycoords="axes fraction", fontsize=8.2,
        color=INK_MUTED, va="top")
    # The title says what this figure SHOWS. It does not show saturation -- the
    # `step` trajectory only visits B=0 and B=4, so the flat 1.124 across
    # B=1..4 is not in this data. That is plot 7's job, and claiming it here
    # would be a caption asserting something the axes do not contain.
    ax1.set_title(
        f"Plot 6 — the selector on `{trajectory}`: reacting one epoch late costs one\n"
        f"full epoch of a 2.7x overrun, then it escalates correctly",
        color=INK, fontsize=11.5, loc="left", pad=10)

    # -- panel 2: makespan against the epoch budget
    cols = [STATUS_CRITICAL if m > epoch_ms else RUNG_COLOR[short[c]]
            for m, c in zip(mk, rung)]
    ax2.bar(eps, mk, width=0.62, color=cols, zorder=3,
            edgecolor=SURFACE, linewidth=0.8)
    ax2.axhline(epoch_ms, color=INK, linewidth=1.4, zorder=4)
    _label_right(ax2, eps[-1], epoch_ms, f"{epoch_ms:.0f} ms epoch", INK, dy=10)
    for e, m, c, b in zip(eps, mk, rung, bursts):
        ax2.annotate(f"{short[c]}\nB={b}", xy=(e, 12), ha="center", va="bottom",
                     fontsize=7.6,
                     color=SURFACE if m > epoch_ms else INK_MUTED)
        if m > epoch_ms:
            ax2.annotate(f"{m:.0f} ms", xy=(e, m), xytext=(0, 4),
                         textcoords="offset points", ha="center",
                         fontsize=8, color=STATUS_CRITICAL, fontweight="bold")
    _style(ax2)
    ax2.set_xticks(eps)
    ax2.set_ylabel("epoch makespan (ms)", color=INK, fontsize=10)
    ax2.set_xlabel("epoch", color=INK, fontsize=10)
    fig.subplots_adjust(right=0.80)
    return _save(fig, out_dir, "plot6_selector_timeline")


def risk_table(rows: Sequence[Dict[str, str]], *, delta: float,
               bursts: Sequence[int]) -> Dict[Tuple[str, int], Tuple[bool, float]]:
    """(rung, burst) -> (admissible, risk). Built here rather than by widening
    headroom.Cell, which carries only what the bound needs."""
    out: Dict[Tuple[str, int], Tuple[bool, float]] = {}
    for lab, cid in LADDER_RUNGS:
        for b in bursts:
            hits = [r for r in rows
                    if r["policy"] == cid
                    and int(float(r["contention_level"])) == b
                    and abs(float(r["delta"]) - delta) < 1e-9]
            if not hits or hits[0].get("max_input_age") in (None, ""):
                continue
            r = hits[0]
            out[(lab, b)] = (r["fits_in_epoch"] == "True",
                             float(r["max_input_age"])
                             / float(r["freshness_window"]))
    return out


def plot_signal_saturation(risks, out_dir: str, *, bursts: Sequence[int]) -> List[str]:
    """Plot 7 -- the cause: the selector's only input stops discriminating.

    This is the figure the negative result rests on, so it is drawn from the same
    measured cells as everything else rather than illustrated.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    fig.patch.set_facecolor(SURFACE)

    # The protective rungs have NUMERICALLY IDENTICAL risk at every burst -- that
    # coincidence is the finding. Drawing three overlapping lines would hide two
    # of them under the third and put their labels on top of each other, implying
    # three distinguishable measurements where there is one. So series are grouped
    # by their risk profile and the shared label names every rung in the group.
    # Grouped on the ADMISSIBLE portion: C1 differs from C2/C3 only at B=4, where
    # it overruns and so has no line to draw. Wherever all three are deployable
    # their risk is bit-identical, which is precisely the point being made.
    groups: Dict[Tuple, List[str]] = {}
    for lab, _cid in LADDER_RUNGS:
        prof = tuple((b, risks[(lab, b)][1]) for b in bursts
                     if (lab, b) in risks and risks[(lab, b)][0])
        if prof:
            groups.setdefault(prof, []).append(lab)

    label_y, label_at, label_col = [], [], []
    for prof, labs in groups.items():
        head = labs[0]
        col, mk = RUNG_COLOR[head], RUNG_MARKER[head]
        ax.plot([b for b, _ in prof], [v for _, v in prof], color=col,
                marker=mk, markersize=6.5, linewidth=2.4, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        lb, lv = prof[-1]
        label_at.append(lb)
        label_y.append(lv)
        label_col.append(
            (RUNG_LABEL[head] if len(labs) == 1
             else " = ".join(labs) + "  (identical)", col))

    # Overrunning cells, per rung, so marker shape still identifies which.
    for lab, _cid in LADDER_RUNGS:
        for b in bursts:
            if (lab, b) not in risks or risks[(lab, b)][0]:
                continue
            ax.plot([b], [risks[(lab, b)][1]], marker=RUNG_MARKER[lab],
                    markersize=6.5, markerfacecolor=SURFACE,
                    markeredgecolor=STATUS_CRITICAL, markeredgewidth=1.6,
                    linestyle="none", zorder=4)
    # A label anchored short of the right edge sits on top of any longer line at
    # the same height -- exactly C1's situation, since its risk coincides with
    # C2/C3 over B=0..3 and only C2/C3 reach B=4. Those go ABOVE the line: below
    # it the band between 1.124 and 0.85 already holds the age=phi rule and the
    # lower entry threshold.
    right = max(label_at)
    # A hidden line needs its label to say it is hidden, or the label points at
    # nothing. C1's risk equals C2/C3's wherever C1 is admissible, so its line is
    # drawn underneath theirs and never visible.
    at_right = {y for x, y in zip(label_at, label_y) if x == right}
    for x, y, (txt, col) in zip(label_at, label_y, label_col):
        hidden = x != right and any(abs(y - yr) < 1e-9 for yr in at_right)
        _label_right(ax, x, y, txt + ("  (same values, hidden)" if hidden else ""),
                     col, dy=0 if x == right else 15)

    ax.axhline(1.0, color=INK, linewidth=1.2, zorder=2)
    _label_right(ax, bursts[-1], 1.0, "age = $\\varphi$", INK, dy=-11)
    for y, lab, col in ((0.85, "entry 0.85", SERIES[0]),
                        (1.10, "entry 1.10", SERIES[1])):
        ax.axhline(y, color=col, linestyle=(0, (5, 3)), linewidth=1.2, zorder=1)

    _style(ax)
    ax.set_yscale("log")
    # Explicit ticks: the default log minor ticks are unlabelled here and the two
    # values that matter (0.752 at B=0, 1.124 at B>=1) are what the reader must
    # be able to read off.
    ticks = [0.752, 1.0, 1.124, 2.0, 5.0]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xticks(list(bursts))
    ax.set_xlabel("offered soft burst B", color=INK, fontsize=10)
    ax.set_ylabel("risk = max input age / $\\varphi$", color=INK, fontsize=10)
    ax.set_title(
        "Plot 7 — why adaptation failed: under every protective rung the risk signal\n"
        f"is flat across B = 1..4, so the selector cannot tell 65% load from 131%",
        color=INK, fontsize=11.5, loc="left", pad=10)
    ax.annotate(
        "the protective rungs pin the worst-case age to one missed producer\n"
        "period, so the age stops reporting how much contention was offered.\n"
        "Only the hollow red points discriminate B=4 — and they belong to\n"
        "schedules that overrun, i.e. are observable only after the damage.",
        xy=(0.015, 0.93), xycoords="axes fraction", fontsize=8.2,
        color=INK_MUTED, va="top")
    fig.subplots_adjust(right=0.80)
    return _save(fig, out_dir, "plot7_signal_saturation")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="results/freshness_cand/*/aggregate.csv")
    ap.add_argument("--adaptive", default="results/freshness_adaptive")
    ap.add_argument("--output", default="figures/freshness_adaptive")
    ap.add_argument("--delta", type=float, default=20.0)
    ap.add_argument("--epoch-ms", type=float, default=300.0)
    ap.add_argument("--trajectory", default="step")
    args = ap.parse_args()

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(_REPO, p)

    rows = load_rows(_abs(args.rows))
    a0 = float(rows[0]["A0"])
    phi = a0 + args.delta
    bursts = sorted({int(float(r["contention_level"])) for r in rows})
    table = build_table(rows, LADDER_RUNGS, delta=args.delta, bursts=bursts)

    out = _abs(args.output)
    written = plot_ladder(table, out, delta=args.delta, bursts=bursts)
    written += plot_headroom(table, out, delta=args.delta)
    written += plot_signal_saturation(
        risk_table(rows, delta=args.delta, bursts=bursts), out, bursts=bursts)

    pe_path = os.path.join(_abs(args.adaptive), "adaptive_per_epoch.csv")
    if os.path.exists(pe_path):
        with open(pe_path) as f:
            per_epoch = list(csv.DictReader(f))
        written += plot_selector_timeline(
            per_epoch, out, phi=phi, trajectory=args.trajectory,
            epoch_ms=args.epoch_ms)
    else:
        print(f"skipping plot 6: {pe_path} not found "
              f"(run benchmarks.freshness_eval.adaptive first)")

    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

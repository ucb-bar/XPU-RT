#!/usr/bin/env python3
"""What sharding a dispatch across harts actually buys, per dispatch.

WHY THIS FIGURE EXISTS, AND WHY IT IS NOT A BAR OF SPEEDUPS
-----------------------------------------------------------
The multi-core profiles say the obvious thing at model level -- ffn_block 2.92x,
yolov8_nano 1.96x, dronet 1.59x on four harts -- and that number is the least
interesting one in the data. It is an average over dispatches that scale 4.02x
and dispatches that come out SLOWER than serial, and reporting only the average
hides the finding: **on every model measured here, some dispatches lose.**

That is what makes per-dispatch core-count selection a correctness question
rather than an optimisation. A scheduler told "use four harts" pays for the
losers. A scheduler holding these per-dispatch costs keeps them narrow. The
figure has to show the losers or it argues for the wrong thing.

Panel a is the measured Gantt: the same model's dispatches, laid out in
execution order at each core width, on one time axis. Dispatches that got
SLOWER are drawn in vermillion. It is a real timeline -- the pool parallelism
lives INSIDE a dispatch, so the model's dispatches are serial and their spans
are exactly the profiled costs.

Panel b is per-dispatch speedup at four harts against the serial cost, so the
eye can see that the losers are all small and the winners are all wide-OC --
i.e. that the effect has a shape and is not noise.

    python scripts/plot_multicore_scaling.py --model dronet
    python scripts/plot_multicore_scaling.py --model dronet --model yolov8_nano_64x96
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle  # noqa: E402

PROFILE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_mb", "profile")

#: The four widths, in the order they are drawn. The label is what a reader
#: needs (how many harts); the tag is what the tree is keyed on.
WIDTHS = [
    ("1 hart", "topo_0"),
    ("2 harts", "topo_0_1"),
    ("4 harts", "topo_0_1_2_3"),
    ("8 harts", "topo_0_1_2_3_4_5_6_7"),
]

#: A dispatch counts as SLOWER only past this ratio, and only if it is at
#: least RESOLVED_TICKS wide. BOTH numbers are measured, not chosen.
#:
#: HOW. Several ops in these models have NO pool path at all -- layernorm,
#: gelu, add, maxpool, the concats. Their cost at 2, 4 and 8 harts must equal
#: their cost at 1 by construction, so whatever deviation they show IS this
#: measurement's noise. Over the 90 (op, width) pairs in the three profiled
#: models:
#:
#:     serial cost      n    median dev    worst dev
#:     < 500 ticks     66        6.8%        102.9%
#:     500-2000        30        1.6%          7.4%
#:     2000-8000       12        1.2%          3.1%
#:     > 8000           9        0.3%          3.5%
#:
#: So below ~2000 ticks (83 us) this profile cannot resolve a 10% change, let
#: alone a 3% one, and above it the floor is ~3.5%. Hence: 2000 ticks and 5%.
#:
#: THE FIRST VERSION OF THIS FIGURE USED A FLAT 2% AND NO SIZE FLOOR. It
#: reported "1 of 5 ffn_block dispatches is slower" on the strength of gelu at
#: 0.97x -- an op with no pool path, which cannot have changed, at a deviation
#: the table above says is indistinguishable from nothing. A figure arguing
#: that some dispatches genuinely lose cannot afford to count noise among
#: them; that is the claim readers would check first.
SLOWER_RATIO = 1.05
RESOLVED_TICKS = 2000.0

#: How many dispatches get a name in the per-dispatch panel. A 90-dispatch
#: model cannot carry 90 tick labels at 5 pt; the ones worth naming are the
#: widest few (where the time is) and the worst regression (what the panel is
#: arguing about).
N_NAMED_WINNERS = 2


def profile_path(model: str, backend: str, quant: str, tag: str) -> str:
    spec = f"{model}_spacemit_x60_{backend}_{model}.{quant}"
    return os.path.join(PROFILE_ROOT, backend, "spacemit_x60", model,
                        f"{model}.{quant}", spec, tag, "results.csv")


def read_width(model: str, backend: str, quant: str, tag: str):
    """`[(module_name, op, shape, ticks)]` in dispatch order, or None."""
    p = profile_path(model, backend, quant, tag)
    if not os.path.exists(p):
        return None
    rows = []
    for r in csv.DictReader(open(p)):
        rows.append((r["module_name"], r["op"], r.get("shape", ""),
                     float(r["cycles"])))
    return rows


def short_op(op: str) -> str:
    """`conv2d_batchnorm2d_silu_s8` -> `conv+bn+silu`. Axis labels are 5 pt."""
    return (op.replace("_s8", "")
              .replace("conv2d_batchnorm2d_silu", "conv+bn+silu")
              .replace("conv2d_batchnorm2d", "conv+bn")
              .replace("conv2d", "conv")
              .replace("maxpool2d", "maxpool")
              .replace("upsample_nearest", "upsample")
              .replace("layernorm", "layernorm")
              .replace("linear", "linear"))


def oc_of(shape: str) -> str:
    for part in (shape or "").split(";"):
        if part.startswith("OC="):
            return part
    for part in (shape or "").split(";"):
        if part.startswith("N="):
            return part
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="model name in the profile tree; repeatable")
    ap.add_argument("--backend", default="rvv_x60")
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stem", default="k1_multicore_scaling")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    figstyle.use()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    models = []
    for m in a.model:
        widths = [(lab, read_width(m, a.backend, a.quant, tag))
                  for lab, tag in WIDTHS]
        have = [(lab, rows) for lab, rows in widths if rows]
        if len(have) < 2:
            print(f"{m}: fewer than two core widths profiled, skipping",
                  file=sys.stderr)
            continue
        models.append((m, have))
    if not models:
        print("nothing to plot", file=sys.stderr)
        return 2

    n_panels = 2 * len(models)
    fig_h = min(figstyle.MAX_HEIGHT, (34.0 * n_panels + 16.0) * figstyle.MM)
    fig, axes = plt.subplots(n_panels, 1, figsize=(figstyle.DOUBLE_COL, fig_h))
    if n_panels == 1:
        axes = [axes]

    letters = "abcdefghij"
    li = 0
    for mi, (model, have) in enumerate(models):
        # ---- panel: the measured timeline at each width --------------------
        ax = axes[2 * mi]
        serial = {k: c for k, _, _, c in have[0][1]}
        # Ticks are the profile's unit; milliseconds are the reader's. rdtime
        # runs at a fixed 24 MHz on this board, so the conversion is exact and
        # not a calibration.
        MS = 1.0 / 24000.0
        span_max = max(sum(c for _, _, _, c in rows) for _, rows in have) * MS
        for row_i, (label, rows) in enumerate(have):
            y = len(have) - 1 - row_i
            t = 0.0
            for key, op, shape, c in rows:
                s0 = serial.get(key, c)
                # A dispatch that got SLOWER is the whole point of the figure.
                slower = (row_i > 0 and s0 >= RESOLVED_TICKS
                          and c > s0 * SLOWER_RATIO)
                colour = (figstyle.VERMILLION if slower
                          else figstyle.model_color(model, figstyle.SKY))
                ax.barh(y, c * MS, left=t, height=0.62, color=colour,
                        edgecolor="white", linewidth=0.25)
                t += c * MS
            ax.text(t + span_max * 0.008, y, f"{t:.2f} ms",
                    va="center", ha="left", fontsize=5)
        ax.set_yticks(range(len(have)))
        ax.set_yticklabels([lab for lab, _ in reversed(have)])
        ax.set_xlim(0, span_max * 1.10)
        ax.set_xlabel("One inference, dispatch by dispatch (ms)")
        figstyle.despine(ax)
        ax.set_title(f"{model} — the same dispatches, at four pool widths",
                     fontsize=7, loc="left")
        figstyle.panel_label(ax, letters[li]); li += 1

        # ---- panel: per-dispatch speedup at four harts ----------------------
        ax = axes[2 * mi + 1]
        by_label = dict(have)
        # Four harts is the comparison width: it is one cluster, so it isolates
        # sharding from the cross-cluster cost that the 8-hart column also
        # carries. Fall back to the widest profiled if 4 is missing.
        four_rows = by_label.get("4 harts", have[-1][1])
        four = {k: c for k, _, _, c in four_rows}
        meta = {k: (op, shape) for k, op, shape, _ in have[0][1]}
        items = sorted(((serial[k], k) for k in serial if k in four),
                       reverse=True)
        xs, ys, cols, resolved = [], [], [], []
        for i, (s0, k) in enumerate(items):
            sp = s0 / four[k] if four[k] else 0.0
            ok = s0 >= RESOLVED_TICKS
            xs.append(i)
            ys.append(sp)
            resolved.append(ok)
            if not ok:
                cols.append(figstyle.C_MUTED)
            else:
                cols.append(figstyle.VERMILLION if sp < 1.0 / SLOWER_RATIO
                            else figstyle.SKY)
        ax.bar(xs, ys, color=cols, width=0.82, linewidth=0)
        ax.axhline(1.0, color=figstyle.BLACK, lw=0.6)
        ax.axhline(4.0, color=figstyle.C_MUTED, lw=0.4, ls=":")
        ax.text(len(xs) - 0.4, 4.0, " 4 harts, perfectly", va="center",
                ha="left", fontsize=4.5, color=figstyle.C_MUTED)

        # Name only what the panel is arguing about: the widest few, and the
        # worst regression. Everything else is a bar whose height is the point.
        named = list(range(min(N_NAMED_WINNERS, len(items))))
        losers = [j for j in range(len(ys))
                  if resolved[j] and ys[j] < 1.0 / SLOWER_RATIO]
        if losers:
            worst = min(losers, key=lambda j: ys[j])
            if worst not in named:
                named.append(worst)
        top = max(4.4, max(ys) * 1.30) if ys else 4.4
        # Two heights, alternating. The widest dispatches sit next to each
        # other by construction, so a single label row always collides.
        for n, j in enumerate(named):
            s0, k = items[j]
            op, shape = meta[k]
            ty = top * (0.94 if n % 2 == 0 else 0.74)
            ha = "left" if j < len(items) * 0.5 else "right"
            dx = len(items) * (0.02 if ha == "left" else -0.02)
            ax.annotate(f"{short_op(op)} {oc_of(shape)}  {ys[j]:.2f}x",
                        xy=(j, ys[j]), xytext=(j + dx, ty),
                        fontsize=4.5, ha=ha, va="center",
                        color=(figstyle.VERMILLION if ys[j] < 1.0 / SLOWER_RATIO
                               else figstyle.BLACK),
                        arrowprops=dict(arrowstyle="-", lw=0.35,
                                        color=figstyle.C_MUTED))

        ax.set_xticks([])
        ax.set_ylabel("Speedup on 4 harts (x)")
        ax.set_xlim(-0.8, len(xs) - 0.2)
        ax.set_ylim(0, top)
        figstyle.despine(ax)
        n_res = sum(resolved)
        n_lose = len(losers)
        ax.set_title(
            f"{model} — one bar per dispatch, widest first. "
            f"{n_lose} of the {n_res} dispatches this profile can resolve are "
            f"more than 5% SLOWER on four harts than on one "
            f"(grey: under {RESOLVED_TICKS:.0f} ticks, below the noise floor).",
            fontsize=7, loc="left")
        figstyle.panel_label(ax, letters[li]); li += 1

    fig.legend(handles=[
        Patch(facecolor=figstyle.SKY, label="faster, or unchanged"),
        Patch(facecolor=figstyle.VERMILLION,
              label="more than 5% SLOWER than the same dispatch on one hart"),
        Patch(facecolor=figstyle.C_MUTED,
              label="under 2000 ticks: below what this profile resolves"),
    ], loc="upper left", bbox_to_anchor=(0.055, 0.975), ncol=3, frameon=False,
        fontsize=5.5)

    if a.title:
        fig.suptitle(a.title, x=0.012, y=0.995, ha="left", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 0.95 if a.title else 0.985))
    print("wrote", figstyle.save(fig, a.stem, a.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())

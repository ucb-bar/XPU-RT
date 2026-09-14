#!/usr/bin/env python3
"""Two wordless diagrams for XPU-RT: a real-time OS for heterogeneous SoCs.

NO TEXT ANYWHERE -- no title, label, legend, axis or number.  Everything is carried by
geometry, colour and arrangement, so the picture survives being dropped into a slide, a
README or a poster in any language.  (Verified: the SVGs contain zero <text> elements.)

  A  STRUCTURAL   who dispatches to whom.  A clock drives one coordinator; the coordinator
                  holds a schedule -- three stages, staggered, three lengths, three
                  colours, standing clear of a deadline bar it must not cross; it fans out
                  to three visibly unlike chips; their results merge into one output whose
                  beat is as even as the clock's.
  B  TEMPORAL     what lands when.  One lane per unit, time left to right, in the cascade
                  grammar of fig_cascade.py: filled bars on a shared window, idle time
                  left visible as a grey track.  A job is a staircase down the three
                  lanes; two or three staircases are always in flight at once, which is
                  what pipelining buys.  Releases drop in from the top on a fixed period;
                  a deadline bar stands after each last stage and every job clears it by
                  the same small margin.

Geometry in B is the measured QRB5165 operating point, to scale: period 125 ms, per-unit
stages 64 / 118 / 77 ms, 259 ms end to end -- the same 259 at 125 that fig_cascade.py
draws against serial 283/283 and CPU-only 685/685.  Bars are clipped at the right edge
rather than stopped, so the window reads as a slice of something ongoing.

House style: matplotlib vectors, default fonts (unused), transparent canvas for a light
page, both .svg and .png at dpi 200.  The three chips are the make_backend_row.py icons,
placed and scaled -- a unit and its icon have to be the same object or the reader learns
the set twice.

Palette is the CVD-validated set unchanged -- CPU #7A1250, Tensor #0097B2, DSP #C77400,
ink #16161C, muted #6B6B78.  NO NEW COLOUR.  Re-checked with the alphas resolved as drawn
on a light ground (idle track #6B6B78 at 0.16 -> #E1E1E1; handoff link #16161C at 0.75 ->
#4E4E52): min OKLab dE 14.9 for normal vision and 8.6 under the worst of protan/deutan/
tritan (Vienot-Brettel), above the 8.0 floor for every pair.  Contrast vs white:
10.35 / 3.46 / 3.54 / 18.02 / 8.28, all above the 3.0 floor.  The idle track is the one
exception at 1.2:1 -- it is a gridline, not a filled shape carrying meaning, and the bars
that sit on it clear it by dE >= 30.7 under every model.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

HERE = Path(__file__).parent
COL = {"CPU": "#7A1250", "Tensor": "#0097B2", "DSP": "#C77400"}
ORDER = ("CPU", "Tensor", "DSP")
INK, DIM = "#16161C", "#6B6B78"
CAP = dict(solid_capstyle="round", solid_joinstyle="round")

# ---- measured operating point (QRB5165, INT8 policy), drawn to scale in panel B
PERIOD = 125.0
STAGE = {"CPU": 64.0, "Tensor": 118.0, "DSP": 77.0}      # sums to 259 ms end to end
OFFSET = {"CPU": 0.0, "Tensor": 64.0, "DSP": 182.0}
DEADLINE = 280.0
WINDOW = 800.0


# --------------------------------------------------------------- the chip vocabulary
# Literally the icons from make_backend_row.py, placed and scaled: a diagram unit and its
# icon have to be the same object or the reader has to learn the set twice.

def chip_icon(ax, kind, cx, cy, k, lw):
    """Draw the backend_row chip for `kind`, centred at (cx, cy); k = canvas units per
    icon unit (the icon is designed in a 100x100 box)."""
    c = COL[kind]
    u = lambda x: cx + (x - 50.0) * k
    v = lambda y: cy + (y - 50.0) * k
    ax.add_patch(FancyBboxPatch((u(22), v(22)), 56 * k, 56 * k,
                                boxstyle=f"round,pad=0,rounding_size={7.8 * k:.3f}",
                                fill=False, ec=c, lw=lw, zorder=6))
    for f in (0.30, 0.50, 0.70):
        p = 22 + 56 * f
        ax.plot([u(p), u(p)], [v(78), v(90)], color=c, lw=lw * 0.72, **CAP, zorder=5)
        ax.plot([u(p), u(p)], [v(10), v(22)], color=c, lw=lw * 0.72, **CAP, zorder=5)
        ax.plot([u(10), u(22)], [v(p), v(p)], color=c, lw=lw * 0.72, **CAP, zorder=5)
        ax.plot([u(78), u(90)], [v(p), v(p)], color=c, lw=lw * 0.72, **CAP, zorder=5)
    if kind == "CPU":                                    # plain die
        ax.add_patch(Rectangle((u(37), v(37)), 26 * k, 26 * k,
                               fill=False, ec=c, lw=lw * 0.9, zorder=7))
    elif kind == "Tensor":                               # 3x3 systolic grid
        for i in (1, 2):
            ax.plot([u(34), u(66)], [v(34 + i * 32 / 3)] * 2,
                    color=c, lw=lw * 0.55, **CAP, zorder=8)
            ax.plot([u(34 + i * 32 / 3)] * 2, [v(34), v(66)],
                    color=c, lw=lw * 0.55, **CAP, zorder=8)
        ax.add_patch(Rectangle((u(34), v(34)), 32 * k, 32 * k,
                               fill=False, ec=c, lw=lw * 0.8, zorder=7))
    else:                                                # DSP: wave in the die
        ax.add_patch(Rectangle((u(34), v(34)), 32 * k, 32 * k,
                               fill=False, ec=c, lw=lw * 0.8, zorder=7))
        x = np.linspace(37, 63, 240)
        ax.plot(u(x), v(50 + 10.5 * np.sin((x - 37) / 26 * 2 * np.pi)),
                color=c, lw=lw * 0.8, **CAP, zorder=8)


# ------------------------------------------------------------------------ panel A
def panel_structural(fig, rect):
    ax = fig.add_axes(rect)
    ax.patch.set_alpha(0.0)
    ax.set_xlim(0, 1300); ax.set_ylim(0, 560)
    ax.set_aspect("equal"); ax.axis("off")
    LWs = 4.6                                            # structural stroke, in points

    # --- the clock: what makes this periodic in the first place
    cx, cy, R = 94.0, 280.0, 54.0
    ax.add_patch(Circle((cx, cy), R, fill=False, ec=INK, lw=LWs * 1.15, zorder=4))
    for a in (0, 90, 180, 270):                          # quarter ticks
        t = np.deg2rad(a)
        ax.plot([cx + (R - 15) * np.cos(t), cx + (R - 7) * np.cos(t)],
                [cy + (R - 15) * np.sin(t), cy + (R - 7) * np.sin(t)],
                color=INK, lw=LWs * 0.8, **CAP, zorder=5)
    ax.plot([cx, cx], [cy, cy + R - 21], color=INK, lw=LWs * 0.9, **CAP, zorder=5)
    ax.plot([cx, cx + (R - 31)], [cy, cy], color=INK, lw=LWs * 0.9, **CAP, zorder=5)
    ax.add_patch(Circle((cx, cy), 6.0, facecolor=INK, ec="none", zorder=6))
    ax.add_patch(FancyArrowPatch((cx + R + 12, cy), (214, cy), arrowstyle="-|>",
                                 mutation_scale=20, lw=LWs, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=4))

    # --- the coordinator: one box, and inside it the schedule it is holding -- three
    # stages, staggered, ending inside a deadline it must not cross. Same shape as the
    # temporal panel, so the two candidates share one grammar.
    ax.add_patch(FancyBboxPatch((222, 168), 304, 224,
                                boxstyle="round,pad=0,rounding_size=22",
                                fill=False, ec=INK, lw=LWs * 1.35, zorder=6))
    x0, k = 258.0, 222.0 / DEADLINE                      # interior scale: units per ms
    for name, y in zip(ORDER, (336, 280, 224)):
        ax.add_patch(Rectangle((x0 + OFFSET[name] * k, y - 22), STAGE[name] * k, 44,
                               facecolor=COL[name], ec="none", zorder=7))
    # the deadline the schedule has to fit inside -- the same thick ink bar the temporal
    # panel uses, kept well clear of the box edge so it cannot be read as an inner border
    ax.plot([x0 + DEADLINE * k] * 2, [200, 360], color=INK, lw=LWs * 1.25,
            **CAP, zorder=8)

    # --- dispatch: one exit, three destinations, each arrow in its unit's colour
    chip_x, chip_y, K = 872.0, (430.0, 280.0, 130.0), 1.34
    for name, y in zip(ORDER, chip_y):
        ax.add_patch(FancyArrowPatch((530, 280), (chip_x - 76, y), arrowstyle="-|>",
                                     mutation_scale=19, lw=LWs * 1.05, color=COL[name],
                                     connectionstyle=f"arc3,rad={(280 - y) / 900:.3f}",
                                     shrinkA=6, shrinkB=8, zorder=5))
        chip_icon(ax, name, chip_x, y, K, LWs * 1.05)

    # --- results merge into one stream
    merge = (1072.0, 280.0)
    for name, y in zip(ORDER, chip_y):
        ax.add_patch(FancyArrowPatch((chip_x + 76, y), merge, arrowstyle="-",
                                     lw=LWs * 0.9, color=COL[name],
                                     connectionstyle=f"arc3,rad={(280 - y) / 900:.3f}",
                                     shrinkA=8, shrinkB=10, zorder=4))
    ax.add_patch(Circle(merge, 11.0, facecolor=INK, ec="none", zorder=6))
    # ...whose beat is as even as the clock's: equal gaps, no numbers needed
    ax.add_patch(FancyArrowPatch((merge[0] + 11, 280), (1290, 280), arrowstyle="-|>",
                                 mutation_scale=20, lw=LWs, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=5))
    for i in range(4):
        xt = 1112 + i * 43
        ax.plot([xt, xt], [256, 304], color=INK, lw=LWs * 1.1, **CAP, zorder=6)
    return ax


# ------------------------------------------------------------------------ panel B
def panel_temporal(fig, rect):
    """Lanes = units, x = time. Bars to scale; a job is a staircase down the lanes."""
    fw, fh = fig.get_size_inches()
    gh = rect[3] * 0.215                                 # glyph axes height, fig coords
    gw = gh * fh / fw                                    # ...made square
    ax = fig.add_axes([rect[0] + gw * 1.35, rect[1],
                       rect[2] - gw * 1.35, rect[3]])
    ax.patch.set_alpha(0.0)
    ax.set_xlim(-0.02 * WINDOW, WINDOW * 1.02)
    ax.set_ylim(3.72, -0.92)                             # lane 0 on top
    ax.axis("off")

    releases = np.arange(0.0, WINDOW, PERIOD)
    for r, name in enumerate(ORDER):
        # the lane itself: idle time has to be visible or utilisation says nothing
        ax.add_patch(Rectangle((0, r + 0.24), WINDOW, 0.52, facecolor=DIM,
                               alpha=0.16, ec="none", zorder=1))
        for s in releases:
            t0 = s + OFFSET[name]
            if t0 >= WINDOW:
                continue
            ax.add_patch(Rectangle((t0, r + 0.24), min(STAGE[name], WINDOW - t0), 0.52,
                                   facecolor=COL[name], ec="none", zorder=3))

    # handoffs: stage i ends exactly where stage i+1 begins, so the link is vertical and
    # each job reads as one staircase. Several staircases overlap -- that IS the point.
    for s in releases:
        for a, b in (("CPU", "Tensor"), ("Tensor", "DSP")):
            t = s + OFFSET[b]
            if t >= WINDOW:
                continue
            ra, rb = ORDER.index(a), ORDER.index(b)
            ax.add_patch(FancyArrowPatch((t, ra + 0.78), (t, rb + 0.22),
                                         arrowstyle="-|>", mutation_scale=9,
                                         lw=1.9, color=INK, alpha=0.75,
                                         shrinkA=0, shrinkB=0, zorder=4))

    # release: a fixed-period drop into the first stage
    for s in releases:
        ax.add_patch(FancyArrowPatch((s, -0.72), (s, -0.10), arrowstyle="-|>",
                                     mutation_scale=13, lw=2.0, color=INK,
                                     shrinkA=0, shrinkB=0, zorder=5))
    # deadline: a wall after the last stage, cleared every period by the same margin
    for s in releases:
        d = s + DEADLINE
        if d > WINDOW:
            continue
        ax.plot([d, d], [2.08, 2.92], color=INK, lw=5.2, zorder=6, **CAP)

    # time, running right
    ax.add_patch(FancyArrowPatch((0, 3.46), (WINDOW, 3.46), arrowstyle="-|>",
                                 mutation_scale=16, lw=2.2, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=5))

    # which lane is which unit -- the icon, never a word
    lo, hi = ax.get_ylim()                               # (bottom, top) -- inverted
    for r, name in enumerate(ORDER):
        f = (lo - (r + 0.5)) / (lo - hi)                 # lane centre, axes fraction
        gax = fig.add_axes([rect[0], rect[1] + rect[3] * f - gh * 0.5, gw, gh])
        gax.set_xlim(0, 100); gax.set_ylim(0, 100)
        gax.set_aspect("equal"); gax.axis("off"); gax.patch.set_alpha(0.0)
        chip_icon(gax, name, 50, 50, 1.0, 5.2)
    return ax


# ------------------------------------------------------------------------- outputs
def render(which, path, figsize):
    fig = plt.figure(figsize=figsize, dpi=200)
    fig.patch.set_alpha(0.0)
    if which == "a":
        panel_structural(fig, [0.01, 0.02, 0.98, 0.96])
    else:
        panel_temporal(fig, [0.015, 0.06, 0.97, 0.88])
    for ext in ("svg", "png"):
        fig.savefig(f"{path}.{ext}", transparent=True,
                    **({"dpi": 200} if ext == "png" else {}))
    plt.close(fig)
    print(f"[ok] {Path(path).name}.svg / .png")


render("a", HERE / "rtos_diagram_a_structural", (12.6, 5.45))
render("b", HERE / "rtos_diagram_b_temporal", (11.6, 5.0))

# both candidates on one sheet
fig = plt.figure(figsize=(12.6, 10.2), dpi=200)
fig.patch.set_alpha(0.0)
panel_structural(fig, [0.01, 0.50, 0.98, 0.48])
panel_temporal(fig, [0.015, 0.045, 0.97, 0.375])
for ext in ("svg", "png"):
    fig.savefig(HERE / f"rtos_diagram.{ext}", transparent=True,
                **({"dpi": 200} if ext == "png" else {}))
plt.close(fig)
print("[ok] rtos_diagram.svg / .png")

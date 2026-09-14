#!/usr/bin/env python3
"""D2 -- THE KEY SHEET. The shared visual vocabulary, rendered.

This is the deliverable that makes events trackable between figures: one page
that fixes every colour channel, every event glyph and the time-axis convention,
so any figure in the set can point at it instead of carrying its own legend.

Generated from vocab.py, so it cannot drift from what the figures draw.

Drawn on ONE axes in a 0-100 x 0-100 coordinate system. An earlier version mixed
fig.text() with a dozen inset axes and every block collided; a single coordinate
system makes the layout checkable by arithmetic instead of by eye.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import vocab as V

V.style()
fig = plt.figure(figsize=(7.16, 5.30))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
CL, CR, W, WR = 3.0, 52.0, 45.0, 46.5
# 1 x-unit = 7.16/100 in. A text run of n chars at p pt is about n*0.60*p/72 in,
# so it spans n*0.60*p/72 * 100/7.16 units. FIT() asserts a run stays in column.
UNIT_IN = 7.16 / 100.0


def FIT(x, s, size, right=99.5):
    w = len(s) * 0.60 * size / 72.0 / UNIT_IN
    if x + w > right:
        print(f"  [overflow {x + w - right:5.1f}u] {size:.1f}pt  {s[:56]!r}")
    return s



def H(x, y, s, size=7.8, right=99.5):
    FIT(x, s, size * 1.06, right)
    ax.text(x, y, s, fontsize=size, color=V.INK, fontweight="bold", va="top")


def S(x, y, s, size=6.1, col=None, w="normal", right=99.5):
    for ln in s.split("\n"):
        FIT(x, ln, size * (1.06 if w == "bold" else 1.0), right)
    ax.text(x, y, s, fontsize=size, color=col or V.MUTED, va="top", fontweight=w)


ax.text(CL, 98.4, "XPU-RT figure vocabulary", fontsize=13, color=V.INK,
        fontweight="bold", va="top")
S(CL, 92.8, "one colour channel per job  ·  one glyph per event  ·  one time-axis "
  "convention\ngenerated from vocab.py  ·  every colour gate reproduced by "
  "palette_check.py", 6.0)

# =================================================== LEFT COLUMN -- colour
H(CL, 88.0, "1   COMPUTE LANE    categorical, n = 3", 7.6, 49)
for i, lane in enumerate(V.LANE_ORDER):
    x = CL + i * (W / 3)
    ax.add_patch(Rectangle((x, 76.6), W / 3 - 1.6, 5.6, facecolor=V.LANE[lane],
                           linewidth=0))
    ax.text(x + (W / 3 - 1.6) / 2, 79.4, lane, ha="center", va="center", fontsize=8.4,
            color="white", fontweight="bold")
    S(x, 75.8, f"{V.LANE_MACHINE[lane]}   {V.LANE[lane]}", 5.9, V.INK2)
S(CL, 72.4, "all-pairs gate CLEAR: worst CVD dE 9.2, normal-vision dE 27.6.\n"
  "HTA is 2.74:1 on white -> the relief rule applies and every lane\n"
  "mark is DIRECT-LABELLED.", 6.0, right=49)

H(CL, 65.0, "2   RELEASE PERIOD    sequential, blue", 7.6, 49)
g = np.linspace(0, 1, 256)[None, :]
ax.imshow(g, aspect="auto", cmap=V.CMAP_PERIOD, extent=(CL, CL + W, 56.8, 61.0),
          zorder=3)
lg = lambda p: CL + W * (np.log(p) - np.log(90)) / (np.log(700) - np.log(90))
for p in (110, 150, 200, 283, 555, 685):
    ax.plot([lg(p), lg(p)], [56.8, 55.8], color=V.MUTED, lw=0.6)
    ax.text(lg(p), 55.4, f"{p}", fontsize=5.8, color=V.MUTED, ha="center", va="top")
S(CL, 52.2, "release period / command cadence (ms)\nlight = fast cadence, dark = slow",
  6.0, V.INK2, right=49)

H(CL, 47.6, "3   OBSERVATION AGE     sequential, orange", 7.6, 49)
ax.imshow(g, aspect="auto", cmap=V.CMAP_LATENCY, extent=(CL, CL + W, 39.4, 43.6),
          zorder=3)
for p in (0, 200, 400, 685):
    x = CL + W * p / 700
    ax.plot([x, x], [39.4, 38.4], color=V.MUTED, lw=0.6)
    ax.text(x, 38.0, f"{p}", fontsize=5.8, color=V.MUTED, ha="center", va="top")
S(CL, 34.8, "observation age (ms)  ·  colourbars only: no discrete steps,\n"
  "because age is normally POSITIONAL (an axis)", 6.0, V.INK2, right=49)

H(CL, 28.4, "4   REFERENCE ARMS    annotated, not re-hued", 7.6, 49)
for i, (lab, mk, col, note, per) in enumerate([
        ("ideal   0 ms age", "o", V.INK, "hollow ring · unreachable reference", 200),
        ("cpu685   QNN baseline", "X", V.STATUS["critical"],
         "status ring + word · the un-scheduled arm", 685)]):
    y = 20.8 - i * 6.2
    ax.add_patch(Rectangle((CL, y), 5.4, 4.0, facecolor=V.c_period(per), linewidth=0))
    ax.plot([CL + 2.7], [y + 2.0], marker=mk, ms=7.5, ls="none", mfc="none", mec=col,
            mew=1.4)
    S(CL + 6.6, y + 3.9, lab, 6.5, V.INK, "bold", right=49)
    S(CL + 6.6, y + 1.7, note, 5.7, right=49)

H(CL, 8.0, "5   ABSENT DATA     never a zero on the ramp", 7.6, 49)
for i, (h, sym, lab) in enumerate([(V.HATCH_INFEASIBLE, "x", "INFEASIBLE"),
                                   (V.HATCH_UNKNOWN, "?", "UNKNOWN"),
                                   (None, "", "not run")]):
    x = CL + i * (W / 3)
    ax.add_patch(Rectangle((x, 0.9), 4.0, 3.6, facecolor=V.ABSENT if h else "white",
                           edgecolor="white" if h else V.AXIS, hatch=h, linewidth=0.7))
    if sym:
        ax.text(x + 2.0, 2.7, sym, ha="center", va="center", fontsize=7.4, color=V.INK2)
    S(x + 5.0, 3.6, lab, 5.7, V.INK2, right=49)

# =================================================== RIGHT COLUMN -- events
H(CR, 88.0, "6   EVENT GLYPHS      shape, not colour", 7.8)
KIND = ["release", "complete", "actuate", "grip", "grasp", "contact", "success",
        "failure"]
MEAN = ["scheduler issues an inference; frame captured",
        "fresh action exists · age steps DOWN",
        "one control tick applies the action it holds",
        "commanded gripper crosses into closed",
        "object held across consecutive ticks",
        "external contact force becomes non-zero",
        "terminal success predicate met",
        "episode ends without it"]
for i, (k, m) in enumerate(zip(KIND, MEAN)):
    y = 82.4 - i * 4.3
    V.ev(ax, [CR + 1.4], y, k, **({"ms": 5.5} if k == "actuate" else {}))
    S(CR + 4.2, y + 1.5, V.GLYPH_LABEL[k], 6.4, V.INK, "bold")
    S(CR + 17.5, y + 1.45, m, 5.7, V.INK2)

H(CR, 46.2, "7   THE RELEASE RAIL      tracked between figures", 7.8)
for j, (lab, per, lat) in enumerate([("pipelined  118 ms", 117.6, 117.7),
                                     ("baseline  685 ms", 684.8, 684.8)]):
    y = 36.0 - j * 9.4
    x0, x1 = CR + 16.0, CR + WR - 0.5
    sc = (x1 - x0) / 1500.0
    ax.plot([x0, x1], [y, y], color=V.AXIS, lw=0.9)
    rel = np.arange(0, 1500, per)
    V.ev(ax, x0 + rel * sc, y - 1.5, "release", ms=6)
    don = rel + lat
    V.ev(ax, x0 + don[don <= 1500] * sc, y, "complete", ms=5.4)
    ax.text(x0 - 1.5, y, lab, fontsize=6.1, color=V.INK2, ha="right", va="center")
    ax.annotate("", xy=(x0 + rel[1] * sc, y - 3.4), xytext=(x0, y - 3.4),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=V.INK2,
                                mutation_scale=5))
    S((x0 + x0 + rel[1] * sc) / 2, y - 4.0, "period", 5.6)
    for s_ in rel[:2]:
        ax.plot([x0 + s_ * sc, x0 + s_ * sc], [y - 1.5, y - 3.4], color=V.MUTED, lw=0.5)
x0, x1 = CR + 16.0, CR + WR - 0.5
ax.plot([x0, x1], [18.4, 18.4], color=V.AXIS, lw=0.8)
for t in (0, 500, 1000, 1500):
    x = x0 + (x1 - x0) * t / 1500
    ax.plot([x, x], [18.4, 17.6], color=V.MUTED, lw=0.6)
    ax.text(x, 17.2, f"{t}", fontsize=5.8, color=V.MUTED, ha="center", va="top")
ax.text(x1 - 0.2, 15.6, "ms", fontsize=5.8, color=V.MUTED, va="top")
S(CR, 15.2, "x is ALWAYS wall-clock time, left to right, zero at the window or\n"
  "episode start.  ms at schedule scale, s at episode scale; the rail is\n"
  "identical in both, so one triangle is followed between them.", 5.9, V.INK2)

RULES = ["task is a FACET, never a colour", "no dual axes, ever",
         "absent is hatched, not zero", "n printed wherever it varies",
         "MEASURED / PREDICTED on the axis",
         "prose in the caption only"]
H(CR, 9.2, "8   RULES IN EVERY PANEL", 7.6)
S(CR, 5.8, "·  " + RULES[0] + "        ·  " + RULES[1] + "\n"
  "·  " + RULES[2] + "            ·  " + RULES[3] + "\n"
  "·  " + RULES[4] + "  ·  " + RULES[5], 5.6, V.INK2)

V.save(fig, "cand_d2_keysheet")

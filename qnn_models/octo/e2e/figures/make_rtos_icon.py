#!/usr/bin/env python3
"""Icon candidates for XPU-RT: a real-time OS for heterogeneous SoCs.

Three DIFFERENT ideas, not one idea at three sizes:

  A  RING       a scheduler loop -- one period, swept clockwise, passing a release token
                at twelve -- closed around three heterogeneous units.  Says: one
                coordinator, many unlike units, and it comes round again.
  B  CLOCK-CHIP a chip shell that IS a clock face.  One hand at the release; the period's
                work laid on the rim as three coloured spans of three different lengths;
                an ink dot marking the deadline, with the last span stopping short of it.
                Says: real time, on silicon, inside a budget.
  C  DISPATCH   one solid coordinator fanning out to three visibly unlike units.
                Says: who dispatches to whom, at a glance.

House style, matched to make_backend_row.py: one 100x100 box, one 8-unit stroke scaled by
role, rounded caps and joins, transparent canvas, drawn for a light page.  B reuses the
backend chip shell at exactly its geometry (die 22..78, pins to 10..90), so an XPU-RT icon
and a backend icon are the same object at the same size.  The unit vocabulary is the one
already in that row -- die, array, wave -- but as three different SILHOUETTES, because at
64 px the internal detail is gone and colour alone will not carry heterogeneity.

Palette is the CVD-validated set unchanged -- CPU #7A1250, Tensor #0097B2, DSP #C77400,
with ink #16161C for everything that is structure rather than a unit.  NO NEW COLOUR.
Re-checked as a four-colour set: min OKLab dE 23.70 for normal vision, 10.37 under the
worst of protan/deutan/tritan (Vienot-Brettel; 13.27 under Machado 2009), both above the
8.0 floor; contrast vs white 10.35 / 3.46 / 3.54 / 18.02, all above the 3.0 floor for
filled shapes.  The binding pair is CPU vs ink under protanopia, and they are never asked
to be told apart -- ink is the coordinator, colour is a unit.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

HERE = Path(__file__).parent
COL = {"CPU": "#7A1250", "Tensor": "#0097B2", "DSP": "#C77400"}
INK = "#16161C"
LW = 8.0
CAP = dict(solid_capstyle="round", solid_joinstyle="round")


# ---------------------------------------------------------------- unit glyphs
# One per backend, echoing the backend row: a die, a systolic array, a waveform.
# Deliberately three different SILHOUETTES -- at 64 px colour alone is not enough and
# internal detail is gone, so the outline has to do the work.

def u_cpu(ax, cx, cy, s, lw, c=None):            # die: a rounded square
    c = c or COL["CPU"]
    ax.add_patch(FancyBboxPatch((cx - s, cy - s), 2 * s, 2 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.30 * s:.2f}",
                                fill=False, ec=c, lw=lw, zorder=6))


def u_tensor(ax, cx, cy, s, lw, c=None):         # array: 2x2 of filled cells
    c = c or COL["Tensor"]
    o, a = 0.55 * s, 0.62 * s
    for dx in (-o, o):
        for dy in (-o, o):
            ax.add_patch(FancyBboxPatch((cx + dx - a / 2, cy + dy - a / 2), a, a,
                                        boxstyle=f"round,pad=0,rounding_size={0.22 * a:.2f}",
                                        facecolor=c, ec="none", zorder=6))


def u_dsp(ax, cx, cy, s, lw, c=None):            # wave: one full period
    c = c or COL["DSP"]
    x = np.linspace(cx - s, cx + s, 200)
    # a stroke has less ink than an area glyph, so it runs a touch heavier and taller
    # to hold the same box at the same optical weight
    ax.plot(x, cy + 0.74 * s * np.sin((x - cx + s) / (2 * s) * 2 * np.pi),
            color=c, lw=lw * 1.14, **CAP, zorder=6)


UNITS = (u_cpu, u_tensor, u_dsp)


def chip_shell(ax, c, x=20, y=20, w=60, lw=LW, pins=True):
    """The shared shell from make_backend_row.py, so this reads as the same family."""
    ax.add_patch(FancyBboxPatch((x, y), w, w,
                                boxstyle=f"round,pad=0,rounding_size={0.14 * w:.2f}",
                                fill=False, ec=c, lw=lw, zorder=4))
    if not pins:
        return
    for f in (0.30, 0.50, 0.70):
        p = x + w * f
        ax.plot([p, p], [y + w, y + w + 10], color=c, lw=lw * 0.72, **CAP, zorder=2)
        ax.plot([p, p], [y - 10, y], color=c, lw=lw * 0.72, **CAP, zorder=2)
        ax.plot([x - 10, x], [p, p], color=c, lw=lw * 0.72, **CAP, zorder=2)
        ax.plot([x + w, x + w + 10], [p, p], color=c, lw=lw * 0.72, **CAP, zorder=2)


def arc_xy(cx, cy, r, a0, a1, n=200):
    """Points on an arc, angles in CLOCK degrees: 0 = 12 o'clock, growing clockwise."""
    t = np.deg2rad(90 - np.linspace(a0, a1, n))
    return cx + r * np.cos(t), cy + r * np.sin(t)


# ------------------------------------------------------------------ candidate A
def icon_ring(ax):
    """Scheduler ring closed over three heterogeneous units."""
    R = 39.0
    a0, a1 = 34.0, 326.0                       # clock degrees, gap straddling 12
    x, y = arc_xy(50, 50, R, a0, a1)
    ax.plot(x, y, color=INK, lw=LW * 1.00, **CAP, zorder=3)
    # arrowhead closing the loop back toward 12 -- the period repeats
    hx, hy = arc_xy(50, 50, R, a1 - 5, a1)
    ax.add_patch(FancyArrowPatch((hx[0], hy[0]), (hx[-1], hy[-1]), arrowstyle="-|>",
                                 mutation_scale=25, lw=LW * 1.00, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=3))
    # the release instant the ring passes every revolution
    ax.add_patch(Circle((50, 50 + R), 5.6, facecolor=INK, ec="none", zorder=5))

    # Triangle points UP, not down. Two units up and one at six o'clock made a face:
    # square and array for eyes, the DSP wave for a mouth. Impossible to unsee at 64 px.
    for fn, ang in zip(UNITS, (0.0, 120.0, 240.0)):    # clock degrees, 120 apart
        t = np.deg2rad(90 - ang)
        fn(ax, 50 + 20.0 * np.cos(t), 47.0 + 20.0 * np.sin(t), 10.6, LW * 0.66)


# ------------------------------------------------------------------ candidate B
def icon_clock(ax):
    """A chip shell that is also a clock: period, deadline, work that must fit."""
    chip_shell(ax, INK, x=22, y=22, w=56, lw=LW)
    cx = cy = 50.0
    R = 18.0
    # the period's work, laid on the rim: three unlike units, three spans, and slack
    for name, (a, b) in zip(("CPU", "Tensor", "DSP"),
                            ((8, 98), (112, 208), (222, 292))):
        x, y = arc_xy(cx, cy, R, a, b)
        ax.plot(x, y, color=COL[name], lw=LW * 0.74, **CAP, zorder=5)
    # Deadline: a DOT on the rim, not a tick. Anything radial near a hub gets read as a
    # second clock hand no matter how it is weighted -- tried R+-7 and R-3..R+8, both
    # read as 10:10. A dot cannot be a hand, so the one radial mark left is the hand.
    dx, dy = arc_xy(cx, cy, R, 320, 320, n=1)
    ax.add_patch(Circle((dx[0], dy[0]), 5.2, facecolor=INK, ec="none", zorder=7))
    # hand, at the release
    ax.plot([cx, cx], [cy, cy + R - 4.5], color=INK, lw=LW * 0.50, **CAP, zorder=6)
    ax.add_patch(Circle((cx, cy), 4.0, facecolor=INK, ec="none", zorder=7))


# ------------------------------------------------------------------ candidate C
def icon_dispatch(ax):
    """One coordinator, three unlike units, three dispatches."""
    # solid, so the coordinator is unmistakably one thing above three
    ax.add_patch(FancyBboxPatch((30, 78), 40, 17,
                                boxstyle="round,pad=0,rounding_size=6.4",
                                facecolor=INK, ec="none", zorder=6))
    xs = (17.0, 50.0, 83.0)
    for x in xs:
        ax.add_patch(FancyArrowPatch((50, 78), (x, 46.0), arrowstyle="-|>",
                                     mutation_scale=17, lw=LW * 0.66, color=INK,
                                     connectionstyle=f"arc3,rad={(50 - x) / 240:.3f}",
                                     shrinkA=2, shrinkB=2, zorder=4))
    for fn, x, s_ in zip(UNITS, xs, (12.8, 13.2, 14.2)):
        fn(ax, x, 28.0, s_, LW * 0.66)


CANDS = [("a_ring", icon_ring), ("b_clock", icon_clock), ("c_dispatch", icon_dispatch)]


def blank(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect("equal"); ax.axis("off"); ax.patch.set_alpha(0.0)


# ---- the row of three
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), dpi=200)
fig.patch.set_alpha(0.0)
for ax, (_, fn) in zip(axes, CANDS):
    blank(ax); fn(ax)
fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98, top=0.98, bottom=0.02)
fig.savefig(HERE / "rtos_icon.svg", transparent=True)
fig.savefig(HERE / "rtos_icon.png", dpi=200, transparent=True)
plt.close(fig)
print("[ok] rtos_icon.svg / .png")

# ---- and each one alone, since a sheet of three is not an icon
for name, fn in CANDS:
    f = plt.figure(figsize=(2.4, 2.4), dpi=200)
    f.patch.set_alpha(0.0)
    ax = f.add_axes([0.01, 0.01, 0.98, 0.98])
    blank(ax); fn(ax)
    f.savefig(HERE / f"rtos_icon_{name}.svg", transparent=True)
    f.savefig(HERE / f"rtos_icon_{name}.png", dpi=200, transparent=True)
    plt.close(f)
    print(f"[ok] rtos_icon_{name}.svg / .png")

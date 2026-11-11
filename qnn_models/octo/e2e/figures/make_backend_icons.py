#!/usr/bin/env python3
"""Icon candidates for the three QRB5165 backends: CPU, DSP, Tensor (HTA).

Drawn as vectors rather than pulled from an icon set: no licence to track in a paper,
exact palette control, and they stay crisp at any size. Writes an SVG (editable) and a
PNG (quick look) with three variants per backend so one can be picked per row.

Every icon is drawn inside the same 100x100 box on the same 8-unit stroke, so they sit
together at one optical weight -- mixing icon sets is what usually makes a figure look
assembled rather than designed.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

HERE = Path(__file__).parent
COL = {"CPU": "#7A1250", "DSP": "#C77400", "Tensor": "#0097B2"}
LW = 8.0
CAP = dict(solid_capstyle="round", solid_joinstyle="round")


def chip(ax, c, pins=True, r=0.14):
    """Shared shell: rounded die + pins. Common shell = one family, not three."""
    ax.add_patch(FancyBboxPatch((22, 22), 56, 56,
                                boxstyle=f"round,pad=0,rounding_size={r*56:.1f}",
                                fill=False, ec=c, lw=LW, zorder=3))
    if not pins:
        return
    for f in (0.30, 0.50, 0.70):
        p = 22 + 56 * f
        ax.plot([p, p], [78, 90], color=c, lw=LW * 0.72, **CAP, zorder=2)
        ax.plot([p, p], [10, 22], color=c, lw=LW * 0.72, **CAP, zorder=2)
        ax.plot([10, 22], [p, p], color=c, lw=LW * 0.72, **CAP, zorder=2)
        ax.plot([78, 90], [p, p], color=c, lw=LW * 0.72, **CAP, zorder=2)


def cpu_a(ax, c):                      # four cores
    chip(ax, c)
    for dx in (0, 1):
        for dy in (0, 1):
            ax.add_patch(Rectangle((34 + dx * 19, 34 + dy * 19), 13, 13,
                                   fill=False, ec=c, lw=LW * 0.75, zorder=4))


def cpu_b(ax, c):                      # single die
    chip(ax, c)
    ax.add_patch(Rectangle((37, 37), 26, 26, fill=False, ec=c, lw=LW * 0.9, zorder=4))


def cpu_c(ax, c):                      # die + instruction pointer
    chip(ax, c)
    ax.add_patch(Rectangle((34, 34), 32, 32, fill=False, ec=c, lw=LW * 0.8, zorder=4))
    ax.plot([42, 58], [50, 50], color=c, lw=LW * 0.8, **CAP, zorder=5)
    ax.plot([52, 58, 52], [44, 50, 56], color=c, lw=LW * 0.8, fillstyle="none",
            **CAP, zorder=5)


def _wave(ax, c, x0, x1, amp, n=220, lw=LW * 0.85):
    x = np.linspace(x0, x1, n)
    ax.plot(x, 50 + amp * np.sin((x - x0) / (x1 - x0) * 2 * np.pi * 1.5),
            color=c, lw=lw, **CAP, zorder=5)


def dsp_a(ax, c):                      # continuous signal through the die
    chip(ax, c)
    _wave(ax, c, 32, 68, 12)


def dsp_b(ax, c):                      # sampled: stems from a baseline, held values
    chip(ax, c)
    ax.plot([32, 68], [50, 50], color=c, lw=LW * 0.4, alpha=0.55, **CAP, zorder=4)
    xs = np.linspace(35, 65, 5)
    ys = 50 + 14 * np.sin((xs - 32) / 36 * 2 * np.pi)
    for x, y in zip(xs, ys):
        ax.plot([x, x], [50, y], color=c, lw=LW * 0.5, **CAP, zorder=6)
        ax.add_patch(Circle((x, y), 4.2, color=c, zorder=7))


def dsp_c(ax, c):                      # filter: noisy in (left), clean out (right)
    chip(ax, c, pins=False)
    ax.plot([6, 22], [50, 50], color=c, lw=LW * 0.72, **CAP, zorder=2)
    ax.plot([78, 94], [50, 50], color=c, lw=LW * 0.72, **CAP, zorder=2)
    ax.plot([50, 50], [26, 74], color=c, lw=LW * 0.45, alpha=0.45, **CAP, zorder=4)
    rng = np.random.default_rng(3)
    x = np.linspace(29, 45, 90)
    ax.plot(x, 50 + 9 * np.sin((x - 29) / 16 * 2 * np.pi) + rng.normal(0, 3.4, x.size),
            color=c, lw=LW * 0.55, **CAP, zorder=5)
    x2 = np.linspace(55, 71, 160)
    ax.plot(x2, 50 + 11 * np.sin((x2 - 55) / 16 * 2 * np.pi), color=c,
            lw=LW * 0.85, **CAP, zorder=5)


def ten_a(ax, c):                      # 3x3 systolic grid
    chip(ax, c)
    for i in range(1, 3):
        ax.plot([34, 66], [34 + i * 32 / 3] * 2, color=c, lw=LW * 0.55, **CAP, zorder=5)
        ax.plot([34 + i * 32 / 3] * 2, [34, 66], color=c, lw=LW * 0.55, **CAP, zorder=5)
    ax.add_patch(Rectangle((34, 34), 32, 32, fill=False, ec=c, lw=LW * 0.8, zorder=4))


def ten_b(ax, c):                      # depth: three separated planes = a tensor
    chip(ax, c)
    for k, (dx, dy) in enumerate(((-7, -7), (0, 0), (7, 7))):
        ax.add_patch(Rectangle((36 + dx, 36 + dy), 22, 22, facecolor="white",
                               ec=c, lw=LW * 0.62, zorder=4 + k))


def ten_c(ax, c):                      # MAC array: grid with dataflow through it
    chip(ax, c, pins=False)
    for i in range(1, 3):
        ax.plot([36, 64], [36 + i * 28 / 3] * 2, color=c, lw=LW * 0.5, **CAP, zorder=5)
        ax.plot([36 + i * 28 / 3] * 2, [36, 64], color=c, lw=LW * 0.5, **CAP, zorder=5)
    ax.add_patch(Rectangle((36, 36), 28, 28, fill=False, ec=c, lw=LW * 0.75, zorder=4))
    # outside the shell: inside it the shell stroke swallows them
    ax.annotate("", xy=(20, 50), xytext=(4, 50),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=LW * 0.55,
                                mutation_scale=15), zorder=6)
    ax.annotate("", xy=(96, 50), xytext=(80, 50),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=LW * 0.55,
                                mutation_scale=15), zorder=6)


ROWS = [("CPU", [("cores", cpu_a), ("die", cpu_b), ("pointer", cpu_c)]),
        ("DSP", [("wave", dsp_a), ("sampled", dsp_b), ("filter", dsp_c)]),
        ("Tensor", [("grid", ten_a), ("stack", ten_b), ("MAC array", ten_c)])]

fig, axes = plt.subplots(3, 3, figsize=(7.6, 8.4), dpi=200)
fig.patch.set_alpha(0.0)
for r, (name, variants) in enumerate(ROWS):
    for k, (vname, fn) in enumerate(variants):
        ax = axes[r, k]
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_aspect("equal"); ax.axis("off"); ax.patch.set_alpha(0.0)
        fn(ax, COL[name])
        ax.set_title(f"{name} · {vname}", fontsize=10.5, color="#22222A", pad=6)
fig.subplots_adjust(hspace=0.16, wspace=0.06, top=0.965, bottom=0.02,
                    left=0.02, right=0.98)
fig.savefig(HERE / "backend_icons.svg", transparent=True)
fig.savefig(HERE / "backend_icons.png", dpi=200, transparent=True)
print(f"[ok] {HERE / 'backend_icons.svg'}")
print(f"[ok] {HERE / 'backend_icons.png'}")

#!/usr/bin/env python3
"""The three QRB5165 backends as one icon row: CPU, Tensor, DSP.

Built from the top-row chip style of make_backend_icons.py, with the two requested
edits: the plain-die icon (the one that reads as CPU) moves to the LEFT, and the
box-with-an-arrow is replaced by a box with a SINE WAVE -- which is the DSP idiom, so
that slot is coloured as DSP. Tensor takes the vacated middle.

One 100x100 box and one 8-unit stroke throughout, so the three sit at a single optical
weight. Transparent canvas; colours are the cascade/splash palette.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
COL = {"CPU": "#7A1250", "Tensor": "#0097B2", "DSP": "#C77400"}
LW = 8.0
CAP = dict(solid_capstyle="round", solid_joinstyle="round")


def chip(ax, c):
    ax.add_patch(FancyBboxPatch((22, 22), 56, 56,
                                boxstyle="round,pad=0,rounding_size=7.8",
                                fill=False, ec=c, lw=LW, zorder=3))
    for f in (0.30, 0.50, 0.70):
        p = 22 + 56 * f
        ax.plot([p, p], [78, 90], color=c, lw=LW * 0.72, **CAP, zorder=2)
        ax.plot([p, p], [10, 22], color=c, lw=LW * 0.72, **CAP, zorder=2)
        ax.plot([10, 22], [p, p], color=c, lw=LW * 0.72, **CAP, zorder=2)
        ax.plot([78, 90], [p, p], color=c, lw=LW * 0.72, **CAP, zorder=2)


def cpu(ax, c):                        # plain die
    chip(ax, c)
    ax.add_patch(Rectangle((37, 37), 26, 26, fill=False, ec=c, lw=LW * 0.9, zorder=4))


def tensor(ax, c):                     # 3x3 systolic grid
    chip(ax, c)
    for i in range(1, 3):
        ax.plot([34, 66], [34 + i * 32 / 3] * 2, color=c, lw=LW * 0.55, **CAP, zorder=5)
        ax.plot([34 + i * 32 / 3] * 2, [34, 66], color=c, lw=LW * 0.55, **CAP, zorder=5)
    ax.add_patch(Rectangle((34, 34), 32, 32, fill=False, ec=c, lw=LW * 0.8, zorder=4))


def dsp(ax, c):                        # sine wave inside the die box
    chip(ax, c)
    ax.add_patch(Rectangle((34, 34), 32, 32, fill=False, ec=c, lw=LW * 0.8, zorder=4))
    x = np.linspace(37, 63, 240)
    ax.plot(x, 50 + 10.5 * np.sin((x - 37) / 26 * 2 * np.pi), color=c,
            lw=LW * 0.8, **CAP, zorder=6)


ROW = [("CPU", cpu), ("Tensor", tensor), ("DSP", dsp)]

for tag, labelled in (("", False), ("_labelled", True)):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2 if labelled else 2.9), dpi=200)
    fig.patch.set_alpha(0.0)
    for ax, (name, fn) in zip(axes, ROW):
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_aspect("equal"); ax.axis("off"); ax.patch.set_alpha(0.0)
        fn(ax, COL[name])
        if labelled:
            ax.set_title(name, fontsize=15, fontweight="bold", color=COL[name], pad=8)
    fig.subplots_adjust(wspace=0.06, left=0.02, right=0.98,
                        top=0.88 if labelled else 0.98, bottom=0.02)
    for ext in ("svg", "png"):
        fig.savefig(HERE / f"backend_row{tag}.{ext}", transparent=True,
                    **({"dpi": 200} if ext == "png" else {}))
    plt.close(fig)
    print(f"[ok] backend_row{tag}.svg / .png")

#!/usr/bin/env python3
"""Bar plot of SmolVLA per-phase performance, and where the gains actually are.

Deliberately separates three things that are easy to conflate:

  * CPU-only baseline        -- sum of measured CPU cells per component
  * best from REAL contexts  -- best backend per tile, counting only backends
                                that have a context binary on the board
  * measured end to end      -- what the Flow C runtime actually clocked

Only the vision encoder has a realized gain. Everything else is either already
fastest on CPU (the projectors, text, state_proj) or blocked (the two experts,
whose ScatterND/Where rewrites landed but have not yet been quantized and run).
Bars for those are drawn flat on purpose -- an empty gain is the finding.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, cpu_ms, best_ms, status)
#   best_ms counts only backends with a context binary that exists.
ROWS = [
    ("vision encoder",      3172.2, 2196.9, "partitioned"),
    ("expert prefill",       583.8,  583.8, "blocked"),
    ("expert decode",        149.6,  149.6, "blocked"),
    ("text encoder",           6.4,    6.4, "cpu-optimal"),
    ("time_in projector",      5.8,    5.8, "cpu-optimal"),
    ("time_out projector",     5.4,    5.4, "cpu-optimal"),
    ("action_in projector",    4.7,    4.7, "cpu-optimal"),
    ("action_out projector",   2.1,    2.1, "cpu-optimal"),
    ("state projector",        1.3,    1.3, "cpu-optimal"),
]
COL = {"partitioned": "#1B7A4B", "blocked": "#B23A18", "cpu-optimal": "#8A97A0"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.2),
                               gridspec_kw={"width_ratios": [1.35, 1]})

# ---- panel 1: per-component baseline vs best from real contexts -------------
lbl = [r[0] for r in ROWS]
cpu = np.array([r[1] for r in ROWS])
best = np.array([r[2] for r in ROWS])
y = np.arange(len(ROWS))[::-1]

ax1.barh(y, cpu, height=.62, color="#D6DDE1", edgecolor="#B9C4CA",
         label="CPU-only baseline", zorder=2)
ax1.barh(y, best, height=.62, color=[COL[r[3]] for r in ROWS],
         edgecolor="none", label="best from real contexts", zorder=3)
ax1.set_yticks(y); ax1.set_yticklabels(lbl, fontsize=10)
ax1.set_xscale("log"); ax1.set_xlim(0.8, 6000)
ax1.set_xlabel("milliseconds (log scale), serial sum of measured cells")
ax1.set_title("Per-phase cost, and where a gain was realized", fontsize=12, pad=10)
ax1.grid(axis="x", ls=":", color="#C8D2D7", zorder=0)
ax1.set_axisbelow(True)
for yy, r in zip(y, ROWS):
    gain = r[1] / r[2]
    txt = f"{r[1]:.0f} → {r[2]:.0f} ms  ({gain:.2f}×)" if gain > 1.01 else f"{r[1]:.0f} ms  —"
    ax1.text(r[1] * 1.15, yy, txt, va="center", fontsize=8.6, color="#333")
h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("partitioned", "blocked", "cpu-optimal")]
h.insert(0, plt.Rectangle((0, 0), 1, 1, color="#D6DDE1"))
ax1.legend(h, ["CPU-only baseline", "partitioned (gain realized)",
               "blocked (rewrites done, not yet run)", "already CPU-optimal"],
           loc="lower right", fontsize=8.4, framealpha=.95)

# ---- panel 2: the vision encoder's configurations --------------------------
CFG = [
    ("published claim\n(not realizable)", 1083.6, None,   "#B23A18"),
    ("141-tile hybrid",                   2196.8, 12876.3, "#9A6206"),
    ("49-tile segments",                  2874.9,  3577.5, "#1B7A4B"),
    ("CPU-only baseline",                 3172.2, None,    "#8A97A0"),
]
x = np.arange(len(CFG)); w = .38
pred = [c[1] for c in CFG]
meas = [c[2] if c[2] else np.nan for c in CFG]
ax2.bar(x - w/2, pred, w, color=[c[3] for c in CFG], alpha=.55,
        edgecolor="none", label="predicted (from cells)", zorder=3)
ax2.bar(x + w/2, meas, w, color=[c[3] for c in CFG],
        edgecolor="none", label="measured end to end", zorder=3)
ax2.set_xticks(x); ax2.set_xticklabels([c[0] for c in CFG], fontsize=8.8)
ax2.set_ylabel("milliseconds")
ax2.set_title("Vision encoder: predicted vs measured", fontsize=12, pad=10)
ax2.grid(axis="y", ls=":", color="#C8D2D7", zorder=0); ax2.set_axisbelow(True)
ax2.legend(fontsize=8.4, framealpha=.95, loc="upper left")
for xi, c in zip(x, CFG):
    ax2.text(xi - w/2, c[1] + 180, f"{c[1]:.0f}", ha="center", fontsize=8.4, color="#333")
    if c[2]:
        ax2.text(xi + w/2, c[2] + 180, f"{c[2]:.0f}", ha="center", fontsize=8.4, color="#333")
    else:
        ax2.text(xi + w/2, 240, "not run", ha="center", fontsize=8, color="#8A97A0", rotation=90)
ax2.annotate("8699 ms of this is context\nbringup — 68% of the wall",
             xy=(1 + w/2 + .04, 11200), xytext=(1.78, 8300), fontsize=8.4,
             color="#9A6206", ha="left",
             arrowprops=dict(arrowstyle="->", color="#9A6206", lw=1.1,
                             connectionstyle="arc3,rad=-0.15"))
ax2.set_ylim(0, 14600)

fig.suptitle("SmolVLA on QRB5165 v66 — realized performance by phase",
             fontsize=14, y=.985)
fig.text(.5, .012,
         "Left: serial sum of measured cells; 'best from real contexts' counts only backends with a context binary on the board. "
         "The two experts are flat because their\nScatterND/Where rewrites are verified bit-exact but not yet quantized or run. "
         "Right: the published 1083.6 ms credits whole segments with HTA times measured on extracted convs.",
         ha="center", fontsize=8, color="#54626C")
fig.tight_layout(rect=[0, .045, 1, .96])
out = "../../plots/smolvla_phase_gains.png"
fig.savefig(out, dpi=160, facecolor="white")
print(f"  wrote {out}")

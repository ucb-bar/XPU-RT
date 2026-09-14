#!/usr/bin/env python3
"""C10v2 -- SIM STRIP, keyed to C9v2.

Same arms, the SAME episode selection rule and therefore the SAME rollouts as
cand_c9v2_episode_timeline.py, so a reader can carry an instant from one figure
into the other. Frames are sampled at fixed EPISODE TIMES (not fixed frame
indices), so column k is the same wall-clock instant in both rows even though
the two episodes are different lengths -- which is the point.

Changes from C10:
  * common absolute time grid, printed under every frame;
  * the rail sits directly under its strip with no dead band;
  * outcome glyph and the aggregate success rate on the row label, so a single
    rollout is never read as the result;
  * frames extracted locally with ffmpeg from videos already on disk.
"""
from __future__ import annotations
import glob, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import vocab as V

V.style()
ARMS = [("pipe110", "XPU-RT pipelined", 117.6, "pipe110fix"),
        ("cpu685", "QNN baseline", 684.8, "cpu685")]
TICK = 40.0
TIMES = np.array([0.0, 1.5, 3.0, 4.5, 6.0, 10.0, 16.0, 22.0])
CUR = V.curated()


def pick(arm):
    d = V.AGE_RUNS[arm][0]
    eps = sorted(int(p.split("ep")[-1][:2]) for p in
                 glob.glob(str(d / "ep*_action_age_ms.npy")))
    means = {e: float(np.nanmean(np.load(d / f"ep{e:02d}_action_age_ms.npy")))
             for e in eps}
    med = np.median(list(means.values()))
    return min(means, key=lambda e: abs(means[e] - med))


fig = plt.figure(figsize=(7.16, 3.10))
X0, X1 = 0.038, 0.995
NF = len(TIMES)
TMAX = 25.0

for k, (arm, lab, per, curated_arm) in enumerate(ARMS):
    ep = pick(arm)
    age, act, ok = V.age_trace(arm, ep)
    t = np.arange(len(age)) * TICK / 1000.0
    got = V.video_frames(arm, ep, n=2, src="runs")
    if not got:
        continue
    _, nfr = got
    import subprocess
    cache = V.HERE / ".frames" / f"runs_egg_{arm}_{ep:02d}"
    fs = sorted(cache.glob("f_*.jpg"))
    top = 0.905 - k * 0.455
    FW = (X1 - X0) / NF
    live = TIMES <= t[-1] + 1e-6
    for j, tt in enumerate(TIMES[live]):
        a = fig.add_axes([X0 + j * FW, top - 0.230, FW * 0.972, 0.230])
        a.axis("off")
        fi = int(round(tt / max(t[-1], 1e-9) * (len(fs) - 1)))
        a.imshow(mpimg.imread(fs[min(fi, len(fs) - 1)]))
        a.set_title(f"{tt:.1f} s", fontsize=6.6, color=V.INK2, pad=1.5)
    nd = int((~live).sum())
    if nd:
        j0 = int(live.sum())
        a = fig.add_axes([X0 + j0 * FW, top - 0.230, FW * nd * 0.985, 0.230])
        a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.add_patch(Rectangle((0.004, 0.02), 0.992, 0.96, facecolor="white",
                              edgecolor=V.AXIS, lw=0.8, ls=(0, (3, 2))))
        a.text(0.5, 0.60, f"episode ended at {t[-1]:.1f} s", ha="center", va="center",
               fontsize=7.6, color=V.INK, fontweight="bold")
        a.text(0.5, 0.36, f"{TIMES[-1] - t[-1]:.0f} s before the baseline finishes",
               ha="center", va="center", fontsize=6.4, color=V.MUTED)

    r = fig.add_axes([X0, top - 0.335, X1 - X0, 0.100])
    r.set_xlim(-0.2, TMAX); r.set_ylim(-1.35, 1.0); r.axis("off")
    r.plot([0, t[-1]], [0.60, 0.60], color=V.AXIS, lw=0.9)
    rel = np.arange(0, t[-1] * 1000, per) / 1000.0
    V.ev(r, rel, 0.35, "release", ms=4.6)
    V.ev(r, t[V.completions(age)], 0.60, "complete", ms=4.4)
    on, off, closed = V.gripper_events(act)
    seg, s0 = [], None
    for i, c in enumerate(closed):
        if c and s0 is None:
            s0 = i
        elif not c and s0 is not None:
            seg.append((s0, i)); s0 = None
    if s0 is not None:
        seg.append((s0, len(closed) - 1))
    for a0, a1 in seg:
        r.add_patch(Rectangle((t[a0], -0.42), t[a1] - t[a0], 0.26, facecolor=V.MUTED,
                              alpha=0.34, linewidth=0))
    if len(on):
        V.ev(r, t[on], -0.29, "grip", ms=5.0)
    V.ev(r, [t[-1] + 0.30], [0.60], "success" if ok else "failure", ms=9)
    for tt in TIMES:
        if tt <= t[-1]:
            r.plot([tt, tt], [0.60, 1.00], color=V.MUTED, lw=0.5, ls=(0, (2, 2)))
    s = json.load(open(V.AGE_RUNS[arm][0] / "summary.json"))["episodes"][ep]
    r.text(0.0, -1.05, f"{lab}   ·   period {per:.0f} ms   ·   "
           f"{s['n_inferences']} commands in {t[-1]:.1f} s   ·   "
           f"mean age {np.nanmean(age):.0f} ms   ·   "
           f"{CUR['egg'][curated_arm]['success']:.0f}% success, n=240",
           fontsize=7.0, color=V.INK, fontweight="bold", va="center")

fig.legend(handles=V.glyph_handles(["release", "complete", "grip", "success", "failure"])
           + [Rectangle((0, 0), 1, 1, facecolor=V.MUTED, alpha=0.34,
                        label="gripper closed")],
           loc="lower left", bbox_to_anchor=(X0, -0.010), ncol=6, handlelength=1.1,
           columnspacing=1.2)
V.save(fig, "cand_c10v2_sim_strip")

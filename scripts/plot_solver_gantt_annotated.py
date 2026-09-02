#!/usr/bin/env python3
"""Annotated onboard-schedule panel built from the REAL solver schedule.

Unlike the conceptual network-level lane view, this renders the actual per-hart dispatches
emitted by the XPU-RT greedy solver (schedules/scheduled_networks_k1_flight_deployed_greedy_
profiled.json) and overlays the period-window coloring + sensor/output arrows on top of them:

  CTRL  mlp_control  100 Hz  (10 ms period)  green   -> thrust+moment
  NAV   fused_full    50 Hz  (20 ms period)  purple  -> yaw-rate / waypoint
  YOLO  yolov8n      spanning perception     orange  -> gate/person boxes

Background: each NAV 20 ms window is shaded very light purple; each CTRL 10 ms window very light
green drawn ON TOP (so the nested ctrl-in-nav period structure reads).  Red arrows from the top
mark each SENSOR input (labelled with the sensor); a network-coloured arrow at the bottom marks
each OUTPUT (labelled with what it emits).  Sharded dispatches are drawn across every physical
hart they occupy, so the YOLO/NAV sharding is visible.
"""
import argparse, json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

C_CTRL, C_NAV, C_YOLO = "#2f8f4e", "#7b52c0", "#e8823a"
LP, LG = "#efe8fa", "#e6f4ec"          # very light purple (nav) / green (ctrl) period fills
HARTS = ["CPU_P#0", "CPU_P#1", "CPU_P#2", "CPU_P#3",
         "CPU_E#0", "CPU_E#1", "CPU_E#2", "CPU_E#3"]


def net_of(job):
    if job.startswith("mlp_control"): return "ctrl"
    if job.startswith("fused_full"):  return "nav"
    if job.startswith("yolov8"):      return "yolo"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", default="schedules/scheduled_networks_k1_flight_deployed_greedy_profiled.json")
    ap.add_argument("--window-ms", type=float, default=70.0)
    ap.add_argument("--nav-period", type=float, default=20.0)
    ap.add_argument("--ctrl-period", type=float, default=10.0)
    ap.add_argument("--out", default="results/hil_figures/solver_gantt_annotated")
    ap.add_argument("--title", default="Onboard K1 schedule — XPU-RT greedy solver (measured K1 profile)")
    ap.add_argument("--desc", default="real XPU-RT greedy schedule, K1 measured profile")
    ap.add_argument("--yolo-deadline", type=float, default=0.0, help="ms; draw YOLO per-frame deadline + miss status (0=off)")
    ap.add_argument("--crash-note", action="store_true", help="add the diverging-backlog / crash annotation")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    W = args.window_ms
    sched = json.load(open(os.path.join(root, args.sched)))
    disp = list(sched["dispatches"].values())
    out = os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    col = {"ctrl": C_CTRL, "nav": C_NAV, "yolo": C_YOLO, "other": "#8a94a3"}
    yof = {h: i for i, h in enumerate(HARTS)}
    lane_h = 0.74
    fig, ax = plt.subplots(figsize=(12.4, 4.0))

    # ---- period-window background: nav 20 ms (light purple) then ctrl 10 ms (green on top) ----
    k = 0
    while k * args.nav_period < W:
        ax.axvspan(k * args.nav_period, min((k + 1) * args.nav_period, W), color=LP, zorder=0.1)
        k += 1
    k = 0
    while k * args.ctrl_period < W:                      # alternate so the 10 ms nesting is legible
        if k % 2 == 0:
            ax.axvspan(k * args.ctrl_period, min((k + 1) * args.ctrl_period, W),
                       color=LG, alpha=0.55, zorder=0.2)  # keep the purple base showing through
        k += 1
    for x in range(0, int(W) + 1, int(args.nav_period)):     # nav deadline lines
        ax.axvline(x, color=C_NAV, lw=0.9, ls=(0, (5, 3)), alpha=0.45, zorder=0.3)
    for x in range(0, int(W) + 1, int(args.ctrl_period)):    # ctrl deadline lines
        ax.axvline(x, color=C_CTRL, lw=0.7, ls=(0, (2, 2)), alpha=0.45, zorder=0.3)

    # ---- real dispatch bars (colored by network; sharded bars span their physical harts) ----
    used = set()
    for x in disp:
        for h in re.split(r"\+", x["hardware_target"]):
            if h in yof and x["start_time"] < W:
                used.add(h)
    for h in HARTS:                                   # mark idle harts (ROS leaves most empty)
        if h not in used:
            y = yof[h]
            ax.axhspan(y - lane_h / 2, y + lane_h / 2, color="#f4d7d7", alpha=0.6, zorder=0.5)
            ax.text(W * 0.5, y, "idle", color="#b23b3b", fontsize=7, ha="center", va="center",
                    style="italic", zorder=3.2)
    for x in disp:
        s, d = x["start_time"], x["duration"]
        if s >= W: continue
        d = min(d, W - s)
        n = net_of(x["job_name"]); c = col[n]
        phys = [h for h in re.split(r"\+", x["hardware_target"]) if h in yof]
        if not phys: continue
        sharded = len(phys) > 1
        # CTRL dispatches are ~0.08 ms: widen a floor + draw on top so the 100 Hz lane is visible
        minw = 0.55 if n == "ctrl" else 0.12
        zz = 4.0 if n == "ctrl" else 3.0
        for h in phys:
            y = yof[h]
            ax.add_patch(Rectangle((s, y - lane_h / 2), max(d, minw), lane_h,
                                   fc=c, ec=("#1c5a30" if n == "ctrl" else "white"),
                                   lw=(0.7 if n == "ctrl" else 0.5),
                                   hatch="////" if sharded else None, zorder=zz))
        if sharded:                                       # bracket linking the sharded lanes
            ys = [yof[h] for h in phys]
            ax.plot([s, s], [min(ys) - lane_h / 2, max(ys) + lane_h / 2],
                    color=c, lw=1.4, alpha=0.5, zorder=3.1)

    # ---- sensor-input arrows (red, from top) at each release; output arrows (colored, at bottom)
    ytop, ybot = len(HARTS) - 0.5 + 0.35, -0.5 - 0.35

    def sensor_tick(x):
        ax.add_patch(FancyArrowPatch((x, ytop + 0.42), (x, ytop + 0.02),
                     arrowstyle="-|>", mutation_scale=9, color="#d62728", lw=1.3, zorder=6))

    def output_tick(x, c):
        ax.add_patch(FancyArrowPatch((x, ybot - 0.02), (x, ybot - 0.42),
                     arrowstyle="-|>", mutation_scale=9, color=c, lw=1.5, zorder=6))

    # per-job release/finish from the actual schedule
    rel = {}          # job -> (min start, max end, net)
    for x in disp:
        j = x["job_name"]; s = x["start_time"]; e = s + x["duration"]
        if j not in rel: rel[j] = [s, e, net_of(j)]
        rel[j][0] = min(rel[j][0], s); rel[j][1] = max(rel[j][1], e)
    for j, (s, e, n) in sorted(rel.items(), key=lambda kv: kv[1][0]):
        if s >= W: continue
        sensor_tick(s)
        output_tick(min(e, W), col[n])

    # staggered legend-style labels (no per-tick text, so nothing collides)
    ax.text(0.5, ytop + 0.50, "sensors in ↓", color="#d62728", fontsize=6.6, ha="left",
            va="bottom", weight="bold")
    for fx, txt in [(0.16, "IMU·flow → CTRL (every 10 ms)"), (0.47, "FPV·ToF → NAV (every 20 ms)"),
                    (0.80, "FPV → YOLO")]:
        ax.text(fx * W, ytop + 0.50, txt, color="#d62728", fontsize=6.3, ha="left", va="bottom")
    ax.text(0.5, ybot - 0.50, "outputs ↑", color="0.25", fontsize=6.6, ha="left", va="top", weight="bold")
    for fx, txt, c in [(0.14, "CTRL → thrust+moment", C_CTRL), (0.45, "NAV → yaw-rate", C_NAV),
                       (0.75, "YOLO → gate/person boxes", C_YOLO)]:
        ax.text(fx * W, ybot - 0.50, txt, color=c, fontsize=6.3, ha="left", va="top", weight="bold")

    # ---- YOLO deadline / miss status (shows why ROS diverges vs why XPU-RT holds) ----
    top_pad = 0.95
    if args.yolo_deadline > 0:
        P = args.yolo_deadline
        yf = {}
        for x in disp:
            if net_of(x["job_name"]) != "yolo": continue
            j = x["job_name"]; s = x["start_time"]; e = s + x["duration"]
            yf.setdefault(j, [s, e]); yf[j][0] = min(yf[j][0], s); yf[j][1] = max(yf[j][1], e)
        frames = sorted(yf.values())
        ystat = ytop + 1.02                        # above the sensor-label row (ytop+0.50)
        any_miss = False
        for k, (s, e) in enumerate(frames):
            dl = (k + 1) * P                       # frame k released at k*P, due at (k+1)*P
            if dl > W + 6: break
            ax.axvline(dl, color="#d62728", lw=1.3, ls=(0, (3, 2)), alpha=0.85, zorder=5.5)
            lag = e - dl
            if lag > 0.2:                          # MISSED: bracket the overrun, growing each frame
                any_miss = True
                ax.annotate("", xy=(min(e, W), ystat), xytext=(dl, ystat),
                            arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.4), zorder=6.5)
                ax.text((dl + min(e, W)) / 2, ystat + 0.08, f"✗ +{lag:.0f} ms", color="#d62728",
                        fontsize=6.6, ha="center", va="bottom", weight="bold", zorder=6.5)
            else:                                  # MET: report the slack
                ax.text(e, ystat, f"✓ {-lag:.0f} ms slack", color="#1c7a3a", fontsize=6.4,
                        ha="center", va="center", weight="bold", zorder=6.5)
        ax.text(0.5, ystat, "YOLO deadlines (red dashed):", color="#d62728", fontsize=6.4,
                ha="left", va="center", weight="bold")
        top_pad = 1.55
        if any_miss and args.crash_note:
            ax.text(W * 0.53, (len(HARTS) - 1) / 2.0,
                    "serial YOLO overruns its 22 ms budget EVERY frame\n→ perception backlog grows without bound → CRASH",
                    color="#b3121b", fontsize=11, ha="center", va="center", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.85, ec="#d62728", lw=1.8),
                    zorder=7)

    ax.set_yticks(range(len(HARTS)))
    ax.set_yticklabels(HARTS, fontsize=7.5)
    ax.set_ylim(ybot - 0.95, ytop + top_pad)
    ax.set_xlim(0, W)
    ax.set_xlabel("onboard schedule time (ms)   —   " + args.desc, fontsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    # cluster divider (P-cores IME-capable, above; E-cores below)
    ax.axhline(3.5, color="0.55", lw=0.8, ls=":", zorder=2)
    ax.text(W * 0.992, 3.5, "cluster0 (IME) ↑  cluster1 ↓", fontsize=6.5, ha="right", va="center",
            color="0.35", zorder=6, bbox=dict(fc="white", alpha=0.78, ec="none", pad=0.6))
    ax.set_title(args.title + "  ·  red = sensor in, coloured = output",
                 fontsize=9.5, weight="bold")
    ax.legend(handles=[Patch(fc=C_CTRL, label="CTRL mlp 100 Hz"),
                       Patch(fc=C_NAV, label="NAV fused 50 Hz"),
                       Patch(fc=C_YOLO, label="YOLO (sharded ⁄⁄)"),
                       Patch(fc=LP, label="NAV 20 ms window"),
                       Patch(fc=LG, label="CTRL 10 ms window")],
              loc="upper right", fontsize=6.4, framealpha=0.92, ncol=5,
              bbox_to_anchor=(1.0, 1.14))
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=180, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("wrote", out + ".{png,pdf}")


if __name__ == "__main__":
    main()

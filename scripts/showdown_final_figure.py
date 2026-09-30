#!/usr/bin/env python3
"""The final showdown figure: from one flight to the whole programme, in reading order.

  S      the two runtimes on one board (schematic)
  A B C  the specific case: the same scene, both runtimes · the control-rate envelope (rate injected, colour = cruise)
         · the same floor on an unseen course
  a–d    the four moments of that pair (chase / FPV + YOLO / cross-ToF)
  D–G    inside the pair: body rate, nav goal heading, forward speed, XPU-RT's velocity along the aisle
  H H′   two scenes each flown 36 times: every success and collision of both runtimes over the obstacles
  I–L    across seeds and speeds: success, paired seeds, where every flight ends, the mechanism
  M–Q    across conditions: the ablation forest — and why, on the board: camera rate, where the latency goes,
         added load, where the work lands
  R      the schedules themselves: four measured Gantt rows
Every number is read at render time; the sidecar <out>_metrics.json lists them.
    scripts/showdown_final_figure.py [--cell tall1005] [--dpi 300]
"""
from __future__ import annotations
import argparse, collections, json, math, os, re, sys, datetime
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
os.environ.setdefault("ENERGY_CSV", os.path.join(RES, "flight_energy_v2.csv"))
sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "sims", "scripts"))
import matplotlib; matplotlib.use("Agg")   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch   # noqa: E402
import showdown_v3_figure as V   # noqa: E402
import showdown_atlas as AT
import figure_constants as FC
import measured_timing as MT   # noqa: E402
import story_figures as SF   # noqa: E402
import showdown_gatecourse as S   # noqa: E402
from hil_story_figure import draw_mechanism, draw_generalization   # noqa: E402
from hil_envelope_panel import draw_envelope, wilson   # noqa: E402

INK = "#22242a"; C_XPU, C_XPU2, C_ROS, C_ROS8, C_P3 = V.C_XPU, V.C_XPU2, V.C_ROS, V.C_ROS8, V.C_P3


def sec(ax, s, fz, dx=-22, dy=26):
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", fontsize=fz["badge"], weight="bold", color="white",
                ha="center", va="center", zorder=40, annotation_clip=False, bbox=dict(boxstyle="circle,pad=0.32", fc="#2f6db0", ec="white", lw=1.8))


def shrink(ax, k, cax=None):
    """bring an imported panel's authored-big fonts down to this figure's scale."""
    for t in ax.texts + [ax.xaxis.label, ax.yaxis.label, ax.title]:
        t.set_fontsize(t.get_fontsize() * k)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontsize(t.get_fontsize() * k)
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontsize(t.get_fontsize() * k)
    if cax is not None:
        cax.yaxis.label.set_fontsize(cax.yaxis.label.get_fontsize() * k)
        for t in cax.get_yticklabels():
            t.set_fontsize(t.get_fontsize() * k)


def draw_system(ax, fz):
    """the two runtimes on the K1 as one left-to-right flow — few words, the measured numbers large: how each is written,
    where it places work, what the board measures, and how that reaches the flights below (centre). Every number comes
    from the registry (`figure_constants`) or the camera-rate points read from the traces."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ft = fz["tick"] * 1.05
    A_, R_ = "xpu_a_cpsat_hard.csv", "ros_vanilla445.csv"
    cp = V.xpu_rate_points("cpsat"); on_time_to = max((h for h, v in cp.items() if v.get("late") == 0), default=None)
    if on_time_to is None:
        raise FC.MissingMeasurement("no CP-SAT camera-rate point is on time; the schematic cannot state a ceiling")
    goals = FC.goals_per_s("vanilla4", 45); goals90 = FC.goals_per_s("vanilla4", 90)
    def box(x, y, w, h, fc, ec=INK, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec, lw=lw, zorder=2))
    def txt(x, y, t, fs=None, weight="normal", col=INK, ha="center", va="center"):
        ax.text(x, y, t, ha=ha, va=va, fontsize=fs or ft, weight=weight, color=col, zorder=3, linespacing=1.35)
    def arrow(x0, y0, x1, y1, col=INK, lw=1.3):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=col, lw=lw, zorder=4, shrinkA=0, shrinkB=0))
    T, B = (52, 96), (4, 44)   # top row (how it is written, how work is placed, the shared kernels) · bottom row (what the board measures, the replay)
    # how each runtime is written
    box(0, T[0], 19, T[1] - T[0], "#fdeceb", ec=C_ROS, lw=1.4); txt(9.5, T[1] - 7, "ROS 2, as one writes it", fz["title"], "bold", C_ROS)
    txt(9.5, (T[0] + T[1]) / 2 - 5, "one node per stage\ndefault executor, DDS queues\nYOLO on a 4-hart pool")
    box(81, T[0], 19, T[1] - T[0], "#e6f3ea", ec=C_XPU, lw=1.4); txt(90.5, T[1] - 7, "XPU-RT", fz["title"], "bold", C_XPU)
    txt(90.5, (T[0] + T[1]) / 2 - 5, "one periodic spec:\nrates, windows, dependencies\nthe board's traces re-cost it")
    # where the work is placed
    box(21, T[0], 18.5, T[1] - T[0], "#fbe0dd", ec=C_ROS); txt(30.25, (T[0] + T[1]) / 2, "the OS places callbacks\nas they become runnable;\nframes queue behind\nthe busy stage")
    box(60.5, T[0], 18.5, T[1] - T[0], "#d8ecdf", ec=C_XPU); txt(69.75, (T[0] + T[1]) / 2, "a solver (greedy or CP-SAT)\nplaces every dispatch on a\nhart and a time: all 8 harts,\nframes overlap in flight")
    # the shared kernels
    box(41, T[0], 18, T[1] - T[0], "#f3f1ec"); txt(50, (T[0] + T[1]) / 2, "the same int8 kernels\nYOLOv8n · fused nav · MLP control\nthe same sensors\ncamera 45–120 Hz · ToF · IMU", weight="bold")
    # what the board measures — the numbers, large
    box(21, B[0], 18.5, B[1] - B[0], "white", ec=C_ROS, lw=1.4); txt(30.25, B[1] - 7, "measured on the K1, 45 Hz camera", fz["tiny"], col="0.35")
    txt(30.25, B[0] + 20, f"{FC.ctrl_hz_label(R_)} · {FC.lat_label(R_)}", fz["head"] * 1.15, "bold", C_ROS); txt(30.25, B[0] + 8, "control rate · camera→control", fz["tiny"], col=C_ROS)
    box(60.5, B[0], 18.5, B[1] - B[0], "white", ec=C_XPU, lw=1.4); txt(69.75, B[1] - 7, "measured on the K1, 45 Hz camera", fz["tiny"], col="0.35")
    txt(69.75, B[0] + 20, f"{FC.ctrl_hz_label(A_)} · {FC.lat_label(A_)}", fz["head"] * 1.15, "bold", C_XPU); txt(69.75, B[0] + 8, "control rate · camera→control", fz["tiny"], col=C_XPU)
    box(0, B[0], 19, B[1] - B[0], "#fdeceb", ec=C_ROS, lw=1.4); txt(9.5, (B[0] + B[1]) / 2, f"{min(goals, goals90):.0f}–{max(goals, goals90):.0f} goals/s at a 45\nand a 90 Hz camera;\nYOLO cannot overlap frames", col=C_ROS)
    box(81, B[0], 19, B[1] - B[0], "#e6f3ea", ec=C_XPU, lw=1.4); txt(90.5, (B[0] + B[1]) / 2, f"every frame on time\nup to a {on_time_to:.0f} Hz camera;\nCP-SAT meets the windows", col=C_XPU)
    box(41, B[0], 18, B[1] - B[0], "#eef2f7", ec="#2f6db0", lw=1.4); txt(50, (B[0] + B[1]) / 2, "each arm's measured cadence\nand latency is replayed into\nthe flights below — same drone,\ncontroller, gain and scene")
    ym = (T[0] + T[1]) / 2; yb = (B[0] + B[1]) / 2
    arrow(19.3, ym, 20.7, ym, col=C_ROS); arrow(80.7, ym, 79.8, ym, col=C_XPU)
    arrow(40.7, ym - 12, 39.8, ym - 12, col="0.35"); arrow(59.3, ym - 12, 60.2, ym - 12, col="0.35")
    arrow(30.25, T[0], 30.25, B[1] + 0.3); arrow(69.75, T[0], 69.75, B[1] + 0.3)
    arrow(39.8, yb, 40.7, yb, col=C_ROS, lw=1.6); arrow(60.2, yb, 59.3, yb, col=C_XPU, lw=1.6)
    arrow(20.7, yb, 19.3, yb, col=C_ROS); arrow(79.8, yb, 80.7, yb, col=C_XPU)
    ax.set_title("Same kernels, same board, two runtimes — and how their measurements reach the flights below", fontsize=fz["title"], weight="bold", loc="left")


def draw_envelope_paper(ax, cax, hx, hr, fz, base):
    """the control-rate envelope of the submitted figure (rate injected, colour = cruise), with the two measured rates marked."""
    csv_path = os.path.join(RES, "hil_ablation.csv")
    draw_envelope(ax, csv_path, compact=True, colorbar_ax=cax, highlight_hz=100.0)
    shrink(ax, base / 22.0, cax)
    rates = [20.0, 25.0, 33.33, 50.0, 100.0]
    for h, col, tag in ((hx, "#0e7c7b", "XPU-RT"), (hr, C_ROS, "ROS 2")):
        if h < rates[0] * 0.97 or h > rates[-1] * 1.03:     # a measured rate a hair past an axis end is still that rate
            continue
        h_ = min(max(h, rates[0]), rates[-1])
        i = max(k for k in range(len(rates) - 1) if rates[k] <= h_); pos = i + (math.log(h_) - math.log(rates[i])) / (math.log(rates[i + 1]) - math.log(rates[i]))
        ax.axvline(pos, color=col, lw=2.0, ls=(0, (4, 2)), alpha=0.9, zorder=2)
        ax.text(pos, 0.905, f"{tag}\n{h:.0f} Hz", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=fz["tiny"], weight="bold", color=col, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_ylim(-0.02, 1.17)
    n_fl = sum(1 for _ in open(csv_path)) - 1
    ax.set_title(f"Control-rate envelope (rate injected, {n_fl} flights): success against\ncontrol rate — ROS 2's {hr:.0f} Hz and XPU-RT's {hx:.0f} Hz marked", fontsize=fz["title"], weight="bold", loc="left")
    return {"rates": rates, "xpu_hz": hx, "ros_hz": hr, "flights": n_fl}


def telemetry_series(X, R):
    """The exact arrays panels E-H draw, as one dict.

    draw_telemetry() and verify_showdown_figure both need these. Deriving them twice would make the
    check a copy of the drawing rather than a test of it, so both call this.
    """
    xxyz, rxyz, xt, rt = X["poses"][:, :3], R["poses"][:, :3], X["t_s"], R["t_s"]
    xw = np.linalg.norm(X["imu_w"], axis=1); rw = np.linalg.norm(R["imu_w"], axis=1)
    xg = np.degrees(np.arctan2(X["goal_cmd"][:, 1], X["goal_cmd"][:, 0]))
    rg = np.degrees(np.arctan2(R["goal_cmd"][:, 1], R["goal_cmd"][:, 0]))
    xs = np.linalg.norm(np.gradient(xxyz[:, :2], xt, axis=0), axis=1)
    rs = np.linalg.norm(np.gradient(rxyz[:, :2], rt, axis=0), axis=1)
    vxy = np.gradient(xxyz[:, :2], xt, axis=0); sel = np.arange(85, len(xxyz), 12)
    return {"E_xpu": S.smooth(xw), "E_ros": S.smooth(rw),        # drawn smoothed
            "F_xpu": xg, "F_ros": rg,
            "G_xpu": S.smooth(xs, 11), "G_ros": S.smooth(rs, 11),
            "H_along": xxyz[sel, 1], "H_across": xxyz[sel, 0],
            "H_u": vxy[sel, 1], "H_v": vxy[sel, 0],
            "H_gates": np.asarray(X["gates_world"]).reshape(-1),
            "_raw": {"xw": xw, "rw": rw, "xs": xs, "rs": rs}}


def series_fingerprint(series):
    """n, range and a digest per drawn series -- enough that a curve cannot come from another flight."""
    import hashlib as _h
    out = {}
    for k, v in series.items():
        if k.startswith("_"):
            continue
        a = np.asarray(v, dtype=float).ravel()
        a = a[np.isfinite(a)]
        out[k] = {"n": int(a.size),
                  "min": round(float(a.min()), 4) if a.size else None,
                  "max": round(float(a.max()), 4) if a.size else None,
                  "mean": round(float(a.mean()), 4) if a.size else None,
                  "sha256": _h.sha256(np.round(a, 6).tobytes()).hexdigest()[:16]}
    return out


def draw_telemetry(axes, X, R, fz, outcome=None):
    """E-H. `outcome` is the baseline's own: a baseline that ran out of time outlasts the completed
    flight instead of ending in a collision, so the axis, the shading and the titles say that rather
    than pointing at a crash it never had. Left unset the panels read as they always have."""
    xxyz, rxyz, xt, rt = X["poses"][:, :3], R["poses"][:, :3], X["t_s"], R["t_s"]
    ran_out = str(outcome) == "timeout"
    axi, axg, axs, axq = axes
    _ser = telemetry_series(X, R); xw, rw = _ser["_raw"]["xw"], _ser["_raw"]["rw"]
    axi.plot(xt, S.smooth(xw), color=C_XPU, lw=2.0, label=f"XPU-RT ({V.eff_hz(X):.0f} Hz control)"); axi.plot(rt, S.smooth(rw), color=C_ROS, lw=2.0, label=f"ROS 2 ({V.eff_hz(R):.0f} Hz)")
    hr_ = V.eff_hz(R) or FC.fallback("eff_hz_ros", 39.0, "ROS 2 dump lacks eff_cmd_hz")
    axi.set_ylabel("IMU |ω| (rad/s), smoothed", fontsize=fz["lab"]); axi.set_title(f"Inside the pair — body rate:\nthe baseline thrashes at {hr_:.0f} Hz", fontsize=fz["title"], weight="bold", loc="left"); axi.legend(fontsize=fz["leg"], frameon=False, loc="upper right")
    xg = np.degrees(np.arctan2(X["goal_cmd"][:, 1], X["goal_cmd"][:, 0])); rg = np.degrees(np.arctan2(R["goal_cmd"][:, 1], R["goal_cmd"][:, 0]))
    axg.plot(xt, xg, color=C_XPU, lw=1.6); axg.plot(rt, rg, color=C_ROS, lw=1.6)
    axg.set_ylim(-80, 100); axg.set_ylabel("goal heading (°)", fontsize=fz["lab"])
    axg.set_title("Nav goal heading: XPU-RT's settles,\nthe baseline's never does" if ran_out
                  else "Nav goal heading: the two commands\ndiverge before the baseline's collision",
                  fontsize=fz["title"], weight="bold", loc="left")
    xs = np.linalg.norm(np.gradient(xxyz[:, :2], xt, axis=0), axis=1); rs = np.linalg.norm(np.gradient(rxyz[:, :2], rt, axis=0), axis=1)
    axs.plot(xt, S.smooth(xs, 11), color=C_XPU, lw=2.0); axs.plot(rt, S.smooth(rs, 11), color=C_ROS, lw=2.0)
    axs.set_ylabel("ground speed (m/s)", fontsize=fz["lab"])
    axs.set_title("Forward speed: XPU-RT holds the cruise,\nthe baseline never reaches it" if ran_out
                  else "Forward speed: the same cruise\nuntil the baseline's collision",
                  fontsize=fz["title"], weight="bold", loc="left")
    for ax in (axi, axg, axs):
        ax.set_xlabel("time (s)", fontsize=fz["lab"]); ax.grid(True, color="0.9", lw=0.5); ax.tick_params(labelsize=fz["tick"])
        if ran_out:   # the baseline is the one still flying, so the axis covers its whole run
            ax.set_xlim(0, max(xt.max(), rt[-1]))
            ax.axvspan(xt.max(), max(xt.max(), rt[-1]), color="#e4efe4", alpha=0.6, zorder=0)
            ax.axvline(xt.max(), color=C_XPU, lw=1.6, ls=(0, (4, 2)), alpha=0.85, zorder=1)
        else:
            ax.set_xlim(0, xt.max())
            ax.axvspan(rt[-1], xt.max(), color="#f6e3e3", alpha=0.5, zorder=0); ax.axvline(rt[-1], color=C_ROS, lw=1.6, ls=(0, (4, 2)), alpha=0.85, zorder=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    if ran_out:
        axi.text(xt.max() + 0.2, 0.60, "XPU-RT ✓\n4 gates", color=C_XPU, fontsize=fz["tiny"], weight="bold", va="top", ha="left", transform=axi.get_xaxis_transform())
    else:
        axi.text(rt[-1] + 0.2, 0.60, "ROS 2 ✗\ncrashes", color=C_ROS, fontsize=fz["tiny"], weight="bold", va="top", ha="left", transform=axi.get_xaxis_transform())
    vxy = np.gradient(xxyz[:, :2], xt, axis=0); sel = np.arange(85, len(xxyz), 12)
    axq.plot(xxyz[:, 1], xxyz[:, 0], color="0.8", lw=1.0, zorder=0)
    axq.quiver(xxyz[sel, 1], xxyz[sel, 0], vxy[sel, 1], vxy[sel, 0], xt[sel], cmap="viridis", angles="xy", scale_units="xy", scale=7.0, width=0.006, headwidth=4, headlength=5)
    for gi, g in enumerate(X["gates_world"]):
        axq.plot(g[1], g[0], "o", mfc="none", mec="#ffd400", mew=2.2, ms=9, zorder=2); axq.text(g[1], g[0] + 0.55, f"G{gi + 1}", ha="center", fontsize=fz["tiny"], color="#b07a00", weight="bold")
    axq.set_xlabel("along the aisle (m)", fontsize=fz["lab"]); axq.set_ylabel("across (m)", fontsize=fz["lab"]); axq.set_title("XPU-RT's velocity through the gates\n(arrow = heading · speed, colour = time)", fontsize=fz["title"], weight="bold", loc="left")
    axq.grid(True, color="0.92", lw=0.5); axq.tick_params(labelsize=fz["tick"]); axq.set_aspect("equal", adjustable="datalim")
    return {"xpu_mean_w": round(float(np.mean(xw)), 3), "ros_mean_w": round(float(np.mean(rw)), 3),
            "xpu_mean_speed": round(float(np.mean(xs)), 3), "ros_mean_speed": round(float(np.mean(rs)), 3),
            # panels E-H carried only these four means, so a whole curve could come from another
            # flight with the mean unchanged; the fingerprint is what the verifier re-derives
            "series": series_fingerprint(_ser)}


def draw_pairs2(ax, rows, fz):
    """same seed, same speed, both runtimes — who gets further (gates cleared; a completed course counts 4): the 1.7 m scene
    with the latency replayed (paired seeds, both seed sets), the 2.4 m scene's totals in the title."""
    def pairs(sub, ph):
        A = SF.cells_by(sub, "xpu_a_cpsat_hard.csv", FC.lat_ms("xpu_a_cpsat_hard.csv"), 0.0, person_h=ph); B = SF.cells_by(sub, "ros_vanilla445.csv", FC.lat_ms("ros_vanilla445.csv"), 0.0, person_h=ph)
        tot = collections.Counter(); per = {}
        for c in sorted(set(A) & set(B)):
            a = {r["seed"]: r for r in A[c]}; b = {r["seed"]: r for r in B[c]}; cnt = collections.Counter()
            for sd in set(a) & set(b):
                ga = 4 if a[sd]["outcome"] == "success" else a[sd]["gates"]; gb = 4 if b[sd]["outcome"] == "success" else b[sd]["gates"]
                cnt["xpu" if ga > gb else ("ros" if gb > ga else "tie")] += 1
            per[c] = cnt; tot.update(cnt)
        return tot, per
    t17, p17 = pairs([r for r in rows if r["camp"] == "campaign_break" and r["dens"] == 0.30], 1.7)
    t24, _ = pairs([r for r in rows if r.get("ph", 2.4) == 2.4 and r["dens"] == 0.30 and r["course"] == "a"], 2.4)
    speeds = sorted(p17); x = np.arange(len(speeds))
    for key, col, lab in (("xpu", C_XPU, "XPU-RT gets further"), ("tie", "0.8", "same gate"), ("ros", C_ROS, "ROS 2 gets further")):
        bottom = np.zeros(len(speeds))
        if key == "tie":
            bottom = np.array([p17[c]["xpu"] for c in speeds], float)
        elif key == "ros":
            bottom = np.array([p17[c]["xpu"] + p17[c]["tie"] for c in speeds], float)
        vals = np.array([p17[c][key] for c in speeds], float)
        ax.bar(x, vals, bottom=bottom, color=col, width=0.7, edgecolor="white", lw=0.6, label=lab)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 3:
                ax.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=fz["tiny"], color="white" if key != "tie" else INK, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{c:.1f}" for c in speeds], fontsize=fz["tick"]); ax.set_xlabel("cruise speed (m/s)", fontsize=fz["lab"]); ax.set_ylabel("seed pairs", fontsize=fz["lab"]); ax.tick_params(labelsize=fz["tick"])
    ax.legend(fontsize=fz["tiny"], frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0)); ax.set_ylim(0, max(sum(p17[c].values()) for c in speeds) * 1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Same seed, 1.7 m: XPU-RT further\n{t17['xpu']} vs {t17['ros']}, {t17['tie']} tied · 2.4 m: {t24['xpu']} vs {t24['ros']}", fontsize=fz["title"], weight="bold", loc="left")
    return {"scene_1.7": dict(t17), "scene_2.4": dict(t24), "per_speed_1.7": {f"c{c}": dict(v) for c, v in p17.items()}}


def draw_crash_position(ax, fz, camps=("campaign_break",), scene="1.7 m scene", other=("campaign_percep", "campaign_seeds24")):
    """where every flight ends, one flight per seed: the controlled 1.7 m scene (both seed sets); the 2.4 m scene's pooled medians in the title."""
    recs = SF.load_records(camps=camps); arms = [a for a in ("xpu_cpsat", "ros_vanilla4", "xpu_greedy") if any(k[0] == a for k in recs)]
    speeds = sorted({c for (a, c) in recs if a == "xpu_cpsat"})
    y0 = min(r["poses"][0, 1] for v in recs.values() for r in v); gw = next(iter(recs.values()))[0]["gates_world"]; gy = gw[:, 1] - y0
    short = {"xpu_cpsat": ("XPU-RT CP-SAT", C_XPU), "ros_vanilla4": ("ROS 2 vanilla", C_ROS), "xpu_greedy": ("XPU-RT greedy", C_XPU2)}
    w = 0.8 / len(arms); out = {}; rng = np.random.RandomState(0)
    for i, a in enumerate(arms):
        lab, col = short[a]
        for j, c in enumerate(speeds):
            v = recs.get((a, c), [])
            if not v:
                continue
            ends = np.array([r["poses"][-1, 1] - y0 for r in v]); succ = np.array([r["outcome"] == "success" for r in v])
            x = j + (i - (len(arms) - 1) / 2) * w + (rng.rand(len(ends)) - 0.5) * w * 0.7
            ax.scatter(x[succ], ends[succ], s=30, marker="o", c=col, edgecolors="white", lw=0.5, zorder=3); ax.scatter(x[~succ], ends[~succ], s=22, marker="x", c=col, lw=1.0, zorder=3)
            med = float(np.median(ends)); ax.plot([j + (i - (len(arms) - 1) / 2) * w - w * 0.45, j + (i - (len(arms) - 1) / 2) * w + w * 0.45], [med, med], color=col, lw=2.6, zorder=4)
            out[f"{a}_c{c}"] = {"n": len(v), "median_along_m": round(med, 2), "completed": int(succ.sum())}
    for gi, y in enumerate(gy):
        ax.axhline(y, color="#f2a900", lw=1.1, ls=(0, (4, 3)), zorder=1); ax.text(len(speeds) - 0.55, y + 0.15, f"G{gi + 1}", color="#b07a00", fontsize=fz["tiny"], ha="right", va="bottom")
    ax.set_xticks(range(len(speeds))); ax.set_xticklabels([f"{c:.1f}" for c in speeds], fontsize=fz["tick"]); ax.set_xlabel("cruise speed (m/s)", fontsize=fz["lab"]); ax.set_ylim(-1.0, 27)
    ax.set_ylabel("where the flight ends, along the aisle (m)", fontsize=fz["lab"]); ax.tick_params(labelsize=fz["tick"]); ax.grid(axis="y", ls=":", lw=0.5)
    ax.legend(handles=[Line2D([], [], color=short[a][1], lw=2.6, label=short[a][0]) for a in arms] + [Line2D([], [], marker="o", ls="", color="0.4", label="completed"), Line2D([], [], marker="x", ls="", color="0.4", label="collision")], fontsize=fz["tiny"], frameon=False, loc="upper left", ncol=2)
    def pooled_med(a):   # median end position over 1.0–1.4 m/s, one flight per seed
        v = [r["poses"][-1, 1] - y0 for c in speeds if 1.0 <= c <= 1.4 for r in recs.get((a, c), [])]
        return float(np.median(v)) if v else float("nan")
    def gate_of(m):
        k = int(np.sum(gy <= m)); return "before G1" if k == 0 else (f"past G{k}" if k < 4 else "the course")
    mx, mr = pooled_med("xpu_cpsat"), pooled_med("ros_vanilla4"); out["pooled_median_1.0-1.4_m"] = {"xpu_cpsat": round(mx, 2), "ros_vanilla4": round(mr, 2)}
    orecs = SF.load_records(camps=other); oy0 = min(r["poses"][0, 1] for v in orecs.values() for r in v)
    def omed(a):
        v = [r["poses"][-1, 1] - oy0 for (aa, c), rs in orecs.items() if aa == a and 1.0 <= c <= 1.4 for r in rs]
        return float(np.median(v)) if v else float("nan")
    ox, orr = omed("xpu_cpsat"), omed("ros_vanilla4"); out["pooled_median_1.0-1.4_m_2.4m_scene"] = {"xpu_cpsat": round(ox, 2), "ros_vanilla4": round(orr, 2)}
    ax.set_title(f"Where flights end ({scene}, one per seed):\nmedian {mx:.1f} vs {mr:.1f} m (2.4 m: {ox:.1f} vs {orr:.1f})".replace(", one per seed", ""), fontsize=fz["title"], weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return out


def censor_and_pair(cells, X_, R_):
    """the breaking-point panel's two rules, applied before the equal-replicate rule: (1) a flight still airborne at the
    simulator's horizon (18 s; the course needs ~22 s at 0.8 m/s) is neither a success nor a collision and is left out
    rather than scored as a failure; (2) at each speed the two runtimes are compared on the seeds both decided — a seed one
    arm only left airborne is dropped for both. Returns the trimmed cells and the number of airborne flights left out."""
    n_air = sum(r["outcome"] == "timeout" for v in cells.values() for r in v)
    cells = {k: [r for r in v if r["outcome"] != "timeout"] for k, v in cells.items()}
    for c in {k[1] for k in cells}:
        if (X_, c) in cells and (R_, c) in cells:
            common = {r["seed"] for r in cells[(X_, c)]} & {r["seed"] for r in cells[(R_, c)]}
            for t in (X_, R_):
                cells[(t, c)] = [r for r in cells[(t, c)] if r["seed"] in common]
    return cells, n_air


def draw_break(ax, rows, fz, show_dense=True, ms=7, lw=2.4):
    """the breaking point: the 1.7 m-people scene (crates set the difficulty), both arms replaying cadence and latency,
    success against cruise — where both complete, where the baseline breaks and XPU-RT continues, where both fail."""
    cells = V.flight_cells(rows, "a", 0.30, 0.0055, person_h=1.7, camps={"campaign_break"})
    X_, R_ = "xpu_a_cpsat_hard.csv", "ros_vanilla445.csv"
    cells, n_air = censor_and_pair(cells, X_, R_)
    arms = [t for t in ("xpu_a_cpsat_hard.csv", "ros_vanilla445.csv", "xpu_a_greedy.csv", "ros_p345.csv") if any(k[0] == t for k in cells)]   # + the hand-pinned ROS 2 (56 ms) when flown
    cells, _ = V.equalise(cells, arms) if arms else (cells, 0)
    out = {}; curves = {}
    for tr in arms:
        lab, col = V.FLIGHT_ARMS[tr]
        pts = sorted((c, sum(x["outcome"] == "success" for x in v), len(v)) for (t, c), v in cells.items() if t == tr and len(v) >= 6)
        if not pts:
            continue
        xs = [c for c, _, _ in pts]; ps = [wilson(k, n) for _, k, n in pts]
        ax.fill_between(xs, [p[1] for p in ps], [p[2] for p in ps], color=col, alpha=0.12, lw=0)
        ax.plot(xs, [p[0] for p in ps], "-o", color=col, lw=lw, ms=ms, mec="white", label=lab, zorder=4)
        if tr == "ros_vanilla445.csv":   # the baseline is named above its first two points (clear of the XPU-RT curve), XPU-RT at its last
            ax.annotate(lab.split(" (")[0], ((xs[0] + xs[min(1, len(xs) - 1)]) / 2, (ps[0][0] + ps[min(1, len(ps) - 1)][0]) / 2), textcoords="offset points", xytext=(0, 10), ha="center", va="bottom", fontsize=fz["tiny"], weight="bold", color=col, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8), zorder=7)
        elif "greedy" not in tr and "p3" not in tr:   # greedy is named in the legend
            ax.annotate(lab.split(" (")[0], (xs[-1], ps[-1][0]), textcoords="offset points", xytext=(8, 0), ha="left", va="center", fontsize=fz["tiny"], weight="bold", color=col)   # inline label, no legend to collide with the zones
        for c, k, n in ([] if ("greedy" in tr or "p3" in tr) else pts):
            ax.annotate(f"{k}/{n}", (c, k / n), textcoords="offset points", xytext=(0, 8 if tr.startswith("xpu") else -13), ha="center", fontsize=fz["tiny"] * 0.9, color=col)
        out[lab] = {f"c{c}": [k, n] for c, k, n in pts}; curves[tr] = {c: k / n for c, k, n in pts}
    dense = V.flight_cells(rows, "a", 0.40, 0.0055, person_h=1.7, camps={"campaign_break"}); dense = {k: [r for r in v if r["outcome"] != "timeout"] for k, v in dense.items()}
    if show_dense and all(any(k[0] == t for k in dense) for t in (X_, R_)):   # the denser scene, both arms: dashed, drawn behind
        dense, _ = V.equalise(dense, [X_, R_])
        for tr in (X_, R_):
            pts = sorted((c, sum(x["outcome"] == "success" for x in v), len(v)) for (t, c), v in dense.items() if t == tr and len(v) >= 6)
            if pts:
                ax.plot([c for c, _, _ in pts], [k / n for _, k, n in pts], "--", color=V.FLIGHT_ARMS[tr][1], lw=1.4, alpha=0.8, zorder=3, marker="s", ms=4)
                out[V.FLIGHT_ARMS[tr][0] + ", density 0.40"] = {f"c{c}": [k, n] for c, k, n in pts}
        ax.plot([], [], "--", color="0.4", lw=1.4, marker="s", ms=4, label="dashed: density 0.40")
    hs = [h for h in ax.get_lines() if h.get_label() in ("XPU-RT · greedy", "ROS 2 hand-tuned (pinned)", "dashed: density 0.40")]
    if hs:
        ax.legend(handles=hs, fontsize=fz["tiny"] * 0.85, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92), handlelength=2.2)
    xa, ra = curves.get("xpu_a_cpsat_hard.csv", {}), curves.get("ros_vanilla445.csv", {})
    # zones, per speed both arms flew: both complete at least a quarter; XPU-RT does and the baseline fewer than one in six; neither reaches one in six
    both = [c for c in xa if c in ra and xa[c] >= 0.25 and ra[c] >= 0.25]; brk = [c for c in xa if c in ra and xa[c] >= 0.25 and ra[c] < 1 / 6]; none = [c for c in xa if c in ra and xa[c] < 1 / 6 and ra[c] < 1 / 6]
    for zone, col, txt in ((both, "#eef4ee", "both\nfly"), (brk, "#e2f0e6", "baseline\nbreaks"), (none, "#f6e3e3", "both\nfail")):
        runs, labelled = [], False   # shade each speed; one label per contiguous run of speeds, the zone's name written once
        for c in sorted(zone):
            ax.axvspan(c - 0.1, c + 0.1, color=col, zorder=0)
            if runs and abs(c - runs[-1][-1]) < 0.25:
                runs[-1].append(c)
            else:
                runs.append([c])
        for run in runs:
            if not labelled:
                ax.text((run[0] + run[-1]) / 2, 0.985, txt, ha="center", va="top", fontsize=fz["tiny"] * 0.85, color="0.3", transform=ax.get_xaxis_transform(), bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8), zorder=6); labelled = True
    out["zones"] = {"both": both, "baseline_breaks": brk, "both_fail": none}; out["airborne_at_horizon_left_out"] = n_air
    ax.set_xlim(0.7, 2.45); ax.set_ylim(-0.04, 1.16); ax.set_ylabel("course completed (fraction)", fontsize=fz["lab"]); ax.set_xlabel("cruise speed (m/s)", fontsize=fz["lab"]); ax.tick_params(labelsize=fz["tick"]); ax.grid(ls=":", lw=0.5)
    n_fl = sum(len(v) for (t, c), v in cells.items() if t in arms)
    if not arms:
        ax.text(0.5, 0.5, "breaking-point campaign in flight", ha="center", va="center", transform=ax.transAxes, fontsize=fz["tick"], color="0.5")
    ax.set_title(f"The breaking point (1.7 m people,\ncadence + latency; {n_fl} flights)", fontsize=fz["title"], weight="bold", loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return out


def scene_panel(ax, Xs, Rs, recdir, cru, letter, fz):
    dirs = [(lab, col, os.path.join(recdir, sub)) for lab, col, sub in [("XPU-RT · CP-SAT", C_XPU, "xpu_cpsat"), ("ROS 2 vanilla", C_ROS, "ros_vanilla"), ("XPU-RT · greedy", C_XPU2, "xpu_greedy")]] if os.path.isdir(recdir) else []
    hx_, hr_ = V.eff_hz(Xs) or FC.fallback("eff_hz_xpu", 100.0, "XPU-RT dump lacks eff_cmd_hz"), V.eff_hz(Rs) or FC.fallback("eff_hz_ros", 39.0, "ROS 2 dump lacks eff_cmd_hz"); _, _, hit_ = V.moments_for(Xs, Rs, 85, hx_, hr_)
    Mk = V.draw_K(ax, Xs, Rs, dirs, fz, f"ROS 2 vanilla ({hr_:.0f} Hz)", f"XPU-RT CP-SAT ({hx_:.0f} Hz)")
    n_ = sum(v["n"] for v in Mk.values()); rg_ = int(Rs["gates_passed"])
    counts = ", ".join(f"{k.split(' · ')[-1] if 'XPU' in k else k.replace(' vanilla', '')} {v['completed']}/{v['n']}" for k, v in Mk.items())
    ax.set_title(f"{letter} ({cru:.1f} m/s), flown {n_} times: {counts}\nevery success and collision; bold = the pair (ROS 2 clears G{rg_}, hits the {hit_})" if n_ else f"{letter} ({cru:.1f} m/s): the pair over its obstacles", fontsize=fz["title"], weight="bold", loc="left")
    return Mk


def strips(fig, gs, moments, X, R, xpu_dir, ros_dir, fz):
    xt, rt = X["t_s"], R["t_s"]
    for c, (src, step, lab) in enumerate(moments):
        dd = ros_dir if src == "ROS" else xpu_dir; Z = R if src == "ROS" else X; tt = (rt if src == "ROS" else xt)[min(step, len(Z["t_s"]) - 1)]
        f = S.frame_at(dd, Z["frame_steps"], step)
        col = gs[c].subgridspec(2, 3, height_ratios=[2.4, 14], width_ratios=[1.15, 1.55, 1.15], hspace=0.02, wspace=0.05); tc = C_ROS if src == "ROS" else C_XPU
        axh = fig.add_subplot(col[0, :]); axh.axis("off")
        axh.scatter([0.02], [0.72], s=fz["badge"] * 14, marker="o", facecolors="black", edgecolors=tc, linewidths=2.2, transform=axh.transAxes, clip_on=False, zorder=5)
        axh.text(0.02, 0.72, chr(ord("a") + c), color="white", fontsize=fz["tiny"], weight="bold", ha="center", va="center", transform=axh.transAxes, zorder=6)
        axh.text(0.065, 0.72, f"{lab} · t={tt:.1f}s", fontsize=fz["tick"], weight="bold", color=tc, va="center", transform=axh.transAxes)
        ac = fig.add_subplot(col[1, 0]); ac.imshow(f["chase"][225:465, 415:655]); ac.axis("off"); ac.set_title("chase", fontsize=fz["tick"])
        af = fig.add_subplot(col[1, 1]); af.imshow(f["fpv"], cmap="gray", vmin=0, vmax=1, aspect="equal")
        for x in [d for d in f["det"] if d[5] >= 0.4]:
            cls, x0, y0, x1, y1, cf = x; _, cc = S.YOLO.get(int(cls), ("obj", "#39f")); af.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=cc, lw=2.2))
        af.set_xticks([]); af.set_yticks([]); af.set_title("FPV + YOLO", fontsize=fz["tick"])
        at = fig.add_subplot(col[1, 2]); S.cross_tof(at, f["tof"]); at.set_title("cross-ToF", fontsize=fz["tick"])


def gantt(ax, fz, base):
    cols = {"xpu": C_XPU, "xpu2": C_XPU2, "ros": C_ROS, "ros8": C_ROS8}; grows, gpaths, gside = [], [], []
    for r in ["xpu:CP-SAT:xpu", "xpu2:greedy:xpu2", "ros8:ROS2 8-core:ros8", "ros:ROS2 vanilla:ros"]:
        name, label, ck = r.split(":"); p = os.path.join(REPO, "schedules", f"measured_gantt_v3_{name}.json")
        if os.path.exists(p):
            grows.append((json.load(open(p)), label, cols[ck], "ros" if ck.startswith("ros") else "xpu")); gpaths.append(p); gside.append(p.replace(".json", "_metrics.json"))
    S.draw_combined_gantt(ax, grows, gpaths); gtitle = ax.get_title(loc="left")
    V.rescale_fonts(ax, base / 16.0, row_label_pt=fz["tick"] * 0.64, tick_pt=fz["tick"])
    for t in ax.texts:
        if t.get_rotation() == 90 and t.get_position()[0] < -5:
            t.set_visible(False)
    ax.get_legend().set_bbox_to_anchor((0.5, -0.14)); ax.get_legend().set_loc("upper center")
    parts = [p.strip() for p in re.split(r"  ·  |\s—\s", gtitle)]
    parts = [re.sub(r"(\d+) of (\d+) frames late", r"\1/\2 late", p).replace("camera→control ", "").replace("control every ", "control ") for p in parts]
    ax.set_title("The schedules themselves — " + "\n".join([parts[0] + " — " + "  ·  ".join(parts[1:3]), "  ·  ".join(parts[3:])]) if len(parts) > 3 else gtitle, fontsize=fz["title"], weight="bold", loc="left", pad=fz["title"] * 0.9)
    return gside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="tall1005"); ap.add_argument("--xpu-dir"); ap.add_argument("--ros-dir"); ap.add_argument("--scene-records", default=os.path.join(RES, "campaign_scene", "tall1005s"))
    ap.add_argument("--scene2-records", default=os.path.join(RES, "campaign_scene", "tall1000s")); ap.add_argument("--scene2-xpu-dir", default=os.path.join(RES, "campaign_v2/display_v3s_c1.4/xpu_s1000_figdata"))
    ap.add_argument("--scene2-ros-dir", default=os.path.join(RES, "campaign_v2/display_v3s_c1.4/ros_s1000_figdata")); ap.add_argument("--scene2-cruise", type=float, default=1.4)
    ap.add_argument("--display-cruise", type=float, default=1.0); ap.add_argument("--width-in", type=float, default=26.0); ap.add_argument("--min-print-pt", type=float, default=3.6)
    ap.add_argument("--dpi", type=int, default=300); ap.add_argument("--out", default=os.path.join(RES, "refined", "warehouse_showdown_final"))
    a = ap.parse_args()
    cells_dir = {"tall1005": ("campaign_v2/display_same/xpu_s1005_figdata", "campaign_v2/display_same/ros_s1005_figdata"), "tall1000": ("campaign_v2/display_v3s_c1.4/xpu_s1000_figdata", "campaign_v2/display_v3s_c1.4/ros_s1000_figdata")}
    xd, rd = cells_dir[a.cell]; xpu_dir = a.xpu_dir or os.path.join(RES, xd); ros_dir = a.ros_dir or os.path.join(RES, rd)
    base = a.min_print_pt * a.width_in / 7.1
    fz = dict(tiny=base * 0.9, tick=base, lab=base * 1.05, leg=base * 0.95, title=base * 1.2, badge=base * 1.1, head=base * 1.35)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": base, "pdf.fonttype": 42})
    X, R = S.load(xpu_dir), S.load(ros_dir)
    xxyz, rxyz, xt = X["poses"][:, :3], R["poses"][:, :3], X["t_s"]; tnorm = (xt - xt.min()) / max(1e-6, xt.max() - xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    hx, hr = V.eff_hz(X) or FC.fallback("eff_hz_xpu", 100.0, "XPU-RT dump lacks eff_cmd_hz"), V.eff_hz(R) or FC.fallback("eff_hz_ros", 39.0, "ROS 2 dump lacks eff_cmd_hz")
    moments, near_miss, hit = V.moments_for(X, R, 85, hx, hr); bg, bg_note, bg_std = V.backdrop(X)
    rows_all = AT.all_flights(); rows = [r for r in rows_all if r["camp"].startswith("campaign_")]
    cells = V.flight_cells(rows, "a", 0.30, 0.0055); b_arms = [t for t in ("xpu_a_cpsat_hard.csv", "xpu_a_greedy.csv", "ros_vanilla445.csv", "ros_rvanilla445.csv", "ros_vanilla4x245.csv") if any(k[0] == t for k in cells)]
    cells, n_aside = V.equalise(cells, b_arms)
    cells_rich = {k: v for k, v in V.flight_cells(rows, "a", 0.30, 0.0055).items() if k[0].startswith(("xpu_b5", "ros_rvanilla"))}
    cells_rich, _ = V.equalise(cells_rich, [t for t in ("xpu_b5_cpsat.csv", "xpu_b5_greedy.csv", "ros_rvanilla445.csv", "ros_rvanilla490.csv") if any(k[0] == t for k in cells_rich)])

    H_in = a.width_in * 1.66
    fig = plt.figure(figsize=(a.width_in, H_in)); ph = float(X["person_h"]) if "person_h" in X.files else None
    fig.text(0.04, 0.995, "From one flight to the whole programme — the same kernels on one K1, scheduled by XPU-RT or run as a ROS 2 graph; drone, controller, gain and scene are shared, only what the board delivers differs",
             fontsize=fz["head"], weight="bold", va="top", ha="left")
    # rows: S | A B C | a–d | D E F G | H H′ | I J K L | M N O ; M P Q | R   (odd rows are spacers)
    hr_ = [3.3, 0.8, 5.2, 0.9, 2.3, 1.1, 2.5, 1.2, 3.4, 1.9, 3.4, 1.3, 3.4, 2.2, 3.4, 2.0, 4.6]
    outer = fig.add_gridspec(len(hr_), 1, height_ratios=hr_, hspace=0.0, left=0.045, right=0.99, top=0.978, bottom=0.03)
    axS = fig.add_subplot(outer[0]); draw_system(axS, fz); sec(axS, "S", fz)
    r2 = outer[2].subgridspec(1, 3, width_ratios=[2.9, 1.6, 1.1], wspace=0.24); axA = fig.add_subplot(r2[0])
    eg = r2[1].subgridspec(1, 2, width_ratios=[30, 1], wspace=0.03); axB = fig.add_subplot(eg[0]); cax = fig.add_subplot(eg[1]); axC = fig.add_subplot(r2[2])
    S.draw_topdown(axA, bg, X["ovK"], X["ovpos"], X["ovquat"], xxyz, rxyz, gates, people, tnorm, 0, False, 85, rxyz[-1], moments, ov_obj=None, near_miss=near_miss)
    xg, rg = int(X["gates_passed"]), int(R["gates_passed"])
    axA.legend(handles=[Line2D([0], [0], color=S.CMAP(0.6), lw=5, label=f"XPU-RT · CP-SAT ({hx:.1f} Hz) · {xg} of 4 gates (colour = time)"), Line2D([0], [0], color=C_ROS, lw=5, label=f"ROS 2 vanilla ({hr:.1f} Hz) · {rg} of 4, then hits {hit}"),
                        Line2D([0], [0], marker="o", color=V.C_MOVER, mec="white", ls="none", ms=9, alpha=0.75, label="patrolling people"), Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=3, ls="none", label="gate")],
               loc="upper left", fontsize=fz["leg"], framealpha=0.93, ncol=2, handlelength=1.9)
    axA.set_title(f"The specific case — one scene, {a.display_cruise:.1f} m/s" + (f", people {ph:.1f} m" if ph else "") + f": XPU-RT clears the course; ROS 2 clears G{rg} and hits a {hit}", fontsize=fz["title"], weight="bold", loc="left")
    MB = draw_envelope_paper(axB, cax, hx, hr, fz, base)
    draw_generalization(axC, fs=1.0, compact=True); shrink(axC, base / 12.0); axC.set_title("Generalisation: the same floor\non an unseen gate course", fontsize=fz["title"], weight="bold", loc="left")
    sec(axA, "A", fz, dx=-30, dy=-24); sec(axB, "B", fz); sec(axC, "C", fz)
    strips(fig, outer[4].subgridspec(1, 4, wspace=0.09), moments, X, R, xpu_dir, ros_dir, fz)
    tg = outer[6].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.30); axT = [fig.add_subplot(tg[i]) for i in range(4)]; MT = draw_telemetry(axT, X, R, fz)
    for ax_, s_ in zip(axT, "DEFG"):
        sec(ax_, s_, fz)
    r8 = outer[8].subgridspec(1, 2, wspace=0.14); axH = fig.add_subplot(r8[0]); axH2 = fig.add_subplot(r8[1])
    MH = scene_panel(axH, X, R, a.scene_records, a.display_cruise, "The display scene", fz)
    X2, R2 = S.load(a.scene2_xpu_dir), S.load(a.scene2_ros_dir); MH2 = scene_panel(axH2, X2, R2, a.scene2_records, a.scene2_cruise, "A second scene", fz)
    sec(axH, "H", fz); sec(axH2, "H′", fz)
    r10 = outer[10].subgridspec(1, 5, width_ratios=[1.2, 1.05, 1.0, 1.05, 0.7], wspace=0.38); axI2 = fig.add_subplot(r10[0]); axI = fig.add_subplot(r10[1]); axJ = fig.add_subplot(r10[2]); axK = fig.add_subplot(r10[3]); axL = fig.add_subplot(r10[4])
    MI2 = draw_break(axI2, rows, fz)
    V.draw_B(axI, cells, b_arms, fz, a.display_cruise); axI.get_legend().set_visible(True); [t.set_fontsize(fz["tiny"] * 0.85) for t in axI.get_legend().get_texts()]
    frac = lambda t: {c: sum(x["outcome"] == "success" for x in v) / len(v) for (tt, c), v in cells.items() if tt == t and v}
    fx, fr = frac("xpu_a_cpsat_hard.csv"), frac("ros_vanilla445.csv")
    ahead = [c for c in sorted(fx) if c in fr and fx[c] >= 1 / 6 and fr[c] < 1 / 6]; block = [c for c in sorted(fx) if c in fr and fx[c] < 1 / 6 and fr[c] < 1 / 6]
    axI.set_title((f"The 2.4 m scene of A: ahead to {max(ahead):.1f} m/s," if ahead else "The 2.4 m scene of A:") + (f"\nfrom {min(block):.1f} the people block every arm" if block else "\nsuccess against cruise speed"), fontsize=fz["title"], weight="bold", loc="left")
    MJ = draw_pairs2(axJ, rows, fz); MK = draw_crash_position(axK, fz)
    draw_mechanism(axL, fs=base / 10.5, compact=True); axL.set_title("Mechanism: moment\n(measured), power (model)", fontsize=fz["title"], weight="bold", loc="left")
    for ax_, s_ in ((axI2, "I"), (axI, "I′"), (axJ, "J"), (axK, "K"), (axL, "L")):
        sec(ax_, s_, fz)
    wr = [1.3, 1.15, 1.15]; r12 = outer[12].subgridspec(1, 3, width_ratios=wr, wspace=0.40); r14 = outer[14].subgridspec(1, 3, width_ratios=wr, wspace=0.40)
    gsM = fig.add_gridspec(1, 3, width_ratios=wr, wspace=0.40, left=0.045, right=0.99, top=outer[12].get_position(fig).y1, bottom=outer[14].get_position(fig).y0); axM = fig.add_subplot(gsM[0])
    ng = r12[1].subgridspec(2, 1, height_ratios=[1.1, 1], hspace=0.16); axN1 = fig.add_subplot(ng[0]); axN2 = fig.add_subplot(ng[1], sharex=axN1); axO = fig.add_subplot(r12[2])
    pg = r14[1].subgridspec(4, 1, hspace=0.32); axP = [fig.add_subplot(pg[i]) for i in range(4)]; axQ = fig.add_subplot(r14[2])
    MM = AT.draw_forest(axM, rows_all, fz, compact=True)
    MN = V.draw_F(axN1, axN2, V.BOARD_ARMS, fz); _lg = axN2.get_legend(); axN2.legend(handles=_lg.legend_handles, labels=[t.get_text() for t in _lg.get_texts()], fontsize=fz["tiny"] * 0.92, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, handlelength=1.2, columnspacing=1.0)
    on_time_to = max((h for h, v in MN.get("XPU-RT · CP-SAT", {}).items() if v.get("late") == 0), default=None)
    axN1.set_title(f"Why, on the board — camera rate: every frame\non time to {on_time_to:.0f} Hz; every ROS 2 layout plateaus" if on_time_to else "Why, on the board — camera rate:\nlatency and frames delivered", fontsize=fz["title"], weight="bold", loc="left")
    MO = AT.draw_waterfall(axO, fz); axO.set_title("Where the latency goes: compute for the schedule,\nqueueing for the callback graph (45 Hz, medians)", fontsize=fz["title"], weight="bold", loc="left")
    MP = V.draw_H(axP, cells_rich, fz); axP[0].set_title("Added load: two more networks on the same harts —\nCP-SAT > greedy > ROS 2, on the board and in flight", fontsize=fz["title"], weight="bold", loc="left")
    MQ = V.draw_G(axQ, [("XPU-RT · CP-SAT", C_XPU, "xpu", ["acpsat_hardr1", "acpsat_hardr2", "acpsat_hardr3"]), ("XPU-RT · greedy", C_XPU2, "xpu", ["agreedyr1", "agreedyr2", "agreedyr3"]),
                        ("ROS 2 on all 8 cores\n(two YOLO nodes)", C_ROS8, "ros", ["45_vanilla4x2_r1", "45_vanilla4x2_r2", "45_vanilla4x2_r3"]), ("ROS 2 vanilla\n(4-hart YOLO pool)", C_ROS, "ros", ["45_vanilla4_r1", "45_vanilla4_r2", "45_vanilla4_r3"])], fz)
    axQ.set_title("Where the work lands: the schedule spreads the kernels over\nall eight harts; ROS 2 can occupy them too, and still queues", fontsize=fz["title"], weight="bold", loc="left")
    for ax_, s_ in ((axM, "M"), (axN1, "N"), (axO, "O"), (axP[0], "P"), (axQ, "Q")):
        sec(ax_, s_, fz)
    axR = fig.add_subplot(outer[16]); gside = gantt(axR, fz, base); sec(axR, "R", fz, dx=-22, dy=30)
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight"); print("wrote", a.out + ".png")
    side = {"figure": a.out + ".png", "written": datetime.datetime.now().isoformat(timespec="seconds"), "cell": a.cell, "sources": {"xpu_dir": xpu_dir, "ros_dir": ros_dir, "scene_records": a.scene_records, "scene2_records": a.scene2_records, "gantt_sidecars": gside},
            "A": {"xpu_gates": xg, "ros_gates": rg, "hit": hit, "backdrop": bg_note, "plate_std": round(bg_std, 1), "xpu_eff_hz": hx, "ros_eff_hz": hr}, "B": MB, "telemetry": MT, "H": MH, "H2": MH2, "I_break": MI2, "J": MJ, "K": MK, "M": MM,
            "N": {re.sub(r"[^A-Za-z]", "", lab): {str(h): {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for h, p in pts.items()} for lab, pts in MN.items()}, "O": MO, "P": MP, "Q": {re.sub(r"[^A-Za-z]", "", k.replace("\n", " ")): v for k, v in MQ.items()},
            "flights_set_aside": n_aside, **FC.sidecar_common("showdown_final_figure", V.LOADED_CSVS + AT.LOADED_EXTRA + AT.WATERFALL_INPUTS + [os.path.join(RES, "ros_traced", "summary.csv")] + list(gside))}
    json.dump(side, open(a.out + "_metrics.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

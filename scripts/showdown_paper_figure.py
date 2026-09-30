#!/usr/bin/env python3
"""The paper's showdown figure in its paper layout, every number measured.

Same skeleton as the paper figure (A top-down pair · B control-rate envelope · C generalisation ·
D mechanism · a–d moments with chase, FPV + YOLO and cross-ToF · E–H telemetry of the pair · I the schedules), with
the two arms measured on the K1 and replayed into the flights:

  A  the display pair: XPU-RT · CP-SAT at its measured control rate against ROS 2 vanilla at its measured control
     rate, same scene, same drone, same gain; the legend counts the same scene's recorded runs per arm
  B  the rate-injected envelope (hil_ablation.csv): success against control rate, colour = cruise speed, the two
     arms' measured control rates marked
  C  the same envelope on an unseen gate course (hil_ablation_courseB.csv)
  D  mechanism: commanded moment (measured) and modelled power per arm, relative to XPU-RT (flight_energy_v2.csv)
  E–H body rate, nav goal heading, forward speed, XPU-RT velocity along the aisle — the pair's own telemetry
  I  the schedules themselves, measured on the K1 at the 45 Hz camera: XPU-RT · CP-SAT and ROS 2 vanilla, one row each
Every number is read at render time; the sidecar <out>_metrics.json lists them and the verifier re-derives them.
    scripts/showdown_paper_figure.py --out results/codesign_feedback/refined/<stem> [--width-in 20 --dpi 300]
"""
from __future__ import annotations
import argparse, collections, csv, glob, json, os, re, sys, textwrap, datetime
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results", "codesign_feedback")
os.environ.setdefault("ENERGY_CSV", os.path.join(RES, "flight_energy_v2.csv"))
sys.path.insert(0, os.path.join(REPO, "scripts")); sys.path.insert(0, os.path.join(REPO, "sims", "scripts"))
import matplotlib; matplotlib.use("Agg")   # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
import showdown_v3_figure as V   # noqa: E402
import showdown_final_figure as F   # noqa: E402
import showdown_gatecourse as S   # noqa: E402
import figure_constants as FC   # noqa: E402
import hil_story_figure as H   # noqa: E402  (ARM_NAMES also fixes panel D's 1x reference)
from hil_story_figure import draw_mechanism, draw_generalization   # noqa: E402
from flight_quarantine import flight_rows  # drops simulator-fault batches (results/codesign_feedback/flight_quarantine.csv)

C_XPU, C_ROS = V.C_XPU, V.C_ROS


def badge(ax, s, fz, dx=-8, dy=26, fc="#2f6db0"):
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points", fontsize=fz["badge"], weight="bold", color="white",
                ha="center", va="center", zorder=40, annotation_clip=False, bbox=dict(boxstyle="circle,pad=0.32", fc=fc, ec="white", lw=1.8))


def scene_counts(recdir):
    """completed / flown per arm in one recorded scene, from the scene's own campaign table."""
    p = os.path.join(recdir, "campaign.csv"); out = {}
    if not os.path.exists(p):
        return out
    for r in flight_rows(p):
        arm = os.path.basename(r.get("ctrl_trace", "") or "")
        k, n = out.get(arm, (0, 0)); out[arm] = (k + (r.get("outcome") == "success"), n + 1)
    return out


def scene_gates(recdir):
    """mean gates reached per arm in one recorded scene. A scene where neither arm completes still
    separates them by how far each gets, so the legend states that rather than two zeroes."""
    p = os.path.join(recdir, "campaign.csv"); tot = {}
    if not os.path.exists(p):
        return {}
    for r in flight_rows(p):
        arm = os.path.basename(r.get("ctrl_trace", "") or "")
        g, n = tot.get(arm, (0.0, 0)); tot[arm] = (g + float(r.get("gates_passed", 0) or 0), n + 1)
    return {k: round(g / n, 2) for k, (g, n) in tot.items() if n}


def envelope_counts(csv_path):
    """k/n per control rate over the rate-injected grid (all cruise speeds pooled)."""
    cell = {}
    for r in flight_rows(csv_path):
        hz = round(float(r["eff_cmd_hz"])); k, n = cell.get(hz, (0, 0)); cell[hz] = (k + (r["outcome"] == "success"), n + 1)
    return {str(h): list(cell[h]) for h in sorted(cell)}


def energy_ratios(csv_path):
    """panel D's bars as numbers: per condition the flights, and mean |moment| and duration-fair power
    relative to the XPU-RT condition (hil_story_figure.draw_mechanism draws exactly these)."""
    rows = list(csv.DictReader(open(csv_path))); by = collections.defaultdict(list)
    for r in rows:
        by[r["flight"].rsplit("_s", 1)[0]].append(r)
    mom = {c: float(np.mean([float(r["mean_absM"]) for r in v])) for c, v in by.items()}
    pw = {c: float(np.mean([float(r["energy"]) / float(r["dur_s"]) for r in v])) for c, v in by.items()}
    # the 1x reference must be the arm the panel actually normalises to. draw_mechanism orders the
    # conditions by ARM_NAMES and takes the first xpu one, so alphabetical order is not the same
    # question: a CSV holding both xpu_p45free and xpu_greedy sorts to greedy and would record ratios
    # against an arm the figure does not draw as 1x.
    _order = list(H.ARM_NAMES)
    xk = min((c for c in by if c.startswith("xpu")),
             key=lambda c: (_order.index(H.canon_arm(c)) if H.canon_arm(c) in _order else len(_order), c))
    return {"energy_csv": csv_path, "xpu_condition": xk,
            "conditions": {c: {"n": len(by[c]), "moment_x": round(mom[c] / mom[xk], 2), "power_x": round(pw[c] / pw[xk], 2)} for c in sorted(by)}}


def merge_dispatches(doc, gap_ms=0.1):
    """join a hart's CONSECUTIVE dispatches of the same network instance into one bar, leaving every idle gap in place.

    A ROS 2 callback occupies its hart continuously for a whole network, so its row is already one bar per frame per
    hart; a solved row is many short dispatches. Joining only back-to-back dispatches puts both rows in the same unit
    of work without claiming a hart was busy while it was idle, so the bars still agree with the measured busy %."""
    runs = {}
    for v in doc["dispatches"].values():
        k = (v["hardware_target"], v["job_name"])
        runs.setdefault(k, []).append((float(v["start_time"]), float(v["start_time"]) + float(v["duration"]), v))
    out = {}
    for k, iv in runs.items():
        iv.sort()
        a, b, proto = iv[0]
        for s_, e_, v in iv[1:]:
            if s_ - b <= gap_ms:
                b = max(b, e_); continue
            w = dict(proto); w["start_time"] = round(a, 6); w["duration"] = round(b - a, 6); out[str(len(out))] = w
            a, b, proto = s_, e_, v
        w = dict(proto); w["start_time"] = round(a, 6); w["duration"] = round(b - a, 6); out[str(len(out))] = w
    return {"metadata": doc["metadata"], "dispatches": out}


GANTT_ROWS = {"xpu": ("XPU-RT", C_XPU, "xpu"), "xpu2": ("XPU-RT · greedy", C_XPU, "xpu"),
              "ros": ("ROS 2", C_ROS, "ros"), "ros8": ("ROS 2 · all 8 cores", C_ROS, "ros"),
              "p3": ("ROS 2 · pinned", C_ROS, "ros"),
              # the two pinned arms told apart: same per-network core sets, differing only in
              # whether control runs on its own timer or in the perception callback
              "p3t": ("ROS 2 timer", C_ROS, "ros"),
              "p3c": ("ROS 2 chained", C_ROS, "ros")}


def _spread_note(sp):
    """Each network's own operators and the lanes they occupy.

    A dispatch is traced on the one hart whose thread recorded it, but a sharded dispatch runs on
    every lane of its shard; the bars draw the lanes, so the note states them and says how many of
    them the trace itself named, rather than printing a lane count the drawing contradicts.
    """
    parts = []
    for k, (tv, dv) in sorted(sp.items(), key=lambda kv: -len(kv[1][1])):
        if not dv:
            continue
        parts.append(f"{k} on {len(dv)} hart" + ("" if len(dv) == 1 else "s")
                     + (f", {len(tv)} traced" if len(dv) > len(tv) else ""))
    return ("  ·  " + ", ".join(parts)) if parts else ""


def gantt_two_rows(ax, fz, base, merge=False, names=("xpu", "ros"), prefix=None, camera_hz=45, xpu_label="XPU-RT · CP-SAT"):
    """One measured row per named arm, from the v3 Gantt sidecars; the solved arm is drawn first."""
    rows, paths, sides, nbars = [], [], [], []
    for name in names:
        label, col, kind = GANTT_ROWS[name]
        p = f"{prefix or os.path.join(REPO, 'schedules', 'measured_gantt_v3')}_{name}.json"
        doc = json.load(open(p)); n_raw = len(doc["dispatches"])
        if merge:
            doc = merge_dispatches(doc)
        rows.append((doc, label, col, kind)); paths.append(p); sides.append(p.replace(".json", "_metrics.json")); nbars.append((label, n_raw, len(doc["dispatches"])))
    S.draw_combined_gantt(ax, rows, paths)
    V.rescale_fonts(ax, base / 16.0, row_label_pt=fz["tick"] * 1.1, tick_pt=fz["tick"])
    ax.get_legend().set_bbox_to_anchor((0.0, 1.0)); ax.get_legend().set_loc("upper left")
    mets = [json.load(open(s)) for s in sides]; mx, mr = mets[0], mets[1]
    # how many distinct harts each network's own operators occupy: the placement the runtime chose,
    # counted from the drawn dispatches rather than from what a manifest declares
    def spread(doc):
        per = {}
        for v in doc["dispatches"].values():
            j = v["job_name"]
            k = "control" if j.startswith("mlp") else ("nav" if j.startswith("fused") else "YOLO")
            t, d = per.setdefault(k, (set(), set()))
            for h in str(v.get("traced_target") or v["hardware_target"]).split("+"):
                t.add(h)
            for h in str(v["hardware_target"]).split("+"):
                d.add(h)
        return per
    spreads = [spread(d) for d, _, _, _ in rows]
    late = f"{mx['frames_late']}/{mx['frames_checked']} frames late" if mx.get("frames_late") else "every frame on time"
    def note(mm):                              # the row name already says which arm this is
        t = mm.get("arm_label") or "OS-scheduled"
        for pre in ("ROS 2 · ", "XPU-RT · "):
            t = t[len(pre):] if t.startswith(pre) else t
        return t
    for t in ax.texts:   # the rotated placement notes beside the lanes repeat what the title says
        if t.get_rotation() == 90 and t.get_position()[0] < -5:
            t.set_visible(False)
    ax.set_title(f"Onboard K1 schedule, measured at the {camera_hz:g} Hz camera — {xpu_label} places the kernels over all 8 harts: camera→control {mx['chain_ms_median']:.0f} ms, "
                 f"control every {mx['ctrl_gap_mean_ms']:.1f} ms, {late}" + _spread_note(spreads[0])
                 + "".join(f"\n{GANTT_ROWS[n][0]} ({note(mm)}): camera→control {mm['chain_ms_median']:.0f} ms, "
                           f"control every {mm['ctrl_gap_mean_ms']:.1f} ms, {mm['frames_late']}/{mm['frames_checked']} frames late"
                           + _spread_note(sp)
                           for n, mm, sp in zip(names[1:], mets[1:], spreads[1:]))
                 + ("\nbars: a hart's consecutive dispatches of one frame joined, idle gaps kept — the same unit of work "
                    f"as a ROS 2 callback ({nbars[0][1]} XPU-RT dispatches in this window, {nbars[0][2]} bars)" if merge else ""),
                 fontsize=fz["title"], weight="bold", loc="left", pad=fz["title"] * 0.8)
    keep = ("chain_ms_median", "ctrl_gap_mean_ms", "frames_late", "frames_checked", "placement_note", "busy_source")
    return sides, {**{n: {**{k: mm.get(k) for k in keep},
                          "harts_per_net": {kk: sorted(vv[0]) for kk, vv in sp.items()},
                          # the lanes the bars occupy, which for a sharded dispatch exceed the traced hart
                          "drawn_harts_per_net": {kk: sorted(vv[1]) for kk, vv in sp.items()}}
                      for n, mm, sp in zip(names, mets, spreads)},
                   "bars": {lab: {"dispatches": n0, "drawn": n1} for lab, n0, n1 in nbars}, "merged_per_hart_and_frame": bool(merge)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xpu-dir", default=os.path.join(RES, "campaign_v2/display_same/xpu_s1005_figdata")); ap.add_argument("--ros-dir", default=os.path.join(RES, "campaign_v2/display_same/ros_s1005_figdata"))
    ap.add_argument("--scene-records", default=os.path.join(RES, "campaign_scene", "tall1005s")); ap.add_argument("--display-cruise", type=float, default=1.0)
    ap.add_argument("--width-in", type=float, default=20.0); ap.add_argument("--min-print-pt", type=float, default=3.6); ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--gantt-rows", default="xpu,ros", help="named Gantt rows, the solved arm first (e.g. xpu,ros,ros8)")
    ap.add_argument("--gantt-merge", action="store_true", help="draw one bar per hart per network instance, the unit a ROS callback already occupies")
    # required, not defaulted: a default stem would name a committed figure, so a bare run would
    # overwrite it from whatever the other defaults point at -- a different display pair and scene
    # from the one committed under that name.
    ap.add_argument("--out", required=True,
                    help="output stem, e.g. results/codesign_feedback/refined/"
                         "showdown_45hz_solver_vs_rospinned_s1007 (no extension)")
    # which pair of deployments the figure is about; the defaults are the 45 Hz pair
    ap.add_argument("--xpu-trace", default="xpu_a_cpsat_hard.csv", help="ctrl trace naming the XPU-RT arm in the scene's campaign table")
    ap.add_argument("--ros-trace", default="ros_vanilla445.csv", help="ctrl trace naming the ROS 2 arm in the scene's campaign table")
    ap.add_argument("--ros-label", default="ROS 2 vanilla", help="how panel A names the baseline")
    ap.add_argument("--xpu-label", default="XPU-RT · CP-SAT", help="how panels A and I name the XPU-RT arm: the solver that produced the schedule it flew")
    ap.add_argument("--gantt-prefix", default=None, help="Gantt sidecars <prefix>_<row>.json (default schedules/measured_gantt_v3)")
    ap.add_argument("--camera-hz", type=float, default=45, help="the camera rate of the board runs panel I draws")
    ap.add_argument("--baseline-fails-before-first-gate", action="store_true",
                    help="declare that this form's baseline is drawn never reaching the first gate, rather than "
                         "crashing inside the course; the verifier then requires the drawn flight to be that, and "
                         "rejects a crash after a gate. Set it for the form that flies the gain each arm's own "
                         "command rate calls for, where the baseline does not enter the course at all.")
    a = ap.parse_args()
    base = a.min_print_pt * a.width_in / 7.1
    fz = dict(tiny=base * 0.9, tick=base, lab=base * 1.05, leg=base * 0.95, title=base * 1.2, badge=base * 1.1, head=base * 1.35)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": base, "pdf.fonttype": 42})
    X, R = S.load(a.xpu_dir), S.load(a.ros_dir)
    xxyz, rxyz, xt = X["poses"][:, :3], R["poses"][:, :3], X["t_s"]; tnorm = (xt - xt.min()) / max(1e-6, xt.max() - xt.min())
    pm = X["person_mask"].astype(bool); people = X["obst_pos"][:, pm, :]; gates = X["gates_world"]
    hx = V.eff_hz(X) or FC.fallback("eff_hz_xpu", 100.0, "XPU-RT dump lacks eff_cmd_hz"); hr = V.eff_hz(R) or FC.fallback("eff_hz_ros", 39.0, "ROS 2 dump lacks eff_cmd_hz")
    r_out = str(R["outcome"]) if "outcome" in R.files else ""
    moments, near_miss, hit = V.moments_for(X, R, 85, hx, hr, outcome=r_out); bg, bg_note, bg_std = V.backdrop(X)
    xg, rg = int(X["gates_passed"]), int(R["gates_passed"]); sc = scene_counts(a.scene_records)
    kx, nx = sc.get(a.xpu_trace, (None, 0)); kr, nr = sc.get(a.ros_trace, (None, 0))
    sg = scene_gates(a.scene_records); gx_sc, gr_sc = sg.get(a.xpu_trace), sg.get(a.ros_trace)

    n_gantt = len(a.gantt_rows.split(","))
    grow = 1.9 * max(0, n_gantt - 2)          # the other six slots sum to 13.2 and the Gantt slot is 3.9
    head = 0.5 * max(0, n_gantt - 2)          # each measured row past the second adds a line to the title
    fig = plt.figure(figsize=(a.width_in, a.width_in * 0.70 * (17.1 + grow + head) / 17.1))
    outer = fig.add_gridspec(7, 1, height_ratios=[5.3, 0.95, 2.3, 0.7, 2.6, 1.35 + head, 3.9 + grow], hspace=0.0, left=0.035, right=0.99, top=0.965, bottom=0.045)
    # A | [B ; C | D]
    r0 = outer[0].subgridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.09); axA = fig.add_subplot(r0[0])
    rc = r0[1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.55); bgrid = rc[0].subgridspec(1, 2, width_ratios=[40, 1], wspace=0.04); axB = fig.add_subplot(bgrid[0]); cax = fig.add_subplot(bgrid[1])
    cd = rc[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.45); axC = fig.add_subplot(cd[0]); axD = fig.add_subplot(cd[1])
    # both paths strobed on ONE clock, so the gap between markers reads as ground covered per second
    STROBE_S = 1.0
    S.draw_topdown(axA, bg, X["ovK"], X["ovpos"], X["ovquat"], xxyz, rxyz, gates, people, tnorm, 0, False, 85, rxyz[-1], moments, ov_obj=None, near_miss=near_miss,
                   strobe_s=STROBE_S, xt=X["t_s"], rt=R["t_s"])
    # the panel draws one flight per arm, so its legend states that flight; the scene's tally over
    # every seed stays in the sidecar, and the rates that pool flights are B, C and D.
    lx = f"XPU-RT ✓ {xg} of 4 gates (colour = time)"
    # a baseline that ran out of time struck nothing, so the legend says what it did instead
    lr = (f"ROS 2 ✗ {rg} of 4 gates, out of time ({hr:.0f} Hz control)" if r_out == "timeout"
          else f"ROS 2 ✗ {rg} of 4 gates, hits a {hit} ({hr:.0f} Hz control)")
    axA.legend(handles=[Line2D([0], [0], color=S.CMAP(0.6), lw=5, label=lx), Line2D([0], [0], color=C_ROS, lw=5, label=lr),
                        Line2D([0], [0], marker="o", color="0.45", mec="white", ls="none", ms=9, label=f"● = where each drone was, every {STROBE_S:.0f} s"),
                        Line2D([0], [0], marker="o", color=V.C_MOVER, mec="white", ls="none", ms=9, alpha=0.75, label="patrolling people"),
                        Line2D([0], [0], marker="o", mfc="none", mec="#ffd400", mew=3, ls="none", label="gate")], loc="lower left", fontsize=fz["leg"], framealpha=0.93, ncol=2, handlelength=1.9)
    gx, gr = float(X["moment_scale"]), float(R["moment_scale"])
    # the whole "what is held equal" clause, because the two gain policies phrase it differently
    gain_note = ("same scene, drone and gain" if abs(gx - gr) < 1e-9
                 else "same scene and drone, each arm at gain 0.5 / its own control rate")
    # one flight per arm is drawn, so the scene's own rate over every seed is stated rather than implied
    # A scene where neither arm completes is still not a tie: the gates each reaches separate them, and
    # that is stated exactly when the completion counts do not -- the counts say it otherwise.
    # the two flights share a scene when they share a layout seed; the episode seed is a separate
    # draw, and a pair flown at two of them says so rather than printing one of the two
    _sx, _sr = int(X["seed"]), int(R["seed"])
    _one = (f"one seed of this scene (seed {_sx})" if _sx == _sr
            else f"scene {int(X['layout_seed'])} · seeds: XPU-RT {_sx}, ROS 2 {_sr}")
    if not (nx and nr):
        seed_note = _one
    elif kx or kr:
        seed_note = f"{_one}; over its {nx} seeds XPU-RT completes {kx if kx else 'none'}, ROS 2 {kr if kr else 'none'}"
    else:
        _reach = f" — {gx_sc:.1f} vs {gr_sc:.1f} of 4 gates" if gx_sc is not None and gr_sc is not None else ""
        seed_note = f"{_one}; over its {nx} seeds neither arm completes{_reach}"
    # how the baseline's flight ended, in its own terms: a crash names what it struck and where in the
    # course; a time-out names the gate it was still short of, because it struck nothing
    ros_note = (f"runs out of time short of G{rg + 1}" if r_out == "timeout"
                else f"clears G{rg} → hits a {hit}" if rg else f"hits a {hit} before G1")
    # The header is a left-aligned axes title, so a long arm label runs past panel A and into panel
    # B's own title. Each line is wrapped to the width panel A actually occupies, measured from the
    # axes rather than assumed, so a longer label makes the header taller instead of overlapping.
    _tfs = fz["title"] * 1.05
    _cols = max(40, int(axA.get_position().width * fig.get_figwidth() * 72.0 / (0.55 * _tfs)))
    _head = [f"Warehouse gate-course showdown — {gain_note}, at {a.display_cruise:.1f} m/s",
             f"{a.xpu_label} ({hx:.0f} Hz control) clears all {xg} gates",
             f"{a.ros_label} ({hr:.0f} Hz control) {ros_note}",
             seed_note]
    axA.set_title("\n".join(textwrap.fill(ln, _cols, subsequent_indent="    ") for ln in _head),
                  fontsize=_tfs, weight="bold", loc="left")
    MB = F.draw_envelope_paper(axB, cax, hx, hr, fz, base); MB["k_n_per_hz"] = envelope_counts(os.path.join(RES, "hil_ablation.csv"))
    draw_generalization(axC, fs=base / 12.0, compact=True); axC.set_title("Generalization: unseen gates (course B)", fontsize=fz["title"], weight="bold", loc="left")
    draw_mechanism(axD, fs=base / 10.5, compact=True)
    # the ratios come from a small energy campaign, so its size is stated rather than left to the caption
    _nE = min(collections.Counter(r["flight"].rsplit("_s", 1)[0] for r in csv.DictReader(open(os.environ["ENERGY_CSV"]))).values())
    axD.set_title(f"Mechanism: the baseline thrashes (n={_nE} flights/arm)", fontsize=fz["title"], weight="bold", loc="left")
    badge(axA, "A", fz, dx=-30, dy=-24); badge(axB, "B", fz); badge(axC, "C", fz); badge(axD, "D", fz)
    # a–d
    F.strips(fig, outer[2].subgridspec(1, 4, wspace=0.09), moments, X, R, a.xpu_dir, a.ros_dir, fz)
    # E–H
    tg = outer[4].subgridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.30); axT = [fig.add_subplot(tg[i]) for i in range(4)]; MT_ = F.draw_telemetry(axT, X, R, fz, outcome=r_out)
    for ax_, s_ in zip(axT, "EFGH"):
        badge(ax_, s_, fz)
    # I
    axI = fig.add_subplot(outer[6]); sides, MI = gantt_two_rows(axI, fz, base, merge=a.gantt_merge, names=tuple(a.gantt_rows.split(",")), prefix=a.gantt_prefix, camera_hz=a.camera_hz, xpu_label=a.xpu_label); badge(axI, "I", fz, dx=-22, dy=30)
    fig.savefig(a.out + ".png", dpi=a.dpi, bbox_inches="tight"); fig.savefig(a.out + ".pdf", bbox_inches="tight"); print("wrote", a.out + ".png")
    inputs = [os.path.join(RES, "hil_ablation.csv"), os.path.join(RES, "hil_ablation_courseB.csv"), os.environ["ENERGY_CSV"], os.path.join(a.scene_records, "campaign.csv"),
              os.path.join(a.xpu_dir, "figure_data.npz"), os.path.join(a.ros_dir, "figure_data.npz")] + sides
    side = {"figure": a.out + ".png", "written": datetime.datetime.now().isoformat(timespec="seconds"),
            "sources": {"xpu_dir": a.xpu_dir, "ros_dir": a.ros_dir, "scene_records": a.scene_records, "gantt_sidecars": sides, "envelope_csv": os.path.join(RES, "hil_ablation.csv"), "courseB_csv": os.path.join(RES, "hil_ablation_courseB.csv"), "energy_csv": os.environ["ENERGY_CSV"]},
            "A": {"gain_xpu": gx, "gain_ros": gr, "xpu_gates": xg, "ros_gates": rg, "hit": hit,
                  # the failure the form is drawn to show, declared at render time and checked against the dump
                  "ros_failure": "before_first_gate" if a.baseline_fails_before_first_gate else "in_the_course", "xpu_eff_hz": round(hx, 1), "ros_eff_hz": round(hr, 1), "backdrop": bg_note, "plate_std": round(bg_std, 1),
                  # the clearance the panel draws, and whether an obstacle was actually found to measure it against
                  "clearance_m": round(float(near_miss[2]), 3), "clearance_measured": near_miss[1] is not None,
                  # the four moments the strips draw, each with the time its own flight reports at that step
                  "moments": [{"arm": _src, "step": int(_st), "label": _lb,
                               "t_s": round(float((R if _src == "ROS" else X)["t_s"][min(int(_st), len((R if _src == "ROS" else X)["t_s"]) - 1)]), 2)}
                              for _src, _st, _lb in moments],
                  # the scene both flights were drawn from, and the episode seed each was flown at
                  "layout_seed": {"xpu": int(X["layout_seed"]), "ros": int(R["layout_seed"])},
                  "episode_seed": {"xpu": _sx, "ros": _sr},
                  "scene_completed": {"xpu": [kx, nx], "ros": [kr, nr]},
                  # what the scene separates the arms by when neither completes it
                  "scene_mean_gates": {"xpu": gx_sc, "ros": gr_sc}, "ros_outcome": r_out, "display_cruise": a.display_cruise,
                  "xpu_trace": a.xpu_trace, "ros_trace": a.ros_trace, "ros_label": a.ros_label, "xpu_label": a.xpu_label, "camera_hz": a.camera_hz},
            "B": MB, "C": {"courseB_k_n_per_hz": envelope_counts(os.path.join(RES, "hil_ablation_courseB.csv"))}, "D": energy_ratios(os.environ["ENERGY_CSV"]), "telemetry": MT_, "I": MI,
            **FC.sidecar_common("showdown_paper_figure", [p for p in inputs if os.path.exists(p)])}
    json.dump(side, open(a.out + "_metrics.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

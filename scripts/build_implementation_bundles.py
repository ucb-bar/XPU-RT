#!/usr/bin/env python3
"""One self-contained directory per comparison, and one per ROS 2 deployment.

The work is spread across registries, sidecars, campaign CSVs, reproduction pages and a figure tree.
That is right for machines and wrong for a person who wants to see one result and how it was made.
This lays the same material out the other way round: a directory per figure holding what that figure
is, the command that rebuilds it, and the small data behind it; and a directory per ROS 2 deployment
holding every rate it was measured at and every figure that draws it.

    scripts/build_implementation_bundles.py [--out artifact/implementations] [--check]

Everything here is generated from the sidecars and the registries, so it cannot drift from them;
`--check` fails when it has. Two deliberate choices:

  * the rendered figure is referenced by relative path, not copied. A 4 MB PNG per bundle would put
    24 MB of duplicated binaries in the tree to show what one path already shows, and the README
    renders it either way.
  * the reproduce script delegates to `scripts/render_audited_set.sh` rather than restating its
    arguments, so a bundle cannot document a command the renderer no longer uses.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
RES = "results/codesign_feedback"

# One line per figure saying what the comparison is for. Facts come from the sidecar; this is the
# part a generator cannot derive -- why the figure exists and how it must not be read.
INTENT = {
    "showdown_36hz_solver_vs_rosallhart_s1006":
        ("The control-rate floor against the strongest all-hart baseline we have measured: two YOLO "
         "pools and a nav pool over every hart, so nothing in the perception chain runs alone. The "
         "baseline clears two gates and loses the course at the third.", None),
    "showdown_36hz_solver_vs_rosallhart_s1006_ladder":
        ("The same pair with panel D drawn as a four-rung ladder over scheduling quality rather than "
         "two bars.", None),
    "showdown_45hz_pinned_vs_rosdefault_s1000":
        ("ROS 2 as it is normally written -- unpinned, one process per node, default executor -- "
         "against a hand-pinned CP-SAT schedule. The baseline's chain is 242 ms and 99.2 % of its "
         "frames are late.",
         "Label it 'ROS 2, default deployment'. It is the representative default, not the best ROS 2 "
         "can do, and the pinned arms are measured separately."),
    "showdown_45hz_pinned_vs_rosdefault_s1003":
        ("The same arms and scene at a second display seed.",
         "Its display pair has no producer script: re-renderable from the archive, not regenerable. "
         "Callout (b) lands on a frame with no usable image."),
    "showdown_45hz_pinned_vs_rospinned_s1011":
        ("The falsification test. A statically partitioned ROS 2 -- three pinned processes, YOLO on "
         "its own four harts -- against a hand-pinned CP-SAT schedule at the same 57 ms chain, so "
         "command rate is the only difference left.",
         "This is a TIE and must never be captioned as a win: 4/12 completions each, and the "
         "baseline's mean gate count is higher (2.83 against 2.75). It bounds the claim."),
    "showdown_45hz_solver_vs_rospinned_s1007":
        ("The same partitioned baseline against a schedule whose placement the solver chose rather "
         "than a person. The gap the tie at s1011 closes, reopens.",
         "Both arms meet their perception window on every frame; the separation is command rate on "
         "top of a 2.1x shorter chain."),
}


def sidecars(stems):
    out = {}
    for s in stems:
        p = os.path.join(REPO, RES, "refined", f"{s}_metrics.json")
        if os.path.exists(p):
            try:
                out[s] = json.load(open(p))
            except Exception:
                pass
    return out


def audited_stems():
    """The showdown stems the artifact's own gate checks, read from the driver so the two agree."""
    p = os.path.join(REPO, "artifact", "verify_no_hardware.sh")
    m = re.search(r'^FIGURES="\$\{FIGURES:-(.*?)\}"', open(p).read(), re.M)
    got = [s for s in (m.group(1).split() if m else []) if s.startswith("showdown_")]
    # the ladder variant is rendered by the same driver and belongs beside its figure
    for s in list(INTENT):
        if s not in got and os.path.exists(os.path.join(REPO, RES, "refined", f"{s}_metrics.json")):
            got.append(s)
    return sorted(got)


def rel(frm, to):
    return os.path.relpath(os.path.join(REPO, to), frm)


def write(path, text, made):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    made[os.path.relpath(path, REPO)] = text
    return text


def _xpurt_manifests():
    """How many runs the board actually holds for XPU-RT, so the index can say what it is not showing."""
    import glob as _g
    return _g.glob(os.path.join(REPO, RES, "xpurt_long", "manifest_*.json"))


def script_arms():
    """arm -> what the board script builds for it, from the case statement in ros_traced_matrix.sh.

    That script is the only place saying what an arm name *means* -- which traced binary, which
    taskset, which node flags. Parsing it here means a page cannot describe a deployment the script
    does not build, and a new arm gets a page the day it is added rather than when someone
    remembers to write one.
    """
    out = {}
    for line in open(os.path.join(REPO, "scripts/ros_traced_matrix.sh")):
        m = re.match(r"\s{2}([a-z0-9_|]+)\)\s+BIN=([A-Za-z0-9_]+)(.*)$", line)
        if not m:
            continue
        rest = m.group(3)
        note = rest.split("#", 1)[1].strip() if "#" in rest else ""
        ex = re.search(r'EXTRA="([^"]*)"', rest)
        pre = re.search(r'PRE="([^"]*)"', rest)
        for n in m.group(1).split("|"):
            out[n] = {"bin": m.group(2), "extra": ex.group(1) if ex else "",
                      "pre": pre.group(1) if pre else "", "note": note}
    return out


# tags are "<hz>_<arm><suffix>_r<rep>" (ros_traced_matrix.sh:100). These are measured but are not
# deployments of the chain: two single-network calibration runs and two smoke tests.
NOT_A_DEPLOYMENT = {"yolo_standalone", "nav_standalone", "smoke", "vanilla_smoke"}


def ros_runs(arms):
    """config -> {rates: {hz: reps}, arm, suffix, manifest} from the run directories themselves.

    Ground truth is the tree, not a registry: a run that happened has a directory. Splitting the
    config into arm and suffix by longest match against the script's own arm list recovers exactly
    the two variables the tag was built from, so the reproduce command can be derived rather than
    restated.
    """
    base = os.path.join(REPO, RES, "ros_traced")
    got = collections.defaultdict(lambda: {"rates": collections.defaultdict(int),
                                           "manifest": None, "nav_pool": 0, "nav_harts": "",
                                           "goals": 0, "procs": 0})
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        m = re.match(r"^(\d+)_(.+)_r(\d+)$", d)
        if not m or not os.path.isdir(os.path.join(base, d)):
            continue
        hz, cfg = int(m.group(1)), m.group(2)
        if cfg in NOT_A_DEPLOYMENT:
            continue
        g = got[cfg]
        g["rates"][hz] += 1
        mf = os.path.join(base, d, "manifest.json")
        if os.path.exists(mf):
            try:
                j = json.load(open(mf))
                procs = list(j["processes"].values()) if "processes" in j else [j]
                if g["manifest"] is None:
                    # A deployment is spread over processes and each one records only its own share:
                    # the control process carries ctrl_mode, the nav process its pool. Merging by
                    # first-non-empty recovers the run's configuration; reading process[0] alone
                    # reports whichever fields the camera happens to carry.
                    merged = {}
                    for q in procs:
                        for k2, v2 in q.items():
                            if k2 not in merged or merged[k2] in (None, "", 0, -1):
                                if v2 not in (None, ""):
                                    merged.setdefault(k2, v2)
                                    if merged[k2] in (None, "", 0, -1):
                                        merged[k2] = v2
                    # ctrl_mode describes ONE node's behaviour, so only the process that runs the
                    # control node reports it meaningfully; the others carry the flag's default.
                    # Taking it from any other process inverts exactly the distinction that decides
                    # whether a command rate is the camera's or the timer's.
                    merged.pop("ctrl_mode", None)          # accumulated over every run below
                    g["manifest"] = merged
                # ctrl_mode describes ONE node's behaviour, so only the process running the control
                # node reports it meaningfully. And the earliest runs of eight arms predate the field
                # (ros_arm_ranking.md 5.1): freezing on the lowest-sorted rate's manifest is what
                # labels a timer arm "chained", so take it from whichever run recorded it.
                for q in procs:
                    if "control" in (q.get("nodes") or "") and q.get("ctrl_mode"):
                        g.setdefault("ctrl_mode", q["ctrl_mode"])
                    g["nav_pool"] = max((q.get("nav_pool") or 0) for q in procs)
                    g["nav_harts"] = next((q.get("nav_harts") for q in procs if q.get("nav_harts")), "")
                # A run that recorded no goal did not get through the chain, so the processes that
                # carry the knobs may never have written a manifest at all -- reading the
                # configuration back off it would produce a command missing exactly the flags the
                # run was made to test. Track it and let ros_command() decline rather than guess.
                g["goals"] = max(g["goals"], sum((q.get("n_goals") or 0) for q in procs))
                g["procs"] = max(g["procs"], len(procs))
            except Exception:
                pass
    for cfg, g in got.items():
        cands = [a for a in arms if cfg == a or cfg.startswith(a)]
        g["arm"] = max(cands, key=len) if cands else cfg
        g["suffix"] = cfg[len(g["arm"]):]
    return got


def ros_measured():
    """config -> hz -> what the runs at that rate recorded about themselves, from their summary.json.

    The registries key only the (arm, rate) pairs a figure or a document prints -- 68 of the 197 pairs
    that have runs. Every run writes its own summary, tracked for all of them, so a page can report
    what a deployment measured at every rate it ran at instead of a dash. Where a registry does key
    the pair the two agree (`--check` compares them), so this is the same measurement read closer to
    the run.
    """
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    base = os.path.join(REPO, RES, "ros_traced")
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        m = re.match(r"^(\d+)_(.+)_r\d+$", d)
        if not m or m.group(2) in NOT_A_DEPLOYMENT:
            continue
        f = os.path.join(base, d, "summary.json")
        if not os.path.exists(f):
            continue
        try:
            out[m.group(2)][int(m.group(1))].append(json.load(open(f)))
        except Exception:
            pass
    med = collections.defaultdict(dict)
    for cfg, rates in out.items():
        for hz, runs in rates.items():
            def mid(key):
                v = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
                return statistics.median(v) if v else None
            med[cfg][hz] = {"reps": len(runs), "gap_mean_ms": mid("gap_mean_ms"),
                            "gap_max_ms": mid("gap_max_ms"), "goal_ms": mid("e2e_goal_med_ms"),
                            "frames": mid("n_frames"), "goals": mid("n_goals")}
    return med


def cadence_census(runs, run_meas):
    """Across every deployment and rate: is a chained command rate ever faster than its camera?

    The showdown rests on one mechanism -- a chained deployment cannot command faster than frames
    arrive, a timer one can because it re-sends a held goal. This counts it over everything measured
    instead of over the two arms a figure happens to draw, so the claim is a census rather than an
    example, and a counterexample would show up here rather than in review.
    """
    ch = tm = over = at = 0
    worst = None
    for cfg, g in runs.items():
        mode = g.get("ctrl_mode")
        for hz, mv in run_meas.get(cfg, {}).items():
            if not mv.get("gap_mean_ms"):
                continue
            rate = 1000.0 / mv["gap_mean_ms"]
            if mode == "chained":
                ch += 1
                if rate > hz * 1.05:
                    over += 1
                    if worst is None or rate / hz > worst[2] / worst[1]:
                        worst = (cfg, hz, rate)
                elif abs(rate - hz) / hz < 0.12:
                    at += 1
            elif mode == "timer":
                tm += 1
    return {"chained": ch, "timer": tm, "over": over, "at_camera": at, "worst": worst}


def ros_command(cfg, g, arms):
    """The one command that re-measures this deployment, derived from the tag and the manifest.

    `ros_traced_matrix.sh` reads RATES, CTRL_HZ, QOS, HOGS and SUFFIX and writes all five into the
    tag and the manifest; so every knob that differs from the script's own default is recoverable,
    and nothing has to be kept in step by hand.
    """
    if not g.get("goals"):
        return None
    mf, env = g.get("manifest") or {}, []
    rates = " ".join(str(h) for h in sorted(g["rates"]))
    # BINSUF selects a differently-LINKED copy of the same program (ros_traced_matrix.sh:36-43):
    # the kernel set, and the nav build a nav pool needs. Neither is a layout, so neither is in
    # the arm name -- one shows up as a token in the suffix, the other as a pool in the manifest.
    # RVV is what the default binary is linked against, so a suffix naming it selects nothing;
    # only a non-default linkage needs BINSUF.
    kern = [t for t in ("ime",) if t in g["suffix"]]
    if kern:
        env.append(f"BINSUF=_{kern[0]}")
    elif g.get("nav_pool") and not arms.get(g["arm"], {}).get("bin", "").endswith("_nav4"):
        env += ["BINSUF=_nav4", f"NAVPOOL={g['nav_pool']}"]
        if g.get("nav_harts"):
            env.append(f"NAVHARTS={g['nav_harts']}")
    if (mf.get("ctrl_hz") or 100) != 100:
        env.append(f"CTRL_HZ={mf['ctrl_hz']}")
    if (mf.get("qos_depth") or 10) != 10:
        env.append(f"QOS={mf['qos_depth']}")
    n = re.search(r"hog(\d+)", g["suffix"])
    if n:
        env.append(f"HOGS={n.group(1)}")
    if g["suffix"]:
        env.append(f'SUFFIX={g["suffix"]}')
    return f'RATES="{rates}" ' + "".join(e + " " for e in env) + \
           f'scripts/ros_traced_matrix.sh {g["arm"]}'


def figure_bundle(stem, d, outdir, made, FC, MT):
    A, S = d.get("A", {}), d.get("sources", {})
    bdir = os.path.join(outdir, "figures", stem)
    os.makedirs(bdir, exist_ok=True)
    what, caveat = INTENT.get(stem, ("", None))

    xt, rt = A.get("xpu_trace"), A.get("ros_trace")
    arms = []
    for side, t in (("XPU-RT", xt), ("ROS 2", rt)):
        a = FC.ARM_BY_TRACE.get(t)
        if not a:
            continue
        reg = (MT.SOLVER_ARMS.get(a.derive[1]) if a.derive and a.derive[0] == "SOLVER_ARMS"
               else (MT.ROS_VANILLA.get(a.derive[1]) if a.derive and a.derive[0] == "ROS_VANILLA"
                     else getattr(MT, "ROS_SENSITIVITY", {}).get(a.derive[1] if a.derive else "", {})))
        arms.append((side, t, a, reg or {}))

    I = d.get("I", {})
    rows = {k: v for k, v in I.items() if isinstance(v, dict) and "chain_ms_median" in v}
    cens = A.get("scene_completed") or {}
    mean = A.get("scene_mean_gates") or {}

    L = [f"# {stem}", "", what, ""]
    if caveat:
        L += [f"> **Read this first.** {caveat}", ""]
    L += [f"![{stem}]({rel(bdir, f'{RES}/refined/{stem}.png')})", "",
          f"Full size: [`{stem}.png`]({rel(bdir, f'{RES}/refined/{stem}.png')}) · "
          f"[`.pdf`]({rel(bdir, f'{RES}/refined/{stem}.pdf')}) · sidecar `metrics.json` in this directory.",
          "", "## The two arms", "",
          "| | arm | camera→control | control rate | census (completed/12, mean gates) |",
          "|---|---|---|---|---|"]
    for side, t, a, reg in arms:
        key = "xpu" if side == "XPU-RT" else "ros"
        c = cens.get(key)
        L.append(f"| {side} | `{a.derive[1] if a.derive else t}` via `{t}` | "
                 f"{reg.get('chain_ms', reg.get('chain_goal_ms', '—'))} ms | "
                 f"{round(1000.0 / reg['ctrl_gap_mean_ms']) if reg.get('ctrl_gap_mean_ms') else '—'} Hz | "
                 f"{f'{c[0]}/{c[1]}' if c else '—'}, {mean.get(key, '—')} |")

    L += ["", "## What the board measured (panel I)", "",
          "| row | camera→control | frames late |", "|---|---|---|"]
    for k, v in rows.items():
        L.append(f"| `{k}` | {v['chain_ms_median']} ms | {v.get('frames_late')}/{v.get('frames_checked')} |")

    L += ["", "## Rebuilding it", "", "```bash", f"bash reproduce.sh          # from this directory",
          "```", "",
          "That renders from data already in the repository and the archives — no board, no GPU. The",
          "board runs and the flights behind it need hardware; the reproduction page below carries them.",
          "", "## Where everything came from", "",
          "| what | where |", "|---|---|"]
    for k, lbl in (("xpu_dir", "XPU-RT flight"), ("ros_dir", "ROS 2 flight"),
                   ("scene_records", "scene census"), ("energy_csv", "panel D energy runs")):
        v = S.get(k)
        if v:
            L.append(f"| {lbl} | `{v.replace(REPO + '/', '')}` |")
    for g in S.get("gantt_sidecars", []):
        L.append(f"| panel I Gantt | `{g.replace(REPO + '/', '')}` |")
    page = {"showdown_36hz": "docs/Evaluation/showdown_cam36_allcores_reproduction.md",
            "showdown_45hz_pinned_vs_rosdefault": "docs/Evaluation/showdown_cam45_ros_unpinned_reproduction.md",
            "showdown_45hz_pinned_vs_rospinned": "docs/Evaluation/showdown_cam45_static6_reproduction.md",
            "showdown_45hz_solver_vs_rospinned": "docs/Evaluation/showdown_cam45_solver_placed_reproduction.md"}
    pg = next((v for k, v in page.items() if stem.startswith(k)), None)
    if pg:
        L.append(f"| full reproduction page | [`{pg}`]({rel(bdir, pg)}) |")
    L += ["", "Small data is copied into `data/` here; the display dumps are 80–300 MB each and stay in",
          "`archive_v3/` with their sha256 in the tracked manifest.", ""]
    write(os.path.join(bdir, "README.md"), "\n".join(L) + "\n", made)

    sh = ["#!/usr/bin/env bash",
          f"# Rebuild {stem} from data already in the repository and the archives.",
          "#",
          "# Delegates to the one renderer rather than restating its arguments, so this cannot document",
          "# a command the renderer no longer uses. Board runs and flights need hardware; see the",
          "# reproduction page named in README.md.",
          "set -eu",
          'cd "$(dirname "$0")/../../../.."           # repository root',
          f'ONLY={stem} bash scripts/render_audited_set.sh', ""]
    write(os.path.join(bdir, "reproduce.sh"), "\n".join(sh), made)
    return bdir


def _same_sidecar(cur, old):
    """Two sidecars record the same measurements, allowing for when and where they were written.

    A sidecar carries the wall-clock time of the render and the absolute path of every input, so the
    same figure re-rendered from the same data in a different directory produces different bytes and
    identical numbers. Comparing bytes reported six bundles stale in the clean-clone run for exactly
    that reason, while the run's own sidecar comparison -- which allows for both, for the same
    reason -- found every figure identical. A path is allowed to differ only when the two name the
    same file relative to their own trees; everything else is compared as it stands.
    """
    def same_path(a, b):
        if not (isinstance(a, str) and isinstance(b, str)):
            return False
        if not (a.startswith(REPO + "/") and b.startswith("/")):
            return False
        return b.endswith("/" + a[len(REPO) + 1:])

    def eq(a, b, key=None):
        if key == "written":
            return True
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(eq(a[k], b[k], k) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(eq(x, y, key) for x, y in zip(a, b))
        return a == b or same_path(a, b)

    return eq(cur, old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "artifact", "implementations"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    import figure_constants as FC
    import measured_timing as MT

    stems = audited_stems()
    side = sidecars(stems)
    made: dict[str, str] = {}
    copies: list[tuple[str, str]] = []

    for s in stems:
        if s not in side:
            continue
        b = figure_bundle(s, side[s], a.out, made, FC, MT)
        S = side[s].get("sources", {})
        copies.append((f"{RES}/refined/{s}_metrics.json", os.path.join(b, "metrics.json")))
        if S.get("scene_records"):
            copies.append((os.path.join(S["scene_records"].replace(REPO + "/", ""), "campaign.csv"),
                           os.path.join(b, "data", "scene_census.csv")))
        if S.get("energy_csv"):
            copies.append((S["energy_csv"].replace(REPO + "/", ""),
                           os.path.join(b, "data", "energy_runs.csv")))
        for t in (side[s]["A"].get("xpu_trace"), side[s]["A"].get("ros_trace")):
            if t:
                copies.append((f"{RES}/ctrl_traces/{t}", os.path.join(b, "data", t)))

    # ---- ROS 2 deployment index -------------------------------------------------
    # Every arrangement that has a run directory, not only the ones a registry keys: a deployment
    # we measured and did not draw is exactly what a reader comparing implementations wants to see.
    figs_by_arm = collections.defaultdict(set)
    for s_, d in side.items():
        for t in (d["A"].get("xpu_trace"), d["A"].get("ros_trace")):
            arm = FC.ARM_BY_TRACE.get(t)
            if arm and arm.derive and arm.derive[0] in ("ROS_VANILLA", "ROS_SENSITIVITY"):
                nm = arm.derive[1][0] if isinstance(arm.derive[1], tuple) else arm.derive[1]
                figs_by_arm[re.sub(r"^\d+_|_r\d+$", "", nm)].add(s_)

    arms = script_arms()
    runs = ros_runs(arms)
    run_meas = ros_measured()
    CEN = cadence_census(runs, run_meas)
    meas = collections.defaultdict(dict)                 # config -> hz -> registry record
    for (arm, hz), v in MT.ROS_VANILLA.items():
        meas[arm][hz] = v
    for k, v in getattr(MT, "ROS_SENSITIVITY", {}).items():
        m = re.match(r"^(\d+)_(.+)_r\d+$", k)
        if m:
            meas[m.group(2)].setdefault(int(m.group(1)), v)

    D = os.path.join(a.out, "ros_deployments")
    idx = ["# ROS 2 deployments — every arrangement measured on the K1", "",
           f"One directory per deployment: **{len(runs)}** of them, from "
           f"{sum(sum(g['rates'].values()) for g in runs.values())} runs under `{RES}/ros_traced/`.",
           "Each says what the arrangement is, every camera rate it ran at, what it measured, the one",
           "command that re-measures it, and which figures draw it.", "",
           "Generated from the run directories and `scripts/ros_traced_matrix.sh` itself, so a "
           "deployment", "gets a page the day it is first run. What each arrangement *is* in full — "
           "process count,",
           f"pinning, executor, pool, QoS, control mode — is in [`ros_arms_catalog.md`]"
           f"({rel(D, 'docs/Baselines/ros_arms_catalog.md')});",
           f"how they rank is [`ros_arm_ranking.md`]({rel(D, 'docs/Baselines/ros_arm_ranking.md')}).", "",
           "*Best control rate* is over every rate the deployment ran at, from the runs' own",
           "summaries rather than from a registry, so a rate that was measured and never drawn counts.",
           "`—` means no run of that deployment recorded a control gap at any rate.", "",
           "**Read the control rate together with the control mode.** A `chained` deployment computes a",
           "command from the frame that produced the goal, so its command rate *is* the rate fresh",
           "information arrives. A `timer` deployment fires on its own clock and, between goals, acts on",
           "the goal it is still holding — so it reports a high command rate while the information under",
           "that command is as old as the camera→goal column beside it. Both are in here because both",
           f"are real deployments; [`ros_arm_ranking.md`]({rel(D, 'docs/Baselines/ros_arm_ranking.md')}) §3 measures",
           "what each choice buys, and §2 is where the two are compared as baselines.", "",
           f"**Measured over every deployment and rate, not only the drawn pair.** Of the {CEN['chained']} "
           f"(deployment, rate) pairs whose",
           f"control node is chained, **{CEN['over']}** command faster than their camera — "
           f"{CEN['at_camera']} command at the camera rate and the",
           "rest slower still, because a chain that cannot keep up emits commands more slowly than "
           "frames arrive.",
           f"The {CEN['timer']} timer pairs are not bound that way: they re-send a held goal, so cadence "
           "and freshness part company.", "",
           "The mode here is read from the process that runs the control node, over every run of the",
           "deployment rather than the lowest-sorted one, because the earliest runs of eight arms",
           f"predate the field. [`ros_arms_catalog.md`]({rel(D, 'docs/Baselines/ros_arms_catalog.md')}) reads one",
           "run and so prints `p3`, `p8`, `yproc`, `nproc`, `multi`, `spin`, `ship` and `smte` as chained;",
           f"[`ros_arm_ranking.md`]({rel(D, 'docs/Baselines/ros_arm_ranking.md')}) §5.1 sets out which reading the",
           "measurements support — a chained control node cannot emit every 10 ms off a 45 Hz camera.", "",
           "| deployment | based on | control | camera rates | runs | best control rate "
           "| camera→goal at that rate | drawn in |", "|---|---|---|---|---|---|---|---|"]
    for cfg in sorted(runs):
        g = runs[cfg]
        rates = sorted(g["rates"])
        # the rate that gives the best cadence, so the latency beside it is that deployment's own
        cand = [(1000.0 / v["gap_mean_ms"], v.get("goal_ms")) for v in run_meas.get(cfg, {}).values()
                if v.get("gap_mean_ms")]
        best, at_lat = max(cand) if cand else (None, None)
        fs = sorted(figs_by_arm.get(cfg, ()))
        mode = g.get("ctrl_mode") or "—"
        idx.append(f"| [`{cfg}`]({cfg}/) | `{g['arm']}`{' + `' + g['suffix'] + '`' if g['suffix'] else ''} "
                   f"| {mode} | {', '.join(str(r) for r in rates)} Hz | {sum(g['rates'].values())} "
                   f"| {f'{best:.0f} Hz' if best else '—'} "
                   f"| {f'{at_lat:.1f} ms' if at_lat else '—'} | {', '.join(fs) or '—'} |")

        d2 = os.path.join(D, cfg)
        os.makedirs(d2, exist_ok=True)
        info = arms.get(g["arm"], {})
        R = [f"# ROS 2 deployment `{cfg}`", ""]
        if info.get("note"):
            R += [f"{info['note'][0].upper() + info['note'][1:]}", ""]
        R += [f"Measured at {len(rates)} camera rate(s) over {sum(g['rates'].values())} runs: "
              f"{', '.join(str(r) + ' Hz' for r in rates)}.", "",
              "## What it runs", "",
              "| | |", "|---|---|",
              f"| layout (arm) | `{g['arm']}` |"]
        if g["suffix"]:
            R.append(f"| variant (suffix) | `{g['suffix']}` |")
        R.append(f"| traced binary | `{info.get('bin', '—')}` |")
        if info.get("pre"):
            R.append(f"| confined to | `{info['pre']}` |")
        if info.get("extra"):
            R.append(f"| node flags | `{info['extra']}` |")
        mf = g.get("manifest") or {}
        for lbl, k in (("executor", "executor"), ("QoS depth", "qos_depth"),
                       ("control timer Hz", "ctrl_hz")):
            if mf.get(k) not in (None, ""):
                R.append(f"| {lbl} | `{mf[k]}` |")
        if g.get("ctrl_mode"):
            R.append(f"| control mode | `{g['ctrl_mode']}`"
                     + (" — fires on its own clock, holding the last goal between arrivals |"
                        if g["ctrl_mode"] == "timer"
                        else " — computed from the frame that produced the goal |"))
        if g.get("nav_pool"):
            R.append(f"| nav pool | `{g['nav_pool']}`"
                     + (f" on harts `{g['nav_harts']}`" if g.get("nav_harts") else "") + " |")
        R += ["", "## What it measured", "",
              "Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in",
              "*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by",
              "`measured_timing.py --verify`; the rest were measured and not drawn.", "",
              "| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals "
              "| in a registry |", "|---|---|---|---|---|---|---|"]
        for hz in rates:
            mv = run_meas.get(cfg, {}).get(hz, {})
            gp = mv.get("gap_mean_ms")
            keyed = meas.get(cfg, {}).get(hz)
            fg = (f"{mv['frames']:.0f}→{mv['goals']:.0f}"
                  if mv.get("frames") is not None and mv.get("goals") is not None else "—")
            goal = mv.get("goal_ms")
            R.append(f"| {hz} | {g['rates'][hz]} | {f'{goal:.1f}' if goal else '—'} | "
                     f"{f'{gp:.2f}' if gp else '—'} | {f'{1000.0 / gp:.0f} Hz' if gp else '—'} | "
                     f"{fg} | {'✔' if keyed else ''} |")
        cmd = ros_command(cfg, g, arms)
        R += ["", "## Re-measuring it", ""]
        if cmd:
            R += ["Needs the K1 board and the node built by `artifact/01_board/`:", "",
                  "```bash", cmd, "```", "",
                  f"Runs land in `{RES}/ros_traced/<hz>_{cfg}_r<n>/`.", ""]
        else:
            R += [f"**No command is derived for this deployment.** Its {sum(g['rates'].values())} run(s) "
                  "recorded no goal, and only", f"{g['procs']} of the deployment's processes wrote a "
                  "manifest, so the knobs it was run with are not",
                  f"recoverable from the run. What is known: the layout is `{g['arm']}` and the tag "
                  f"carries `{g['suffix']}`.",
                  f"[`ros_arm_ranking.md`]({rel(d2, 'docs/Baselines/ros_arm_ranking.md')}) §4 carries the row "
                  "for it, and points at the", "sibling deployments whose command form is recorded.", ""]
        if fs:
            R += ["## Figures drawing this deployment", ""] + \
                 [f"* [`{f}`]({rel(d2, 'artifact/implementations/figures/' + f)}/)" for f in fs] + [""]
        else:
            R += ["No figure draws this deployment; it is here because it was measured.", ""]
        R += [f"Ranking and what each choice buys: [`ros_arm_ranking.md`]"
              f"({rel(d2, 'docs/Baselines/ros_arm_ranking.md')}) · "
              f"full layout: [`ros_arms_catalog.md`]({rel(d2, 'docs/Baselines/ros_arms_catalog.md')})", ""]
        write(os.path.join(d2, "README.md"), "\n".join(R) + "\n", made)
    write(os.path.join(D, "README.md"), "\n".join(idx) + "\n", made)

    # ---- XPU-RT schedule index ----------------------------------------------------
    # The XPU-RT side of the same question. Two registries hold two different kinds of thing and
    # the distinction matters to a reader: a solved schedule the board then executed, against a
    # hand-built deployment point measured directly. Both get a directory; the index keeps them apart.
    figs_by_xarm = collections.defaultdict(set)
    for s_, d in side.items():
        for t in (d["A"].get("xpu_trace"), d["A"].get("ros_trace")):
            arm = FC.ARM_BY_TRACE.get(t)
            if arm and arm.derive and arm.derive[0] in ("SOLVER_ARMS", "XPURT_POINTS"):
                figs_by_xarm[arm.derive[1]].add(s_)

    import run_index as RI
    X = os.path.join(a.out, "xpurt_schedules")
    groups = [("Solved schedules executed on the board", "SOLVER_ARMS", MT.SOLVER_ARMS,
               "Each is a schedule a solver produced for one workload spec, carried through the "
               "codegen contract and run on the K1."),
              ("Measured deployment points", "XPURT_POINTS", MT.XPURT_POINTS,
               "Each is a hand-built arrangement measured directly, without a solve to reproduce.")]
    xidx = ["# XPU-RT schedules and deployment points — every arm measured on the K1", "",
            f"One directory per arm: **{sum(len(r) for _, _, r, _ in groups)}** of them. Each says "
            "what the arm is, what", "the board measured, and the recipe that rebuilds it.", "",
            "These are the arms whose numbers a figure or a document prints. The board also holds "
            f"{len(_xpurt_manifests())}", "run manifests under `" + RES + "/xpurt_long/`, most of them "
            "points in a solver sweep rather than",
            f"deployments; [`run_index.md`]({rel(X, 'docs/Artifact/run_index.md')}) indexes those.", ""]
    for title, reg_name, reg, blurb in groups:
        xidx += [f"## {title}", "", blurb, "",
                 "| arm | what it is | camera→control | control gap | frames late | drawn in |",
                 "|---|---|---|---|---|---|"]
        for k in sorted(reg):
            v = reg[k]
            what = v.get("solver") or (f"{v.get('camera_hz')} Hz camera"
                                       + (f", {v['policy']}" if v.get("policy") else ""))
            fs = sorted(figs_by_xarm.get(k, ()))
            xidx.append(f"| [`{k}`]({k}/) | {what} | {v.get('chain_ms', '—')} ms | "
                        f"{v.get('ctrl_gap_mean_ms', '—')} ms | {v.get('frames_late', '—')} | "
                        f"{', '.join(fs) or '—'} |")

            d3 = os.path.join(X, k)
            os.makedirs(d3, exist_ok=True)
            tbl, recipe = RI.board_recipe(k)
            L3 = [f"# XPU-RT arm `{k}`", "", what[0].upper() + what[1:] + ".", "",
                  "## What the board measured", "", "| | |", "|---|---|"]
            for lbl, kk, unit in (("camera→control (median)", "chain_ms", " ms"),
                                  ("camera→control (p95)", "chain_p95_ms", " ms"),
                                  ("control gap (mean)", "ctrl_gap_mean_ms", " ms"),
                                  ("control gap (max)", "ctrl_gap_max_ms", " ms"),
                                  ("control gaps over 15 ms", "ctrl_gaps_over_15ms", ""),
                                  ("frames late", "frames_late", ""),
                                  ("camera rate", "camera_hz", " Hz"),
                                  ("meets its window", "on_time", "")):
                if v.get(kk) is not None:
                    val = {True: "yes", False: "no"}.get(v[kk], v[kk])
                    L3.append(f"| {lbl} | {val}{unit} |")
            if v.get("ctrl_gap_mean_ms"):
                L3.append(f"| command rate | {1000.0 / v['ctrl_gap_mean_ms']:.0f} Hz |")
            L3 += ["", "## Rebuilding it", ""]
            if tbl:
                L3 += [f"Executed schedule: `{tbl}`", ""]
            if recipe:
                L3 += [f"Board recipe: {recipe}", "",
                       "Needs the K1 board and the kernels built by `artifact/01_board/`.", ""]
            else:
                L3 += ["No solve to re-run: the arrangement was built by hand and measured "
                       "directly. Its run", "manifest under `" + RES + "/xpurt_long/` records what "
                       "executed.", ""]
            L3 += [f"Registry entry: `scripts/measured_timing.py` → `{reg_name}[\"{k}\"]`, "
                   "re-derived from the", f"traces by `measured_timing.py --verify`.", ""]
            if fs:
                L3 += ["## Figures drawing this arm", ""] + \
                      [f"* [`{f}`]({rel(d3, 'artifact/implementations/figures/' + f)}/)" for f in fs] + [""]
            else:
                L3 += ["No figure draws this arm; it is here because it was measured.", ""]
            L3 += [f"Ranking: [`xpurt_arm_ranking.md`]({rel(d3, 'docs/Evaluation/xpurt_arm_ranking.md')}) · "
                   f"run index: [`run_index.md`]({rel(d3, 'docs/Artifact/run_index.md')})", ""]
            write(os.path.join(d3, "README.md"), "\n".join(L3) + "\n", made)
        xidx.append("")
    write(os.path.join(X, "README.md"), "\n".join(xidx) + "\n", made)

    # ---- top index ---------------------------------------------------------------
    top = ["# Implementations — one directory per result", "",
           "Each figure below is a comparison between one XPU-RT schedule and one ROS 2 deployment.",
           "Its directory holds what the figure is, the command that rebuilds it, the small data behind",
           "it, and pointers to the large data in `archive_v3/`.", "",
           "Generated by `scripts/build_implementation_bundles.py`; `--check` fails if it and the",
           "registries disagree.", "", "## Figures", "",
           "| comparison | camera | XPU-RT | ROS 2 |", "|---|---|---|---|"]
    for s in stems:
        if s not in side:
            continue
        A = side[s]["A"]
        top.append(f"| [`{s}`](figures/{s}/) | {A.get('camera_hz')} Hz | "
                   f"`{A.get('xpu_trace', '').replace('.csv', '')}` | "
                   f"`{A.get('ros_trace', '').replace('.csv', '')}` |")
    top += ["", "## Implementations, one directory each", "",
            f"| tree | what is in it | how many |", "|---|---|---|",
            f"| [`ros_deployments/`](ros_deployments/) | every ROS 2 arrangement measured on the K1 — "
            f"what it runs, every rate it ran at, the command that re-measures it | {len(runs)} |",
            f"| [`xpurt_schedules/`](xpurt_schedules/) | every XPU-RT arm — the solved schedules the "
            f"board executed, and the deployment points measured by hand | "
            f"{len(MT.SOLVER_ARMS) + len(MT.XPURT_POINTS)} |",
            f"| [`figures/`](figures/) | the audited comparisons, each drawing one arm from each tree "
            f"| {len([x for x in stems if x in side])} |", "",
            "A deployment with no figure is still here: what was measured and not drawn is part of",
            "the comparison.", "",
            "## Reading the differences", "",
            f"[`run_index.md`]({rel(a.out, 'docs/Artifact/run_index.md')}) is the one page that says what differs",
            "between any two runs, and carries the board recipe per arm.", ""]
    write(os.path.join(a.out, "README.md"), "\n".join(top) + "\n", made)

    # ---- emit or check -------------------------------------------------------------
    stale = []
    for relp, text in made.items():
        p = os.path.join(REPO, relp)
        cur = open(p).read() if os.path.exists(p) else None
        if cur != text:
            stale.append(relp)
            if not a.check:
                open(p, "w").write(text)
    for src, dst in copies:
        s_ = os.path.join(REPO, src)
        if not os.path.exists(s_):
            continue
        same = os.path.exists(dst) and open(s_, "rb").read() == open(dst, "rb").read()
        # A sidecar carries the wall-clock time it was written, so re-rendering a figure from
        # identical inputs produces different bytes and the same measurements. Comparing bytes
        # made this check report six bundles stale in the clean-clone run purely on that field,
        # while the run's own sidecar comparison -- which skips it for the same reason -- found
        # every figure identical. The copy on disk stays a byte copy; only the decision changes.
        if not same and s_.endswith(".json") and os.path.exists(dst):
            try:
                same = _same_sidecar(json.load(open(s_)), json.load(open(dst)))
            except Exception:
                pass
        if not same:
            stale.append(os.path.relpath(dst, REPO))
            if not a.check:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(s_, dst)
    # The pages report what each run recorded about itself; the registries report the subset a
    # figure prints. Where both cover a (deployment, rate) they must agree, or a page and a figure
    # are saying different things about one run.
    drift = []
    if CEN["over"]:
        c, hz, r = CEN["worst"]
        drift.append(f"a chained deployment commands faster than its camera: {c} at {hz} Hz "
                     f"-> {r:.1f} Hz ({CEN['over']} such pairs)")
    for cfg, rates in run_meas.items():
        for hz, mv in rates.items():
            rv = (meas.get(cfg, {}).get(hz) or {}).get("ctrl_gap_mean_ms")
            if rv and mv.get("gap_mean_ms") and abs(rv - mv["gap_mean_ms"]) > 0.35:
                drift.append(f"{cfg} @ {hz} Hz: registry {rv} ms, runs' summaries "
                             f"{mv['gap_mean_ms']:.2f} ms")
    n_pair = sum(1 for cfg, r in run_meas.items() for hz in r
                 if (meas.get(cfg, {}).get(hz) or {}).get("ctrl_gap_mean_ms"))
    if a.check:
        for s_ in stale[:6]:
            print(f"FAIL  {s_} is stale")
        for d_ in drift[:6]:
            print(f"FAIL  {d_}")
        print(f"\n{len(made) + len(copies) - len(stale)} up to date, {len(stale)} stale; "
              f"{n_pair - len(drift)}/{n_pair} registry values match the runs they came from")
        return 1 if (stale or drift) else 0
    if drift:
        for d_ in drift:
            print(f"FAIL  {d_}")
        return 1
    for d in [p for p in made if p.endswith("reproduce.sh")]:
        os.chmod(os.path.join(REPO, d), 0o755)
    print(f"wrote {len(made)} page(s) and {len(copies)} data file(s) under "
          f"{os.path.relpath(a.out, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

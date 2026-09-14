#!/usr/bin/env python3
"""XPURT paper figures -- the SHARED VISUAL VOCABULARY.

One import for every candidate in this directory, so that a reader can follow the
same episode and the same event from the cascade figure into the sim visualisation
into the metric panel.

=============================================================================
COLOUR -- three channels, each with ONE job, in disjoint hue families
=============================================================================

The house rules (dataviz skill) say: pick the form, then assign colour BY THE JOB
IT DOES, then VALIDATE. These are print figures on a white ground, so one mode is
selected (light); `palette_check.py` is the Python port of the skill's validator
(node is not installed here) and every set below is held to the ALL-PAIRS gate,
because scatter and small multiples are the dominant forms in this figure set.

  LANE       categorical, n=3   CPU / DSP / HTA compute lanes.
                                Hues: documented slots 7 / 2 / 3.
                                Used ONLY in story-A (mechanism) panels.
                                Sub-3:1 on white for DSP and HTA, so the RELIEF
                                rule applies: lanes are ALWAYS direct-labelled.

  PERIOD     sequential, blue   release period / command cadence, 0 -> 700 ms.
                                Light = fast cadence, dark = slow. This is the
                                primary sequential channel in stories B and C
                                because PERIOD is the variable that dominates.

  LATENCY    sequential, orange the second sequential context (observation age),
                                per the house rule that context 2 takes the next
                                categorical slot's hue as its own one-hue ramp.
                                Used sparingly -- latency is normally positional.

  STATUS     reserved           solver-cell state and success/failure. Never
                                reused for a series. Always ships icon + label.

Blue is deliberately NOT a lane colour: it is reserved for the period ramp, so a
lane bar and a schedule point can never be confused across facing pages.

TASK IS NEVER A COLOUR. Four tasks are shown as small multiples; task identity is
carried by a monochrome label chip plus the robot-family rule (widowx 40 ms grid /
google 333 ms grid), because a task facet already separates them positionally.

The two named reference arms sit INSIDE the period ramp and are marked by
annotation, not by a hue of their own:
  ideal (0 ms)     neutral grey fill  + hollow ring  + "unreachable reference"
  cpu685 baseline  ramp fill          + status ring  + "QNN baseline"

=============================================================================
EVENT GLYPHS -- shape carries the event, ink is monochrome
=============================================================================

  release ...............  |   thin vertical tick, hanging DOWN from the rail
  inference complete ....  v   filled down-triangle, sitting ON the rail
  actuation applied .....  .   dot on the tick rail (one per control tick)
  gripper closes ........  P   plus-filled marker
  grasp acquired ........  o   filled circle with a white ring
  contact ...............  D   diamond
  task success ..........  *   filled star
  task failure ..........  X   cross
  stall / miss ..........  ::  45-degree hatched block
  absent: infeasible ....  x   45-degree hatch + grey
  absent: unknown .......  ?   135-degree hatch + grey

=============================================================================
TIME AXIS CONVENTION
=============================================================================

x is ALWAYS wall-clock time, left to right, zero at episode (or window) start.
Milliseconds at schedule scale, seconds at episode scale, and the two are linked
by the RELEASE RAIL: a 1-row strip pinned to the top of every time-based panel
carrying release ticks and completion triangles at the same pitch. The rail is the
object a reader tracks between figures; `release_rail()` draws it identically
everywhere.
"""
from __future__ import annotations
import csv, io, json, glob, os, re, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

# --------------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
BASE = PAPER.parent
BOARD = Path(os.environ.get("OCTO_REPRO_RUNS",
                            "/scratch2/dima/misc_sw/XPU-RT/qnn_models/octo/repro_runs"))
CURATED = BASE / "traces_torque3"
PLANE = BASE / "g5grid" / "plane3_runs"
ARMS_TSV = BASE / "g5grid" / "arms.tsv"

# ------------------------------------------------------------------ colours
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

LANE = {"CPU": "#4a3aa7", "DSP": "#eb6834", "HTA": "#1baf7a"}
LANE_MACHINE = {"CPU": "CPU_X", "DSP": "CPU_E", "HTA": "CPU_P"}
LANE_ORDER = ["CPU", "DSP", "HTA"]

# documented blue ramp, 100 -> 700
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGE = ["#fbe0d4", "#f7c3ab", "#f3a582", "#ef8a5b", "#eb6834", "#d95926",
          "#bd4c1f", "#9c3f19", "#7c3213"]
CMAP_PERIOD = LinearSegmentedColormap.from_list("period", BLUE)
CMAP_LATENCY = LinearSegmentedColormap.from_list("latency", ORANGE)
# discrete ordinal steps for the 9-arm ladder (validated with --ordinal)
LADDER_STEPS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a",
          "critical": "#d03b3b"}
ABSENT = "#dcdbd5"          # a cell with NO data -- never a zero on the ramp
REF_GREY = "#8a8880"        # the ideal / unreachable reference arm

PERIOD_MAX = 700.0
LAT_MAX = 700.0


def c_period(p_ms, lo=90.0, hi=PERIOD_MAX):
    """Colour for a release period. Light = fast cadence, dark = slow."""
    return CMAP_PERIOD(np.clip((np.log(np.clip(p_ms, lo, hi)) - np.log(lo))
                               / (np.log(hi) - np.log(lo)), 0.06, 1.0))


def c_latency(l_ms, lo=0.0, hi=LAT_MAX):
    return CMAP_LATENCY(np.clip((l_ms - lo) / (hi - lo), 0.10, 1.0))


def norm_period(lo=90.0, hi=PERIOD_MAX):
    from matplotlib.colors import LogNorm
    return LogNorm(vmin=lo, vmax=hi)


# ------------------------------------------------------------------- glyphs
GLYPH = {
    "release":  dict(marker="|", ms=7, mew=1.5, color=INK2, ls="none"),
    "complete": dict(marker="v", ms=7, color=INK, ls="none", mec="white", mew=0.7),
    "actuate":  dict(marker=".", ms=2.6, color=MUTED, ls="none"),
    "grip":     dict(marker="P", ms=8, color=INK, ls="none", mec="white", mew=0.8),
    "grasp":    dict(marker="o", ms=8, color=INK, ls="none", mec="white", mew=1.4),
    "contact":  dict(marker="D", ms=5.5, color=INK2, ls="none", mec="white", mew=0.7),
    "success":  dict(marker="*", ms=13, color=STATUS["good"], ls="none", mec="white", mew=0.8),
    "failure":  dict(marker="X", ms=9, color=STATUS["critical"], ls="none", mec="white", mew=0.8),
}
GLYPH_LABEL = {
    "release": "release", "complete": "inference done", "actuate": "actuation",
    "grip": "gripper closes", "grasp": "grasp held", "contact": "contact",
    "success": "success", "failure": "failure",
}
HATCH_INFEASIBLE, HATCH_UNKNOWN = "///", "\\\\\\"


def ev(ax, x, y, kind, **kw):
    """Draw one class of event glyph. `y` may be a scalar or an array."""
    s = dict(GLYPH[kind]); s.update(kw)
    x = np.atleast_1d(x)
    y = np.full_like(x, y, dtype=float) if np.isscalar(y) else np.asarray(y)
    return ax.plot(x, y, **s, zorder=6, clip_on=False)


def glyph_handles(kinds):
    return [Line2D([], [], **GLYPH[k], label=GLYPH_LABEL[k]) for k in kinds]


# ------------------------------------------------------------ the time rail
def release_rail(ax, y, t0, t1, period, latency, *, complete=True,
                 lw=1.1, label=None, drop=None, ms=None):
    """The object a reader tracks between figures.

    A hairline at `y` spanning [t0, t1]; RELEASE ticks hang BELOW the rail and
    COMPLETION triangles sit ON it. The offset is not decoration: when latency
    equals the period the two events fall on the same instant, and a release tick
    drawn at the same y disappears under the triangle -- which is exactly the
    case (the un-pipelined baseline) the figure most needs to show.

    `drop` defaults to 1.6% of the axis' y-range so the same call works whether
    the panel is 3 lanes tall or 1400 ms tall.
    """
    if drop is None:
        lo, hi = ax.get_ylim()
        drop = abs(hi - lo) * 0.030
    inv = ax.get_ylim()[0] > ax.get_ylim()[1]      # inverted axes: hang the other way
    dy = -drop if not inv else drop
    ax.plot([t0, t1], [y, y], color=AXIS, lw=lw, zorder=2, solid_capstyle="butt")
    rel = np.arange(np.ceil(t0 / period) * period, t1, period)
    kw = {} if ms is None else {"ms": ms}
    ev(ax, rel, y + dy, "release", **kw)
    if complete:
        don = rel + latency
        ev(ax, don[don <= t1], y, "complete", **kw)
    if label:
        ax.text(t0, y, label + "  ", ha="right", va="center", fontsize=7.2,
                color=INK2, clip_on=False)
    return rel


# --------------------------------------------------------------- text style
def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8.0,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 8.6, "axes.labelsize": 7.8, "axes.titleweight": "bold",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 7.0,
        "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2,
        "lines.solid_capstyle": "round",
        "figure.dpi": 170, "savefig.dpi": 170,
    })


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, axis=grid, color=GRID, lw=0.6)
    return ax


def save(fig, name, note=""):
    """No bbox_inches='tight' -- LAYOUTS.md documents the 61,744-px failure."""
    p = HERE / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[ok] {p}  {fig.get_size_inches()[0]:.1f}x{fig.get_size_inches()[1]:.1f} in"
          + (f"  {note}" if note else ""))
    return p


# ====================================================================== DATA
# ---------------------------------------------------------- A: board traces
BOARD_GROUPS = {"mono": "mono_2*.log", "serial": "ungated_*.log",
                "p150w300": "p150w300*.log", "p130w275": "m130w275*.log",
                "pipe110": "pipe110x10*.log", "gc105": "gc105_2*.log"}


def board(group):
    """Every replicate of one board configuration, as parsed dispatch rows."""
    out = []
    for p in sorted(BOARD.glob(BOARD_GROUPS[group])):
        body = p.read_text(errors="replace")
        if "AGENTS_QNN_TRACE_BEGIN" not in body:
            continue
        body = body.split("AGENTS_QNN_TRACE_BEGIN", 1)[1].split("AGENTS_QNN_TRACE_END", 1)[0]
        raw = [l for l in body.splitlines() if l.strip() and not l.lstrip().startswith("===")]
        i = next((j for j, l in enumerate(raw) if l.startswith("seg_id,")), None)
        if i is None:
            continue
        rows = list(csv.DictReader(io.StringIO("\n".join(raw[i:]))))
        if not rows:
            continue
        t0 = min(float(r["actual_start_ms"]) for r in rows)
        for r in rows:
            r["s"] = float(r["actual_start_ms"]) - t0
            r["e"] = float(r["actual_end_ms"]) - t0
            r["lane"] = (r.get("actual_backend") or "").strip()
            r["inst"] = int(r.get("instance", 0) or 0)
            r["seg"] = (r.get("backend_label") or "").strip()
            r["pred"] = float(r["predicted_duration_ms"])
        out.append((p.name, rows))
    return out


def board_median(group):
    """The replicate whose per-inference span is closest to the group median."""
    reps = board(group)
    if not reps:
        raise SystemExit(f"no board logs for {group} under {BOARD}")

    def span(rows):
        iv = collections.defaultdict(lambda: [1e18, -1e18])
        for r in rows:
            v = iv[r["inst"]]
            v[0], v[1] = min(v[0], r["s"]), max(v[1], r["e"])
        return float(np.median([b - a for a, b in iv.values()]))
    sp = [span(r) for _, r in reps]
    k = int(np.argmin(np.abs(np.array(sp) - np.median(sp))))
    return reps[k][0], reps[k][1], sp


def inferences(rows):
    """{instance: (start, end)} in start order."""
    iv = {}
    for r in rows:
        a, b = iv.get(r["inst"], (r["s"], r["e"]))
        iv[r["inst"]] = (min(a, r["s"]), max(b, r["e"]))
    return dict(sorted(iv.items(), key=lambda kv: kv[1][0]))


# The three headline schedules, MEASURED on the board (LAYOUTS.md / fig_cascade).
CASCADE = [("QNN CPU baseline", 684.8, 684.8, "mono"),
           ("XPU-RT serial", 283.4, 283.4, "serial"),
           ("XPU-RT pipelined", 258.7, 124.8, "p150w300")]

# mix3 partition, from the board bring-up log (52 single-graph contexts).
PARTITION = {"CPU": 26, "DSP": 14, "HTA": 12}
N_DISPATCH, N_EVICT = 710, 0


# --------------------------------------------------------------- B: the grid
def grid_warm():
    d = json.load(open(PAPER / "grid_warm.json"))
    return d["cells"], d.get("tally", {}), d


def grid_greedy():
    return json.load(open(PAPER / "grid_greedy.json"))


def arms():
    """44 operating points: name -> (latency_ms, period_ms, makespan_ms)."""
    out = {}
    for line in open(ARMS_TSV):
        f = line.split()
        if f:
            out[f[0]] = (float(f[1]), float(f[2]), float(f[3]))
    return out


# -------------------------------------------------------------- C: the sims
TASKS = [("egg", "eggplant in basket", "widowx", 40.0),
         ("spoon", "spoon on towel", "widowx", 40.0),
         ("coke", "pick coke can", "google", 333.0),
         ("drawer", "close drawer", "google", 333.0)]
TASK_NAME = {t: n for t, n, _, _ in TASKS}
TASK_ROBOT = {t: r for t, _, r, _ in TASKS}
TASK_GRID = {t: g for t, _, _, g in TASKS}
ECH = "t2_drive_arm_sus"

# the curated ladder, slowest schedule last
CURATED_ARMS = ["lat0", "pipe110fix", "p105w300", "p130w275", "p150w300",
                "pipe200fix", "serial283", "fp32_555", "cpu685"]
CURATED_LABEL = {"lat0": "ideal", "pipe110fix": "pipe110", "p105w300": "p105/w300",
                 "p130w275": "p130/w275", "p150w300": "p150/w300",
                 "pipe200fix": "pipe200", "serial283": "serial", "fp32_555": "fp32",
                 "cpu685": "cpu685"}
REFERENCE, BASELINE = "lat0", "cpu685"


def curated():
    """task -> arm -> dict(seeds, success, energy_med, mission, latency, period, eps)."""
    out = collections.defaultdict(dict)
    for f in glob.glob(str(CURATED / "*_rng*" / "energy2.json")):
        m = re.match(r"(egg|spoon|coke|drawer)_(.+)_rng(\d+)$", Path(f).parent.name)
        if not m:
            continue
        task, arm, seed = m.group(1), m.group(2), int(m.group(3))
        out[(task, arm)][seed] = json.load(open(f))
    T = collections.defaultdict(dict)
    for (task, arm), sd in out.items():
        d0 = next(iter(sd.values()))
        sr = [100 * d["n_success"] / d["n_episodes"] for d in sd.values()]
        en = [float(np.median([e[ECH] for e in d["episodes"]])) for d in sd.values()]
        mt = [e["duration_s"] for d in sd.values() for e in d["episodes"] if e["success"]]
        eps = [e for d in sd.values() for e in d["episodes"]]
        T[task][arm] = dict(n_seeds=len(sd), latency=d0["latency_ms"],
                            period=d0["issue_period_ms"],
                            success=float(np.mean(sr)), success_seeds=sr,
                            energy=float(np.mean(en)), energy_seeds=en,
                            mission=float(np.mean(mt)) if mt else np.nan,
                            episodes=eps)
    return dict(T)


def plane():
    """The 44-arm plane. Delegates to the paper's own plane3_lib so the two
    cannot disagree."""
    import sys
    sys.path.insert(0, str(PAPER))
    import plane3_lib as L
    return L.table(), L.arm_plane()


def episode_series(cell, ep):
    """Per-actuation series for one curated episode: tick, drive tau^2, contact
    force, gripper force. Local only -- nothing is fetched."""
    z = np.load(CURATED / cell / "series.npz")
    k = f"{ep:02d}"
    return dict(tick=z[f"tick_{k}"], tau2=z[f"tau2_{k}"], tot2=z[f"tot2_{k}"],
                cF=z[f"cF_{k}"], cFmax=z[f"cFmax_{k}"], cG=z[f"cG_{k}"])


def episode_summary(cell):
    return json.load(open(CURATED / cell / "summary.json"))


# --------------------------------- measured per-tick action age (5 arms only)
AGE_RUNS = {
    "cpu685":    (BASE / "runs_video" / "vid_cpu685", 684.8, 684.8),
    "serial283": (BASE / "runs_video" / "vid_serial283", 283.4, 283.4),
    "pipe110":   (BASE / "runs_video" / "vid_pipe110", 117.7, 117.6),
    "ideal":     (BASE / "runs_video" / "vid_ctrl_lat0", 0.0, 200.0),
}
LADDER_RUNS = {a: (BASE / "videos_ladder" / f"egg_{a}", ) for a in
               ("cpu685", "serial283", "pipe200", "pipe110")}


def age_trace(arm, ep=0):
    """MEASURED per-tick observation age (ms) for one episode, plus the applied
    7-DoF action array and the episode's success flag."""
    d = AGE_RUNS[arm][0]
    age = np.load(d / f"ep{ep:02d}_action_age_ms.npy")
    act = np.load(d / f"ep{ep:02d}_applied_actions.npy")
    ok = bool(glob.glob(str(d / f"ep{ep:02d}_success_True.mp4")))
    return age, act, ok


def ladder_age(arm):
    d = BASE / "videos_ladder" / f"egg_{arm}"
    f = sorted(d.glob("ep*_action_age_ms.npy"))[0]
    n = f.name[:4]
    age = np.load(f)
    act = np.load(d / f"{n}_applied_actions.npy")
    s = json.load(open(d / "summary.json"))
    ok = bool(list(d.glob(f"{n}_success_True.mp4")))
    return age, act, ok, s


def video_frames(arm, ep=0, n=6, task="egg", src="ladder"):
    """Evenly spaced frames from a rendered rollout, extracted locally with
    ffmpeg into the scratch cache. No remote access, no GPU."""
    import subprocess, tempfile
    d = (BASE / "videos_ladder" / f"{task}_{arm}") if src == "ladder" \
        else AGE_RUNS[arm][0]
    mp4 = sorted(list(d.glob(f"ep{ep:02d}_success_*.mp4")))
    if not mp4:
        return []
    cache = HERE / ".frames" / f"{src}_{task}_{arm}_{ep:02d}"
    cache.mkdir(parents=True, exist_ok=True)
    if not list(cache.glob("f_*.jpg")):
        # 384 px wide at 4 fps, as JPEG. A rollout frame lands in the figure about
        # 150 px wide and is sampled at ~8 instants, so a full-rate full-size PNG
        # cache is ~300 MB in the paper directory for no visible gain; this is
        # ~1 MB and hits any requested instant to within 0.25 s. Frames stay
        # uniformly spaced over the episode, so index/time mapping is unchanged.
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4[0]),
                        "-vf", "fps=4,scale=384:-2", "-q:v", "3",
                        str(cache / "f_%04d.jpg")], check=True)
    fs = sorted(cache.glob("f_*.jpg"))
    if not fs:
        return []
    idx = np.linspace(0, len(fs) - 1, n).round().astype(int)
    return [(int(i), fs[i]) for i in idx], len(fs)


# ------------------------------------------------------------- the key sheet
def gripper_events(act, close_thresh=0.5):
    """Ticks where the commanded gripper crosses from open to closed.

    Column 6 of the applied action is the gripper command; the sign convention is
    read from the data itself (the closed state is whichever sign the episode
    spends its grasping phase in) rather than assumed.
    """
    g = act[:, -1]
    closed = g < close_thresh
    on = np.flatnonzero(closed[1:] & ~closed[:-1]) + 1
    off = np.flatnonzero(~closed[1:] & closed[:-1]) + 1
    return on, off, closed


def completions(age):
    """Ticks at which a FRESH inference landed -- the age series steps DOWN."""
    a = np.asarray(age, float)
    ok = np.isfinite(a)
    d = np.full(a.shape, np.nan)
    d[1:] = a[1:] - a[:-1]
    return np.flatnonzero(ok & (d < -1e-9))


if __name__ == "__main__":
    style()
    print("LANE      ", LANE)
    print("partition ", PARTITION, "dispatches", N_DISPATCH, "evictions", N_EVICT)
    a = arms()
    print(f"arms      {len(a)} operating points")
    c = grid_warm()[1]
    print("grid      ", c)
    T = curated()
    print("curated   ", {k: len(v) for k, v in T.items()})

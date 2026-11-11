#!/usr/bin/env python3
"""Emit the XPU-RT vs naive-ROS comparison as markdown, from the recorded results."""
import json, statistics as st

SW = "/scratch2/dima/misc_sw/XPU-RT/qnn_models/flow_c/sweeps/qrb5165_ros_pinsweep_20260909-233318"
OUT = f"{SW}/NAIVE_ROS_COMPARISON.md"

main = json.load(open(f"{SW}/results/analysis.json"))
tri = json.load(open(f"{SW}/results/analysis_3net.json"))
BAND = main["noise_floor"]["np_pct"]

rows = []
for c in main["cells"]:
    note = []
    if c.get("np_degenerate"):
        note.append("no aperiodic work")
    if not c.get("np_work_equal", True):
        note.append("unequal timed work")
    rows.append(dict(arm="wl_sweep", name=c["cell"][len("networks_"):], cfg=c.get("config"),
                     ros=c.get("iso_np_corrected_ms"), xrt=c.get("xrt_np_warmbest_corrected_ms"),
                     ratio=c.get("iso_over_xrt_np_warmbest_corrected"),
                     inside=c.get("inside_noise_iso_over_xrt_np_warmbest_corrected"),
                     note=", ".join(note)))
for s in tri["shapes"]:
    rows.append(dict(arm="3net", name=s["shape"], cfg="all lanes",
                     ros=s.get("iso_np_corrected_ms"), xrt=s.get("xrt_np_warmbest_corrected_ms"),
                     ratio=s.get("iso_over_xrt_np_warmbest_corrected"),
                     inside=s.get("inside_noise_iso_over_xrt_np_warmbest_corrected"),
                     note="no aperiodic work" if s.get("np_degenerate") else ""))
live = [r for r in rows if r["ratio"] and not r["note"]]


def scope(sel):
    v = [r["ratio"] for r in sel]
    return dict(n=len(v), med=st.median(v), xrt=sum(1 for x in v if x > 1),
                ros=sum(1 for x in v if x < 1), band=sum(1 for r in sel if r["inside"]))


S = [("`wl_sweep`, quad (the real machine)",
      scope([r for r in live if r["arm"] == "wl_sweep" and r["cfg"] == "quad"])),
     ("`wl_sweep`, all lane subsets",
      scope([r for r in live if r["arm"] == "wl_sweep"])),
     ("`3net`", scope([r for r in live if r["arm"] == "3net"])),
     ("**Every comparable cell**", scope(live))]

CFG = {"quad": "hta + dsp + cpu + gpu", "hd": "hta + dsp", "dc": "dsp + cpu",
       "cg": "cpu + gpu", "all lanes": "hta + dsp + cpu"}

L = []
w = L.append
w("# XPU-RT against the ROS deployment a team would actually write")
w("")
w("Every workload cell run twice on the same QRB5165, both timed from their own first")
w("dispatch: once under XPU-RT's per-operation scheduler solving with `cpsat:warmbest`,")
w("and once as ordinary ROS 2 nodes with each network pinned whole to the backend it is")
w("fastest on **in isolation**. No placement search on the ROS side — that is the point.")
w("")
w("The ratio is **ROS ÷ XPU-RT** on the non-periodic makespan, so **above 1.00 the")
w("scheduler is ahead**. Medians of 3 reps.")
w("")
w("## Where it lands")
w("")
w("| scope | cells | median | scheduler ahead | pinning ahead | within noise |")
w("|---|---:|---:|---:|---:|---:|")
for k, v in S:
    w(f"| {k} | {v['n']} | **{v['med']:.4f}** | {v['xrt']} | {v['ros']} | {v['band']} |")
w("")
w(f"A cell is comparable only if it has aperiodic work to time and both runtimes ran the")
w(f"same number of instances of it — **{len(live)} of {len(rows)}** qualify.")
w("")
w("## What each side is")
w("")
w("**XPU-RT** places individual operations across HTA, Hexagon DSP, CPU and GPU lanes and")
w("gates each dispatch to a scheduled start. The schedule comes from `cpsat:warmbest` —")
w("CP-SAT warm-started from the best feasible heuristic — which the solver study recommends")
w("for the build-time path.")
w("")
w("**Naive ROS** is one node per network, one executor, the whole dispatch graph on one")
w("backend, chosen by that network's own cost in isolation. No search, no coordination.")
w("An earlier version of this comparison let ROS pick the best of every legal placement;")
w("that is a placement oracle nobody has, and removing it reversed the conclusion.")
w("")
w("## Every cell")
w("")
w(f"Rows marked *excluded* are not part of any median above. `±{BAND}%` is this board's")
w("rep-to-rep spread on the objective; a difference inside it is a direction, not a result.")
w("")
w("| cell | lanes | ROS ms | XPU-RT ms | ratio | |")
w("|---|---|---:|---:|---:|---|")
last = None
for r in sorted(rows, key=lambda x: (x["arm"] != "wl_sweep", x["cfg"] or "", x["name"])):
    key = (r["arm"], r["cfg"])
    if key != last:
        w(f"| **{r['arm']} · {r['cfg']}** | *{CFG.get(r['cfg'],'')}* | | | | |")
        last = key
    if r["note"]:
        tag = f"*excluded — {r['note']}*"
        rat = f"{r['ratio']:.3f}" if r["ratio"] else "—"
    else:
        tag = "within noise" if r["inside"] else ("scheduler" if r["ratio"] > 1 else "pinning")
        rat = f"**{r['ratio']:.3f}**"
    w(f"| `{r['name']}` | {r['cfg']} | {r['ros']:.2f} | {r['xrt']:.2f} | {rat} | {tag} |")
w("")
w("## Why the two arms disagree")
w("")
w("On `wl_sweep` at quad the naive rule costs a median **1.17×** against the best placement")
w("available, and up to **1.86×** on `depth_contended_quad`, where all three networks prefer")
w("the DSP alone and all three land there. On `3net` the same rule is nearly optimal — a")
w("median gap of **1.018×** — because those networks spread naturally across lanes.")
w("")
w("**So whether naive pinning is good enough is not a property of the scheduler. It is a")
w("property of whether your networks contend for the same preferred lane.**")
w("")
w("The sharpest case is `3net_fused2_mlp8_yolo1`, where both placements run the timed")
w("`yolov8n` on the DSP and the naive one is still 17% slower:")
w("")
w("```")
w("naive    mlp_control@cpu   24 inst x 0.127 ms    yolov8n@dsp  34.676 ms")
w("oracle   mlp_control@dsp   24 inst x 0.580 ms    yolov8n@dsp  28.811 ms")
w("```")
w("")
w("Moving `mlp_control` to the DSP makes mlp itself 4.6× slower per instance and the timed")
w("network 17% faster, because QnnCpu builds its thread pool with full-machine affinity and")
w("CPU work starves the DSP lane's host-side driver thread. **An isolation cost model cannot")
w("see this**: `mlp_control` genuinely is faster on the CPU alone, 110.2 µs against 523.6.")
w("The naive rule picks correctly by its own criterion and still loses, because the cost")
w("lands on a different network on a different lane.")
w("")
w("## Read with these")
w("")
w(f"* **The band is wide.** {S[3][1]['band']} of {S[3][1]['n']} comparable cells sit inside")
w(f"  ±{BAND}%. Those are directions, not results.")
w("* **Only `quad` is a real machine.** A QRB5165 always has all four backends; `hd`, `dc`")
w("  and `cg` model lane subsets you cannot buy and distort individual cells badly — in `hd`")
w("  there is no CPU lane, so `mlp_control_sd` is forced onto the DSP at 523.6 µs though CPU")
w("  runs it in 110.2. They are kept as a lane-scarcity sensitivity study, not the headline.")
w("* **`scale_ladder_quad` is sampled, not enumerated** — 64 of 729 legal placements.")
w("  Widening that sample already improved the best known placement by 10.7%, which bounds")
w("  how wrong the sampled cells can be.")
w("* **Both sides are offset-corrected.** Each run is timed from its own first dispatch, so")
w("  neither runtime is charged for the other's start barrier. Raw and corrected figures are")
w("  both in `results/analysis.json`; only `perception_heavy` moves more than the band.")
w("")
w("---")
w("")
w("Generated from `results/analysis.json` and `results/analysis_3net.json` by")
w("`scripts/mk_naive_comparison.py`. Performance governor, one tenant at a time behind the")
w("board lock. Full method and corrections: `SETUP.md` and `ANALYSIS.md` in this directory.")
w("")

open(OUT, "w").write("\n".join(L))
print("wrote", OUT)

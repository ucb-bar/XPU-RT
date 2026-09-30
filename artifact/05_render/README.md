# 05 — rendering the figures, and verifying them

**Needs no hardware.** Every render below reads only on-disk artifacts; the verifier re-derives
every number the figure prints from the same files.

Six figures verify 0 FAIL today and are the current set (`docs/Artifact/artifact_checklist.md` §5):
`warehouse_showdown_paper_r36` — the 36 Hz rate-sweep form — and
`showdown_45hz_pinned_vs_rosdefault_s1003{,_ladder,_allcores,_allcores_merged,_c14}`. All six are drawn by
`scripts/showdown_paper_figure.py`; they differ in the display pair, the scene census and which
Gantt rows panel I carries.

> **No sidecar records the command line that wrote it.** `docs/Artifact/artifact_checklist.md` §5 says so.
> A command is recovered by one method: read the
> sidecar's `sources` and `I` blocks, take `--out` from the figure's own filename, and leave every
> flag the sidecar does not constrain at its default. §A below is recorded verbatim in a document.
> §B is recovered by that method and is **marked as such per flag**.

## A. `warehouse_showdown_paper_r36` — the 36 Hz rate-sweep form

`docs/Evaluation/showdown_rate_sweep_reproduction.md` §6, verbatim:

```bash
R=results/codesign_feedback
ENERGY_CSV=$R/flight_energy_r36.csv .venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $R/campaign_rate3640/display36/search_c1.2/xpu_s1000_figdata \
  --ros-dir $R/campaign_rate3640/display36/search_c1.2/ros_s1000_figdata \
  --scene-records $R/campaign_scene/r36_l1000 --display-cruise 1.2 \
  --xpu-trace xpu_w2pg36.csv --ros-trace ros_vanilla4x236.csv \
  --xpu-label "XPU-RT · greedy on measured costs" --ros-label "ROS 2 on all 8 cores" \
  --camera-hz 36 --gantt-prefix $R/refined/rate36/measured_gantt_r36 --gantt-rows xpu,ros8 \
  --out $R/refined/warehouse_showdown_paper_r36
```

Its panel I inputs are built first, from board traces alone (same document, §5):

```bash
D=results/codesign_feedback
.venv/bin/python scripts/make_measured_gantt_pair.py \
  --arm "xpu:xpu:$D/xpurt_long/trace_w2pg36r1_other_run1.csv:$D/xpurt_long/cpu_w2pg36r1_other_run1.csv:$D/xpurt_long/manifest_w2pg36r1_other_run1.json:schedules/fig_w2pg36_greedy_clamped.json" \
  --arm "ros8:ros:$D/ros_traced/36_vanilla4x2_r1/trace.csv:$D/ros_traced/36_vanilla4x2_r1/cpu.csv:$D/ros_traced/36_vanilla4x2_r1/manifest.json" \
  --spec data/toplevel/wh_chain36_w2p.json --window-ms 160 --out-prefix $D/refined/rate36/measured_gantt_r36
```

Each Gantt row opens its window where the row's own frames have the run's median latency; the
sidecar records both medians.

The panel A / a–d / telemetry inputs are the two display dumps, which are ignored and archived —
see [`../04_fly/`](../04_fly/) §2 for the tar and where to unpack it.

## B. The `showdown_45hz_pinned_vs_rosdefault_s1003` family

Same script, same defaults except where listed. Flags **recovered from the sidecar** are marked ▲;
flags left at the script's default are simply absent.

```bash
R=results/codesign_feedback
D=$R/campaign_v2/display_lat_c1.4               # ▲ sources.xpu_dir / sources.ros_dir
.venv/bin/python scripts/showdown_paper_figure.py \
  --xpu-dir $D/xpu_s1003_figdata --ros-dir $D/ros_s1003_figdata \
  --scene-records $R/campaign_scene/tall1000s \
  --display-cruise 1.4 \
  --gantt-rows <rows> [--gantt-merge] \
  --out $R/refined/<figure>
```

| figure | `--gantt-rows` ▲ (from the `I` block) | `--gantt-merge` ▲ (`I.merged_per_hart_and_frame`) | display pair ▲ (`sources`) |
|---|---|---|---|
| `showdown_45hz_pinned_vs_rosdefault_s1003` | `xpu,ros` (the default) | no | `display_lat_c1.4/{xpu,ros}_s1003_figdata` |
| `warehouse_showdown_paper_ladder` | `xpu,p3,ros` | no | `display_lat_c1.4/{xpu,ros}_s1003_figdata` |
| `warehouse_showdown_paper_allcores` | `xpu,ros,ros8` | no | `display_lat_c1.4/{xpu,ros}_s1003_figdata` |
| `warehouse_showdown_paper_allcores_merged` | `xpu,ros,ros8` | **yes** | `display_lat_c1.4/{xpu,ros}_s1003_figdata` |
| `showdown_45hz_pinned_vs_rosdefault_s1000` | `xpu,ros` | no | `display_v3s_c1.4/{xpu,ros}_s1000_figdata` |

All five record `sources.scene_records = campaign_scene/tall1000s`, `A.display_cruise = 1.4`,
`sources.energy_csv = flight_energy_v2.csv` (the repository default, so no `ENERGY_CSV` is needed)
and Gantt sidecars under `schedules/measured_gantt_v3` (the default `--gantt-prefix`). `--camera-hz`
is the default 45 for all five, `--width-in 20.0`, `--min-print-pt 3.6` and `--dpi 300` likewise.
`--xpu-trace`, `--ros-trace`, `--xpu-label` and `--ros-label` are **not recorded in the sidecar**;
the defaults are `xpu_a_cpsat_hard.csv`, `ros_vanilla445.csv`, `XPU-RT · CP-SAT` and `ROS 2 vanilla`,
which are the arms these five draw.

`--gantt-merge` joins each hart's consecutive dispatches of one frame into a single bar and keeps
every idle gap, so both rows carry the same unit of work as a ROS 2 callback
(`docs/Evaluation/figure_runbook.md` §3b item 11b).

## The verifier — this is the acceptance test

```bash
R=results/codesign_feedback
.venv/bin/python scripts/verify_showdown_figure.py --metrics $R/refined/warehouse_showdown_paper_r36_metrics.json
```

Must print `0 FAIL`. It re-derives panel A's per-scene tally, panels B and C's k/n per control rate,
panel D's energy ratios, panel I's per-row latencies and hart placement; it checks that each
sidecar's `inputs` hashes match the files on disk, that every Gantt row's executed table matches the
ledger `results/codesign_feedback/executed_tables.json`, that `fallbacks_used` is empty, that every
display literal comes from the registry `scripts/figure_constants.py`, and that the XPU-RT arm's
label names the solver recorded in the board run behind the trace it replays.

All six at once, together with the rest of the no-hardware suite (~45 s):

```bash
artifact/verify_no_hardware.sh
```

After any render, `docs/Evaluation/figure_runbook.md` §12 requires all four of these to report no failure:

```bash
. scripts/env.sh
$HOST_PY scripts/verify_showdown_figure.py --all     # every sidecar in refined/ — the backlog too
$HOST_PY scripts/measured_timing.py --verify
$HOST_PY scripts/executed_tables.py
$HOST_PY -m pytest tests xpu-rt/tests -q
```

`--all` reports far more than these six; `docs/Artifact/artifact_checklist.md` §5 splits that total three ways
(superseded-by-a-growing-census, a board number the roll-up no longer carries, and one attribution
gap).

## Other figures

`docs/Evaluation/figure_runbook.md` is the per-figure index: `schedule_evolution_mega` (§1), `hil_ablation_phase`
(§2), the warehouse forms (§3–3b), the story figures (§8), the atlas (§9), the final figure (§10),
`paper10` (§11), `solver_win_sensor` (§5), the supporting Gantts (§6) and `loop_ablation` (§7), with
`bash scripts/make_all_codesign_figures.sh` to regenerate what can be regenerated from cached
artifacts.

## Not verified here

The render commands were **not executed**: `results/codesign_feedback/refined/` is the output
directory of a figure build that was running while this page was written. §A is recorded verbatim in
`docs/Evaluation/showdown_rate_sweep_reproduction.md` §6 and §5; §B is recovered from the sidecars by the method
at the top of this page, flag by flag, with what each flag was recovered from named
above. The verifier invocations **were** run, over all six figures, and each printed `0 FAIL`.

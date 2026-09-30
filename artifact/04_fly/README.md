# 04 — the flights: census, displayed pair, scene census, energy runs

**Needs a GPU** (the Isaac Sim / IsaacLab conda env, `ISAAC_PY`). Nothing on this page was run while
this directory was written; every command is quoted from the document named beside it.

## What you have to supply, and what we do

Isaac Sim is yours to install — `docs/Artifact/environment.md` names the version (Isaac Sim 5.1.0.0, Python
3.11) and `requirements-isaac.txt` pins the interpreter. **Everything a flight loads is ours**, and
one command says whether it is all here:

```bash
.venv/bin/python scripts/verify_flight_inputs.py      # 0 failed
```

It checks, because a flight leaves no sidecar and no other check looks at these:

| what a flight loads | where it is |
|---|---|
| guidance net, controller, YOLO weights | `sims/models/warehouse/*.pt`, tracked (13 MB) |
| the replayed control cadence | `results/codesign_feedback/ctrl_traces/*.csv`, tracked, 26 named by drivers |
| the scene | `sims/isaaclab_tasks/` (54 files tracked) on IsaacLab pinned at `4df6560e` |
| the props | Isaac's own `Environments/Simple_Warehouse/…` — they come with Isaac |

So after `conda env create` and `git submodule update --init sims/IsaacLab`, a flight command from
this page has every input it names.

A flight replays two board measurements per arm: the control-output cadence (`--ctrl_trace`, cut in
[`../03_measure/`](../03_measure/)) and the camera-to-control latency (`--percep_latency_ms`). They
are **separate switches** — a pair flown with the cadence alone differs only in control rate — so
read a dump's own record rather than assuming. The navigation goal is held for one camera period in
both arms, since perception cannot refresh faster than the camera.

Flights are admitted one at a time under `results/codesign_feedback/gpu_admit.lock`; every driver
takes `MAX_SIMS` and `NEED_MB` and waits for GPU room. A batch whose simulator log shows a GPU fault
is written to `quarantine.txt` and listed in `results/codesign_feedback/flight_quarantine.csv`, which
every reader drops and `scripts/flight_quarantine.py` re-checks. Flight outcomes are **not
bit-reproducible**: the same seed re-flown in the display script can differ from the census, which is
why a pair is re-verified on the scene it flies and never carried over.

## 1. The census

```bash
MAX_SIMS=3 NEED_MB=10000 scripts/campaign_rate30.sh          # 30 Hz
MAX_SIMS=3 NEED_MB=10000 scripts/campaign_rate3640.sh        # 36 and 40 Hz  -> the paper figure
MAX_SIMS=3 NEED_MB=10000 scripts/campaign_submitted_config.sh   # ROS 2 out of the box (vanilla_c50)
```

Standard cell: course a, prop density 0.30, people 2.4 m, gain 0.0055, cruise 1.0–1.8 m/s, twelve
seeds — 60 paired cells per rate, one flight per (cruise, seed) per arm. Sources:
`docs/Evaluation/showdown_rate_sweep_reproduction.md` §3 and `docs/Evaluation/showdown_cam45_ros_out_of_box_reproduction.md` §1.

Read a census back with the paired statistics the figures quote — rows only through
`flight_quarantine.flight_rows`, a cell only when both arms flew it, completions over the
non-timeout flights, a Newcombe (Wilson) interval on the completion difference and a percentile
bootstrap (4000 resamples, seed 11) over the paired per-cell gate differences:

```bash
.venv/bin/python scripts/census_submitted.py \
    --csv results/codesign_feedback/campaign_rate3640/campaign.csv \
    --xpu xpu_w2pg36.csv --ros ros_vanilla4x236.csv
```

That one **needs no GPU** — it reads the committed CSV — and is `docs/Evaluation/showdown_cam45_ros_out_of_box_reproduction.md`
§1 verbatim. It was run here and reproduces the §3 r36 row.

## 2. The displayed pair

```bash
scripts/display_pairs_rate36.sh      # the census cells, re-flown with figure data dumped
scripts/display_search_rate36.sh     # every seed per cruise, in the display script itself
MAX_SIMS=3 NEED_MB=10000 scripts/display_search_submitted.sh   # env CRUISES, OD
```

The display script flies one episode through `sims/scripts/record_sensor_demo.py` with
`layout_seed = seed`, so a census cell is a *different scene* and a census outcome never carries
over. A pair is kept only when, on the scene it flies, XPU-RT completes and the baseline crashes
having entered the course (`ROS_GATES`; `verify_showdown_figure.py`'s `_compare_display_pair` is what
accepts it). Seeds are ordered with the census's completing seeds first — a search order, not a
selection: every attempt is logged and the pair is accepted on what the display flight itself does.
`docs/Evaluation/showdown_rate_sweep_reproduction.md` §4 records the pair the paper figure draws (cruise
1.2 m/s, seed 1000).

The dumps these write (`*_figdata/figure_data.npz`, 78–314 MB each) are ignored; the paper
figure's pair is archived as `results/codesign_feedback/archive_v3/display_dumps_r36.tar` (391 MB),
listed in that directory's tracked `MANIFEST.sha256`. Without it, panels A, a–d and the telemetry row
cannot be re-rendered at all; with it, unpack into
`results/codesign_feedback/campaign_rate3640/display36/search_c1.2/`.

## 3. Panel A's scene census

Twelve flights per arm on the display scene's layout, each arm replaying its own cadence, latency and
goal hold, every flight recorded. `docs/Evaluation/showdown_rate_sweep_reproduction.md` §4:

```bash
T=results/codesign_feedback/ctrl_traces
CELL=r36_l1000 LAYOUT_SEED=1000 CRUISE=1.2 MAX_SIMS=3 NEED_MB=10000 \
  ARMS="xpu:$PWD/$T/xpu_w2pg36.csv:30.1:27.8 ros8:$PWD/$T/ros_vanilla4x236.csv:32.2:27.8" \
  scripts/scene_runs_pair.sh
```

`ARMS` is `name:trace:latency_ms:hold_ms`. `scripts/scene_runs.sh` is the same thing with its arms
named in code.

## 4. Panel D's energy runs

One recorded flight per seed per condition with the commanded wrench logged, each condition replaying
its measured cadence, at the display cruise, with no latency or goal hold — the panel isolates what
the control cadence alone does. Totals via `scripts/flight_energy_model.py`.
`docs/Evaluation/showdown_rate_sweep_reproduction.md` §5:

```bash
T=$PWD/results/codesign_feedback/ctrl_traces
MAX_SIMS=3 NEED_MB=10000 CONDS="xpu_w2pg36:$T/xpu_w2pg36.csv ros_x2_36:$T/ros_vanilla4x236.csv" CRUISE=1.2 \
  ER=$PWD/results/codesign_feedback/energy_runs_r36 \
  OUTCSV=$PWD/results/codesign_feedback/flight_energy_r36.csv \
  scripts/run_energy_pair.sh
```

## 5. Panel I's Gantt — no hardware

The measured Gantt pair is built from board traces alone and needs neither the board nor the GPU;
it is in [`../05_render/`](../05_render/) with the render commands.

## Cost

Each flight is one episode of up to 1800 simulator steps under `record_sensor_demo.py` /
`sweep_rate_demo.py`, admitted one at a time. A census rate is 120 flights (5 cruise speeds × 12
seeds × 2 arms), a scene census 24, an energy pair 12. The wall-clock cost is GPU-hours and depends
on the machine; `docs/Artifact/reproduction_full.md` holds the queues that sequence a whole study.

## Older campaigns

`docs/Evaluation/figure_runbook.md` §3b item 4 and `docs/Artifact/reproduction_full.md` cover `campaign_v2.sh`,
`env_sweep.sh`, the `queue_v3*.sh` sequence and the rest of the second-form study; `scripts/attic/`
indexes the drivers that ran once and are not on the documented path.

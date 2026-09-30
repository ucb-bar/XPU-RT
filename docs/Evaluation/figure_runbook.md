# Figure runbook — how each headline PNG is made (inputs + exact command)

Every figure below is reproducible from on-disk artifacts. Environments, interpreters and machine paths:
`docs/Artifact/environment.md` §Environments (`scripts/env.sh` sets `REPO`, `SIM_TREE`, `HOST_PY`, `ISAAC_PY`; paths
below are relative to `$REPO`). Scheduler figures use the host venv (`$HOST_PY`); the two warehouse mega
plots + the HIL scatter need the Isaac env (`$ISAAC_PY`) to (re)generate their flight data.
`export XPURT_CPSAT_WORKERS=0` for any CP-SAT solve run by hand. Drivers that ran once for the study and
are not on the documented path are indexed in `scripts/attic/README.md` and `sims/scripts/attic/README.md`;
what this runbook names is what is outside `attic/`.

Composition scripts marked `scripts/…` live in the repo. Scripts marked `(scratchpad)` are the newer
figure builders that should be copied into `scripts/` for a clean checkout (they only read on-disk artifacts).

---

## 1. `schedule_evolution_mega` — the co-design "Gantt after Gantt" (the 3rd mega plot)
**Story** (contended sensor-fusion workload; HERO metric = **network-instance deadline misses** — hard-real-time
control, so makespan is secondary context — trajectory **2 → 0 → 4 → 0**, each panel a real solve or a
K1-calibrated re-cost). Panel 4 re-solves on the calibrated costs to **34.16 ms / 0 misses**, essentially the
same makespan as panel 3 but with every instance deadline restored:
1. **og** — RVV singleton dispatches: misses two FFN instance deadlines.
2. **+ shard + IME** — AOT levers (multi-hart widths + matrix-engine routing): meets every deadline ON THE GANTT
   (2 → 0).
3. **runtime feedback** — panel-2 schedule RE-COST with heterogeneous K1 calibration (per-dispatch/per-op
   measurements where available, aggregate fallback otherwise): deadlines the optimistic Gantt promised
   are now MISSED (0 → 4). This is a calibrated model,
   not a direct execution trace. Panel tinted as a board-calibrated round.
4. **re-schedule on board-calibrated costs** — CP-SAT re-solves knowing the true costs: deadlines RECOVERED
   (4 → 0). This is the loop closing: adjust after runtime feedback, before any further optimization.

Idle stretches are compressed into grey break columns (real-ms x-axis kept); misses are ringed red.
- **Generate the sequence** (`XPURT_CPSAT_WORKERS=0`; CP-SAT solves for og/AOT/fix + a board re-cost):
  `XPURT_PY=$(command -v python) python scripts/gen_schedule_evolution.py`
  → writes the 4 panel schedules + `panels.json` under `results/codesign_feedback/sensor_evo/`.
  (The board re-cost is `scripts/recost_schedule_on_board.py` — re-times a fixed schedule using measured
  per-dispatch/per-op calibration where available and an aggregate fallback otherwise; the software twin of
  "how the run differs from the Gantt".)
- **Render**:
  `.venv/bin/python scripts/compose_schedule_evolution.py \
     --spec data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json \
     --panels-json results/codesign_feedback/sensor_evo/panels.json --layout grid`
  → `results/codesign_feedback/schedule_evolution_mega.{png,pdf}` plus a metrics sidecar. The default grid is
  a 2×2 four-stage layout authored at 7.16×4.35 in for the top of a two-column page; `--layout vertical`
  preserves the legacy single-column stack. All four grid panels share one time scale so the makespan change
  remains visually comparable.
- **Inputs**: the sensor workload spec `_4w_networks_k1_sensor_sharded_rich_shard_ime_s4.0.json` + its profiles
  under `gen/mb/…`; `k1_board_calibration.json` (drives both the re-cost and the board-calibrated re-solve).

## 2. `hil_ablation_phase` — HIL command-rate phase diagram (GPU for the grid)
- **Generate** the per-flight grid (real Isaac flights; 5 speeds × 4 rates × 6 seeds = 120 flights, GPU-hours).
  Needs the conda `env_isaaclab` python — pass it via `ISAAC_PY` (see `docs/Artifact/REPRODUCE.md` §1):
  `ISAAC_PY=<env_isaaclab>/bin/python bash scripts/hil_ablation_grid.sh`
  → appends rows to `results/codesign_feedback/hil_grid/hil_ablation.csv` (via
  `sims/scripts/sweep_rate_demo.py --sweep-csv`; one row/episode:
  `seed,cruise_speed,sim_dt,decimation,control_dt_ms,sched_latency_ms,hold_steps,eff_cmd_hz,…,outcome`).
  Overridable env: `XPURT_REPO`, `HIL_OUTDIR`, `HIL_WEIGHTS`. The 120-flight CSV is committed at
  `results/codesign_feedback/hil_ablation.csv` so the diagram re-renders without the GPU sweep.
- **Render**:
  `.venv/bin/python scripts/hil_ablation_phase.py --csv results/codesign_feedback/hil_ablation.csv`
  → `results/codesign_feedback/hil_ablation_phase.{png,pdf}`. **Two panels sharing the command-rate axis**:
  LEFT ties each runtime's control-loop cadence to the rate it sustains (rate = 1000/gap — ROS under its
  default executor 24.94 ms→33 Hz, XPU-RT 6.30→100 Hz), against the shrinking loop budget; RIGHT is the
  flight phase diagram (a smooth safe→crash surface over speed×rate, crash frontier drawn, raw 6-seed
  cells) with those rates carried across. Honest: >100 Hz is hatched "not flight-tested" (no extrapolated
  surface). The cadences come from `scripts/measured_timing.py`, which holds the board measurements and
  their provenance and re-derives them under `--verify`; the script imports them rather than carrying its
  own copy. Replaces the old `hil_ablation_scatter` (kept for reference).

## 3–4. Warehouse figures — combined showdown + the two mega plots (GPU to regen flights)
- **Flight data** — TWO dumps: the XPU-RT successful weave and the ROS crash:
  `<env_isaaclab>/python sims/scripts/record_sensor_demo.py --headless --controller rl \
     --weights sims/models/warehouse/nav_fused_v12_cnn.pt --sched_latency_ms 24.94 --decimation 1 \
     --prop_density 0.35 --obstacle_level 8 --fixed_speed 1.2 --episodes 4 --seed 2000 \
     --dump_figure_data <xpu-dir>` (+ once with `--clean_overview --clean_out <xpu-dir>` for `clean_bg.npz`);
  the ROS crash dump is the same command with the ROS-rate latency (crashes ~y=10, past gate 1) → `<ros-dir>`.
- **`warehouse_showdown_board`** (combined overview) and **`warehouse_schedule_board`** (publication-sized
  schedule panel): both selected paths on one top-down aisle, 2 fixed-baseline + 2 XPU snapshots,
  IMU/goal/speed/velocity, and a separate matched-horizon schedule-model comparison:
  `<env_isaaclab>/python sims/scripts/compose_warehouse_showdown.py \
     --xpu-dir results/codesign_feedback/crash_demo/complete_figdata \
     --ros-dir results/codesign_feedback/crash_demo/crash_figdata --rot 0 \
     --out results/codesign_feedback/warehouse_showdown_board \
     --schedule-out results/codesign_feedback/warehouse_schedule_board`
  → both `.png` and `.pdf` variants plus metrics sidecars. The defaults are the matched 5/6/12-instance
  board-model schedules `scheduled__flight_deployed_matched_board_cpsat_profiled.json` (XPU-RT) and
  `scheduled_ros_partition_deployed_matched_board.json` (ROS-style fixed partition), comparing interior
  YOLO frame 3 against the adapted 23 ms period/deadline. Trace provenance and selection rules are recorded
  in `results/codesign_feedback/crash_demo/trace_manifest.json`. Use the standalone schedule panel in the
  main paper; treat the dense combined overview as a supplement/full-page asset.
- **`mega_warehouse_xpurt` / `mega_warehouse_ros`** (the per-scheduler mega plots) — Gantt strip via
  `scripts/plot_solver_gantt_annotated.py --sched schedules/scheduled__flight_deployed_2frame_cpsat_profiled.json`
  (the CP-SAT deployed schedule: 40.4 ms 0-miss, balanced), then
  `<env_isaaclab>/python sims/scripts/compose_mega_figure.py --data-dir <xpu-dir> --gantt <gantt.png>
   [--crash-step 779] --out results/codesign_feedback/mega_warehouse_{xpurt,ros}`.
- **Notes**: people are projected at their real height (z≈0.85, not 2.0); moment markers are placed at the
  actual gate crossings; IMU is smoothed. `make_all_codesign_figures.sh §5` drives all three (set
  `WAREHOUSE_FIGDATA`/`WAREHOUSE_ROS_FIGDATA`).

## 3b. Warehouse showdown, second form — vanilla ROS 2 against the solvers, both on the K1

The complete recipe behind this section (environments, board staging, solves, cadence traces, the simulator
campaigns and their queues, the guidance-net training, the feedback study, aggregation and verification) is
`docs/Artifact/reproduction_full.md`.

Every number is measured on the board or replayed from a board measurement; the scripts are the recipe.

1. **Board, ROS 2 as one writes it** — `scripts/board_vanilla.sh` (arms `vanilla`, `rvanilla`), `board_vanilla2.sh`
   (`vanilla4`, `vanilla4t`), `board_rp3_90.sh` (`rvanilla4`): one process per node, unpinned, default executor and
   QoS, control in the goal callback (or on a timer, `vanilla4t`); camera rates 15–90 Hz, three replicates; pulled by
   `scripts/pull_ros_traced.py` into `ros_traced/<hz>_<arm>_r<k>/` and `ros_traced/summary.csv`.
2. **Board, the solvers on one spec** — specs `data/toplevel/wh_chain45_solve*.json` (45 Hz chain),
   `wh_chain90_rich_solve_500.json` (90 Hz + heavier stack), `wh_chain{30,60,90}_solve_500.json`,
   `wh_chain45_shard_solve*.json` (YOLO free to shard, costed from `gen/mb_cal`, built by
   `scripts/calibrate_yolo_shard_profile.py`). Certificates on the hyperperiod: `scripts/solve_certificates.sh`
   (`--solver cpsat`, hard windows, must be feasible; `greedy_periodic` misses counted). Tables:
   `scripts/solve_stage2_hard.sh <spec> <tag>` → `schedules/fig_<tag>_{cpsat_hard,greedy}.json`; then
   `scripts/board_stage2.sh <tag> <spec>` (clamp widths, feasibility check, three interleaved runs each) →
   `xpurt_long/trace_<tag><solver>r<k>_other_run1.csv`. Read with `scripts/xpurt_trace_report.py --windows ...`.
   Chains: `scripts/chain_rates2.sh`, `scripts/chain_shard_solve.sh`; costing sensitivity `scripts/solve_certificates_sensitivity.sh`.
3. **Cadence traces** — `scripts/ctrl_trace_from_board.py <trace.csv | ctrl_gaps.csv> --out ctrl_traces/<arm>.csv`:
   the control-output times of one board run, looped over the table from warm-up to its end.
4. **Flights** — `scripts/campaign_v2.sh` with `ARMS="name:trace ..."` and `SPEEDS=...` (`campaign_ros_family2.sh`,
   the solver arms, `campaign_courseB_v2.sh`): `sweep_rate_demo.py --ctrl_trace`, 12 seeds per cell, one simulator at a
   time, GPU-room gated → `campaign_v2/campaign_v2.csv`. Selection: `scripts/campaign_select_v2.py --xpu <trace> --ros <trace>`
   (fastest cruise where XPU-RT completes ≥ 3/12 and the baseline enters then crashes in ≥ half). Mechanism:
   `scripts/run_energy_v2.sh` → `flight_energy_v2.csv`. Envelope 20 Hz cell: `scripts/envelope_20hz.sh`.
5. **Displayed flights and videos** — `scripts/dump_display_serial.sh` / `dump_display_ros_attempts.sh` (one simulator
   at a time; the baseline re-flown over its mid-course-crash seeds until the rule's flight reproduces; attempts logged),
   `scripts/record_display_videos.sh` (`--ctrl_trace`, `--gantt_schedule schedules/measured_gantt_<arm>.json`).
4b. **Environment sweep** — `scripts/env_sweep.sh <id> "<arm:trace ...>"` (one simulator per driver, GPU-room
   gated): gate course a/b × obstacle density 0.20/0.30/0.40 × cruise 1.0–1.8 × the replayed cadences, 12 seeds a
   cell, people 2.4 m (`mdp_obstacles.PERSON_H`, so they must be avoided); every flight recorded by
   `sweep_rate_demo.py --record_dir` (path, commanded wrench, body rates, goal command, gates, outcome, settings)
   → `campaign_env/env_sweep.csv` + `campaign_env/records/<cell>/ep<k>_s<seed>.npz`. `scripts/env_sweep_summary.py`
   aggregates success/gates per cell and, through the rotor model, mean propulsive power and commanded moment →
   `campaign_env/env_sweep_summary.csv`, `refined/env_sweep.png`. One scene for both displayed flights:
   `--layout_seed` (props and people from a fixed seed) and `scripts/display_same_env.sh`.
6. **Render** — `SPEC=data/toplevel/wh_chain45_solve.json XPU_ARM=acpsat_hardr1 XPU_SCHED=schedules/fig_a_cpsat_hard_clamped.json
   XPU2_ARM=agreedyr1 XPU2_SCHED=schedules/fig_a_greedy_clamped.json ROS_TAG=45_vanilla4_r1 LABEL_ROS="ROS 2 vanilla"
   XPU_DIR=<xpu dump> ROS_DIR=<ros dump> scripts/render_showdown_measured.sh <out>`; the layers figure
   `scripts/plot_deployment_layers.py`; the throughput map `scripts/plot_throughput_latency.py`;
   `ENERGY_CSV=results/codesign_feedback/flight_energy_v2.csv scripts/hil_story_figure.py`;
   `scripts/verify_showdown_figure.py --xpu-arm ... --ros-tag ... --xpu-dir ... --ros-dir ...` must print 0 FAIL;
   `scripts/measured_timing.py --verify` must print 0 drift.
7. **Render, third form** — `refined/warehouse_showdown_v3[_<cell>_<tuned>].{png,pdf,_metrics.json}` and the
   column-width companion `hil_envelope_story_v3[...]`: `CELL=tall1005|tall1000|cal17 TUNED=board|inb|none|all
   scripts/render_showdown_v3.sh` (the full environment contract — `PAPER`, `MAIN`, `MAIN_CELL`, `DPI`, `PY`, `XPU_DIR`,
   `ROS_DIR`, `SCENE_RECORDS`, `DISPLAY_CRUISE`, `ENERGY_CSV` — is in `docs/Artifact/reproduction_full.md` §8 step viii; `MAIN=1`
   also copies the cell's `board` variant to the unsuffixed name; `cal17` has scene runs but no display dump on disk). Panels:
   A the same-scene pair (backdrop = per-pixel median of the dump's overhead sequence `ov_seq[t ≥ 2 s]`, so props stay
   and the drone and walkers vanish), a–d strips, B success against cruise per replayed arm (Wilson 95 %,
   `campaign_percep` + `campaign_seeds24` pooled), C gates cleared before the collision at the display speed,
   D′ the environment map (`campaign_*` latency-replayed cells pooled over 1.0–1.4 m/s), D mechanism, E body rate,
   F the camera-rate envelope on the K1 (XPU-RT from `xpurt_long/trace_a{30,,60,90,120h}{cpsat_hard,greedy}r*`, frames
   late by the spec window; ROS 2 from `ros_traced/summary.csv`, latency `e2e_goal_med_ms`, delivered
   `n_consumed/(seconds − warm-up)`), G per-hart share of executed inference time from the traces (`worker_hart`; a
   ROS 2 YOLO callback credited to its node's `pool_harts` ∪ its hart), H the added-load bars + `campaign_rich`
   flights, J the feedback drift (`hil_feedback_closeup.load`), K every recorded run of the display scene
   (`scripts/scene_runs.sh` → `campaign_scene/<cell>/<arm>/ep*.npz`, obstacles from the record), I the measured Gantt
   rows (`make_measured_gantt_pair.py --arm xpu|xpu2|ros8|ros …`, 100 ms window; `ros8` = `45_vanilla4x2`, the two-YOLO-node
   graph on all eight cores). `verify_showdown_figure.py --v3-metrics <sidecar>` re-derives every panel's numbers and
   must print 0 FAIL; `emit_figure_numbers.py` turns the sidecar into caption macros. Simulator work for the cells:
   `scripts/queue_v3.sh` (display pairs with `display_same_env.sh ROS_GATES=2`, scene runs, the eight-core ROS 2
   flight arms `ros_vanilla4x2[_90]`, `ros_multi`), then in order `queue_v3c.sh` (the baseline-first pair search,
   `display_pair_search.sh`, at 1.2 then 1.4 m/s, scene runs and render of the pair found), `queue_v3d.sh` (the
   eight-core flight arms and the final renders; prints QUEUE_V3_DONE), `queue_v3b.sh` (the baseline's and greedy's
   replicates so every arm carries three flights per seed, then re-renders), `queue_v3e.sh` (a two-gate pair search at
   1.2 m/s over seeds 1012–1023) and `queue_v3f.sh` (six more XPU-RT attempts on each two-gate seed). Each queue
   waits on the previous one's marker in its log and gates the simulator on GPU room. `PAPER=1` renders the **paper form** instead
   (`refined/warehouse_showdown_v3_paper[_<cell>_<tuned>]`): the paper-form skeleton — top-down with the
   envelope / generalisation / mechanism column beside it, the four strips, a four-panel row (body rate, camera-rate
   envelope, where the work lands, added load) and the Gantt — with every slot's content measured.
8. **Story figures from the same data** — `scripts/story_figures.py [names]` → `refined/<name>.{png,pdf,_metrics.json}`:
   `crash_position` (where every flight ends along the aisle, per arm and cruise, from the records),
   `course_progress` (course fraction covered, mean and 95 % bootstrap band, against cruise), `rate_speed_map`
   (camera rate × cruise, courses completed per runtime arm), `ros_ladder` (the hand-tuning ladder on the board at
   45 and 90 Hz and in flight), `seed_pairs` (the same seed under both runtimes, gates cleared side by side),
   `latency_waterfall` (camera→control split into queue wait / YOLO / nav / control per arm from the traces).
   All flight counts are one flight per seed (the first recorded); each sidecar carries the numbers drawn.
9. **The atlas** — `scripts/showdown_atlas.py` → `refined/warehouse_showdown_atlas.{png,pdf,_metrics.json}`: the
   census of every campaign and board-run family (A), the same-scene pair (B), the camera-rate envelope (C), the
   hand-tuning ladder at 45/90 Hz (D), the latency waterfall (E), the ablation forest — Δ courses completed
   XPU-RT − ROS 2 with Newcombe 95 % intervals, one row per condition plus the pooled row (F), paired seeds per
   speed (G), course fraction against cruise (H), CP-SAT against greedy per camera rate (I), the added load (J).
   Every panel reads the CSVs/records/traces at render time; counts are one flight per seed.
10. **The final figure** — `scripts/showdown_final_figure.py` → `refined/warehouse_showdown_final.{png,pdf,_metrics.json}`,
   in reading order: S the two runtimes on the board as one left-to-right flow (how each is written → where it places
   work → what the board measures → the replay into the simulator); A the specific case, B the control-rate envelope
   of the paper figure (rate injected, colour = cruise, `hil_envelope_panel.draw_envelope`) with the two measured
   control rates marked on it, C generalisation to the unseen course (`hil_story_figure.draw_generalization`); the
   moments a–d; D–G the pair inside (body rate, nav goal heading, forward speed, XPU-RT's velocity through the gates —
   the paper figure's telemetry row); H / H′ two scenes each flown 36 times (the display scene at 1.0 m/s and the
   two-gate scene at 1.4 m/s: every success and collision of both arms over the obstacles; `scripts/scene_runs.sh`);
   I success against cruise, J paired seeds, K where every flight ends, L mechanism; M the ablation forest (short
   condition labels, `showdown_atlas.SHORT`), N camera rate on the K1, O the latency waterfall, P added load, Q where
   the work lands; R the four measured Gantt rows. Panel functions come from `showdown_v3_figure.py`,
   `showdown_atlas.py`, `story_figures.py`, `hil_envelope_panel.py` and `hil_story_figure.py`; imported panels are
   authored at their own size and brought to the figure's scale by `shrink()`; the sidecar carries every number.
   Iterate at `--dpi 60` (≈10 s) and inspect crops of each row before the 300 dpi render.

11. **Figure 10 for the paper, sized to its slot** — `scripts/showdown_paper10_figure.py` →
   `refined/warehouse_showdown_paper10.{pdf,png,_metrics.json}`, 7.0 in × 5.1 in (the full text width; the height grew
   from the earlier 3.6 in by one row of results; `--width-in/--height-in` to change). Type is set to the paper's
   (caption 8 pt, body 9 pt): `--base-pt 6.0` ticks, titles 6.6 pt bold, nothing under 5 pt. Eight panels and the four
   moments, drawn by the same functions as the final figure: row 1 — A the specific case at the plate's true 3.3:1 aspect
   across the left half, B the breaking point (1.7 m scene; CP-SAT, greedy, ROS 2 vanilla, hand-pinned), C camera rate on
   the K1 (colours as E's arm labels); a–d chase + FPV at the moments across the full width; row 2 — S the second scene
   (seed 1000, 1.4 m/s) with every one of its 36 flights over the crates, racks and people, the pair bold (ROS 2 clears
   G1 and G2 and hits a gate frame), F paired seeds (who gets further), G body rate and nav goal heading inside the pair
   of A; row 3 — D the four measured schedule rows across the width with each arm, chain latency and control period
   labelled in its row, E the latency waterfall. Row heights are given in inches in the script (`top_in … bot_in`).
   `--audit` re-reads every drawn text after layout and prints those that leave the page, fall under 5 pt or collide
   with another panel's text or axes box — the acceptance test of the render loop (exit code 3 on findings; the list is
   also written to the sidecar). Still only in the final figure and the atlas: envelope, generalisation, speed/quiver
   telemetry, the 1.0 m/s scene runs, 2.4 m success curve, forest, added load, hart map.
   Include with `\includegraphics[width=\textwidth]{warehouse_showdown_paper10.pdf}`; text is embedded (DejaVu, Type 42).

11b. **The paper's figure in its paper layout, every number measured** — `scripts/showdown_paper_figure.py` →
   `refined/showdown_45hz_pinned_vs_rosdefault_s1003.{pdf,png,_metrics.json}` (20 in wide, `--dpi 300`). The same skeleton as the paper figure
   (A top-down pair with the same scene's completed/flown per arm in the legend · B the rate-injected
   control-rate envelope with the two arms' measured control rates marked · C the same envelope on course B · D mechanism ·
   a–d chase, FPV + YOLO, cross-ToF · E–H the pair's telemetry · I the measured K1 schedule, one XPU-RT · CP-SAT row and
   one ROS 2 vanilla row from `schedules/measured_gantt_v3_{xpu,ros}.json`). Titles and legend text are built from the
   dumps, the sidecars and the registry; `verify_showdown_figure.py --metrics …_paper_metrics.json` re-counts the
   envelopes from `hil_ablation*.csv`, the scene from `campaign_scene/tall1005s/campaign.csv` and the Gantt rows.
   Two display cells are kept. The default is the same-gain pair (tall seed 1005 at 1.0 m/s; that scene completes 1 of
   12 against the baseline's 0 of 12). `warehouse_showdown_paper_cal17` is the 1.7 m scene, seed 1010 at 1.2 m/s, with
   each arm at the gain its own control rate calls for (0.0052 against 0.0128): the scene completes 10 of 12 against 1
   of 12 and the baseline clears a gate before it hits a crate, the shape the paper figure shows. Render it with
   `--xpu-dir/--ros-dir campaign_v2/display_same_cal17/{xpu,ros}_s1010_figdata --scene-records campaign_scene/cal17
   --display-cruise 1.2`; panel A's title states which gain policy the pair flew and the verifier holds the dumps to it.
   `--gantt-merge` joins each hart's consecutive dispatches of one frame into a single bar and keeps every idle gap, so
   both rows carry the same unit of work as a ROS 2 callback and the bars still agree with the measured busy per hart
   (545 dispatches become 409 bars in the 100 ms window; the baseline's 12 callbacks are unchanged).
   A ROS 2 perception callback whose kernel runs on a worker pool is recorded on one hart: the callback thread's. The
   pool's helpers run the kernel's shards on the other pool harts over the same interval — the callback takes the
   four-hart standalone time and the sampler shows those harts busy — but the trace does not record them. Each
   dispatch therefore carries `traced_target`, and the Gantt draws the traced hart solid and the pool's other lanes
   hatched at low opacity, with a legend key, so a measured placement is distinguishable from a credited one. XPU-RT's
   bars are one measured dispatch each, on the hart it ran on.
   `scripts/compose_pair_video.sh <xpu.mp4> <ros.mp4> <out.mp4> [left banner] [right banner]` builds the side-by-side
   video of a cell's pair and its half-size copy; the half-size copies are tracked, the full-resolution ones are
   regenerated from the display clips.

11c. **The 36 Hz family — the control-rate floor against a baseline that uses the whole machine** —
   `scripts/render_allcores36.sh` and `scripts/render_navpool36.sh` →
   `refined/warehouse_showdown_cam36_{allcores_s1003,allcores_s1007,allcores_s1009,navpool_s1006}`.
   Same `showdown_paper_figure.py` skeleton as 11b; what differs is the baseline and where its Gantt
   row comes from. The baseline runs two YOLO pools over all eight harts (`vanilla4x2`) and, in the
   `navpool` form, a four-way pool under navigation as well (`vanilla4x2ns4c`), so it meets its
   deadline and leaves no hart idle. Gantt rows are built by `make_measured_gantt_pair.py` into
   `refined/{allcores36,navshard36}/` against `data/toplevel/wh_chain36_free.json`, so the 27.78 ms
   perception period is the one drawn. **Full recipe, every arm, every seed's outcome and the
   caveats for a caption: `docs/Evaluation/showdown_cam36_allcores_reproduction.md`.**

   Two things the earlier forms did differently. *Both* YOLO pools now record their per-shard
   slices, so a pooled dispatch is drawn on the harts that actually ran it rather than credited to
   the callback thread with the other lanes hatched (11b above describes the hatched form, which is
   still what an arm with one traced pool gets); a lane drawn from measured slices carries
   `lanes_measured` and `showdown_gatecourse.py` suppresses the hatch for it. And the slices of one
   dispatch are coalesced into a single bar spanning its lanes, so a frame reads as one four-lane
   block instead of a few hundred sub-millisecond slivers.

   The Gantt comes from the instrumented replicate (`36_vanilla4x2d2_r1`) and the flights replay the
   un-instrumented one (`36_vanilla4x2_r1`): recording the second pool leaves the mean cadence
   unchanged, 27.77 ms either way, and raises p95 from 31.4 to 36.2 ms. The arm is flown as deployed
   and only the picture comes from the instrumented run.

   `scripts/render_gantt_compare.py --rows <path>:<label>:<xpu|ros> …` draws any set of those rows on
   their own, which is how the three-arm comparison
   `refined/gantt_navpool_36hz_v2.{png,pdf}` is made.

12. **Verification, after any render** — `. scripts/env.sh` first (it names the interpreters and the results tree), then:
   `$HOST_PY scripts/verify_showdown_figure.py --all` re-derives every `refined/*_metrics.json` sidecar (v3 in every cell
   and placement, the atlas, the final figure, paper10 and the six story figures) from the campaign CSVs, records, traces
   and board manifests under the same counting rules the scripts use, checks that display literals come from the registry
   `scripts/figure_constants.py`, that `fallbacks_used` is empty, that each sidecar's `inputs` hashes match the files on
   disk, that every Gantt row's executed table matches the ledger `results/codesign_feedback/executed_tables.json`, and
   that every png/pdf in `refined/` has a sidecar or is on the script's allowlist (`REFINED_ALLOWLIST`); `--metrics <sidecar>`
   checks one figure. `$HOST_PY scripts/measured_timing.py --verify` re-derives every board constant and the registry
   (`figure_constants.check_registry`). `$HOST_PY scripts/executed_tables.py` checks the ledger against the schedule files
   (`--write` rebuilds it from the manifests). `$HOST_PY -m pytest tests xpu-rt/tests -q` pins the counting rules
   (`tests/test_showdown_figures.py`). All four must report no failure before a figure is committed.

## 5. `solver_win_sensor` — CP-SAT beats greedy (contended sensor workload)
`.venv/bin/python scripts/compose_solver_win.py \
   --greedy schedules/scheduled__4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0_greedy_profiled.json \
   --cpsat  schedules/scheduled__4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0_cpsat_profiled.json \
   --spec   data/toplevel/_4w_networks_k1_sensor_sharded_rich_shard_ime_s5.0.json --window 32 \
   --out results/codesign_feedback/solver_win_sensor`
(greedy 4 misses / INFEASIBLE vs CP-SAT 0 misses / PROVEN OPTIMAL, 27.85 vs 30.64 ms).

## 6. Supporting Gantts / comparison
- `gantt_annotated_{cpsat,ros}` — `scripts/plot_solver_gantt_annotated.py --sched <schedule> --window-ms 44
  [--crash-note --yolo-deadline 22]` (ROS variant shows serial-YOLO backlog → crash).
- `warehouse_crash_speed` / `warehouse_solver_flight` / `solver_comparison_*` — scratchpad builders reading
  `schedules/*_metrics.json` + the Isaac sweep CSV.

## 7. `loop_ablation` — which feedback loop does the work (and does the exact solver earn its cost)

**Story.** The channel closes at two distances, and one showcase run cannot separate
them. Four cells — **A** neither, **B** inner only, **C** outer only, **D** both — with a
bar per solver arm in each. Read across the cells for which loop does the work; read
within a cell for whether CP-SAT beats greedy. *Inner* = AOT co-design with ModelBlaster
(graph rewrites + per-dispatch implementation choice, decided against isolated profiles;
the board appears only as a **profiler**). *Outer* = HIL (measured multipliers from
real-time runs of the whole schedule, returned and re-solved against; the board is the
**runtime**). HERO metric = network-instance deadline misses.

**Generate** (CPU only; no board needed — the calibration is a committed artifact).
The current population is the **2→5 network ladder**, not the K1 corpus:

```bash
export XPURT_NO_COMPACT=1                 # the script sets it too, and records it
$PY scripts/ablate_feedback_loops.py \
    --workloads data/toplevel/scaling/w2_ffn_tight.json \
                data/toplevel/scaling/w3_ffn_dronet.json \
                data/toplevel/scaling/w4_ffn_dronet_sensor.json \
                data/toplevel/scaling/w5_ffn_dronet_yolo.json \
    --solvers cpsat,greedy --time-limit 90 --cpsat-time-limit 300 --repeats 3 \
    --max-rounds 3 --out-dir results/loop_ablation_ladder_v2
```

**Why the ladder and not the 25 K1 specs.** That corpus is bimodal and answers a
different question: 12 of 25 have a baseline that misses no deadline at all (nothing at
stake), and most of the rest ask for something no schedule can deliver — 44 of the 50
residual misses in the corpus-wide run were **infeasible by construction**, because
`yolov8_nano_64x96` needs 23.95 ms at 8 cores against a 22 ms window and its core scaling
has saturated. `scripts/make_scaling_workloads.py --check` builds the ladder so that every
rung's singleton baseline MISSES and a measured wider implementation FITS, with **zero
infeasible misses** — the band where a scheduler decides the outcome. The corpus-wide
command (`--workloads $($PY scripts/loop_over_workloads.py --list-k1) --one-per-family`)
still works and is the right thing for a generality claim, with the family de-duplication
that claim requires.

Writes `<out-dir>/ablation_summary.json` (+ `ablation.log`). Greedy over the four rungs is
~4 min; a CP-SAT arm is budget-bound — its cost is set by `--cpsat-time-limit`, not by the
instance — so expect a couple of hours and run it `nice`d.

**Render:**

```bash
$PY scripts/plot_loop_ablation.py \
    --summary results/loop_ablation_ladder_v2/ablation_summary.json \
    --out-dir results/codesign_feedback --stem loop_ablation
```

Outputs `results/codesign_feedback/loop_ablation.{png,pdf}` plus a metrics sidecar.
Caption numbers come from the sidecar via `scripts/emit_figure_numbers.py --prefix abl…`
(use a per-arm prefix — `ORDINALS` holds 8 entries and cell×arm overflows it). The
renderer reads the summary only and never re-solves.

**Inputs.** `data/toplevel/*.json` (the K1 specs), the profiles under `gen/profile_mb/`,
and `results/codesign_feedback/k1_board_calibration.json`.

**Honesty notes — read these before quoting the figure.**
- **The population is families, not files.** Two specs are byte-identical and several
  variants return bit-identical results; `--one-per-family` is the honest unit. The
  family table is in the summary's `population` block.
- **Only the at-stake stratum is plotted.** A workload whose naive deployment already
  meets every deadline cannot show a loop helping. The stratum (`cell A misses ≥ 1`) is
  pre-registered in code, and the count of excluded workloads is printed on the figure.
- **Every cell is scored on board costs.** A/B are solved on predicted costs and re-cost
  with their assignment fixed; C/D are solved with `--board-calibration` — re-costing
  those would apply the multiplier twice.
- **CP-SAT's result is not reproducible; the experiment is.** Repeats are run only where
  the solver did not prove optimality, and reported as median with the spread.
- **MOSEK is absent on purpose**: it exhausted ~89 GiB on the rich workload and no memory
  guard exists in code.
- **Every cell obeys the codegen contract, and the two arms obey it differently.** A
  schedule that gives a packed-weight (convolution) dispatch different core widths in
  different periodic instances cannot be code-generated, so it is not a result. CP-SAT is
  CONSTRAINED (`XPURT_UNIFORM_PACKED_WIDTH=1`, set automatically for shard-mode solves):
  it still chooses the width and must choose one. Greedy has no combination-selection
  variable to couple, so its unbuildable candidates are REJECTED by the inner search
  instead. That asymmetry disadvantages greedy on exactly the workloads where sharding is
  the answer, it is recorded in the summary's `codegen_contract` field, and it must be
  stated wherever the two arms are compared. See `docs/Feature/board_and_model_gaps.md` gap 3.
- **A number this gate already retracted.** Before it existed, `shard` was credited with
  11 → 5 instance misses on `w5_ffn_dronet_yolo` using a schedule the compiler refuses.
  Any older table carrying that halving is carrying a number that was never deployable.

---
**Regenerate all** (from cached artifacts where possible): `bash scripts/make_all_codesign_figures.sh`.

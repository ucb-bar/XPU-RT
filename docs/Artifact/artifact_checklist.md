# Artifact checklist — what is stored, what is generated, and how to check both

Every number in the warehouse showdown comes from one of three places: the K1 board, the flight
simulator, or a solver. This lists, for each, what the repository holds, what a reviewer regenerates,
and the command that checks the two agree. Nothing here needs this machine except where it says
"needs the board" or "needs a GPU".

## 1. What is stored, and why it cannot be regenerated

| what | where | why stored rather than generated |
|---|---|---|
| ROS 2 baseline node + samplers + the traced worker pool | `board/k1_ros_mb/` (+ `MANIFEST.sha256`) | the arms are flags into this one program; it lived only on the board |
| board traces, per-core samples, manifests | `results/codesign_feedback/{ros_traced,xpurt_long}/` | a measurement of one run of one binary on one board |
| cadence traces replayed by the flights | `results/codesign_feedback/ctrl_traces/` | cut from the board traces, one command, kept so flights are reproducible without a board |
| flight outcomes | `results/codesign_feedback/campaign*/campaign*.csv` | Isaac flights are not bit-reproducible across machines |
| faulted-batch quarantine | `results/codesign_feedback/flight_quarantine.csv` | a judgement about data, with the log line as evidence |
| measured IME-vs-RVV conv table | `ModelBlaster/artifacts/ime_conv*/` | board measurement the kernel picker reads |

Display dumps (`*_figdata/figure_data.npz`, 78-314 MB each) are `.gitignore`d as before; the figure's
sidecar records their sha256, so a render can still be tied to the exact dumps it read. The paper
figure's pair is archived beside the older ones as
`results/codesign_feedback/archive_v3/display_dumps_r36.tar` (391 MB, listed in that directory's
`MANIFEST.sha256`, which is tracked -- the tars themselves are not). Without that archive panels A,
a-d and the telemetry row cannot be re-rendered at all; with it, unpack into
`results/codesign_feedback/campaign_rate3640/display36/search_c1.2/`.

The executed schedules the verifier checks (`schedules/fig_w2pg36_greedy_clamped.json` and the other
tables a figure names) are force-added past the blanket `/schedules/*` ignore for the same reason.

## 2. What is generated

| what | command | needs |
|---|---|---|
| per-network kernels + skeleton (the deployed build) | `docs/Baselines/ros_baseline_reproduction.md` §1 | cross toolchain |
| the sharded YOLO build and its standalone timing | `scripts/board_campaign.sh` step 1 | board |
| the ROS node binaries on a board | `scripts/board_deploy_ros_node.sh` | board |
| ROS rungs across the camera sweep | `scripts/board_rate_sweep_x2.sh` | board |
| ROS 2 out of the box (`vanilla_c50`, 50 Hz control timer) | `scripts/board_ctrl50_arms.sh` | board |
| XPU-RT's arm at any camera rate | `scripts/xpu_greedy_shard_at_rate.sh <hz>` | board |
| XPU-RT's CP-SAT arm with board feedback | `scripts/rate_feedback_loop.sh <hz>` | board + ~3 h CPU |
| flight censuses | `scripts/campaign_rate30.sh`, `scripts/campaign_rate3640.sh` | GPU |
| the displayed pair | `scripts/display_pairs_rate36.sh`, `scripts/display_search_rate36.sh` | GPU |
| panel A's scene census | `scripts/scene_runs_pair.sh` | GPU |
| panel D's energy runs | `scripts/run_energy_pair.sh` | GPU |
| panel I's Gantt | `scripts/make_measured_gantt_pair.py` | — |
| the figure | `scripts/showdown_paper_figure.py` (see `docs/Evaluation/showdown_rate_sweep_reproduction.md` §6) | — |
| the effort-ladder table | `scripts/ros_effort_ladder.py` | — |
| the IME kernel, its measurements and builds | `docs/K1/ime_kernel_reproduction.md` | board + cross toolchain |

## 3. The checks

Run from the repository root; none needs the GPU.

```bash
.venv/bin/python -m pytest tests xpu-rt/tests -q          # the unit and property tests (8 skipped need hardware)
.venv/bin/python scripts/measured_timing.py --verify      # every timing constant re-derived from the board traces
.venv/bin/python scripts/flight_quarantine.py             # every quarantined batch still matches its evidence
.venv/bin/python -c "import sys;sys.path.insert(0,'scripts');import figure_constants as F;print(F.check_registry() or 'clean')"
.venv/bin/python scripts/verify_showdown_figure.py --metrics \
    results/codesign_feedback/refined/warehouse_showdown_paper_r36_metrics.json   # 0 FAIL
scripts/board_source_snapshot.sh --verify                 # needs the board: its sources match the repo
```

`measured_timing --verify` is the one that matters most: it re-reads the board traces and recomputes
every latency, control gap and late-frame count the figure prints, and reports DRIFT if any recorded
constant no longer follows from the data.

## 3b. Board traces: tracked, or archived with their bytes

Board traces are 3-12 MB each and there are 236 that a documented check opens, so `.gitignore` keeps
them out of the tree and `archive_v3/` holds them with the tars' sha256 in the tracked MANIFEST --
the arrangement the display dumps use. `scripts/verify_board_traces.py` is what makes that
arrangement mean something: it enumerates the traces `measured_timing.py --verify` globs for (one per
`SOLVER_ARMS` prefix, one per `ROS_VANILLA` arm and rate) and the trace each `measured_gantt_*`
sidecar names as its source, and requires every one to be tracked, or present in an archive with the
bytes that are on disk.

Without it the failure is silent rather than loud: `--verify` prints "no runs" for a constant whose
trace is missing and carries on, so a constant with no evidence behind it reads the same as one with
evidence. It is part of `artifact/verify_no_hardware.sh`.

## 3c. The ModelBlaster side: pinned, reachable, re-derivable

`scripts/verify_modelblaster_inputs.py` is the same guarantee for the half of the artifact that has
no figure sidecars. It checks three things, and each one has already caught something:

* **the pin is reachable.** The pinned commit is not on `origin` (§5), so what makes it recoverable is
  `artifact/history/modelblaster.bundle` — and only while the bundle's recorded tip is still the
  commit the superproject pins. Re-cut one without moving the other and this is what says so.
* **every submodule path a doc names exists at the PINNED commit**, not merely in a working tree.
  The paths are scraped from `docs/` and `artifact/` rather than listed -- both the ones written with
  a `ModelBlaster/` prefix and the ones a fenced block reaches after `cd ModelBlaster`, which it gets
  from `verify_doc_commands.py`'s working-directory pass rather than a second copy of the rule. So a
  doc that starts naming a new file is covered from the moment it does. That is how
  `docs/Evaluation/measurements_and_ablations.md`'s citation of `run_xpurt_k1.sh` was found: the pipeline table
  gave it at the submodule's top level, where it has never been; it lives under `scripts/`.
* **the IME numbers re-derive from the files behind them.** The measured tables `pipeline/ime_cost.py`
  names (read out of the pinned source, so a third conv op-kind is covered automatically) exist, carry
  a bit-exactness verdict on every row, and reproduce the per-shape win count the prose states; the
  two schedules `ime_mixed_schedule_36hz.json` names are tracked and carry the `impl: ime` dispatch
  count it records; the six board traces behind the two medians are tracked.

Its content checks need the submodule's objects. In a clean clone before the bundle is fetched they
report **SKIP**, with the count printed on the summary line — a skipped check is not a passed one.

## 4. What a reviewer can check without any hardware

The figure, its sidecar and the CSVs behind it are in the repository, so `verify_showdown_figure.py`
re-derives panel A's per-scene tally, panel B and C's k/n per control rate, panel D's energy ratios,
panel I's per-row latencies and hart placement, and checks that the XPU-RT arm's label names the
solver recorded in the board run behind the trace it replays. The effort-ladder table and the paired
census statistics regenerate from the CSVs. Everything else needs the K1 or a GPU, and each of those
has a script above.

## 4b. The current figure set

`artifact/verify_no_hardware.sh` verifies nineteen figure stems and each re-derives 0 FAIL. The four
newest are the 36 Hz family, which is where the control-rate claim is made against a baseline that
uses the whole machine:

| stem | baseline drawn | camera→control | seed | reproduction |
|---|---|---|---|---|
| `showdown_36hz_solver_vs_rosallhart_s1006` | two YOLO pools **and** a nav pool, all 8 harts | 37.4 ms | 1006, baseline at 2 gates | [`showdown_cam36_allcores_reproduction.md`](../Evaluation/showdown_cam36_allcores_reproduction.md) |
| `warehouse_showdown_cam36_allcores_s1003` | two YOLO pools, nav on one hart | 32.3 ms | 1003 | same |
| `warehouse_showdown_cam36_allcores_s1007` | same | 32.3 ms | 1007 | same |
| `warehouse_showdown_cam36_allcores_s1009` | same | 32.3 ms | 1009 | same |

Renders kept on disk **outside** the gate, each with its sidecar, because
`verify_showdown_figure.py:506` requires panel A's baseline to end having passed one or two gates and
these fall outside that window — the drone must enter the course and lose it before the third gate
for the panel to have anything to show:

| stem | why it is not in the gate |
|---|---|
| `warehouse_showdown_cam36_allcores_s1001` | the baseline reached 3 gates |
| `warehouse_showdown_cam30_allcores_*` (7 stems) | at 30 Hz the baseline hits a crate before G1 — 0 gates |
| `..._s1006` / `..._s1005` nav-pool forms named `_xpu_Nof4` | the scheduled arm did not complete the course on that seed |

A render the panel-A rule rejects is still a recorded flight. Naming it distinctly and keeping it is
how that is said; dropping it would leave the population invisible. The whole twelve-seed display
population for all three 36 Hz arms — 36 one-episode flights — is archived in
`archive_v3/display_dumps_allcores36.tar` and tabulated in the reproduction page.

## 4c. The five figures under audit

[`five_figure_audit.md`](five_figure_audit.md) is the panel-by-panel account of the audited set: what
each panel claims, which check re-derives it, what could not be rebuilt and why, and the findings the
audit produced. Two commands sit behind it:

```bash
bash scripts/render_audited_set.sh     # rebuild all five here, from their recorded inputs
bash scripts/repro_clean_clone.sh      # rebuild them in a fresh clone + the archives, and diff the
                                       # regenerated sidecars against the committed ones
```

The second is what tests reproducibility rather than consistency: the in-place checks run in a tree
where every input already happens to be present.

Two checks cannot pass inside a clone and are not defects: `verify_archived_dumps.py` and
`verify_board_traces.py` ask whether an untracked input is in an archive, and `archive_v3/` is
gitignored, so a clone has none of its own. `repro_clean_clone.sh` points both at the real archive
directory, which is what a reviewer has beside a clone. Two that *are* gaps: `pytest` cannot collect
`tests/test_ime_profile_from_picks.py` without the ModelBlaster submodule content, and
`flight_quarantine.py` cites campaign logs that are untracked and in no archive — the quarantine
decisions are committed, the log lines behind them are not.

`scripts/verify_doc_commands.py` checks the other half of a recipe: that every script a document
tells a reader to run is at the path it gives. A citation is resolved against **the directory the
command actually runs in**, not the repository root: a fenced block carries its own working
directory, set by a literal `cd` and reset at the fence. Resolving everything against the root is how
a command that cannot run reads as fine -- `cd ModelBlaster` followed by
`ModelBlaster/scripts/ime_fused_conv_bench.py` names a path that exists from the root and resolves to
`ModelBlaster/ModelBlaster/...` where the reader is standing. Five such lines were found that way, in
`ime_kernel_reproduction.md` (both benches), `ros_with_ime.md`, `ros_baseline_reproduction.md` and
`nav_sharding.md` (the kernel-coverage check).

233 of 244 cited (path, working directory) pairs resolve. The eleven that do not are nine distinct
scripts -- two are cited from two different working directories and so appear twice -- and all are in
peripheral pages: Firesim, the Demo/Qualcomm walkthroughs, the K1 closed-loop note and the
ModelBlaster integration page. None is in a showdown reproduction page or in the artifact path. They
are stale citations of retired or never-committed helpers (`install_conda.sh`,
`apply_compile_advice.py`, `merlin_adapter.py` and six more), several of them inside the
`zephyr-chipyard-sw` submodule that this artifact does not initialise; the check names each with the
documents citing it and the directory it would be run from. It is not in `verify_no_hardware.sh`
while those stand.

## 5. Known gaps

* The IME arm is measured but not drawn. `schedules/fig_w2pg36ime_greedy_clamped.json` places 1492 of
  its 4768 dispatches on the IME (all on cluster 0, the only cluster where `smt.vmadot` is legal), and
  the same source tree built twice and run back to back on the board reads 30.1 ms camera->control on
  the RVV kernels against 22.3 ms on the mixed schedule -- 1.35x, over three runs each, both arms
  bit-identical under `MODELBLASTER_VERIFY`. Both arms are registered in `measured_timing.XPURT_POINTS`
  (`w2pg36base`, `w2pg36ime`) and re-derive under `--verify`, and the IME arm's cadence trace is
  `ctrl_traces/xpu_w2pg36ime.csv`. It is not in any figure: no flights have been flown against it, and
  the drawn arm is always the arm that flew. Its control cadence (100.0 Hz, gap max 11.3 ms) is the
  same as the RVV arm's (100.9 Hz, 14.3 ms) -- both already meet the 100 Hz control target -- so what
  the IME buys is latency and schedule occupancy, which is where a figure would have to claim it.
* The ModelBlaster submodule pin is correct but **unpushed**, so a clone cannot fetch it. The
  superproject pins `ed776fd1`, and that commit does contain the fused IME kernel, both benches, both
  measured tables and the pool's per-slice tracing -- `docs/K1/ime_kernel_reproduction.md` §2-§3 and
  `ros_baseline_reproduction.md` §2b are all followable *from the pinned state*. What is missing is
  reachability: `ed776fd1` is three commits ahead of `origin/feat/split-linear-along-m` and
  `git ls-remote` does not have it, so `git submodule update --init ModelBlaster` fails and the
  directory comes up empty rather than old. Advancing the pointer is not the fix -- no pushed commit
  carries this work. The commit travels in `artifact/history/modelblaster.bundle` instead (tip and
  sha256 in the tracked `MANIFEST.sha256`), which is what a reviewer must use until the branch is
  pushed; `docs/K1/ime_kernel_reproduction.md` §0 is the three-command recipe and
  `scripts/verify_modelblaster_inputs.py` checks the bundle's tip has not drifted from the pin.
  (An earlier revision of this bullet said the pointer predated the IME work. It does not; the
  checked-in pin and the branch tip are the same commit.)
* `verify_showdown_figure.py --all` reports **64 FAIL** over 19 historical figures. Six figures verify
  0 FAIL and are current: `warehouse_showdown_paper_r36` (the 36 Hz rate-sweep form) and
  `showdown_45hz_pinned_vs_rosdefault_s1003{,_ladder,_allcores,_allcores_merged,_c14}`. The count is not a verdict on
  the paper figures, and it is not one kind of failure. It splits three ways:

  **44 stale by design — superseded, not contradicted.** A sidecar records the sha256 of every file
  its render read, and the campaign CSVs have grown since. `showdown_v3_figure.load_campaigns()`
  globs `results/codesign_feedback/campaign_*/campaign.csv`, so *every* campaign directory flown
  after a render is pooled into the re-derivation: there are 25 such directories today against the
  10 the v3 sidecars name, and ten of them are not yet tracked. Nothing these figures draw is
  contradicted — panel D′ of every v3 form re-derives all seven drawn columns unchanged and merely
  finds two newer ones (`c_d0.30`, `c_d0.40`); panel B's drawn k/n all pass and only the
  equal-replicates bookkeeping moves (384 flights set aside at render, 504 now). Re-rendering them
  *today* chases a moving target: the atlas census went 10236 → 10248 in the ten minutes between two
  runs of the verifier, because `campaign_submitted` is still being flown. Redraw them when the
  campaigns settle, and commit the CSVs in the same commit as the render (a sidecar hashes the
  committed input, so a render committed without its CSV fails for everyone else).

  **19 state a board number the board no longer produces — these must be redrawn, not ignored.**
  Since commit `cdd10967` the ROS 2 harness selects nodes by exact name; before it, a process launched
  as `--nodes perception2` also started `perception`, so the two-YOLO-node arm ran the network twice on
  every frame. Figures drawn from runs before that commit carry that arm (`vanilla4x2`, drawn as "ROS 2
  two YOLO nodes, 8 cores" / "ROS 2 on all cores") at the two-network cost, 5–12x the current run:

  | camera rate | drawn | `ros_traced/summary.csv` today |
  | --- | --- | --- |
  | 45 Hz | 454.6 ms | **37.0 ms** |
  | 60 Hz | 359.8 ms | **68.4 ms** |
  | 90 Hz | 254.3 ms | 252.8 ms (unaffected) |

  At 45 Hz the current run places this arrangement ahead of hand-pinning (56.4 ms) and
  XPU-RT · CP-SAT (56.3 ms) on the effort ladder, rather than four times slower than either. The arm was re-flown
  (`campaign_x2fix`, `campaign_x2fix90`) and `story_figures.RATE_ARMS` already names it at 37.0 ms, so
  `rate_speed_map` re-derives 3/12 completions at 1.0 m/s where the figure draws 0/12. The current
  measurement is what `schedules/measured_gantt_v3_ros8_metrics.json` (37.237 ms, 0 of 448 frames
  late) and the paper figure already carry. The stale bars are in `latency_waterfall`,
  `ros_ladder`, `rate_speed_map`, the atlas (panel E), the final figure (panels N and O), `paper10`
  (panels C and E) and panel F of all nine v3 forms. Two further FAILs are a different defect:
  `warehouse_showdown_paper_cal17` panel A presents its pair as the 56.8 ms XPU-RT arm against the
  242.0 ms ROS 2 arm, but both display dumps
  (`campaign_v2/display_same_cal17/{xpu,ros}_s1010_figdata`) record `percep_latency_ms = 0.0` — no
  camera→control latency was injected into either flight, and the arms differ only in replayed
  control cadence (96.2 against 39.1 Hz) and gain (0.0052 against 0.01277). No render repairs that;
  the pair has to be re-flown with the latency on.

  **1 attribution gap.** See the next bullet.

  A reviewer with no hardware should run `verify_showdown_figure.py --metrics <sidecar>` on the
  figure they care about rather than reading the `--all` total, and should expect 0 FAIL from the six
  current figures above. `--all` is the backlog, not the artifact's state.
* `showdown_45hz_pinned_vs_rosdefault_s1003`'s display pair has **no producer script**. Its dumps are
  `campaign_v2/display_lat_c1.4/{xpu,ros}_s1003_figdata`; that directory is named by its sidecar,
  `artifact/05_render/README.md` and one line of `measurements_and_ablations.md`, and by no script in
  the repository -- `git log -S"display_lat_c1.4" -- scripts/` is empty across every branch and the
  directory's Isaac logs record no argv. The dumps are archived in
  `archive_v3/display_dumps_lat_c14.tar` with their sha256, so the figure **re-renders** from a clean
  clone; the flights behind it are **not regenerable** from anything recorded. It is the only one of
  the audited stems in that position.
  Its sidecar also predates the seed fields, so six of the verifier's checks do not run on it: the
  same-scene layout seed, each flight's episode seed, the baseline's drawn-versus-recorded ending,
  the scene's mean gates, and the two panel-I lane checks. 42 checks against the other stems' 48.
* Four figures have no sidecar (`gantt_8core_timer`, `gantt_loaded_stack`) and eight carry a sidecar
  with no producer recorded; their commands are in `docs/Evaluation/figure_runbook.md` rather than the artifact.
  The two unattributed Gantts were drawn ad hoc for commits `98f36ac6` and `2d842623`
  (`scripts/render_gantt_compare.py` is the likely producer, but nothing records the `--rows` it was
  given), so they cannot be reproduced from the artifact alone. No sidecar records the command line
  that wrote it either, so re-rendering a figure means reconstructing its arguments from the
  sidecar's `sources` and `I` blocks and checking that every previously recorded number comes back
  unchanged.
* The figures the paper carries are `refined/warehouse_showdown_cam30_solver_placed.pdf`, copied to
  `plots/fig_hil_showdown.pdf` (Figure 13), and `refined/warehouse_showdown_cam30_solver_placed_rate_gain.pdf`,
  copied to `plots/fig_hil_showdown_rate_gain.pdf` (Figure 14), both in the evaluation section. Each
  carries a sidecar, neither is on `verify_showdown_figure.REFINED_ALLOWLIST`, and `--metrics`
  re-derives both at 0 FAIL, so the paper's evaluation figures are inside the verifier's reach. They
  are the pair in which the scheduled arm is given no placement instruction and the solver derives
  the mapping; `docs/Evaluation/showdown_cam30_solver_placed_reproduction.md` is their reproduction page, and its §7 records
  which form draws which failure and the render flag that declares it.
  `refined/warehouse_showdown_paper_r30.pdf` held the Figure 13 slot before them and remains
  current and verified, with its own page at `docs/Evaluation/showdown_r30_reproduction.md`.

  The form that stood in that slot before, `refined/warehouse_showdown_story.pdf`, has no sidecar and
  is named in the allowlist, so `--all` skips it. No single committed script builds it
  end to end: its panel I text is in `results/codesign_feedback/refined_src/showdown_recovered.py`, its
  B, C and D come from `scripts/hil_envelope_panel.py` and `scripts/hil_story_figure.py`, and the
  composer named beside it on the allowlist, `sims/scripts/compose_warehouse_showdown.py`, draws four
  panels rather than this layout. Its panel I is a K1-calibrated *schedule model* of the pair
  `schedules/scheduled__flight_deployed_matched_board_cpsat_profiled.json` and
  `schedules/scheduled_ros_partition_deployed_matched_board.json` -- a ROS-style fixed partition of the
  eight harts -- and that composer takes the baseline's control rate as `--r-resp-ms` on the command
  line rather than from a ROS 2 deployment. The forms whose baseline is a measured ROS 2 run on the
  board are `warehouse_showdown_cam30_solver_placed` and `warehouse_showdown_cam30_solver_placed_rate_gain` (the two
  the paper carries, a hand-placed baseline against a solver-placed schedule at a 30 Hz camera, in the
  two gain policies), `warehouse_showdown_paper_r30` (a baseline on all eight cores at a 30 Hz camera),
  `warehouse_showdown_paper_submitted` (the same configuration as that earlier form, measured) and
  `warehouse_showdown_paper_r36`. Each has a reproduction page in `docs/`.

  The modelled form is reproducible on the same terms:
  [`showdown_analytical_reproduction.md`](../Evaluation/showdown_analytical_reproduction.md) is its page, and
  `scripts/ros_pinning_model.py --verify` -- run by `artifact/verify_no_hardware.sh` beside
  `measured_timing.py --verify` -- re-derives what its panel I prints from the model's stated
  inputs and re-reads those inputs out of the schedules they came from. That page's §6 names the
  parts that rest on a model rather than a measurement, the `--r-resp-ms` among them.

  What that leaves open -- how much of the modelled baseline is the model and how much is the cost
  input it was given -- is separated in
  [`ros_pinning_model_profiled.md`](../Baselines/ros_pinning_model_profiled.md), which runs the same recurrence on
  per-node costs read from the board's own ROS 2 runs (`scripts/ros_pinning_profiled.py`, no hardware)
  and puts a board measurement next to every prediction.

* `results/loop_ablation_ladder_v2/ablation_summary.json` is absent, so `loop_ablation.{png,pdf}`
  cannot be rendered; `solver_win_sensor.png` and `mega_warehouse_ros.png` are likewise absent.
* Flight outcomes are not bit-reproducible: the same seed re-flown in the display script can differ
  from the census (a pair is therefore re-verified on the scene it flies, never carried over).
* `p3_hog2`, `multi_hog2`, `spin_hog2` carry a background load generator (`cpu_hog`); they are
  sensitivity arms, not rungs of the ladder.

## 5b. The simulator the flights ran in

Every flight number is Isaac Lab, not the board: the board supplies the measured cadence and latency
each flight replays. [`flight_environment.md`](../Evaluation/flight_environment.md) records the interpreter, Isaac
Sim 5.1.0.0, Isaac Lab `4df6560e` (which is what `sims/IsaacLab` is pinned to), PyTorch, NumPy and
warp versions, and how `sweep_rate_demo.py` resolves Isaac Lab -- `$ISAACLAB_SOURCE`, then the pinned
submodule, then the checkout the results were produced against -- so a clean clone needs only
`git submodule update --init sims/IsaacLab`.

## 6. Where a reviewer starts

`artifact/README.md` is the entry point: the same material as this page, laid out as an ordered path
with the hardware each step needs marked. `artifact/01_board` … `artifact/05_render` each hold a short
page naming the exact commands and the `scripts/` that run them, `artifact/history/` holds git bundles
of this repository and of the ModelBlaster submodule at the commits the results were produced at (both
branches are unpushed, so a plain clone cannot fetch them), and `artifact/verify_no_hardware.sh` runs
§3 above — every line except the board one — in a single command, reporting pass or fail per check.

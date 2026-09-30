# The ROS 2 baseline in three tiers

Every ROS 2 number in this artifact is produced by one of three methods. They answer different
questions, take different inputs and carry different assumptions, so a figure is only read correctly
once its tier is known. This page defines the tiers and assigns every figure under
`results/codesign_feedback/refined/` that draws a ROS 2 arm to one of them.

To select an arm by name and see all three tiers for it — its launch flags, its board runs, its
replay trace, the Tier A / B model evaluated for it — use `scripts/ros_baseline.py`; the arms
ordered by the engineering effort each costs are in [`ros_baseline_ladder.md`](ros_baseline_ladder.md).

| tier | what it is | the ROS 2 timing comes from | runs ROS 2 software? |
|---|---|---|---|
| **A — analytical** | a placement-policy recurrence | XPU-RT's own per-dispatch durations, re-laid under the policy | no |
| **B — calibrated analytical** | the same recurrence | per-node costs read from the board's ROS 2 traces | the inputs come from ROS 2 runs; the prediction does not |
| **C — measured** | ROS 2 C++ nodes on the SpaceMiT K1, their cadence replayed into flights | `rdtime()` brackets and per-core samplers on the board | yes, on the board |

No flight in any tier runs a runtime live. Every flight is an Isaac Lab simulation that replays a
control cadence and injects a camera→control latency ([`run_index.md`](../Artifact/run_index.md) §3); the tiers
differ in where that cadence and latency come from.

---

## Tier A — analytical

**What it computes.** One ROS 2 node per network, each owning a static partition of harts and
released by its own periodic timer at `k·T`. Inside a node the network's dispatch graph runs
sequentially. For one-hart serial compute `C`:

```
start_k    = k*T                         (executor policy "release")
start_k    = max(k*T, finish_{k-1})      (executor policy "queued")
response_k = start_k + C - k*T
```

and a response converts into a command rate by one of two rules: the zero-order hold at the
simulator's 10 ms control step, `1000 / (10 · ceil(response / 10))`, or the loop budget,
`1000 / response`.

**Inputs.** XPU-RT's K1-calibrated per-dispatch durations, summed per network and divided by a
standalone OC-shard speedup table (`scripts/ros_pinning_generic.py:63`). Both arms therefore pay the
same per-op compute, and the comparison isolates placement policy.

**Assumptions.** Serial graphs inside a node; static partitions; jitter-free periodic release;
middleware is free (no DDS serialisation, copies, executor wake-ups or callback jitter); one
single-threaded executor per node. The middleware omission makes the arm faster than a deployment;
the single-thread assumption can make it slower than a deployment that overlaps callbacks.
`scripts/ros2_middleware_tax_k1.py` measures the first direction on the board.

**Implementations.**

* `scripts/ros_pinning_model.py` — the model, its inputs and its numbers, with `--verify`
  (in `artifact/verify_no_hardware.sh`).
* `scripts/ros_pinning_generic.py`, `scripts/ros_pinning_periodic.py` — schedule writers.
* `sims/scripts/compose_warehouse_showdown.py` — the showdown composer that draws a Tier A pair;
  both worst-case control responses enter it as flags, `--r-resp-ms` and `--x-resp-ms`.
* `ros_pinning_baseline.py` and `results/microros_baseline_k1/` in the separate checkout
  `/scratch2/agustin/XPU-RT` (untracked, not part of this artifact) — the per-network-pin
  prediction on `networks_k1_tri_exact_100ms.json` that the Tier A showdown's 12.40 ms and 4.89 ms
  come from.
* `scripts/cores_yolo_service.py` — the ROS arm of the cores × YOLO figure (`--ros flat|published`).
* Schedules: `schedules/scheduled_ros_partition_deployed.json`,
  `schedules/scheduled_ros_partition_deployed_matched_board.json` (the same schedule re-timed by
  per-dispatch board multipliers), `schedules/scheduled_ros_pin_{predicted,board}.json`
  (`policy: ros_pinning_periodic`), and every `schedules/cmp_*ROS*_board.json` (`_board` means
  board-calibrated costs, not board-executed).

**When to use it.** To ask what placement policy alone does at equal per-op cost, and to explore
arrangements that have not been deployed. It is not a statement about ROS 2 software on the K1.

Reproduction: [`showdown_analytical_reproduction.md`](../Evaluation/showdown_analytical_reproduction.md).

## Tier B — calibrated analytical

**What it computes.** The Tier A recurrence, unchanged: same partitions, same periodic release, plus
a keep-last queue term where the arm is saturated.

**Inputs.** Per-node costs read from the K1's own ROS 2 runs under
`results/codesign_feedback/ros_traced/`, where `board/k1_ros_mb/ros_mb_chain_traced.cpp` writes one
`node_callback` row per callback. A camera cost is derived as a residual.

**Assumptions.** Those of Tier A, except that the per-node cost is the one ROS 2 paid on the board.

**Implementation.** `scripts/ros_pinning_profiled.py` (`--costs`, `--table`, `--submitted-spec`,
`--json`).

**How close it is.** Over the 22 unsaturated rows of `--table`, the mean of predicted minus measured
camera→goal chain is:

| cost input | mean residual |
|---|---|
| profiled (Tier B) | **−1.74 ms** |
| submitted (`scheduled_ros_partition_deployed.json`) | **−2.46 ms** |
| board-recost (`..._matched_board.json`) | **+5.18 ms** |

The negative residual is the three topic hops and their executor wake-ups, which the recurrence does
not charge for. Above saturation the recurrence diverges where the board stays flat, because the
board's camera self-clocks and its keep-last queue drops.

**When to use it.** To predict an arrangement's chain and command cadence from traced per-node
costs, and to separate what the policy model contributes from what its inputs contribute.

Reproduction: [`ros_pinning_model_profiled.md`](ros_pinning_model_profiled.md).

## Tier C — measured on K1, replayed into flights

**What it measures.** ROS 2 C++ nodes (`board/k1_ros_mb/ros_mb_chain_traced.cpp`) running the same
staged YOLO IR as XPU-RT, deployed on the SpaceMiT K1 under a named arrangement — pinning,
executor, pool width, QoS depth, control mode — and traced per callback and per core.

**Pipeline.**

1. `scripts/board_deploy_ros_node.sh` builds and deploys the node; `scripts/ros_traced_matrix.sh`
   runs the arrangement × rate matrix; `scripts/pull_ros_traced.py` pulls the traces.
2. `results/codesign_feedback/ros_traced/` holds the runs — 65 deployments, about 440 runs
   (`summary.csv` has 445 rows, two of them smoke runs; [`ros_arm_ranking.md`](ros_arm_ranking.md)
   counts 443) — and `summary.csv` their per-run statistics.
3. `scripts/ros_arms_catalog.py` lays each arrangement out
   ([`ros_arms_catalog.md`](ros_arms_catalog.md)); `scripts/measured_timing.py --verify` re-derives
   every recorded number.
4. `scripts/ctrl_trace_from_board.py` cuts a control-output cadence from a run into
   `results/codesign_feedback/ctrl_traces/ros_*.csv`, and `scripts/figure_constants.py`
   registers each as a `ReplayArm` (`ARM_BY_TRACE`) so no figure types a timing in.
5. Flight campaigns replay the cadence with the arm's measured latency; the results are
   `results/codesign_feedback/campaign_*/campaign.csv`.

**Assumptions.** The flight is simulated; the board supplies the cadence and latency it replays.
A replayed cadence loops the traced series, so bursts and gaps are kept, but the flight does not
feed back into the board's timing.

**When to use it.** Any claim of the form "ROS 2 on the K1 does X". Every measured arm, as an
ordered ladder of deployment effort, is in [`ros_baseline_ladder.md`](ros_baseline_ladder.md).

---

## Every ROS figure under `refined/`, by tier

Assigned from each sidecar's `ros_trace` / `ctrl_trace` / `sources` / `inputs` / `provenance`
fields. Where a sidecar names only a run directory, the tier was read from that directory's
`figure_data.npz` (`ctrl_trace`), and the row says so.

| stem | tier | ROS 2 source recorded | note |
|---|---|---|---|
| `warehouse_showdown_story` | **A** | none — no sidecar; control responses 12.40 / 4.89 ms from `microros_baseline_k1/microros_baseline_k1.json`, a Tier A prediction on the four-network workload | both arms modelled; on `REFINED_ALLOWLIST`; see [`showdown_analytical_reproduction.md`](../Evaluation/showdown_analytical_reproduction.md) |
| `cores_yolo_service_derived_flat` | **A** | `schedules/scheduled_ros_pin_{predicted,board}.json` | `ros_sched` / `ros_board` series |
| `cores_yolo_service_derived_flat_a22` | **A** | as above | |
| `cores_yolo_service_derived_published` | **A** | as above | the K4–K7 ROS points are carried as published, flagged in `unsourced_points` |
| `cores_yolo_service_derived_published_a24p5` | **A** | as above | as above |
| `warehouse_showdown_paper_submitted` | C | `ros_vanilla_c5045.csv` | ROS 2 out of the box (`vanilla_c50`), measured |
| `warehouse_showdown_paper_r30` | C | `ros_vanilla4x230.csv` | |
| `warehouse_showdown_paper_r36` | C | `ros_vanilla4x236.csv` | |
| `warehouse_showdown_paper_allcores` | C | `ros_vanilla445.csv` | |
| `warehouse_showdown_paper_allcores_merged` | C | `ros_vanilla445.csv` | |
| `warehouse_showdown_paper_ladder` | C | `ros_vanilla445.csv` | |
| `warehouse_showdown_paper_cal17` | C | `ros_dir` only → figdata `ctrl_trace` `ros_vanilla445.csv` | inferred from figdata |
| `warehouse_showdown_cam30_solver_placed` | C | `ros_cp3n430.csv` | |
| `warehouse_showdown_cam30_solver_placed_v2` | C | `ros_cp3n430.csv` | |
| `warehouse_showdown_cam30_solver_placed_rate_gain` | C | `ros_cp3n430.csv` | |
| `warehouse_showdown_cam30_solver_placed_rate_gain_v2` | C | `ros_cp3n430.csv` | |
| `warehouse_showdown_cam30_static6` | C | `ros_cp330.csv` | |
| `warehouse_showdown_cam30_threeway` | C | `ros_cp330.csv` | |
| `warehouse_showdown_cam30_allcores_s1001_xpu_3of4` | C | `ros_vanilla4x230.csv` | |
| `warehouse_showdown_cam30_allcores_s1003` | C | `ros_vanilla4x230.csv` | |
| `warehouse_showdown_cam30_allcores_s1007` | C | `ros_vanilla4x230.csv` | |
| `warehouse_showdown_cam30_allcores_v2_s1001_xpu_3of4` | C | `ros_vanilla4x230.csv` | |
| `warehouse_showdown_cam30_allcores_v2_s1003` | C | `ros_vanilla4x230.csv` | |
| `warehouse_showdown_cam30_allcores_v2_s1007` | C | `ros_vanilla4x230.csv` | |
| `warehouse_showdown_cam30_allcores_v3_s1003` | C | `ros_vanilla4x230.csv` | |
| `warehouse_showdown_cam36_allcores_s1001` | C | `ros_vanilla4x236.csv` | |
| `warehouse_showdown_cam36_allcores_s1003` | C | `ros_vanilla4x236.csv` | |
| `warehouse_showdown_cam36_allcores_s1007` | C | `ros_vanilla4x236.csv` | |
| `warehouse_showdown_cam36_allcores_s1009` | C | `ros_vanilla4x236.csv` | |
| `warehouse_showdown_cam36_allcores_s1003_ladder` | C | `ros_vanilla4x236d2.csv` | |
| `warehouse_showdown_cam36_allcores_s1007_ladder` | C | `ros_vanilla4x236d2.csv` | |
| `warehouse_showdown_cam36_allcores_s1009_ladder` | C | `ros_vanilla4x236d2.csv` | |
| `warehouse_showdown_cam36_navpool_s1000_xpu_3of4` | C | `ros_vanilla4x236ns4.csv` | |
| `warehouse_showdown_cam36_navpool_s1005_xpu_2of4` | C | `ros_vanilla4x236ns4.csv` | |
| `warehouse_showdown_cam45_solver_placed` | C | `ros_cp345.csv` | |
| `warehouse_showdown_cam45_static6` | C | `ros_cp345.csv` | |
| `warehouse_showdown_cam45_unpinned_best` | C | `ros_vanilla445.csv` | |
| `warehouse_showdown_cam45_unpinned_best_s1007` | C | `ros_vanilla445.csv` | |
| `warehouse_showdown_cam45_unpinned_best_v2` | C | `ros_vanilla445.csv` | |
| `warehouse_showdown_cam45_unpinned_seed1000` | C | `ros_vanilla445.csv` | |
| `showdown_36hz_solver_vs_rosallhart_s1006` | C | `ros_vanilla4x236ns4.csv` | |
| `showdown_36hz_solver_vs_rosallhart_s1006_ladder` | C | `ros_vanilla4x236ns4.csv` | panel D's `ros_shipped` and `xpu_greedy` rows are 45 Hz-camera energy flights |
| `showdown_improved_hero`, `_compact`, `_slide`, `_slide_v2`, `_envelope` | C | every ROS 2 arm of 13 paired censuses (`census` field) | `scripts/showdown_improved_figure.py`; equal cells and replicates per pair |
| `ros_model_fidelity_three_tier`, `_column`, `_ladder` | A, B, C | the Tier A and Tier B predictions and the measured rows, side by side | `scripts/ros_model_fidelity_figure.py`; each panel badged with its tier |
| `ros_model_fidelity_submitted_metric`, `_column` | A, B, C | the Tier A showdown's metric under submitted, recost and profiled inputs; the measured `p3` arm | as above |
| `ros_model_fidelity_story`, `_story_tall` | B, C | Tier B timeline of `p3`; flights replay the measured `ros_p345.csv` | as above; its flight paths read gitignored `campaign_percep/records/*.npz` |
| `showdown_45hz_pinned_vs_rosdefault_s1000` | C | `ros_vanilla445.csv` | |
| `showdown_45hz_pinned_vs_rosdefault_s1003` | C | `ros_vanilla445.csv` | |
| `showdown_45hz_pinned_vs_rospinned_s1011` | C | `ros_cp345.csv` | |
| `showdown_45hz_solver_vs_rospinned_s1007` | C | `ros_cp345.csv` | |
| `warehouse_showdown_v3` | C | `ros_dir` only → figdata `ctrl_trace` `ros_vanilla445.csv` | inferred from figdata; panel F carries pre-`cdd10967` `vanilla4x2` bars |
| `warehouse_showdown_v3_paper` | C | as above | as above |
| `warehouse_showdown_v3_paper_tall1005_board` | C | as above | as above |
| `warehouse_showdown_v3_paper_tall1005_inb` | C | as above | as above |
| `warehouse_showdown_v3_paper_tall1005_none` | C | as above | as above |
| `warehouse_showdown_v3_tall1000_board` | C | `ros_dir` `display_v3s_c1.4/ros_s1000` → figdata `ros_vanilla445.csv` | as above |
| `warehouse_showdown_v3_tall1005_board` | C | as `warehouse_showdown_v3` | as above |
| `warehouse_showdown_v3_tall1005_inb` | C | as `warehouse_showdown_v3` | as above |
| `warehouse_showdown_v3_tall1005_none` | C | as `warehouse_showdown_v3` | as above |
| `warehouse_showdown_final` | C | `ros_dir` only → figdata `ros_vanilla445.csv`; `inputs` include `ros_traced/` traces | inferred from figdata; panels N, O carry pre-`cdd10967` `vanilla4x2` bars |
| `warehouse_showdown_paper10` | C | as `warehouse_showdown_final` | inferred from figdata; panels C, E as above |
| `warehouse_showdown_atlas` | C | `inputs`: `ros_traced/45_*/trace.csv`, `summary.csv`, campaign CSVs | panel E as above |
| `latency_waterfall` | C | `inputs`: `ros_traced/45_{p3,vanilla,vanilla4,vanilla4x2}_r*/trace.csv` | `vanilla4x2` bar as above |
| `ros_ladder` | C | `inputs`: campaign CSVs + `ros_traced/summary.csv` | as above |
| `rate_speed_map` | C | `inputs`: campaign CSVs (`ctrl_trace` column) | as above |
| `course_progress` | C | `inputs`: `campaign_percep`, `campaign_seeds24` | |
| `crash_position` | C | `inputs`: `campaign_percep`, `campaign_seeds24` | |
| `seed_pairs` | C | `inputs`: campaign CSVs + `ros_traced/summary.csv` | |
| `ros_effort_ladder` | C | `ctrl_trace` per rung (`ros_vanilla45.csv` … `ros_vanilla845.csv`) | |
| `ros_effort_ladder_v2` | C | as above | |
| `control_rate_response` | C | `trace` per point (`ros_cp345.csv`, `ros_vanilla*.csv`, `ros_vanilla_c5045.csv`) | |
| `control_rate_response_v2` | C | `trace` per point (`ros_cp3{15,25,30,45}.csv`, `ros_vanilla*.csv`) | |
| `control_rate_response_cp3` | C | as `_v2` | |

The "pre-`cdd10967`" notes: before that commit the ROS 2 harness started `perception` alongside
`perception2`, so the two-YOLO-node arrangement ran the network twice per frame. The affected bars
and the current measurement are listed in [`artifact_checklist.md`](../Artifact/artifact_checklist.md) §5.

**Sidecars with no ROS 2 arm**, not in the table: `audit_showdown_claims` (the control-rate envelope
of one controller), `hil_feedback_a90`, `hil_feedback_a90h`, `hil_feedback_a120e`,
`hil_feedback_a120h`, `hil_feedback_b5`, `hil_feedback_close_120` (XPU-RT feedback rounds), and
`schedule_evolution_short`, `schedule_evolution_tall` (XPU-RT schedule stages).

**Tier A forms outside `refined/`:** `results/codesign_feedback/warehouse_showdown_board` and
`warehouse_schedule_board` (the composer's form on the board-recost pair), whose sidecars sit beside
them; see [`showdown_analytical_reproduction.md`](../Evaluation/showdown_analytical_reproduction.md).

Envelope panels (B/C of the showdown forms) sweep one controller over control rate and are not a
ROS 2 arm; the ROS 2 marker drawn on them takes the tier of the figure's ROS 2 arm.

---

## Reproducing each tier

* The Tier A form from its stated inputs: [`showdown_analytical_reproduction.md`](../Evaluation/showdown_analytical_reproduction.md).
* The same model on board-profiled ROS 2 costs (Tier B), set against the measurements:
  `scripts/ros_model_fidelity_figure.py` → `refined/ros_model_fidelity_*`.
* The measured arms (Tier C), every rung and how to deploy, replay or model it:
  [`ros_baseline_ladder.md`](ros_baseline_ladder.md) and [`ros_arm_ranking.md`](ros_arm_ranking.md).

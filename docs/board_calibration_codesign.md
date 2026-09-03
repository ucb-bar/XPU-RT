# Board-calibrated co-design scheduling — findings & reproduction

The scheduler's Gantt is a *prediction*. This work measures how far that prediction is from real
SpaceMiT K1 silicon, folds the correction back into the cost model, and shows what it buys:
a fairness win over ROS, a genuine schedule improvement, and a closed feedback loop.

All figures land in `results/codesign_feedback/`; the published gallery is the fastest way to browse.

---

## 1. The predicted-vs-actual gap (and it is NOT contention)

We already had **60 real-time K1 runs** on disk (`results/k1_feedback_exact/board_runs*/`, 178
concurrent dispatches each, carrying *both* the additive prediction and the measured board cycles).

- The additive Gantt under-predicts the board by **+31%** (critical response 8.0 → 10.5 ms).
- Disaggregation (`scratchpad/disaggregate_gap.py`): overhead vs. #overlapping co-runners has
  **R²≈0.01** — it is *not* resource contention (confirms the prior `contention_model` null).
- It is **per-op execution inflation**: the isolated `iree-benchmark-module` systematically
  under-times real on-board dispatch. Worst on f16 (`linear_f16` **2.05×**, `lstm` 1.44×,
  `layernorm` 1.50×) and standalone memory-bound elementwise (`batchnorm` 1.74×); near-zero on int8
  linear (1.10×).
- Held-out validation (fit on one schedule, predict the other): additive |err| 23% → calibrated 12%,
  heavy ops stable across schedules.

Artifact: `results/codesign_feedback/k1_board_calibration.json` — primary key `network/dispatch_id`
(48 exact keys, aggregate ×1.261), per-op fallback for extrapolation. Fig: `predicted_vs_actual_gap`.

Reproduce: `python3 scratchpad/disaggregate_gap.py` · `python3 scratchpad/emit_calibration.py`

## 2. Injecting the calibration into the cost model

`--board-calibration [PATH]` on `run_xpurt_schedule.py` scales each dispatch's time at the
profile-load boundary (`xpu-rt/profile_loader.py`, `_board_calibration_mult` at the `base_t`
assignment) → **CP-SAT, MOSEK, and greedy all become board-honest from one injection point**.
Verified exact: all-ones calibration reproduces the additive schedule byte-for-byte; a single ×3 key
scales exactly that dispatch and the solver re-places it.

```
export XPURT_CPSAT_WORKERS=0
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json <spec> \
  --solver greedy --profiled --max-periodic-iters 1 --board-calibration
```

## 3. Beat-ROS feasibility flip (the fairness win)

ROS-style static partitioning has **no cost model** — it pins each net serially to one hart, so it
cannot ingest a board correction or re-place when a net turns out costlier. Both schedulers suffer
the board gap; only XPU-RT can re-solve. At a 9.3 ms dronet deadline, ROS's own Gantt certifies it
feasible (8.3 ms) but on real silicon it runs **10.4 ms → MISS**, frozen; XPU-RT recalibrates and
meets at **6.1 ms** (sharding dronet across the harts ROS left idle). `mlp_control` (int8) correctly
shows no flip. Figs: `ros_vs_xpurt_calibrated`, `scheduled_vs_real_gantt` (all 8 cores).

## 4. The improvement, and the trap (contended workloads)

On a contended-but-feasible workload (**s4.0**, mlp/fused @ 4 ms): greedy's additive schedule looks
fine but on real silicon misses **3** deadlines (`fused_full` overruns to 1.9×). Re-solving with
board-honest costs (**CP-SAT + calibration**) meets every one: **3 → 0**. The trap (s5.0): naively
switching to the "optimal" solver on *additive* costs is *worse* than greedy — you must feed the
solver real costs. Figs: `improvement_before_after`, `improvement_loop_s5`.

Example menu across networks (`scratchpad/examples_figures.py`, `ex_contact_sheet`): the clean X→0
improvement is specific to the **sensor sharded family** (s3.0 2→0, s3.5 2→0, s4.0 3→0) — it has the
f16+contention combo. int8-dominated sets (transformer, rich-fusion) don't inflate enough to break,
even fly-faster-tightened (honest flat).

## 5. Warehouse e2e + flight A/B

Calibration breaks the 22 ms perception budget for *everyone* including XPU-RT (20.4 → 25.6 ms) — the
YOLO conv-chain inflation is intrinsic, so the honest fix is fly-slower or a cheaper kernel, not
scheduling. Flight A/B (6 real Isaac seeds/condition): the 20.4→25.6 ms gap changes the flight only
across a ZOH control-step boundary (83% → 0% when it straddles one; invisible at the deployed 100 Hz
rate). Figs: `warehouse_calibration`, `warehouse_flight_ab`.

## 6. The feedback loops

- **Multi-lever (manual)** `scratchpad/multilever_loop.py`: fuse/unfuse → shard → board-readjust, each
  a real solve, measured accept/reject. On YOLO the loop accepts unfuse (−40 ms) and correctly rejects
  shard (YOLO is a dependency chain already filling 8 cores by placement). Fusion pays via the board,
  not the additive model (standalone elementwise is memory-bound). Fig: `multilever_loop`.
- **Closed auto-feedback loop** `scratchpad/auto_feedback_loop.py`: closes the previously-OPEN loop —
  `--emit-feedback` hints (`prefer_finer`→shard, `pin_target`→ime, `consider_fuse_with_pred`→fuse)
  now auto-drive each round, re-solving with **board-readjustment every round** (CP-SAT + calibration).
  On s4.0: baseline 2 misses → auto-shard+readjust **0 (ACCEPT)** → auto-ime (no gain, reject) →
  converged. A measuring loop, not a blind one. Fig: `auto_feedback_loop`.

## 7. Solver beats greedy — CP-SAT wins on BOTH multiworkloads (the exact-solver headline)

The user's standing ask: for *our* side always launch CP-SAT, greedy, and MOSEK, and have a non-greedy
solver genuinely beat greedy — real, not faked. It does, on two workloads. The lever is legitimate solver
settings, not a weakened baseline: `XPURT_CPSAT_WORKERS=0` (the default `num_search_workers=1` cripples
CP-SAT) + the model-consistent HEFT warm-start (`scheduler_cpsat.py:cpsat_with_heft_warm_start`; greedy
warm-start was tried twice and abandoned — inconsistent alpha indexing) + an adequate `--time-limit`.

**Feedback workload (contended sensor-fusion + IME, `_4w_..._s5.0`, 217 ops, ~0.9 util):**

| solver | misses | makespan | lateness | wall | status |
|---|---|---|---|---|---|
| greedy | **4** | 30.64 ms | 24.6 ms | 4 s | feasible |
| **CP-SAT** | **0** | **27.85 ms** | 0.0 | 210 s | **PROVEN OPTIMAL** |
| MOSEK monolithic | — | — | — | — | non-converging (no incumbent) |
| MOSEK per-network decomp | — | (bounded) | — | — | bounded upper bound, not a joint schedule |

CP-SAT is feasible where greedy is not, and *certified optimal* — it remaps the contended dispatches onto
the IME-capable P-cores greedy left oversubscribed (`ffn_block0` overruns its 20 ms deadline under greedy).
This corrects a stale on-disk artifact: the previous `s5.0_cpsat` was run with the crippling `workers=1`
default, hit the time limit, and returned 29 misses — *worse* than greedy. The win only appears with the
settings above. Fig: `solver_win_sensor`, `solver_comparison_sensor_headline`.

**Warehouse workload (`_flight_1frame`, 120 ops):** CP-SAT packs the perception frame to **18.43 ms (0
miss)** — 9.7 % tighter than greedy's **20.41 ms** and well under the ROS static partition's **24.35 ms**
(which misses the 22 ms budget). The winner is CP-SAT, explicitly not greedy, and significantly better than
ROS+greedy. Figs: `solver_comparison_warehouse_flight1`, `warehouse_crash_speed`.

That 2 ms is not cosmetic — it translates into flight. In fresh Isaac flights (6 seeds/arm, RL controller,
ZOH latency injection) at control_dt = 24 ms, the board-calibrated latencies split on the ZOH step
boundary: CP-SAT 23.22 ms → refreshes the command every **1 step (fresh)**, greedy 25.72 ms and ROS
30.68 ms → **2 steps (stale)**. Result at cruise 1.0×: **CP-SAT 3/6 completes; greedy 0/6; ROS 0/6** — the
non-greedy winner is the only arm that finishes the gate weave. Fig: `warehouse_solver_flight`.

MOSEK is launched on both (per the ask) and reported honestly: the monolithic MILP does not converge on a
multi-network workload (`docs/solvers.md`), and the per-network decomposition converges only as a bounded
upper bound — neither is a joint-optimal schedule, so CP-SAT is the real winner over greedy.

**Reproduce:** `scratchpad/solver_arms_lean.sh` (all arms, `XPURT_CPSAT_WORKERS=0`) ·
`scripts/compose_solver_win.py --greedy … --cpsat … --spec …` ·
`scratchpad/{solver_comparison_table,warehouse_crash_speed,warehouse_solver_flight}.py` ·
`scratchpad/isaac_sweep_driver.sh` (fresh Isaac flights).

## 8. Coupled workloads, HIL sim fidelity, and the crate-tower crash (co-design realism)

To make the multiworkloads *relate* (a miss in one propagates), we couple them into a `YOLO -> nav ->
control` chain (`data/toplevel/_flight_coupled.json` adds `yolov8_nano_64x96 -> fused_full`; the deployed
and sensor specs already carry `fused_full -> mlp_control`).

**Honest finding — freshness is a workload property here, not a schedule lever.** Driving the repo's own
`xpu-rt/freshness.evaluate_freshness` (via `scratchpad/fresh_eval.py`) on the produced schedules, greedy,
CP-SAT and ROS come out *identical* (e.g. s5.0 nav->ctrl 80%; coupled flight chain 33%). This is by design:
freshness edges are evaluated post-hoc against **producer release times** (`SAMPLE_AT_RELEASE`), so the
metric depends on the workload's periods/windows, not on where the solver places dispatches (confirmed in
`xpu-rt/freshness.py`; CP-SAT/greedy read no freshness fields at all). The scheduler's flight-relevant lever
is therefore **control latency** (schedule-dependent: makespan / critical response), *not* freshness. So the
coupling's real payoff is a richer flight *consequence*: a faster schedule delivers a fresher command and
**reacts to obstacles in time**; a slower one (ROS, "can't process fast enough") reacts late and hits them.

**HIL sim fidelity.** The ZOH command-refresh quantizes latency into `ceil(latency/control_dt)` integer
steps, so a coarse `control_dt` collapses small latency gaps into the same bucket. Two improvements: (1)
**finer sim step** — drop `sim_dt` (0.01->0.005, 200 Hz physics) so each arm's actual latency separates
continuously instead of sharing a hold bucket; (2) **schedule-driven sync** (future) — drive the refresh
from the scheduled `mlp_control` per-instance completion times (present in `scheduled_*.json`) rather than a
single scalar worst-case, the faithful RoSE-style sync.

**Crate-tower crash demo.** The warehouse scene has real collidable 2.0-3.5 m box/crate **towers**
(`--prop_density 0.35 --obstacle_level 8`; `sims/isaaclab_tasks/warehouse_nav/mdp_obstacles.py`), and any
body contact fires the `collision` termination = crash. ROS keeps its resources (fair static partition,
YOLO sharded across its P-cores) but its slow schedule -> stale control (`--sched_latency_ms 12.40`, 2-step
hold at 100 Hz) -> the drone drifts off the x=-8 gate line and collides with a tower; the fresh CP-SAT
schedule (`4.89` ms, 1-step) stays centered and weaves through. Recorded with
`sims/scripts/record_sensor_demo.py --controller rl --save_video` (chase+FPV+overhead+Gantt composite) and
`--dump_figure_data` for a top-down trajectory overlay (ROS red path ending at the tower vs CP-SAT green
completing). Reproduce: `scratchpad/crash_demo/` runs; `sims/scripts/compose_ros_vs_ours_flight.py` overlay.

---

## Honest limitations
- Calibration is measured on the deployed no-YOLO workload; YOLO/warehouse multipliers are extrapolated
  (conv2d_batchnorm2d_silu_s8 via measured conv2d_s8), labelled as such.
- Greedy re-solve rarely improves on slack-rich workloads; the wins need CP-SAT + calibration on
  contended frames.
- The graph-rewrite ingest (compile_advice → apply_*_hint) remains driver-mediated; the closed loop
  above ingests the *scheduling* hints, not graph rewrites.

## Not yet done — external, see REMAINING_EXTERNAL_STEPS.md
- **Real-K1 board confirmation** of the recalibrated schedules: the board (10.44.86.251) is currently
  offline; staged one-command-ready for when it returns.
- **Republish to the original gallery URL**: content is fully published at
  https://claude.ai/code/artifact/84dab88a-a438-4124-af48-983403de8a13 (current org); the historical
  URL is in another org and needs `/login`.

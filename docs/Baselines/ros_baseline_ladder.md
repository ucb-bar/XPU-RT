# The ROS 2 baseline as an effort ladder

The ROS 2 baseline is not one deployment. It exists in sixty-odd configurations ("arms"), from
ROS 2 as it ships to layouts a ROS 2 expert would need a day of profiling to arrive at. A
comparison is only honest when the reader can see which of them the baseline is, what the next
one up would take, and what the board measured for each. `scripts/ros_baseline.py` is the one entry
point to all of it: every arm selectable by name, ordered by the engineering effort each step costs.

It defines nothing itself. The arms and their flags come from `scripts/ros_traced_matrix.sh` and the
run manifests (through `data/ros_arms.json`), the rung order and labels from
`scripts/ros_effort_ladder.py`, the board numbers from `scripts/measured_timing.py`, the paired
flights from `scripts/ros_ladder_paired.py`, the replay traces from `scripts/figure_constants.py`,
and the models from `scripts/ros_pinning_model.py` and `scripts/ros_pinning_profiled.py`. What each
method is and assumes is in [`ros_baseline_tiers.md`](ros_baseline_tiers.md); what each deployment
choice buys, one change at a time, is in [`ros_arm_ranking.md`](ros_arm_ranking.md) §3.

```bash
scripts/ros_baseline.py ladder --camera-hz 45     # the ordered rungs at one camera rate
scripts/ros_baseline.py list                      # every arm, its base arm + suffix, its measured rates
scripts/ros_baseline.py show p3_q1                # one arm in full
scripts/ros_baseline.py deploy p3_q1 --camera-hz 45 --reps 3    # print the board commands
scripts/ros_baseline.py model p3 --tier A --camera-hz 45        # the analytical model for it
```

No command here touches hardware. `deploy` prints; `model` and the rest read the repository.

---

## 1. The ladder

Each rung is one real deployment of the same camera → YOLO → navigation → control chain on the K1.
The order is the ladder figure's (`ros_effort_ladder.LADDER`), with the rungs that figure does not
draw inserted where they belong (`EXTRA_RUNGS` in `ros_baseline.py`: the out-of-the-box
`vanilla_c50`, the chained `cp3`, and the nav-pool arms).

| rung | arm | what it changes | what it takes |
|---:|---|---|---|
| 0 | `vanilla` | nothing: every ROS 2 default — one node per process (4 processes), single-threaded executor, keep-last-10 QoS, nothing pinned, control chained to perception, YOLO run on the callback thread | nothing |
| 1 | `vanilla_c50` | `--ctrl-hz 50`; inert, because a chained control node runs no timer | one flag |
| 2 | `vanilla4` | a 4-hart worker pool for YOLO | one flag, and the pool-enabled build (`MB_WITH_POOL`) linked with the 4-wide YOLO kernels |
| 3 | `vanilla8` | the pool widened to all 8 harts | one flag |
| 4 | `vanilla4_q1` | keep-last-1 on every topic | one QoS setting |
| 5 | `vanilla4tm` | control on its own 100 Hz timer (it is already its own process) | one flag on the control node |
| 6 | `cp3` | camera and perception in one process on harts 0–3 (main thread on 0), nav on hart 4, control on hart 5 | a launch-layout change and expert per-network core choice |
| 7 | `p3` | `cp3` with control on its timer | one flag on the control node |
| 8 | `p3_q1` | `p3` with keep-last-1 | one QoS setting |
| 9 | `vanilla4x2` | a second perception process with its own pool on harts 4–7; the camera alternates frames | a second model instance |
| 10 | `vanilla4x2tm` | `vanilla4x2` with control on its timer | one flag on the control node |
| 11–13 | `vanilla4x2ns4c`, `…a`, `…b` | a 4-way nav worker pool, left to the OS / pinned to 0–3 / pinned to 4–7 | a nav build sharded at codegen (`BINSUF=_nav4`) and one flag; the pinned variants add a core choice |
| 14–15 | `cp3n4`, `cp3n4_d` | the pinned layout with nav (and control) across the E cluster; `_d` with nav sharded 4 ways | a core choice; a sharded nav build |

This table is the shape; the columns *what it changes* and *what it takes* are **derived**, not
written: for each rung, `ros_baseline.py` picks the earlier rung whose launch differs from it the
least, diffs the two per node (process, cores, executor, QoS, pools, control mode and rate,
binary), and names each difference by what an engineer does to make it. Run `ladder` for the
derived text and the measured columns.

### Measured columns

For the chosen `--camera-hz`, each rung carries:

* **control rate** = `1000 / gap_mean_ms` and **camera→goal** = `e2e_goal_med_ms`, from
  `results/codesign_feedback/ros_traced/summary.csv`, pooled over the arm's replicates at that rate
  the way [`ros_arm_ranking.md`](ros_arm_ranking.md) §0 pools them (`measured_timing.derive()`:
  gap means averaged, latency median of the per-run medians). The run count and the number of
  cores above 20 % busy are shown beside them.
* **flights**: the rung against XPU-RT (`xpu_a_cpsat_hard.csv`) over the same (cruise, seed) cells
  and an equal number of replicates per cell, with the two-sided Fisher p, from
  `ros_ladder_paired.py`. Only the rungs that script pairs appear, at the camera rate of their
  replay trace.

A rung with no run at the chosen rate reads **"not measured at N Hz"** and lists the rates it was
measured at. It is never shown as zero. A rung with no paired flights reads "no paired flights at
this rate".

A rung whose launch, as the script stands today, differs from what its recorded runs show carries a
**note**. Today that is `cp3n4`: the script passes `--nav-pool 4 --nav-harts 4,5,6,7`, and its runs
were made before the manifest recorded nav pools, so they neither confirm nor contradict it.
[`ros_arm_ranking.md`](ros_arm_ranking.md) §4 records that the plain `cp3n4` rows predate that line;
`cp3n4_d` is the arm the current script reproduces.

## 2. Picking a rung

* **To show ROS 2 as a user would first deploy it**, draw rung 0 (`vanilla`) and say so in the label.
  It is a default, not the best ROS 2 can do.
* **To claim a win over ROS 2**, the baseline must be the strongest rung measured at that camera
  rate. At 45 Hz that is `p3_q1` or `vanilla4x2tm` (100 Hz control, 31 / 38 ms);
  [`ros_arm_ranking.md`](ros_arm_ranking.md) §1–2 ranks every arm per rate and says which figures
  draw a weaker one.
* **To separate the two axes**, read adjacent rungs: `vanilla4 → vanilla4_q1` changes latency and
  not cadence; `vanilla4 → vanilla4tm` and `cp3 → p3` change cadence and not latency.
* Read the control rate together with the control mode (`show` prints it per node). A chained arm's
  command rate is the rate fresh goals arrive; a timer arm re-sends a held goal, so a high command
  rate there does not mean fresh information.

## 3. One arm in full: `show`

```bash
scripts/ros_baseline.py show vanilla4x2
```

prints:

* the matrix command (`RATES="{hz}" <knobs> scripts/ros_traced_matrix.sh <base arm>`) and, per
  process, the exact command line the script launches on the board, with the camera rate, the
  replicate and the kernels sha as placeholders;
* the effective configuration per node — the node binary's defaults, read from
  `board/k1_ros_mb/ros_mb_chain_traced.cpp`, overlaid with the flags;
* whether the latest run's `manifest.json` agrees with that launch, field by field;
* which tiers exist for it: whether the Tier A and Tier B models support its shape (§5), and its
  Tier C board runs per camera rate with their tags;
* its cadence traces under `results/codesign_feedback/ctrl_traces/` — attributed by each trace's own
  `# source=` header — the latency and hold each is replayed with in `figure_constants`, and every
  `refined/` figure whose sidecar names that trace.

## 4. Measuring and replaying an arm: `deploy`

```bash
scripts/ros_baseline.py deploy vanilla4_q1 --camera-hz 36 --reps 3
```

prints, without running any of it:

0. the one-time node build, `scripts/board_deploy_ros_node.sh`, and the binary the arm needs
   (the kernels are staged as [`ros_baseline_reproduction.md`](ros_baseline_reproduction.md) §1–2 describe);
1. one `scripts/ros_traced_matrix.sh` invocation per replicate, and the per-process board commands each
   one launches. Replicates already on disk at that rate are skipped (`--first-rep` overrides),
   because a successful pull replaces its tag's directory;
2. `scripts/pull_ros_traced.py <tags>` to roll the runs into `summary.csv`, then
   `scripts/measured_timing.py --verify`;
3. `scripts/ctrl_trace_from_board.py …/ctrl_gaps.csv --out …/ctrl_traces/ros_<arm><hz>.csv --warmup-ms 3000`,
   and whether that trace is already registered as a `ReplayArm` in `scripts/figure_constants.py`
   (if not, register it with the measured camera→goal as its latency before any figure uses it);
4. the flight command form (`sims/scripts/sweep_rate_demo.py --ctrl_trace … --percep_latency_ms …`),
   whose remaining flags are in [`reproduction_full.md`](../Artifact/reproduction_full.md) "One flight".

An arm whose knobs cannot be recovered from its runs (`vanilla4x2ns4`: no goal recorded, only the
camera wrote a manifest) gets no command; `deploy` exits 2 and names the siblings whose form is
recorded.

## 5. Modelling an arm: `model`

```bash
scripts/ros_baseline.py model p3 --tier A --camera-hz 45
scripts/ros_baseline.py model p3_q1 --tier B --camera-hz 45     # ~25 s: reads the traced callbacks
```

* **Tier A** calls `ros_pinning_model.model()` with the arm's perception width (its YOLO pool, 1 if
  none) re-costed through the measured speedup table, the camera rate, and control on its timer or
  chained; the control period follows the arm's `--ctrl-hz`. It reports both executor readings
  (`release`, `queued`). QoS depth is not in this model.
* **Tier B** calls `ros_pinning_profiled.predict()` with the per-node costs
  `pooled_inputs()` reads from the board's own traces, the arm's control mode and rate, its QoS
  depth and whether the camera shares perception's process. It reports the model chain, the queue
  term, the predicted control rate and whether the partition saturates.

Both print the Tier C measurement at that rate beside the prediction, or "not measured".

Both models are **per-node pinning** models. `model` refuses, exit 2, an arm whose shape they do not
represent and says why:

| shape | why it is outside |
|---|---|
| two perception instances (`vanilla4x2*`, `x2*`) | one perception node per chain |
| the heavier stack (`r*` arms) | no cost input for `ffn_block` / `dronet` |
| a nav worker pool (`*ns4*`, `cp3n4*`) | nav runs serially on its partition |
| a multi-threaded executor (`multi`, `smte`, …) | one executor thread per node (assumption 6) |
| control in perception's process (`ship`, `spin`, `vanilla4t`, `nproc`, …) | each node its own thread, so a starved timer is invisible |
| unpinned processes (`vanilla*`) | static partitions (assumption 3); `--force` evaluates it as if pinned and says so |

## 6. The arm table and its check

`data/ros_arms.json` is generated by `scripts/gen_ros_arms.py`. It runs `ros_traced_matrix.sh`
once per arm with `ssh` and `sleep` replaced by stubs that record what would have been sent (and the
host set to an unresolvable name), so every process line in the table is the script's own; it
recovers each measured configuration's knobs (`QOS`, `CTRL_HZ`, `HOGS`, `BINSUF`, `NAVPOOL`,
`SUFFIX`) from its tag and manifest through `build_implementation_bundles.ros_command`; and it
compares each expansion with its latest run's manifest.

```bash
scripts/gen_ros_arms.py            # regenerate after changing ros_traced_matrix.sh or adding runs
scripts/gen_ros_arms.py --check    # exit 1 when the table has drifted from either
.venv/bin/python -m pytest -q tests/test_ros_baseline.py
```

The tests check the ladder order, that every `LADDER` arm resolves to a launch that agrees with
its latest manifest, that the out-of-the-box arm's flags are exactly the ROS 2 defaults listed in
rung 0, that `show` and `deploy` carry each arm's flags, that the models refuse the shapes above,
and that the table is fresh.

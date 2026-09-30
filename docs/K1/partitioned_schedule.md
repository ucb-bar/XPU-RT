# A spatially partitioned schedule for the deployed chain

The three deployed networks are scheduled onto disjoint sets of the K1's eight harts:
`yolov8_nano_64x96` on the four P cores and nowhere else, `fused_full` (nav) alone on `CPU_E#0`,
`mlp_control` alone on `CPU_E#1`. This document is the mechanism, the solves, the board runs and
the plain answer to what the partition costs.

The partition is not a tidiness preference. `smt.vmadot` — the K1's IME matrix engine — is legal
only on cluster 0; a per-hart SIGILL probe of all eight harts is at
`ModelBlaster/artifacts/ime_isa_probe/FINDINGS.md`, and harts 4-7 trap on the instruction rather
than running it slowly. Pinning the whole of YOLO to `CPU_P#0-3` is therefore the placement under
which the whole of YOLO is IME-eligible: the partition and the matrix engine are the same lever.

## 1. The mechanism

### What was already there, and why it could not express this

`xpu-rt/profile_loader.py` has `preferred_hw`, but it matches the *profile* hw name
(`hardware.profile_hw` values). In this spec both clusters map to `rvv_x60`, so it cannot
separate `CPU_P` from `CPU_E` here; asking it to raises a `ValueError` naming the available
profile hw.

`workload_factory.parse_infeasible_combinations` maps a dispatch's `{"infeasible_machines": [...]}`
to combination indices — but it mapped names into the **machines** list. Under
`machine_combination_mode: "shard"` a combination is an aligned *block* of harts, not a hart, so
the two index spaces disagree: with the K1's 14 shard combinations, machine index 4 is `CPU_E#0`
while combination index 4 is `['CPU_P#0','CPU_P#1']`. That gap is recorded in
`docs/Feature/board_and_model_gaps.md`.

### What was added

Two general, spec-expressed fields, both read by `create_workload_from_network_hierarchy` and both
applied per network (a periodic network's instances inherit them through `network_info.copy()`):

| field | meaning |
|---|---|
| `networks.<net>.allowed_machines` | which harts this network may occupy: a core (`"CPU_E#0"`) or a whole kind (`"CPU_P"`), case-insensitive |
| `networks.<net>.machine_width` | how many harts one dispatch of it may occupy: an int, or a list of ints |

`workload_factory.machine_restriction_to_infeasible_combinations` turns the pair into the set of
machine-combination indices to exclude. A combination survives only if **every** machine in it is
allowed and its size is one of the permitted widths — a block that straddles the allowed boundary
is not a partial placement, it occupies every hart in the block for its whole duration. Which harts
and how many are separate questions, and a spec has to be able to ask both: pinning YOLO to cluster
0 says nothing about whether a frame runs on one hart or spreads over four, and on this workload
that second choice turns out to be worth more than the first (§4).

A selector that names no machine, a width the hardware does not offer, or a restriction that leaves
no combination at all is a `ValueError` naming what *is* available — a typo that silently restricts
nothing, or everything, is worse than a stop.

`parse_infeasible_combinations` now also answers in combination space when it is given the
combination list, which closes the gap above; without one it reads the machines list as a list of
singletons, which is what the two non-shard modes produce and what callers predating shard mode
assume.

**The restriction is written twice, as an exclusion and as a cost.** The MILP path
(`scheduler.py`) and both CP-SAT paths read `Operation.infeasible_combinations`; the greedy list
scheduler does not read it at all — it takes the combination with the earliest completion, full
stop. A set-only exclusion would be honoured by two solvers out of three, silently. So every
excluded cell also gets `workload_factory.PINNED_OUT_COST_MS` (1e8 ms), the same value and the same
reasoning as `codegen_contract._PINNED_OUT_COST_MS`, which prices widths out for exactly this
reason; CP-SAT folds a cost that large back into its own exclusions. The processing-times list is
copied before it is written to, because a periodic network's instances share one cached list.

Tests: `xpu-rt/tests/test_machine_restriction.py` (25 cases) — combination-space semantics, the
straddling block, width composition, every error path, and an end-to-end check that both fields
reach every instance's `Operation` as an exclusion *and* as a cost while leaving another network's
costs untouched.

Also generalised: `scheduler.shard_only_networks` accepts the literal `"none"`, the empty set,
meaning no network may be widened. An empty list already meant "no restriction", so before this the
only way to say "hold everything at one core" was to name a network that happened to be unable to
widen anyway — which states the intent nowhere.

### One lever that does not reach every solver

`shard_only_networks` is applied by the greedy scheduler and by the registry CP-SAT
(`xpu-rt/scheduler_cpsat.py`), but **not** by the `--solver cpsat` subprocess path
(`xpu-rt/cpsat_scheduler.py`), which is the path `solve_stage2_hard.sh` and the arms below use. The
width-1 CP-SAT table in §3 therefore came back with mixed widths (1790 dispatches at one hart, 196
at two, 59 at four) despite `shard_only_networks: "none"`. Its *hart* partition was still exact,
because `allowed_machines` is applied in the workload builder and so reaches every path. This is
left as it stands rather than repaired here: `wh_chain45_shard_solve*` and `wh_chain90_rich_*`
carry non-empty shard sets and are solved on that path, so changing it would move arms that have
already been measured. `machine_width` is the lever that binds on every path.

## 2. The specs

All derived from `data/toplevel/wh_chain30_solve_500.json` — the half-second table of the 30 Hz
chain, `fig_a30_cpsat_hard_clamped`'s own spec — changing nothing but placement freedom. Windows,
periods, instance counts, horizon and the 80 ms chain deadline are untouched.

| spec | yolo placement | impls |
|---|---|---|
| `data/toplevel/wh_chain30_part.json` | `allowed_machines: ["CPU_P"]`, `shard_only_networks: "none"` | RVV |
| `data/toplevel/wh_chain30_partime.json` | the same | `enable_impls: true` |
| `data/toplevel/wh_chain30_part4.json` | `allowed_machines: ["CPU_P"]`, `machine_width: 4` | RVV |
| `data/toplevel/wh_chain30_part4ime.json` | the same | `enable_impls: true` |

`fused_full` carries `allowed_machines: ["CPU_E#0"]` and `mlp_control` `["CPU_E#1"]` in all four.

The IME specs solve against `gen_root: gen/mb_shard`, which is where the `ime_x60` cells live; its
`rvv_x60` `topo_0` rows for this net are byte-identical to `gen/mb`'s (both total 55.748 ms over 90
dispatches), so the RVV and IME arms at one hart are costed from the same numbers. The width-4 RVV
spec also uses `gen/mb_shard`, because only there is the 4-hart profile a board measurement
(24.66 ms a frame) rather than a repeat of the 1-hart number (`gen/mb`'s `topo_0_1_2_3` totals
55.65 ms, essentially its `topo_0`).

Solved as the other arms are, with the board calibration table:

```bash
export XPURT_CPSAT_WORKERS=0 XPURT_UNIFORM_PACKED_WIDTH=1 XPURT_NO_COMPACT=1
CAL=results/codesign_feedback/k1_board_calibration_yolo110.json
.venv/bin/python scripts/run_xpurt_schedule.py --networks-json data/toplevel/wh_chain30_part4ime.json \
    --solver cpsat --max-periodic-iters 1 --cpsat-time-limit 1800 --use-profiled --board-calibration $CAL
scripts/board_partitioned30.sh schedules/fig_p30w4ime_cpsat_hard.json p30w4imecp gen/mb_shard
```

`scripts/board_partitioned30.sh` clamps to the codegen contract, checks feasibility, switches to
the three-backend walker when the table contains any `impl: ime`, takes
`results/codesign_feedback/board.lock` and runs three replicates.

## 3. The solves

Every solve at the 30 Hz camera, half-second table, `k1_board_calibration_yolo110.json`, seed 42.

| spec | solver | status | wall | makespan | periodic window misses |
|---|---|---|---|---|---|
| `wh_chain30_solve_500` (unpartitioned) | greedy | — | 0.8 s | 525.16 ms | 0 |
| `wh_chain30_solve_500` (unpartitioned) | CP-SAT hard | OPTIMAL | 1376 s | 515.79 ms | 0 |
| `wh_chain30_part`, widening left free | greedy | — | 1.0 s | 771.50 ms | **1543** |
| `wh_chain30_part`, widening left free | CP-SAT hard | **UNKNOWN, no solution** | 3000 s | — | — |
| `wh_chain30_part` | greedy | — | 1.1 s | 518.69 ms | 0 |
| `wh_chain30_part` | CP-SAT hard | FEASIBLE | 3000 s | 516.66 ms | 0 |
| `wh_chain30_partime` | greedy | — | 1.2 s | 505.83 ms | 0 |
| `wh_chain30_part4` | greedy | — | 0.9 s | 498.70 ms | 0 |
| `wh_chain30_part4` | CP-SAT hard | **OPTIMAL** | 3.0 s | 498.75 ms | 0 |
| `wh_chain30_part4ime` | greedy | — | 1.2 s | 490.08 ms | 0 |
| `wh_chain30_part4ime` | CP-SAT hard | **OPTIMAL** | 9.2 s | 490.08 ms | 0 |
| `wh_chain30_part4force` (blanket IME, §7) | greedy | — | 1.3 s | 490.26 ms | 0 |
| `wh_chain30_part4force` (blanket IME, §7) | CP-SAT hard | **OPTIMAL** | 5.2 s | 490.31 ms | 0 |

The two rows worth reading twice are the ones where widening is left free. Four harts cannot both
carry a wide YOLO block and pipeline frames: a 4-hart block occupies the whole cluster for its
duration, so no two frames overlap, and greedy — which picks the earliest completion per dispatch —
takes wide blocks and then has nowhere to put the next frame. It lands on 1543 window misses and a
771 ms makespan against 518 ms when every dispatch is held at one hart. CP-SAT on the same model
did not find a single feasible point in 3000 s (`objective 544444, best_bound 516075, status
UNKNOWN`). The same spec with the width *decided* — one hart per frame, or the whole cluster per
frame — is solved to proven optimality in 3 and 9 seconds. Fixing the width is what makes this
model tractable, and `machine_width` is how the spec says it.

Buildability, before any board run (`clamp_schedule_widths.py`, then
`check_schedule_feasibility.py`, which returned rc=0 on all six):

| table | narrowed by the contract | executed width histogram |
|---|---|---|
| `fig_p30_greedy` | **0** | {1: 2045} |
| `fig_p30ime_greedy` | **0** | {1: 2045} |
| `fig_p30_cpsat_hard` | 120 | {1: 1910, 2: 105, 4: 30} |
| `fig_p30w4_greedy` / `fig_p30w4_cpsat_hard` | 45 | {1: 575, 2: 45, 4: 1425} |
| `fig_p30w4ime_greedy` / `fig_p30w4ime_cpsat_hard` | 45 | {1: 575, 2: 45, 4: 1425} |
| `fig_p30w4force_cpsat_hard` | 45 | {1: 575, 2: 45, 4: 1425} |

The 45 are three dispatches a frame whose packed-weight kernel cannot be built at four harts; the
contract narrows them to two and the schedule stays feasible. The width-1 tables need no narrowing
at all.

## 4. On the board

K1, `SCHED_OTHER`, three replicates per arm, one arm at a time under
`results/codesign_feedback/board.lock`. Every run reports `MODELBLASTER_VERIFY
[yolov8_nano_64x96] max_abs_err=0 max_rel_err=0` and `[mlp_control] max_abs_err=0`; `[fused_full]`
reports 1.83e-4, the nav head's fp16 output, identically on the RVV and IME arms. Traces are
`results/codesign_feedback/xpurt_long/trace_<label>r{1,2,3}_other_run1.csv`; all values are pooled
over the three replicates, warm frames only, and re-derived by `scripts/measured_timing.py
--verify`.

| arm | label | camera→control (median) | camera→goal | YOLO span | control gap mean / max | frames late |
|---|---|---|---|---|---|---|
| unpartitioned, CP-SAT, RVV | `a30cpsat_hardr` | 55.16 ms | 50.40 ms | 45.34 ms | 9.98 / 15.44 ms | 0/36 |
| unpartitioned, greedy, RVV | `a30greedyr` | 70.09 ms | 63.38 ms | 58.42 ms | 10.00 / 15.52 ms | 0/36 |
| partitioned w1, greedy, RVV | `p30greedyr` | 60.09 ms | 55.55 ms | 50.63 ms | 10.00 / 10.03 ms | 0/36 |
| partitioned w1, CP-SAT, RVV | `p30cpsat_hardr` | 60.08 ms | 54.45 ms | 49.58 ms | 10.00 / 10.08 ms | 0/36 |
| partitioned w1, greedy, IME | `p30imer` | 43.42 ms | 39.16 ms | 34.34 ms | 10.00 / 10.05 ms | 0/36 |
| partitioned w4, greedy, RVV | `p30w4r` | 36.75 ms | 32.03 ms | 27.12 ms | 10.00 / 10.03 ms | 0/36 |
| partitioned w4, CP-SAT, RVV | `p30w4cpr` | 36.75 ms | 32.09 ms | 27.23 ms | 10.00 / 10.02 ms | 0/36 |
| partitioned w4, greedy, IME | `p30w4imer` | **26.75 ms** | **22.27 ms** | 17.36 ms | 10.00 / 10.02 ms | 0/36 |
| partitioned w4, CP-SAT, IME | `p30w4imecpr` | **26.75 ms** | 22.32 ms | 17.44 ms | 10.00 / 10.34 ms | 0/36 |

Every arm meets the 80 ms camera→control chain deadline on every frame; the worst single frame
across the partitioned arms is 40.09 ms (w4) and 70.08 ms (w1 CP-SAT), against 66.83 ms on the
unpartitioned arm.

### Does the partition cost or save latency?

**At one hart per frame it costs 4.9 ms** — 60.09 ms against the unpartitioned CP-SAT arm's
55.16 ms. That is the honest price of the restriction taken by itself: four harts for perception
instead of eight means YOLO's span grows from 45.34 to 50.63 ms. Against the unpartitioned *greedy*
arm the partition is 10 ms ahead (60.09 vs 70.09), so the comparison depends on which unpartitioned
arm you take.

**The partition pays for itself twice over as soon as the width and the engine are chosen.**
Giving YOLO the whole P cluster per frame takes it to 36.75 ms, 18.4 ms *under* the unpartitioned
CP-SAT arm; adding the IME takes it to 26.75 ms, 2.06x the unpartitioned arm and 1.37x its own
all-RVV twin — the same ratio the 36 Hz IME pair measured (22.01 vs 29.73 ms). The IME is only
reachable because YOLO is on cluster 0: the restriction that costs 4.9 ms on its own is what the
10 ms saving is bought with.

**The partition also buys a control cadence the unpartitioned arm cannot hold.** With
`mlp_control` alone on `CPU_E#1`, the worst control gap over 144 warm gaps is 10.02-10.08 ms
against 15.44 ms unpartitioned — control never waits behind a perception dispatch, because nothing
else is scheduled on its hart. That is the separation of concerns the partition is named for, and
it is visible in the measurement rather than only in the Gantt.

### Against the ROS 2 baseline

The ROS 2 arms come from `docs/Baselines/ros_with_ime.md`, measured at the same 30 Hz camera on the same
board; they are quoted here, not re-derived. The like-for-like column is camera→**goal**, since
that is what the ROS runs record.

| arm | camera→goal median | p95 | frames late | control cadence |
|---|---|---|---|---|
| ROS 2, 8 harts, RVV (the figure's baseline) | 31.38 ms | 32.40 ms | 14/509 | 33.33 ms (30 Hz) |
| ROS 2, cluster 0, IME | 23.96 ms | 40.52 ms | 126/509 | 33.33 ms (30 Hz) |
| XPU-RT unpartitioned, CP-SAT, RVV (today's figure) | 50.40 ms | 63.86 ms | 0/36 | 9.98 / 15.44 ms |
| XPU-RT partitioned w4, RVV | 32.03 ms | 32.03 ms | 0/36 | 10.00 / 10.03 ms |
| **XPU-RT partitioned w4 + IME** | **22.27 ms** | **22.27 ms** | **0/36** | **10.00 / 10.02 ms** |

The combination gets below the baseline on the baseline's own measure — 22.27 against 23.96 ms —
and the distributions are not comparable in the same direction either: the ROS cluster-0 IME arm's
median is 23.96 with a 40.52 ms p95 and a quarter of its frames late, while the partitioned arm's
p95 equals its median (worst single frame 31.74 ms) with no frame late, and it holds a 100 Hz
control loop whose worst gap over 144 warm gaps is 10.02 ms against the ROS arm's 33.33 ms cadence.
Adding the control leg (camera→control, 26.75 ms) puts us 2.8 ms above the ROS camera→goal figure,
which is not a like-for-like row: the extra is the phase-locked control tick, and it is what buys
the cadence. The argument no longer rests on control rate alone.

## 5. The Gantt sidecars

`results/codesign_feedback/refined/partitioned/measured_gantt_{unpartitioned,partitioned_w1,partitioned_w4,partitioned_w4_ime,partitioned_w4_ime_blanket}.json`
plus `*_metrics.json`, built by `scripts/make_measured_gantt_pair.py` from replicate 1 of each arm
over a 200 ms window at the run's representative offset. `rate30/` is untouched.

```bash
D=results/codesign_feedback/xpurt_long
.venv/bin/python scripts/make_measured_gantt_pair.py \
  --arm unpartitioned:xpu:$D/trace_a30cpsat_hardr1_other_run1.csv:$D/cpu_a30cpsat_hardr1_other_run1.csv:$D/manifest_a30cpsat_hardr1_other_run1.json:schedules/fig_a30_cpsat_hard_clamped.json \
  --arm partitioned_w1:xpu:$D/trace_p30greedyr1_other_run1.csv:$D/cpu_p30greedyr1_other_run1.csv:$D/manifest_p30greedyr1_other_run1.json:schedules/fig_p30_greedy_clamped.json \
  --arm partitioned_w4:xpu:$D/trace_p30w4cpr1_other_run1.csv:$D/cpu_p30w4cpr1_other_run1.csv:$D/manifest_p30w4cpr1_other_run1.json:schedules/fig_p30w4_cpsat_hard_clamped.json \
  --arm partitioned_w4_ime:xpu:$D/trace_p30w4imecpr1_other_run1.csv:$D/cpu_p30w4imecpr1_other_run1.csv:$D/manifest_p30w4imecpr1_other_run1.json:schedules/fig_p30w4ime_cpsat_hard_clamped.json \
  --spec data/toplevel/wh_chain30_part4ime.json --window-ms 200 \
  --out-prefix results/codesign_feedback/refined/partitioned/measured_gantt
```

Each network's hart set, as the sidecars' own bars record it:

| row | yolov8_nano_64x96 | fused_full | mlp_control |
|---|---|---|---|
| unpartitioned | all eight harts | `CPU_E#0-2`, `CPU_P#0-3` | `CPU_E#0-3`, `CPU_P#0,1,3` |
| partitioned w1 | `CPU_P#0-3` | `CPU_E#0` | `CPU_E#1` |
| partitioned w4 | `CPU_P#0-3` (one 4-hart block) | `CPU_E#0` | `CPU_E#1` |
| partitioned w4 + IME | `CPU_P#0-3` (one 4-hart block) | `CPU_E#0` | `CPU_E#1` |
| partitioned w4 + blanket IME | `CPU_P#0-3` (one 4-hart block) | `CPU_E#0` | `CPU_E#1` |

Per-hart busy % over the run, from the per-core sampler, with the harness's own per-hart kernel
fraction beside it (the sampler counts the runtime's poll loop as busy, the kernel fraction does
not — which is why `CPU_E#1` samples at 94-96 % while running 0.8 % kernel):

| row | sampler busy % | kernel % |
|---|---|---|
| unpartitioned | P 50.3 / 13.5 / 24.4 / 13.3 · E 33.3 / 13.3 / 26.8 / 15.8 | P 41.8 / 12.4 / 22.8 / 9.6 · E 28.4 / 11.3 / 22.9 / 13.9 |
| partitioned w1 | P 88.5 / 88.0 / 21.7 / 23.8 · E 14.1 / 96.0 / 1.8 / 0.0 | P 67.7 / 35.5 / 18.4 / 21.3 · E 13.5 / 0.8 |
| partitioned w4 | P 84.4 / 61.0 / 55.6 / 56.2 · E 10.2 / 84.0 / 3.8 / 0.0 | P#0 75.4 · E 14.2 / 0.7 |
| partitioned w4 + IME | P 92.7 / 41.8 / 38.4 / 40.9 · E 15.9 / 94.0 / 0.0 / 5.5 | P#0 51.5 · E 14.4 / 0.8 |
| partitioned w4 + blanket IME | P 92.2 / 44.9 / 44.7 / 44.0 · E 17.0 / 96.0 / 0.0 / 1.8 | — |

(The w4 rows report a kernel fraction on `CPU_P#0` only because a 4-hart dispatch is accounted to
the hart that issued it.) `CPU_E#2` and `CPU_E#3` carry nothing in every partitioned arm: the
partition leaves two harts idle, which is the other half of its cost and is plainly visible here.

## 6. Where the IME lands

`fig_p30ime_greedy_clamped` places 735 of its 2045 dispatches on the matrix engine — 49 a frame, at
one hart — and `fig_p30w4ime_*` places 690 — 46 a frame, at four harts. Every one of them is on
cluster 0, structurally rather than luckily: `allowed_machines` keeps YOLO there and
`capabilities.K1_CAPABILITIES` gives `CPU_E` no `ime`, so no ime combination is ever emitted for
cluster 1. The placement is table-guided by `ModelBlaster/pipeline/ime_cost.py` (one table per conv
op-kind; an op with no table stays on RVV), and the counts match
`docs/K1/ime_kernel_reproduction.md`'s: 49 of 57 fused convs win at one hart, 46 at four.

The `ime_x60` backend carries its own `-march=rv64gcv_zvl256b_zfh_zvfh` in
`ModelBlaster/pipeline/backends.py`, so the ordinary
`BACKENDS=rvv_x60,rvv_x60,ime_x60 CORE_KINDS=rvv,rvv_c1,ime` path compiles the engine kernels with
it; the board logs show the walker built with `kinds: ['rvv', 'rvv_c1', 'ime']`.

## 7. Ablation: is the per-dispatch IME pick worth anything?

The arms above route each YOLO dispatch to the engine only where the measured tables say the
engine wins (`ModelBlaster/pipeline/ime_cost.py`) — 46 dispatches a frame at four harts. The
deployment an engineer with an accelerator and no per-shape profiling would build instead turns the
engine on for everything it has a kernel for: `MB_IME_FORCE=1`, 63 of the net's 98 dispatches, the
switch `docs/Baselines/ros_with_ime.md` gave the ROS 2 baseline. Until now the two sides of that comparison
ran different IME policies and this arm had never been measured under the blanket one.

### Saying "blanket" to a per-dispatch scheduler

A cost model cannot express this deployment. A solver handed both cells always takes the cheaper
one, so the only placements it will ever produce are the ones the table already calls wins — costing
a forced build and re-solving reproduces the table-guided pick exactly. The policy is a
restriction, not a cost, so it is written as one: `networks.<net>.prefer_impls`, resolved by
`workload_factory.impl_restriction_to_infeasible_combinations` next to `allowed_machines` and
`machine_width`.

**Prefer, not allow, and the difference is load-bearing.** `impl` is per dispatch, and 35 of YOLO's
98 dispatches are `add`/`cat`/`chunk`/`maxpool`/`upsample`, for which no matrix kernel exists at
all; `profile_loader` files those cells at its own 1e8 sentinel precisely so nothing is ever placed
on an engine that cannot run it. A strict reading makes those 35 unplaceable and the network
unschedulable — which is not what the forced binary does: it runs them on the ordinary kernels out
of the same build. So the restriction is applied per dispatch, to the dispatches that have a cell
under it, and the solve log says how many took the other route:

```
restricted: yolov8_nano_64x96 -> allowed_machines=['CPU_P'], machine_width=4, prefer_impls=['ime'],
            1/21 combination(s) kept (['CPU_P#0'..'CPU_P#3'], width(s) [4], impl(s) ['ime'])
prefer_impls: 35 of 98 dispatch(es) of yolov8_nano_64x96 have no cell under ['ime'] and keep the
            network's other combinations (no kernel there, not a cost decision)
```

63 = 98 − 35, so every dispatch that *can* be on the engine is, which is the definition of the
blanket deployment.

### Costing the forced build

`scripts/make_ime_profile.py` gained `--ime-ops-from-picks <kernel_picks.json>`: ask the BUILD which
ops it put on the engine rather than asking the table which ops deserve to be there. The two
answers differ for a forced build, and `ime_cost` deliberately does not read `MB_IME_FORCE` outside
the generator, so the artifact is the honest source. With that flag the tool also records the
measured ratio **whichever way it points** — a build that has already chosen runs those dispatches
on the engine whether they win or lose, and filtering the losses out would file a cell the
deployment cannot take.

```bash
R=results/codesign_feedback/ros_traced/yolo_standalone
PICKS=ModelBlaster/build/k1_ime_force_shard4/yolov8_nano_64x96/int8/generated/kernel_picks.json
scripts/make_ime_profile.py --net yolov8_nano_64x96 --gen-root gen/mb_force/profile     --topo topo_0_1_2_3 --from-runs $R/4core_w.txt:$R/ime_force_shard4_4harts.txt     --ime-ops-from-picks $PICKS
#   63/90 dispatches given a MEASURED IME cell (46 faster than RVV, 17 slower -- this build
#   runs them on the engine either way)
```

`gen/mb_force/profile` is `gen/mb_shard`'s `rvv_x60`, `scalar` and `ime_x60` trees with YOLO's two
`ime_x60` cells replaced; nav and control keep `gen/mb_shard`'s. The RVV side of the ratio is
`4core_w.txt`, the same run the table-guided profile was derived from, so the only thing that
changes between the two derived profiles is the IME run. Predicted YOLO cost at four harts: 16.99 ms
blanket against 15.79 ms table-guided. `data/toplevel/wh_chain30_part4force.json` is the spec;
CP-SAT solves it OPTIMAL in 5.2 s, 0 window misses, makespan 490.31 ms.

### The placement, and the build that ran it

| | dispatches on the engine | per frame | ime dispatches on a CPU_E hart |
|---|---|---|---|
| `fig_p30w4ime_cpsat_hard` (table-guided) | 690 / 2045 | 46 | **0** |
| `fig_p30w4force_cpsat_hard` (blanket) | 945 / 2045 | **63** | **0** |

Both keep the partition exactly — YOLO on `CPU_P#0-3`, nav on `CPU_E#0`, control on `CPU_E#1` — and
neither puts an `ime` dispatch on cluster 1, where `smt.vmadot` raises SIGILL. The board run set
`MB_IME_FORCE=1`, and the built tree's own `kernel_picks.json` records `table_guided: false`,
`ime_force: true`, `ime_forced_ops: ['conv2d_batchnorm2d_silu_s8', 'conv2d_s8']` against the
table-guided arm's build, which carries none of those fields. The `ime_x60` backend supplies its own
`-march=rv64gcv_zvl256b_zfh_zvfh` (`ModelBlaster/pipeline/backends.py`), so both IME arms are built
with it. All three replicates verify `max_abs_err=0`.

### The three-way, measured

30 Hz camera, three replicates each, pooled warm frames, same harness and same board session:

| arm | engine dispatches | camera→control | camera→goal | YOLO span | control gap max |
|---|---|---|---|---|---|
| partitioned, all-RVV (`p30w4cpr`) | 0 | 36.75 ms | 32.09 ms | 27.23 ms | 10.02 ms |
| partitioned, **table-guided IME** (`p30w4imecpr`) | 690 (46/frame) | **26.75 ms** | **22.32 ms** | **17.44 ms** | 10.34 ms |
| partitioned, blanket IME (`p30w4forcer`) | 945 (63/frame) | 30.08 ms | 23.64 ms | 18.85 ms | 10.03 ms |

**Table-guided wins.** Against blanket it is 3.33 ms better on camera→control, 1.32 ms on
camera→goal, and 1.41 ms — 7.5 % — on YOLO's own span, with 255 fewer dispatches on the engine.
Both beat all-RVV; the engine is worth 6.7-10.0 ms either way. So `pipeline/ime_cost.py` is not
decoration on this workload: the six `conv2d_s8` detect heads and the eleven other dispatches the
tables measure as losses cost 1.41 ms of perception every frame when they are forced onto the
engine anyway, and the per-shape tables are what keeps them off it.

**Why this points the other way from the standalone numbers.** `docs/Baselines/ros_with_ime.md` measures the
model alone on four harts at 17.69 ms table-guided against 17.26 ms blanket — blanket ahead by
0.4 ms. That is not the same experiment. Those are two single binaries, each running every dispatch
out of itself; the table-guided one has to run the losing convs on its own RVV path within one
build. XPU-RT dispatches per op across two builds, so its table-guided arm sends each dispatch to
whichever binary is faster for it. The 1.41 ms is therefore the value of per-dispatch ROUTING,
which is the runtime's to have and a single-binary deployment's to lack — and it is only available
because the cost model says which way to route.

## 8. Registered

`scripts/measured_timing.SOLVER_ARMS` carries all eight new arms (`p30greedyr`, `p30cpsat_hardr`,
`p30imer`, `p30w4r`, `p30w4cpr`, `p30w4imer`, `p30w4imecpr`, `p30w4forcer`);
`scripts/measured_timing.py --verify`
re-derives every one from the traces with no DRIFT, and `figure_constants.check_registry()` returns
clean. None of them is in a flight campaign, so none is in `figure_constants._ARMS`.

---

## 6. The same chain without the restriction

The tables above name a cluster per network. That is the mechanism working, but it is worth recording
what the solver does when it is given the same machine and told nothing about placement:
`data/toplevel/wh_chain30_free.json` carries no `allowed_machines` and no `machine_width`.

| arm | placement | camera→control |
|---|---|---|
| `p30free` | unconstrained | **26.75 ms** |
| `p30w4imecp` | `CPU_P` + width 4 for the detector, a hart each for nav and control | 26.75 ms |
| `a30cpsat` | unconstrained, engine not offered, one width profiled | 55.16 ms |

Unconstrained, CP-SAT places 808 of the detector's 1470 dispatches four-wide on the P cluster and
takes 679 IME dispatches against the 690 the restricted table takes, and the board measures the same
26.75 ms across three replicates. All eight harts carry work.

So the restriction is not what the chain rests on. What it rests on is the pair of co-design inputs:
the hardware's capability statement -- `smt.vmadot` is legal on cluster 0, which is why placing the
detector there is worth anything at all -- and per-width profile tables that make a wide placement
costable. The same solver without those two measures 55.16 ms.

The restriction mechanism remains what makes the *ablations* above possible, and what a spec needs in
order to ask a narrower question. It is not needed to reach the number.

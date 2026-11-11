# ROS 2 whole-network-pinning baseline over `sched_algo_sweep10` — QRB5165 — PLANNED

Written **before any measurement ran**, in the shape of
`sweeps/qrb5165_sched_algo_sweep10_20260908-210226/SETUP.md`, so the two can be
read side by side. Results land in `results/`, `measured.json` and
`ANALYSIS.md`; **nothing below is revised afterwards** — corrections go in
ANALYSIS.md §0.

Two things ran before this contract was written, and both are inputs rather
than results:

* **The enumeration and the cost table.** `scripts/pinsweep.py` is a pure
  function of the frozen `cost_model.json` and the binding manifests; it
  touched no hardware. Its output is quoted below because it decides what
  gets measured.
* **A single harness smoke run and the executor-floor measurement**
  (`results/floor.json`). Those establish that the harness works and what its
  L3 limit is. No cell result comes from either.

---

## 1. What this is

`qnn_models/runtime/ros_baseline/src/pin_harness.cpp` makes each network **one
ROS 2 node**, with **its own `SingleThreadedExecutor`** on its own thread,
running **its whole dispatch graph on one backend**. No per-op placement, no
tile splitting, no sharding. The only free variable is the map
`network -> backend`.

That is deliberately the negation of what XPU-RT's scheduler does. It is the
fixed-pinning reference the scheduler is measured against — the number a team
gets by writing ordinary ROS nodes and picking a lane per network.

The reference experiment is
`/scratch/dima/rose-infra/RoSE/experiments/microros_pinsweep/`, the same design
on FireSim with micro-ROS. What ports, and what does not, is §6.

**Backends: CPU, DSP, HTA.** GPU is not a pinning candidate. Its measured
per-dispatch floor is 2307.9 µs warm (`qnn_models/opsweep/README.md`) and it is
not the fastest lane for a single network anywhere in this zoo — every
`mlp_control` rung prefers CPU, every `dronet`, `yolov8_nano` and `fastdepth`
prefers DSP, and `vint` runs nowhere but CPU. A whole-model pin to GPU is a
choice no one makes. GPU still appears in the *cells* (`cg` declares CPU+GPU,
`quad` declares all four); it is simply never assigned. §3.3 says what that
costs and how it is reported.

Unlike the FireSim reference — where a network's backend followed from which
hart its node sat on, so hart relabellings (0≡1, 2≡3) had to be canonicalised —
**here the three backends are genuinely distinguishable hardware and there is
nothing to canonicalise.** Two networks pinned to the same backend do contend
for it, and that contention is exactly what the score models and the board
measures.

## 2. Questions

1. **Per cell, how does whole-model ROS pinning compare with measured XPU-RT
   scheduling?** Both sides run the same 42 cells, the same networks, the same
   instance counts, the same periods, the same context binaries, at the same
   governor. The XPU-RT numbers already exist
   (`sweeps/qrb5165_sched_algo_sweep10_20260908-210226/results/phase4_results.json`)
   and are not re-run.
2. **How much is the placement decision worth, measured?** Ranking the legal
   placements is what makes this a *fair* baseline rather than a straw man — a
   badly-placed baseline flatters the scheduler by an amount that has nothing
   to do with scheduling.
3. **Does the whole-model cost model predict the board?** The reference's §6.6
   found its own cost model breaking by up to −44 % on exactly one structural
   case. This one has a different structure and its own failure modes.
4. **Where does whole-model pinning win?** Not a rhetorical question. The
   scheduler pays gating and cross-lane handoff that a pinned node does not,
   and on cells where the best placement is also the best schedule the
   baseline should be at least as fast. If it wins, that gets reported.

## 3. The matrix

### 3.1 Cells

The 42 ported cells of `data/toplevel/s10port/`, which are exactly the 42 the
XPU-RT sweep measured: 11 families × the lane configs each family was
generated for.

| config | lanes declared | pinning candidates (∩ CPU/DSP/HTA) |
|---|---|---|
| `hd` | hta + dsp | hta, dsp |
| `dc` | dsp + cpu | dsp, cpu |
| `cg` | cpu + gpu | **cpu only** |
| `quad` | hta + dsp + cpu + gpu | hta, dsp, cpu |

The 8 cells the XPU-RT sweep generated but never measured (`*_dcg`, `*_dg`) are
out of scope: there is nothing to compare them against.

### 3.2 Whole-model costs, and where the backend preference inverts

The sum of a network's tile costs on the one backend it is pinned to, from the
frozen `cost_model.json` resolved through each network's binding manifest —
the same file `flow_c.py artifacts` re-emits `gen/profile/<HW>/qrb5165_flowc/
.../topo_0/results.csv` from, so both sides of the comparison read identical
numbers. Cached in `model_costs.json`.

| network | cpu | dsp | hta | gpu | best | spread |
|---|---|---|---|---|---|---|
| dronet_sa | 0.860 | **0.627** | 2.593 | 1.404 | dsp | 4.13× |
| dronet_sb | 1.310 | **0.592** | 2.615 | 1.481 | dsp | 4.41× |
| dronet_sc | 1.212 | **0.667** | 2.633 | 1.522 | dsp | 3.95× |
| dronet_sd | 1.474 | **0.672** | 2.667 | 1.530 | dsp | 3.97× |
| dronet_se | 1.452 | **0.838** | 2.788 | 1.659 | dsp | 3.33× |
| dronet_sf | 1.990 | **0.810** | 1.923 | 1.694 | dsp | 2.46× |
| dronet_sg | 2.497 | **0.882** | 1.998 | 1.877 | dsp | 2.83× |
| fastdepth | 3.643 | **2.822** | 3.746 | 3.624 | dsp | 1.33× |
| mlp_control_sa | **0.060** | 0.522 | — | 0.451 | cpu | 8.74× |
| mlp_control_sb | **0.066** | 0.516 | — | 0.439 | cpu | 7.85× |
| mlp_control_sd | **0.110** | 0.524 | — | 0.445 | cpu | 4.75× |
| mlp_control_sf | **0.176** | 0.527 | — | 0.447 | cpu | 2.99× |
| vint | **121.989** | — | — | — | cpu | (only lane) |
| yolov8_nano_sc | 3.887 | **3.461** | 7.928 | 14.936 | dsp | 2.29× |
| yolov8_nano_se | 6.805 | **5.152** | 8.508 | 20.072 | dsp | 1.65× |
| yolov8_nano_sf | 8.665 | **6.798** | 8.681 | 26.067 | dsp | 1.28× |
| yolov8_nano_sh | 16.627 | **8.691** | 10.001 | 41.917 | dsp | 1.91× |

ms; `—` means some tile of that network does not compose there, and a
whole-model pin has nowhere else to put it.

Three things this table decides:

* **The preference is bimodal by family, not by rung.** Every `mlp_control`
  wants CPU by 3–9×; everything else wants DSP. So the cells that mix an
  `mlp_control` with a vision network are exactly the ones where the placement
  decision has teeth, and the cells that do not (`scale_ladder`,
  `depth_chain`) turn on contention instead.
* **HTA is a candidate for 12 of the 16 networks only because Phase 1R of the
  XPU-RT sweep made it one.** Phase 1 found HTA rejecting fifteen of sixteen on
  Batchnorm / Elu / StridedSlice; eleven were unlocked by numerics-preserving
  graph rewrites (`results/rewrite_ledger.json`). The four `mlp_control` rungs
  were not, and that is why they carry no `hta` cell here. Without those
  rewrites this baseline would have had one fewer lane on every `hd` and `quad`
  cell.
* **`vint` is pinnable to CPU and nothing else.** Its encoder tile composes on
  {dsp, cpu}, its decoder on {cpu, gpu}; the intersection is CPU. A whole-model
  pin pays 121.989 ms — 84.163 (encoders) + 37.826 (decoder) — where XPU-RT can
  put the encoders on DSP at 14.213. This is the single clearest forfeiture in
  the matrix and it is a **prediction to be checked**, not a result.

### 3.3 What whole-model pinning forfeits, stated up front

1. **Per-tile placement.** `vint` above.
2. **Per-instance placement.** XPU-RT already does this on this board:
   `saturation_hd__greedy` runs `dronet_sf` instance 0 on HTA and instance 1 on
   DSP in the same window (`runs/saturation_hd__greedy/rep1/run.log`, lanes 0
   and 3). Whole-model pinning cannot.
3. **The GPU lane as an overflow.** On `cg`, dropping GPU as a candidate leaves
   one legal assignment: everything on CPU. That makes 11 cells DEGENERATE
   (below). It is a real narrowing of the baseline relative to the machine
   XPU-RT was given on those cells, it favours the scheduler, and it is
   reported per cell rather than buried. The corresponding *un*narrowed number
   already exists in the same matrix: `quad` gives the same family the full
   CPU+DSP+HTA choice, so **the `quad` row is what a team actually gets**, and
   the `cg` row is what the pinning model can express on the machine that cell
   declares.

## 4. Expressibility — classified before anything was built

Ported from the reference's §1: decide per cell whether this deployment model
can express it, and never report a number for a cell it cannot.

`scripts/pinsweep.py classify` → `results/expressibility.json`.

| verdict | cells | meaning |
|---|---|---|
| `OK` | **31** | expressed as written, ≥2 legal assignments |
| `DEGENERATE (1-LANE)` | **11** | expressed, but exactly one legal assignment: no placement decision exists |
| `INEXPRESSIBLE` | **0** | — |

The 11 degenerate cells are the 11 `cg` cells, for the reason in §3.3.

### 4.1 The reference's three findings, and what changes here

* **The shard arm.** The reference's `wl_sweep_shard` — 44 more cells — is
  inexpressible under whole-model pinning, which is the negation of intra-op
  sharding. **It does not arise here at all**: the QRB5165 port has no shard
  arm, because the work inside a QNN dispatch belongs to HVX, the tensor
  accelerator or the CPU op package and the host cannot subdivide it
  (`sweeps/qrb5165_sched_algo_sweep10.../SETUP.md`, "2 arms → 1 arm"). So this
  is not a loss this baseline introduces; it is a loss the port already took.
* **The depth families' cross-network edge — EXPRESSED, not dropped.**
  `depth_chain`, `depth_contended` and `depth_nav` carry
  `edges: [{from: fastdepth, to: dronet_sf}]` in `data/toplevel/s10port/`, and
  XPU-RT schedules it (`depth_chain_dc__greedy` rep1 shows `dronet_sf` #0 with
  `dep_wait_ms=5.327` behind `fastdepth` #0). The micro-ROS reference could not
  express it, ran the two networks on independent timers, and had to label
  those cells EDGE-DROPPED. **ROS 2 can, and this harness does**: `fastdepth`'s
  node publishes its instance index on `/pin/fastdepth/done`, `dronet_sf`'s node
  is driven by that subscription, and its instance k is released at
  `max(k·period, completion of upstream instance k)` — the same
  instance-to-instance rule XPU-RT applies. **These 12 cells are therefore
  genuine `depth_*` results here**, which is a real gain over the reference, and
  the DDS hop latency it costs is measured and reported per cell rather than
  hidden.
* **`scale_ladder` — expressible.** The reference excluded it only because of a
  compile-time 2–3 node cap in the C harness. ROS 2 on Linux has no such cap;
  the harness is generic over the network list. All 4 `scale_ladder` cells are
  in, and `scale_ladder_quad` is the largest legal space in the matrix at 729
  assignments.

### 4.2 Harness limits — L1..L5, reported per cell, not in a footnote

* **L1 — iteration semantics.** Every network runs exactly its declared
  `num_instances`; the taskset is finite and a pass ends when the last instance
  of the last network completes. There is **no** iteration cap (the reference's
  `NET_A_MAX_ITERS=30`) and **no** one-shot rewrite (the reference's "a one-shot
  network runs exactly once", which silently halved `saturation`'s
  `yolov8_nano_se`). This is the same semantics XPU-RT's runtime uses
  (`[summary] N/N entries executed`), which is what makes the two makespans
  comparable at all.
* **L2 — release semantics, and a deliberate departure from `rcl`.** Instance k
  of a periodic network is released at `t0 + k·period` and runs as soon as its
  executor is free after that. **Releases are never skipped.** `rcl`'s wall
  timer, after an overrun, advances `next_call_time` to the next period
  boundary strictly greater than now — which would idle a busy node for up to a
  period per late instance. That is a penalty with nothing to do with pinning,
  so the callback drains every release already due. XPU-RT gates each entry on
  its release and then runs it when the lane frees, so this is the *matched*
  rule, not a favour.
  **Consequence, stated loudly:** this removes the timer-drift cascade that is
  the headline finding of `qnn_models/runtime/ROS_BASELINE_RESULTS.md` (one
  18 ms exec spike desynchronising `dronet` for the following 1.99 s, 399/400
  deadline misses). Per-instance deadline misses reported here are therefore
  **not** comparable with that document's miss rates: they measure genuine
  overload, not `rcl` timer phase.
* **L3 — executor floor, measured not assumed.** `results/floor.json`: one
  `SingleThreadedExecutor` firing an empty callback tracks 0.5 ms at p50 0.5000
  / p95 0.5030 ms, 1.0 ms at p50 1.0000 / p95 1.0224. The tightest period in
  the whole matrix is 0.670 ms (`saturation_cg`), so **no cell in this matrix
  sits below the floor** — unlike the reference, where a ~200 µs floor degraded
  `bimodal` and `tight_loop`. Any cell that turns out to sit below it is
  reported as DEGRADED per cell.
* **L4 — one process.** All of a cell's nodes live in one process, each with
  its own node, its own executor and its own thread, sharing **one QNN backend
  handle per `.so`**. That sharing is what makes two contexts on one backend
  legal in a process, and it is exactly what XPU-RT's emitted runtime does
  (`SharedBackend` in `runtime_main.cpp`). This is a **departure from the prior
  ROS baseline** (`ROS_BASELINE_PLAN.md`), which used one process per node
  because its `qnn_lib.cpp` created a backend per context. The departure buys a
  single shared `t0` — the prior harness took `t0` separately inside each
  process's constructor and its side-by-side gantt silently assumed the three
  coincided — and it matches XPU-RT's process model. It is *not* free: nodes in
  one process share an address space and a FastRPC client. That is a confound,
  and it is listed as such in §7.
* **L5 — no affinity, no real-time priority.** Nodes are not pinned to cores
  and get no `SCHED_FIFO`. XPU-RT pins each lane to the core its registry
  assigns and asks for `SCHED_FIFO`. This asymmetry is deliberate — an ordinary
  ROS 2 deployment does neither — and it is reported rather than corrected.

## 5. The selection strategy

### 5.1 Enumerate

Every legal `network -> backend` map per cell. Legal = the cell's declared
lanes ∩ {cpu, dsp, hta} ∩ the backends on which that network's every tile
composes. Nothing is canonicalised (§1).

**`n_legal` distribution over the 42 cells** — reported, not assumed:

| n_legal | 1 | 2 | 4 | 6 | 8 | 9 | 12 | 18 | 27 | 64 | 729 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cells | 11 | 5 | 8 | 3 | 6 | 1 | 1 | 3 | 1 | 2 | 1 |

min 1, median 4, max 729, **1078 legal assignments in total**.

### 5.2 Score

Each assignment is scored against a **whole-model** cost — the sum of that
model's tile costs on the one backend it is pinned to (§3.2), which is what
whole-model pinning actually pays. The simulation gives each network its own
serial node, makes each backend one FIFO server shared by every network pinned
to it, releases instance k at `k·period` (all instances at `t0` for an
aperiodic network), and additionally gates an edge's downstream on its
upstream's instance k.

Two objectives come out, because the wall clock alone hides the placement:

* **makespan** — t0 to the last completion of any instance. This is what
  XPU-RT's runtime prints as `[summary] wall=` and what `measured_median_ms`
  records, so it is the directly comparable quantity. On cells with a long
  periodic tail it is pinned by the last *release*, `(n−1)·period`, and is
  nearly identical whatever the placement.
* **np makespan** — the makespan over **non-periodic** networks only. This is
  the objective `sweep10_runner`'s `evaluate(ctx, t, alpha, True)` minimises
  and the one the solver ranking is actually about. Where a cell has no
  aperiodic network it degenerates to the makespan, exactly as `evaluate()`
  does, and is flagged.

### 5.3 Rank

**`(networks starved to zero, then makespan)`** — deliberately not lowest
makespan alone. The reference's §6.5 found on hardware that the
minimum-makespan placement was, on two of seven cells, the one that got there
by not running a network at all.

**A structural difference to state before the numbers exist:** in this harness
the taskset is finite and every instance is run to completion (L1), so on any
completed run `starved` is 0 and the key reduces to makespan. The reference's
starvation arose because its window was closed by a one-shot and a co-resident
could be dropped entirely; nothing here can drop an instance. The rule is still
applied, an incomplete run is still ranked below a complete one, and the live
analogue — a network whose whole-model latency on its pinned backend exceeds
its own period, so it can never hold its cadence — is recorded per assignment
as `overrun_nets` and reported separately.

**Window feasibility is reported separately and never folded into the
makespan.** A miss is an instance completing later than `release +
window_duration`.

### 5.4 Measure on the board, not in the model

`scripts/pinsweep.py enumerate` marks which assignments go to hardware.

* **Full enumeration where `n_legal ≤ 8`** — 33 of 42 cells, every legal
  assignment measured.
* **Sampled where `n_legal > 8`** — 9 cells (`control_mix_quad`,
  `depth_chain_quad`, `depth_contended_quad`, `depth_nav_quad`,
  `saturation_quad`, `tight_loop_quad`, `scale_ladder_{dc,hd,quad}`). The
  measured set is the top 3 of the makespan ranking, the top 3 of the np
  ranking, both uniform (all-on-one-backend) placements, and the bottom-ranked
  assignment — so the measured span always brackets the predicted range rather
  than only its optimistic end. Each such cell records `measure_mode:
  "sampled"` in its plan and in `measured.json`.

**172 assignments, 3 reps each = 516 board runs.**

## 6. What ports from the reference, and what does not

| microros_pinsweep | here |
|---|---|
| expressibility classifier + enumerator + scorer (`pinsweep.py`) | same three jobs, re-implemented for this machine model; hart canonicalisation dropped because the backends are distinguishable |
| `(missed windows, makespan)` prediction ranking, `(starved, makespan)` measured ranking | **kept**, with §5.3's structural caveat |
| whole-model cost = sum of per-op profile costs on one backend | **kept**; here the sum is over binding tiles, which for 15 of 16 networks is one tile |
| hart → backend implied by the bitstream | **does not port.** Backends are chosen directly |
| broker placement (Micro XRCE-DDS session owner takes a hart) | **does not port.** ROS 2 on Linux has no broker; DDS discovery is in-process |
| `NET_A/B/C` compile-time macros, 2–3 node cap | **does not port.** Generic over the network list, which is what makes `scale_ladder` expressible |
| MODELS ordering bug (§4.7) — a short-period network exhausting the iteration cap and force-closing the window on a long-period peer | **cannot arise.** No iteration cap and no shared window close (L1) |
| FireSim TARGET cycles, deterministic | **wall clock on real silicon**, 3 reps, median with spread |
| gantt per run | trace CSV per run; plots in `plots/` |

## 7. Confounds, listed before they can be blamed after the fact

1. **One process (L4).** All nodes share an address space and one FastRPC
   client per backend. The prior ROS baseline used one process per node. If a
   cell's contention behaviour depends on that, this harness will not see it.
2. **No affinity, no FIFO (L5).** XPU-RT pins lanes and asks for real-time
   priority; this does neither. It is the ordinary-ROS-node choice, and it is
   an asymmetry in the scheduler's favour.
3. **The DDS hop on edge cells.** `depth_*` pays a publish→callback latency
   XPU-RT does not (it gates on a semaphore in-process). Measured per instance
   as `dep_wait_ms` minus the upstream's end, and reported.
4. **Governor.** Saved before the campaign, forced to `performance` on all 8
   cores, restored after — the same conditions `cost_model.json` was captured
   under and the same the XPU-RT runs used (`flow_c.py run --tuned`). The prior
   ROS baseline never set it, which left its ROS-vs-MILP comparison with an
   unstated confound.
5. **A shared board.** Other tenants use it. Every run is behind
   `flock -w 900 /tmp/qnn_board.lock`, the lock wait is recorded per call, and
   runs are strictly serial.
6. **Warmth.** Contexts are brought up once per process with 2 warm
   `graphExecute`s each, and every measured pass is preceded by a discarded
   full pass — the analogue of the XPU-RT runs' `FLOWC_ITERATIONS=2`. XPU-RT
   reloads its contexts per rep (new process per rep) where this harness reuses
   them across the 3 reps; that favours this side and is stated.
7. **The cost model is frozen, the board is not.** `measurements/
   qrb5165_v66.json` keeps being rebuilt; every prediction here is against
   `sweeps/qrb5165_sched_algo_sweep10_20260908-210226/cost_model.json` and
   nothing else.

## 8. The noise floor, stated before it is used

From the XPU-RT sweep's own 97 measured points
(`results/phase4_results.json`, recomputed):

| quantity | median rep spread | p90 | max |
|---|---|---|---|
| wall-clock makespan | **7.62 %** | 29.8 % | 159.6 % |
| non-periodic makespan | **9.18 %** | — | — |
| wall clock, excluding `scale_ladder` | 6.94 % | 29.2 % | 61.8 % |

**Every ROS-vs-XPU-RT difference below ~8 % on the wall clock, or ~9 % on the
non-periodic objective, is inside the noise and will be reported as such.**
Both sides quote medians over 3 reps with their spreads. No single-rep number
appears as a result.

## 9. Deliverables

| artifact | what |
|---|---|
| `SETUP.md` | this contract, written first, not revised |
| `scripts/pinsweep.py` | classifier + enumerator + scorer |
| `scripts/drive.py` | stage / floor / run / collect, board discipline |
| `scripts/board.sh` | the `timeout`+`ssh -n`+`flock` wrapper and governor save/restore |
| `model_costs.json` | whole-model cost per (network, backend) |
| `results/expressibility.json` | the §4 verdicts, per cell |
| `results/enumeration.json` | `n_legal`, best / runner-up / worst per cell |
| `plans/<cell>.json` | every legal assignment, scored, ranked, with the ones marked for hardware |
| `configs/<cell>__<aid>.cfg` | what the harness actually reads |
| `results/floor.json` | L3, measured |
| `measured.json` | the board results, medians over 3 reps with spreads, plus the XPU-RT numbers being compared against |
| `plots/` | per-cell ROS-vs-XPU-RT and placement-spread figures |
| `ANALYSIS.md` | written after, corrections in its §0 |
| `reproduce.py` | re-derives every predicted number from the frozen inputs and re-checks the measured tables |

`logs/`, `plots/`, `configs/` and `raw/` are not committed; everything needed
to replicate is.

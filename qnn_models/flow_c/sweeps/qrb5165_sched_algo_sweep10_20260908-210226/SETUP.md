# sched_algo_sweep10 on real silicon — QRB5165 / Flow C — PLANNED

Written before Phases 2-4 ran, in the same shape as
`sweeps/qrb5165_20260829-200620/SETUP.md`, so the two can be read side by
side. Results land in `results/` and `ANALYSIS.md` here; **nothing below is
revised afterwards** — corrections go in ANALYSIS.md, as §0 of the precedent
sweep's does.

One ordering note, stated because it matters for what this document can
claim: **Phase 1 (build the networks, measure their cells) ran BEFORE this
contract was written**, exactly as the precedent's SETUP.md quotes cells it
had already measured. Phase 1's compose survey is an *input* here — it decides
which lanes exist — and its numbers are quoted below. Nothing from Phase 2
onward had run.

## What is being replicated, and what is new

`../RoSE/experiments/sched_algo_sweep10/` is a
ten-solver scheduler bench over 44 workloads x 2 arms, 880 solves plus
supplementary arms, targeting `firesim_f2_rocket_saturn`. Its headline is

| solver | mean impr vs greedy | its rank |
|---|---|---|
| `cpsat:warmbest` | **+9.75%** | 1 of 12 |
| `pso` | **+9.10%** | 2 |
| `sa` | +8.75% | 3 |
| `cpsat:warm` | +8.30% | 4 |
| `best-of-fast` | +7.63% | 5 |
| `cpsat` | +6.03% | 6 |
| `heft_edf` | +2.37% | 7 |
| `greedy` | 0 | 8 |
| **`decomposed`** | **−4.92%** | 9 |
| `greedy_reserved` | −0.10% (57/78 feasible) | 10 |
| `greedy_periodic` | +1.04% (56/78 feasible) | 11 |
| `heft` | −1.66% (50/78 feasible) | 12 |

and its **first honest caveat is that none of it was ever executed**: "These
are predicted makespans, not measurements." Its `fpga/schedules/`,
`fpga/logs/` and `elf/` trees are empty. So the ranking above is a ranking of
what a cost model believes.

**This sweep executes schedules on hardware.** That is the whole point of it.

## Questions

1. **Does the reference's solver ranking survive measurement?** Specifically,
   is the +8-10% the top group claims over greedy — the one conclusion the
   reference calls "safe" and "far outside the noise" — visible in a measured
   makespan on this board?
2. **Does `decomposed`'s reversal survive?** The reference's most interesting
   negative result is that the four-solver sweep's "decomposed wins, +2.18%"
   became −4.92% once six more solvers were in. Both of those are predictions.
3. **Is the rep-to-rep noise floor on this board narrower than the effects
   being claimed?** The precedent QRB5165 sweep measured ~4.4% median rep
   spread. If that is representative, a +9% claim is measurable and a +2%
   claim is not, and the honest report says which differences are inside it.
4. **Where predicted and measured disagree, which documented confound
   accounts for it?** The two candidates already on record for this target are
   the CPU-lane contention term the cost model has no way to express, and the
   fixed per-dispatch cost of each backend.

## What ports from sweep10, and what does not

| sweep10 | here |
|---|---|
| `scripts/sweep10_runner.py` — one (arm, workload, solver) solve + `validate()` | **ports verbatim, unmodified.** Its `build()` is `wl_sweep_bench.build`, its `make_solver` defines the twelve entries, and its `validate()` is the independent float-arithmetic feasibility audit. The comparison being replicated is of METHOD, so this file is not edited. |
| `scripts/sweep10_dispatch.py` — two-pool fan-out | **ports verbatim** (`--arms s10port`), because CP-SAT still asks for 8 search threads while the other eight solvers are single-threaded. |
| `fpga/emit_schedule.py`, `fpga/emit_all.py`, `fpga/pick_winners.py` | **port verbatim.** `emit_schedule.py` was already written for exactly this step and never used; `_sched_hash` is the dedupe key here. |
| `scripts/sweep10_analyze.py`, `sweep10_breakdown.py` | **port with two edits**: the config axis (`PAIRS`) and the hardcoded `/88` denominator. Ranking rule, feasible-first key, `best-of-fast` synthesis and every aggregate are untouched. |
| `experiments/workload_gen/mk_workloads.py` | **re-implemented as `scripts/mk_workloads_qrb5165.py`.** Family definitions (networks, instance counts, period multiples) carried over verbatim; periods re-derived from this board's own cells; the machine axis replaced (below). |
| 11 families x 4 machine configs | **11 families x 4 LANE configs.** |
| 2 arms (`wl_sweep`, `wl_sweep_shard`) | **1 arm.** The shard arm shards operators across two harts, which shows up as more dispatches. This target has no such primitive — the work inside a QNN dispatch belongs to HVX, the tensor accelerator or the CPU op package, and the host cannot subdivide it (`flow_c/README.md`, "Threading and tiling — why not modelblaster's"). A granularity arm here would mean re-slicing and recompiling all 16 networks: a different experiment. **This is a real loss of coverage and it is not disguised.** The reference's shard-specific findings (cold `cpsat` falls from +10.18% to +1.87% at 801 ops; `cpsat:warmbest` holds +8.42%) therefore cannot be re-tested here at all. |
| `milp` | not in the reference's study either (it failed 66 of 80 cells at 126-801 ops). Not here. |
| FireSim TARGET cycles, 1 cycle = 1 ns, deterministic | **wall clock on real silicon.** Every measured point runs **3 reps** and reports a median with its spread. No single-rep number appears as a result. |
| 96-vCPU EC2 box, 880 solves in 43 s + 447 s | **48-core host** for the solves; **one board, shared with other tenants** for the runs, strictly serial behind `flock /tmp/qnn_board.lock`. |

## The machine axis: why it does not port one-for-one

FireSim's configs are two machine kinds at two counts each -- `rvvpair` (0+2),
`gempair` (2+0), `hetero` (1+1), `quad` (2+2). Two things stop that porting:

* **This board has exactly one core of each kind.** `cores/qrb5165_qnn.json`
  declares `hta0`, `dsp0`, `cpu0`, `gpu0` -- one each. `flowc/mb.py`'s
  `install_slot_map` raises on a slot that indexes past the number of cores of
  its kind, so `cpu_x: 2` is not something this flow can emit a runtime for.
  Doubling a lane would also be fiction: there is one HTA and one cDSP.
* **What this board offers instead is a choice of lanes**, and they are far
  more heterogeneous than FireSim's two harts. Measured per-dispatch floors
  (`qnn_models/opsweep/README.md`, warm): `cpu/fp32` 2.0 us, `cpu/int8`
  23.4 us, `dsp/int8` 425.3 us, `hta/int8` 1345.1 us, `gpu/fp16` 2307.9 us.

So the config axis becomes **which lane subset is available**:

| config | lanes | the role it plays | FireSim analogue |
|---|---|---|---|
| `hd` | hta + dsp | two accelerator lanes | `gempair` |
| `dc` | dsp + cpu | accelerator + general purpose | `hetero` |
| `cg` | cpu + gpu | general purpose + the slow accelerator | `rvvpair` |
| `quad` | hta + dsp + cpu + gpu | the whole part | `quad` |

Three two-lane configs -- a fast pair, a mixed pair and a slow pair -- plus one
four-lane config. That is the same shape as the reference's three two-machine
configs plus `quad`.

**HTA is a lane only because Phase 1R made it one, and that is the single
biggest thing this port had to get right.** Phase 1 built every network whole
and found HTA rejected fifteen of sixteen, on three distinct blockers. Had the
sweep proceeded from there, twelve of eighteen tiles would have reached the
solver with one fewer placement than the silicon actually supports, and the
scheduling problem the whole study is about would have been silently narrowed.
Section "Blockers, rewrites and granularity" below is what was done instead.

**The arms are matched WITHIN THIS TARGET ONLY.** No absolute latency here is
comparable with the reference's FireSim numbers, and nothing in this sweep
rests on such a comparison. The precedent QRB5165 sweep makes exactly this
point, and its ANALYSIS.md §0 corrects an earlier draft that had leaned on a
cross-target claim which turned out not to hold.

## Periods and windows: re-derived, not transported

The reference derives each periodic task's period from an analytic MAC count
anchored to the model's *measured* single-hart FireSim runtime. Those anchors
(e.g. `dronet_se` 4.992 ms on gemmini, `vint` 5257 ms) are wrong here by two
orders of magnitude in both directions, and `tight_loop` is already known to
be infeasible by construction in the reference for exactly this reason: its
anchor was wrong for `dronet_sa`, so it shipped 2.3 ms windows against a
35.3 ms critical path.

Here, every period comes from Phase 1's own gap-phase cells:

    period(net) = pmult(net) * (slowest lane cost for that net IN THIS CONFIG)

with `pmult` and the instance counts carried over verbatim from the reference,
and a dependency chain sharing one period covering the whole chain's latency
(the reference's own chain rule, unchanged). `window_duration = period`, as in
the reference. Because the lane set differs per config, a family's periods
differ per config — which is also true in the reference, whose cost table is
per machine pair. **Within a config every solver sees identical durations**,
and that is the comparison the ranking claim is about.

Generation is deterministic: the family table is a literal, the instance
counts are fixed, and the only inputs are this directory's frozen
`cost_model.json` and the binding manifests. There is no RNG in it.

## Pinning

`cost_model.json` is frozen in this directory and every spec points at it, so
`flow_c.py artifacts` re-emits the shared `gen/` profile CSVs from it
immediately before the solves. That matters for the same reason the precedent
sweep documents: `measurements/qrb5165_v66.json` keeps being rebuilt (in-situ
promotion, `feedback` runs), and it has already moved one recorded makespan
from 67.571 ms to 80.23 ms between two runs that were both correct for their
own cost model.

## Validation — a cell that fails a predicate is REJECTED, never built

The reference has no explicit predicates; the precedent QRB5165 sweep has
seven. The ones that apply to a fixed-taskset generator are carried over:

  5. **compose coverage** — every tile of every network in the cell has at
     least one lane of that config with a measured cell, so the solver is
     never offered a placement the board would reject. A cell that fails this
     is not generated at all.
  6. **context staged** — every (tile, backend) the emitted dispatch table
     names has a context binary on the board. Checked by a locked probe before
     the run, not discovered during it.
  7. **no capability-excluded sentinel** in the chosen placement — every
     (tile, kind) pair the table selects has a real measured cell in the
     frozen model rather than the 100x-peer exclusion cost `flowc/artifacts.py`
     writes for an unmeasured one.

Predicates 1-4 of the precedent (periodic coverage, sporadic containment,
uniform stop time, non-empty) are about a *randomly generated* taskset. This
generator is not random — the family table fixes instance counts and period
multiples — so 1 and 3 are properties of the reference's own family design
rather than of a draw, and are reported per cell instead of gated on.
`tight_loop` and `saturation` are deliberately over-subscribed by design and
would fail such a gate on purpose.

Independent of those, **`sweep10_runner.validate()` runs on every returned
schedule**, unmodified: predecessor-plus-transfer ordering, per-machine
no-overlap charging every machine a multi-core combination occupies, negative
starts, starts before `min_start`, and operations assigned to a combination
they cannot run on. It is an audit of the solver's output in float arithmetic,
not the solver's own report.

## Blockers, rewrites and granularity — what Phase 1 had to do first

Phase 1 built all 16 of the reference's networks whole and composed each on
five (backend, precision) pairs. **65 of 80 composed. The 15 failures were the
entire HTA column except `fastdepth`**, and they were three distinct blockers,
each recorded with the verbatim validator string:

| networks | backend | log says | who wrote the op |
|---|---|---|---|
| `dronet_s[a-g]` | HTA | `QnnHtaunsupported op Batchnorm` | the author (BNs after the residual `Add`, so the converter cannot fold them into a conv) |
| `mlp_control_s[abdf]` | HTA | `unsupported elementwise neuson op 0` / `failed to create IHtaOp type name ElementWiseNeuron` | the author (`Elu`) |
| `yolov8_nano_s[cefh]` | HTA | `QnnHtaunsupported op StridedSlice` | **the converter** — modelblaster exports the C2f channel chunk as a `Slice` pair, and the converter lowers it to `StridedSlice` |

Recording those and moving on would have been a defect in this experiment, not
a finding: `qnn_models/PARTITIONING_GUIDE.md` §5 says a rewrite is the first
move and a cut is the fallback, and measured 4.14 ms for dronet on full HTA
after the BN rewrite against 7.23 ms for the best cut and 31.89 ms full DSP.

### The rewrite ledger

Each blocker was removed and the rewritten graph carried as a **candidate**,
built exactly as the base was — fresh ONNX, fresh DLC, fresh calibration, fresh
`qairt-quantizer` pass (§8 rule 5: never splice pre-quantized constants into a
rewritten graph) — and composed and measured on all five pairs.

| base | chain | numerics | result |
|---|---|---|---|
| `dronet_s[a-g]` | `onnxsim`, `bn_to_mul_add`, `flatten_gemm_to_conv` | **preserving** | composes on all 5; HTA unlocked; DSP also 3-18% faster |
| `yolov8_nano_s[cefh]` | `onnxsim`, `channel_slice_to_conv1x1` | **preserving** | composes on all 5; HTA unlocked; DSP 15-24% faster |
| `mlp_control_sf` | `onnxsim`, `elu_to_relu` | **CHANGING** | composes on all 5; **HTA measures 2.186 ms against 0.176 ms on the CPU** |

`bn_to_mul_add` rewrites BN into its own affine form `Mul(x, γ/√(σ²+ε)) +
Add(β − μγ/√(σ²+ε))`; `flatten_gemm_to_conv` re-indexes the head's inner
product as a convolution over the whole spatial extent, which removes the
`Flatten` the converter turns into a `Transpose`; `channel_slice_to_conv1x1`
writes a static channel gather as a 1x1 convolution with a one-hot selection
weight. All three are algebra, not approximation. **Neither existed in the
tree** — `optimization_flow.md`'s punch-list names both `bn_to_mul_add.py` and
`split_to_conv1x1.py` as unwritten, and `slicing_study/RESULTS.md` says
`yolov8n_nosplit.onnx` "was taken as given, not rebuilt" — so
`scripts/onnx_rewrites.py` implements them, following the guide's own math.

**A rewrite is adopted only where it MEASURES better** (`scripts/phase1_adopt.py`,
per network per lane, minimum over {base, variant} x {fp32, int8}), and **a
numerics-changing variant is never adopted at all**. The `elu_to_relu` probe
exists to answer "would HTA ever be worth it for the control loop?" by
measurement rather than by argument. The answer is no, by 12x, and that is now
a measured statement about this board rather than an inference from the
dispatch floor. It is labelled numerics-changing wherever it appears and does
not enter the cost model.

Net effect on the cost model: **HTA cells go from 1 to 12** (7 dronet rungs,
4 yolov8_nano rungs, fastdepth), and 26 of 70 lanes are served by a rewritten
graph.

### Precision is a per-tile decision, and it moved the numbers a long way

The slicing study found CPU int8 beats fp32 by 4x on ViNT's encoders and loses
by 5x on dronet. This sweep measured both precisions for all 28 candidate
graphs and reproduces that split on its own networks:

| family | cpu fp32 | cpu int8 | winner |
|---|---|---|---|
| `dronet_s*` | 0.86–2.61 ms | 5.02–10.69 ms | **fp32**, by 4.1–6.7x |
| `mlp_control_s*` | 0.060–0.176 ms | 0.062–0.179 ms | tie |
| `yolov8_nano_s*` | 17.8–60.3 ms | 3.9–16.6 ms | **int8**, by 3.6–4.6x |
| `fastdepth` | 6.47 ms | 3.64 ms | **int8**, by 1.8x |

A blanket per-flow choice would have put `yolov8_nano_sh` on the CPU lane at
60.3 ms instead of 16.6 ms — a 3.6x error on the largest tile in the sweep, in
the lane the scheduler is most likely to overflow into.

### Granularity: every network keeps one tile, for two different measured reasons

`scripts/granularity.py` applies the slicing study's break-even rule — an extra
DSP dispatch costs `0.37 ms + 5.4 ns x boundary_bytes`, where `boundary_bytes`
counts **every** tensor crossing the cut — per network and per rung, with the
boundary bytes read out of each rung's own ONNX:

| family | candidate cut | boundary bytes | added DSP dispatch | whole-net critical path | decision |
|---|---|---|---|---|---|
| `dronet_s[a-g]` | 3 residual sums | 1.0–18.8 KB | 1.13–1.41 ms | 0.59–1.08 ms | **keep whole** — the dispatch alone exceeds the network |
| `mlp_control_s*` | none | — | — | 0.060–0.176 ms | **keep whole** — no interior worth a dispatch |
| `yolov8_nano_s*` | FPN neck + head (the study's k=3 shape) | 36.9–230.4 KB | 1.14–3.23 ms | 3.46–8.69 ms | **keep whole** — affordable, but unlocks nothing |
| `fastdepth` | decoder upsample boundary | 32.8 KB | 0.55 ms | 2.82 ms | **keep whole** — same |

The second reason is the interesting one and it is a consequence of the
rewrites: after Phase 1R every network composes on every lane whole, so a cut
can no longer buy placement — and the study's conclusion 1 is that placement is
the *only* thing slicing buys. The rewrite was the cheaper way to buy it.

`vint` is the exception and keeps its shipped two-tile split, because there the
cut genuinely does buy placement: the monolith runs nowhere but the CPU, while
the encoder tile composes on the DSP.

**The scaled ladders make the no-cut decision stronger, not weaker.** The
0.37 ms dispatch is a FastRPC property independent of the graph, so scaling a
rung *down* leaves the fixed cost unchanged against less work. Every rung below
dronet's ~0.6 ms whole-network DSP time is uncuttable by construction.

## Cells are bimodal on the accelerators, so each is measured five times

The first single-pass measurement put `dronet_sc/dronet_sc_full@dsp` at
2.666 ms between rungs measured at 0.673 and 0.646 ms, with a gap/loop ratio
of 4.9x where every other cell sat at 1.0–1.5x. That is the bimodal
power-collapse `qnn_models/opsweep/README.md` documents — "they power-collapse
mid-loop even at zero gap, so repeat passes scatter 4x" — and one pass cannot
tell it from a real cost.

Rather than hand-pick which cells to redo, **every (tile, lane) pair is
measured in five independent passes and the cost model takes the per-cell
median**, with all five values and the spread recorded as provenance. That is
a rule stated in advance, not a judgement applied after seeing the numbers.

It also produces a prediction this sweep can test: because the collapse is a
property of the target rather than of the harness (the gap phase inserts
exactly the idle the runtime's own gate inserts), **some measured runs should
pay the collapsed cost the median does not carry**, so measured > predicted
should have a heavy right tail rather than being symmetric noise.

## Reporting rules, fixed in advance

* Medians over reps, always, with the spread quoted next to them. **No
  single-rep number is a result.**
* Every difference is held against the measured rep spread. A predicted
  difference smaller than the spread of the two points it separates is
  reported as **unresolved by measurement**, not as a result in either
  direction.
* The **non-periodic makespan** is the primary quantity, because that is what
  `sweep10_runner` ranks on (`evaluate(..., restrict_to_nonperiodic=True)`).
  The wall clock is reported alongside it, but it is usually pinned by the last
  periodic instance and would flatten the ranking by construction.
* Where a prediction and a measurement disagree, ANALYSIS.md names which of
  the confounds above accounts for it, or says it does not know.
* The realised matrix is stated exactly. Coverage that was planned and not run
  is listed as not run.

## Hardware / software under test

    board       QRB5165 (SM8250, Hexagon v66) at qrb5165, QAIRT 2.45
    lanes       HTA (hta0, core 7) / Hexagon v66 cDSP (dsp0, core 6) /
                Kryo 585 CPU (cpu0, cores 5,4) / Adreno 650 (gpu0) -- all four
                are scheduler lanes, HTA only after Phase 1R's rewrites
    registry    qnn_models/flow_c/cores/qrb5165_qnn.json
    runtime     generated per point by flowc/emit_runtime.py,
                --lane-mode kind-network, built on-board with g++
    conditions  `--tuned`: performance governor on all 8 cores plus one
                warm-up walk (the trace reports walk 2), governor restored
    solver host 48 cores / 125 GB. ortools 9.15.6755 in a venv at
                `<repo>/.cpsat-venv` — see "CP-SAT" below.
    profiles    gen/profile/<HW>/qrb5165_flowc/... written by modelblaster's
                profile_writer with clock_mhz=1.0, so a "cycle" is a µs

## CP-SAT

The brief flagged that no interpreter on this host had `ortools`, and that if
there were no network access the four CP-SAT entries would have to be dropped.
**There is network access.** `python3 -m venv <repo>/.cpsat-venv && pip install
ortools numpy` succeeded (ortools 9.15.6755), and a smoke solve of a 66-op
Flow C workload returned `OPTIMAL` in 0.47 s. So **all twelve solver entries
run**, including `cpsat`, `cpsat:warm` and `cpsat:warmbest`, and none of the
reference's conclusions is untestable for want of a solver.

`XPURT_CPSAT_PYTHON` points at that venv; `xpu-rt/cpsat_scheduler.py` shells
out to it, which is also why the dispatcher's two-pool split still matters.

## The networks

16 networks were rebuilt for this sweep from modelblaster's own model zoo —
the `_s*` scaled rungs plus `fastdepth`, which are exactly the reference's
network set. They are thin wrapper modules that set a scale env var before
importing the backing module, generated by
`experiments/workload_gen/mk_scaled_variants.py`. Flow C ingests PyTorch
through the same `extract_graph`, so no new model was invented.

The zoo used is the RoSE working tree
(`../RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster`),
**the same one the reference used**, because the `modelblaster` submodule in
this repo is pinned at 2026-08-27, which predates the `_s*` wrappers
(2026-09-03) and `fastdepth.py` (2026-09-05). Nothing is written into either
tree; the IR and ONNX land under this sweep.

**These are latency and scheduling targets, never accuracy results.**
`mk_scaled_variants.py` is explicit that non-default rungs keep seeded random
init, because the trained checkpoint only fits the default geometry. That
applies to every `dronet_s*` except `se` and every `mlp_control_s*` except
`sd`. `fastdepth` has no committed checkpoint at all and is always
seeded-random. `yolov8_nano_s*` is the exception in the other direction:
channel counts are input-size independent, so `get_model()` loads the COCO
checkpoint at every rung.

`vint` is the one network not rebuilt. Its shipped manifest is a real two-tile
split (encoders / decoder) cut at the compress projections after a documented
GELU rewrite, with measured cells on dsp, cpu and gpu. Rebuilding it as one
tile would have made both `vint_*` families use a network that runs nowhere
but the CPU — a different workload from the reference's. Its cells are copied
verbatim into the frozen model and flagged there as `in_situ_p50_pooled` while
the 16 new ones are `gap_median`. Two provenances in one cost model, stated
rather than blended.

## Cells: gap phase, not back-to-back

`remeasure_cells.py`'s docstring is the reason. Back-to-back cells predicted
in-situ tile duration at 0.999x for tiles >= 1 ms but **1.655x for tiles
< 1 ms**, missing by a roughly fixed **+0.234 ms**, because the runtime calls
each tile once per period from a lane thread that was asleep until its gate
fired. These cells feed exactly such a scheduler, so every cell here is
`gap_median_us` at `--iters 40 --gap-us 3000`, with `loop_median_us` kept
alongside for comparison. CPU cells are measured UNMASKED: the lane's exec
mask binds only the lane thread, while QnnCpu builds its thread pool at
bringup with full-machine affinity.

CPU tiles are `fp32`, not `int8`, even where the int8 context composed. Two
reasons, both already on record for this target: QnnCpu's int8 conv is a
reference kernel (12.2 ms for fused_split's vision branch against 0.014 ms
for its 8x8 depth branch), and for `mlp_control` the int8 CPU path is
*numerically dead* — three different inputs give byte-identical outputs,
verified with `qnn-net-run`. Since `mlp_control` rungs are 4 of the 16
networks, taking int8 on the CPU lane would put a wrong number in the cost
model for a quarter of the set. The int8 CPU cell is still measured and
recorded so that choice is auditable rather than asserted.

## The matrix

    arm       s10port                     (1; the shard arm does not port)
    families  11                          (verbatim from the reference)
    configs   hd, dc, cg, quad            (4 lane subsets)
    cells     11 x 4 = 44 generated, 42 kept
              vint_intro_hd and vint_multi_hd REJECTED on predicate 5:
              vint's decoder has no HTA and no DSP cell, so in a
              two-accelerator config the solver would see only exclusion
              costs for it. That is the predicate working, and those two
              cells are not run rather than run on a fiction.
    networks  17                          (16 rebuilt + vint reused)
    tiles     18                          (17 whole + vint's two)
    solvers   12                          (the ten + best-of-fast + cpsat:warmbest)
    solves    42 x 12 = 504               host only
    reps      3 per executed point        medians reported, spreads quoted

**Board time is the binding constraint**, so Phase 4 is explicitly tiered and
the realised matrix is reported exactly, with no implied coverage:

  A. **all 12 solvers on a representative subset of cells** — this is what
     actually tests the ranking claim, and it is the highest-value data here.
  B. **winner + greedy for every cell** — coverage, winner by
     `pick_winners.py`'s feasible-first rule (discard invalid, prefer
     `misses == 0`, then min objective).
  C. widen from there as time allows.

Points are deduped by schedule content hash (`emit_schedule.py::_sched_hash`,
the op -> (combination, start, duration) map) so two solvers that produced an
identical assignment do not consume board time twice. That dedupe is measured,
not assumed: the hash is recorded per point and the duplicate is named.

## Known confounds, stated up front

* **CPU cells are load-dependent and the cost model cannot express it.** The
  shipped README documents `feedback` oscillating rather than converging for
  ViNT's CPU decoder (12.7 ms alone, 37.8 ms in situ, 12.8 ms next run). Every
  config here except `dg` has a CPU lane, so expect the CPU-lane placements to
  be the least accurate part of the sweep. Per-tile ratios are reported, not
  just makespans.
* **Per-dispatch cost is large and lane-dependent** — 425 µs on the DSP,
  2308 µs on the GPU, warm. A solver that wins by fragmenting work across
  lanes pays that on hardware and not in the model, which is a mechanism by
  which a predicted ranking could invert. This is the specific thing this
  sweep is able to see and the reference is not.
* **GPU is a genuinely slow lane** and its warm timings are bimodal — Adreno
  power-collapses mid-loop even at zero gap, with repeat passes scattering ~4x
  on small ops. `dg` and `cg` lean on it hardest.
* **Other agents share this board.** Lock wait is probed immediately before
  every rep and recorded with it.
* **Board disk.** cDSP crash dumps have filled `/` to 94% earlier in this
  session. `df -h /` is recorded with every staging probe.
* **Cross-family absolute comparisons are unsound**, as in the reference: the
  valid comparison is *within* a workload cell, where all twelve solvers see
  identical durations.
* **`tight_loop` is reported apart from every aggregate**, as in the
  reference. It is over-subscribed by construction there; whether it still is
  here with re-derived periods is itself a Phase 2 observation.

## Outputs

    bindings/            one manifest per rebuilt network, generated from the
                         board's own compose verdicts
    cost_model.json      FROZEN; every predicted number is solved against it
    specs/               Flow C workload spec per cell
    schedules/           scheduled_*.json + .meta.json (hash, objective, audit)
    results/             phase1_compose.json, phase2_generated.json,
                         phase3_all_results.json, winners.json,
                         phase4_results.json, RESULTS.md, breakdown.md
    runtimes/, runs/     dispatch tables and board logs (not committed)
    ANALYSIS.md          written after the run
    reproduce.py         tiered reproduction, in the precedent's style

---

## The cells this sweep solves against

Frozen in `cost_model.json`; regenerate this table with
`python3 scripts/cell_table.py`. Median over five passes of the gap-phase
median, per (tile, lane), with the graph and precision Phase 1A adopted.

| tile | dsp (ms) | cpu (ms) | gpu (ms) | hta (ms) | statistic |
|---|---|---|---|---|---|
| `dronet_sa/dronet_sa_full` | 0.627 | 0.860 | 1.404 | 2.593 | gap_median |
| `dronet_sb/dronet_sb_full` | 0.592 | 1.310 | 1.481 | 2.615 | gap_median |
| `dronet_sc/dronet_sc_full` | 0.667 | 1.212 | 1.522 | 2.633 | gap_median |
| `dronet_sd/dronet_sd_full` | 0.672 | 1.474 | 1.530 | 2.667 | gap_median |
| `dronet_se/dronet_se_full` | 0.838 | 1.452 | 1.659 | 2.788 | gap_median |
| `dronet_sf/dronet_sf_full` | 0.810 | 1.990 | 1.694 | 1.923 | gap_median |
| `dronet_sg/dronet_sg_full` | 0.882 | 2.497 | 1.877 | 1.998 | gap_median |
| `fastdepth/fastdepth_full` | 2.822 | 3.643 | 3.624 | 3.746 | gap_median |
| `mlp_control_sa/mlp_control_sa_full` | 0.522 | 0.060 | 0.451 | — | gap_median |
| `mlp_control_sb/mlp_control_sb_full` | 0.516 | 0.066 | 0.439 | — | gap_median |
| `mlp_control_sd/mlp_control_sd_full` | 0.524 | 0.110 | 0.445 | — | gap_median |
| `mlp_control_sf/mlp_control_sf_full` | 0.527 | 0.176 | 0.447 | — | gap_median |
| `vint/vint_decoder` | — | 37.826 | 16.425 | — | in_situ_p50_pooled |
| `vint/vint_encoders` | 14.213 | 84.163 | 55.854 | — | in_situ_p50_pooled |
| `yolov8_nano_sc/yolov8_nano_sc_full` | 3.461 | 3.887 | 14.936 | 7.928 | gap_median |
| `yolov8_nano_se/yolov8_nano_se_full` | 5.152 | 6.805 | 20.072 | 8.508 | gap_median |
| `yolov8_nano_sf/yolov8_nano_sf_full` | 6.798 | 8.665 | 26.067 | 8.681 | gap_median |
| `yolov8_nano_sh/yolov8_nano_sh_full` | 8.691 | 16.627 | 41.917 | 10.001 | gap_median |

`vint`'s two cells are the shipped model's, copied verbatim and flagged
`in_situ_p50_pooled` rather than `gap_median` — two provenances in one cost
model, stated rather than blended.

The dashes are real and each has a recorded reason in
`results/compose_failures.json`: `mlp_control` has no HTA cell because `Elu`
has no algebraic equivalent and the numerics-changing probe measured 12x worse
than the CPU; `vint`'s decoder has no DSP cell (`gelu_s8` and its action
post-process) and neither vint tile has an HTA cell.

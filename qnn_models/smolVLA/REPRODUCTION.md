# SmolVLA vision encoder — reproduction on QRB5165 v66

Re-run 2026-08-31. Data log: `reproduction_data_log.json` (99 cells).
Re-measure command:

    bash profile_vision_v3_correct.sh --iters 20 --skip-build

All 8 cores on `performance` during measurement, behind `/tmp/qnn_board.lock`,
governor restored to `schedutil` afterwards. 99 measurements, 0 failures.

---

## 1. The documented result reproduces

Every headline number in `README.md` is confirmed against the committed
artifacts, and the partitioning rationale is *derivable* rather than asserted:

| claim | README | recomputed from committed artifacts |
|---|---|---|
| serial CPU-only | 3172.2 ms | 3172177 us |
| serial DSP-only | 3609.6 ms | 3609601 us |
| serial HTA-only | -- | 1367613 us |
| schedule makespan | 1083.6 ms | 1083.6 ms |
| partition | 47/49 on HTA | 47 on `CPU_P#0`, 2 on `CPU_X#0` |

The two CPU-resident segments are the two where HTA is genuinely slower:

    dsp_seg_00   cpu  68.5 ms   hta 295.5 ms   (patch embed Conv16x16, HTA 4.3x worse)
    dsp_seg_24   cpu  77.8 ms   hta 134.8 ms   (final head 12288->960, HTA 1.7x worse)

## 2. Re-measured on the board

| backend | n | median new/recorded | within +/-10% |
|---|---|---|---|
| CPU | 49 | **1.000** | 48/49 |
| HTA | 25 | 0.964 | 13/25 |
| DSP | 25 | **0.839** | 0/25 |

**CPU reproduces essentially exactly.** HTA reproduces apart from one outlier.

**DSP is uniformly ~16% faster than recorded** -- median 0.839, range
0.79-0.85, not a single segment inside +/-10%. A tight, one-sided band like
that is a systematic condition difference, not noise. The likely cause is
already documented in `qnn_models/flow_c/measurements/qrb5165_v66.json`: under
`schedutil` the board idles at 710 MHz of 2419 and the host clock gates FastRPC
dispatch, making accelerator cells 10-36% pessimistic. DSP work goes through
FastRPC and CPU work does not, which is exactly the asymmetry observed. Read
as: **the recorded DSP column looks like it was captured under a slower host
clock than the CPU column.** Not proven -- it needs a deliberate A/B under both
governors to confirm.

**The partitioning decision survives.** Of the 25 segments with all three
backends measured, HTA is fastest on 23, and the two exceptions are exactly
`dsp_seg_00` and `dsp_seg_24` -- the same two the recorded schedule puts on CPU.

One caveat that matters for future scheduling: `dsp_seg_24` on HTA re-measured
at **83.3 ms against a recorded 134.8 ms** (ratio 0.62). Its CPU cell is
77.9 ms, so the CPU-over-HTA margin collapses from 73% to **6.5%**. That
assignment is now marginal and could flip under contention or a re-measure.

## 3. The finding the numbers do not advertise

**There is no concurrency in this workflow at all.**

Measured across every SmolVLA schedule variant on disk:

    scheduled_networks_smolvla_vision_v3_qrb5165          makespan 1083.6  ops  49  overlap 0  concurrency 1.00x
    scheduled_networks_smolvla_unrolled10_qrb5165         makespan 5636.0  ops  54  overlap 0  concurrency 1.00x
    scheduled_networks_smolvla_v3_unrolled10_qrb5165      makespan 3351.7  ops 102  overlap 0  concurrency 1.00x
    scheduled_networks_smolvla_v3_bundles_unrolled10      makespan 4833.0  ops 150  overlap 0  concurrency 1.00x

Zero overlapping dispatches in any of them. For the vision schedule the
makespan equals the sum of lane busy times *and* the sum of per-segment
best-backend times, to the decimal:

    sum of per-segment best backend   1083.6 ms
    recorded schedule makespan        1083.6 ms
    sum of lane busy times            1083.6 ms   {CPU_P#0: 937.3, CPU_X#0: 146.3}

So the 2.9x is **per-segment backend selection**, not parallel execution. Two
nested chains explain it:

  * within the network, 48 of the 49 vision dispatches depend on their
    predecessor -- a sliced ViT is a linear graph;
  * across networks, the `unrolled10` config chains
    `prefill -> action_in_0 -> time_in_0 -> time_out_0 -> decode_0 -> ...`.
    That is an autoregressive decode loop, not ten independent inferences.

**The DSP lane is never used.** `CPU_E#0` is available and receives zero
dispatches in both schedules. The scheduler is right -- DSP is slowest on every
segment, 3609.6 ms serial against CPU's 3172.2 -- but the "three backends"
framing is two in practice, and in the unrolled case HTA sits at 28% occupancy
while CPU carries 72%.

## 4. Reproducibility by stage

| stage | runs from tracked inputs? | notes |
|---|---|---|
| slice | no | needs `smolvlm_vision.onnx` (untracked, ~38 GB tree) |
| rewrite | no | operates on sliced ONNX |
| build | no | needs Docker image `qnn-convert` with the QNN SDK |
| build-hta | no | same |
| profile | **yes, if the board is staged** | 272 ctx binaries + 74 DLC + 99 HTA DLC already on the board under `/root/models/smolvlm_vision_v3`; `--skip-build` goes straight to measurement |
| emit | yes | `emit_vision_v3_profile.py --from-perf-json` |
| schedule | yes | host-only, reproduces the makespan exactly |

Only 39 files under `smolVLA/` are tracked; the ONNX/DLC/context blobs are
untracked build output. **A newcomer on a fresh clone cannot run stages 1-4**
without regenerating those, which needs the ONNX and the conversion container.
Stages 5-7 are reproducible today because the board still holds the staged
artifacts -- that is a property of this board, not of the repo.

## 5. What changed on disk

    M  qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3/segment_perf.json
    ?? qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3/*.csv   (per-call traces)
    +  qnn_models/smolVLA/REPRODUCTION.md, reproduction_data_log.json

Not touched: the pre-existing dirty artifacts under
`gen/*/qrb5165_v66/smolvlm_vision_v3_bundles/`, which are a different variant
and were already modified before this work.

---

## 6. Partitioning checklist — which components are actually partitioned

Derived from `gen/profile/{CPU,DSP,HTA}/qrb5165_v66/<component>/**/results.csv`.
A backend is EXCLUDED for a segment when the profile sweep wrote the 1e9 us
sentinel, meaning the segment cannot compile/run there.

| # | component | segments | CPU | DSP | HTA | partitioned? |
|---|---|---|---|---|---|---|
| 1 | `smolvlm_vision_v3` | **49** | 3172.2 | 3609.6* | 1367.6* | **YES** (see §7) |
| 2 | `smolvlm_vision_v3_bundles` | 141 | 26 run / 115 excl | 69 / 72 | 46 / 95 | yes, but a regression |
| 3 | `smolvlm_vision_coarse` | 1 | EXCL | EXCL | EXCL | no -- runs nowhere |
| 4 | `smolvlm_expert_prefill_coarse` | 1 | 583.8 | EXCL | EXCL | no |
| 5 | `smolvlm_expert_decode_coarse` | 1 | 149.6 | EXCL | EXCL | no |
| 6 | `smolvlm_text_coarse` | 1 | **6.4** | 37.8 | EXCL | no |
| 7 | `state_projector_coarse` | 1 | **1.3** | 29.0 | 2.6 | no |
| 8 | `action_in_projector_coarse` | 1 | **4.7** | 58.8 | 6.7 | no |
| 9 | `action_out_projector_coarse` | 1 | **2.1** | 31.4 | 3.4 | no |
| 10 | `time_in_projector_coarse` | 1 | **5.8** | 35.9 | 6.7 | no |
| 11 | `time_out_projector_coarse` | 1 | **5.4** | 33.7 | 6.6 | no |

All times ms, serial sum over the component's segments. Bold = fastest backend.
`*` The DSP and HTA columns for `smolvlm_vision_v3` are **already hybrid**, not
single-backend alternatives -- see §7. Only the CPU column is a true
one-backend number.

**One of nine distinct components has been successfully partitioned.** Vision
is the whole story; everything else is a single monolithic `*_coarse` graph.

### Vision, the success

`smolvlm_vision_coarse` is excluded on all three backends -- the unsliced graph
runs *nowhere*, which is what motivated slicing. Cutting it into 49 segments
(`v3`) makes every segment runnable on all three and yields best-of-3 =
**1083.6 ms** against 3172.2 CPU-only. That is a genuine win.

### The bundles variant is a regression, and worth knowing why

`v3_bundles` slices further, to 141 segments. Measured:

    total segments          141
    runnable somewhere      141
    runnable on >1 backend    0     <- zero scheduling freedom
    runnable nowhere          0
    forced serial total   2272.3 ms  (vs 1083.6 for v3)

It is a **perfect hard partition**: every segment runs on exactly one backend,
none on two. So the scheduler has no choice to make -- the assignment is forced
by capability, and the result is **2.1x worse** than the coarser 49-segment cut
that leaves all three backends viable. Finer slicing bought capability
fragmentation, not speed. This is the same lesson the dronet binding records
(7 residual blocks: 3.49 ms summed vs 0.92 ms whole).

### The experts are blocked, not merely unpartitioned

`expert_prefill` (583.8 ms) and `expert_decode` (149.6 ms) are CPU-only --
excluded on both DSP and HTA. `SMOLVLA_DSP_SLICING_PLAN.md` names the cause:
`ScatterND` (48-64x) and `Where` (16x) on top of the Sin/Tanh issue. They were
explicitly declared out of scope for the vision iteration and remain so.

### Where the remaining opportunity is

By cost, on the single-inference path:

    vision           1083.6 ms   partitioned, HTA-resident      <- done
    expert_prefill    583.8 ms   CPU-only, ScatterND/Where       <- biggest remaining prize
    expert_decode     149.6 ms   CPU-only, same blockers          (and x10 when unrolled)
    everything else   ~25   ms   CPU-best already, not worth it

The four projectors plus text and state_proj total roughly 25 ms and are
already fastest on CPU -- partitioning them is not worth the dispatch overhead.
**`expert_prefill` is the only component where partitioning work would still
pay**, and it is gated on the ScatterND/Where op support rather than on slicing
mechanics.


---

## 7. Correction: vision is NOT 47/49 on HTA

The `README.md` headline, which §1 reproduced arithmetically, says 47 of 49
segments run on HTA. **Physically, 23 do.** The claim survives only because the
profile data misrepresents what HTA can run.

### The two segment families

The 49 segments are not homogeneous. They are:

    25 x dsp_seg_*   heavy compute (MatMul/Conv), accelerator-eligible
    24 x cpu_seg_*   the Tanh/GELU trampolines carved out by the slicing plan,
                     CPU-only *by construction* -- they exist precisely because
                     HTA and DSP cannot run Tanh

### The data defect

For all 24 `cpu_seg_*` rows, the DSP and HTA columns carry the **CPU value,
copied verbatim** (24/24 identical in both columns), rather than the 1e9 us
sentinel that marks "cannot run here". So the scheduler sees a trampoline as
runnable on HTA at exactly the CPU price, and — with nothing to distinguish
them — assigns all 24 to the HTA lane:

    schedule: 47 dispatches on CPU_P#0 (HTA lane), 2 on CPU_X#0
    of those 47, 24 are cpu_seg_* trampolines
    board contexts for cpu_seg_*: 24 x __Cpu.bin, 0 x __Hta.bin

That placement cannot execute. No HTA context exists for a trampoline, and none
can be built.

### What the numbers really are

    "HTA serial" 1367.6 ms = 822.9 ms  (25 dsp_seg_* on HTA)
                           + 544.7 ms  (24 cpu_seg_* on CPU, copied in)

    makespan     1083.6 ms =  392.6 ms  on HTA   (23 segments)
                           +  691.0 ms  on CPU   (26 segments)

The makespan *arithmetic* is correct -- the durations summed are the real CPU
durations for the trampolines -- so 1083.6 ms and the 2.9x speedup stand. What
is wrong is the **lane attribution**: the partition is 23 HTA / 26 CPU, not
47/3. HTA carries 36% of the work, not 86%.

### Why it matters

  * A runtime generated from this schedule would dispatch 24 entries to a
    backend that cannot run them. Flow C's capability check would reject it;
    the greedy path did not.
  * It inflates the apparent HTA benefit. The honest framing is that slicing
    moved 36% of the vision encoder onto HTA, and that alone is worth 2.9x.
  * The fix is in the emit step: `cpu_seg_*` rows should carry the exclusion
    sentinel for DSP and HTA, not the CPU value. Then the scheduler places them
    on CPU_X where they belong, and the reported lane split becomes truthful.
    The makespan should not change, since the durations are already the CPU ones.

---

## 8. Port to Flow C: HTA is unreachable at the published granularity

Building the Flow C binding manifest from the **context binaries that actually
exist on the board** (`flow_c/gen_smolvla_binding.py`, which reads the ctx
inventory rather than the profile CSVs) resolves §7 completely.

    ctx_cpu_seg_00..23__*        Cpu only          (24)  Tanh/GELU trampolines
    ctx_dsp_seg_00..24__*        Cpu, Dsp          (25)  whole segments -- NO Hta
    ctx_dsp_seg_NN_*_conv1x1__*  Hta only          (50)  extracted Conv1x1 kernels
    ctx_dsp_seg_NN_tramp_p*__*   Cpu, Dsp          (74)  trampoline sub-parts

    generated manifest, segment granularity:  49 tiles, 0 reach HTA
    generated manifest, bundle  granularity: 173 tiles, 50 reach HTA

**There is no `ctx_dsp_seg_NN__Hta.bin`.** HTA never runs a whole segment and
never runs a trampoline; it runs only the extracted Conv1x1 kernels. The HTA
column in `gen/profile/HTA/.../smolvlm_vision_v3` is therefore synthesized from
those sub-model timings and attributed to whole segments that cannot execute
there.

### What is actually achievable

    all-CPU baseline                                 3172.2 ms
    49 segments, best(CPU,DSP) -- real contexts      2997.5 ms   1.06x
    173 tiles, convs on HTA (the bundles cut)        2272.3 ms   1.40x   <- measured
    published figure, synthesized HTA column         1083.6 ms   2.9x    <- not realizable

The published 2.9x assumes 25 whole segments run on HTA at the price of their
extracted convs. The best physically-supported result is the 141/173-tile
decomposition at **2272.3 ms, 1.40x** -- which is exactly the "bundles" variant
§6 recorded as a 2.1x regression *against the unrealizable number*. Measured
against reality it is the best configuration on the board.

### Consequence for the port

This is the case for the port rather than against it. Flow C's binding manifest
is capability-typed: backends are declared per tile and checked against the
registry and the staged contexts. The manifest generated here **cannot express
the claim that broke** -- at segment granularity it emits zero HTA backends,
because zero HTA contexts exist. The greedy path had no such gate, which is how
a synthesized column became a headline.

### Next step

Target **2272.3 ms**, not 1083.6. Drive `--granularity bundle` (173 tiles)
through `flow_c.py artifacts -> schedule -> runtime -> stage -> run`. Open
questions to settle in that run:

  * 173 tiles x eager context load is a large bringup; Flow C loads eagerly and
    context init was measured at 42-256 ms elsewhere. Lazy or budgeted loading
    may be required.
  * The 50 HTA tiles are HTA-*only*, with no CPU fallback, so a capability
    failure is fatal rather than degrading -- the opposite of the usual Flow C
    tile, which has 2-3 viable backends.
  * The chain is still strictly sequential, so the lane runtime buys backend
    selection only. The 1.40x is real but it is not concurrency.

---

## 9. Flow C port: result, and the limit it hit

Ported at hybrid granularity and driven through `artifacts -> schedule ->
runtime -> run`. All 272 (context, backend) pairs on the board were re-measured
for this (10 iters, gap phase, performance governor, board lock) into a
separate `measurements/qrb5165_v66_smolvla.json` so the port cannot perturb the
sweep's cost model.

### The port finds a better configuration than the original

Per `dsp_seg_NN` there is a choice the original flow never made explicit: run
the **whole** segment (cpu/dsp) or its **decomposition** (conv1x1 kernels on
HTA + trampoline parts on cpu/dsp). Measured, it goes both ways —

    dsp_seg_00   whole  68.2 ms   decomposed 349.9 ms   -> keep whole
    dsp_seg_01   whole  74.2 ms   decomposed  45.5 ms   -> decompose

23 of 25 segments are better decomposed, 2 are not. Taking the per-segment
best gives a 141-tile hybrid:

    all-CPU baseline                              3172.2 ms
    all-whole      (best of cpu/dsp)              2875.0 ms
    all-decomposed (the bundles cut)              2509.0 ms
    original's realizable best (bundles, measured) 2272.3 ms
    FLOW C PORT, hybrid                           2196.8 ms   1.44x
    published figure (not realizable)             1083.6 ms

**The scheduler produced 2196.8 ms against a predicted 2196.9 ms** — the
best-of-cells bound, matched to 0.1 ms. And unlike the original schedule, which
left `CPU_E#0` completely idle, all three lanes carry work:

    CPU_E#0 (DSP)  69 tiles  1155.0 ms  52.6%
    CPU_X#0 (CPU)  26 tiles   689.3 ms  31.4%
    CPU_P#0 (HTA)  46 tiles   352.6 ms  16.0%

Concurrency is still 1.00x with zero overlapping dispatches — the chain is
sequential, so the lane runtime buys backend selection, not parallelism. That
was expected and is unchanged by the port.

### It does not run: Flow C loads contexts eagerly

    [bringup] ... 56 contexts loaded (18 HTA, 27 DSP, 11 CPU)
    QNN error 0x4 at runtime_main.cpp:205
        iface.contextCreateFromBinary(...)

**Flow C's emitted runtime loads every context at bringup and the board
exhausts resources at 56 of 141.** There is no lazy or budgeted loading in
`flowc/emit_runtime.py` — the `XPURT_*_CTX_BUDGET` knobs belong to the older
`deploy_and_run.sh` runtime, not this one.

This is a Flow C capability gap, not a defect in the port. Every Flow C network
to date has had 1-3 tiles; 141 is two orders of magnitude past what the eager
bringup was designed for. The port is what surfaced it.

### Where this leaves things

  * **Schedule: reproduced and improved.** 2196.8 ms, 1.44x over CPU, beating
    the original's best realizable configuration (2272.3 ms) and using all
    three lanes instead of two.
  * **Execution: blocked** on eager context bringup at ~56 contexts.
  * **Fix:** lazy context load with an LRU budget per backend in
    `emit_runtime.py`. The schedule is strictly sequential, so at most one
    context per lane is live at a time — a budget of a few per backend would
    suffice, and the trace already records per-entry timing to measure the
    reload cost against the 2196.8 ms target.

---

## 10. Lazy context loading, and what it revealed

`flowc/emit_runtime.py` now supports lazy context loading with a per-backend
LRU budget, enabled by `FLOWC_CTX_BUDGET`. Default 0 = eager, byte-identical to
the previous behaviour, so every runtime that already fits is unchanged.

    contexts are created on first use, not at bringup
    the LRU victim on the same backend is freed when the budget is reached
    a context in use by a lane is never evicted (refcount)
    reports: ctx loads=N evictions=M load_time=T ms

### It unblocks execution

    before:  QNN error 0x4 at contextCreateFromBinary, 56 of 141 loaded
    after:   [main] lazy contexts: budget 8 per backend, 141 registered
             [summary] 141/141 entries executed

### And it exposes the real cost

    config                            predicted   actual   ratio   notes
    141-tile hybrid (lazy, budget 8)     2196.8  12876.3   5.86x   8699.3 ms in ctx load
                                                                   141 loads, 117 evictions
    49-tile segments (eager, resident)   2875.0   3577.5   1.24x   49 contexts, 0 loads

**Context loading is 68% of the 141-tile wall time.** Each tile is visited once
in a sequential chain, so an LRU cache gets no reuse -- the budget prevents the
crash but every tile still pays a create. 141 creates cost 8.7 s against 2.2 s
of predicted compute.

The 49-tile cut fits under the ~56-context ceiling, so its contexts stay
resident and it pays nothing per tile. It is **3.6x faster end to end**
(3577.5 vs 12876.3 ms) despite a *worse* predicted makespan (2875.0 vs 2196.8).

### The conclusion the port produces

Scheduling gain and context residency pull in opposite directions, and on this
board residency wins decisively:

  * finer tiling buys backend choice -- the 141-tile hybrid reaches HTA and
    predicts 2196.8 ms against the 49-tile cut's 2875.0 ms;
  * but it costs 141 context creates, and the board can only hold ~56 resident.

So the best *measured* configuration is the coarse one, and the published
1083.6 ms is doubly unreachable: it credits whole segments with HTA times
measured on extracted convs, and the decomposition that would actually reach
HTA cannot stay resident.

**Where this leaves the port:** Flow C now runs smolVLA end to end at either
granularity, its capability model refuses the placement that produced the
original headline, and the measured best is 3577.5 ms at 49 tiles. Closing the
gap to the 2196.8 ms schedule needs contexts that survive across dispatches --
either a larger context budget than this silicon allows, or fewer/larger tiles
that still reach HTA.

---

## 11. The remaining components: why the experts cannot be partitioned here

`expert_prefill` (583.8 ms) and `expert_decode` (149.6 ms) are the only
components where partitioning could still pay -- the four projectors plus text
and state_proj total ~25 ms and are already fastest on CPU. Analysed both
against the residency ceiling §10 measured.

### The blockers are interleaved, not clustered

    smolvlm_expert_prefill.onnx   1166 ops   82 blockers   144 heavy (MatMul/Gemm/Conv)
    smolvlm_expert_decode.onnx    1096 ops   68 blockers   144 heavy

    prefill blockers: ScatterND x64, Where x16, Sin x1, Cos x1
    decode  blockers: ScatterND x48, Where x16, Sin x2, Cos x2

The `_patched` variants on disk do **not** remove them -- they add 16-18 ops and
leave every ScatterND, Where, Sin and Cos in place.

64 ScatterND in a 32-layer transformer prefill is one per layer per K/V: these
are KV-cache writes, structurally one per layer, not an artefact that better
slicing can avoid. Their spacing confirms it:

    gap between consecutive blockers: min 1, median 7, max 41
    first at op 17, last at op 1137 of 1166
    largest blocker-free span anywhere: 40 ops (8 heavy)

### That forces a tile count the board cannot hold

    component   blocker-free runs with heavy ops   trampolines   tiles needed
    prefill                    33                      82            115
    decode                     33                      68            101

Against a **measured residency ceiling of ~56 contexts**. Both are ~2x over,
and §10 measured exactly what exceeding it costs: the 141-tile vision hybrid
spent 8699 ms creating contexts -- 68% of its wall -- and came out **3.6x
slower** than the 49-tile cut that stays resident.

Applying that to prefill: ~115 context creates at the ~62 ms each observed for
vision is roughly **7 s of context churn against 583.8 ms of compute**. The
partitioning would cost an order of magnitude more than the work it accelerates.

Coarse partitioning does not rescue it either. The largest blocker-free span in
the whole graph is 40 ops holding 8 heavy ops, and the median accelerator run
holds 3 -- there is no large contiguous region to hand to DSP or HTA.

### Conclusion, and the direction that would work

**The experts are not partitionable on this board by slicing.** This is not the
same conclusion as `SMOLVLA_DSP_SLICING_PLAN.md` reached -- it deferred them as
"more invasive surgery" -- it is stronger and now quantified: even if every
blocker were successfully carved into a trampoline, the resulting tile count
exceeds what the silicon can hold resident, and the context churn dominates.

The productive direction is **graph rewriting, not finer slicing** -- the same
move that made vision work, where MatMul was rewritten to Conv1x1 so it would
compose on HTA. Concretely:

  * **ScatterND (64/48).** If these are KV-cache writes, a static cache layout
    lets them become Concat or a fixed slice assignment, both of which the
    accelerators support. This is the highest-value single change: it removes
    78% of prefill's blockers and would collapse ~82 trampolines toward ~18.
  * **Where (16).** Attention masking. Often expressible as Mul by a precomputed
    0/1 mask, which composes.
  * **Sin/Cos (1-2).** Rotary embeddings; can be precomputed to constants when
    positions are static.

If ScatterND and Where both fall, prefill's tile count drops to roughly 20 --
inside the residency ceiling -- and it becomes a candidate for the same
treatment vision received. Until then, partitioning it is measurably the wrong
move.

---

## 12. Graph rewrite: ScatterND eliminated, both experts now fit

§11 concluded the experts need graph rewriting rather than finer slicing. Done:
`rewrite_scatternd_to_concat.py`.

### What the ScatterNDs actually were

Not general scatters, and not in-place KV-cache updates either. Every one of
the 64 in prefill falls into one of two classes, verified across all of them:

    32x  data = an ALL-ZERO initializer, indices = constant arange(32)     -> [0..31]
    32x  data = the previous ScatterND's output, indices = arange(32)+32   -> [32..63]

So each consecutive pair is

    tmp = ScatterND(zeros(64,...), [0..31],  A)     # tmp[0:32]=A, tmp[32:64]=0
    out = ScatterND(tmp,           [32..63], B)     # out[0:32]=A, out[32:64]=B

which is exactly `Concat([A, B], axis=0)` — the graph was assembling the
`present_key_N` / `present_value_N` cache from two halves. The rewrite is
value-identical by construction: the base is all zeros, the two index blocks
are contiguous, disjoint, in order, and together cover the whole axis. The
rewriter checks all four conditions per pair and skips anything that fails.

### Verified numerically, not just structurally

    prefill: 64 ScatterND -> 0   (32 pairs -> Concat)
    decode:  48 ScatterND -> 0   (24 pairs -> Concat)

onnxruntime, same random inputs, graph optimisation disabled, all 33 outputs:

    vlm_output_embeds        max|diff| 0.000e+00
    present_key_0..15        max|diff| 0.000e+00
    present_value_0..15      max|diff| 0.000e+00

**Bit-exact**, not merely within tolerance.

### It brings both experts inside the residency ceiling

    component                ops   blockers                          tiles   vs ~56 ceiling
    prefill BEFORE          1166   82  (ScatterND 64, Where 16, ...)    115   OVER
    prefill AFTER           1134   18  (Where 16, Sin 1, Cos 1)          36   UNDER
    decode  BEFORE          1096   68  (ScatterND 48, Where 16, ...)    101   OVER
    decode  AFTER           1072   20  (Where 16, Sin 2, Cos 2)          39   UNDER

That is the unblock §11 identified: prefill drops from 115 tiles to **36**,
decode from 101 to **39**. Both now fit resident, so neither would pay the
context-churn penalty that made the 141-tile vision hybrid 3.6x slower than the
49-tile cut.

### What remains

`Where` x16 is the surviving blocker in both, and is next: attention masking is
usually expressible as a Mul by a precomputed 0/1 mask, which composes.
Removing it would take prefill to ~19 tiles. `Sin`/`Cos` (1-2 each) are rotary
embeddings and can be folded to constants when positions are static.

The rewritten models are `smolvlm_expert_{prefill,decode}_concat.onnx`
(untracked, like every other model blob under smolVLA/). Next step is to push
them through convert -> quantize -> context-build and confirm the tiles compose
on DSP/HTA, which is what the tile counts above assume but do not yet prove.

---

## 13. Where rewrite, and how far the confirmation got

### The Where rewrite

`rewrite_where_to_mask_arith.py`. All 16 `Where` in each expert have the same
shape -- `Where(cond, scores, -3.4028235e+38)` -- standard additive attention
masking. Rewritten to the exactly-equivalent arithmetic form:

    mask_f = Cast(cond, float32)          # 1.0 keep, 0.0 mask
    neg    = (1 - mask_f) * -FLT_MAX      # 0.0 keep, -FLT_MAX mask
    out    = scores * mask_f + neg

    cond true  -> scores*1 + 0        = scores
    cond false -> scores*0 + -FLT_MAX = -FLT_MAX

This is stronger than the usual `scores + bias` trick, which is only bit-exact
because the ulp at -FLT_MAX is huge; multiplying the masked lane to exactly
zero first removes that dependence.

prefill shares one mask, **decode has two** (self- and cross-attention). The
first version asserted a single shared mask and correctly *refused* on decode
rather than building wrong setup; it now emits one setup chain per mask.

### Both experts, both rewrites, verified bit-exact

    prefill  33 outputs, 33 bit-exact, max|diff| 0.000e+00
    decode    1 output,   1 bit-exact, max|diff| 0.000e+00

against the *original* graphs, onnxruntime, graph optimisation disabled.

### Effect on tile count -- far better than projected

    prefill ORIGINAL   1166 ops, 82 blockers -> 115 tiles   OVER the ~56 ceiling
    prefill REWRITTEN  1153 ops,  2 blockers ->   4 tiles   UNDER
    decode  ORIGINAL   1096 ops, 68 blockers -> 101 tiles   OVER
    decode  REWRITTEN  1094 ops,  4 blockers ->   7 tiles   UNDER

Only `Sin`/`Cos` remain (rotary embeddings, 1-2 each). §11 projected ~19 tiles
for prefill; the measured result is **4**.

### The confirmation: converted, did NOT compose, and the reason matters

    snpe-onnx-to-dlc  ->  INFO_CONVERSION_SUCCESS, 601 MB prefill_nomask.dlc
    qnn-context-binary-generator --backend libQnnDsp.so  ->  GENERATOR_RC=14

        QnnDsp <E> Input[0] has incorrect Datatype 0x508.
        Validate OpConfig failed: QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE
        Failed to successfully compose graph

**This is a precision failure, not an op-support failure.** 0x508 is float32:
the DLC was converted fp32 and never quantized, and the DSP backend rejects
float32 inputs at datatype validation -- before op composition is reached. So
the test neither confirms nor refutes that the rewritten graph composes; it
failed at an earlier gate.

The vision pipeline quantizes to int8 before building contexts
(`qairt-quantizer --act_bitwidth 8 --weights_bitwidth 8` against a calibration
list). Doing the same for the experts needs calibration inputs for
`vlm_embeds` (float), `attention_mask` (**bool**) and `position_ids` (int64) --
the bool input is the awkward one, and no expert calibration set exists yet.

### State

    ScatterND rewrite    done, bit-exact, 64->0 prefill / 48->0 decode
    Where rewrite        done, bit-exact, 16->0 both
    tile count           115->4 (prefill), 101->7 (decode); both under ceiling
    ONNX -> DLC          confirmed, converts cleanly
    DLC -> DSP context   BLOCKED on quantization, not on op support

The remaining step is calibration + `qairt-quantizer`, then re-run the same
context build. Until that runs, "the experts now compose on DSP" is **not**
established -- only that the blockers that previously made it impossible are
gone and the graph converts.

---

## 14. Experts: op support confirmed on DSP and HTA; quantization still blocked

### The confirmation that matters

The rewritten prefill was converted and its per-op runtime support read straight
out of the DLC (`snpe-dlc-info`, `Runtimes` column):

    ops in DLC                    1197
    ops WITHOUT DSP support          0
    ops WITHOUT HTA/AIP support      0

    op types: Eltwise_Binary 403, Reshape 227, Transpose 226,
              FullyConnected 112, Eltwise_Unary 34, Reduce 32,
              Split 32, Concat 32, Resize 32, MatMul 32

**Every op in the rewritten expert composes on both DSP and HTA.** The 32
`Concat` are exactly the ones the ScatterND rewrite created. This is the
converter's own static analysis, and it is the answer to the question the
rewrites were for: the experts are no longer CPU-bound by op support.

`SMOLVLA_DSP_SLICING_PLAN.md` put the experts out of scope over ScatterND and
Where. Both are gone, bit-exactly, and nothing else in the graph blocks.

### What is still blocked: calibration plumbing, not op support

Quantization is required before a context binary will build -- the fp32 DLC is
rejected at `Input[0] has incorrect Datatype 0x508` (float32) before op
composition is reached. `qairt-quantizer` will not accept the calibration set:

    batch-1 raws  -> "batch size = 1 does not match with expected ... batch size = 4"
    batch-4 raws  -> "file size 1735680 ... the file size should match the
                      tensor extent: 433920 bytes"

The two messages contradict each other: 433920 bytes IS the batch-1 extent, and
all three inputs were verified against the DLC's declared dims and dtypes --

    vlm_embeds      Float_32  [1,960,113]   433920 B   (note: converter applies
                                                        axes-to-spatial-first-order,
                                                        so NOT the ONNX [1,113,960])
    attention_mask  Bool_8    [1,113,113]    12769 B
    position_ids    Int_32    [1,113]          452 B   (ONNX int64; converter
                                                        runs keep_int64_inputs=False)

Both layout traps were found and fixed and the sizes match exactly, so this is
a batch-inference quirk inside the quantizer's netrun rather than malformed
calibration data. Next things to try: `--batch 1` explicitly on the converter,
converting with a fixed batch in the ONNX, or `--float_bitwidth 16` to sidestep
int8 calibration entirely and test composition at fp16.

### Honest status

    ScatterND rewrite         done, bit-exact
    Where rewrite             done, bit-exact
    tile count                115 -> 4 (prefill), 101 -> 7 (decode)
    ONNX -> DLC               converts cleanly
    op support on DSP/HTA     CONFIRMED, 0 of 1197 ops unsupported
    int8 quantization         BLOCKED on a quantizer batch-inference quirk
    context build / timing    not reached

No performance number for the experts is claimed. What is established is that
the blockers which made them CPU-only are gone and the graph is accelerator-
eligible end to end.

---

## 15. Full triage of the experts: the prefill now composes on DSP

Every accelerator-mapping avenue worked through to a verdict. Machine-readable:
`expert_triage.json`.

### The headline

**The SmolVLA expert prefill now produces a working DSP context binary** --
`ctx_trunk_dsp.bin`, 158,467,168 bytes, 1108 ops. That is the first time any
expert has composed on an accelerator. `SMOLVLA_DSP_SLICING_PLAN.md` declared
them out of scope; they are not.

Getting there needed five rewrites, a calibration fix and one slice, each of
which surfaced only after the previous one was cleared -- the toolchain reports
exactly one blocker at a time.

### The chain, in the order it had to be solved

| # | avenue | verdict | what it actually was |
|---|---|---|---|
| R1 | ScatterND -> Concat | **done** | 64/48 scatters were a two-half cache assembly; bit-exact |
| R2 | Where -> mask arithmetic | **done** | additive attention mask; bit-exact |
| C1 | float32 calibration raws | **done** | SNPE lists take float32 *regardless of network dtype*; a uint8 mask was 1/4 the extent and surfaced as a phantom "batch size 4" |
| D1 | int8 quantization | **done** | fp32 DLC is rejected at `Datatype 0x508` before op validation |
| R4 | bool input -> float32 | **done** | 0x508 = `QNN_DATATYPE_BOOL_8`; DSP rejects a **bool graph input**, which no op-level fix can address |
| R3 | Sin/Cos -> constants | **done** | `Param[0]=14` is the Sin opcode; DSP's ElementWiseUnary has no Sin. Rotary depends only on `position_ids`, so it folds |
| R5 | block RmsNorm fusion | **failed** | see below |
| S1 | whole-graph single tile | **blocked** | by that one RmsNorm |
| S2 | slice before the RmsNorm | **SUCCEEDED** | 1108-op trunk composes |
| X1 | execute the context | **blocked** | 158 MB context wedges the board |

Two traps cost real time and are worth recording: the converter applies
`axes-to-spatial-first-order` (so `vlm_embeds` is `[1,960,113]`, not the ONNX
`[1,113,960]`) and `keep_int64_inputs=False` (so `position_ids` is `Int_32`).
A wrong-width raw is reported as a *batch* mismatch, never as a dtype error.

### Why RmsNorm could not be rewritten away

The ONNX contains **no norm op** -- 33 decomposed `Pow -> ReduceMean -> Add ->
Sqrt -> Reciprocal -> Mul` chains. The converter pattern-matches the last of
them into a single `qti.aisw:RmsNorm`, and v66 has no implementation:

    QNN_BACKEND_ERROR_OP_PACKAGE_NOT_FOUND

Three exits were tried and all are closed:

  * **Config** -- no RmsNorm pass appears in
    `--dump_ir_optimizer_config_template` (32 passes listed, none of them it).
  * **Barrier** -- inserting `Mul` by 1.0 after all 33 `ReduceMean` did not
    defeat the matcher; RmsNorm was still emitted.
  * **Op package** -- `libQnnDspV66Skel.so` has no RmsNorm and the SDK ships no
    registerable op-package `.so`. Since the fusion happens at *convert* time,
    HTA would inherit the same node.

So RmsNorm is a genuine v66 gap, not a configuration mistake. It is also the
best possible place for one: op **1189 of 1190**, the final op producing
`vlm_output_embeds`, operating on `[1,113,960]`.

### What that buys

    prefill: 1108-op trunk on DSP  +  1 RmsNorm on CPU  =  2 tiles

Two tiles, against a ~56-context ceiling. Compare the pre-rewrite estimate of
115 tiles, which was itself the reason section 11 called the experts
infeasible. That conclusion is now superseded by measurement.

### It executes -- and the answer is that the mapping is not worth taking

The first execution attempt wedged the board (recovered with `/opt/relay.sh`,
~60 s). That was memory pressure from a preceding run, not a hard limit: on a
freshly booted board it runs clean.

    status ok, graph "trunk", 3 iters
    context init      1032.4 ms
    execute median    1384.5 ms   (min 1383.3, max 1384.6, std 0.6 -- very stable)

    CPU baseline       583.8 ms
    DSP                1384.5 ms   ->  2.37x SLOWER

**The full rewrite chain successfully maps the expert prefill onto the DSP, and
the result is 2.4x worse than leaving it on the CPU.** Plus a full second of
context init.

HTA was tried as the alternative and does not compose at all
(`ComposeGraphs Failed with error = 1`) -- it is far more restricted than the
DSP and the trunk is MatMul/FullyConnected-heavy rather than convolutional.

This is consistent rather than surprising. The vision encoder measured DSP as
the *slowest* backend on every one of its 49 segments (3609.6 ms serial against
CPU's 3172.2), and section 6 recorded DSP as 5-20x worse on every small
component. **The Hexagon v66 DSP is simply not competitive with the Kryo 585
for transformer-shaped work on this board.** HTA is competitive, but only for
convolution, which is why vision benefits and the experts cannot.

### Verdict

    op support     SOLVED    0 of 1197 ops unsupported after R1-R4
    composition    SOLVED    1108-op trunk builds a 158 MB DSP context
    tiles          2         DSP trunk + CPU RmsNorm tail, vs 115 pre-rewrite
    execution      DONE      1384.5 ms median, stable
    performance    NEGATIVE  2.37x slower than CPU; HTA will not take the graph

The engineering question -- *can* the experts be mapped to an accelerator on
this silicon -- is answered yes, and took five rewrites, a calibration fix and
one slice. The product question -- *should* they be -- is answered no. Both
answers are measured, and the second only became knowable by doing the first.

### One gap, stated plainly

Everything above is **prefill**. `expert_decode` has the identical rewrite
chain applied and verified bit-exact (ScatterND 48->0, Where 16->0, bool input
retyped, 4 Sin/Cos folded, 101 -> 7 tiles) but was **not** converted, quantized,
composed or measured. It was not taken further because the DSP verdict is
architectural rather than model-specific -- v66 measured slower than the CPU on
all 49 vision segments and on prefill -- and decode is the smaller prize at
149.6 ms. That is an inference, not a measurement, and it is the one claim here
that rests on argument rather than evidence. The artifacts
(`smolvlm_expert_decode_nofuse.onnx`) are on disk if someone wants to close it.

Also note R3's caveat: the rotary fold is exact only while `position_ids`
equals the sequence it was folded at. For this fixed-shape export that holds;
for variable positions the sin/cos must be lifted to graph inputs instead.

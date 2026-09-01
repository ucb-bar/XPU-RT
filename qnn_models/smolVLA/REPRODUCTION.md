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

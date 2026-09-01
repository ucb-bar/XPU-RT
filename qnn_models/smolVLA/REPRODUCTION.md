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

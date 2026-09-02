# Operator sweep — QRB5165 (SM8250, Hexagon v66)

What it costs to run each operator in this repo's model family on each backend,
at each precision, across a structured size ladder — and, separately, **how much
of that cost is arithmetic and how much is the runtime charging you to dispatch
at all**.

That second number is the one that decides placements on this board. HTA's
per-dispatch overhead is ~543 us warm and ~2470 us cold *independent of the
work*, so an operator only belongs there if the CPU would need longer than that.
A sweep that reported only total time would hide it.

## Why single-op models

A network's time is a sum over ops plus a per-dispatch charge, and those are not
separable after the fact. Sweeping ONE op over a geometric size ladder makes
them separable: fit `t = overhead + macs/throughput` and the intercept **is** the
dispatch cost. Ladders span from 64 MACs upward for exactly this reason — the
two terms are only identifiable across decades.

## Layout

    opzoo.py     18 parametrised single-op ONNX builders. FAMILIES records which
                 model each kind comes from; nothing here is invented.
    grid.py      the size ladders. One dimension varied at a time against a
                 shared nominal, so series intersect and a heatmap has axes.
    genmodels.py runs opzoo inside the converter container to emit the ONNX and
                 the calibration raws.
    sweep.py     gen -> build -> measure -> fit. Resumable; every finished row is
                 appended to results.jsonl and skipped on re-run.
    plot_opsweep.py   placement heatmap, crossover heatmap, the same map with
                 dispatch subtracted, overhead bars, size ladders.

The host needs only python3 + numpy: onnx and the QAIRT converters live in the
`qnn-convert` container, the runtime lives on the board.

## Run it

    python3 sweep.py --all                     # ~185 points x 7 lanes
    python3 sweep.py --only measure --ops conv2d,linear
    python3 sweep.py --only fit                # re-fit, no board needed
    python3 plot_opsweep.py --out ../../plots

`--ops` and `--limit` cut it down; the board is the slow part, and the sweep can
be interrupted at any point and resumed.

## Things this encodes that cost real time to learn

* **Calibration raws must be float32 for every input**, whatever dtype the DLC
  declares. A uint8 raw is 1/4 the expected extent and `qairt-quantizer` reports
  it as a bogus "batch size 4" mismatch, never a dtype error.
* **Conv models are built natively NCHW and converted WITHOUT
  `--preserve_io layout`**, so the converter declares the input NHWC and emits no
  layout ops. Wrapping a conv in a rank-3 `[B,M,K]` dance is what makes the v66
  DSP refuse to finalize a graph — at *any* size, down to a single layer.
* **A discard run precedes the timed ones.** The first `profile_seg` call on a
  fresh context pays bringup that belongs to neither phase; without it the warm
  measurement (taken first) absorbs it and can read *higher* than the cold one.
* **One `profile_seg` call yields both phases.** It already runs a back-to-back
  loop *and* a gapped loop, so `median_us` is the warm number and
  `gap_median_us` (at `--gap-us 3000`) is the cold one. Timing it twice, as the
  first version did, doubled board time for nothing. The gap is power collapse,
  and it only bites dispatches shorter than about a millisecond — so it changes
  small-op numbers by ~4.7x and large-op numbers by nothing.
* **Warm is a median, and the spread is kept.** CPU and DSP are tight (sd well
  under 10 us), but HTA and GPU are *bimodal* back-to-back — they power-collapse
  part way through even a zero-gap loop, so repeat passes over the same point
  scatter by 4x (relu[64] on HTA: 541 us on one pass, 2504 on the next) and sd
  runs ~900 us. The row therefore carries `warm_us` (median, the number a packed
  pipeline sees), `warm_min_us` (the resident floor) and `warm_std_us` (the
  evidence for how much to trust it).
* **DSP and HTA are never asked to take a float DLC.** They reject it outright
  (`Input[0] has incorrect Datatype 0x508`), so those lanes are not attempted and
  the grid records why instead of emitting a blank cell.
* **`linear` and `conv1x1` are swept at matched arithmetic on purpose.** QNN's
  FullyConnected kernel measured ~13x slower than its own Conv2d kernel for the
  same MACs on HTA and DSP; the sweep should reproduce that, and it is the single
  highest-leverage mapping choice on this board.

## Two backend limits the sweep found on its own

* **The Adreno 650 backend rejects quantized tensors outright** — every int8
  point fails at `GPU_ERROR_INVALID_TYPE(10012)` -> `OpPackage (qti.aisw)
  validation failure`, before any op-specific check. The `gpu/int8` lane is
  therefore empty by construction, which is why nothing in this repo has ever
  run int8 on the GPU.
* **HTA has a tensor-size ceiling**, not just an op whitelist: `relu` composes
  up to 262144 elements and fails at 1048576 with `Fail to prepare graph m in
  HTA backend`. Ops it nominally supports still fall off the map at size.

## What the full sweep found

**The dispatch floor is the whole story.** Median fitted intercept per lane:

    cpu/fp32       2.0 us warm      144.6 us cold   (72x)
    cpu/int8      23.4 us warm      192.8 us cold   (8.3x)
    dsp/int8     425.3 us warm      625.8 us cold   (1.5x)
    hta/int8    1345.1 us warm     2127.0 us cold   (1.6x)
    gpu/fp16    2307.9 us warm     2801.9 us cold   (1.2x)

An accelerator has to find ~400 us (DSP) or ~1.3 ms (HTA) of arithmetic before
it breaks even, and most single ops in these models are nowhere near that.  At
each op's largest measured size, against the **best** CPU lane:

    4.69x  conv2d            hta   38.6 ms ->  8.2 ms
    3.30x  conv2d            dsp   11.9 ms ->  3.6 ms
    3.21x  layernorm         dsp  161.1 ms -> 50.2 ms
    1.59x  layernorm         gpu  161.1 ms -> 101.2 ms   <- the GPU's only win
    1.53x  concat_c          hta   21.9 ms -> 14.3 ms
    1.41x  conv1x1           hta    7.2 ms ->  5.1 ms
    1.22x  add               dsp   40.5 ms -> 33.3 ms
    1.13x  elu               dsp   31.6 ms -> 28.0 ms
    1.10x  conv1x1           dsp    1.8 ms ->  1.7 ms
    1.01x  depthwise_conv2d  hta    5.7 ms ->  5.6 ms

Compare against the *best* CPU lane, not against cpu/int8.  They differ enough
to invent wins that are not there: int8 `elu` on the CPU takes 197 ms where
fp32 `elu` takes 31 ms, so a DSP number of 28 ms reads as 7.1x when what you
would actually run makes it 1.1x.  `--only coverage` uses the best lane.

**Conv1x1 vs FullyConnected, at matched arithmetic.** The single mapping choice
worth making, with a number per backend:

    hta/int8   linear is 45.3x the time of conv1x1 (median), up to 127.8x
    dsp/int8   linear is  1.3x            (median), up to  13.5x
    gpu/fp16   linear is  1.2x
    cpu/int8   linear is  0.8x  -- on the CPU the mapping is a slight LOSS

So it is an accelerator-side rewrite, not a universal one, and it matters far
more on HTA than the earlier single-shape measurement suggested.

**Precision.** int8 is the only precision DSP and HTA accept; the GPU is the
reverse.  On the CPU int8 beats fp32 for the heavy ops (conv2d, linear,
matmul_dyn, conv1x1) and *loses* for cheap elementwise ones, where the
quantize/dequantize is the work -- `elu` is the extreme case at 6.3x worse.

## Ladder length is a result, not a setting

Several conclusions above only appeared after the ladders were extended, and
`sweep.py --only coverage` exists so that is checked rather than eyeballed.

The test is the **local** slope at the top of each ladder, not a global fit: the
CPU's own throughput degrades at large sizes, so a global fit hides crossovers.
The ratio tends to `g_acc / g_cpu`, and that asymptote decides everything.

* Extending `layernorm` rows to 16384 moved DSP from 1.4x to **3.21x**, and
  surfaced the GPU's only win in the entire sweep at 1.59x.
* `concat_c` topped out at 0.23 MMAC against conv2d's 9664 -- a 42,000x spread.
  Extended to 16.8 MMAC it went from 0.13x to **1.53x** on HTA.
* `mul` on HTA was the one crossover the grid provably failed to reach (18.3
  MMAC predicted, 12.6 swept).  Extended to 33.5M it reaches 0.95x and its
  asymptote is 0.99 -- it creeps toward parity and never crosses.
* `transpose`, `maxpool2d`, `avgpool_global` are the counter-case: their rows
  keep rising because the fixed overhead amortises, but the local slope stays
  4-5x worse than the CPU's, so they asymptote at 0.16-0.50.  A 4-point local
  fit on the old short `avgpool_global` ladder predicted a crossover at 6.7
  MMAC; extending to 16.8 MMAC disproved it.  Short ladders make confident
  extrapolations and they are not reliable.

## Two backend limits the sweep found on its own

* **The Adreno 650 backend rejects quantized tensors outright** — every int8
  point fails at `GPU_ERROR_INVALID_TYPE(10012)` -> `OpPackage (qti.aisw)
  validation failure`, before any op-specific check. The `gpu/int8` lane is
  therefore empty by construction, which is why nothing in this repo has ever
  run int8 on the GPU.
* **HTA has a tensor-size ceiling**, not just an op whitelist: `relu` composes
  up to 262144 elements and fails at 1048576 with `Fail to prepare graph m in
  HTA backend`. Ops it nominally supports still fall off the map at size.

## What the full sweep found

1267 distinct lanes, 1021 measured, 246 refused by a backend.

**The dispatch floor is the whole story.** Median fitted intercept per lane:

    cpu/fp32       2.0 us warm      144.6 us cold   (72x)
    cpu/int8      23.4 us warm      192.8 us cold   (8.3x)
    dsp/int8     425.3 us warm      625.8 us cold   (1.5x)
    hta/int8    1345.1 us warm     2127.0 us cold   (1.6x)
    gpu/fp16    2307.9 us warm     2801.9 us cold   (1.2x)

An accelerator has to find ~400 us (DSP) or ~1.3 ms (HTA) of arithmetic before
it breaks even, and almost nothing in these models is that big per op. **24 of
659 accelerator lanes beat the CPU at all**, and every one of the top eight is
`conv2d`; the best is 4.69x (conv2d, 1024 channels, HTA: 38.6 ms -> 8.2 ms).
The aggregate placement map has no green cell, because a median over sizes
averages the large-size wins away — read the crossover map for placement.

**Conv1x1 vs FullyConnected, at matched arithmetic.** This is the single
mapping choice worth making, and the sweep puts numbers on it per backend:

    hta/int8   linear is 45.3x the time of conv1x1 (median), up to 127.8x
    dsp/int8   linear is  1.3x            (median), up to  13.5x
    gpu/fp16   linear is  1.2x
    cpu/int8   linear is  0.8x  -- on the CPU the mapping is a slight LOSS

So it is an accelerator-side rewrite, not a universal one, and it matters far
more on HTA than the earlier single-shape measurement suggested. Rewriting a
FullyConnected as a 1x1 convolution on the CPU makes things marginally worse.

**Precision.** int8 is the only precision the DSP and HTA accept at all; the GPU
is the reverse, taking fp16/fp32 and rejecting int8 outright. On the CPU, int8
beats fp32 for the heavy ops (conv2d, linear, matmul_dyn, conv1x1) and *loses*
for the cheap elementwise ones, where the quantize/dequantize is the work.

## Compose failures are results

A backend that refuses an op is recorded with its verbatim validator message,
not dropped. "HTA has no two-dynamic-operand MatMul at any rank" is why attention
cannot be placed there, and it belongs on the map next to the timings — the
heatmap hatches those cells rather than leaving them empty.

## Board discipline

Every board interaction is serialised behind `flock /tmp/qnn_board.lock` and
wrapped in `timeout -s KILL`; the CPU governor is set to `performance` for the
measurement and restored to whatever was found. Contexts are deleted as they are
measured — the sweep builds thousands and the board has ~63 GB free.

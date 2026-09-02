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
    plot_opsweep.py   placement heatmap, dispatch-overhead bars, size ladders.

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

# Why HTA needs ~2.5 ms to do nothing

> **Superseded in part by section 'The floor is mostly power collapse' below.**
> Everything above it was measured at `--gap-us 3000`, which leaves HTA idle
> long enough to power down between every execute. The ~2.5 ms floor is real
> for that duty cycle, but roughly 2 ms of it is wake-up, not dispatch.

2.5 ms to run a block the Kryo finishes in 1.7 ms is not a plausible compute
result, so it was measured directly rather than reasoned about.

## The experiment

A single `Conv(1x1)` -- one op, no elementwise, nothing else -- swept across
seven sizes spanning **17,356,797x in arithmetic** and 153,600x in weight bytes,
each quantized identically and profiled the same way (50 iters, 3 repeats,
performance governor, gap-phase median, context already resident so init is
excluded).

| probe | Cin | Cout | S | MMAC | weights | HTA us | CPU us |
|-------|-----|------|---|------|---------|--------|--------|
| t01 |    8 |    8 |   1 |   0.000064 | 256 B  | **2467.8** |   14.53 |
| t02 |   64 |   64 |   8 |   0.033    | 16 KB  | 2694.9 |   23.28 |
| t03 |  256 |  256 |  50 |   3.277    | 262 KB | 1261.0 |  132.45 |
| t04 |  512 |  512 |  50 |  13.107    | 1.0 MB | 2511.9 |  303.29 |
| t05 |  720 | 2048 |  50 |  73.728    | 5.9 MB | 1965.4 |  775.80 |
| t06 |  960 | 2560 | 113 | 277.709    | 9.8 MB | 2063.5 | 1754.63 |
| t07 | 1920 | 5120 | 113 | 1110.835   | 39 MB  | 3057.0 | 7734.61 |

**A 64-MAC convolution costs HTA 2467.8 us.** A 1.11-GMAC convolution -- 17
million times the work -- costs 3057.0 us. Across the whole sweep HTA's time
spans **2.42x** while the work spans 17,356,797x. The CPU spans 532x and
tracks the arithmetic.

Fitting `t = fixed + work/throughput`:

    HTA   fixed 2.13 ms   marginal 1310 GMAC/s   R2 = 0.286
    CPU   fixed 0.089 ms  marginal  146 GMAC/s   R2 = 0.998

The R2 values are the finding. The CPU is a clean compute-bound line; HTA
barely correlates with work at all because the fixed term dominates every
point. Taking the marginal from the top two points alone (least contaminated by
the noisy floor) gives **839 GMAC/s**.

## It is not data movement

t01 has **256 bytes** of weights and a 32-byte activation. There is nothing to
stage, and it still costs 2467.8 us -- within 20% of t07's 3057.0 us, which
moves 39 MB of weights. Whatever the cost is, it is paid before any data is
touched, so it is a control-path cost, not a bandwidth one.

## Decomposing it against the other backends

The same three probes on every backend:

| probe | MMAC | CPU us | DSP us | HTA us |
|-------|------|--------|--------|--------|
| t01 | 0.000064 | **14.5** | 549.3 | 2467.8 |
| t04 | 13.107   | 303.3 | 660.0 | 2511.9 |
| t07 | 1110.835 | 7734.6 | 3149.9 | **3057.0** |

    floor        CPU  ~14 us    DSP  ~549 us    HTA  ~2468 us
    marginal     CPU  146       DSP  427        HTA  1885  GMAC/s

**HTA is the fastest compute engine on the SoC and the slowest to start.** At
1885 GMAC/s marginal it is 12.9x the Kryo's throughput; it just charges 2.5 ms
for the privilege. DSP sits between on both axes.

The DSP and HTA floors are not independent. On SM8250 the HTA is not directly
addressable from the CPU -- SNPE exposes the accelerator as **AIP**, "a software
abstraction of Q6, HVX and HTA into a single entity", where the network is split
into HTA subnets and HNN (Hexagon NN) subnets and the **Q6 coordinates
execution**. So an HTA execute is a FastRPC hop to the Hexagon *plus* HTA
programming, and the DSP floor is a lower bound on the shared part:

    ~0.55 ms   Hexagon / FastRPC control path   (measured as the DSP floor)
    ~1.92 ms   HTA-specific, on top of it       (measured by difference)

Published FastRPC round-trip latency is **200-300 us**, with kernel-launch
overhead around 20 us on Hexagon 690. The 549 us DSP floor is consistent with
that plus QNN's own per-execute backend work. The ~1.92 ms HTA increment is
measured by subtraction and is NOT attributed to a named internal cause here --
that would need Qualcomm internals. What is established is that it is fixed,
it is not data movement, and it is paid on every single execute.

Corroborating that HTA rides on the DSP path, the SDK's own release notes
record: *"Fixed a segmentation fault that could occur when executing a cached
model on the HTA backend if a subgraph fell back to the DSP backend."*

## The consequence

Break-even, from the fitted lines, is **335 MMAC per dispatch**. Below it HTA
cannot win no matter how efficient its arithmetic is. That single number
predicts every expert-block result measured:

| block | MMAC | predicted | measured |
|-------|------|-----------|----------|
| prefill MLP | 833 | HTA | HTA 2452 vs CPU 4415 -- **HTA wins** |
| decode MLP  | 221 | CPU | HTA 2512 vs CPU 1701 -- CPU wins |
| prefill qkv | 174 | CPU | HTA 2455 vs CPU 1193 -- CPU wins |
| decode qkv  |  58 | CPU | HTA 1769 vs CPU  594 -- CPU wins |
| decode oproj|  35 | CPU | HTA 2524 vs CPU  381 -- CPU wins |

So the answer to "why can't decode's MLP go on HTA" is not that it cannot --
it composes and runs correctly. It is that decode's MLP is 221 MMAC, below the
335 MMAC break-even, and the CPU already finishes it in 1.7 ms, which is below
HTA's floor. You cannot beat 1.7 ms with a device that takes 2.5 ms to start.

And decode cannot batch its way over the line: layers are sequentially
dependent, the denoising steps are sequentially dependent, and qkv/o_proj are
separated from the MLP by attention, which HTA has no kernel for at any rank.
The only lever is a longer action chunk -- the crossover sits between 50 and
113 tokens.

## Sources

* SNPE AIP runtime architecture (Q6 + HVX + HTA as one entity; HTA vs HNN
  subnets): https://developer.qualcomm.com/docs/snpe/aip_runtime.html
* Qualcomm AI Runtime (QAIRT) overview:
  https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/general_overview.html
* FastRPC round-trip latency 200-300 us:
  https://mdeore.medium.com/hexagon-dsp-cpu-offload-4fb8e4077fe8
* Kernel-launch overhead on Hexagon 690, and the general point that launch
  overhead negates acceleration for fine-grained kernels:
  https://arxiv.org/pdf/2309.02680
* Local: `QAIRT_ReleaseNotes.txt` (HTA/DSP subgraph fallback),
  `include/SNPE/DlSystem/DlEnums.hpp` (`AIP_FIXED8_TF` = "Snapdragon AIX+HVX")

## The floor is mostly power collapse, not dispatch

The sweep above held the idle gap fixed at 3000 us. Varying it changes the
answer completely. Same 64-MAC conv (t01), 40 iters per point:

| tile | loop | gap 0 | gap 250 | gap 1000 | gap 3000 | gap 10000 |
|------|------|-------|---------|----------|----------|-----------|
| t01 / **HTA** | **543.5** | 542.6 | 564.9 | **2537.9** | 2662.6 | 3063.9 |
| t01 / DSP     | 404.5 | 401.9 | 410.1 | 405.9 | 620.2 | 801.9 |
| t07 / HTA     | 2842.9 | 2796.5 | 2916.4 | 2990.4 | 2945.8 | 3148.0 |

**HTA's floor is 543 us kept busy and ~2540 us after about a millisecond of
idle** -- a 4.7x penalty, with the knee between 250 us and 1 ms. The DSP shows
the same effect, later (knee between 1 and 3 ms) and much milder (~400 us).

So the ~2.1 ms "fixed cost" fitted earlier is really two things:

    ~0.54 ms   genuine per-execute dispatch (warm)
    ~2.0 ms    wake-up from power collapse, paid only if HTA idled >~1 ms

Refitting on the WARM points (t01 543.5 us at 0.000064 MMAC, t07 2842.9 us at
1110.8 MMAC):

    HTA warm   fixed 0.543 ms   marginal ~483 GMAC/s
    CPU        fixed 0.089 ms   marginal  146 GMAC/s
    break-even 95 MMAC per dispatch   (was 335 MMAC cold)

That is a different regime. At 95 MMAC the break-even now sits BELOW decode's
MLP (221 MMAC), which predicts warm HTA would do it in roughly
543 + 221*2.07 = ~1.0 ms against the Kryo's 1.70 ms.

### Nothing in this project has ever kept HTA warm, or asked it to

Two levers exist in the SDK and neither is used anywhere in the repo:

* `QnnHtaPerfInfrastructure.h` defines a power ladder --
  `DEFAULT=0, LOW_POWER_SAVER, POWER_SAVER, HIGH_POWER_SAVER, BALANCED,
  HIGH_PERFORMANCE, BURST=6`. `grep` for `PerfInfrastructure|PowerMode|BURST`
  finds **zero** hits in `runtime/profile_segments.cpp` and **zero** in
  `flow_c/flowc/emit_runtime.py`. Every number in this repo was taken at
  `POWERMODE_DEFAULT`.
* `QnnGraph_executeAsync` exists, with an execution queue and a completion
  callback. The harness and the Flow C runtime both use synchronous
  `QnnGraph_execute` only.

In a real pipeline HTA would be cold at every dispatch anyway: decode's
attention block alone is 4.3 ms, so consecutive HTA dispatches are ~5 ms apart,
far past the ~1 ms knee. Warm numbers are only reachable if something keeps the
accelerator awake -- which is exactly what "prefetching" would have to mean
here.

### ...but warming does NOT rescue the blocks we care about

The warm/cold prediction above was extrapolated from the t01->t07 line. It is
wrong. Measured directly, every expert block at gap 0 against gap 3000:

| tile | backend | WARM (gap 0) | COLD (gap 3000) | ratio |
|------|---------|--------------|-----------------|-------|
| `ncd_qkv`   | HTA | 1368.5 | 3208.5 | **2.34x** |
| `ncd_oproj` | HTA |  731.2 | 1308.2 | 1.79x |
| `ncd_mlp`   | HTA | **2518.3** | 2593.2 | **1.03x** |
| `nc_qkv`    | HTA | 2140.6 | 2356.9 | 1.10x |
| `nc_oproj`  | HTA | 1447.6 | 1631.4 | 1.13x |
| `nc_mlp`    | HTA | **2392.1** | 2459.6 | **1.03x** |
| `ncd_mlp`   | CPU | 1674.1 | 1713.7 | 1.02x |
| `nc_mlp`    | CPU | 4243.7 | 4401.7 | 1.04x |

**The wake-up penalty is only visible on short dispatches.** A block carrying
~2.4 ms of real work absorbs the DVFS ramp inside its own execution, so the two
MLP blocks -- the only ones worth exporting -- gain 3% from being warm. The
blocks that DO gain (ncd_qkv 2.34x) are the ones the CPU already wins by 2-4x,
so recovering their wake-up changes nothing.

Warm-to-warm, both verdicts are unchanged:

    prefill MLP   HTA 2392.1  vs  CPU 4243.7   HTA wins 1.77x
    decode  MLP   HTA 2518.3  vs  CPU 1674.1   CPU wins 1.50x

The predicted ~1.0 ms warm decode MLP does not exist; it is 2518 us. The linear
`fixed + work/throughput` model does not hold across different op counts and
tensor shapes, so it should be used for intuition about the floor and not for
extrapolating any particular block.

Note the CPU also runs faster warm (`nc_oproj` 584.6 vs 920.3, 1.57x), so the
gap-phase methodology penalises both engines, not just HTA.

## Can the HTA cost be prefetched or hidden?

Four mechanisms, three of which do not apply here:

1. **Keep HTA warm** (power mode, or a keep-alive dispatch during the CPU's
   attention block). Measured above: worth 3% on the MLP blocks. Not useful.
2. **`POWERMODE_BURST`.** Never set anywhere in this repo -- everything ran at
   `POWERMODE_DEFAULT`. Untested, and the expected value is low for the MLP
   blocks specifically, because at 2.4 ms warm they are compute-bound rather
   than floor-bound. Worth one experiment, not a plan.
3. **`QnnGraph_executeAsync`.** The API exists (execution queue + completion
   callback) and neither the harness nor the Flow C runtime uses it. But inside
   an expert there is nothing to overlap it WITH: norm -> qkv -> attention ->
   o_proj -> norm -> MLP is a strict chain, layers are sequentially dependent,
   and the 10 denoising steps are too. Async buys nothing within the component.
4. **True prefetch -- starting the MLP before its input exists.** Impossible.
   The MLP input is the RMSNorm of the attention output; there is no branch to
   speculate on and no value to guess.

Where async DOES pay is across networks, which is XPU-RT's actual job: issue
the expert's HTA MLP, let the CPU run another network's work, collect on the
callback. That fills the idle band the Gantt chart shows (39 ms per prefill).
It does not make the expert itself faster, and on decode it cannot -- the
offload is a net loss there before any overlap is considered.

## Is it just that decode's MLP is smaller?

Partly, but the mechanism is worth stating precisely, because "too small for
HTA" implies HTA is inefficient on small work and that is not what happens.

Fitting the warm HTA measurements on three terms:

    HTA_us = 873 + 158*ops + 1.393*MMAC        R2 = 0.742
             ^^^   ^^^^^^^   ^^^^^
        per-execute per-op   718 GMAC/s

For a 6-op MLP block that is **1.82 ms of overhead before a single multiply**.
The arithmetic then adds 308 us (decode, 221 MMAC) or 1160 us (prefill, 833
MMAC). Hence:

    ncd_mlp  221 MMAC   HTA 2518.3   CPU 1674.1
    nc_mlp   833 MMAC   HTA 2392.1   CPU 4243.7
    HTA varies 1.05x across a 3.77x work range; CPU varies 2.53x

**HTA costs ~2.45 ms for either MLP.** It is not slower on the small one -- it
is the same. What differs is the CPU, which needs 4.24 ms for prefill's and
only 1.67 ms for decode's. So the rule is simply: **HTA wins when the CPU would
need more than ~2.45 ms**, and decode's MLP is already under that.

## Can a full model run benefit from warm dispatches? No.

The warm/cold gap is real physics but it is irrelevant to this workload,
because it only bites dispatches shorter than roughly a millisecond and every
block worth offloading is longer than that.

Measured on the six real vision Conv1x1 HTA kernels that are actually resident
on the board (49 exist under `/root/models/smolvlm_vision_v3/ctx`):

| kernel | WARM (gap 0) | COLD (gap 3000) | penalty |
|--------|--------------|-----------------|---------|
| dsp_seg_00_sng_MatMul_0    |  6559.4 |  6545.3 |  -14 us |
| dsp_seg_01_node_MatMul_549 |  3471.0 |  3561.3 |  +90 us |
| dsp_seg_01_node_MatMul_554 |  9320.5 |  9163.8 | -157 us |
| dsp_seg_02_node_MatMul_570 | 14410.2 | 14078.6 | -332 us |
| dsp_seg_02_sng_MatMul_2    |  6794.4 |  6437.8 | -357 us |
| dsp_seg_03_node_MatMul_669 |  3395.5 |  3586.3 | +191 us |

Mean penalty **-96 us per kernel** -- noise, and if anything negative. These
kernels run 3.4 to 14.4 ms each and absorb the DVFS ramp entirely, exactly like
the expert MLP blocks do.

So the full inventory of blocks this project would ever place on HTA:

    49 vision conv1x1 kernels   3.4 - 14.4 ms   warm/cold penalty ~0
    prefill MLP                 2.4 ms          1.03x
    decode MLP                  2.5 ms          1.03x

Nothing on that list benefits. The blocks that DO benefit -- `ncd_qkv` 2.34x,
`ncd_oproj` 1.79x, the 64-MAC probe 4.7x -- are all short dispatches the CPU
already beats by 2-4x even when HTA is warm. **There is no configuration in
this pipeline where keeping HTA warm changes a placement decision.**

A corollary for the cost model: the gap-blindness of the per-cell numbers
(one value per tile/backend, when the true value varies 4.7x with idle history)
does not matter here, because every tile in this pipeline is long. It would
matter for a workload built from many short accelerator dispatches -- and that
is precisely the workload shape one should avoid on this silicon.

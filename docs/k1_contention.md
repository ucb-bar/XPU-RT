# Contention on the K1, measured on the path we actually ship

`runtime/scripts/k1_contention_mb.py`. The artifact it writes
(`artifacts/k1_run/contention_mb.json`) is NOT in git -- `artifacts/` is
untracked and the script regenerates it from the board in about twenty
minutes. This document is the part worth keeping.

## The short version

**Co-runner contention is below this measurement's resolution for up to four
co-runners.** The IREE-path numbers in `contention.json` — same-cluster 1.043×,
cross-cluster 1.185× — **do not reproduce** on the ModelBlaster path.

Nothing should install a contention model from either artifact today.
`contention_model.load()` returns `None` when the file is absent and the
scheduler wiring is off unless a model is explicitly installed, so the default
is already the right one.

## What was measured

DroNet pinned to hart 0, co-runners looping on chosen other harts, median of 6
warm iterations per run, solo re-measured immediately before every arm.
Cost-weighted ratio (total co-run / total solo), because the median over 21
dispatches said 1.000 for an arm whose model got 3.7% slower — the slowdown
concentrates in the few big dispatches, so the median is the wrong summary for
something that multiplies per-dispatch durations. Both are recorded.

    arm                    n   cost-weighted ratios              spread
    same_cluster  x1       4   0.999  1.012  1.010  1.051         5.2%
    other_cluster x1       4   1.061  0.995  1.002  1.004         6.6%
    same_cluster  x3       1   1.025
    other_cluster x4       1   1.039
    same_cluster  x3 yolo  1   1.024
    other_cluster x4 yolo  1   1.006

The two four-sample distributions straddle 1.0 and overlap completely. The
arms with more co-runners land inside that same band, and are not monotonic in
co-runner count — `other_cluster x1` measured 1.061 while `other_cluster x4`
measured 1.039, which cannot be physical and is therefore noise.

## Three ways this measurement was wrong before it was right

Each was caught by a check, and each would have produced a publishable-looking
number.

**1. The co-runner was not where it said it was.** A co-runner respawned per
inference spends most of its wall time in fork, exec, loader and page faults —
work done *before* `main` calls `sched_setaffinity`, so it is unpinned and
lands wherever the scheduler puts it. Measured with per-CPU `/proc/stat`
sampling (loadavg is useless on this board: it has a permanent floor of
exactly 2.00 from two D-state kernel threads):

    respawn per inference   cpu1 80%   and cpu3 8%, cpu4 4%, cpu5 6%
    ITERS inside one proc   cpu1 100%  and every other hart 0%

DroNet is 8.3 ms and the binary is 3.5 MB. With the respawn form, both the
same-cluster and the other-cluster arm were contaminated by the *same*
off-target load — which is exactly why the first sweep reported 1.011× and
1.007×, a null result that looked like "these kernels do not contend."

**2. The survivor check counted itself.** `pgrep -cf <pattern>` matches the
`pgrep` command line, which contains the pattern. The check reported a
surviving co-runner on a clean board and refused to continue — permanently.
The bracket trick (`[d]ronet…`) matches the process and not the check.

**3. The design was unpaired.** Two solo runs twenty minutes apart differed by
2.6% with nothing else on the board. Against effects of 1–6% that is not a
correction, it is the whole signal. Solo is now re-measured immediately before
every arm, and each arm records its own drift against the sweep reference
(`solo_drift_vs_reference`, now 0.996–1.002). An arm whose drift is the size
of its effect is not evidence and a reader cannot tell without that number.

## What is NOT claimed

That contention does not exist on this board. A one-off stress sweep — not
paired, not repeated — showed a much larger effect for heavy co-runners:

    4 DroNet co-runners, other cluster    1.139x
    3 YOLO   co-runners, same  cluster    1.113x

Neither reproduced under the paired design (1.039× and 1.024×). So the honest
statement is that any real effect at this co-runner count is smaller than a
few percent, and settling whether cross-cluster genuinely costs more than
same-cluster needs either many more repetitions or a heavier co-runner than a
DroNet inference.

That question matters — it is the same one the multi-core sweep raised from
the other side, where DroNet came out *slower* on 8 harts than on 4 (5.32 vs
5.25 ms) and yolov8_nano did not. There the effect was large enough to see.
Here it is not, and saying so is the finding.

## Why the old artifact should not simply be carried forward

`contention.json` measures `iree-benchmark-module` running `.vmfb` files. That
path is retired; every kernel on this board today comes out of ModelBlaster's
curated tree. A multiplier measured against IREE-compiled kernels is a
multiplier for code nobody runs, and its per-dispatch shape is different too:
IREE needed one process invocation per dispatch, so a 21-dispatch model was 21
separate runs with 21 separate cache states, rather than one run in which the
dispatches see each other.

It is kept, not deleted, because it is the provenance of the numbers quoted in
`contention_model.py`'s docstring — which now need the qualifier that they
describe a compiler we no longer use.

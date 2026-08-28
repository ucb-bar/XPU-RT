# B4, the sharding rung: what it costs before it buys anything

Host-side gate, run before any board time. Nothing here is a board measurement;
everything here is the solver's prediction from the *measured* per-dispatch
costs (`pdb_hash = sha256:602e075cc1182f00...`, the same four CSVs for all three
cells, re-solved in one sitting with one command).

## Why this gate exists

The walker refuses to give a core kind an intra-op pool when that kind already
has more than one scheduler worker:

    OVERSUBSCRIPTION GUARD: inter-dispatch parallelism (one scheduler worker
    per hart) and intra-op parallelism (pool helpers) draw on the same
    physical harts.
    -- ModelBlaster/pipeline/generate_xpurt_main.py, ~line 940

That guard is right, and it means "turn sharding on" is not a flag. Reaching a
pool requires giving the kind ONE machine in the network spec, so the real
question is a placement question:

    four scheduler workers per cluster, or one worker plus three pool helpers?

## The three cells

    baseline   cpu_p=4 cpu_e=4    data/toplevel/networks_k1_mb_4model_4hz_fused.json
    S1         cpu_p=1 cpu_e=4    data/toplevel/networks_k1_mb_4model_B4_shard_c0.json
    S2         cpu_p=1 cpu_e=1    data/toplevel/networks_k1_mb_4model_B4_shard_both.json

All three: `--solver milp --scheduler edf --profiled`, 1201 dispatches, and an
identical busy-sum of 1086.56 ms. Same work, three ways to place it.

| cell     | machines | makespan  | op deadline misses | max lateness |
|----------|----------|-----------|--------------------|--------------|
| baseline | 8        | 902.65 ms | **0**              | 0 ms         |
| S1       | 5        | 902.65 ms | 229 (19.1%)        | 110.68 ms    |
| S2       | 2        | 970.71 ms | 314 (26.1%)        | 162.06 ms    |

## The result that matters, and the one that would have misled

Makespan does not move at all between the baseline and S1. Dropping three of
cluster 0's four cores changes it by 0.00 ms.

Read alone, that says the cores are idle and therefore free to hand to a pool.
That reading is wrong, and it is worth writing down because it is the reading
the headline number invites. Makespan here is set by yolov8_nano's long serial
conv chain, which no amount of extra same-kind cores shortens -- the earlier
interleaving analysis already measured zero self-pipelining. So makespan is
exactly the metric that cannot see what those cores are doing.

What they are doing is absorbing the latency-critical periodic work. mlp_control
and fused_full both run at a 10 ms period; on a single P core they queue behind
yolov8_nano instead. Hence **0 -> 229 predicted misses at an unchanged
makespan**, and a max lateness that goes from 0 to 110.68 ms.

Under the lexicographic objective, hard deadline miss count ranks first and
makespan seventh. So the correct summary is the opposite of the makespan
reading: S1 is a large regression that happens to be invisible in the headline
number.

## The bar for B4

This sets an explicit, quantified bar rather than a hope. To be accepted, intra-op
sharding on cluster 0 must buy back more than **229 predicted deadline misses**
and **110.68 ms of max lateness**.

That is reachable in principle -- yolov8_nano is 226.87 ms of which 96.1% is one
op type -- but it is not free, and it is not what "the cores are idle" suggested.

## What is NOT yet true, and must be before the board runs

1. The solver does not model the pool. Every number above assumes *unsharded*
   per-dispatch costs. The sharding upside is invisible here by construction.
   Closing that needs a re-profile with the pool active, written to its own
   profile tree, and fed back in -- measurement first, then the scheduler
   consumes that measurement.
2. The shard path is not correct yet on rvv. `parallel_conv2d_s8` slices
   IHWOC-packed weights with an OIHW offset formula, and the fused convs -- 96.1%
   of yolov8_nano -- do not route through the parallel wrapper at all. Profiling
   before those land would measure the bug.

So: gated, quantified, and not yet run. The next artifact in this directory
should be a board measurement or nothing.

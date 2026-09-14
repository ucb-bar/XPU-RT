# Archived 2026-09-06 — pipelined arms carried the wrong latency

These results are kept because they are real measurements, but the PIPELINED arms are
parameterised wrongly and their rows must not be quoted.

`wall / n_instances` from the board trace is THROUGHPUT, not latency. For a pipelined
schedule the two diverge, and the sweep used throughput as the latency parameter:

| arm       | used latency | measured latency (per-inference span) | measured throughput |
|-----------|--------------|---------------------------------------|---------------------|
| pipe110   | 117.7 ms     | **385.1 ms**  (3.3x understated)       | 111.4 ms            |
| pipe200   | 231.8 ms     | **260.5 ms**  (11% understated)        | 219.2 ms            |
| serial283 | 283.4 ms     | 239-270 ms (4 runs, median ~255)       | same (1 in flight)  |
| cpu685    | 684.8 ms     | 677.1 ms                               | same (1 in flight)  |

At the pipe110 cadence, 4 inferences are in flight, so 385 ms ~= 4 x 111 ms: a fresh
result arrives every ~111 ms but each one is ~385 ms old.

AFFECTED: every pipe110 and pipe200 number, and the headline that pipelining at 110 ms
is free or better than zero latency (+13.3 points on eggplant).

NOT AFFECTED: lat0, serial283, fp32_555, cpu685 (single instance, latency == throughput);
the monotone degradation across those; grasp-vs-push; completion time; the energy
results for the non-pipelined arms.

Superseded by the corrected run with --latency-ms 385.1 / --issue-period-ms 111.4 and
--latency-ms 260.5 / --issue-period-ms 219.2.

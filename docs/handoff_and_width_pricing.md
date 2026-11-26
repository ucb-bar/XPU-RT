# Hand-off cost and the price of core time

Two things the scheduler's model treated as free.

1. **Hand-off.** Moving consecutive dependent dispatches of one network between
   cores. On this tree the transfer matrix is filled from `scheduler.handoff_ms`
   (`workload_factory.build_handoff_transfer_times`, {from kind: {to kind: ms}},
   0 on the same core; absent = zeros). Section 1 says how a K3 hand-off
   calibration maps onto it and what `xpu-rt/handoff_metrics.py` reports.
2. **Core time.** The decoder family picks, per dispatch, the machine
   combination that finishes earliest. Helper cores cost nothing in that rule,
   so a width-2/4/8 combination wins whenever it finishes any earlier at all. A
   core-time price now enters the rule (section 2).

Both are off by default; a solve without them writes the same schedule as before.

## 1. Hand-off cost

A calibration measured as `same_kind_ms` per kind (between two different cores
of one kind) and `cross_kind_ms` per direction ("A->B") is the table

```json
"scheduler": {"handoff_ms": {"cpu_e": {"cpu_e": <same_kind_ms[CPU_E]>, "cpu_p": <cross_kind_ms["CPU_E->CPU_P"]>},
                             "cpu_p": {"cpu_p": <same_kind_ms[CPU_P]>, "cpu_e": <cross_kind_ms["CPU_P->CPU_E"]>}}}
```

which gives the same matrix (tests/test_handoff_table.py checks the builder).
The decoder family reads it exactly: a successor on combination `c` may not
start before `end(p) + transfer[first machine of p][first machine of c]` for
every predecessor `p`.

(An earlier version of this work, on the deprecated history, carried its own
`--handoff-calibration` file and an additive per-width `split_overhead_ms`. The
first is what `handoff_ms` already does here; the second was measured as 0 at
every width on the K3 and is not carried.)

### Metrics

`xpu-rt/handoff_metrics.py <schedule.json>` reports, per network instance and in
total: core changes along consecutive dispatches in start order (master core =
first machine of `hardware_target`), kind changes, distinct cores, composite
dispatches, core-ms, and the dependency edges that change core or kind (the
count the matrix prices). `--network REGEX` restricts the listing and prints
medians; `--skip-zero-duration` leaves zero-duration dispatches out; `--json`
prints everything.

## 2. Price of core time

### The rule

`schedule_decoder.decode(..., price=...)` chooses, for op `i`, the combination
`c` that minimises

    finish(i, c) + price_i * (core_ms(i, c) - min over c' of core_ms(i, c'))

with `core_ms(i, c) = len(c) * dur[i, c]` over the combinations the op can use.
A millisecond of finish time is traded against `price_i` milliseconds of extra
core time. `price` None or 0 is the earliest-finish rule, tie-breaks included.

The price is not only about width. A singleton on the slower machine kind also
burns more core time than the op's cheapest combination and is priced the same
way.

What price 1 means for a split, for an op whose singleton time is `d1`: a
width-`w` combination with time `dw` on idle cores is chosen when
`dw + (w * dw - d1) < d1`, that is when the speed-up `d1 / dw` exceeds
`(w + 1) / 2`: 1.5x at width 2, 2.5x at width 4, 4.5x at width 8. An op with
durations {width 1: 1.0, width 2: 0.6} therefore still splits at price 1 and
stops splitting above price 2.

### The policy `core_time_price`

Spec key `scheduler.core_time_price`, or `--core-time-price` (the CLI wins): a
number, or `auto`. Default unset. It travels on the workload object
(`workload.core_time_price`) and is read by `DecoderContext`.

- a number: that price for every op.
- `auto`: with deadlines, finishing early has no value by itself; core time is
  the scarce resource, so a split, or a slow-kind placement that burns more core
  time, is taken only when it is needed to meet a window. The levels 1.0, 0.5,
  0.25, 0.0 are tried and the highest price whose schedule has the fewest missed
  periodic windows (`schedule_decoder.evaluate`'s miss count) is kept. Between
  the global levels there is one staged step: the price stays at 1.0 for the ops
  of every network all of whose instances met their windows in the price-1.0
  schedule, and is lowered (0.5, 0.25, 0) only for the ops of networks with a
  missed instance. A staged schedule is preferred over a global lowering with
  the same miss count. The preference order is

      1.0, staged 0.5, staged 0.25, staged 0, 0.5, 0.25, 0

  and the first entry with the minimum miss count wins. A price-1.0 schedule
  with no misses ends the search. Inside one level `heft_edf` runs its laxity
  gate enumeration and ranking unchanged.

`metadata.core_time_price` in the schedule records the policy, the solver that
won, the level that won (`level`, `price`, `lowered_price`,
`lowered_networks`), its miss count and what every level tried scored.

### Which solvers implement it

| family | status |
|---|---|
| `heft`, `heft_edf` | Implemented, including `auto`. |
| `pso`, `sa` | Every decode is priced. Under `auto` they adopt the per-op price vector `heft_edf` settled on. With a price set they are seeded from `heft` and `heft_edf` only, because their incumbent is ranked by (missed windows, makespan) and an unpriced greedy seed could win on makespan alone. Their fitness itself does not include core time. |
| `greedy`, `greedy_periodic`, `greedy_reserved`, `decomposed` | Not implemented. Their loops choose the op and its combination together by earliest completion (and `greedy_reserved` by its own lane key), so a price term would change which op is placed next as well as where. That is not a small change. |
| CP-SAT (both), MILP (both) | Not implemented. |

Because of that, `--solver auto` with `core_time_price` set restricts its
candidate list to `heft` and `heft_edf` and prints that it did. Its ranking,
(window misses, converged, makespan, all-ops end), is unchanged; with a price
set it also prints, per candidate, total core-ms and the hand-off count from
`handoff_metrics`. Naming a solver that does not implement the price prints a
note that the price has no effect.

### The codegen contract (one width per packed-weight dispatch)

On this tree `run_xpurt_schedule.py` applies ModelBlaster's codegen contract to
the workload itself (`codegen_contract.apply_to_workload` in `_build_workload`),
before any solver runs, so the decoder family sees the pinned widths whatever
the candidate list is. A price does not undo the pin
(tests/test_core_time_price.py). The pin picks each dispatch's width by summed
duration over its instances, before any price is applied: for a network with
several instances per hyperperiod the price cannot narrow a convolution the pin
has already fixed at a wider width.

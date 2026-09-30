# The top rungs: what the inner and outer loops each do on w4 and w5

Both rungs used to report that no lever helps — w4 10 → 10 deadline misses, w5 11 → 11 —
and the ladder figure could show it without explaining it. Neither was a scheduling
limit. This is what they actually do, why they looked flat, and how to reproduce it.

## Result

Instance-level deadline misses, on the accept rule's own counter. Greedy; two independent
runs of every row are bit-identical.

| rung | baseline | inner (AOT) | board re-cost | board re-solve | levers found |
|------|---------:|------------:|--------------:|---------------:|--------------|
| w4 · 4 nets, profile-sized | 10 | **5** | 4 | 4 (refused) | `shard:ffn_block` |
| w5 · 5 nets, profile-sized | 11 | **7** | **16** | 14 | `shard:ffn_block`, `shard:dronet`, `ime` |
| b4 · 4 nets, board-sized | 10 | **3** | 4 | 4 (refused) | `shard` |
| b5 · 5 nets, board-sized | 10 | **6** | — | — | `shard:dronet`, `shard:ffn_block`, … |

Both profile-sized rungs used to report 10 → 10 and 11 → 11. Figure:
`results/codesign_feedback/inner_outer_arc.png`.

Four defects were between the loop and these numbers, and **none of them was scheduling**.
Two are described below; the other two were in the accept path:

* **A measurement tolerance was applied to a deterministic count.** `miss_rate_frac` is
  8% because seven repeated *board runs* of one schedule gave MLP 7–9 misses of 38 — real
  execution jitter. The AOT search is not a measurement: a candidate's misses are computed
  from a solved schedule against fixed costs, deterministic to the last digit. On b4, 34
  instances put the term-1 tolerance at **2.72 misses**, so `shard:dronet` — which takes
  misses 5 → 3 and clears dronet entirely — was called "indistinguishable on every term"
  and the decision fell through to p99, where it loses. `DETERMINISTIC_TOLERANCES` zeroes
  the miss tolerance and keeps the continuous ones, which guard real CP-SAT nondeterminism.
* **The winner was ranked by makespan** — the term the rule places *seventh*. The code read
  `min(winners, key=(c["score"], c["mk"]))` and its comment claimed it minimised "the
  objective first", but `score` *is* the makespan metric. With the miss tolerance at zero
  this shows directly: more candidates are accepted per round, and w5's final goes from 7
  misses to **nine**, a hill-climb steered by the seventh term.

## Why they looked flat: two defects, neither of them scheduling

**The list scheduler could not be held to the codegen contract.** CP-SAT couples a
dispatch's instances with a constraint, so it can be *asked* for one width per
packed-weight dispatch. Greedy picks each op's combination as it walks the ready set,
with nothing to couple, so the contract could only be checked afterwards and the whole
candidate discarded. On w4 that threw away the best schedule anyone has produced for the
workload — 5 misses against 10, worst lateness 3.09 ms against 17.95 — over **three**
dronet dispatches whose instances took different widths. `codegen_contract.pin_uniform_widths`
now restricts each such dispatch to one width up front, chosen by *measured* cost.

Two traps worth knowing if you touch this:

* Choosing the *widest* usable width is wrong on real data — yolo's OC=2 detect-head
  convs measure slower on four cores than on one. Choosing the *narrowest* silently
  undoes sharding, which is the point of the lever. The width is picked by summed
  measured duration over the dispatch's instances.
* The width must be **priced out**, not flagged. Greedy never reads
  `infeasible_combinations`; flagging alone left dronet 0/8/9 still mixing widths
  `[1, 4]`. `processing_times` is what both schedulers read, and CP-SAT folds a sentinel
  cost back into its exclusions, so one edit binds both arms.

**Sharding was all-or-nothing across networks.** `machine_combination_mode: "shard"`
opens multi-hart combinations for every network at once. On w5 that also widens
`yolov8_nano_64x96` — 191.6 core-ms monopolising all eight harts while the 5 ms-period
networks wait — so worst lateness went 24.67 → 34.87 ms and the whole lever was rejected,
including the part that helps. `restrict_shard_to_networks` plus a per-network
`shard:<net>` lever lets a round widen one network and hold the rest at one core;
successive rounds compose, so the loop *discovers* the set. w5 converges on
`{ffn_block, dronet}` and correctly leaves yolo single-core.

## What the outer loop contributes, stated precisely

It is not "re-place the dispatches". On these rungs it does two things a purely AOT flow
cannot:

1. **It reveals what the cost model missed.** w5's AOT schedule predicts 7 misses; the
   board says **16**, and five of them are in `fused_full`, a network the AOT solve
   believed was entirely clean. That is a discrepancy no amount of offline solving finds.
2. **It attributes the residual.** Of those 16, five *cannot* be scheduled away:
   `net_times` reports a single measured core width for `fused_full` (against
   `{1,2,4,8}` for ffn_block/dronet/yolo), its window is 5.0 ms against a 3.62 ms
   singleton, and the board inflates its dispatches up to 3.12×, which puts it over.
   Adding `shard:fused_full` was tried and changes nothing — identical 16 misses, same
   breakdown — because sharding a single-width net is a no-op by construction. So the
   residual splits **5 unschedulable + 9 contention**, and the first five are an
   actionable ModelBlaster task (generate a multi-core `fused_full`), not a scheduling
   failure.

That also explains the weak fix: the greedy re-solve recovers 2 of the 9 revealed misses
because five of the nine were never recoverable by any scheduler.

## The windows were not achievable, and that is why the profile-sized rungs stall

The ladder's premise is that a schedule meeting every deadline *exists* using
implementations already measured on the board. That was verified against **profile**
times, and the board disagrees. Warm execution per instance, summed from the trace's own
cycles (`scripts/attribute_board_misses.py`):

| net | profile @ best width | measured warm | window | board/profile |
|---|---:|---:|---:|---:|
| ffn_block | 7.72 ms (8c) | **10.03–10.34** | 10.0 | 1.34× |
| yolov8_nano_64x96 | 23.95 ms (8c) | **42.77** | 26.0 | 1.79× |
| dronet | 6.05 ms (2c) | 6.02 | 7.0 | 1.00× |
| fused_full | 3.62 ms (1c) | 4.42 | 5.0 | 1.22× |

So `ffn_block` misses its 10 ms window at its *fastest measured width*, and yolo misses
its 26 ms window by 1.64×. yolo's multi-hart time had never been measured before this
work — 42.77 ms is the first one, against a profile that promised 23.95. Three of five ffn
instances and the yolo instance are over window **by execution alone**, which no scheduler
recovers, and the earlier "0 infeasible misses" classification says otherwise only because
it trusts profiles.

`b4`/`b5` (`LADDER_BOARD`) size windows from these measurements with ~15% slack. On b4 the
board then confirms it: `ffn_block` fits (warm 8.40 ms in a 12 ms window, **zero**
instances over), and the only execution-bound misses left are **two cold-start**
instances — the first instance of a network runs 2.74× its warm median for `fused_full`
and 1.34× for `dronet`, which the warm steady-state profile models not at all.

## Known limits

* **w4 has no reveal.** Its board re-cost is 5 → 4: the AOT model was already right
  there, and the greedy re-solve makes it worse (4 → 8) and is correctly refused. Bar 4
  is drawn as the beat it kept, marked `[refused: nothing better]` — an empty bar there
  would read as zero misses, the opposite of the truth.
* **A list scheduler is the wrong re-solver, but CP-SAT is the worse solver here.**
  Greedy does not optimise deadline misses, so re-solving it against better costs moves
  the schedule without aiming at the metric — board-aware greedy *search* gives 10 → 8 on
  w4 and 11 → 12 on w5, worse than solving blind and measuring afterwards. CP-SAT is
  indeed the better *re-optimiser*: on w5 its board re-solve recovers **9** misses
  (29 → 20) where greedy recovers 2 (16 → 14). But its own AOT solve of the same spec is
  far worse — 25 misses against greedy's 7 at a 400 s budget on 492 dispatches — so its
  arc ends at 20 against greedy's 14. Greedy wins end to end on these rungs; the exact
  solver wins only the beat where the metric is already the objective.
* **The w5 board run excludes the `ime` lever.** The harness is built one backend per
  core kind, and the converged schedule's IME dispatches need an `ime_x60` backend that
  the RVV build lacks (`FATAL entry 0 of ffn_block asks for impl 'ime'`). The measured
  run is the shard-only spec; the IME lever is present in the AOT arc but not in the
  board beats.

## Reproduce

```bash
# inner loop (fast, deterministic)
scripts/run_codesign_loop.py --workload data/toplevel/scaling/w4_ffn_dronet_sensor.json \
    --solver greedy --max-rounds 4 --out-dir results/w4_inner

# execute that schedule on a real K1 and calibrate from its own trace
eval "$(scripts/setup_spacemit_toolchain.sh)"
ModelBlaster/scripts/run_xpurt_k1.sh --schedule <the converged schedule>.json \
    --models mlp_control,fused_full,ffn_block,dronet \
    --staged-ir mlp_control:ModelBlaster/build/k1_xpurt/mlp_control/int8 \
    ... --backends rvv_x60,rvv_x60 --quant int8 --out-root /tmp/w4_board
scripts/emit_board_calibration.py --trace <...>_trace.csv --schedule <...>.json \
    --out results/codesign_feedback/k1_cal_w4_measured.json

# the four beats, scored on those measured costs
scripts/run_codesign_loop.py --workload data/toplevel/scaling/w4_ffn_dronet_sensor.json \
    --solver greedy --max-rounds 4 \
    --board-calibration results/codesign_feedback/k1_cal_w4_measured.json \
    --board-solver greedy --out-dir results/w4_outer

scripts/plot_inner_outer_arc.py \
    --rung "w4 · 4 networks=results/w4_outer/w4_ffn_dronet_sensor/loop_report.json=data/toplevel/scaling/w4_ffn_dronet_sensor.json" \
    --rung "w5 · 5 networks=results/w5_outer/w5_ffn_dronet_yolo/loop_report.json=data/toplevel/scaling/w5_ffn_dronet_yolo.json"
```

`--backends` is one entry per **core kind**, not per model — two kinds on the K1, so two
entries regardless of how many networks are in the schedule. Graph extraction needs
torch, which the solver venv does not have, so board runs reuse built IR via
`--staged-ir` / `MB_IR` rather than re-extracting.

## The five-network cases, and what "suggest the transformation / apply CP-SAT" actually buys

**`b5y_board_holds` — the clean one. `3 → 0 → 0 → 0`, verified on a K1.** Baseline
misses, the AOT search clears every deadline with `shard:ffn_block`, the board re-cost
agrees at zero, the re-solve has nothing to do, and `attribute_board_misses.py` reports
**0 execution-bound instances**: every network fits its window, cold start included.

It is the only rung of its kind because it is the only one whose windows were sized from
**board** measurements at the width the scheduler actually falls back to. Two earlier
attempts show what that sentence is worth:

| rung | outcome | why |
|---|---|---|
| `b5_board_sized` | 6 misses | ffn window **equals** its period → execution fits, no slack for queueing |
| `b5x_board_slack` | 0 predicted, fails on board | windows sized from *widened* dronet/yolo (6.03, 42.77 ms), but the loop widened only ffn and left them at one core, where the board gives 9.56 and 60.02 against 9.0 and 50.0 |
| `b5y_board_holds` | **0, holds** | every window exceeds the measured **cold** time at one core |

**`b5z_reveal` — the reveal-and-fix rung, and an honest negative.** Its dronet window
(9.0 ms) sits between dronet's 1-core *profile* time (8.33 → the AOT solve predicts it
fits) and its 1-core *measured* cold time (11.74 → it does not), and above its sharded
cold time (8.06 → widening should fix it). Arc: **`3 → 0 → 1 → 1`**. The board reveals a
dronet miss no offline solve could see. Then:

* **Suggesting the transformation works as a mechanism.** Board-aware search picks
  `shard:dronet` *first* — the lever the profile-blind search never proposed.
* **It does not pay off.** Sweeping the entire shard lever space against board costs:
  `ffn` 4, `dronet` **3**, `ffn+dronet` **3**, `ffn+dronet+yolo` 5, `ffn+yolo` 6,
  `dronet+yolo` 5. The minimum over the whole space is 3 — and the *blind* AOT schedule
  recost on the board is **1**. Widening dronet fixes dronet and costs ffn more than it
  saves; eight harts cannot carry both at these windows.
* **CP-SAT does not rescue it: 9 misses** against greedy's 3 on identical input, with
  every phase FEASIBLE and `best_bound = 0.0` after 421 s. On a five-network board-cost
  model it is nowhere near converged, so this is not a case where more solver power helps.

The consistent result across w4, w5 and b5z is that the outer loop's **re-solve** half
does not pay off with either solver, while its **reveal and attribution** half does: it
is what showed that ffn's 10 ms window and yolo's 26 ms window were never meetable, and
what turns "the loop failed" into "the target was wrong, here is the achievable one".

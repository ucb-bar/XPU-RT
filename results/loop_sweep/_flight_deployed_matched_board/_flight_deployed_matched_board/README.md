# Automatic co-design loop — `_flight_deployed_matched_board`

Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →
measure each → accept the largest measured makespan win with 0 added misses → repeat.

```
scripts/run_codesign_loop.py --workload data/toplevel/_flight_deployed_matched_board.json --max-rounds 3
```

**Baseline → final: 126.6 → 112.5 ms (-11.2%)** — levers applied: ['shard'].

| round | lever | before (ms) | after (ms) | % | misses |
|--:|--|--:|--:|--:|--:|
| 1 | +shard | 126.6 | 112.5 | -11.2 | 0 |

Honest note — fuse: NOT applied — roofline decision-aid says the stack is compute-bound (max fusible-epilogue ceiling 16% on dronet); fusion collapses dispatches (scheduling) but is measured ~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.

Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, `round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), `specs/` (every candidate spec), `loop_log.txt`.

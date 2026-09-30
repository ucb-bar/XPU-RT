# Automatic co-design loop — `networks_k1_tri_small_feedback`

Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →
measure each → accept the largest measured makespan win with 0 added misses → repeat.

```
scripts/run_codesign_loop.py --workload data/toplevel/networks_k1_tri_small_feedback.json --max-rounds 3
```

**Baseline → final: 26.6 → 20.1 ms (-24.5%)** — levers applied: ['ime', 'shard'].

| round | lever | before (ms) | after (ms) | % | misses |
|--:|--|--:|--:|--:|--:|
| 1 | +ime | 26.6 | 20.1 | -24.5 | 0 |
| 2 | +shard | 20.1 | 20.1 | -0.0 | 0 |

Honest note — fuse: NOT applied — roofline decision-aid says the stack is compute-bound (max fusible-epilogue ceiling 16% on dronet); fusion collapses dispatches (scheduling) but is measured ~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.

Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, `round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), `specs/` (every candidate spec), `loop_log.txt`.

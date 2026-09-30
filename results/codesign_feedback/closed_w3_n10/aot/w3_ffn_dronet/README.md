# Automatic co-design loop — `w3_ffn_dronet`

Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →
measure each → accept the largest measured makespan win with 0 added misses → repeat.

```
scripts/run_codesign_loop.py --workload data/toplevel/scaling/w3_ffn_dronet.json --max-rounds 4
```

**Baseline → final: 74.6 → 59.6 ms (-91.1%)** — levers applied: ['shard'].

| round | lever | before (ms) | after (ms) | % | misses |
|--:|--|--:|--:|--:|--:|
| 1 | +shard | 74.6 | 59.6 | -91.1 | 5 |
| 2 | _(none accepted)_ | 7.794 | — | — | — |

Honest note — fuse: NOT applied — roofline decision-aid says the stack is compute-bound (max fusible-epilogue ceiling 16% on dronet); fusion collapses dispatches (scheduling) but is measured ~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.

Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, `round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), `specs/` (every candidate spec), `loop_log.txt`.

# Automatic co-design loop — `w5_ffn_dronet_yolo`

Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →
measure each → accept the largest measured makespan win with 0 added misses → repeat.

```
scripts/run_codesign_loop.py --workload data/toplevel/scaling/w5_ffn_dronet_yolo.json --max-rounds 3
```

**Baseline → final: 74.6 → 63.9 ms (-18.4%)** — levers applied: ['ime', 'shard'].

| round | lever | before (ms) | after (ms) | % | misses |
|--:|--|--:|--:|--:|--:|
| 1 | +ime | 74.6 | 69.6 | -9.8 | 11 |
| 2 | +shard | 69.6 | 63.9 | -9.5 | 5 |
| 3 | _(none accepted)_ | 89.572 | — | — | — |

Honest note — fuse: NOT applied — roofline decision-aid says the stack is compute-bound (max fusible-epilogue ceiling 16% on dronet); fusion collapses dispatches (scheduling) but is measured ~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.

Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, `round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), `specs/` (every candidate spec), `loop_log.txt`.

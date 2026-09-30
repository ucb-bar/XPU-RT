# Automatic co-design loop — `w5_ffn_dronet_yolo`

Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →
measure each → accept the largest measured makespan win with 0 added misses → repeat.

```
scripts/run_codesign_loop.py --workload data/toplevel/scaling/w5_ffn_dronet_yolo.json --max-rounds 3
```

**Baseline → final: 74.6 → 64.0 ms (-14.2%)** — levers applied: ['shard:dronet', 'shard:ffn_block', 'ime'].

| round | lever | before (ms) | after (ms) | % | misses |
|--:|--|--:|--:|--:|--:|
| 1 | +shard:dronet | 74.6 | 79.2 | 6.2 | 74 |
| 2 | +shard:ffn_block | 79.2 | 66.0 | -16.7 | 74 |
| 3 | +ime | 66.0 | 64.0 | -3.1 | 74 |

Honest note — fuse: NOT applied — roofline decision-aid says the stack is compute-bound (max fusible-epilogue ceiling 16% on dronet); fusion collapses dispatches (scheduling) but is measured ~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.

Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, `round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), `specs/` (every candidate spec), `loop_log.txt`.

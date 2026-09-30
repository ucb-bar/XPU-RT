# Automatic co-design loop — `w4_ffn_dronet_sensor`

Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →
measure each → accept the largest measured makespan win with 0 added misses → repeat.

```
scripts/run_codesign_loop.py --workload data/toplevel/scaling/w4_ffn_dronet_sensor.json --max-rounds 4
```

**Baseline → final: 74.6 → 84.8 ms (--13.7%)** — levers applied: ['shard:dronet', 'shard:fused_full'].

| round | lever | before (ms) | after (ms) | % | misses |
|--:|--|--:|--:|--:|--:|
| 1 | +shard:dronet | 74.6 | 110.7 | 48.4 | 139 |
| 2 | +shard:fused_full | 110.7 | 84.8 | -23.4 | 20 |
| 3 | _(none accepted)_ | 84.808 | — | — | — |

Honest note — fuse: NOT applied — roofline decision-aid says the stack is compute-bound (max fusible-epilogue ceiling 16% on dronet); fusion collapses dispatches (scheduling) but is measured ~+0.85% on cycles, so the loop does not credit a makespan gain it cannot measure.

Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, `round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), `specs/` (every candidate spec), `loop_log.txt`.

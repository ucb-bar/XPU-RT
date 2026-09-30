# Automatic co-design loop — `networks_k1_mb_3model_4hz_yolo_ctrl`

Fully automatic ModelBlaster↔XPU-RT feedback loop: solve → propose every lever →
measure each → accept the largest measured makespan win with 0 added misses → repeat.

```
scripts/run_codesign_loop.py --workload data/toplevel/networks_k1_mb_3model_4hz_yolo_ctrl.json --max-rounds 3
```

**Baseline → final: 898.1 → 866.4 ms (-3.5%)** — levers applied: ['unfuse'].

| round | lever | before (ms) | after (ms) | % | misses |
|--:|--|--:|--:|--:|--:|
| 1 | +unfuse | 898.1 | 866.4 | -3.5 | 0 |

Honest note — fuse: decision-aid unavailable.

Artifacts: `loop_report.json`, `makespan_vs_round.{png,pdf}`, `round_<k>_<lever>_gantt.{png,pdf}` (IME dispatches drawn darker + hatched), `specs/` (every candidate spec), `loop_log.txt`.

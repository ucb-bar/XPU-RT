# XPU-RT arm `best45alt2long`

45 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 40.1 ms |
| control gap (mean) | 10.0 ms |
| control gap (max) | 10.03 ms |
| camera rate | 45 Hz |
| meets its window | yes |
| command rate | 100 Hz |

## Rebuilding it

No solve to re-run: the arrangement was built by hand and measured directly. Its run
manifest under `results/codesign_feedback/xpurt_long/` records what executed.

Registry entry: `scripts/measured_timing.py` → `XPURT_POINTS["best45alt2long"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


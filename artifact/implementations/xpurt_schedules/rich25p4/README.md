# XPU-RT arm `rich25p4`

25 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 29.6 ms |
| control gap (mean) | 10.0 ms |
| camera rate | 25 Hz |
| meets its window | yes |
| command rate | 100 Hz |

## Rebuilding it

No solve to re-run: the arrangement was built by hand and measured directly. Its run
manifest under `results/codesign_feedback/xpurt_long/` records what executed.

Registry entry: `scripts/measured_timing.py` → `XPURT_POINTS["rich25p4"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


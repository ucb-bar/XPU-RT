# XPU-RT arm `p45detr`

CP-SAT (hard windows), placement unconstrained, single-worker solve, 45 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 30.07 ms |
| camera→control (p95) | 43.6 ms |
| control gap (mean) | 9.83 ms |
| control gap (max) | 24.27 ms |
| frames late | 0/63 |
| command rate | 102 Hz |

## Rebuilding it

Executed schedule: `scheduled_wh_chain45_free_det1_cpsat_profiled_clamped.json`

Board recipe: `run_xpurt_schedule.py --networks-json data/toplevel/wh_chain45_free_det1.json` then `board_partitioned30.sh <table> p45det gen/mb_shard_nav`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["p45detr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


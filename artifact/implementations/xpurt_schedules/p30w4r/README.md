# XPU-RT arm `p30w4r`

Greedy, partitioned, yolo on the whole P cluster.

## What the board measured

| | |
|---|---|
| camera→control (median) | 36.75 ms |
| control gap (mean) | 10.0 ms |
| control gap (max) | 10.03 ms |
| frames late | 0/36 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `fig_p30w4_greedy_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> p30w4` then `board_stage2.sh p30w4 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["p30w4r"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


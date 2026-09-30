# XPU-RT arm `w2pg36r`

Greedy + measured shard costs, 36 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 30.1 ms |
| control gap (mean) | 9.98 ms |
| control gap (max) | 17.23 ms |
| frames late | 0/96 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `fig_w2pg36_greedy_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> w2pg36` then `board_stage2.sh w2pg36 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["w2pg36r"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


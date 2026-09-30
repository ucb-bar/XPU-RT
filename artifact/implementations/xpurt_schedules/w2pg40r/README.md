# XPU-RT arm `w2pg40r`

Greedy + measured shard costs, 40 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 39.8 ms |
| control gap (mean) | 10.06 ms |
| control gap (max) | 16.68 ms |
| frames late | 0/108 |
| command rate | 99 Hz |

## Rebuilding it

Executed schedule: `fig_w2pg40_greedy_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> w2pg40` then `board_stage2.sh w2pg40 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["w2pg40r"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


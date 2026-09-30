# XPU-RT arm `agreedyr`

Greedy (list scheduling).

## What the board measured

| | |
|---|---|
| camera→control (median) | 748.0 ms |
| camera→control (p95) | 1064.7 ms |
| control gap (mean) | 13.25 ms |
| control gap (max) | 1036.96 ms |
| control gaps over 15 ms | 3/294 |
| frames late | 120/120 |
| command rate | 75 Hz |

## Rebuilding it

Executed schedule: `fig_a_greedy_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> a` then `board_stage2.sh a <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["agreedyr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


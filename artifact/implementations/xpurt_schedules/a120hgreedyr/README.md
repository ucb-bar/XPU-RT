# XPU-RT arm `a120hgreedyr`

Greedy, 120 Hz camera, 200 ms table.

## What the board measured

| | |
|---|---|
| camera→control (median) | 566.3 ms |
| control gap (mean) | 35.77 ms |
| control gap (max) | 513.81 ms |
| frames late | 36/36 |
| meets its window | no |
| command rate | 28 Hz |

## Rebuilding it

Executed schedule: `fig_a120h_greedy_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> a120h` then `board_stage2.sh a120h <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["a120hgreedyr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


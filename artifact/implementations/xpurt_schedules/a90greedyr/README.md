# XPU-RT arm `a90greedyr`

Greedy, 90 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 947.5 ms |
| control gap (mean) | 25.67 ms |
| control gap (max) | 991.65 ms |
| frames late | 111/111 |
| command rate | 39 Hz |

## Rebuilding it

Executed schedule: `fig_a90_greedy_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> a90` then `board_stage2.sh a90 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["a90greedyr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


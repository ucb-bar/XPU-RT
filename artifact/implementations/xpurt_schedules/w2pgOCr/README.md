# XPU-RT arm `w2pgOCr`

Greedy, shard costs measured, perception in two camera periods.

## What the board measured

| | |
|---|---|
| camera→control (median) | 36.0 ms |
| camera→control (p95) | 44.2 ms |
| control gap (mean) | 9.99 ms |
| control gap (max) | 17.02 ms |
| frames late | 0/120 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `fig_w2p_greedy_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> w2p` then `board_stage2.sh w2p <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["w2pgOCr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


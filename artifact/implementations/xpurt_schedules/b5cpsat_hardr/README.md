# XPU-RT arm `b5cpsat_hardr`

CP-SAT (hard windows), 90 Hz + heavier stack.

## What the board measured

| | |
|---|---|
| camera→control (median) | 58.4 ms |
| camera→control (p95) | 68.8 ms |
| control gap (mean) | 9.95 ms |
| control gap (max) | 16.35 ms |
| frames late | 0/108 |
| command rate | 101 Hz |

## Rebuilding it

Executed schedule: `fig_b5_cpsat_hard_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> b5` then `board_stage2.sh b5 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["b5cpsat_hardr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


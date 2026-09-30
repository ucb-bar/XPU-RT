# XPU-RT arm `p30cpsat_hardr`

CP-SAT (hard windows), partitioned, one hart per yolo frame.

## What the board measured

| | |
|---|---|
| camera→control (median) | 60.08 ms |
| control gap (mean) | 10.0 ms |
| control gap (max) | 10.08 ms |
| frames late | 0/36 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `fig_p30_cpsat_hard_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> p30` then `board_stage2.sh p30 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["p30cpsat_hardr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


# XPU-RT arm `fb30r1r`

CP-SAT, per-rate feedback round 1, 30 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 30.1 ms |
| control gap (mean) | 9.99 ms |
| control gap (max) | 19.7 ms |
| frames late | 0/81 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `fig_fb30r1_cpsat_hard_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> fb30r1` then `board_stage2.sh fb30r1 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["fb30r1r"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


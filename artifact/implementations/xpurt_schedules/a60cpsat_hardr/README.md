# XPU-RT arm `a60cpsat_hardr`

CP-SAT (hard windows), 60 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 58.5 ms |
| control gap (mean) | 10.06 ms |
| control gap (max) | 15.3 ms |
| frames late | 0/72 |
| command rate | 99 Hz |

## Rebuilding it

Executed schedule: `fig_a60_cpsat_hard_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> a60` then `board_stage2.sh a60 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["a60cpsat_hardr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


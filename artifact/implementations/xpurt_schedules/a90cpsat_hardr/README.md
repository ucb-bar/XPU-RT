# XPU-RT arm `a90cpsat_hardr`

CP-SAT (hard windows), 90 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 55.9 ms |
| control gap (mean) | 10.0 ms |
| control gap (max) | 15.04 ms |
| frames late | 0/108 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `fig_a90_cpsat_hard_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> a90` then `board_stage2.sh a90 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["a90cpsat_hardr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


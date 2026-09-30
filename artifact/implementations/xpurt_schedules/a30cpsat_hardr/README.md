# XPU-RT arm `a30cpsat_hardr`

CP-SAT (hard windows), 30 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 55.2 ms |
| control gap (mean) | 9.98 ms |
| control gap (max) | 15.44 ms |
| frames late | 0/36 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `fig_a30_cpsat_hard_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> a30` then `board_stage2.sh a30 <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["a30cpsat_hardr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


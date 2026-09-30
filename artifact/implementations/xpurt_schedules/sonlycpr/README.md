# XPU-RT arm `sonlycpr`

CP-SAT (hard windows), shard costs measured.

## What the board measured

| | |
|---|---|
| camera→control (median) | 53.2 ms |
| camera→control (p95) | 62.6 ms |
| control gap (mean) | 10.03 ms |
| control gap (max) | 20.52 ms |
| frames late | 0/120 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `fig_sonly_cpsat_hard_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> sonly` then `board_stage2.sh sonly <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["sonlycpr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


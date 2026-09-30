# XPU-RT arm `p30freer`

CP-SAT (hard windows), placement unconstrained, conv on the IME.

## What the board measured

| | |
|---|---|
| camera→control (median) | 26.75 ms |
| camera→control (p95) | 30.1 ms |
| control gap (mean) | 9.99 ms |
| control gap (max) | 18.06 ms |
| control gaps over 15 ms | 3/144 |
| frames late | 0/36 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `scheduled_wh_chain30_free_cpsat_profiled_clamped.json`

Board recipe: `run_xpurt_schedule.py --networks-json data/toplevel/wh_chain30_free.json` then `board_partitioned30.sh <table> p30free gen/mb_shard_nav`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["p30freer"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


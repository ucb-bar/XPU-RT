# XPU-RT arm `p45freer`

CP-SAT (hard windows), placement unconstrained, conv on the IME, 45 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 27.46 ms |
| camera→control (p95) | 37.2 ms |
| control gap (mean) | 9.92 ms |
| control gap (max) | 20.19 ms |
| control gaps over 15 ms | 6/147 |
| frames late | 0/66 |
| command rate | 101 Hz |

## Rebuilding it

Executed schedule: `scheduled_wh_chain45_free_cpsat_profiled_clamped.json`

Board recipe: `run_xpurt_schedule.py --networks-json data/toplevel/wh_chain45_free.json` then `board_partitioned30.sh <table> p45free gen/mb_shard_nav`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["p45freer"]`, re-derived from the
traces by `measured_timing.py --verify`.

## Figures drawing this arm

* [`showdown_45hz_solver_vs_rospinned_s1007`](../../figures/showdown_45hz_solver_vs_rospinned_s1007/)

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


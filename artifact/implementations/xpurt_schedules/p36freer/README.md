# XPU-RT arm `p36freer`

CP-SAT (hard windows), placement unconstrained, conv on the IME, 36 Hz camera.

## What the board measured

| | |
|---|---|
| camera→control (median) | 25.82 ms |
| camera→control (p95) | 34.3 ms |
| control gap (mean) | 9.84 ms |
| control gap (max) | 19.69 ms |
| control gaps over 15 ms | 9/144 |
| frames late | 0/42 |
| command rate | 102 Hz |

## Rebuilding it

Executed schedule: `scheduled_wh_chain36_free_cpsat_profiled_clamped.json`

Board recipe: `run_xpurt_schedule.py --networks-json data/toplevel/wh_chain36_free.json` then `board_partitioned30.sh <table> p36free gen/mb_shard_nav`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["p36freer"]`, re-derived from the
traces by `measured_timing.py --verify`.

## Figures drawing this arm

* [`showdown_36hz_solver_vs_rosallhart_s1006`](../../figures/showdown_36hz_solver_vs_rosallhart_s1006/)
* [`showdown_36hz_solver_vs_rosallhart_s1006_ladder`](../../figures/showdown_36hz_solver_vs_rosallhart_s1006_ladder/)

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


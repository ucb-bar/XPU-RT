# XPU-RT arm `p30efullr`

CP-SAT (hard windows), both clusters whole, conv on the IME.

## What the board measured

| | |
|---|---|
| camera→control (median) | 25.3 ms |
| camera→control (p95) | 30.2 ms |
| control gap (mean) | 10.0 ms |
| control gap (max) | 29.88 ms |
| control gaps over 15 ms | 3/144 |
| frames late | 0/36 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `scheduled_wh_chain30_part4ime_efull_cpsat_profiled_clamped.json`

Board recipe: `run_xpurt_schedule.py --networks-json data/toplevel/wh_chain30_part4ime_efull.json` then `board_partitioned30.sh <table> p30efull gen/mb_shard_nav`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["p30efullr"]`, re-derived from the
traces by `measured_timing.py --verify`.

No figure draws this arm; it is here because it was measured.

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


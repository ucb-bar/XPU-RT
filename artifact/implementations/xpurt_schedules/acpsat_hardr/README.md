# XPU-RT arm `acpsat_hardr`

CP-SAT (hard windows).

## What the board measured

| | |
|---|---|
| camera→control (median) | 56.8 ms |
| camera→control (p95) | 72.3 ms |
| control gap (mean) | 10.0 ms |
| control gap (max) | 19.36 ms |
| control gaps over 15 ms | 12/294 |
| frames late | 0/120 |
| command rate | 100 Hz |

## Rebuilding it

Executed schedule: `fig_a_cpsat_hard_clamped.json`

Board recipe: `solve_stage2_hard.sh <spec> a` then `board_stage2.sh a <spec>`

Needs the K1 board and the kernels built by `artifact/01_board/`.

Registry entry: `scripts/measured_timing.py` → `SOLVER_ARMS["acpsat_hardr"]`, re-derived from the
traces by `measured_timing.py --verify`.

## Figures drawing this arm

* [`showdown_45hz_pinned_vs_rosdefault_s1000`](../../figures/showdown_45hz_pinned_vs_rosdefault_s1000/)
* [`showdown_45hz_pinned_vs_rosdefault_s1003`](../../figures/showdown_45hz_pinned_vs_rosdefault_s1003/)
* [`showdown_45hz_pinned_vs_rospinned_s1011`](../../figures/showdown_45hz_pinned_vs_rospinned_s1011/)

Ranking: [`xpurt_arm_ranking.md`](../../../../docs/Evaluation/xpurt_arm_ranking.md) · run index: [`run_index.md`](../../../../docs/Artifact/run_index.md)


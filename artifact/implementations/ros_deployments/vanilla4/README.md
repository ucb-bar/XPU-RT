# ROS 2 deployment `vanilla4`

Measured at 7 camera rate(s) over 21 runs: 15 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 90 Hz, 120 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4` |
| traced binary | `ros_mb_chain_traced_pool` |
| executor | `single` |
| QoS depth | `10` |
| control timer Hz | `100` |
| control mode | `chained` — computed from the frame that produced the goal |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 15 | 3 | 31.0 | 66.67 | 15 Hz | 268→255 |  |
| 25 | 3 | 31.0 | 40.00 | 25 Hz | 446→424 |  |
| 30 | 3 | 31.1 | 33.33 | 30 Hz | 536→509 |  |
| 45 | 3 | 242.4 | 25.88 | 39 Hz | 802→647 | ✔ |
| 60 | 3 | 189.4 | 25.64 | 39 Hz | 1076→656 |  |
| 90 | 3 | 136.9 | 25.68 | 39 Hz | 1604→657 | ✔ |
| 120 | 3 | 110.6 | 26.12 | 38 Hz | 2139→646 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="15 25 30 45 60 90 120" scripts/ros_traced_matrix.sh vanilla4
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4_r<n>/`.

## Figures drawing this deployment

* [`showdown_45hz_pinned_vs_rosdefault_s1000`](../../figures/showdown_45hz_pinned_vs_rosdefault_s1000/)
* [`showdown_45hz_pinned_vs_rosdefault_s1003`](../../figures/showdown_45hz_pinned_vs_rosdefault_s1003/)

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


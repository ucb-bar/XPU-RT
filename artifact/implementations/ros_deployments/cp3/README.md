# ROS 2 deployment `cp3`

Measured at 7 camera rate(s) over 15 runs: 15 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 75 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `cp3` |
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
| 15 | 3 | 30.6 | 66.66 | 15 Hz | 268→254 | ✔ |
| 25 | 3 | 30.6 | 40.00 | 25 Hz | 445→424 | ✔ |
| 30 | 3 | 30.7 | 33.40 | 30 Hz | 545→508 | ✔ |
| 45 | 3 | 56.2 | 25.81 | 39 Hz | 693→658 | ✔ |
| 60 | 1 | 56.2 | 25.63 | 39 Hz | 693→661 | ✔ |
| 75 | 1 | 56.1 | 25.60 | 39 Hz | 696→662 | ✔ |
| 90 | 1 | 56.0 | 25.57 | 39 Hz | 702→663 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="15 25 30 45 60 75 90" scripts/ros_traced_matrix.sh cp3
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_cp3_r<n>/`.

## Figures drawing this deployment

* [`showdown_45hz_pinned_vs_rospinned_s1011`](../../figures/showdown_45hz_pinned_vs_rospinned_s1011/)
* [`showdown_45hz_solver_vs_rospinned_s1007`](../../figures/showdown_45hz_solver_vs_rospinned_s1007/)

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


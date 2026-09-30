# ROS 2 deployment `cp3n4_d`

Measured at 1 camera rate(s) over 3 runs: 30 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `cp3n4` |
| variant (suffix) | `_d` |
| traced binary | `ros_mb_chain_traced_pool_nav4` |
| executor | `single` |
| QoS depth | `10` |
| control timer Hz | `100` |
| control mode | `chained` — computed from the frame that produced the goal |
| nav pool | `4` on harts `4,5,6,7` |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 30 | 3 | 30.1 | 33.40 | 30 Hz | 545→508 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="30" SUFFIX=_d scripts/ros_traced_matrix.sh cp3n4
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_cp3n4_d_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


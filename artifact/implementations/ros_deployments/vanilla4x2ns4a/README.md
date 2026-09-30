# ROS 2 deployment `vanilla4x2ns4a`

Pipelining by hand: the camera alternates frames between two perception processes (4-hart pools on 0-3 and 4-7); nav, control unpinned

Measured at 1 camera rate(s) over 1 runs: 36 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4x2` |
| variant (suffix) | `ns4a` |
| traced binary | `ros_mb_chain_traced_pool` |
| executor | `single` |
| QoS depth | `10` |
| control timer Hz | `100` |
| control mode | `chained` — computed from the frame that produced the goal |
| nav pool | `4` on harts `0,1,2,3` |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 36 | 1 | 38.7 | 27.79 | 36 Hz | 656→611 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="36" BINSUF=_nav4 NAVPOOL=4 NAVHARTS=0,1,2,3 SUFFIX=ns4a scripts/ros_traced_matrix.sh vanilla4x2
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4x2ns4a_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


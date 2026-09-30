# ROS 2 deployment `vanilla4x2d2`

Pipelining by hand: the camera alternates frames between two perception processes (4-hart pools on 0-3 and 4-7); nav, control unpinned

Measured at 2 camera rate(s) over 2 runs: 30 Hz, 36 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4x2` |
| variant (suffix) | `d2` |
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
| 30 | 1 | 31.6 | 33.33 | 30 Hz | 547→509 | ✔ |
| 36 | 1 | 32.8 | 27.77 | 36 Hz | 656→611 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="30 36" SUFFIX=d2 scripts/ros_traced_matrix.sh vanilla4x2
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4x2d2_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


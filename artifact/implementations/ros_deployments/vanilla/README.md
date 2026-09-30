# ROS 2 deployment `vanilla`

ROS 2 as written out of the box: one node per stage, one process each, unpinned, serial kernels, control in the goal callback

Measured at 6 camera rate(s) over 18 runs: 15 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla` |
| traced binary | `ros_mb_chain_traced` |
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
| 15 | 3 | 53.8 | 66.67 | 15 Hz | 267→254 |  |
| 25 | 3 | 433.3 | 48.06 | 21 Hz | 446→345 |  |
| 30 | 3 | 370.5 | 48.39 | 21 Hz | 535→344 |  |
| 45 | 3 | 264.8 | 48.22 | 21 Hz | 802→347 | ✔ |
| 60 | 3 | 212.1 | 48.33 | 21 Hz | 1069→347 |  |
| 90 | 3 | 159.2 | 48.36 | 21 Hz | 1604→348 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh vanilla
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


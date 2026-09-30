# ROS 2 deployment `vanilla_c50`

ROS 2 as written out of the box: one node per stage, one process each, unpinned, serial kernels, control in the goal callback

Measured at 1 camera rate(s) over 3 runs: 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla` |
| variant (suffix) | `_c50` |
| traced binary | `ros_mb_chain_traced` |
| executor | `single` |
| QoS depth | `10` |
| control timer Hz | `50` |
| control mode | `chained` — computed from the frame that produced the goal |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 45 | 3 | 265.9 | 49.22 | 20 Hz | 806→341 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="45" CTRL_HZ=50 SUFFIX=_c50 scripts/ros_traced_matrix.sh vanilla
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla_c50_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


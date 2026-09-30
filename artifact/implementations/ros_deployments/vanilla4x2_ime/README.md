# ROS 2 deployment `vanilla4x2_ime`

Pipelining by hand: the camera alternates frames between two perception processes (4-hart pools on 0-3 and 4-7); nav, control unpinned

Measured at 1 camera rate(s) over 3 runs: 30 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4x2` |
| variant (suffix) | `_ime` |
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
| 30 | 3 | 24.1 | 66.67 | 15 Hz | 546→255 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="30" BINSUF=_ime SUFFIX=_ime scripts/ros_traced_matrix.sh vanilla4x2
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4x2_ime_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


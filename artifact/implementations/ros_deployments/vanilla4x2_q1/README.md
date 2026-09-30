# ROS 2 deployment `vanilla4x2_q1`

Pipelining by hand: the camera alternates frames between two perception processes (4-hart pools on 0-3 and 4-7); nav, control unpinned

Measured at 1 camera rate(s) over 3 runs: 36 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4x2` |
| variant (suffix) | `_q1` |
| traced binary | `ros_mb_chain_traced_pool` |
| executor | `single` |
| QoS depth | `1` |
| control timer Hz | `100` |
| control mode | `chained` — computed from the frame that produced the goal |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 36 | 3 | 32.3 | 27.78 | 36 Hz | 642→611 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="36" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh vanilla4x2
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4x2_q1_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


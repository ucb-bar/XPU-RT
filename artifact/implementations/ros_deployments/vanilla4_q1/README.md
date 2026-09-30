# ROS 2 deployment `vanilla4_q1`

Measured at 2 camera rate(s) over 6 runs: 45 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4` |
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
| 45 | 3 | 42.0 | 25.57 | 39 Hz | 803→663 | ✔ |
| 90 | 3 | 36.7 | 25.61 | 39 Hz | 1612→662 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="45 90" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh vanilla4
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4_q1_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


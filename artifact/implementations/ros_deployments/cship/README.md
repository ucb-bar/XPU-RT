# ROS 2 deployment `cship`

Measured at 3 camera rate(s) over 9 runs: 15 Hz, 25 Hz, 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `cship` |
| traced binary | `ros_mb_chain_traced` |
| node flags | `--executor single --ctrl-mode chained` |
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
| 15 | 3 | 53.9 | 66.67 | 15 Hz | 256→255 |  |
| 25 | 3 | 212.6 | 53.26 | 19 Hz | 319→315 |  |
| 45 | 3 | 212.7 | 53.28 | 19 Hz | 319→316 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="15 25 45" scripts/ros_traced_matrix.sh cship
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_cship_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


# ROS 2 deployment `rvanilla4`

Vanilla4 plus ffn_block and dronet as their own unpinned processes: every core carries a node

Measured at 6 camera rate(s) over 18 runs: 15 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `rvanilla4` |
| traced binary | `ros_mb_chain_traced_rich` |
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
| 15 | 3 | 31.5 | 66.67 | 15 Hz | 267→255 |  |
| 25 | 3 | 31.3 | 40.01 | 25 Hz | 445→424 |  |
| 30 | 3 | 31.5 | 33.33 | 30 Hz | 534→509 |  |
| 45 | 3 | 243.4 | 26.36 | 38 Hz | 800→636 | ✔ |
| 60 | 3 | 190.8 | 26.25 | 38 Hz | 1066→640 |  |
| 90 | 3 | 137.5 | 26.14 | 38 Hz | 1598→645 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh rvanilla4
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_rvanilla4_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


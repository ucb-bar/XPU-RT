# ROS 2 deployment `ship_q1`

Measured at 2 camera rate(s) over 2 runs: 25 Hz, 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `ship` |
| variant (suffix) | `_q1` |
| traced binary | `ros_mb_chain_traced` |
| node flags | `--executor single` |
| executor | `single` |
| QoS depth | `1` |
| control timer Hz | `100` |
| control mode | `timer` — fires on its own clock, holding the last goal between arrivals |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 25 | 1 | 107.7 | 36.59 | 27 Hz | 351→312 |  |
| 45 | 1 | 107.2 | 35.82 | 28 Hz | 387→314 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="25 45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh ship
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_ship_q1_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


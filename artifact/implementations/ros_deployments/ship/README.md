# ROS 2 deployment `ship`

Measured at 12 camera rate(s) over 24 runs: 5 Hz, 8 Hz, 10 Hz, 12 Hz, 15 Hz, 20 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 75 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `ship` |
| traced binary | `ros_mb_chain_traced` |
| node flags | `--executor single` |
| executor | `single` |
| QoS depth | `10` |
| control timer Hz | `100` |
| control mode | `timer` — fires on its own clock, holding the last goal between arrivals |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 5 | 2 | 53.8 | 12.13 | 82 Hz | 86→85 |  |
| 8 | 2 | 53.4 | 14.07 | 71 Hz | 137→136 |  |
| 10 | 2 | 53.4 | 14.44 | 69 Hz | 171→170 |  |
| 12 | 2 | 53.3 | 18.01 | 56 Hz | 205→204 |  |
| 15 | 3 | 53.9 | 24.93 | 40 Hz | 256→255 |  |
| 20 | 2 | 210.5 | 49.65 | 20 Hz | 323→319 |  |
| 25 | 3 | 212.4 | 53.22 | 19 Hz | 320→316 |  |
| 30 | 2 | 210.4 | 52.70 | 19 Hz | 323→319 |  |
| 45 | 3 | 213.2 | 53.41 | 19 Hz | 319→315 |  |
| 60 | 1 | 213.2 | 53.53 | 19 Hz | 318→314 |  |
| 75 | 1 | 213.0 | 53.37 | 19 Hz | 319→315 |  |
| 90 | 1 | 214.4 | 53.71 | 19 Hz | 317→313 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="5 8 10 12 15 20 25 30 45 60 75 90" scripts/ros_traced_matrix.sh ship
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_ship_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


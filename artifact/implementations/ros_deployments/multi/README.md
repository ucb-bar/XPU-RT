# ROS 2 deployment `multi`

Measured at 12 camera rate(s) over 24 runs: 5 Hz, 8 Hz, 10 Hz, 12 Hz, 15 Hz, 20 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 75 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `multi` |
| traced binary | `ros_mb_chain_traced` |
| node flags | `--executor multi` |
| executor | `multi` |
| QoS depth | `10` |
| control timer Hz | `100` |
| control mode | `timer` — fires on its own clock, holding the last goal between arrivals |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 5 | 2 | 53.2 | 10.00 | 100 Hz | 86→85 |  |
| 8 | 2 | 53.2 | 10.00 | 100 Hz | 137→136 |  |
| 10 | 2 | 53.2 | 10.00 | 100 Hz | 171→170 |  |
| 12 | 2 | 53.2 | 10.00 | 100 Hz | 205→204 |  |
| 15 | 3 | 53.8 | 10.00 | 100 Hz | 256→255 |  |
| 20 | 2 | 53.2 | 10.00 | 100 Hz | 341→339 |  |
| 25 | 3 | 433.8 | 10.00 | 100 Hz | 426→334 |  |
| 30 | 2 | 369.5 | 10.00 | 100 Hz | 511→342 |  |
| 45 | 3 | 264.7 | 10.00 | 100 Hz | 766→338 | ✔ |
| 60 | 1 | 212.5 | 10.00 | 100 Hz | 1021→338 |  |
| 75 | 1 | 180.1 | 10.00 | 100 Hz | 1276→340 |  |
| 90 | 1 | 159.4 | 10.00 | 100 Hz | 1531→340 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="5 8 10 12 15 20 25 30 45 60 75 90" scripts/ros_traced_matrix.sh multi
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_multi_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


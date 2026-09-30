# ROS 2 deployment `p3`

Measured at 13 camera rate(s) over 24 runs: 5 Hz, 8 Hz, 10 Hz, 12 Hz, 15 Hz, 20 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 75 Hz, 90 Hz, 120 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `p3` |
| traced binary | `ros_mb_chain_traced_pool` |
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
| 5 | 1 | 30.4 | 10.00 | 100 Hz | 92→85 |  |
| 8 | 1 | 30.5 | 10.00 | 100 Hz | 143→136 |  |
| 10 | 1 | 30.5 | 10.00 | 100 Hz | 179→170 |  |
| 12 | 1 | 30.5 | 10.00 | 100 Hz | 215→204 |  |
| 15 | 3 | 30.5 | 10.00 | 100 Hz | 270→254 |  |
| 20 | 1 | 30.5 | 10.00 | 100 Hz | 359→339 |  |
| 25 | 3 | 30.5 | 10.00 | 100 Hz | 450→424 |  |
| 30 | 1 | 30.4 | 10.00 | 100 Hz | 534→509 | ✔ |
| 45 | 6 | 56.4 | 10.00 | 100 Hz | 690→656 | ✔ |
| 60 | 1 | 55.8 | 10.00 | 100 Hz | 700→665 |  |
| 75 | 1 | 56.1 | 10.00 | 100 Hz | 695→663 |  |
| 90 | 1 | 56.2 | 10.00 | 100 Hz | 695→660 | ✔ |
| 120 | 3 | 56.4 | 10.00 | 100 Hz | 690→656 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="5 8 10 12 15 20 25 30 45 60 75 90 120" scripts/ros_traced_matrix.sh p3
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_p3_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


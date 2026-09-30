# ROS 2 deployment `rp3`

Measured at 3 camera rate(s) over 6 runs: 25 Hz, 45 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `rp3` |
| traced binary | `ros_mb_chain_traced_rich` |
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
| 25 | 2 | 30.9 | 10.00 | 100 Hz | 448→424 |  |
| 45 | 2 | 57.4 | 10.00 | 100 Hz | 678→644 | ✔ |
| 90 | 2 | 56.3 | 10.00 | 100 Hz | 739→652 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="25 45 90" scripts/ros_traced_matrix.sh rp3
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_rp3_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


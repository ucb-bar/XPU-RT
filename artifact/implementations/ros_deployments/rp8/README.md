# ROS 2 deployment `rp8`

The p8 layout with the heavier stack: perception 0-3, nav 4-5, control 6-7, extras unpinned

Measured at 1 camera rate(s) over 3 runs: 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `rp8` |
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
| 45 | 3 | 56.9 | 10.00 | 100 Hz | 685→650 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="45" scripts/ros_traced_matrix.sh rp8
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_rp8_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


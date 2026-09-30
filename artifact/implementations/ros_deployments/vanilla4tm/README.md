# ROS 2 deployment `vanilla4tm`

As vanilla4, control on its own 100 Hz timer in its own unpinned process (the held goal), the other natural default

Measured at 2 camera rate(s) over 6 runs: 45 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4tm` |
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
| 45 | 3 | 242.5 | 10.00 | 100 Hz | 802→649 | ✔ |
| 90 | 3 | 136.6 | 10.00 | 100 Hz | 1603→658 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="45 90" scripts/ros_traced_matrix.sh vanilla4tm
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4tm_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


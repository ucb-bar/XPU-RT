# ROS 2 deployment `vanilla4x2tm`

Pipelining by hand: the camera alternates frames between two perception processes (4-hart pools on 0-3 and 4-7); nav, control unpinned

Measured at 5 camera rate(s) over 15 runs: 25 Hz, 30 Hz, 45 Hz, 75 Hz, 120 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4x2tm` |
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
| 25 | 3 | 31.6 | 10.00 | 100 Hz | 455→424 | ✔ |
| 30 | 3 | 31.6 | 10.00 | 100 Hz | 536→509 | ✔ |
| 45 | 3 | 37.8 | 10.00 | 100 Hz | 805→763 | ✔ |
| 75 | 3 | 291.5 | 10.00 | 100 Hz | 1340→1037 | ✔ |
| 120 | 3 | 200.4 | 10.00 | 100 Hz | 2148→1013 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="25 30 45 75 120" scripts/ros_traced_matrix.sh vanilla4x2tm
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4x2tm_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


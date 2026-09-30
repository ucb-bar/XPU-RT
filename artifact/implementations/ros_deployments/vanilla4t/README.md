# ROS 2 deployment `vanilla4t`

One process, default executor, 4-hart YOLO, control on its 100 Hz timer; nothing pinned by hand

Measured at 6 camera rate(s) over 18 runs: 15 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4t` |
| traced binary | `ros_mb_chain_traced_pool` |
| node flags | `--executor single --yolo-pool 4 --pool-harts 0,1,2,3` |
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
| 15 | 3 | 30.9 | 13.33 | 75 Hz | 256→255 |  |
| 25 | 3 | 30.9 | 13.33 | 75 Hz | 426→425 |  |
| 30 | 3 | 30.9 | 20.00 | 50 Hz | 511→510 |  |
| 45 | 3 | 121.9 | 30.61 | 33 Hz | 555→551 | ✔ |
| 60 | 3 | 121.8 | 30.57 | 33 Hz | 556→552 |  |
| 90 | 3 | 121.2 | 30.44 | 33 Hz | 559→555 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh vanilla4t
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4t_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


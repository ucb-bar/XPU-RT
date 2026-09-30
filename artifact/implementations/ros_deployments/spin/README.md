# ROS 2 deployment `spin`

Measured at 12 camera rate(s) over 24 runs: 5 Hz, 8 Hz, 10 Hz, 12 Hz, 15 Hz, 20 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 75 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `spin` |
| traced binary | `ros_mb_chain_traced_pool` |
| confined to | `taskset -c 0-3` |
| node flags | `--executor single --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0` |
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
| 5 | 2 | 30.5 | 10.53 | 95 Hz | 86→85 |  |
| 8 | 2 | 30.5 | 11.36 | 88 Hz | 137→136 |  |
| 10 | 2 | 30.5 | 11.12 | 90 Hz | 171→170 |  |
| 12 | 2 | 30.4 | 12.32 | 81 Hz | 205→204 |  |
| 15 | 3 | 30.4 | 12.69 | 79 Hz | 256→255 |  |
| 20 | 2 | 30.4 | 12.50 | 80 Hz | 341→340 |  |
| 25 | 3 | 30.4 | 13.35 | 75 Hz | 426→425 |  |
| 30 | 2 | 30.4 | 17.80 | 56 Hz | 511→510 |  |
| 45 | 3 | 119.0 | 29.93 | 33 Hz | 568→564 |  |
| 60 | 1 | 119.4 | 30.02 | 33 Hz | 566→562 |  |
| 75 | 1 | 119.8 | 30.21 | 33 Hz | 563→559 |  |
| 90 | 1 | 119.5 | 30.05 | 33 Hz | 566→562 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="5 8 10 12 15 20 25 30 45 60 75 90" scripts/ros_traced_matrix.sh spin
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_spin_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


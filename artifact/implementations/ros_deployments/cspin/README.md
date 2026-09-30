# ROS 2 deployment `cspin`

Measured at 6 camera rate(s) over 12 runs: 15 Hz, 25 Hz, 45 Hz, 60 Hz, 75 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `cspin` |
| traced binary | `ros_mb_chain_traced_pool` |
| confined to | `taskset -c 0-3` |
| node flags | `--executor single --ctrl-mode chained --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0` |
| executor | `single` |
| QoS depth | `10` |
| control timer Hz | `100` |
| control mode | `chained` — computed from the frame that produced the goal |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 15 | 3 | 30.5 | 66.66 | 15 Hz | 256→255 |  |
| 25 | 3 | 30.4 | 40.00 | 25 Hz | 426→425 |  |
| 45 | 3 | 120.8 | 30.54 | 33 Hz | 557→553 |  |
| 60 | 1 | 121.9 | 30.75 | 33 Hz | 553→549 |  |
| 75 | 1 | 122.1 | 30.71 | 33 Hz | 554→550 |  |
| 90 | 1 | 119.6 | 30.11 | 33 Hz | 564→560 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="15 25 45 60 75 90" scripts/ros_traced_matrix.sh cspin
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_cspin_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


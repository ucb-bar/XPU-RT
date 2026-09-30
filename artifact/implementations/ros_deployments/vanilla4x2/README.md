# ROS 2 deployment `vanilla4x2`

Pipelining by hand: the camera alternates frames between two perception processes (4-hart pools on 0-3 and 4-7); nav, control unpinned

Measured at 11 camera rate(s) over 33 runs: 25 Hz, 30 Hz, 36 Hz, 38 Hz, 40 Hz, 45 Hz, 50 Hz, 60 Hz, 75 Hz, 90 Hz, 120 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4x2` |
| traced binary | `ros_mb_chain_traced_pool` |
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
| 25 | 3 | 31.4 | 40.00 | 25 Hz | 447→424 | ✔ |
| 30 | 3 | 31.4 | 33.33 | 30 Hz | 544→509 | ✔ |
| 36 | 3 | 32.2 | 27.78 | 36 Hz | 656→611 | ✔ |
| 38 | 3 | 36.4 | 26.31 | 38 Hz | 676→644 |  |
| 40 | 3 | 37.5 | 24.99 | 40 Hz | 712→678 | ✔ |
| 45 | 3 | 37.0 | 22.22 | 45 Hz | 804→763 | ✔ |
| 50 | 3 | 36.8 | 20.00 | 50 Hz | 907→848 |  |
| 60 | 3 | 68.4 | 16.63 | 60 Hz | 1068→1016 | ✔ |
| 75 | 3 | 291.9 | 16.19 | 62 Hz | 1337→1038 | ✔ |
| 90 | 3 | 252.8 | 16.20 | 62 Hz | 1611→1034 | ✔ |
| 120 | 3 | 200.0 | 16.38 | 61 Hz | 2148→1025 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="25 30 36 38 40 45 50 60 75 90 120" scripts/ros_traced_matrix.sh vanilla4x2
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_vanilla4x2_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


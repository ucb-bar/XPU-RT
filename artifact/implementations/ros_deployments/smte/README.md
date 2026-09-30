# ROS 2 deployment `smte`

Measured at 12 camera rate(s) over 18 runs: 5 Hz, 8 Hz, 10 Hz, 12 Hz, 15 Hz, 20 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 75 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `smte` |
| traced binary | `ros_mb_chain_traced_pool` |
| confined to | `taskset -c 0-3` |
| node flags | `--executor multi --yolo-pool 4 --pool-harts 0,1,2,3` |
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
| 5 | 2 | 44.9 | 10.00 | 100 Hz | 86→85 |  |
| 8 | 2 | 42.7 | 10.00 | 100 Hz | 137→136 |  |
| 10 | 2 | 41.7 | 10.00 | 100 Hz | 171→170 |  |
| 12 | 2 | 38.6 | 10.00 | 100 Hz | 205→204 |  |
| 15 | 1 | 45.4 | 10.00 | 100 Hz | 256→255 |  |
| 20 | 1 | 41.8 | 10.00 | 100 Hz | 341→339 |  |
| 25 | 1 | 391.1 | 10.00 | 100 Hz | 425→407 |  |
| 30 | 1 | 364.9 | 10.00 | 100 Hz | 510→391 |  |
| 45 | 3 | 258.6 | 10.00 | 100 Hz | 766→385 |  |
| 60 | 1 | 207.8 | 10.00 | 100 Hz | 1021→389 |  |
| 75 | 1 | 174.3 | 10.00 | 100 Hz | 1276→397 |  |
| 90 | 1 | 154.8 | 10.00 | 100 Hz | 1531→387 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="5 8 10 12 15 20 25 30 45 60 75 90" scripts/ros_traced_matrix.sh smte
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_smte_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


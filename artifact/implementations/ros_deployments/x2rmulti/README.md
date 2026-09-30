# ROS 2 deployment `x2rmulti`

The multi-threaded executor, under the combined load

Measured at 1 camera rate(s) over 2 runs: 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `x2rmulti` |
| traced binary | `ros_mb_chain_traced_rich` |
| node flags | `--executor multi --cameras 2 --nodes camera,perception,camera2,perception2,nav,control,extra --extra ffn_block:10,dronet:30 --yolo-pool 4 --pool-harts 0,1,2,3` |
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
| 45 | 2 | 247.9 | 10.00 | 100 Hz | 1532→503 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="45" scripts/ros_traced_matrix.sh x2rmulti
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_x2rmulti_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


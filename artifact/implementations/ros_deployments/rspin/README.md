# ROS 2 deployment `rspin`

Measured at 2 camera rate(s) over 4 runs: 25 Hz, 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `rspin` |
| traced binary | `ros_mb_chain_traced_rich` |
| confined to | `taskset -c 0-3` |
| node flags | `--executor single --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0 --extra ffn_block:10,dronet:30` |
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
| 25 | 2 | 96.3 | 28.64 | 35 Hz | 297→296 |  |
| 45 | 2 | 198.2 | 53.25 | 19 Hz | 319→316 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="25 45" scripts/ros_traced_matrix.sh rspin
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_rspin_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


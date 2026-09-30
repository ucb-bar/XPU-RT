# ROS 2 deployment `rmulti`

Measured at 2 camera rate(s) over 6 runs: 25 Hz, 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `rmulti` |
| traced binary | `ros_mb_chain_traced_rich` |
| node flags | `--executor multi --yolo-pool 4 --pool-harts 0,1,2,3 --extra ffn_block:10,dronet:30` |
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
| 25 | 2 | 34.0 | 10.00 | 100 Hz | 426→424 |  |
| 45 | 4 | 248.0 | 10.02 | 100 Hz | 766→506 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="25 45" scripts/ros_traced_matrix.sh rmulti
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_rmulti_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


# ROS 2 deployment `multi_c200`

Measured at 1 camera rate(s) over 1 runs: 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `multi` |
| variant (suffix) | `_c200` |
| traced binary | `ros_mb_chain_traced` |
| node flags | `--executor multi` |
| executor | `multi` |
| QoS depth | `10` |
| control timer Hz | `200` |
| control mode | `timer` — fires on its own clock, holding the last goal between arrivals |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 45 | 1 | 265.0 | 5.00 | 200 Hz | 766→337 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="45" CTRL_HZ=200 SUFFIX=_c200 scripts/ros_traced_matrix.sh multi
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_multi_c200_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


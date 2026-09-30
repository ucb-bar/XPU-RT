# ROS 2 deployment `spin_c200`

Measured at 1 camera rate(s) over 1 runs: 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `spin` |
| variant (suffix) | `_c200` |
| traced binary | `ros_mb_chain_traced_pool` |
| confined to | `taskset -c 0-3` |
| node flags | `--executor single --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0` |
| executor | `single` |
| QoS depth | `10` |
| control timer Hz | `200` |
| control mode | `timer` — fires on its own clock, holding the last goal between arrivals |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 45 | 1 | 119.7 | 30.11 | 33 Hz | 565→561 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="45" CTRL_HZ=200 SUFFIX=_c200 scripts/ros_traced_matrix.sh spin
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_spin_c200_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


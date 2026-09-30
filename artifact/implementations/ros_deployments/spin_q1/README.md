# ROS 2 deployment `spin_q1`

Measured at 2 camera rate(s) over 2 runs: 25 Hz, 45 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `spin` |
| variant (suffix) | `_q1` |
| traced binary | `ros_mb_chain_traced_pool` |
| confined to | `taskset -c 0-3` |
| node flags | `--executor single --yolo-pool 4 --pool-harts 0,1,2,3 --pin-main 0` |
| executor | `single` |
| QoS depth | `1` |
| control timer Hz | `100` |
| control mode | `timer` — fires on its own clock, holding the last goal between arrivals |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 25 | 1 | 30.4 | 13.38 | 75 Hz | 426→425 |  |
| 45 | 1 | 60.1 | 21.37 | 47 Hz | 682→557 | ✔ |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="25 45" QOS=1 SUFFIX=_q1 scripts/ros_traced_matrix.sh spin
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_spin_q1_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


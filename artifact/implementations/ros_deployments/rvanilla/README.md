# ROS 2 deployment `rvanilla`

The same plus ffn_block and dronet, each its own unpinned process

Measured at 6 camera rate(s) over 18 runs: 15 Hz, 25 Hz, 30 Hz, 45 Hz, 60 Hz, 90 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `rvanilla` |
| traced binary | `ros_mb_chain_traced_rich` |
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
| 15 | 3 | 721.2 | 83.45 | 12 Hz | 267→196 |  |
| 25 | 3 | 470.0 | 83.92 | 12 Hz | 445→197 |  |
| 30 | 3 | 406.3 | 83.82 | 12 Hz | 533→198 |  |
| 45 | 3 | 300.7 | 83.95 | 12 Hz | 800→199 | ✔ |
| 60 | 3 | 247.2 | 83.85 | 12 Hz | 1067→200 |  |
| 90 | 3 | 195.0 | 83.91 | 12 Hz | 1600→200 |  |

## Re-measuring it

Needs the K1 board and the node built by `artifact/01_board/`:

```bash
RATES="15 25 30 45 60 90" scripts/ros_traced_matrix.sh rvanilla
```

Runs land in `results/codesign_feedback/ros_traced/<hz>_rvanilla_r<n>/`.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


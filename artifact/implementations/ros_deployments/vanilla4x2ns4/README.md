# ROS 2 deployment `vanilla4x2ns4`

Pipelining by hand: the camera alternates frames between two perception processes (4-hart pools on 0-3 and 4-7); nav, control unpinned

Measured at 1 camera rate(s) over 1 runs: 36 Hz.

## What it runs

| | |
|---|---|
| layout (arm) | `vanilla4x2` |
| variant (suffix) | `ns4` |
| traced binary | `ros_mb_chain_traced_pool` |
| executor | `single` |
| QoS depth | `10` |
| control timer Hz | `100` |

## What it measured

Median over the replicates at each rate, from each run's own `summary.json`. A **✔** in
*in a registry* marks a rate whose numbers a figure or a document prints, re-derived by
`measured_timing.py --verify`; the rest were measured and not drawn.

| camera Hz | runs | camera→goal ms | control gap ms | control rate | frames→goals | in a registry |
|---|---|---|---|---|---|---|
| 36 | 1 | — | — | — | 676→0 |  |

## Re-measuring it

**No command is derived for this deployment.** Its 1 run(s) recorded no goal, and only
1 of the deployment's processes wrote a manifest, so the knobs it was run with are not
recoverable from the run. What is known: the layout is `vanilla4x2` and the tag carries `ns4`.
[`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) §4 carries the row for it, and points at the
sibling deployments whose command form is recorded.

No figure draws this deployment; it is here because it was measured.

Ranking and what each choice buys: [`ros_arm_ranking.md`](../../../../docs/Baselines/ros_arm_ranking.md) · full layout: [`ros_arms_catalog.md`](../../../../docs/Baselines/ros_arms_catalog.md)


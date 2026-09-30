# ROS 2 deployments — every arrangement measured on the K1

One directory per deployment: **65** of them, from 443 runs under `results/codesign_feedback/ros_traced/`.
Each says what the arrangement is, every camera rate it ran at, what it measured, the one
command that re-measures it, and which figures draw it.

Generated from the run directories and `scripts/ros_traced_matrix.sh` itself, so a deployment
gets a page the day it is first run. What each arrangement *is* in full — process count,
pinning, executor, pool, QoS, control mode — is in [`ros_arms_catalog.md`](../../../docs/Baselines/ros_arms_catalog.md);
how they rank is [`ros_arm_ranking.md`](../../../docs/Baselines/ros_arm_ranking.md).

*Best control rate* is over every rate the deployment ran at, from the runs' own
summaries rather than from a registry, so a rate that was measured and never drawn counts.
`—` means no run of that deployment recorded a control gap at any rate.

**Read the control rate together with the control mode.** A `chained` deployment computes a
command from the frame that produced the goal, so its command rate *is* the rate fresh
information arrives. A `timer` deployment fires on its own clock and, between goals, acts on
the goal it is still holding — so it reports a high command rate while the information under
that command is as old as the camera→goal column beside it. Both are in here because both
are real deployments; [`ros_arm_ranking.md`](../../../docs/Baselines/ros_arm_ranking.md) §3 measures
what each choice buys, and §2 is where the two are compared as baselines.

**Measured over every deployment and rate, not only the drawn pair.** Of the 72 (deployment, rate) pairs whose
control node is chained, **0** command faster than their camera — 31 command at the camera rate and the
rest slower still, because a chain that cannot keep up emits commands more slowly than frames arrive.
The 124 timer pairs are not bound that way: they re-send a held goal, so cadence and freshness part company.

The mode here is read from the process that runs the control node, over every run of the
deployment rather than the lowest-sorted one, because the earliest runs of eight arms
predate the field. [`ros_arms_catalog.md`](../../../docs/Baselines/ros_arms_catalog.md) reads one
run and so prints `p3`, `p8`, `yproc`, `nproc`, `multi`, `spin`, `ship` and `smte` as chained;
[`ros_arm_ranking.md`](../../../docs/Baselines/ros_arm_ranking.md) §5.1 sets out which reading the
measurements support — a chained control node cannot emit every 10 ms off a 45 Hz camera.

| deployment | based on | control | camera rates | runs | best control rate | camera→goal at that rate | drawn in |
|---|---|---|---|---|---|---|---|
| [`cp3`](cp3/) | `cp3` | chained | 15, 25, 30, 45, 60, 75, 90 Hz | 15 | 39 Hz | 56.0 ms | showdown_45hz_pinned_vs_rospinned_s1011, showdown_45hz_solver_vs_rospinned_s1007 |
| [`cp3n4`](cp3n4/) | `cp3n4` | chained | 30 Hz | 3 | 30 Hz | 30.1 ms | — |
| [`cp3n4_d`](cp3n4_d/) | `cp3n4` + `_d` | chained | 30 Hz | 3 | 30 Hz | 30.1 ms | — |
| [`cship`](cship/) | `cship` | chained | 15, 25, 45 Hz | 9 | 19 Hz | 212.6 ms | — |
| [`cspin`](cspin/) | `cspin` | chained | 15, 25, 45, 60, 75, 90 Hz | 12 | 33 Hz | 119.6 ms | — |
| [`multi`](multi/) | `multi` | timer | 5, 8, 10, 12, 15, 20, 25, 30, 45, 60, 75, 90 Hz | 24 | 100 Hz | 53.2 ms | — |
| [`multi_c200`](multi_c200/) | `multi` + `_c200` | timer | 45 Hz | 1 | 200 Hz | 265.0 ms | — |
| [`multi_hog2`](multi_hog2/) | `multi` + `_hog2` | timer | 45 Hz | 1 | 100 Hz | 265.4 ms | — |
| [`multi_q1`](multi_q1/) | `multi` + `_q1` | timer | 25, 45 Hz | 2 | 100 Hz | 64.9 ms | — |
| [`nproc`](nproc/) | `nproc` | timer | 15, 25, 45 Hz | 9 | 76 Hz | — | — |
| [`p3`](p3/) | `p3` | timer | 5, 8, 10, 12, 15, 20, 25, 30, 45, 60, 75, 90, 120 Hz | 24 | 100 Hz | 55.8 ms | — |
| [`p3_c200`](p3_c200/) | `p3` + `_c200` | timer | 45 Hz | 1 | 200 Hz | 56.0 ms | — |
| [`p3_c50`](p3_c50/) | `p3` + `_c50` | timer | 45 Hz | 3 | 50 Hz | 56.4 ms | — |
| [`p3_hog2`](p3_hog2/) | `p3` + `_hog2` | timer | 45 Hz | 1 | 100 Hz | 55.9 ms | — |
| [`p3_q1`](p3_q1/) | `p3` + `_q1` | timer | 25, 45 Hz | 2 | 100 Hz | 31.1 ms | — |
| [`p8`](p8/) | `p8` | timer | 5, 8, 10, 12, 15, 20, 25, 30, 45 Hz | 11 | 100 Hz | 30.4 ms | — |
| [`part8`](part8/) | `part8` | timer | 45 Hz | 6 | 100 Hz | 56.2 ms | — |
| [`rmulti`](rmulti/) | `rmulti` | timer | 25, 45 Hz | 6 | 100 Hz | 34.0 ms | — |
| [`rp3`](rp3/) | `rp3` | timer | 25, 45, 90 Hz | 6 | 100 Hz | 56.3 ms | — |
| [`rp8`](rp8/) | `rp8` | timer | 45 Hz | 3 | 100 Hz | 56.9 ms | — |
| [`rspin`](rspin/) | `rspin` | timer | 25, 45 Hz | 4 | 35 Hz | 96.3 ms | — |
| [`rvanilla`](rvanilla/) | `rvanilla` | chained | 15, 25, 30, 45, 60, 90 Hz | 18 | 12 Hz | 721.2 ms | — |
| [`rvanilla4`](rvanilla4/) | `rvanilla4` | chained | 15, 25, 30, 45, 60, 90 Hz | 18 | 38 Hz | 137.5 ms | — |
| [`rvanilla4x2tm`](rvanilla4x2tm/) | `rvanilla4x2tm` | timer | 45 Hz | 3 | 100 Hz | 89.7 ms | — |
| [`rvanilla8`](rvanilla8/) | `rvanilla8` | chained | 45 Hz | 3 | 37 Hz | 243.6 ms | — |
| [`rvanilla8tm`](rvanilla8tm/) | `rvanilla8tm` | timer | 45 Hz | 3 | 100 Hz | 244.0 ms | — |
| [`ship`](ship/) | `ship` | timer | 5, 8, 10, 12, 15, 20, 25, 30, 45, 60, 75, 90 Hz | 24 | 82 Hz | 53.8 ms | — |
| [`ship_c200`](ship_c200/) | `ship` + `_c200` | timer | 45 Hz | 1 | 19 Hz | 213.3 ms | — |
| [`ship_q1`](ship_q1/) | `ship` + `_q1` | timer | 25, 45 Hz | 2 | 28 Hz | 107.2 ms | — |
| [`smte`](smte/) | `smte` | timer | 5, 8, 10, 12, 15, 20, 25, 30, 45, 60, 75, 90 Hz | 18 | 100 Hz | 391.1 ms | — |
| [`spin`](spin/) | `spin` | timer | 5, 8, 10, 12, 15, 20, 25, 30, 45, 60, 75, 90 Hz | 24 | 95 Hz | 30.5 ms | — |
| [`spin_c200`](spin_c200/) | `spin` + `_c200` | timer | 45 Hz | 1 | 33 Hz | 119.7 ms | — |
| [`spin_hog2`](spin_hog2/) | `spin` + `_hog2` | timer | 45 Hz | 1 | 33 Hz | 119.5 ms | — |
| [`spin_q1`](spin_q1/) | `spin` + `_q1` | timer | 25, 45 Hz | 2 | 75 Hz | 30.4 ms | — |
| [`vanilla`](vanilla/) | `vanilla` | chained | 15, 25, 30, 45, 60, 90 Hz | 18 | 21 Hz | 433.3 ms | — |
| [`vanilla4`](vanilla4/) | `vanilla4` | chained | 15, 25, 30, 45, 60, 90, 120 Hz | 21 | 39 Hz | 189.4 ms | showdown_45hz_pinned_vs_rosdefault_s1000, showdown_45hz_pinned_vs_rosdefault_s1003 |
| [`vanilla4_q1`](vanilla4_q1/) | `vanilla4` + `_q1` | chained | 45, 90 Hz | 6 | 39 Hz | 42.0 ms | — |
| [`vanilla4f`](vanilla4f/) | `vanilla4f` | chained | 45 Hz | 3 | 39 Hz | 242.5 ms | — |
| [`vanilla4t`](vanilla4t/) | `vanilla4t` | timer | 15, 25, 30, 45, 60, 90 Hz | 18 | 75 Hz | 30.9 ms | — |
| [`vanilla4tm`](vanilla4tm/) | `vanilla4tm` | timer | 45, 90 Hz | 6 | 100 Hz | 136.6 ms | — |
| [`vanilla4x2`](vanilla4x2/) | `vanilla4x2` | chained | 25, 30, 36, 38, 40, 45, 50, 60, 75, 90, 120 Hz | 33 | 62 Hz | 291.9 ms | — |
| [`vanilla4x2_ime`](vanilla4x2_ime/) | `vanilla4x2` + `_ime` | chained | 30 Hz | 3 | 15 Hz | 24.1 ms | — |
| [`vanilla4x2_q1`](vanilla4x2_q1/) | `vanilla4x2` + `_q1` | chained | 36 Hz | 3 | 36 Hz | 32.3 ms | — |
| [`vanilla4x2c0_ime`](vanilla4x2c0_ime/) | `vanilla4x2c0` + `_ime` | chained | 30 Hz | 3 | 30 Hz | 24.0 ms | — |
| [`vanilla4x2c0_rvv`](vanilla4x2c0_rvv/) | `vanilla4x2c0` + `_rvv` | chained | 30 Hz | 3 | 30 Hz | 31.1 ms | — |
| [`vanilla4x2d2`](vanilla4x2d2/) | `vanilla4x2` + `d2` | chained | 30, 36 Hz | 2 | 36 Hz | 32.8 ms | — |
| [`vanilla4x2ns4`](vanilla4x2ns4/) | `vanilla4x2` + `ns4` | — | 36 Hz | 1 | — | — | — |
| [`vanilla4x2ns4a`](vanilla4x2ns4a/) | `vanilla4x2` + `ns4a` | chained | 36 Hz | 1 | 36 Hz | 38.7 ms | — |
| [`vanilla4x2ns4b`](vanilla4x2ns4b/) | `vanilla4x2` + `ns4b` | chained | 36 Hz | 1 | 36 Hz | 39.2 ms | — |
| [`vanilla4x2ns4c`](vanilla4x2ns4c/) | `vanilla4x2` + `ns4c` | chained | 36 Hz | 1 | 36 Hz | 37.8 ms | showdown_36hz_solver_vs_rosallhart_s1006, showdown_36hz_solver_vs_rosallhart_s1006_ladder |
| [`vanilla4x2tm`](vanilla4x2tm/) | `vanilla4x2tm` | timer | 25, 30, 45, 75, 120 Hz | 15 | 100 Hz | 37.8 ms | — |
| [`vanilla4x2tm_c50`](vanilla4x2tm_c50/) | `vanilla4x2tm` + `_c50` | timer | 45 Hz | 3 | 50 Hz | 39.8 ms | — |
| [`vanilla8`](vanilla8/) | `vanilla8` | chained | 45 Hz | 3 | 38 Hz | 242.5 ms | — |
| [`vanilla8_q1`](vanilla8_q1/) | `vanilla8` + `_q1` | chained | 45 Hz | 3 | 39 Hz | 42.5 ms | — |
| [`vanilla8f`](vanilla8f/) | `vanilla8f` | chained | 45 Hz | 3 | 39 Hz | 242.3 ms | — |
| [`vanilla8tm`](vanilla8tm/) | `vanilla8tm` | timer | 45 Hz | 3 | 100 Hz | 241.9 ms | — |
| [`vanilla8tm_c50`](vanilla8tm_c50/) | `vanilla8tm` + `_c50` | timer | 45 Hz | 3 | 50 Hz | 242.9 ms | — |
| [`vanilla_c50`](vanilla_c50/) | `vanilla` + `_c50` | chained | 45 Hz | 3 | 20 Hz | 265.9 ms | — |
| [`vanilla_q1`](vanilla_q1/) | `vanilla` + `_q1` | chained | 45 Hz | 1 | 21 Hz | 64.9 ms | — |
| [`x2p`](x2p/) | `x2p` | timer | 45 Hz | 2 | 100 Hz | 62.7 ms | — |
| [`x2rmulti`](x2rmulti/) | `x2rmulti` | timer | 45 Hz | 2 | 100 Hz | 247.9 ms | — |
| [`x2rp3`](x2rp3/) | `x2rp3` | timer | 45 Hz | 2 | 100 Hz | 63.3 ms | — |
| [`x2rspin`](x2rspin/) | `x2rspin` | timer | 45 Hz | 2 | 11 Hz | 623.8 ms | — |
| [`x2spin`](x2spin/) | `x2spin` | timer | 45 Hz | 2 | 17 Hz | 417.7 ms | — |
| [`yproc`](yproc/) | `yproc` | timer | 15, 25, 45 Hz | 9 | 100 Hz | 55.6 ms | — |

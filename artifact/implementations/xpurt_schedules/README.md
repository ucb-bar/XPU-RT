# XPU-RT schedules and deployment points — every arm measured on the K1

One directory per arm: **43** of them. Each says what the arm is, what
the board measured, and the recipe that rebuilds it.

These are the arms whose numbers a figure or a document prints. The board also holds 481
run manifests under `results/codesign_feedback/xpurt_long/`, most of them points in a solver sweep rather than
deployments; [`run_index.md`](../../../docs/Artifact/run_index.md) indexes those.

## Solved schedules executed on the board

Each is a schedule a solver produced for one workload spec, carried through the codegen contract and run on the K1.

| arm | what it is | camera→control | control gap | frames late | drawn in |
|---|---|---|---|---|---|
| [`a120hcpsat_hardr`](a120hcpsat_hardr/) | CP-SAT (hard windows), 120 Hz camera, 200 ms table | 59.9 ms | 10.0 ms | 0/36 | — |
| [`a120hgreedyr`](a120hgreedyr/) | greedy, 120 Hz camera, 200 ms table | 566.3 ms | 35.77 ms | 36/36 | — |
| [`a30cpsat_hardr`](a30cpsat_hardr/) | CP-SAT (hard windows), 30 Hz camera | 55.2 ms | 9.98 ms | 0/36 | — |
| [`a30greedyr`](a30greedyr/) | greedy, 30 Hz camera | 70.1 ms | 10.0 ms | 0/36 | — |
| [`a60cpsat_hardr`](a60cpsat_hardr/) | CP-SAT (hard windows), 60 Hz camera | 58.5 ms | 10.06 ms | 0/72 | — |
| [`a60greedyr`](a60greedyr/) | greedy, 60 Hz camera | 583.6 ms | 17.48 ms | 72/72 | — |
| [`a90cpsat_hardr`](a90cpsat_hardr/) | CP-SAT (hard windows), 90 Hz camera | 55.9 ms | 10.0 ms | 0/108 | — |
| [`a90greedyr`](a90greedyr/) | greedy, 90 Hz camera | 947.5 ms | 25.67 ms | 111/111 | — |
| [`acpsat_hardr`](acpsat_hardr/) | CP-SAT (hard windows) | 56.8 ms | 10.0 ms | 0/120 | showdown_45hz_pinned_vs_rosdefault_s1000, showdown_45hz_pinned_vs_rosdefault_s1003, showdown_45hz_pinned_vs_rospinned_s1011 |
| [`agreedyr`](agreedyr/) | greedy (list scheduling) | 748.0 ms | 13.25 ms | 120/120 | — |
| [`b5cpsat_hardr`](b5cpsat_hardr/) | CP-SAT (hard windows), 90 Hz + heavier stack | 58.4 ms | 9.95 ms | 0/108 | — |
| [`b5greedyr`](b5greedyr/) | greedy, 90 Hz + heavier stack | 150.3 ms | 11.56 ms | 108/108 | — |
| [`fb30r1r`](fb30r1r/) | CP-SAT, per-rate feedback round 1, 30 Hz camera | 30.1 ms | 9.99 ms | 0/81 | — |
| [`p30cpsat_hardr`](p30cpsat_hardr/) | CP-SAT (hard windows), partitioned, one hart per yolo frame | 60.08 ms | 10.0 ms | 0/36 | — |
| [`p30efullr`](p30efullr/) | CP-SAT (hard windows), both clusters whole, conv on the IME | 25.3 ms | 10.0 ms | 0/36 | — |
| [`p30freer`](p30freer/) | CP-SAT (hard windows), placement unconstrained, conv on the IME | 26.75 ms | 9.99 ms | 0/36 | — |
| [`p30greedyr`](p30greedyr/) | greedy, partitioned, one hart per yolo frame | 60.09 ms | 10.0 ms | 0/36 | — |
| [`p30imer`](p30imer/) | greedy, partitioned, one hart per yolo frame, conv on the IME | 43.42 ms | 10.0 ms | 0/36 | — |
| [`p30w4cpr`](p30w4cpr/) | CP-SAT (hard windows), partitioned, yolo on the whole P cluster | 36.75 ms | 10.0 ms | 0/36 | — |
| [`p30w4forcer`](p30w4forcer/) | CP-SAT (hard windows), partitioned, yolo on the whole P cluster, blanket IME | 30.08 ms | 10.0 ms | 0/36 | — |
| [`p30w4imecpr`](p30w4imecpr/) | CP-SAT (hard windows), partitioned, yolo on the whole P cluster, conv on the IME | 26.75 ms | 10.0 ms | 0/36 | — |
| [`p30w4imer`](p30w4imer/) | greedy, partitioned, yolo on the whole P cluster, conv on the IME | 26.75 ms | 10.0 ms | 0/36 | — |
| [`p30w4r`](p30w4r/) | greedy, partitioned, yolo on the whole P cluster | 36.75 ms | 10.0 ms | 0/36 | — |
| [`p36freer`](p36freer/) | CP-SAT (hard windows), placement unconstrained, conv on the IME, 36 Hz camera | 25.82 ms | 9.84 ms | 0/42 | showdown_36hz_solver_vs_rosallhart_s1006, showdown_36hz_solver_vs_rosallhart_s1006_ladder |
| [`p45detr`](p45detr/) | CP-SAT (hard windows), placement unconstrained, single-worker solve, 45 Hz camera | 30.07 ms | 9.83 ms | 0/63 | — |
| [`p45freer`](p45freer/) | CP-SAT (hard windows), placement unconstrained, conv on the IME, 45 Hz camera | 27.46 ms | 9.92 ms | 0/66 | showdown_45hz_solver_vs_rospinned_s1007 |
| [`sonlycpr`](sonlycpr/) | CP-SAT (hard windows), shard costs measured | 53.2 ms | 10.03 ms | 0/120 | — |
| [`w2pg36r`](w2pg36r/) | greedy + measured shard costs, 36 Hz camera | 30.1 ms | 9.98 ms | 0/96 | — |
| [`w2pg40r`](w2pg40r/) | greedy + measured shard costs, 40 Hz camera | 39.8 ms | 10.06 ms | 0/108 | — |
| [`w2pgOCr`](w2pgOCr/) | greedy, shard costs measured, perception in two camera periods | 36.0 ms | 9.99 ms | 0/120 | — |

## Measured deployment points

Each is a hand-built arrangement measured directly, without a solve to reproduce.

| arm | what it is | camera→control | control gap | frames late | drawn in |
|---|---|---|---|---|---|
| [`best45alt2`](best45alt2/) | 45 Hz camera, other_hog2 | 45.6 ms | 9.97 ms | — | — |
| [`best45alt2c200`](best45alt2c200/) | 45 Hz camera | 37.9 ms | 5.0 ms | — | — |
| [`best45alt2long`](best45alt2long/) | 45 Hz camera | 40.1 ms | 10.0 ms | — | — |
| [`best60alt2`](best60alt2/) | 60 Hz camera | 39.6 ms | 10.0 ms | — | — |
| [`best75alt2`](best75alt2/) | 75 Hz camera | 119.6 ms | 10.0 ms | — | — |
| [`best90alt1`](best90alt1/) | 90 Hz camera | 57.4 ms | 10.0 ms | — | — |
| [`cam2alt1`](cam2alt1/) | 90 Hz camera | 71.8 ms | 10.0 ms | — | — |
| [`cam2rich`](cam2rich/) | 90 Hz camera | 74.3 ms | 9.96 ms | — | — |
| [`rich25p4`](rich25p4/) | 25 Hz camera | 29.6 ms | 10.0 ms | — | — |
| [`rich45alt2`](rich45alt2/) | 45 Hz camera | 40.8 ms | 9.98 ms | — | — |
| [`rich45alt2ime`](rich45alt2ime/) | 45 Hz camera | 40.9 ms | 9.97 ms | — | — |
| [`w2pg36base`](w2pg36base/) | 36 Hz camera | 30.1 ms | 9.98 ms | — | — |
| [`w2pg36ime`](w2pg36ime/) | 36 Hz camera | 22.3 ms | 9.94 ms | — | — |


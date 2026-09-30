# board/ — code that runs on the SpacemiT K1

Everything is under `k1_ros_mb/`:

* `ros_mb_chain_traced.cpp` — the ROS 2 camera → perception → nav → control chain, instrumented per callback
  and per pool shard; built and deployed by `scripts/board_deploy_ros_node.sh`. `ros_mb_chain.cpp` is the
  same chain without tracing; `ros_ctrl_starve.cpp` isolates control-callback starvation.
* `pool/` — the worker pool the perception node uses.
* `cpu_sampler.c`, `rdtime_now.c`, `cpu_hog.c` — the per-core busy sampler, a timer probe and a load generator.
* `MANIFEST.sha256` — hashes of the sources the board runs were built from.

The ROS 2 baseline measured with this code is Tier C in `docs/Baselines/ros_baseline_tiers.md`.

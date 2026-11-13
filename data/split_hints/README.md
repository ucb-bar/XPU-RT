# K3 split hints (branch k3-split)

- `k3_advice_ime_shard.json`: `emit_compile_advice.py` output on
  `schedules/scheduled_networks_k3_mb_3model_ime_greedy_profiled.json`
  (profiles rvv_k3_x100 + ime_k3_a100, IR from ModelBlaster/build/k3).
  It contains **no `split` advice**: every dispatch fits its free slot
  (free slots: mlp_control 9.97 ms, dronet 32.2 ms, yolov8_nano 241.8 ms vs.
  max single dispatch 5.3 ms), so `blocking_advice` has nothing to cut. It does
  emit 72 `choose_implementation` items (A100 is 78-94% faster on yolo convs).
- The hand-picked hint `k3_split_v1.json` (13 ops, singletons mode) was removed
  on this branch; it remains on branch `k3` (commit 64dec882).


# K3 split hints (branch k3-split)

- `k3_advice_ime_shard.json`: `emit_compile_advice.py` output on
  `schedules/scheduled_networks_k3_mb_3model_ime_greedy_profiled.json`
  (profiles rvv_k3_x100 + ime_k3_a100, IR from ModelBlaster/build/k3).
  It contains **no `split` advice**: every dispatch fits its free slot
  (free slots: mlp_control 9.97 ms, dronet 32.2 ms, yolov8_nano 241.8 ms vs.
  max single dispatch 5.3 ms), so `blocking_advice` has nothing to cut. It does
  emit 72 `choose_implementation` items (A100 is 78-94% faster on yolo convs).
- `k3_split_v1.json`: hand-picked hint (`modelblaster.split_hints/v1`),
  singletons mode (no shard). Picks are the heaviest dispatches by
  `max(x100_time, a100_time)` from the single-core (topo_0) profiles.

Rules used: OC/C tiles are multiples of 16 (IME packed-weight tile); OH only on
plain `conv2d_s8` (fused conv family has no OH splitter); C for `maxpool2d_s8`.
The yolo OC=80 convs have no even 2..4 split with 16-multiple tiles (80 = 5x16),
so they use explicit `tile_sizes` ([32,32,16], [48,32]). The yolo stem
(id0, OC=16, fused silu, 835us on A100 -- the top a100-only dispatch) cannot be
split: OC=16 is one tile and fused convs have no OH axis. `upsample_nearest_s8`
id53 (a100 402us) has no splitter either.

Each op carries a `reason` with its x100/a100 times. Dispatch ids refer to the
unsplit `build/k3/<m>/int8/graph.json`.

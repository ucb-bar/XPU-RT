# Graph Digest & Focused Chunks

The LLM is never handed the raw module. Two MCP tools produce the
views it reads:

* `analyze_graph` — shape-free overall digest.
* `focus_chunk` — one region with its full knob + DoF surface.

Both are driven by `python/compgen/analysis/graph_digest.py`. No LLM
call is needed to build them; they're deterministic views over the
analyzer output and the xDSL module.

## `analyze_graph` — the overall digest

`GraphDigest` fields (all JSON-encoded):

| Field | Notes |
|-------|-------|
| `pattern_histogram` | Count per `DetectedPattern.pattern_type` |
| `pattern_size_histogram` | Count per cluster size (# of ops) |
| `dim_spectrum.rank_histogram` | Count per tensor rank |
| `dim_spectrum.{parallel,reduce,batch,broadcast}_dims` | From `compgen.dim_role` attrs stamped by `analyze/dim_semantics.py` |
| `dtype_spectrum` / `quant_spectrum` | Count per short dtype code (`f32`, `bf16`, `fp8_e4m3`, `int8`, …) |
| `flop_distribution.total` | From `NetworkAnalysis.total_flops`; falls back to an IR-walk estimator when analyzer returns 0. `source` = `"analyzer"` or `"ir_walk_fallback"` |
| `byte_distribution` | Total bytes + top-5 clusters |
| `memory_footprint_bytes` | Sum of tensor-result byte counts across all non-structural ops |
| `critical_path` | `GraphAnalysisDossier.critical_path` |
| `fusion_opportunity_count` | Count of optimization opportunities mentioning "fus" |
| `bottleneck_ops` | `NetworkAnalysis.bottleneck_clusters` |
| `region_index` | IDs from the dossier regions (usable as `selector.region_id`) |

`to_prompt_summary(max_bytes=2048)` returns a ≤2 KB string safe to drop
into an LLM prompt.

## `focus_chunk` — one region, with knobs + DoF

`ChunkView` fields:

| Field | Notes |
|-------|-------|
| `region_id`, `pattern_type` | Resolved from selector |
| `ops`, `edges` | Nodes + DAG edges within the chunk |
| `symbolic_shapes` | Shape-free view; concrete shapes opt-in |
| `concrete_shapes` | Only when `include_concrete_shapes=True` |
| `dim_roles` | Read from `compgen.dim_role` attrs (not empty on synthetic clusters) |
| `dtypes` | From xDSL op result types; FX↔xDSL name bridge in `_cluster_ops` |
| `quant_attrs` | Booleans flagging fp8/int dtypes |
| `envelope_facts` | Target name, vector lanes, scratchpad bytes, peak bandwidth, MMA shapes |
| `decision_knobs` | Oracle-enumerated options with advisory flag (see below) |
| `dof_description` | Open-ended design-space: axes, memory tiers, archetypes, fusion boundaries, heuristic hints |

### Knob schema is non-binding

The knobs emphasise their non-binding nature:

```json
{
  "advisory_nature": "non-binding; agent is the decider",
  "granularity_options": [
    {"granularity": "MICRO", "source": "oracle:granularity", "oracle_advisory": false, "reason": "", "confidence": 0.0},
    {"granularity": "NORMAL", "source": "oracle:granularity", "oracle_advisory": false, "reason": "", "confidence": 0.0},
    {"granularity": "MEGA", "source": "oracle:granularity", "oracle_advisory": true,  "reason": "chain of 2 ops fits in scratchpad + every pair FUSE; combined speedup 2.00× ≥ 1.50× → MEGA", "confidence": 0.8}
  ],
  "tile_options":    [ ...multi-dtype × multi-shape sweep from recommend_tile... ],
  "memory_tier_options": ["register", "scratchpad", "device_dram", "host"],
  "fusion_options":  [ ...adjacent pairs with should_fuse verdicts + est_speedup + reason, all {binding: false}... ],
  "alternatives":    []   // slot for LLM/history-supplied options
}
```

`oracle_advisory: true` marks the oracle's pick — **suggestion, not
command**. Every candidate carries `source` naming which oracle/cost
model produced it. Fusion entries carry `oracle_verdict` (not
`verdict`) and `binding: false` so no reader can mistake them for a
binding decision.

### DoF description — the creative surface

`DoFDescription` gives the LLM the abstract design space:

* `axes` — abstract dim names (`dim0`, `dim1`, …) present in the module
* `memory_tiers` — the full 5-value enum (the LLM can invent a routing
  strategy oracle knobs don't enumerate)
* `archetypes` — `COMPUTE_TILED`, `POINTWISE`, `REDUCE`, `MEMORY`, `ACTIVATION`
* `fusion_boundaries` — source→dest pairs within the cluster
* `heuristic_hints` — codegen-hint strings from the envelope + any
  `optimization_opportunities` strings from the analyzer that mention
  this cluster

## Selector resolution

`focus_chunk(selector={...})` accepts any of:

* `{"region_id": "rmsnorm_0"}` — matches cluster id or dossier region id
* `{"pattern_type": "rmsnorm"}` — first cluster of this type
* `{"cluster_id": "..."}` — exact cluster id
* `{"node_names": ["mm_3", "sigmoid", "mul_4"]}` — cluster containing these FX names
* Empty selector — falls back to first cluster; if no clusters exist,
  synthesizes one from the first dossier region

## What this doesn't claim

* The digest's FLOP fallback is element-count heuristics, not real
  FLOP counts. Use it as a ranking signal, not a measurement.
* `fusion_options` entries carry oracle-model cost breakdowns
  (`dram_savings_us`, `launch_savings_us`, etc.) — those are computed
  from authored constants (`_LAUNCH_OVERHEAD_US`), not measured
  latencies. The trace records `oracle_advisory` events so downstream
  analysis can compute the oracle's accuracy against bench results.

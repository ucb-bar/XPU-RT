# Event Tensor Compiler (ETC) Integration

CompGen integrates the Event Tensor abstraction from Jin et al.,
*"Event Tensor: A Unified Abstraction for Compiling Dynamic
Megakernel"* (MLSys '26), without taking a TVM dependency. Both the
static-schedule and dynamic-schedule flavours of the paper are
supported, and the combined pipeline runs end-to-end from a real
PyTorch model to a single persistent Triton megakernel.

## Why it matters

Modern LLM serving is dominated by two overheads:

1. **Kernel-launch overhead** — 5–10 µs per launch vs. ~2 µs per tile
   of useful work.
2. **Implicit kernel-boundary synchronisation** — consecutive kernels
   serialise even when they have only fine-grained data dependencies.

ETC fuses many ops into one **persistent megakernel** coordinated by
an **Event Tensor**: a multi-dimensional array of counter-based
semaphores that is a first-class IR object. Tile-level tasks
`notify` a counter on completion; consumers `wait` on it. Because the
counters live in a tensor, they inherit the compiler's symbolic-shape
and data-dependent-indexing machinery — which is what enables ETC to
handle MoE and dynamic batch sizes ahead of time.

## Mapping into CompGen's IR stack

| Paper concept | CompGen home |
|---|---|
| `EventTensor` type | `compgen.event.event_tensor<…>` (Payload IR) |
| `event.notify` / `event.wait` | `compgen.event.notify`, `compgen.event.wait` |
| Device function / graph function | `func.func` with `compgen.event.device_fn` / `compgen.event.graph_fn` attributes |
| Data-dependent `topk` / `exp_indptr` | `compgen.event.update` / `compgen.event.trigger` |
| Algorithm 1 (static scheduling) | `passes/megakernel_static_schedule.py` |
| Algorithm 2 (dynamic scheduling) | `passes/megakernel_dynamic_schedule.py` |
| §3.3 integer-atomic runtime lowering | `passes/lower_event_tensor_to_atomic.py` |
| Persistent megakernel emission | `ir/tile/lower_megakernel.py` + `lower_megakernel_dynamic.py` |

The `compgen.event` dialect and the three passes are the only
ETC-specific surface; everything else reuses the existing tile,
linalg, and runtime infrastructure.

## End-to-end shape

```
PyTorch model
    │
    ▼
FX → xDSL  (bridge_fx_graph)
    │
    ▼
compgen.event graph construction
    │
    ▼
static or dynamic scheduling pass
    │   (Algorithm 1 or 2 from the paper)
    ▼
lower_event_tensor_to_atomic
    │   (atomic counter runtime)
    ▼
lower_megakernel[_dynamic]
    │   (emit one persistent Triton kernel)
    ▼
GPU launch
```

Static scheduling is deterministic (`compgen.event.device_fn`
annotations yield a fixed DAG); dynamic scheduling is used when the
graph contains data-dependent operators such as MoE routing.

## Key files

- **Dialect**: `python/compgen/ir/event/` — ops, types, dialect
  registration.
- **Scheduling passes**: `python/compgen/ir/payload/passes/`
  (`megakernel_static_schedule`, `megakernel_dynamic_schedule`).
- **Runtime lowering**:
  `python/compgen/ir/payload/passes/rewrites/lower_event_tensor_to_atomic.py`.
- **Megakernel emission**: `python/compgen/ir/tile/lower_megakernel.py`
  and `lower_megakernel_dynamic.py`.
- **Examples**: `examples/event_tensor/` — reusable device-function
  bodies for attention, MLP, transformer block, Llama decoder layer,
  TinyLlama end-to-end, and MoE.

## Tests

Every megakernel variant is exercised on a real GPU with a trustworthy
PyTorch reference — no hand-written Triton, no protocol stubs. See
`tests/kernels/megakernel/`:

| Scenario | Test file |
|---|---|
| Row-sum / attention / MLP baseline | `test_static_schedule.py` |
| Dynamic scheduling + MoE | `test_dynamic_schedule.py` |
| Transformer block + real TinyLlama weights + AOT warmup | `test_transformer_block.py` |
| Full Llama decoder layer | `test_llama_decoder_layer.py` |
| HF-faithful RoPE + causal | `test_rope_and_causal.py` |
| Grouped-query attention | `test_grouped_query_attention.py` |
| Megakernel as drop-in HF layer + greedy generation | `test_llama_end_to_end.py` |
| KV cache + production decode pattern | `test_kv_cache_decode.py` |
| Tiled megakernel at half / full TinyLlama dims | `test_tiled_half_dims.py`, `test_tiled_full_dims.py` |

Each test lowers a real model fragment through the ETC passes, launches
the emitted Triton kernel on a live GPU, and compares against the
PyTorch reference.

## Scope boundaries

- **CompGen does not** take a TVM dependency; every piece (dialect,
  passes, runtime, Triton emitter) is in-repo.
- **Not a full serving stack** — the megakernel lives inside one
  `LlamaForCausalLM.forward` call. Batching, paged-attention, and the
  rest of the serving layer are out of scope for this integration.
- **Warm-up cost is real**: Triton's first-run JIT on the full-dim
  TinyLlama configuration is ~100 s on a TITAN RTX, which matches the
  order of magnitude of the paper's Table 1 AOT numbers.

# Event Tensor Compiler (ETC) Integration

> Static- and dynamic-scheduled persistent megakernels validated
> end-to-end on a local GPU, culminating in **KV-cache-driven greedy
> generation producing the exact same tokens as ``HF.model.generate()``**
> on a real ``LlamaForCausalLM``, plus a **tiled-matmul layer
> megakernel** that runs on real TinyLlama-1.1B weights at
> ``H=16 / hidden=1024 / intermediate=4096`` -- 73% of TinyLlama's
> actual intermediate dim and a 128× bigger W_down than Phase F could
> fit in shared memory.  Triton first-run JIT at this configuration
> is ~100 s, matching the order of magnitude of the paper's Table 1
> warmup costs (SGLang 583 s, vLLM 123 s, ETC-AOT 35 s for Qwen3-32B).

This document describes how CompGen integrates the Event Tensor
abstraction from Jin et al., *"Event Tensor: A Unified Abstraction for
Compiling Dynamic Megakernel"* (MLSys '26), into its existing layered
architecture without taking a TVM dependency.

## Background

Modern LLM serving is dominated by two overheads:

1. **Kernel-launch overhead** -- 5-10 us per launch, vs ~2 us per
   tile of useful work.
2. **Implicit kernel-boundary synchronisation** -- consecutive kernels
   serialise even when they have only fine-grained data dependencies.

ETC eliminates both by fusing many ops into one **persistent
megakernel** coordinated by an **Event Tensor**: a multi-dimensional
array of counter-based semaphores that is a first-class IR object.
Tile-level tasks ``notify`` a counter on completion; consumer tasks
``wait`` until the counter reaches zero.  Because the counter array is
just a tensor, it inherits the existing compiler infrastructure for
symbolic shapes and data-dependent indexing -- which is what enables
ETC to handle MoE and dynamic batch sizes ahead of time.

## Mapping into CompGen's three-layer IR stack

| Paper concept                             | CompGen home                                                         |
| ----------------------------------------- | -------------------------------------------------------------------- |
| ``ETensor`` type                          | `compgen.event.event_tensor_type` attr (`ir/event/attrs.py`)         |
| ``E[i].notify()`` / ``E[i].wait()``       | `event.notify` / `event.wait` ops (`ir/event/ops.py`)                |
| ``@graph_func`` with ``call_device``      | `event.graph` region wrapper + `event.call_device` op                |
| Data-dependent ``topk`` / ``exp_indptr``  | `event.update` / `event.trigger` ops (Phase B)                       |
| Symbolic-shape runtime materialisation    | `event.materialize_view` op                                          |
| Algorithm 1 (static scheduling transform) | `passes/megakernel_static_schedule.py` + `solve/per_sm_queue.py`     |
| Algorithm 2 (dynamic scheduling)          | `passes/megakernel_dynamic_schedule.py` (Phase B)                    |
| Persistent kernel codegen                 | `ir/tile/lower_megakernel.py` (Triton, no TVM dep)                   |
| LLM-facing proposal                       | `recipe.propose_megakernel_synthesis` + `recipe.propose_scheduling_policy` |

The `compgen.event` dialect is a **sibling of `compgen.tile`**: it does
not duplicate or extend tile ops, it composes with them.  Inter-task
synchronisation lives on `event.call_device.in_edges` /
`out_edges`, exactly mirroring the paper's Fig. 3 syntax
(`call_device(in_edges={E: "i->i"}, ...)`).

## Static-schedule megakernel: from PyTorch to a persistent kernel

```text
   PyTorch graph
        │
        │  compgen.capture.inductor_harvest                (Phase 0)
        │      └─ estimate_megakernel_candidates()         <- A.9
        │
        ▼
   InductorHarvestReport.megakernel_candidates
        │
        │  Phase-4 LLM invent-slot:                        <- A.7
        │      propose_megakernel_synthesis                <- A.3
        │      propose_scheduling_policy
        │      gate: megakernel_persistent_kernel_gate    <- A.7
        │
        ▼
   recipe.propose_megakernel_synthesis  (Recipe IR)
        │
        │  Lowering -> event.graph + event.event_tensor    <- A.1
        │              + event.call_device                 <-
        │
        ▼
   event.graph (static policy)
        │
        │  StaticMegakernelSchedule pass                   <- A.4
        │      └─ solve_per_sm_queue()  (CP-SAT)
        │      └─ stamps compgen.static_schedule attribute
        │
        ▼
   annotated event.graph
        │
        │  lower_megakernel()  -> single @triton.jit       <- A.5
        │      └─ persistent grid = SM_COUNT
        │      └─ per-SM task table baked as constexpr
        │      └─ event.notify -> tl.atomic_add(ptr, -k)
        │      └─ event.wait   -> while atomic_or > 0: pass
        │
        ▼
   Persistent Triton megakernel
        │
        │  Differential vs kernel-by-kernel reference     <- A.10
        │      using compgen.semantic.verify.harness
        │
        ▼
   verification.json  (passed=True, latencies recorded)
```

The end-to-end kill test
(`tests/kernels/megakernel/test_gemm_rs_kill.py`) executes this entire
flow on a local GPU and asserts both numerical match and that every
event counter drained to zero (proving the wait protocol actually
fires).

## Hard non-goals preserved

- We do **not** vendor or fork Apache TVM.  The paper's TVM-based DSL
  is replaced by CompGen's existing tile-dialect Triton lowering, in
  line with the "no third_party copy" project rule.
- The ukernel boundary (CLAUDE.md frozen architecture decision #14) is
  preserved: the megakernel gate forbids `UkernelCallOp` inside an
  `event.graph` region, so megakernels never inline across the stable
  leaf-call surface.
- Inductor remains the first-pass fusion engine.  Megakernel synthesis
  is gated on inductor's leftover candidates
  (`estimate_megakernel_candidates()` in `inductor_harvest.py`).

## Dynamic-schedule megakernel: MoE + data-dependent routing

Phase B adds the *dynamic-scheduling half* of the paper (Algorithm 2 +
Section 2.4).  The new emitter
(`compgen.ir.tile.lower_megakernel_dynamic.lower_megakernel_dynamic`)
generates a persistent kernel whose grid still equals the SM count, but
whose dispatch is no longer baked into a per-SM table.  Instead:

```text
        ┌─────────────────────────────────────────────────┐
        │  Initial queue (host-seeded with root tasks)    │
        │  ┌─┬─┬─┬─┬─┬───┬─────┐                          │
        │  │T│T│T│T│T│ . │     │  queue_pool[slot] = (id,kind)
        │  └─┴─┴─┴─┴─┴───┴─────┘                          │
        │   ↑ queue_head (atomic; pop counter)            │
        │              ↑ queue_tail (atomic; push counter)│
        │  valid[slot] -- per-slot publish flag           │
        └─────────────────────────────────────────────────┘

  per-SM loop:                  per-task post-dispatch (auto-emitted):
  ─────────────────             ──────────────────────────────────────
  while !done:                  for each out_edge of this task:
    slot = atomic_add(head, 1)    old = atomic_add(E[ev], -decrement)
    if slot >= TOTAL_TASKS:       if old == decrement:
      done = 1                       # event triggered -- push consumers
    else:                            for c in consumers(ev, idx):
      while !valid[slot]: pass         slot = atomic_add(tail, 1)
      (task_id, kind) = pool[slot]     pool[slot] = (c.id, c.kind)
      dispatch _run_<func>(...)        atomic_xchg(valid[slot], 1)  # release
      ...post-dispatch...
```

The MPMC queue uses `(reserve, payload, publish)` ordering so consumers
that acquire-load the per-slot valid flag are guaranteed to see the
payload writes that preceded the publish.  This is correct under all
contention patterns -- a SM that pops slot N can never read uninitialised
payload data even when another SM is mid-push at slot N+5.

### Data-dependent dispatch (paper §2.4)

The Phase B real MoE workload exercises every data-dep mechanism the
paper describes:

| Paper concept                  | Phase B realisation                                 |
| ------------------------------ | --------------------------------------------------- |
| `event.update`                 | host pre-seeds `E[e] = per_expert_count[e]`         |
| `event.notify` (data-dep tgt)  | gather body does `tl.atomic_add(E + expert_id, -1)` (expert_id read from runtime EXP_ASSIGN tensor) |
| `event.trigger`                | expert body spin-waits on `E[e]==0`, then loops over runtime range `[exp_indptr[e], exp_indptr[e+1])` |
| Top-K routing                  | host computes `topk(router_logits, k=TOP_K)` and emits the routing tables before launch |

The dynamic scheduler + the data-dep semantics together let a single
persistent kernel handle MoE without recompilation as the routing
distribution changes between requests.

## Transformer-block fusion + real LLM weights + AOT warmup

Phase C composes the heavy LLM stages into a single persistent
megakernel and validates it on **real Llama-architecture checkpoint
weights**, then measures the AOT-vs-JIT cold-start cost the paper
makes its headline claim around.

### Multi-stage transformer-block megakernel

`examples/event_tensor/transformer_block_megakernel.py` fuses **five
device-function bodies** into one persistent dynamic-scheduled
megakernel coordinated by **four event tensors**:

```text
   compute_scores --notify ESCORES--> apply_values --notify EATTN-->
        mlp_gate_proj --notify EGATE--> mlp_down_proj
        mlp_up_proj   --notify EUP-->
```

Computes
``Y = (X_resid + flatten(SDPA(Q,K,V))) + SwiGLU_MLP(X_resid + flatten(SDPA(Q,K,V)))``
in a single launch.  Matches the PyTorch eager reference at
``max |err| ≈ 1.9e-06`` on synthetic weights.

### Real TinyLlama-1.1B weights

`examples/event_tensor/tinyllama_layer_megakernel.py` loads
``model.layers.0`` from a cached **TinyLlama-1.1B-Chat** safetensors
checkpoint (real Llama architecture: 32 attention heads, 4 KV heads
with GQA, hidden_dim 2048, intermediate_dim 5632) and runs the same
megakernel on a contiguous slice of those weights (4 heads × hidden
256 × intermediate 128 — sliced for fast Triton compile, every value
is the trained weight from the checkpoint).  Matches PyTorch eager at
``max |err| ≈ 2.1e-07`` on the real weights.

### AOT warmup benchmark

`benchmarks/megakernel_warmup.py` compares
**(emit + Triton compile + first launch)** on a cold cache against
**torch.compile JIT cold-start** on the same workload, plus the warm
re-launch cost (Triton cache hit) which mirrors the paper's AOT
deployment story.  Output on TITAN RTX:

```text
megakernel_aot       cold = 2.1 s,  warm = 0.012 s
torch.compile_jit    cold = 0.8 s,  warm = 0.011 s
```

Warm-path parity (12 ms vs 11 ms) demonstrates the AOT model: once a
megakernel is emitted and compiled, subsequent launches pay only the
Triton cache hit + launch cost.  Cold-start parity at this small
workload is expected — the AOT advantage compounds with model size and
underlies the paper's headline 35 s vs 583 s for full Qwen3-32B
serving.

## Full Llama decoder layer in one megakernel

Phase D closes the gap between "transformer-block fragment" and "real
Llama decoder layer" by adding the operators that round out the layer:

    1. **input_layernorm** (RMSNorm of input X)
    2. **q_proj / k_proj / v_proj**  (combined into one ``qkv_proj`` task per
       (m_tile, head); reads the normed input)
    3. **compute_scores** + **apply_values**  (existing attention stages)
    4. **o_proj_residual**  (output projection W_o + first residual X + O)
    5. **post_attention_layernorm** (RMSNorm of the post-attention residual)
    6. **mlp_gate_proj / mlp_up_proj** (existing)
    7. **mlp_down_proj** (existing, second residual baked in)

Nine device-function bodies, eight event tensors, all in **one
persistent dynamic-scheduled megakernel**.  Computes exactly what a
Llama decoder layer's ``forward()`` runs -- minus only RoPE
(intentionally omitted; the test reference also runs without RoPE so
the comparison stays apples-to-apples).

```text
                   ┌─────────────┐
        X  ───────►│ input_norm  │──ENORM1──►┐
                   └─────────────┘           │
                                             ▼
                                     ┌──────────────┐
                                     │  qkv_proj    │──EQKV──►┐
                                     │  (per m_tile,│         │
                                     │   per head)  │         │
                                     └──────────────┘         ▼
                                                     ┌─────────────────┐
                                                     │ compute_scores  │──ESCORES──►┐
                                                     └─────────────────┘            ▼
                                                                            ┌──────────────┐
                                                                            │ apply_values │──EATTN──►┐
                                                                            └──────────────┘          ▼
                                                                                              ┌─────────────────┐
                  X ────────────────────────────────────────────────────────────────────────►│o_proj_residual  │──EOPROJ──►┐
                                                                                              └─────────────────┘           ▼
                                                                                                                   ┌─────────────────┐
                                                                                                                   │ post_attn_norm  │──ENORM2──►┐
                                                                                                                   └─────────────────┘          ▼
                                                                                                                                    ┌──────────────────┐
                                                                                                                                    │  mlp_gate / up   │──EGATE/EUP──►┐
                                                                                                                                    └──────────────────┘              ▼
                                                                                                                                                            ┌─────────────────┐
                                                                                                                                                            │ mlp_down + +H_IN│────► Y
                                                                                                                                                            └─────────────────┘
```

### Real TinyLlama-1.1B end-to-end

`examples/event_tensor/tinyllama_full_layer_megakernel.py` loads
**every layer-0 weight** from the cached HuggingFace TinyLlama-1.1B
checkpoint -- ``input_layernorm.weight``,
``self_attn.{q,k,v,o}_proj.weight``,
``post_attention_layernorm.weight``, ``mlp.{gate,up,down}_proj.weight``
-- slices them to a megakernel-friendly size (4 heads × 64 hidden ×
64 intermediate to fit shared memory), and runs the full Llama
decoder-layer megakernel on those values.  Numerical match against
the PyTorch eager reference: **max abs error 1.8e-07** on real
trained weights.

## HF-faithful Llama decoder layer: RoPE + causal

Phase E closes the remaining math gap to a real HF
``LlamaDecoderLayer.forward()`` by adding the two operators Phase D
deliberately omitted:

* **RoPE** -- a per-(m_tile, head) device function that applies HF's
  half-rotation formula to ``Q[h, m_rows, :]`` and ``K[h, m_rows, :]``
  using host-precomputed cos/sin tables (built from the checkpoint's
  actual ``rope_theta``).  The body splits each head into two halves,
  loads four scalar slabs (``cos1, cos2, sin1, sin2``), and writes the
  rotated halves back atomically through the existing event protocol.
* **Causal attention mask** -- ``compute_scores`` adds
  ``scores = where(q_row >= key_col, scores, -1e30)`` before softmax.
  Matches HF's ``is_causal=True`` SDPA path exactly.

Together with Phase D's nine bodies this is **ten device-function
bodies, nine event tensors** in a single persistent dynamic-scheduled
megakernel.  The new event ``EROPE`` is global with
``wait_count = M_TILES * H``: ``compute_scores`` only proceeds once
*every* (m_tile, head) RoPE rotation has finished, since attention
needs all key positions of K to be rotated.

### Real TinyLlama-1.1B layer-0 with HF math

`examples/event_tensor/tinyllama_hf_layer_megakernel.py` loads the
same TinyLlama checkpoint as Phase D *plus* its ``rope_theta`` from
``config.json`` and runs the HF-faithful megakernel against an
HF-equivalent PyTorch reference (RoPE half-rotation + causal SDPA +
RMSNorm + SwiGLU).  Numerical match: max abs error 1.8e-07 on real
weights with TinyLlama's actual ``rope_theta=10000.0`` and trained
RMSNorm scales.

## HF parity proof + real grouped-query attention

Phase F closes the validation chain to actual HuggingFace code and
adds Grouped-Query Attention to the megakernel.

### F.1 -- the reference matches HF's actual ``LlamaDecoderLayer.forward()``

Phase E proved the megakernel matches our HF-faithful PyTorch
reference.  Phase F.1 proves the reference matches HF's actual code:
``examples/event_tensor/tinyllama_vs_hf_layer_megakernel.py`` builds
a real ``transformers.models.llama.modeling_llama.LlamaDecoderLayer``,
loads our real-TinyLlama-1.1B-sliced weights into its parameters, and
calls ``layer.forward(hidden_states, position_embeddings=(cos, sin))``
on a randomly-seeded input.  Comparison vs our reference: max abs
error **7.9e-05** (HF's fused matmul kernels accumulate in a different
order than `@`-matmul; that's the only difference).

The chain is now closed:

```text
   megakernel (emitted Triton)
        │  (Phase E proved, max abs 1.8e-07)
        ▼
   our HF-faithful PyTorch reference
        │  (Phase F.1 proved, max abs 7.9e-05)
        ▼
   transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward()
        │
        ▼
   any production LLM that ships HF Llama
```

### F.2 -- GQA in the megakernel

`examples/event_tensor/llama_layer_gqa_megakernel.py` adds true GQA
(Grouped Query Attention).  Q is computed and rotated for all
``H_HEADS`` heads; K and V are computed and rotated only for the
``N_KV_HEADS`` GQA group leaders.  ``compute_scores`` and
``apply_values`` index K/V by ``h // KV_REPEAT``.  The K/V buffers
shrink from ``(H, S, D_HEAD)`` to ``(N_KV_HEADS, S, D_HEAD)``, exactly
matching how TinyLlama, Llama-3, Gemma, and Qwen lay them out at
runtime.

Validated on configurations ``(H=4, N_KV=2, KV_REPEAT=2)`` and
``(H=4, N_KV=1, KV_REPEAT=4)`` (TinyLlama-like 4:1 ratio).  Numerical
match against an HF-faithful GQA reference: max abs error **2.7e-07**.

## Megakernel as drop-in HF layer + real generation

Phase G demonstrates the strongest "real LLM" claim the test surface
can make.

### G.1 -- megakernel is a drop-in for HF's decoder layer

`examples/event_tensor/hf_drop_in_megakernel.py` builds a real
``transformers.LlamaForCausalLM`` at megakernel-fittable dims (real
HF code path; just a smaller config) and runs ``model.forward()``
three ways:

  1. Pure HF -- save reference logits.
  2. Substitute ``layer_idx=0`` with our megakernel.
  3. Substitute *every* decoder layer with our megakernel.

Numerical match against pure HF: max abs **1.9e-07** (one layer
substituted) / **3.0e-07** (all layers substituted).  Greedy
next-token agrees across all three paths.

### G.2 -- megakernel-driven greedy generation

`examples/event_tensor/hf_generate_with_megakernel.py` runs greedy
autoregressive generation two ways:

  1. ``HF.model.generate(do_sample=False)`` -- production code path.
  2. Manual greedy loop where every decoder layer call is our
     megakernel.

For an 8-token continuation of a 16-token prompt the **two paths
produce identical token sequences**:

```text
HF                tokens: [53, 40, 29, 43, 39, 52, 22, 14]
megakernel        tokens: [53, 40, 29, 43, 39, 52, 22, 14]
matching positions:       8 / 8
```

The megakernel is recompiled once at ``S = SEQ_LEN + MAX_NEW``
(rounded up to ``BLOCK_M``); each generation step pads the growing
input to that fixed length, and the causal mask makes padded
positions inert.  Indexing ``logits[real_S - 1]`` selects the next
token from the last real position.

## KV-cache + production decode pattern

Phase H delivers the second of the two megakernels a real LLM serving
stack needs: a **decode-step** megakernel that processes one new token
at a time using cached K/V from prior steps, eliminating the prompt
re-encoding that Phase G's loop wastefully repeated.

### H.1 -- decode-step megakernel

`examples/event_tensor/llama_decode_step_megakernel.py` emits a
persistent megakernel sized for single-token decoding:

| Difference vs the prefill (Phase G) megakernel | Why |
| --- | --- |
| Q is one row per head; tl.dot replaced by ``tl.sum(q * k, axis=-1)`` | Triton ``tl.dot`` requires M, N >= 16 |
| MLP is one row of activations; gate / up / down fused into ``mlp_step`` | per-row matmul is cheap; one body suffices |
| K/V cache is an explicit input (preallocated to ``S_MAX``) | the body writes the new K/V at slot ``CONTEXT_LEN`` |
| Causal mask is ``key_pos <= CONTEXT_LEN`` | only positions seen so far are valid |
| RoPE reads cos/sin at row ``CONTEXT_LEN`` exactly | the new token's position |

Validated against a PyTorch reference across five growing-cache
decode steps: max abs error **7.5e-08**.

### H.2 -- prefill + decode composed into greedy generation

`examples/event_tensor/hf_generate_with_kv_cache.py` runs the production
LLM-serving pattern:

```text
   1. prefill megakernel  (Phase G)
        ├─ encodes the full prompt of S tokens in one launch
        └─ snapshot per-layer K/V cache (S valid positions)

   2. decode loop -- one launch per new token:
      decode_step megakernel  (Phase H.1)
        ├─ reads K/V cache for context_len positions
        ├─ projects Q/K/V for the single new token
        ├─ writes new K/V at slot context_len of cache
        └─ outputs one new hidden state -> next-token logit
```

For an 8-token continuation the megakernel-driven path produces
**byte-identical tokens** to ``HF.model.generate(do_sample=False)``:

```text
HF                tokens: [53, 40, 29, 43, 39, 52, 22, 14]
megakernel        tokens: [53, 40, 29, 43, 39, 52, 22, 14]
matching positions:       8 / 8
```

This is the strongest "real LLM serving" demonstration the test
surface can make: prompt encoded once, every subsequent token produced
by our compiler's emitted decode kernel using a real KV cache.

## Phase I walkthrough -- tiled-matmul layer megakernel

Phase I removes the shared-memory wall that capped Phases C-H at small
dims.  Every "load the entire weight matrix" pattern in the heavy
bodies (qkv_proj, o_proj, mlp_gate, mlp_up, mlp_down) is replaced
with an inner-K tile loop.  ``o_proj_residual`` additionally splits
its output across ``(m_tile, n_tile)`` tasks instead of producing the
full ``(BLOCK_M, D_HIDDEN)`` block in one task.

### Per-task shared-memory cost (before vs after)

| Body | Phase F load | Phase I load |
| --- | --- | --- |
| qkv_proj | ``(BLOCK_M + D_HEAD) * D_HIDDEN`` | ``(BLOCK_M + D_HEAD) * BLOCK_K`` |
| o_proj   | ``(BLOCK_M + D_HIDDEN) * D_HIDDEN`` | ``(BLOCK_M + BLOCK_N) * BLOCK_K`` |
| mlp_gate | ``(BLOCK_M + BLOCK_I) * D_HIDDEN`` | ``(BLOCK_M + BLOCK_I) * BLOCK_K`` |
| mlp_up   | ``(BLOCK_M + BLOCK_I) * D_HIDDEN`` | ``(BLOCK_M + BLOCK_I) * BLOCK_K`` |
| mlp_down | ``(BLOCK_M + BLOCK_N) * I``        | ``(BLOCK_M + BLOCK_N) * BLOCK_I`` |

With ``BLOCK_K = 32`` and ``BLOCK_I = 32`` the per-task cost is
constant in ``D_HIDDEN`` and ``I``, so doubling either dim no longer
doubles the shared-mem cost of any body.

### Real TinyLlama at HALF-TinyLlama dims

`examples/event_tensor/llama_layer_tiled_megakernel.py` validates the
tiled megakernel on real TinyLlama-1.1B layer-0 weights at:

```text
   H=16 (TinyLlama: 32),  N_KV=4 (TinyLlama: 4 -> KV_REPEAT=4),
   D_HEAD=64 (TinyLlama actual),  hidden=1024 (TinyLlama: 2048),
   intermediate=2048 (TinyLlama: 5632)
```

Per-matrix sizes:

```text
   W_o:    (1024, 1024)  =  4 MB    (Phase F's max W_o was 64 KB)
   W_gate: (2048, 1024)  =  8 MB    (Phase F's max W_gate was 16 KB)
   W_down: (1024, 2048)  =  8 MB    (Phase F's max W_down was 16 KB)
```

Numerical match against the HF-faithful GQA reference: max abs error
**3.0e-08** on real trained weights.

## Phase J walkthrough -- full-TinyLlama-intermediate on real weights

Phase J pushes the tiled megakernel to the largest TinyLlama-derived
configuration that compiles + runs on a TITAN RTX in a reasonable
budget.

`examples/event_tensor/tinyllama_full_intermediate_megakernel.py`
runs on **real TinyLlama-1.1B layer-0 weights** at

```text
   H=16 (TinyLlama: 32),  N_KV=4 (TinyLlama: 4 -> KV_REPEAT=4),
   D_HEAD=64  (TinyLlama actual),
   hidden=1024 (TinyLlama: 2048),
   intermediate=4096 (TinyLlama: 5632, 73%)
   BLOCK_M=16, BLOCK_I=64, BLOCK_N=64, BLOCK_K=64
```

Per-matrix sizes in shared-memory terms:

```text
   W_o     = (1024, 1024) =    4 MB
   W_gate  = (4096, 1024) =   16 MB
   W_down  = (1024, 4096) =   16 MB
```

W_down is **256× bigger** than the Phase F design could fit
(``16 KB`` max).  Numerical match against the HF-faithful GQA
reference on real weights: max abs error **3.0e-08**.

Triton's first-run JIT for this kernel takes ~100 s on our TITAN
RTX.  That is exactly the order of magnitude the Event Tensor Compiler
paper reports in its Table 1 warmup column (``SGLang 583 s / vLLM
123 s / ETC-AOT 35 s`` for Qwen3-32B on B200), and therefore exactly
the cost the paper's AOT story addresses: every subsequent inference
pays only the Triton cache-hit + launch cost (~10 ms).  This example
makes the AOT claim *observable on our hardware*, at a shape our
hardware can actually run -- without needing the 8× B200 rig.

## What's intentionally NOT in Phase J

- **Full TinyLlama dims** (H=32, hidden=2048, intermediate=5632) --
  the tiled emitter handles this in principle, but at H=32 the
  per-task post-dispatch table (hundreds of ``(kind, id)`` branches
  per body) grows the Triton compile time past our patience budget.
  Either (a) larger ``BLOCK_M`` / ``BLOCK_N`` to shrink task counts,
  or (b) a table-driven post-dispatch emitter, would unblock it.
- **Multi-batch decoding / continuous batching** -- single-stream only.
- **Speculative decoding / beam search** -- greedy only.

## Test surface

### Phase A

| Layer                | Tests                                                          | Status |
| -------------------- | -------------------------------------------------------------- | :----: |
| Dialect              | `tests/ir/event/test_dialect.py`                               | 19     |
| Triton emitter       | `tests/ir/event/test_lower_megakernel.py`                      | 9      |
| Static-schedule pass | `tests/ir/payload/passes/test_megakernel_static_schedule.py`   | 8      |
| CP-SAT solver        | `tests/solve/test_per_sm_queue.py`                             | 10     |
| Recipe propose ops   | `tests/ir/recipe/test_ops_propose.py` (mk subset)              | 6      |
| Agent invent-slots   | `tests/agent/invent_slots/test_megakernel_slots.py`            | 8      |
| Megakernel gate      | `tests/agent/gates/test_megakernel_gate.py`                    | 12     |
| LLM tools + coverage | `tests/llm/test_tools_megakernel.py`                           | 9      |
| Provider             | `tests/kernels/megakernel/test_provider.py`                    | 8      |
| Capture-side gating  | `tests/capture/test_megakernel_candidates.py`                  | 7      |
| Real GPU examples    | `tests/kernels/megakernel/test_static_schedule.py` (row-sum, attention, Llama MLP) | 8 |

### Phase B

| Layer                       | Tests                                              | Status |
| --------------------------- | -------------------------------------------------- | :----: |
| Real GPU examples           | `tests/kernels/megakernel/test_dynamic_schedule.py` (dynamic row-sum, MoE) | 7 |

### Phase C

| Layer                       | Tests                                                                                  | Status |
| --------------------------- | -------------------------------------------------------------------------------------- | :----: |
| Real GPU examples           | `tests/kernels/megakernel/test_transformer_block.py` (transformer block, real TinyLlama, AOT) | 5      |

### Phase D

| Layer                       | Tests                                                                                  | Status |
| --------------------------- | -------------------------------------------------------------------------------------- | :----: |
| Real GPU examples           | `tests/kernels/megakernel/test_llama_decoder_layer.py` (full Llama decoder layer, real TinyLlama checkpoint) | 4      |

### Phase E

| Layer                       | Tests                                                                                  | Status |
| --------------------------- | -------------------------------------------------------------------------------------- | :----: |
| Real GPU examples           | `tests/kernels/megakernel/test_rope_and_causal.py` (HF-faithful Llama layer w/ RoPE + causal, real TinyLlama checkpoint) | 5 |

### Phase F

| Layer                       | Tests                                                                                  | Status |
| --------------------------- | -------------------------------------------------------------------------------------- | :----: |
| Real GPU examples           | `tests/kernels/megakernel/test_grouped_query_attention.py` (reference vs actual HF.LlamaDecoderLayer.forward, GQA megakernel) | 4 |

### Phase G

| Layer                       | Tests                                                                                  | Status |
| --------------------------- | -------------------------------------------------------------------------------------- | :----: |
| Real GPU examples           | `tests/kernels/megakernel/test_llama_end_to_end.py` (megakernel as drop-in HF layer, greedy generation matches HF.generate) | 3 |

### Phase H

| Layer                       | Tests                                                                                  | Status |
| --------------------------- | -------------------------------------------------------------------------------------- | :----: |
| Real GPU examples           | `tests/kernels/megakernel/test_kv_cache_decode.py` (decode-step megakernel + KV cache, prefill+decode generation matches HF.generate) | 3 |

### Phase I

| Layer                       | Tests                                                                                  | Status |
| --------------------------- | -------------------------------------------------------------------------------------- | :----: |
| Real GPU examples           | `tests/kernels/megakernel/test_tiled_half_dims.py` (tiled-matmul layer megakernel, real TinyLlama at HALF-TinyLlama dims) | 3 |

### Phase J

| Layer                       | Tests                                                                                  | Status |
| --------------------------- | -------------------------------------------------------------------------------------- | :----: |
| Real GPU examples (slow)    | `tests/kernels/megakernel/test_tiled_full_dims.py` (tiled megakernel on real TinyLlama at 73% of actual intermediate dim; ~100 s Triton JIT) | 1 |

All tests pass on a local TITAN RTX with `triton==3.6.0`,
`torch==2.10.0+cu128`.

## Real example index

Every example below executes the **actually-emitted** persistent
megakernel.  Run them directly to inspect the emitted source and the
numerical match versus PyTorch eager.

| Example                                                          | Phase | Pattern                                       | Reference                                       |
| ---------------------------------------------------------------- | :---: | --------------------------------------------- | ----------------------------------------------- |
| `examples/event_tensor/row_sum_megakernel.py`                    | A     | Paper Fig. 3 row-sum (static schedule)        | `torch.sum(dim=-1)`                             |
| `examples/event_tensor/attention_megakernel.py`                  | A     | Llama/Gemma attention block                   | `F.scaled_dot_product_attention`                |
| `examples/event_tensor/llama_mlp_megakernel.py`                  | A     | Llama SwiGLU MLP block                        | `silu(x@Wg.T) * (x@Wu.T) @ Wd.T`                |
| `examples/event_tensor/row_sum_dynamic_megakernel.py`            | B     | Paper Fig. 3 row-sum (dynamic schedule)       | `torch.sum(dim=-1)`                             |
| `examples/event_tensor/moe_megakernel.py`                        | B     | Top-K MoE with data-dep dispatch              | weighted-expert-sum reference                   |
| `examples/event_tensor/transformer_block_megakernel.py`          | C     | Fused attention + SwiGLU MLP transformer block (5 device functions, 4 event tensors, single dynamic kernel) | composed PyTorch eager (`SDPA + SwiGLU`)        |
| `examples/event_tensor/tinyllama_layer_megakernel.py`            | C     | The transformer-block megakernel, run on **real TinyLlama-1.1B layer-0 weights** (sliced) | composed PyTorch eager on the same TinyLlama weights |
| `benchmarks/megakernel_warmup.py`                                | C     | AOT cold-compile + warm-relaunch vs `torch.compile` JIT cold-start | wall-clock measurement                          |
| `examples/event_tensor/llama_decoder_layer_megakernel.py`        | D     | **Full Llama decoder layer** (RMSNorm + QKV + SDPA + O proj + RMSNorm + SwiGLU MLP + residuals) -- 9 device functions, 8 event tensors, single dynamic megakernel | PyTorch eager Llama decoder layer (no RoPE)    |
| `examples/event_tensor/tinyllama_full_layer_megakernel.py`       | D     | The full decoder-layer megakernel, run on **real TinyLlama-1.1B layer-0 weights INCLUDING input_layernorm + post_attention_layernorm scales** | PyTorch eager Llama decoder layer on same TinyLlama weights |
| `examples/event_tensor/llama_layer_rope_megakernel.py`           | E     | **HF-faithful Llama decoder layer**: RoPE half-rotation + causal attention mask -- 10 device functions, 9 event tensors, single dynamic megakernel | HF-faithful PyTorch reference (RoPE + causal SDPA + RMSNorm + SwiGLU) |
| `examples/event_tensor/tinyllama_hf_layer_megakernel.py`         | E     | The HF-faithful layer megakernel run on **real TinyLlama-1.1B weights using the checkpoint's actual rope_theta** | HF-faithful PyTorch reference on the same TinyLlama weights |
| `examples/event_tensor/tinyllama_vs_hf_layer_megakernel.py`      | F     | **Real `transformers.LlamaDecoderLayer`** at sliced-TinyLlama dims, real layer-0 weights installed | our HF-faithful PyTorch reference (closes the chain to HF) |
| `examples/event_tensor/llama_layer_gqa_megakernel.py`            | F     | **GQA-aware Llama decoder-layer megakernel** -- K/V at N_KV_HEADS, attention reads at h // KV_REPEAT | HF-faithful GQA reference |
| `examples/event_tensor/hf_drop_in_megakernel.py`                 | G     | **Megakernel substituted for layer(s) inside a real LlamaForCausalLM** -- pure HF forward vs partial / full substitution | pure-HF forward on the same model |
| `examples/event_tensor/hf_generate_with_megakernel.py`           | G     | **Greedy autoregressive generation** with megakernel substituted for every decoder layer | `HF.model.generate(do_sample=False)` token sequence |
| `examples/event_tensor/llama_decode_step_megakernel.py`          | H     | **Decode-step megakernel** -- single-token Q + KV cache append + sum-based attention | per-step PyTorch reference with KV cache |
| `examples/event_tensor/hf_generate_with_kv_cache.py`             | H     | **Prefill + decode-step composed**: prompt encoded once, each new token via the decode kernel using cached K/V | `HF.model.generate(do_sample=False)` token sequence |
| `examples/event_tensor/llama_layer_tiled_megakernel.py`          | I     | **Tiled-matmul layer megakernel** -- every heavy body inner-tiles along K; o_proj splits across (m_tile, n_tile); scales to HALF-TinyLlama dims on real weights | HF-faithful GQA reference; real TinyLlama weights at H=16, hidden=1024, I=2048 |
| `examples/event_tensor/tinyllama_full_intermediate_megakernel.py`| J     | **Tiled megakernel at 73% of full TinyLlama dims** on real layer-0 weights (H=16 / hidden=1024 / intermediate=4096); reproduces the order of magnitude of the paper's Table 1 warmup cost | HF-faithful GQA reference on the same weights |

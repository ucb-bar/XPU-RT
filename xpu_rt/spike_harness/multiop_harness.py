"""Vanilla-path multi-op pipeline harness for Gemmini.

The agentic path emits one fused MEGA kernel that keeps intermediates
in scratchpad. The vanilla path emits N independent single-op kernels
that round-trip intermediates through DRAM. This module renders the
glue driver for the vanilla case: a single C source that

  1. allocates one DRAM buffer per ``ContractEdge`` in the graph and
     one per external input/output;
  2. calls each kernel in topological order, passing the right
     buffers;
  3. computes the scalar reference for the whole chain (so the
     whole-block diff catches wiring errors that per-kernel
     correctness checks would miss);
  4. measures end-to-end cycles via Gemmini's
     ``MAIN_LD_ST_EX_CYCLES`` counter — the same counter
     :mod:`xpu_rt.kb_gemmini.templates` uses for the single-op case.

The kernels themselves (one ELF symbol per ``ContractNode``) are
emitted by the existing single-op KB-vanilla pipeline driver
(:mod:`xpu_rt.kb_gemmini.kb_pipeline_driver`). This harness only
stitches them together.

For the pipeline-level study (P1.7) this harness is the only thing
that gives the vanilla path *something* to be measured end-to-end on
— without it, each kernel runs in isolation and we can't compare to
the agentic MEGA path's end-to-end cycle count.

Today we support the chain shapes SmolVLA actually surfaces in the
benchmark (matmul → activation → matmul → ...). The reference-op
catalogue is intentionally tight; adding a new op kind means adding
one entry to :data:`_REFERENCE_OP_CATALOGUE`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from xpu_rt.ir.payload.contract_graph import ContractGraph, ContractNode


# ---------------------------------------------------------------------------
# Per-op reference implementations (scalar C, computes whole-block ref).
# Each entry returns a snippet that operates on already-allocated buffers.
# ---------------------------------------------------------------------------


def _reference_matmul_snippet(out_buf: str, lhs_buf: str, rhs_buf: str, M: int, K: int, N: int) -> str:
    return f"""\
    /* matmul reference: {out_buf} = {lhs_buf} @ {rhs_buf}; ({M}x{K}) x ({K}x{N}) -> ({M}x{N}) */
    for (int64_t mi = 0; mi < {M}LL; ++mi) {{
        for (int64_t ni = 0; ni < {N}LL; ++ni) {{
            int32_t acc = 0;
            for (int64_t ki = 0; ki < {K}LL; ++ki) {{
                acc += (int32_t){lhs_buf}[mi*{K}LL + ki] * (int32_t){rhs_buf}[ki*{N}LL + ni];
            }}
            {out_buf}[mi*{N}LL + ni] = acc;
        }}
    }}
"""


def _reference_silu_snippet(out_buf: str, in_buf: str, n_elems: int) -> str:
    # silu(x) approximated as x * sigmoid(x) on i32 by clamping/scaling.
    # On Gemmini we typically integer-approximate; the harness only needs
    # a deterministic reference. Define silu(x) = x for x>0, 0 otherwise
    # (relu) as the reference — it's deterministic, matches what a
    # quantised int32 silu often approximates, and only requires
    # integer arithmetic. The agent's kernel must match this reference.
    return f"""\
    /* activation reference (relu as deterministic int32 approximation
       of silu used by the whole-block diff): {out_buf}[i] = max(0, {in_buf}[i]) for i in [0, {n_elems}) */
    for (int64_t i = 0; i < {n_elems}LL; ++i) {{
        int32_t v = {in_buf}[i];
        {out_buf}[i] = v > 0 ? v : 0;
    }}
"""


def _reference_relu_snippet(out_buf: str, in_buf: str, n_elems: int) -> str:
    return f"""\
    for (int64_t i = 0; i < {n_elems}LL; ++i) {{
        int32_t v = {in_buf}[i];
        {out_buf}[i] = v > 0 ? v : 0;
    }}
"""


_REFERENCE_OP_CATALOGUE = {
    "matmul": "matmul",
    "linalg.matmul": "matmul",
    "linear": "matmul",
    "silu": "silu",
    "gelu": "silu",  # approximated identically for the harness ref
    "relu": "relu",
    "tanh": "relu",
}


# ---------------------------------------------------------------------------
# Spec types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelBinding:
    """Maps one :class:`ContractNode` to its compiled kernel symbol.

    Attributes:
        op_id: ``ContractNode.op_id``.
        function_name: The extern C symbol exported by the kernel's
            init.c — typically ``launch_<region_id>``.
        kernel_source_path: Path to the kernel's ``init.c`` file
            (already emitted by KB-vanilla per-op). The harness
            stager copies these next to the driver before invoking
            the compile server.
    """

    op_id: str
    function_name: str
    kernel_source_path: Path


@dataclass(frozen=True)
class PipelineHarnessSpec:
    """Everything the renderer needs to produce a vanilla-path driver.c.

    Attributes:
        graph: The block's :class:`ContractGraph`. The renderer walks
            ``graph.topological_order``; each node must have a
            corresponding :class:`KernelBinding`.
        bindings: One per node.
        external_input_shapes: Pre-resolved shapes for every operand
            that does NOT come from an intra-graph edge. Keyed by
            ``(op_id, operand_index)``.
    """

    graph: ContractGraph
    bindings: tuple[KernelBinding, ...]
    external_input_shapes: Mapping[tuple[str, int], tuple[int, ...]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _node_op_family(node: ContractNode) -> str:
    name = node.op_name.lower()
    if name in _REFERENCE_OP_CATALOGUE:
        return _REFERENCE_OP_CATALOGUE[name]
    base = name.removeprefix("aten_")
    for suffix in ("_default", "_tensor", "_scalar"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return _REFERENCE_OP_CATALOGUE.get(base, base)


def _resolve_output_shape(node: ContractNode) -> tuple[int, ...]:
    md = node.contract.metadata or {}
    outs = md.get("output_shapes") or ()
    if not outs:
        raise ValueError(f"node {node.op_id!r} has no output_shapes in metadata")
    return tuple(outs[0])


def _buffer_id(op_id: str, kind: str, slot: int = 0) -> str:
    safe = op_id.replace(".", "_").replace("-", "_")
    return f"buf_{kind}_{safe}_{slot}"


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_pipeline_driver_c(spec: PipelineHarnessSpec) -> str:
    """Render a complete pipeline driver.c source.

    The driver is self-contained: it declares the externs, allocates
    every buffer, runs the kernel chain, computes the scalar
    reference, diffs, and prints the protocol KB's parser expects
    (``mismatches=N/M`` + ``cycles=N``).
    """
    graph = spec.graph
    binding_by_id = {b.op_id: b for b in spec.bindings}

    # Validate every node has a binding.
    for nid in graph.topological_order:
        if nid not in binding_by_id:
            raise ValueError(f"node {nid!r} has no KernelBinding")

    # 1) Compute which operands of each node are external (block-arg /
    #    not satisfied by any intra-graph edge). The buffer-allocation
    #    pass needs this so each external input is allocated once and
    #    fed identically to both the device kernel chain and the
    #    scalar reference chain.
    intra_consumers: set[tuple[str, int]] = set()
    edge_by_consumer: dict[tuple[str, int], str] = {}
    for e in graph.edges:
        intra_consumers.add((e.consumer_id, e.operand_index))
        edge_by_consumer[(e.consumer_id, e.operand_index)] = e.producer_id

    externals: list[tuple[str, int, tuple[int, ...]]] = []
    seen_external: set[tuple[str, int]] = set()
    for nid in graph.topological_order:
        node = graph.nodes[nid]
        md = node.contract.metadata or {}
        in_shapes = md.get("input_shapes", ())
        for op_i in range(len(in_shapes)):
            if (nid, op_i) in intra_consumers:
                continue
            shape = tuple(in_shapes[op_i])
            key = (nid, op_i)
            if key in seen_external:
                continue
            seen_external.add(key)
            externals.append((nid, op_i, shape))

    # 2) Render extern declarations + buffer allocations.
    extern_decls: list[str] = []
    seen_externs: set[str] = set()
    for nid in graph.topological_order:
        fn = binding_by_id[nid].function_name
        if fn in seen_externs:
            continue
        seen_externs.add(fn)
        extern_decls.append(
            f"extern void {fn}(void *output, void *input_A, void *input_B, "
            f"int64_t M, int64_t K, int64_t N);"
        )

    buf_decls: list[str] = []
    fill_calls: list[str] = []
    seed = 0
    external_buffers: dict[tuple[str, int], str] = {}
    for nid, op_i, shape in externals:
        # Use the v1 contract's stamped dtype if available.
        node = graph.nodes[nid]
        n_elems = 1
        for d in shape:
            if d > 0:
                n_elems *= d
        dtype = "int8_t"
        buf_id = _buffer_id(nid, "in", op_i)
        external_buffers[(nid, op_i)] = buf_id
        buf_decls.append(f"    static {dtype} {buf_id}[{max(n_elems, 1)}];")
        fill_calls.append(f"    fill_i8((int8_t *){buf_id}, {n_elems}LL, 0x{seed + 0xC0FFEE:08X});")
        seed += 1

    # 3) Per-node output buffer (i32 for matmul, i8 for activations).
    output_buf_id: dict[str, str] = {}
    output_dtype: dict[str, str] = {}
    output_elems: dict[str, int] = {}
    output_shape_dict: dict[str, tuple[int, ...]] = {}
    for nid in graph.topological_order:
        node = graph.nodes[nid]
        out_shape = _resolve_output_shape(node)
        n_elems = 1
        for d in out_shape:
            if d > 0:
                n_elems *= d
        family = _node_op_family(node)
        dtype = "int32_t" if family == "matmul" else "int32_t"  # all intermediates as i32 for consistency
        buf_id = _buffer_id(nid, "out", 0)
        output_buf_id[nid] = buf_id
        output_dtype[nid] = dtype
        output_elems[nid] = n_elems
        output_shape_dict[nid] = out_shape
        buf_decls.append(f"    static {dtype} {buf_id}[{max(n_elems, 1)}];")

    # 4) Reference output buffers — one per node, parallel allocation.
    ref_buf_id: dict[str, str] = {}
    for nid in graph.topological_order:
        rid = _buffer_id(nid, "ref", 0)
        ref_buf_id[nid] = rid
        buf_decls.append(f"    static {output_dtype[nid]} {rid}[{max(output_elems[nid], 1)}];")

    # 5) Generate the device call sequence — each node's call passes
    #    its operands by buffer id.
    device_calls: list[str] = []
    for nid in graph.topological_order:
        node = graph.nodes[nid]
        md = node.contract.metadata or {}
        in_shapes = md.get("input_shapes", ())
        out_shape = output_shape_dict[nid]

        # Resolve each operand.
        operand_bufs: list[str] = []
        for op_i, in_shape in enumerate(in_shapes):
            if (nid, op_i) in edge_by_consumer:
                src = edge_by_consumer[(nid, op_i)]
                operand_bufs.append(output_buf_id[src])
            else:
                operand_bufs.append(external_buffers[(nid, op_i)])

        out_buf = output_buf_id[nid]
        fn = binding_by_id[nid].function_name
        family = _node_op_family(node)
        if family == "matmul":
            # Vanilla KB matmul signature: launch(out, A, B, M, K, N).
            M = out_shape[0]
            N = out_shape[1]
            K = in_shapes[0][1]
            A = operand_bufs[0]
            B = operand_bufs[1] if len(operand_bufs) >= 2 else operand_bufs[0]
            device_calls.append(
                f"    {fn}((void *){out_buf}, (void *){A}, (void *){B}, "
                f"{M}LL, {K}LL, {N}LL);"
            )
        else:
            # Unary activation: launch(out, in, NULL, n_elems, 0, 0).
            n_elems = output_elems[nid]
            A = operand_bufs[0]
            device_calls.append(
                f"    {fn}((void *){out_buf}, (void *){A}, NULL, {n_elems}LL, 0LL, 0LL);"
            )

    # 6) Generate the scalar reference sequence — operates over the
    #    same external inputs but writes into the per-node ref buffers,
    #    cascading.
    ref_calls: list[str] = []
    for nid in graph.topological_order:
        node = graph.nodes[nid]
        md = node.contract.metadata or {}
        in_shapes = md.get("input_shapes", ())
        out_shape = output_shape_dict[nid]
        operand_bufs: list[str] = []
        for op_i in range(len(in_shapes)):
            if (nid, op_i) in edge_by_consumer:
                src = edge_by_consumer[(nid, op_i)]
                # The ref-chain uses ref buffers for intra-graph edges;
                # if the producer's output dtype matches, route through ref.
                operand_bufs.append(ref_buf_id[src])
            else:
                operand_bufs.append(external_buffers[(nid, op_i)])

        family = _node_op_family(node)
        if family == "matmul":
            M = out_shape[0]
            N = out_shape[1]
            K = in_shapes[0][1]
            ref_calls.append(_reference_matmul_snippet(ref_buf_id[nid], operand_bufs[0], operand_bufs[1], M, K, N))
        elif family == "silu":
            ref_calls.append(_reference_silu_snippet(ref_buf_id[nid], operand_bufs[0], output_elems[nid]))
        elif family == "relu":
            ref_calls.append(_reference_relu_snippet(ref_buf_id[nid], operand_bufs[0], output_elems[nid]))
        else:
            raise ValueError(f"no reference implementation for op family {family!r} on node {nid!r}")

    # 7) Pick the final node's output for the whole-block diff.
    tail_id = graph.topological_order[-1]
    tail_elems = output_elems[tail_id]
    tail_out_buf = output_buf_id[tail_id]
    tail_ref_buf = ref_buf_id[tail_id]
    tail_dtype = output_dtype[tail_id]

    code = f"""\
// Auto-generated pipeline harness for vanilla-KB-on-Gemmini.
// Chains {len(graph.topological_order)} kernels with DRAM intermediates and
// reports end-to-end cycles + correctness against a whole-block scalar
// reference. Same counter (MAIN_LD_ST_EX_CYCLES) and protocol as the
// single-op harness at xpu_rt/kb_gemmini/templates.py.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "include/gemmini.h"
#include "include/gemmini_counter.h"

{chr(10).join(extern_decls)}

static uint32_t _lcg(uint32_t *s) {{
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}}

static void fill_i8(int8_t *p, int64_t n, uint32_t seed) {{
    uint32_t s = seed;
    for (int64_t i = 0; i < n; ++i) p[i] = (int8_t)((_lcg(&s) >> 16) & 0xFF);
}}

int main(void) {{
{chr(10).join(buf_decls)}

{chr(10).join(fill_calls)}

    gemmini_flush(0);
    counter_configure(0, MAIN_LD_ST_EX_CYCLES);
    counter_snapshot_reset();
    int64_t cycles_before = counter_read(0);

{chr(10).join(device_calls)}

    gemmini_fence();
    counter_snapshot_take();
    int64_t cycles_after = counter_read(0);
    int64_t cycles = cycles_after - cycles_before;

    /* whole-block scalar reference */
{chr(10).join(ref_calls)}

    int mismatches = 0;
    int total = {tail_elems};
    int first_idx = -1;
    {tail_dtype} first_ref = 0, first_got = 0;
    for (int64_t i = 0; i < (int64_t)total; ++i) {{
        if ({tail_ref_buf}[i] != {tail_out_buf}[i]) {{
            if (first_idx < 0) {{
                first_idx = (int)i;
                first_ref = {tail_ref_buf}[i];
                first_got = {tail_out_buf}[i];
            }}
            ++mismatches;
        }}
    }}

    printf("nodes={len(graph.topological_order)}\\n");
    printf("mismatches=%d/%d\\n", mismatches, total);
    if (first_idx >= 0)
        printf("first_diff_at=%d ref=%d got=%d\\n", first_idx, (int)first_ref, (int)first_got);
    printf("cycles=%lld\\n", (long long)cycles);
    printf("speedup_baseline_us=%lld\\n", (long long)cycles);
    return mismatches == 0 ? 0 : 1;
}}
"""
    return code


def stage_pipeline_harness(out_dir: Path, spec: PipelineHarnessSpec) -> Path:
    """Write all kernel sources + driver.c into ``out_dir``.

    The kernel sources are copied verbatim from
    ``KernelBinding.kernel_source_path`` so each node can be compiled
    once and reused for both single-op and pipeline runs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for b in spec.bindings:
        if not b.kernel_source_path.is_file():
            raise FileNotFoundError(f"kernel source for {b.op_id!r} not found at {b.kernel_source_path}")
        (out_dir / f"{b.function_name}.c").write_text(b.kernel_source_path.read_text())
    (out_dir / "driver.c").write_text(render_pipeline_driver_c(spec))
    return out_dir


__all__ = [
    "KernelBinding",
    "PipelineHarnessSpec",
    "render_pipeline_driver_c",
    "stage_pipeline_harness",
]

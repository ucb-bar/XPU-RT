"""Agentic-path templates: fused MEGA kernel + whole-block harness.

The vanilla path (:mod:`xpu_rt.kb_gemmini.multiop_harness`) chains N
single-op kernels with DRAM intermediates. The agentic path replaces
that with one fused kernel that keeps intermediates in scratchpad —
the whole point of the MEGA contract.

This module renders two artifacts from a MEGA :class:`KernelContractV3`:

  * ``init.c`` — the starter source the LLM rewrites. The body of
    ``launch_gpu_implementation`` initially computes the chain
    scalar-by-scalar; the agent's job is to replace it with a
    Gemmini-accelerated fused version that uses ``gemmini_mvin`` /
    ``gemmini_preload`` / ``gemmini_compute_preloaded`` (or
    ``tiled_matmul_auto``) and never writes intermediates back to
    DRAM.

  * ``driver.c`` — the harness. Allocates one DRAM buffer per
    external input, one per external output, calls
    ``launch_gpu_implementation`` once, computes the whole-block
    scalar reference, diffs, and prints the
    ``mismatches=N/M`` + ``cycles=N`` protocol that
    :mod:`xpu_rt.kernels.kernelblaster_v2.evaluators.c_riscv`
    parses.

The harness uses the same ``MAIN_LD_ST_EX_CYCLES`` counter as the
single-op (``templates.py``) and vanilla-chain
(``multiop_harness.py``) paths, so cycle counts are directly
comparable across the three runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xpu_rt.kernels.contract_v3 import (
    Granularity,
    KernelArchetype,
    KernelContractV3,
    TensorIO,
)


# ---------------------------------------------------------------------------
# Shape helpers (concrete-only — error on dynamic dims)
# ---------------------------------------------------------------------------


def _require_concrete_shape(io: TensorIO, *, role: str) -> tuple[int, ...]:
    """Return ``io.shape.dims`` as a concrete int tuple.

    Raises:
        ValueError: If any dim is ``None``. The fused harness
            needs concrete shapes to allocate DRAM buffers.
    """
    if any(d is None for d in io.shape.dims):
        raise ValueError(
            f"MEGA harness requires concrete shapes on {role} IO {io.name!r}; "
            f"got dims={io.shape.dims}. Symbolic dims must be resolved at planning time."
        )
    return tuple(int(d) for d in io.shape.dims)


def _elem_count(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _io_dtype_ctype(io: TensorIO) -> str:
    """Map the v3 TensorIO dtype_class[0] to a C ctype.

    The whole-block harness operates over the *external* IO dtype.
    Internal scratchpad dtypes (for body[] sub-kernels) are an
    agent codegen concern — the harness doesn't allocate them.
    """
    if not io.dtype_class:
        return "int8_t"
    dt = io.dtype_class[0]
    return {
        "i8": "int8_t",
        "i32": "int32_t",
        "f32": "float",
    }.get(dt, "int8_t")


# ---------------------------------------------------------------------------
# Reference-snippet catalogue — mirrors multiop_harness's catalogue
# ---------------------------------------------------------------------------


def _matmul_ref_snippet(out_buf: str, lhs: str, rhs: str, M: int, K: int, N: int) -> str:
    return f"""\
    /* matmul reference: {out_buf} = {lhs} @ {rhs}; ({M}x{K}) x ({K}x{N}) -> ({M}x{N}) */
    for (int64_t mi = 0; mi < {M}LL; ++mi) {{
        for (int64_t ni = 0; ni < {N}LL; ++ni) {{
            int32_t acc = 0;
            for (int64_t ki = 0; ki < {K}LL; ++ki) {{
                acc += (int32_t){lhs}[mi*{K}LL + ki] * (int32_t){rhs}[ki*{N}LL + ni];
            }}
            {out_buf}[mi*{N}LL + ni] = acc;
        }}
    }}
"""


def _activation_ref_snippet(out_buf: str, in_buf: str, n: int) -> str:
    """Deterministic int32 activation reference (relu) — matches the
    vanilla path's reference exactly so the cross-backend diff is
    apples-to-apples."""
    return f"""\
    for (int64_t i = 0; i < {n}LL; ++i) {{
        int32_t v = {in_buf}[i];
        {out_buf}[i] = v > 0 ? v : 0;
    }}
"""


# ---------------------------------------------------------------------------
# Public renderers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedKernelArtifacts:
    init_c: str
    driver_c: str


def render_fused_init_c(mega: KernelContractV3) -> str:
    """Render the starter ``init.c`` body for an LLM to rewrite.

    The initial implementation is the scalar chain — the same
    semantics the harness checks against. Whichever way the agent
    rewrites the body of ``launch_gpu_implementation``, it must
    keep this signature (the driver depends on it).
    """
    if mega.granularity is not Granularity.MEGA:
        raise ValueError(f"render_fused_init_c expects Granularity.MEGA; got {mega.granularity}")
    if not mega.body:
        raise ValueError("MEGA contract has empty body")

    n_inputs = len(mega.io.inputs)
    n_outputs = len(mega.io.outputs)

    # External-input ctype list, ordered as in io.inputs.
    input_ctypes = [_io_dtype_ctype(t) for t in mega.io.inputs]
    output_ctypes = [_io_dtype_ctype(t) for t in mega.io.outputs]

    # The starter "fused" body is just the chain of references. The
    # agent's first repair step is typically to replace this with a
    # tiled Gemmini implementation.
    member_names = [sub.op_name for sub in mega.body]
    chain_doc = " -> ".join(member_names)

    return f"""\
// Starter MEGA fused kernel for {mega.op_name!r}.
// Chain: {chain_doc}
//
// Vanilla KB signature preserved (KB's parser depends on it):
//   void launch_gpu_implementation(void *output,
//                                  void *input_A,
//                                  void *input_B,
//                                  int64_t M, int64_t K, int64_t N)
//
// For the MEGA case the harness packs *all* external inputs into a
// single contiguous DRAM region pointed to by input_A; input_B is
// reserved for the weight matrix of the dominant matmul (when
// applicable). M/K/N carry the outer-loop bounds of the chain (the
// agent reads body[] metadata via the prompt context for the
// inner-loop shapes).
//
// External inputs ({n_inputs}): {[t.name for t in mega.io.inputs]}
// External outputs ({n_outputs}): {[t.name for t in mega.io.outputs]}

#include <stdint.h>
#include <stddef.h>
#include "include/gemmini.h"

void launch_gpu_implementation(void *output,
                               void *input_A,
                               void *input_B,
                               int64_t M, int64_t K, int64_t N) {{
    /* AGENT: replace this body with a fused Gemmini kernel that keeps
       all intermediates in scratchpad and never writes them back to
       DRAM. The reference behaviour (what the harness checks) is
       captured by the per-op snippets in mega_templates.py.

       For the matmul-silu-matmul chain the canonical fused pattern is:
         1) tiled_matmul_auto(... A=input_A, B=input_B, OUT=scratchpad)
         2) elementwise activation kept in scratchpad
         3) tiled_matmul_auto(... A=scratchpad, B=... , OUT=output)
       with shared scratchpad addressing across (1)->(2)->(3). */
    (void)output;
    (void)input_A;
    (void)input_B;
    (void)M; (void)K; (void)N;
}}
"""


def render_fused_driver_c(mega: KernelContractV3) -> str:
    """Render the whole-block driver harness for a MEGA contract.

    The harness:
      * allocates per-external-input + per-external-output DRAM
        buffers (concrete shapes required);
      * calls ``launch_gpu_implementation`` once (passing the
        chain's outer-loop M/K/N from the dominant matmul if any);
      * runs the scalar reference for every body[] sub-contract,
        cascading the outputs to mimic the chain semantics;
      * diffs the device output vs. reference, prints the standard
        ``mismatches=N/M`` + ``cycles=N`` protocol.

    Raises:
        ValueError: When the MEGA's IO shapes are not concrete, or
            when no body[] sub-kernel yields a concrete output shape
            for the outer-loop bounds.
    """
    if mega.granularity is not Granularity.MEGA:
        raise ValueError(f"render_fused_driver_c expects Granularity.MEGA; got {mega.granularity}")
    if not mega.body:
        raise ValueError("MEGA contract has empty body")

    # External-input + external-output allocations.
    in_decls: list[str] = []
    fill_calls: list[str] = []
    in_buf_names: list[str] = []
    seed = 0
    for i, t in enumerate(mega.io.inputs):
        shape = _require_concrete_shape(t, role="external input")
        n = _elem_count(shape)
        ctype = _io_dtype_ctype(t)
        buf = f"mega_in_{i}"
        in_buf_names.append(buf)
        in_decls.append(f"    static {ctype} {buf}[{max(n, 1)}];")
        if ctype == "int8_t":
            fill_calls.append(f"    fill_i8({buf}, {n}LL, 0x{seed + 0xC0FFEE:08X});")
        elif ctype == "int32_t":
            fill_calls.append(
                f"    for (int64_t i=0;i<{n}LL;++i) {buf}[i] = (int32_t)((_lcg(&seed_state) >> 8) & 0xFFFF) - 32768;"
            )
        else:
            fill_calls.append(f"    memset({buf}, 0, sizeof({buf}));")
        seed += 1

    out_decls: list[str] = []
    out_ref_decls: list[str] = []
    out_buf_names: list[str] = []
    out_ref_names: list[str] = []
    out_elem_counts: list[int] = []
    out_ctypes: list[str] = []
    for j, t in enumerate(mega.io.outputs):
        shape = _require_concrete_shape(t, role="external output")
        n = _elem_count(shape)
        ctype = _io_dtype_ctype(t)
        # Force int32_t for matmul-tailed chains so the reference math
        # (which accumulates in i32) compares exactly.
        device_ctype = "int32_t" if mega.archetype is KernelArchetype.COMPUTE_TILED else ctype
        out_ctypes.append(device_ctype)
        buf = f"mega_out_{j}"
        ref = f"mega_ref_{j}"
        out_buf_names.append(buf)
        out_ref_names.append(ref)
        out_elem_counts.append(n)
        out_decls.append(f"    static {device_ctype} {buf}[{max(n, 1)}];")
        out_ref_decls.append(f"    static {device_ctype} {ref}[{max(n, 1)}];")

    # Whole-block scalar reference: cascade through body[].
    # Each sub-contract has lifted inputs / outputs; we walk them in
    # order and route by external_input routing metadata. The
    # MegaContractEmitter recorded that routing in
    # ``mega.metadata['external_inputs_routing']`` / outputs_routing.
    # For the reference, we need a *concrete* cascade — each sub's
    # inputs come from the external buffers or the previous sub's
    # ref output buffer.
    sub_ref_decls: list[str] = []
    sub_ref_names: list[str] = []
    for k, sub in enumerate(mega.body):
        if not sub.io.outputs:
            raise ValueError(f"MEGA body[{k}] has no outputs")
        shape = _require_concrete_shape(sub.io.outputs[0], role=f"body[{k}].output[0]")
        n = _elem_count(shape)
        # Use i32 for matmul outputs, i32 for activation outputs (we
        # cascade in i32 across the chain for deterministic compare).
        ref = f"sub_ref_{k}"
        sub_ref_decls.append(f"    static int32_t {ref}[{max(n, 1)}];")
        sub_ref_names.append(ref)

    # Refer the inputs of each sub-kernel back to external buffers
    # (operand_index < n_external_inputs that map to this sub) OR
    # to the previous sub's ref buffer (linear chain assumption).
    # We treat the chain as: sub[0] external inputs only; sub[k>0]
    # consumes sub[k-1].output as its first operand and external for
    # the rest. This matches what MegaContractEmitter currently
    # routes for chain-shaped clusters.
    ref_call_blocks: list[str] = []
    ext_in_routing = list(mega.metadata.get("external_inputs_routing", []))
    # Map (member_idx, operand_idx) → external_buf_name
    member_op_ids: list = list(mega.metadata.get("member_op_ids", []))
    op_id_to_member_idx = {mid: i for i, mid in enumerate(member_op_ids)}
    ext_op_to_buf: dict[tuple[int, int], str] = {}
    for ext_idx, (mid, op_i) in enumerate(ext_in_routing):
        if mid not in op_id_to_member_idx:
            continue
        m_idx = op_id_to_member_idx[mid]
        # routing may carry -1 sentinel for padded archetype-fix inputs.
        if op_i < 0:
            continue
        ext_op_to_buf[(m_idx, op_i)] = in_buf_names[ext_idx]

    for k, sub in enumerate(mega.body):
        archetype = sub.archetype
        sub_out = _require_concrete_shape(sub.io.outputs[0], role=f"body[{k}].output[0]")
        if archetype is KernelArchetype.COMPUTE_TILED:
            # Two inputs: input0 is either a previous sub's ref or external; input1 is external (weight).
            in0_shape = _require_concrete_shape(sub.io.inputs[0], role=f"body[{k}].input[0]")
            M = sub_out[0]
            N = sub_out[1]
            K = in0_shape[1]
            in0_buf = ext_op_to_buf.get((k, 0))
            in1_buf = ext_op_to_buf.get((k, 1))
            if in0_buf is None and k > 0:
                in0_buf = sub_ref_names[k - 1]
            if in0_buf is None or in1_buf is None:
                raise ValueError(
                    f"body[{k}] is COMPUTE_TILED but operand routing incomplete: "
                    f"in0={in0_buf} in1={in1_buf}"
                )
            ref_call_blocks.append(_matmul_ref_snippet(sub_ref_names[k], in0_buf, in1_buf, M, K, N))
        else:
            # Unary: input from prev ref or external.
            in0_buf = ext_op_to_buf.get((k, 0))
            if in0_buf is None and k > 0:
                in0_buf = sub_ref_names[k - 1]
            if in0_buf is None:
                raise ValueError(f"body[{k}] unary op has no operand 0 routing")
            ref_call_blocks.append(_activation_ref_snippet(sub_ref_names[k], in0_buf, _elem_count(sub_out)))

    # Final ref result: copy from sub_ref of the last body into mega_ref_0.
    tail_ref = sub_ref_names[-1]
    tail_elems = out_elem_counts[0]
    final_copy = f"""\
    for (int64_t i = 0; i < {tail_elems}LL; ++i) {out_ref_names[0]}[i] = {tail_ref}[i];
"""

    # Outer-loop M/K/N for the launch call: pull from the chain's first
    # COMPUTE_TILED member if any, otherwise zeros.
    outer_M = 0
    outer_K = 0
    outer_N = 0
    for sub in mega.body:
        if sub.archetype is KernelArchetype.COMPUTE_TILED:
            sub_out = _require_concrete_shape(sub.io.outputs[0], role="body.outer-loop output")
            in0_shape = _require_concrete_shape(sub.io.inputs[0], role="body.outer-loop input0")
            outer_M, outer_K, outer_N = sub_out[0], in0_shape[1], sub_out[1]
            break

    return f"""\
// Auto-generated MEGA harness for fused chain {mega.op_name!r}.
// Single device-side call to launch_gpu_implementation runs the whole
// fused body[]; the scalar reference cascade computes the same
// semantics op-by-op for the end-to-end diff. Counter is
// MAIN_LD_ST_EX_CYCLES (matches the single-op + vanilla-chain paths).

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "include/gemmini.h"
#include "include/gemmini_counter.h"

extern void launch_gpu_implementation(void *output,
                                      void *input_A,
                                      void *input_B,
                                      int64_t M, int64_t K, int64_t N);

static uint32_t _lcg(uint32_t *s) {{
    *s = (*s) * 1103515245u + 12345u;
    return *s;
}}

static void fill_i8(int8_t *p, int64_t n, uint32_t seed) {{
    uint32_t s = seed;
    for (int64_t i = 0; i < n; ++i) p[i] = (int8_t)((_lcg(&s) >> 16) & 0xFF);
}}

int main(void) {{
    uint32_t seed_state = 0xC0FFEE;
{chr(10).join(in_decls)}
{chr(10).join(out_decls)}
{chr(10).join(out_ref_decls)}
{chr(10).join(sub_ref_decls)}

{chr(10).join(fill_calls)}

    gemmini_flush(0);
    counter_configure(0, MAIN_LD_ST_EX_CYCLES);
    counter_snapshot_reset();
    int64_t cycles_before = counter_read(0);

    /* The harness passes the dominant matmul's (M, K, N) so KB-style
       kernels that read these parameters keep working; the agent's
       fused kernel may ignore them and dispatch over body[] inner
       loops directly. */
    launch_gpu_implementation((void *){out_buf_names[0]},
                              (void *){in_buf_names[0] if in_buf_names else "NULL"},
                              (void *){in_buf_names[1] if len(in_buf_names) >= 2 else "NULL"},
                              {outer_M}LL, {outer_K}LL, {outer_N}LL);

    gemmini_fence();
    counter_snapshot_take();
    int64_t cycles_after = counter_read(0);
    int64_t cycles = cycles_after - cycles_before;

    /* Whole-block scalar reference */
{chr(10).join(ref_call_blocks)}
{final_copy}

    int mismatches = 0;
    int total = {tail_elems};
    int first_idx = -1;
    {out_ctypes[0]} first_ref = 0, first_got = 0;
    for (int64_t i = 0; i < (int64_t)total; ++i) {{
        if ({out_ref_names[0]}[i] != {out_buf_names[0]}[i]) {{
            if (first_idx < 0) {{
                first_idx = (int)i;
                first_ref = {out_ref_names[0]}[i];
                first_got = {out_buf_names[0]}[i];
            }}
            ++mismatches;
        }}
    }}

    printf("mega=1 nodes={len(mega.body)}\\n");
    printf("mismatches=%d/%d\\n", mismatches, total);
    if (first_idx >= 0)
        printf("first_diff_at=%d ref=%d got=%d\\n", first_idx, (int)first_ref, (int)first_got);
    printf("cycles=%lld\\n", (long long)cycles);
    printf("speedup_baseline_us=%lld\\n", (long long)cycles);
    return mismatches == 0 ? 0 : 1;
}}
"""


def render_fused_artifacts(mega: KernelContractV3) -> FusedKernelArtifacts:
    """Convenience: render both init.c and driver.c together."""
    return FusedKernelArtifacts(
        init_c=render_fused_init_c(mega),
        driver_c=render_fused_driver_c(mega),
    )


def stage_mega_contract_dir(out_dir: Path, mega: KernelContractV3) -> Path:
    """Write ``init.cu`` + ``driver.cpp`` (vanilla KB filename
    convention) into ``out_dir``. Returns the directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = render_fused_artifacts(mega)
    (out_dir / "init.cu").write_text(artifacts.init_c)
    (out_dir / "driver.cpp").write_text(artifacts.driver_c)
    return out_dir


__all__ = [
    "FusedKernelArtifacts",
    "render_fused_artifacts",
    "render_fused_driver_c",
    "render_fused_init_c",
    "stage_mega_contract_dir",
]

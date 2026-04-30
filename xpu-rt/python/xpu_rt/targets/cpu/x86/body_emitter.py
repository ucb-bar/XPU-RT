"""x86 CPU body emitter — emits portable C++ for GEMM + elementwise.

The stub deliberately does NOT use SIMD intrinsics. Goal of Wave
1.15 is to validate that the abstraction holds outside NVIDIA-land,
not to ship a fast CPU implementation. Bodies are scalar-fmaf C++
with three nested loops; the C++ compiler's auto-vectorizer
exploits AVX where available.

Future arch-leaves under ``targets/cpu/x86/avx512/``,
``targets/cpu/x86/avx2/`` etc. would specialize with intrinsics.

Each body returned satisfies the same
:class:`compgen.transforms.emit_cuda_megakernel.DeviceFunctionSource`
contract the GPU emitters use — the universal lowering matcher
treats CPU and GPU bodies identically.
"""

from __future__ import annotations

from compgen.transforms.emit_cuda_megakernel import DeviceFunctionSource


class X86BodyEmitter:
    """Emit portable C++ kernel bodies for the universal matcher."""

    def preferred_tile_shape(self, *, op: str, dtype: str) -> tuple[int, int, int]:
        """Cache-line-friendly tile defaults. 32×32×32 keeps each
        tile in L1 (~3 KB at fp32) and aligns with the GPU
        fmaf-path tile so the matcher's divisibility checks port
        cleanly."""
        del op, dtype  # vendor-blind for the stub
        return (32, 32, 32)

    def gemm(
        self,
        *,
        b_dim: int,
        k_dim: int,
        n_dim: int,
        n_tiles_per_row: int,
        x_buf: int,
        w_buf: int,
        out_buf: int,
        precision: str,
        tile_m: int,
        tile_n: int,
        tile_k: int,
    ) -> DeviceFunctionSource:
        """Emit a CPU GEMM body. One C function per task; the
        runtime wraps multiple of these into a single .so. The
        body computes a (tile_m × tile_n) output tile by
        accumulating over k."""
        del precision  # the stub only does fp32
        body = f"""
// CPU GEMM body — Wave 1.15 stub. Vendor-blind C++ that the
// system C++ compiler auto-vectorizes for the available SIMD width.
// task_id, sm_id, buffers come from the universal megakernel
// wrapper; we ignore sm_id (no per-SM concept on CPU).
const int B = {b_dim};
const int IN = {k_dim};
const int OUT = {n_dim};
const int TM = {tile_m}, TN = {tile_n}, TK = {tile_k};
const int TILES_PER_ROW = {n_tiles_per_row};

(void)sm_id;
const int row_tile_idx = task_id / TILES_PER_ROW;
const int col_tile_idx = task_id % TILES_PER_ROW;
const int row_start = row_tile_idx * TM;
const int col_start = col_tile_idx * TN;

const float *x = (const float *)buffers[{x_buf}];
const float *w = (const float *)buffers[{w_buf}];
float *y       = (float *)buffers[{out_buf}];

for (int m = 0; m < TM; ++m) {{
    int out_row = row_start + m;
    if (out_row >= B) break;
    for (int n = 0; n < TN; ++n) {{
        int out_col = col_start + n;
        if (out_col >= OUT) break;
        float acc = 0.0f;
        for (int k = 0; k < IN; ++k) {{
            // x is (B, IN) row-major; w is (OUT, IN) row-major
            // (nn.Linear weight layout) — same as GPU side.
            acc += x[out_row * IN + k] * w[out_col * IN + k];
        }}
        y[out_row * OUT + out_col] = acc;
    }}
}}
"""
        return DeviceFunctionSource(name=f"cpu_gemm_t{x_buf}_{w_buf}_{out_buf}", body=body)

    def elementwise(
        self,
        *,
        op: str,
        total_elems: int,
        in_bufs: tuple[int, ...],
        out_buf: int,
        tile_m: int,
        tile_n: int,
    ) -> DeviceFunctionSource:
        """Emit a CPU elementwise body. Tile-aware loop covers
        TM × TN per task — same shape as the GPU's loop pattern,
        just no thread-level parallelism (SIMD comes from the
        compiler's auto-vectorizer)."""
        del tile_m, tile_n  # task_id indexes flat output
        if op == "relu":
            assert len(in_bufs) == 1
            in_buf = in_bufs[0]
            kernel_op = (
                f"const float *in_p = (const float *)buffers[{in_buf}];\n"
                f"float       *out_p = (float *)buffers[{out_buf}];\n"
                f"for (int i = 0; i < {total_elems}; ++i) {{\n"
                f"    float v = in_p[i];\n"
                f"    out_p[i] = v > 0.0f ? v : 0.0f;\n"
                f"}}"
            )
            name = f"cpu_relu_o{out_buf}"
        elif op == "add":
            assert len(in_bufs) == 2
            a_buf, b_buf = in_bufs
            kernel_op = (
                f"const float *a = (const float *)buffers[{a_buf}];\n"
                f"const float *b = (const float *)buffers[{b_buf}];\n"
                f"float       *out_p = (float *)buffers[{out_buf}];\n"
                f"for (int i = 0; i < {total_elems}; ++i) {{\n"
                f"    out_p[i] = a[i] + b[i];\n"
                f"}}"
            )
            name = f"cpu_add_o{out_buf}"
        else:
            raise ValueError(f"X86BodyEmitter.elementwise: unsupported op {op!r}. Supported in stub: 'relu', 'add'.")

        body = f"""
// CPU {op} body — Wave 1.15 stub.
(void)task_id; (void)sm_id;
{kernel_op}
"""
        return DeviceFunctionSource(name=name, body=body)

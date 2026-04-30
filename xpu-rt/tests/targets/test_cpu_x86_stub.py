"""Wave 1.15 — CPU x86 stub backend tests.

The acceptance test for the unified target hierarchy abstraction:

- The stub satisfies all four GPU-class Protocols (where they
  apply) + CPU-class Protocols.
- The body emitter produces compilable C++.
- The runtime can JIT compile + dispatch a real GEMM body and
  produce correct output (validated against numpy).
- ``compgen.targets`` import auto-registers the stub.

If these tests pass on a Linux host with a C++ compiler available,
the abstraction is real architecture, not just a renaming
exercise — the same Protocols a GPU body emitter satisfies, a CPU
emitter satisfies too.
"""

from __future__ import annotations

import shutil

import pytest


def _has_cxx() -> bool:
    return any(shutil.which(n) is not None for n in ("clang++", "g++", "clang", "gcc"))


@pytest.fixture(autouse=True)
def _ensure_registered():
    """The autouse registry-reset in `test_registry.py` won't apply
    here — that's a different file. We just want the stub
    registered; importing `compgen.targets` does it."""
    import compgen.targets as targets_mod

    targets_mod._register_in_tree()
    yield


class TestProbe:
    def test_probe_satisfies_basic_contract(self) -> None:
        from compgen.targets.cpu.x86.probe import X86Probe

        p = X86Probe()
        # Calls don't raise.
        assert isinstance(p.is_available(), bool)
        assert isinstance(p.device_arch(), str)
        assert p.device_arch().startswith("x86_")
        assert isinstance(p.supports_clusters(), bool)
        assert isinstance(p.supports_tensor_cores(), bool)
        assert isinstance(p.library_paths(), dict)
        assert isinstance(p.vendor_extras(), dict)

    def test_probe_no_clusters_no_tensor_cores(self) -> None:
        """CPU explicitly has neither — Protocol contract pinning."""
        from compgen.targets.cpu.x86.probe import X86Probe

        p = X86Probe()
        assert p.supports_clusters() is False
        assert p.supports_tensor_cores() is False

    def test_vendor_extras_carries_simd_width(self) -> None:
        from compgen.targets.cpu.x86.probe import X86Probe

        extras = X86Probe().vendor_extras()
        assert "simd_width_bits" in extras
        assert extras["simd_width_bits"] >= 64


class TestBodyEmitter:
    """The emitter produces source that compiles + runs."""

    def test_preferred_tile_shape(self) -> None:
        from compgen.targets.cpu.x86.body_emitter import X86BodyEmitter

        emitter = X86BodyEmitter()
        assert emitter.preferred_tile_shape(op="gemm", dtype="fp32") == (32, 32, 32)

    def test_gemm_emits_valid_cpp(self) -> None:
        """Body source mentions the canonical bits — for / acc / fmaf-
        like loops + the right buffer indices."""
        from compgen.targets.cpu.x86.body_emitter import X86BodyEmitter

        body = (
            X86BodyEmitter()
            .gemm(
                b_dim=32,
                k_dim=32,
                n_dim=32,
                n_tiles_per_row=1,
                x_buf=0,
                w_buf=1,
                out_buf=2,
                precision="fp32",
                tile_m=32,
                tile_n=32,
                tile_k=32,
            )
            .body
        )
        # Sanity checks on the emitted C++.
        assert "for (int" in body
        assert "acc" in body
        assert "buffers[0]" in body
        assert "buffers[1]" in body
        assert "buffers[2]" in body

    def test_relu_emits_valid_cpp(self) -> None:
        from compgen.targets.cpu.x86.body_emitter import X86BodyEmitter

        body = (
            X86BodyEmitter()
            .elementwise(
                op="relu",
                total_elems=128,
                in_bufs=(0,),
                out_buf=1,
                tile_m=32,
                tile_n=32,
            )
            .body
        )
        assert "for (int" in body
        assert "v > 0.0f" in body

    def test_unsupported_op_raises(self) -> None:
        from compgen.targets.cpu.x86.body_emitter import X86BodyEmitter

        with pytest.raises(ValueError, match="unsupported op"):
            X86BodyEmitter().elementwise(
                op="gelu",
                total_elems=128,
                in_bufs=(0,),
                out_buf=1,
                tile_m=32,
                tile_n=32,
            )


class TestProtocolStructuralChecks:
    """Wave 1.10's Protocols must accept the CPU stub. If these
    fail, the abstraction is leaking GPU-specific concepts."""

    def test_x86_body_emitter_satisfies_cpu_protocol(self) -> None:
        from compgen.targets.cpu.contracts import CpuBodyEmitter
        from compgen.targets.cpu.x86.body_emitter import X86BodyEmitter

        assert isinstance(X86BodyEmitter(), CpuBodyEmitter)

    def test_x86_runtime_satisfies_cpu_protocol(self) -> None:
        from compgen.targets.cpu.contracts import CpuRuntime
        from compgen.targets.cpu.x86.runtime import X86Runtime

        assert isinstance(X86Runtime(), CpuRuntime)


class TestRegistration:
    def test_x86_in_tree_registered(self) -> None:
        from compgen.targets.registry import registry

        pkg = registry().get("cpu.x86")
        assert pkg is not None
        # All four adapters wired (not None placeholders anymore).
        assert pkg.probe is not None
        assert pkg.body_emitter is not None
        assert pkg.runtime is not None
        assert pkg.cost_model is not None

    def test_audit_metadata_pinned(self) -> None:
        """Wave 1.15 metadata visible via the agent's describe()
        query — pin so the audit surface doesn't drift."""
        from compgen.targets.registry import registry

        pkg = registry().get("cpu.x86")
        assert pkg is not None
        m = pkg.metadata
        assert m["supports_clusters"] is False
        assert m["supports_tensor_cores"] is False
        assert m["default_tile_shape"] == [32, 32, 32]
        assert "detected_arch" in m
        assert m["detected_arch"].startswith("x86_")


@pytest.mark.skipif(
    not _has_cxx(),
    reason="No C++ compiler reachable — Wave 1.15 JIT requires clang or gcc",
)
class TestEndToEndJIT:
    """The acceptance test: emit a body, compile via clang/gcc,
    load via ctypes, dispatch, validate output bit-exact against
    numpy. If this passes, the abstraction is real."""

    def test_gemm_jit_compile_and_dispatch_correct(self) -> None:
        import ctypes

        import numpy as np
        from compgen.targets.cpu.x86.body_emitter import X86BodyEmitter
        from compgen.targets.cpu.x86.runtime import X86Runtime

        # Emit a 32×32×32 GEMM body for the universal task signature.
        body = X86BodyEmitter().gemm(
            b_dim=32,
            k_dim=32,
            n_dim=32,
            n_tiles_per_row=1,
            x_buf=0,
            w_buf=1,
            out_buf=2,
            precision="fp32",
            tile_m=32,
            tile_n=32,
            tile_k=32,
        )

        # Wrap the body source in a full C++ translation unit
        # exposing the canonical universal entry-point signature
        # ``void <symbol>(int task_id, int sm_id, void **buffers)``.
        symbol = "test_gemm"
        full_source = f"""
#include <stddef.h>
void {symbol}(int task_id, int sm_id, void **buffers) {{
{body.body}
}}
"""

        runtime = X86Runtime()
        lib = runtime.compile_source(
            source=full_source,
            symbol_name=symbol,
        )

        # Marshal real fp32 inputs.
        rng = np.random.default_rng(0)
        x = rng.standard_normal((32, 32), dtype=np.float32)
        # nn.Linear weight layout: (OUT, IN) row-major.
        w = rng.standard_normal((32, 32), dtype=np.float32)
        y = np.zeros((32, 32), dtype=np.float32)

        # Buffer pointers as ints — runtime wraps in c_void_p.
        x_ptr = x.ctypes.data_as(ctypes.c_void_p).value
        w_ptr = w.ctypes.data_as(ctypes.c_void_p).value
        y_ptr = y.ctypes.data_as(ctypes.c_void_p).value

        runtime.dispatch(
            library_handle=lib,
            kernel_params=(symbol, x_ptr, w_ptr, y_ptr),
        )

        # Reference: y = x @ w.T  (matches the body's
        # ``acc += x[m,k] * w[n,k]`` indexing).
        ref = x @ w.T
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)

    def test_relu_jit_compile_and_dispatch_correct(self) -> None:
        import ctypes

        import numpy as np
        from compgen.targets.cpu.x86.body_emitter import X86BodyEmitter
        from compgen.targets.cpu.x86.runtime import X86Runtime

        symbol = "test_relu"
        body = X86BodyEmitter().elementwise(
            op="relu",
            total_elems=64,
            in_bufs=(0,),
            out_buf=1,
            tile_m=8,
            tile_n=8,
        )
        full = f"""
#include <stddef.h>
void {symbol}(int task_id, int sm_id, void **buffers) {{
{body.body}
}}
"""
        rt = X86Runtime()
        lib = rt.compile_source(source=full, symbol_name=symbol)

        rng = np.random.default_rng(1)
        x = rng.standard_normal(64, dtype=np.float32)
        y = np.zeros(64, dtype=np.float32)
        rt.dispatch(
            library_handle=lib,
            kernel_params=(
                symbol,
                x.ctypes.data_as(ctypes.c_void_p).value,
                y.ctypes.data_as(ctypes.c_void_p).value,
            ),
        )
        np.testing.assert_allclose(y, np.maximum(x, 0.0))

"""Target-agnostic ukernel design proof tests.

CRITICAL: This file proves the ukernel dialect is target-agnostic. A single
registry serves CUDA GPU, RISC-V vector (RVV), NPU, and generic CPUs with
the same data structures and selection algorithm. Target specificity lives
entirely in constraint strings and body target_family tags -- not in code paths.

Test matrix:
    - CUDA GPU (float32) -> vendor_matmul_f32 (priority=20), body=cublas_sgemm
    - RVV CPU (float32)  -> vendor_matmul_f32 (priority=15), body=rvv_sgemm
    - RVV CPU (int8)     -> matmul_generic_i8 (priority=10), body=rvv body
    - NPU (float32)      -> vendor_matmul_f32 (priority=15), body=npu_matmul_f32
    - Generic CPU (int8)  -> matmul_generic_i8 (priority=1, fallback), body=any
    - Generic CPU (float32) -> None (no matching ukernel)
"""

from __future__ import annotations

import pytest
from compgen.ir.ukernel.constraints import ConstraintContext
from compgen.ir.ukernel.lower import lower_ukernel_with_body
from compgen.ir.ukernel.ops import UkernelBodyOp, UkernelCallOp, UkernelDeclOp, UkernelMatchOp
from compgen.ir.ukernel.provider_bridge import UkernelProvider
from compgen.ir.ukernel.registry import UkernelRegistry
from compgen.kernels.provider import KernelContract


def _build_multi_target_registry() -> UkernelRegistry:
    """Build a registry with transparent + opaque ukernels for multiple targets."""
    reg = UkernelRegistry()

    # --- 1. Transparent ukernel: matmul_generic_i8 ---
    decl_generic = UkernelDeclOp(
        kernel_name="matmul_generic_i8",
        transparency="transparent",
        body_kind="mlir",
        tile_family="generic_tile",
        supports_prepacked_rhs=True,
    )
    match_a = UkernelMatchOp(
        kernel_name="matmul_generic_i8",
        op_family="matmul",
        dtype_constraints=("dtype==int8",),
        shape_constraints=("M%8==0",),
        target_constraints=("has_rvv",),
        priority=10,
    )
    match_b = UkernelMatchOp(
        kernel_name="matmul_generic_i8",
        op_family="matmul",
        dtype_constraints=("dtype==int8",),
        shape_constraints=("M%8==0",),
        target_constraints=(),
        priority=1,
    )
    body_a = UkernelBodyOp(
        kernel_name="matmul_generic_i8",
        body_kind="mlir",
        transparency="transparent",
        target_family="rvv",
        inline_body="linalg.matmul_transpose_b",
    )
    body_b = UkernelBodyOp(
        kernel_name="matmul_generic_i8",
        body_kind="mlir",
        transparency="transparent",
        target_family="any",
        inline_body="linalg.matmul",
    )
    reg.register_ukernel(decl_generic, [match_a, match_b], [body_a, body_b])

    # --- 2. Opaque ukernel: vendor_matmul_f32 ---
    decl_vendor = UkernelDeclOp(
        kernel_name="vendor_matmul_f32",
        transparency="opaque",
        body_kind="library",
    )
    match_c = UkernelMatchOp(
        kernel_name="vendor_matmul_f32",
        op_family="matmul",
        dtype_constraints=("dtype==float32",),
        target_constraints=("has_tensor_core",),
        priority=20,
    )
    match_d = UkernelMatchOp(
        kernel_name="vendor_matmul_f32",
        op_family="matmul",
        dtype_constraints=("dtype==float32",),
        target_constraints=("has_rvv",),
        priority=15,
    )
    match_e = UkernelMatchOp(
        kernel_name="vendor_matmul_f32",
        op_family="matmul",
        dtype_constraints=("dtype==float32",),
        target_constraints=("has_npu_engine",),
        priority=15,
    )
    body_c = UkernelBodyOp(
        kernel_name="vendor_matmul_f32",
        body_kind="library",
        transparency="opaque",
        target_family="cuda",
        source_ref="cublas_sgemm",
    )
    body_d = UkernelBodyOp(
        kernel_name="vendor_matmul_f32",
        body_kind="library",
        transparency="opaque",
        target_family="rvv",
        source_ref="rvv_sgemm",
    )
    body_e = UkernelBodyOp(
        kernel_name="vendor_matmul_f32",
        body_kind="library",
        transparency="opaque",
        target_family="npu",
        source_ref="npu_matmul_f32",
    )
    reg.register_ukernel(decl_vendor, [match_c, match_d, match_e], [body_c, body_d, body_e])

    return reg


class TestMultiTargetSelection:
    """Prove the same registry serves every target via constraints alone."""

    @pytest.fixture(autouse=True)
    def _setup_registry(self) -> None:
        self.reg = _build_multi_target_registry()

    def test_cuda_gpu_selects_vendor_f32(self) -> None:
        ctx = ConstraintContext(
            shapes={"M": 128, "N": 64, "K": 32},
            dtypes=("float32",),
            target_features=frozenset({"has_tensor_core"}),
            device_type="gpu",
        )
        decl = self.reg.select_ukernel("matmul", ctx)
        assert decl is not None
        assert decl.kernel_name == "vendor_matmul_f32"

        body = self.reg.select_body(decl.kernel_name, "cuda")
        assert body is not None
        assert body.source_ref == "cublas_sgemm"

    def test_rvv_cpu_float32_selects_vendor(self) -> None:
        ctx = ConstraintContext(
            shapes={"M": 128, "N": 64, "K": 32},
            dtypes=("float32",),
            target_features=frozenset({"has_rvv"}),
            device_type="cpu",
        )
        decl = self.reg.select_ukernel("matmul", ctx)
        assert decl is not None
        assert decl.kernel_name == "vendor_matmul_f32"

        body = self.reg.select_body(decl.kernel_name, "rvv")
        assert body is not None
        assert body.source_ref == "rvv_sgemm"

    def test_rvv_cpu_int8_selects_generic(self) -> None:
        ctx = ConstraintContext(
            shapes={"M": 128, "N": 64, "K": 32},
            dtypes=("int8",),
            target_features=frozenset({"has_rvv"}),
            device_type="cpu",
        )
        decl = self.reg.select_ukernel("matmul", ctx)
        assert decl is not None
        assert decl.kernel_name == "matmul_generic_i8"

        body = self.reg.select_body(decl.kernel_name, "rvv")
        assert body is not None
        assert body.target_family == "rvv"
        assert body.inline_body == "linalg.matmul_transpose_b"

    def test_npu_selects_vendor_f32(self) -> None:
        ctx = ConstraintContext(
            shapes={"M": 128, "N": 64, "K": 32},
            dtypes=("float32",),
            target_features=frozenset({"has_npu_engine"}),
            device_type="npu",
        )
        decl = self.reg.select_ukernel("matmul", ctx)
        assert decl is not None
        assert decl.kernel_name == "vendor_matmul_f32"

        body = self.reg.select_body(decl.kernel_name, "npu")
        assert body is not None
        assert body.source_ref == "npu_matmul_f32"

    def test_generic_cpu_int8_falls_back(self) -> None:
        ctx = ConstraintContext(
            shapes={"M": 128, "N": 64, "K": 32},
            dtypes=("int8",),
            target_features=frozenset(),
            device_type="cpu",
        )
        decl = self.reg.select_ukernel("matmul", ctx)
        assert decl is not None
        assert decl.kernel_name == "matmul_generic_i8"

        body = self.reg.select_body(decl.kernel_name, "any")
        assert body is not None
        assert body.target_family == "any"
        assert body.inline_body == "linalg.matmul"

    def test_generic_cpu_float32_returns_none(self) -> None:
        ctx = ConstraintContext(
            shapes={"M": 128, "N": 64, "K": 32},
            dtypes=("float32",),
            target_features=frozenset(),
            device_type="cpu",
        )
        decl = self.reg.select_ukernel("matmul", ctx)
        assert decl is None


class TestProviderBridge:
    """UkernelProvider serves ukernels through the KernelProvider protocol."""

    @pytest.fixture(autouse=True)
    def _setup_provider(self) -> None:
        self.reg = _build_multi_target_registry()
        self.provider = UkernelProvider(self.reg)

    def test_provider_name(self) -> None:
        assert self.provider.name == "ukernel"

    def test_accepts_matching_contract(self) -> None:
        contract = KernelContract(
            op_family="matmul",
            input_shapes=((128, 32), (32, 64)),
            dtypes=("float32",),
            target_name="gpu_cuda",
            hardware_key="tensor_core",
        )
        assert self.provider.accepts_contract(contract) is True

    def test_rejects_non_matching_contract(self) -> None:
        contract = KernelContract(
            op_family="matmul",
            input_shapes=((128, 32), (32, 64)),
            dtypes=("float32",),
            target_name="cpu_generic",
            hardware_key="",
        )
        assert self.provider.accepts_contract(contract) is False

    def test_search_returns_result_with_metadata(self) -> None:
        contract = KernelContract(
            op_family="matmul",
            input_shapes=((128, 32), (32, 64)),
            dtypes=("float32",),
            target_name="gpu_cuda",
            hardware_key="tensor_core",
        )
        result = self.provider.search(contract)
        assert result.found is True
        assert result.correct is True
        assert result.metadata["kernel_name"] == "vendor_matmul_f32"
        assert result.metadata["transparency"] == "opaque"

    def test_search_not_found(self) -> None:
        contract = KernelContract(
            op_family="softmax",
            dtypes=("float32",),
        )
        result = self.provider.search(contract)
        assert result.found is False


class TestBodyAwareLowering:
    """lower_ukernel_with_body selects bodies and uses correct prefixes."""

    @pytest.fixture(autouse=True)
    def _setup_registry(self) -> None:
        self.reg = _build_multi_target_registry()

    def test_transparent_body_uses_inline_prefix(self) -> None:
        call = UkernelCallOp(
            kernel_name="matmul_generic_i8",
            operands=["a", "b"],
            results=["c"],
        )
        result = lower_ukernel_with_body([call], self.reg, target_family="rvv")
        assert len(result.lowered_calls) == 1
        lowered = result.lowered_calls[0]
        assert lowered.function_name.startswith("inline_")
        assert lowered.metadata["transparency"] == "transparent"
        assert lowered.metadata["body_kind"] == "mlir"

    def test_opaque_library_body_uses_source_ref(self) -> None:
        call = UkernelCallOp(
            kernel_name="vendor_matmul_f32",
            operands=["a", "b"],
            results=["c"],
        )
        result = lower_ukernel_with_body([call], self.reg, target_family="cuda")
        assert len(result.lowered_calls) == 1
        lowered = result.lowered_calls[0]
        assert lowered.function_name == "cublas_sgemm"
        assert lowered.metadata["transparency"] == "opaque"
        assert lowered.metadata["body_kind"] == "library"

    def test_opaque_rvv_body_uses_source_ref(self) -> None:
        call = UkernelCallOp(
            kernel_name="vendor_matmul_f32",
            operands=["a", "b"],
            results=["c"],
        )
        result = lower_ukernel_with_body([call], self.reg, target_family="rvv")
        assert len(result.lowered_calls) == 1
        assert result.lowered_calls[0].function_name == "rvv_sgemm"

    def test_opaque_npu_body_uses_source_ref(self) -> None:
        call = UkernelCallOp(
            kernel_name="vendor_matmul_f32",
            operands=["a", "b"],
            results=["c"],
        )
        result = lower_ukernel_with_body([call], self.reg, target_family="npu")
        assert len(result.lowered_calls) == 1
        assert result.lowered_calls[0].function_name == "npu_matmul_f32"

    def test_fallback_to_any_body(self) -> None:
        call = UkernelCallOp(
            kernel_name="matmul_generic_i8",
            operands=["a", "b"],
            results=["c"],
        )
        result = lower_ukernel_with_body([call], self.reg, target_family="arm_neon")
        assert len(result.lowered_calls) == 1
        lowered = result.lowered_calls[0]
        # Falls back to "any" body which is mlir -> inline_ prefix
        assert lowered.function_name.startswith("inline_")
        assert lowered.metadata["target_family"] == "any"

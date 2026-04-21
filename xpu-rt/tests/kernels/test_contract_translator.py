"""Tests for ``compgen.kernels.contract_translator``.

Locks in:
  * select_translator picks the right translator per target taxonomy
  * Triton translator emits a skeleton + autotune grid + prompt context
  * Hexagon translator emits a C header + ISA hints
  * Autocomp translator builds a duck-typed ``Prob`` shape
  * MEGA + PERSISTENT contracts produce documented compatibility notes
  * to_autocomp_prob raises a clear error when autocomp isn't installed
    (skipped when it IS installed — this is just a contract-shape test)
"""

from __future__ import annotations

import pytest

from compgen.kernels.contract_translator import (
    AutocompContractTranslator,
    AutocompProblem,
    HexagonContractTranslator,
    HexagonTranslation,
    KernelContractTranslator,
    TritonContractTranslator,
    TritonTranslation,
    select_translator,
)
from compgen.kernels.contract_v3 import (
    DispatchModel,
    DispatchSpec,
    EventDecl,
    ExecutionEnvelope,
    Granularity,
    HardwareEnvelope,
    InternalEventEdge,
    IOContract,
    KernelArchetype,
    KernelContractV3,
    LayoutKind,
    MemorySpec,
    MemoryTier,
    NumericsSpec,
    OrchestrationSpec,
    ShapeClass,
    StaticAttr,
    SyncSpec,
    TensorIO,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _envelope(name: str) -> HardwareEnvelope:
    return HardwareEnvelope(
        target_name=name, vector_lanes=64,
        scratchpad_bytes=49152, register_bytes=256,
        native_dtypes=("f16", "f32"), peak_bandwidth_gbps=672.0,
        codegen_hints=("hint A", "hint B"),
    )


def _matmul(target: str = "cuda-a100") -> KernelContractV3:
    env = _envelope(target)
    return KernelContractV3(
        op_name="linalg.matmul",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="lhs", shape=ShapeClass(dims=(None, None)),
                         dtype_class=("f16",)),
                TensorIO(name="rhs", shape=ShapeClass(dims=(None, None)),
                         dtype_class=("f16",)),
            ),
            outputs=(TensorIO(name="out", shape=ShapeClass(dims=(None, None)),
                              dtype_class=("f16",)),),
            numerics=NumericsSpec(accumulator_dtype="f32"),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )


def _softmax(target: str = "cuda-a100") -> KernelContractV3:
    env = _envelope(target)
    return KernelContractV3(
        op_name="softmax",
        archetype=KernelArchetype.REDUCE,
        io=IOContract(
            inputs=(TensorIO(name="x", shape=ShapeClass(dims=(None, None)),
                             dtype_class=("f32",)),),
            outputs=(TensorIO(name="y", shape=ShapeClass(dims=(None, None)),
                              dtype_class=("f32",)),),
            attributes=(StaticAttr(name="axis", value=-1),),
        ),
        orchestration=OrchestrationSpec(execution=ExecutionEnvelope(hardware=env)),
    )


# ---------------------------------------------------------------------------
# Factory walks target taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target,expected_translator_name", [
    ("cuda-a100",       "triton"),
    ("rocm-mi250",      "triton"),
    ("test-gpu-simt",   "triton"),
    ("openq_5165rb",    "hexagon_c"),
    ("hexagon-v69",     "hexagon_c"),
    ("totally-unknown", "triton"),     # default fallback
])
def test_select_translator_matches_target_taxonomy(target, expected_translator_name) -> None:
    t = select_translator(target)
    assert t.name == expected_translator_name
    assert isinstance(t, KernelContractTranslator)


# ---------------------------------------------------------------------------
# Triton translator
# ---------------------------------------------------------------------------


def test_triton_translator_supports_cuda_and_rocm() -> None:
    t = TritonContractTranslator()
    assert t.supports(_matmul("cuda-a100"))
    assert t.supports(_matmul("rocm-mi250"))


def test_triton_translator_rejects_hexagon() -> None:
    t = TritonContractTranslator()
    assert not t.supports(_matmul("openq_5165rb"))


def test_triton_translation_carries_skeleton_and_grid() -> None:
    t = TritonContractTranslator()
    out = t.translate(_matmul("cuda-a100"))
    assert isinstance(out, TritonTranslation)
    assert "@triton.jit" in out.kernel_skeleton
    assert "linalg_matmul_kernel" in out.kernel_skeleton
    # Compute-tiled archetype gets a 3-config grid
    assert len(out.autotune_configs) >= 3
    # Prompt context surfaces the IO + hints
    assert "linalg.matmul" in out.prompt_context
    assert "lhs" in out.prompt_context
    assert "hint A" in out.prompt_context     # codegen_hints surfaced


def test_triton_translation_arch_for_rocm_target() -> None:
    t = TritonContractTranslator()
    out = t.translate(_matmul("rocm-mi250"))
    assert out.target_arch == "rocm"


def test_triton_translation_notes_mega_compatibility() -> None:
    """A MEGA contract should emit a compatibility note about persistent
    kernel codegen + body splicing."""
    env = _envelope("cuda-a100")
    sub = KernelContractV3(
        op_name="sub", archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="a", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
                TensorIO(name="b", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
            ),
            outputs=(TensorIO(name="o", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),),
        ),
        orchestration=OrchestrationSpec(
            memory=MemorySpec(
                input_tiers=(MemoryTier.SCRATCHPAD, MemoryTier.SCRATCHPAD),
                output_tiers=(MemoryTier.SCRATCHPAD,),
            ),
        ),
    )
    mega = KernelContractV3(
        op_name="mega.test",
        archetype=KernelArchetype.COMPUTE_TILED,
        io=IOContract(
            inputs=(
                TensorIO(name="a", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
                TensorIO(name="b", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),
            ),
            outputs=(TensorIO(name="o", shape=ShapeClass(dims=(None,)), dtype_class=("f32",)),),
        ),
        granularity=Granularity.MEGA,
        orchestration=OrchestrationSpec(
            execution=ExecutionEnvelope(hardware=env),
            dispatch=DispatchSpec(model=DispatchModel.PERSISTENT),
        ),
        body=(sub,),
    )
    out = TritonContractTranslator().translate(mega)
    assert any("MEGA" in n for n in out.compatibility_notes)
    assert any("PERSISTENT" in n for n in out.compatibility_notes)


# ---------------------------------------------------------------------------
# Hexagon translator
# ---------------------------------------------------------------------------


def test_hexagon_translator_supports_only_hexagon_targets() -> None:
    h = HexagonContractTranslator()
    assert h.supports(_matmul("openq_5165rb"))
    assert not h.supports(_matmul("cuda-a100"))


def test_hexagon_translation_emits_c_header_with_signature() -> None:
    h = HexagonContractTranslator()
    out = h.translate(_matmul("openq_5165rb"))
    assert isinstance(out, HexagonTranslation)
    assert "void linalg_matmul" in out.c_header
    assert "lhs" in out.c_header and "rhs" in out.c_header and "out" in out.c_header


def test_hexagon_translation_carries_isa_hints_for_compute_tiled() -> None:
    h = HexagonContractTranslator()
    out = h.translate(_matmul("openq_5165rb"))
    assert any("vmpyubacc" in hint for hint in out.isa_hints)


# ---------------------------------------------------------------------------
# Autocomp translator
# ---------------------------------------------------------------------------


def test_autocomp_translator_accepts_any_v3_contract() -> None:
    a = AutocompContractTranslator()
    assert a.supports(_matmul("cuda-a100"))
    assert a.supports(_softmax("cuda-a100"))
    assert a.supports(_matmul("openq_5165rb"))


def test_autocomp_translation_produces_prob_shape() -> None:
    a = AutocompContractTranslator()
    out = a.translate(_matmul("cuda-a100"))
    assert isinstance(out, AutocompProblem)
    assert out.prob_type == "matmul"
    assert isinstance(out.prob_id, int) and out.prob_id > 0
    # Context mirrors the v3 fields
    assert "linalg.matmul" in out.context
    assert "cuda-a100" in out.context
    assert "Inputs" in out.context and "Outputs" in out.context


def test_autocomp_archetype_to_prob_type_taxonomy() -> None:
    a = AutocompContractTranslator()
    assert a.translate(_matmul()).prob_type == "matmul"
    assert a.translate(_softmax()).prob_type == "reduce"


def test_autocomp_prob_id_stable_for_same_op_target() -> None:
    """Same (op_family, target) → same prob_id across calls."""
    a = AutocompContractTranslator()
    p1 = a.translate(_matmul("cuda-a100"))
    p2 = a.translate(_matmul("cuda-a100"))
    assert p1.prob_id == p2.prob_id


def test_autocomp_prob_id_differs_for_different_targets() -> None:
    a = AutocompContractTranslator()
    p_cuda = a.translate(_matmul("cuda-a100"))
    p_rocm = a.translate(_matmul("rocm-mi250"))
    assert p_cuda.prob_id != p_rocm.prob_id


def test_autocomp_to_autocomp_prob_lazy_imports() -> None:
    """The conversion to a real autocomp.Prob is lazy. If autocomp is
    installed it returns a Prob; if not it raises ImportError with a
    helpful message. Either branch is correct."""
    a = AutocompContractTranslator()
    try:
        prob = a.to_autocomp_prob(_matmul("cuda-a100"))
    except ImportError as e:
        assert "autocomp" in str(e)
        return
    # If we got here, autocomp IS installed — sanity check the result.
    assert hasattr(prob, "prob_type")
    assert hasattr(prob, "prob_id")

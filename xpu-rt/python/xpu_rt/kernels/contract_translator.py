"""Translate ``KernelContractV3`` into the per-backend / per-search-tool
representation each consumer expects.

One v3 contract describes the kernel canonically; this module is the
adapter layer that converts it into:

  * **TritonContractTranslator** — a Triton-flavoured prompt + an
    autotune config grid suitable for both NVIDIA CUDA and AMD ROCm
    Triton paths. Reuses the ``codegen_hints`` from the contract's
    ``HardwareEnvelope``.
  * **HexagonContractTranslator** — an NPU-dialect prompt with HVX +
    VTCM hints. Pulled from the same hints field; produces a prompt
    flavored for Hexagon C.
  * **AutocompContractTranslator** — a duck-typed object that mirrors
    autocomp's ``Prob`` shape (``prob_type``, ``prob_id``, ``context``,
    ``test_file``). Lets any v3 contract be sent to autocomp's beam
    search for the long-tail 5% of kernels — same v3, same code path,
    no per-target prompt forking.

Design principle: the translator is **lossless on IO + lossy on
orchestration**. Every backend cares about IO shape/dtype/numerics; not
every backend cares about (or can implement) the v3 sync graph. Each
translator's ``compatibility_notes`` field documents what was dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from compgen.kernels.contract_v3 import (
    DispatchModel,
    Granularity,
    KernelArchetype,
    KernelContractV3,
)

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TritonTranslation:
    """What ``TritonContractTranslator.translate()`` returns."""

    kernel_skeleton: str  # Triton stub the codegen fills in
    autotune_configs: tuple[dict[str, Any], ...]  # ``triton.Config``-shaped dicts
    prompt_context: str  # human-readable prompt string for codegen
    target_arch: str  # "cuda" | "rocm"
    compatibility_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HexagonTranslation:
    """Output for Hexagon NPU C codegen."""

    c_header: str  # function signature + buffer decls
    isa_hints: tuple[str, ...]  # vmpyubacc, vrmpybu, etc.
    prompt_context: str
    compatibility_notes: tuple[str, ...] = ()


@dataclass
class AutocompProblem:
    """Duck-typed mirror of ``autocomp.search.Prob``.

    Kept as a plain dataclass (not subclass of autocomp.Prob) so this
    module doesn't hard-import autocomp — agents that don't have the
    autocomp package installed can still build the translation; the
    actual autocomp invocation site does the lazy import + adaptation.
    """

    prob_type: str  # autocomp's op-family taxonomy
    prob_id: int  # generated stable id
    context: str  # prompt-friendly description
    test_file: Path | None = None  # path to a generated test (optional)
    sol_file: Path | None = None
    tests: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class KernelContractTranslator(Protocol):
    """Convert ``KernelContractV3`` → backend-specific representation."""

    @property
    def name(self) -> str: ...

    def supports(self, contract: KernelContractV3) -> bool:
        """Whether this translator can handle this contract."""
        ...

    def translate(self, contract: KernelContractV3) -> Any:
        """Return the backend-specific translation object."""
        ...


# ---------------------------------------------------------------------------
# Triton — covers CUDA + ROCm
# ---------------------------------------------------------------------------


def _arch_for_triton(target_name: str) -> str:
    n = target_name.lower()
    if n.startswith("rocm") or n.startswith("mi"):
        return "rocm"
    return "cuda"  # default


def _triton_autotune_grid(contract: KernelContractV3) -> tuple[dict, ...]:
    """A small, conservative grid keyed off archetype.

    Real generators (the ones the codegen prompt produces) extend this
    with shape-aware overrides.
    """
    arch = contract.archetype
    if arch is KernelArchetype.COMPUTE_TILED:
        return (
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 8, "num_stages": 3},
        )
    if arch is KernelArchetype.REDUCE:
        return (
            {"BLOCK_N": 1024, "num_warps": 4},
            {"BLOCK_N": 2048, "num_warps": 8},
        )
    if arch in (KernelArchetype.POINTWISE, KernelArchetype.ACTIVATION):
        return (
            {"BLOCK": 1024, "num_warps": 4},
            {"BLOCK": 4096, "num_warps": 8},
        )
    return ()


def _kernel_skeleton_for(contract: KernelContractV3) -> str:
    """Tiny Triton skeleton — a starting point the codegen prompt completes."""
    inputs = ", ".join(t.name for t in contract.io.inputs)
    outputs = ", ".join(t.name for t in contract.io.outputs)
    return (
        f"@triton.jit\n"
        f"def {contract.op_name.replace('.', '_')}_kernel(\n"
        f"    # inputs:  {inputs}\n"
        f"    # outputs: {outputs}\n"
        f"    *args, **constexpr,\n"
        f"):\n"
        f"    # TODO: codegen body — see prompt_context for IO + numerics + autotune grid\n"
        f"    pass\n"
    )


@dataclass
class TritonContractTranslator:
    """v3 → Triton skeleton + autotune grid + prompt for CUDA/ROCm Triton."""

    name_str: str = "triton"

    @property
    def name(self) -> str:
        return self.name_str

    def supports(self, contract: KernelContractV3) -> bool:
        env = contract.orchestration.execution
        if env is None:
            return False
        n = env.hardware.target_name.lower()
        return n.startswith(("cuda", "rocm", "mi", "test-gpu", "titan"))

    def translate(self, contract: KernelContractV3) -> TritonTranslation:
        env = contract.orchestration.execution
        target = env.hardware.target_name if env else "unknown"
        arch = _arch_for_triton(target)
        configs = _triton_autotune_grid(contract)

        notes: list[str] = []
        if contract.granularity is Granularity.MEGA:
            notes.append(
                "MEGA contract — Triton emitter must produce a persistent kernel "
                "with the body[] sub-kernels' compute graph spliced inline; "
                "internal_events are inserted as bar.arrive / mbarrier"
            )
        if contract.orchestration.dispatch.model is DispatchModel.PERSISTENT:
            notes.append("PERSISTENT dispatch — launch NUM_SMS CTAs, not per-tile")

        # Render the prompt context
        view = contract.kernel_facing()
        prompt_lines = [
            f"Triton kernel for {contract.op_name!r} ({contract.archetype.value}, {contract.granularity.value})",
            f"Target arch: {arch} ({target})",
            "",
            "IO:",
        ]
        for t in view.io.inputs:
            prompt_lines.append(
                f"  IN  {t.name}: shape={t.shape.dims} dtype_class={t.dtype_class} layout={t.layout.value}"
            )
        for t in view.io.outputs:
            prompt_lines.append(
                f"  OUT {t.name}: shape={t.shape.dims} dtype_class={t.dtype_class} layout={t.layout.value}"
            )
        if view.io.attributes:
            prompt_lines.append("")
            prompt_lines.append("Static attrs:")
            for a in view.io.attributes:
                prompt_lines.append(f"  {a.name}={a.value!r}")
        if view.execution and view.execution.hardware.codegen_hints:
            prompt_lines.append("")
            prompt_lines.append("Hardware hints:")
            for h in view.execution.hardware.codegen_hints:
                prompt_lines.append(f"  - {h}")
        if configs:
            prompt_lines.append("")
            prompt_lines.append(f"Autotune over: {len(configs)} configs")
        if notes:
            prompt_lines.append("")
            prompt_lines.append("Compatibility notes:")
            for n in notes:
                prompt_lines.append(f"  - {n}")

        return TritonTranslation(
            kernel_skeleton=_kernel_skeleton_for(contract),
            autotune_configs=configs,
            prompt_context="\n".join(prompt_lines),
            target_arch=arch,
            compatibility_notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# Hexagon NPU
# ---------------------------------------------------------------------------


@dataclass
class HexagonContractTranslator:
    """v3 → Hexagon C function header + HVX/VTCM hints."""

    name_str: str = "hexagon_c"

    @property
    def name(self) -> str:
        return self.name_str

    def supports(self, contract: KernelContractV3) -> bool:
        env = contract.orchestration.execution
        if env is None:
            return False
        n = env.hardware.target_name.lower()
        return n.startswith(("hexagon", "openq"))

    def translate(self, contract: KernelContractV3) -> HexagonTranslation:
        env = contract.orchestration.execution
        target = env.hardware.target_name if env else "hexagon"

        # ISA hints — pulled from envelope's codegen_hints + archetype.
        hints: list[str] = []
        if env is not None:
            hints.extend(env.hardware.codegen_hints)
        if contract.archetype is KernelArchetype.COMPUTE_TILED:
            hints.append("Use vmpyubacc for int8 matmul accumulation")
        if contract.archetype is KernelArchetype.REDUCE:
            hints.append("HVX vrmpy_acc for partial reductions; finish with scalar fold")

        # C header derived from IO
        params = []
        for t in (*contract.io.inputs, *contract.io.outputs):
            dt = t.dtype_class[0] if t.dtype_class else "f32"
            ctype = {"f16": "float16_t", "bf16": "bfloat16_t", "f32": "float", "i8": "int8_t", "i32": "int32_t"}.get(
                dt, "void"
            )
            params.append(f"{ctype} const* {t.name}")
        c_header = (
            f"// {contract.op_name} ({contract.archetype.value}, target={target})\n"
            f"void {contract.op_name.replace('.', '_')}(\n  " + ",\n  ".join(params) + "\n);\n"
        )

        notes: list[str] = []
        if contract.granularity is Granularity.MEGA:
            notes.append(
                "MEGA — Hexagon C codegen must emit one combined function "
                "calling all body[] sub-kernels in order; events become "
                "scalar flags polled via L2 fence."
            )

        prompt_context = (
            f"Hexagon C kernel for {contract.op_name!r}\n"
            f"Target: {target}\n\n"
            f"Header:\n{c_header}\n"
            "ISA hints:\n  - " + "\n  - ".join(hints)
        )
        return HexagonTranslation(
            c_header=c_header,
            isa_hints=tuple(hints),
            prompt_context=prompt_context,
            compatibility_notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# Autocomp
# ---------------------------------------------------------------------------


# Stable id → (op_family, target) so re-translations of the same contract
# get the same prob_id and autocomp can cache against it. Lives in-process;
# real deployment persists via the kernel store's fingerprint.
_PROB_ID_COUNTER = 1
_PROB_ID_BY_KEY: dict[tuple[str, str], int] = {}


def _stable_prob_id(op_family: str, target: str) -> int:
    global _PROB_ID_COUNTER
    key = (op_family, target)
    if key in _PROB_ID_BY_KEY:
        return _PROB_ID_BY_KEY[key]
    pid = _PROB_ID_COUNTER
    _PROB_ID_BY_KEY[key] = pid
    _PROB_ID_COUNTER += 1
    return pid


@dataclass
class AutocompContractTranslator:
    """v3 → autocomp ``Prob``-shaped object.

    Lets the escalating router send any v3 contract to autocomp without
    per-target prompt forking. The translator builds the same context
    string from the v3 fields that the Triton/Hexagon translators do —
    autocomp's beam search reads it as the problem statement.
    """

    name_str: str = "autocomp"

    @property
    def name(self) -> str:
        return self.name_str

    def supports(self, _contract: KernelContractV3) -> bool:
        # Autocomp consumes any v3 contract — it's the universal escalation.
        return True

    def translate(self, contract: KernelContractV3) -> AutocompProblem:
        # Map archetype → autocomp prob_type taxonomy.
        prob_type = {
            KernelArchetype.COMPUTE_TILED: "matmul",
            KernelArchetype.REDUCE: "reduce",
            KernelArchetype.POINTWISE: "pointwise",
            KernelArchetype.MEMORY: "memory",
            KernelArchetype.ACTIVATION: "activation",
            KernelArchetype.TYPE_CONV_INDEX: "type_conv",
        }.get(contract.archetype, "generic")

        env = contract.orchestration.execution
        target = env.hardware.target_name if env else "unknown"
        prob_id = _stable_prob_id(contract.op_name, target)

        # Build the context string — same surface a human would write
        # for an autocomp problem statement.
        view = contract.kernel_facing()
        context_lines = [
            f"# Kernel: {contract.op_name} ({contract.archetype.value}, {contract.granularity.value})",
            f"# Target: {target}",
            "",
            "## Inputs",
        ]
        for t in view.io.inputs:
            context_lines.append(
                f"  - {t.name}: shape={t.shape.dims} dtype_class={t.dtype_class} layout={t.layout.value}"
            )
        context_lines.append("")
        context_lines.append("## Outputs")
        for t in view.io.outputs:
            context_lines.append(f"  - {t.name}: shape={t.shape.dims} dtype_class={t.dtype_class}")
        if view.io.attributes:
            context_lines.append("")
            context_lines.append("## Static attributes")
            for a in view.io.attributes:
                context_lines.append(f"  - {a.name}={a.value!r}")
        context_lines.append("")
        context_lines.append("## Numerics")
        context_lines.append(
            f"  accumulator_dtype={view.io.numerics.accumulator_dtype} "
            f"fast_math={view.io.numerics.fast_math} "
            f"max_relative_error={view.io.numerics.max_relative_error}"
        )
        if view.execution:
            context_lines.append("")
            context_lines.append("## Hardware envelope")
            hw = view.execution.hardware
            context_lines.append(
                f"  vector_lanes={hw.vector_lanes} scratchpad_bytes={hw.scratchpad_bytes} "
                f"native_dtypes={hw.native_dtypes} peak_bandwidth_gbps={hw.peak_bandwidth_gbps}"
            )
            for h in hw.codegen_hints:
                context_lines.append(f"  - {h}")

        return AutocompProblem(
            prob_type=prob_type,
            prob_id=prob_id,
            context="\n".join(context_lines),
        )

    def to_autocomp_prob(self, contract: KernelContractV3):
        """Build an actual ``autocomp.search.Prob`` instance.

        Lazy-imports autocomp so this module is usable without it.
        Returns the ``Prob`` configured with our generated context.
        Raises ImportError if autocomp isn't installed.
        """
        translation = self.translate(contract)
        try:
            from autocomp.search.prob import Prob  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "autocomp package is not installed; install third_party/autocomp "
                "to use AutocompContractTranslator.to_autocomp_prob"
            ) from exc

        prob = Prob(
            prob_type=translation.prob_type,
            prob_id=translation.prob_id,
            context=translation.context,
        )
        return prob


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def select_translator(target_name: str) -> KernelContractTranslator:
    """Pick the right translator for ``target_name``.

    Mirrors the runtime-adapter + knowledge-store target taxonomy so all
    three layers stay in sync.
    """
    n = (target_name or "").lower()
    if n.startswith(("hexagon", "openq")):
        return HexagonContractTranslator()
    if n.startswith(("cuda", "rocm", "mi", "test-gpu", "titan")):
        return TritonContractTranslator()
    # Fall-through: Triton. Most novel targets the user adds will have a
    # Triton path before they have a hand-written backend.
    return TritonContractTranslator()


__all__ = [
    "AutocompContractTranslator",
    "AutocompProblem",
    "HexagonContractTranslator",
    "HexagonTranslation",
    "KernelContractTranslator",
    "TritonContractTranslator",
    "TritonTranslation",
    "select_translator",
]

"""``VendorDialectAdapter`` — protocol + base class.

An adapter is what a user-space package registers to teach CompGen how
to drive a third-party MLIR toolchain end-to-end. It composes two roles
from the existing infrastructure:

* :class:`compgen.targets.backend.TargetBackendProtocol` — the target-side
  compile pipeline (Payload IR → vendor IR → binary).
* :class:`compgen.kernels.provider.KernelProvider` (optional) — when the
  vendor has no direct linalg/stablehlo ingress and needs per-op kernel
  authoring (e.g. CUDA Tile IR).

Adapters hold the frozen :class:`VendorDialectDescriptor` that produced
them. The descriptor is the source of truth for what the adapter
promises; runtime code should prefer inspecting ``adapter.descriptor``
over re-deriving facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from compgen.extensions.vendor_dialect.descriptor import VendorDialectDescriptor
from compgen.kernels.provider import KernelProvider
from compgen.targets.backend import CompiledArtifact, TargetBackendProtocol

log = structlog.get_logger()


# --------------------------------------------------------------------------- #
# Artifacts that flow through the adapter
# --------------------------------------------------------------------------- #


@dataclass
class LoweringResult:
    """Output of Payload-IR → vendor-IR lowering.

    Attributes:
        vendor_mlir: Text of the lowered MLIR module in the vendor dialect.
        kernels: Per-region kernel sources keyed by region id (filled when
            the adapter has a kernel provider).
        metadata: Free-form diagnostics (pass stats, LLM iterations, etc.).
    """

    vendor_mlir: str = ""
    kernels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Adapter base class (concrete default implementations where sensible)
# --------------------------------------------------------------------------- #


class VendorDialectAdapter:
    """Base class for vendor MLIR dialect integrations.

    User-space packages subclass this, fill in the four abstract hooks,
    and optionally attach a :class:`KernelProvider`. CompGen never
    instantiates this class directly — it consumes adapters produced by
    the scaffold engine or hand-written by users.

    Subclasses MUST override:

    * :meth:`lower_payload` — Payload IR → vendor IR
    * :meth:`emit_artifact` — vendor IR → runnable binary

    Subclasses MAY override:

    * :meth:`validate` — differential validation against golden data
    * :meth:`kernel_provider` — return an attached kernel provider
    """

    descriptor: VendorDialectDescriptor

    def __init__(
        self,
        descriptor: VendorDialectDescriptor,
        *,
        kernel_provider: KernelProvider | None = None,
    ) -> None:
        self.descriptor = descriptor
        self._kernel_provider = kernel_provider

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """Canonical vendor name (matches ``descriptor.name``)."""
        return self.descriptor.name

    @property
    def target(self) -> str:
        """CompGen target this adapter binds to."""
        return self.descriptor.target

    # ------------------------------------------------------------------ #
    # TargetBackendProtocol compatibility
    # ------------------------------------------------------------------ #

    def supports_target(self, target_name: str) -> bool:
        return target_name == self.descriptor.target

    def get_compilation_stages(self) -> list[str]:
        return ["lower_to_vendor", "emit"]

    # ------------------------------------------------------------------ #
    # Abstract hooks
    # ------------------------------------------------------------------ #

    def lower_payload(
        self,
        payload_mlir: str,
        *,
        output_dir: str | Path,
        options: dict[str, Any] | None = None,
    ) -> LoweringResult:
        """Lower CompGen Payload IR (text) to vendor dialect MLIR (text).

        Subclasses implement the vendor-specific transformation. When
        ``kernel_authoring_required`` is set on the descriptor, the
        adapter should delegate per-op kernel synthesis to the attached
        :class:`KernelProvider`.
        """
        raise NotImplementedError(f"{type(self).__name__}.lower_payload must be overridden")

    def emit_artifact(
        self,
        lowering: LoweringResult,
        *,
        output_dir: str | Path,
        options: dict[str, Any] | None = None,
    ) -> CompiledArtifact:
        """Drive the vendor toolchain to produce a runnable artifact."""
        raise NotImplementedError(f"{type(self).__name__}.emit_artifact must be overridden")

    def validate(
        self,
        artifact: CompiledArtifact,
        golden_inputs: dict[str, Any],
        golden_output: Any,
    ) -> bool:
        """Differential test against golden data.

        Default implementation returns True; adapters targeting a real
        device should override.
        """
        del artifact, golden_inputs, golden_output
        return True

    # ------------------------------------------------------------------ #
    # Optional kernel provider surface
    # ------------------------------------------------------------------ #

    def kernel_provider(self) -> KernelProvider | None:
        return self._kernel_provider

    # ------------------------------------------------------------------ #
    # Convenience: full drive from Payload IR to CompiledArtifact
    # ------------------------------------------------------------------ #

    def compile(
        self,
        payload_mlir: str,
        *,
        output_dir: str | Path,
        options: dict[str, Any] | None = None,
    ) -> CompiledArtifact:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        log.info("vendor_adapter.compile.start", vendor=self.name, target=self.target)
        lowering = self.lower_payload(payload_mlir, output_dir=out, options=options)
        artifact = self.emit_artifact(lowering, output_dir=out, options=options)
        log.info(
            "vendor_adapter.compile.done",
            vendor=self.name,
            format=artifact.format,
            target=artifact.target_name,
        )
        return artifact


# Runtime compatibility check against TargetBackendProtocol.
assert hasattr(VendorDialectAdapter, "supports_target")
assert hasattr(VendorDialectAdapter, "get_compilation_stages")
_ = TargetBackendProtocol  # imported so the docstring cross-reference resolves.


__all__ = ["LoweringResult", "VendorDialectAdapter"]

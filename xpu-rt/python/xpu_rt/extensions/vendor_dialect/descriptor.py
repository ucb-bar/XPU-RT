"""Vendor MLIR dialect integration descriptor.

A ``VendorDialectDescriptor`` is the frozen, reviewable spec that the
exploration agent produces from a vendor repo. It is the handoff
between the *explore* phase and the *scaffold* phase: users gate on
it before generating a user-space adapter package.

Descriptors are YAML on disk for human review and Python dataclasses in
memory for type-safe construction. Keep this module light — no LLM or
filesystem work; just schema definitions + (de)serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Nested records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompileEntry:
    """How to invoke the vendor compile pipeline.

    A vendor may expose a CLI tool (e.g. ``cuda-tile-translate``), a
    Python API (e.g. ``cuda_tile._mlir._mlir_libs._cuda_tile``), or both.
    At least one must be populated.
    """

    cli_tools: tuple[str, ...] = ()
    python_module: str = ""
    python_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpEntry:
    """One vendor dialect op surfaced by the scanner or TableGen reader."""

    name: str
    summary: str = ""
    operands: tuple[str, ...] = ()
    results: tuple[str, ...] = ()
    source_file: str = ""


@dataclass(frozen=True)
class LoweringStrategy:
    """How Payload IR reaches the vendor dialect.

    ``mode`` is one of:
      * ``"direct_linalg"``   — vendor accepts linalg (Hexagon)
      * ``"kernel_authoring"`` — Claude/LLM emits dialect ops per op-family (CUDA Tile)
      * ``"torch_mlir"``       — vendor ingests via torch-mlir bridge
      * ``"stablehlo"``        — vendor ingests stablehlo
    """

    mode: str = "direct_linalg"
    op_families: tuple[str, ...] = ()
    template_ops: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class BundlePlan:
    """Concrete pipeline from vendor MLIR text to a runnable artifact."""

    steps: tuple[str, ...] = ()
    output_format: str = "binary"
    runtime_entry: str = ""


@dataclass(frozen=True)
class VerificationPlan:
    """Gates the scaffolded adapter must clear."""

    structural: bool = True
    matmul_diff_test: bool = True
    workload_diff_test: bool = False
    workloads: tuple[str, ...] = ()
    tolerance_rtol: float = 1e-3
    tolerance_atol: float = 1e-3


# --------------------------------------------------------------------------- #
# Top-level descriptor
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VendorDialectDescriptor:
    """Frozen specification for integrating a third-party MLIR dialect.

    Produced by :mod:`compgen.agent.vendor_integration.explore` and
    consumed by :mod:`compgen.extensions.vendor_dialect.scaffold`.

    Attributes:
        name: Canonical vendor name (e.g. ``"cuda_tile"``, ``"hexagon"``).
        package_name: Python package name for the scaffolded user-space
            adapter (e.g. ``"compgen_cuda_tile"``).
        repo_path: Absolute path to the cloned vendor repository.
        target: CompGen target name this adapter binds to
            (e.g. ``"nvidia-h100"``).
        input_ir: Which IR(s) the vendor toolchain accepts as input.
        output_format: What the final compile artifact is
            (``"cubin"``, ``"hexagon_elf"``, ``"llvm_ir"``, ...).
        compile_entry: CLI / Python entry points to drive compilation.
        td_files: Relative paths to detected TableGen (.td) files.
        op_registry: Parsed list of vendor dialect ops.
        lowering: How Payload IR is brought to the vendor dialect.
        bundle: The vendor-side pipeline to produce the artifact.
        verification: Verification gates this adapter must clear.
        kernel_authoring_required: Whether the adapter needs an LLM-backed
            kernel provider (no direct linalg path from CompGen's Payload IR).
        dependencies: External tools / SDKs the user must install.
        license: Detected upstream license identifier (SPDX-like).
        extras: Free-form provenance (scanner version, explore-agent
            notes, etc.).
    """

    name: str
    package_name: str
    repo_path: str
    target: str
    input_ir: tuple[str, ...] = ()
    output_format: str = ""
    compile_entry: CompileEntry = field(default_factory=CompileEntry)
    td_files: tuple[str, ...] = ()
    op_registry: tuple[OpEntry, ...] = ()
    lowering: LoweringStrategy = field(default_factory=LoweringStrategy)
    bundle: BundlePlan = field(default_factory=BundlePlan)
    verification: VerificationPlan = field(default_factory=VerificationPlan)
    kernel_authoring_required: bool = False
    dependencies: tuple[str, ...] = ()
    license: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, safe to ``yaml.safe_dump``."""
        return _dataclass_to_plain(self)

    def to_yaml(self) -> str:
        """YAML dump ready for ``descriptor.yaml``."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def write(self, path: str | Path) -> Path:
        """Write the descriptor to ``path`` and return the absolute path."""
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml())
        return p

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VendorDialectDescriptor:
        """Construct from plain dict (e.g. from YAML)."""
        d = dict(data)
        if "compile_entry" in d and d["compile_entry"] is not None:
            d["compile_entry"] = CompileEntry(
                cli_tools=tuple(d["compile_entry"].get("cli_tools", ())),
                python_module=str(d["compile_entry"].get("python_module", "")),
                python_symbols=tuple(d["compile_entry"].get("python_symbols", ())),
            )
        if "op_registry" in d and d["op_registry"] is not None:
            d["op_registry"] = tuple(
                OpEntry(
                    name=str(op.get("name", "")),
                    summary=str(op.get("summary", "")),
                    operands=tuple(op.get("operands", ())),
                    results=tuple(op.get("results", ())),
                    source_file=str(op.get("source_file", "")),
                )
                for op in d["op_registry"]
            )
        if "lowering" in d and d["lowering"] is not None:
            d["lowering"] = LoweringStrategy(
                mode=str(d["lowering"].get("mode", "direct_linalg")),
                op_families=tuple(d["lowering"].get("op_families", ())),
                template_ops=tuple(d["lowering"].get("template_ops", ())),
                notes=str(d["lowering"].get("notes", "")),
            )
        if "bundle" in d and d["bundle"] is not None:
            d["bundle"] = BundlePlan(
                steps=tuple(d["bundle"].get("steps", ())),
                output_format=str(d["bundle"].get("output_format", "binary")),
                runtime_entry=str(d["bundle"].get("runtime_entry", "")),
            )
        if "verification" in d and d["verification"] is not None:
            d["verification"] = VerificationPlan(
                structural=bool(d["verification"].get("structural", True)),
                matmul_diff_test=bool(d["verification"].get("matmul_diff_test", True)),
                workload_diff_test=bool(d["verification"].get("workload_diff_test", False)),
                workloads=tuple(d["verification"].get("workloads", ())),
                tolerance_rtol=float(d["verification"].get("tolerance_rtol", 1e-3)),
                tolerance_atol=float(d["verification"].get("tolerance_atol", 1e-3)),
            )
        # Coerce plain tuples / primitives.
        for k in ("input_ir", "td_files", "dependencies"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return cls(**d)

    @classmethod
    def from_yaml(cls, text: str) -> VendorDialectDescriptor:
        """Parse from a YAML string."""
        data = yaml.safe_load(text) or {}
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> VendorDialectDescriptor:
        """Load from a YAML file on disk."""
        return cls.from_yaml(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _dataclass_to_plain(obj: Any) -> Any:
    """Convert a (possibly nested) frozen dataclass tree into YAML-safe dicts.

    ``dataclasses.asdict`` already does most of the work, but we convert
    the top-level tuples to lists so that PyYAML emits a clean flow-free
    style, and we drop empty-container keys on the inner records to keep
    the YAML compact without altering semantics.
    """
    d = asdict(obj)
    return _normalize(d)


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_normalize(v) for v in value]
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


__all__ = [
    "BundlePlan",
    "CompileEntry",
    "LoweringStrategy",
    "OpEntry",
    "VendorDialectDescriptor",
    "VerificationPlan",
]

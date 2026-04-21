"""Frontend capture paths for CompGen.

Captures a PyTorch nn.Module into either a ``torch.export`` ExportedProgram or
TorchDynamo FX graph partitions and builds the canonical capture artifact for
the CompGen frontend boundary.

Invariants:
    - ``capture_model()`` preserves the legacy behaviour and returns the raw
      ``ExportedProgram`` for existing call sites.
    - ``capture_frontend_artifact()`` is the canonical frontend boundary:
      export, optional export decompositions, diagnostics, and unsupported-op
      preparation are all recorded before CompGen takes ownership of the IR.
    - Dynamic shapes, guards, and decomposition provenance are serialized into
      stable dataclasses.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from compgen.capture.dynamo_baseline import DynamoReport, collect_diagnostics
from compgen.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from compgen.capture.unsupported import UnsupportedOpResolution, recover_unsupported_operators
from compgen.capture.unsupported.introspect import runtime_versions
from compgen.ir.payload.decompositions import DECOMPOSITION_TABLE
from compgen.models import CaptureMode


@dataclass(frozen=True)
class ExportValidation:
    """Result of validating an exported program.

    Attributes:
        valid: Whether the export passed all checks.
        round_trip_ok: Whether re-export produces the same graph.
        num_ops: Number of ops in the exported graph.
        graph_breaks: List of graph break descriptions (should be empty).
        warnings: Non-fatal validation warnings.
    """

    valid: bool
    round_trip_ok: bool
    num_ops: int
    graph_breaks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RangeConstraint:
    """Serializable representation of an export dynamic-shape constraint."""

    symbol: str
    minimum: int | None = None
    maximum: int | None = None


@dataclass
class CaptureArtifact:
    """Canonical frontend boundary for CompGen.

    Attributes:
        original_exported_program: Raw ``torch.export.export`` output, if available.
        exported_program: Prepared export after ``run_decompositions()``, if available.
        validation: Validation for the prepared export.
        diagnostics: TorchDynamo diagnostics collected before export.
        decomposition_targets: Export decomposition table keys applied.
        unsupported_resolutions: Unsupported-op recovery results.
        synthesized_payload_translations: Dynamic Payload translations keyed by target.
        explicit_blackboxes: Targets explicitly approved for opaque lowering.
        capture_mode: Which frontend path produced the artifact.
        graphs: Captured FX graph partitions for Dynamo capture.
    """

    original_exported_program: Any | None
    exported_program: Any | None
    validation: ExportValidation
    diagnostics: DynamoReport
    graph_signature: str
    module_call_graph: tuple[str, ...]
    capture_mode: CaptureMode = CaptureMode.TORCH_EXPORT
    graphs: tuple[torch.fx.GraphModule, ...] = ()
    graph_break_count: int = 0
    analysis_success: bool = False
    range_constraints: tuple[RangeConstraint, ...] = ()
    decomposition_targets: tuple[str, ...] = ()
    quantization_config: QuantizationConfig | None = None
    runtime_versions: dict[str, str] = field(default_factory=dict)
    unsupported_resolutions: list[UnsupportedOpResolution] = field(default_factory=list)
    synthesized_payload_translations: dict[str, Any] = field(default_factory=dict)
    explicit_blackboxes: tuple[str, ...] = ()

    def strict_import_options(self) -> dict[str, Any]:
        """Options used by strict Payload import on the prepared graph."""

        return {
            "allow_opaque_fallback": False,
            "explicit_blackboxes": set(self.explicit_blackboxes),
            "dynamic_decompositions": dict(self.synthesized_payload_translations),
        }

    @property
    def graph_count(self) -> int:
        """Number of captured graphs represented by this artifact."""

        if self.graphs:
            return len(self.graphs)
        return 1 if self.exported_program is not None else 0


def _load_model_from_path(model_path: str | Path) -> Any:
    """Load an nn.Module from a Python file.

    Expects the file to define a function ``get_model_and_inputs()``
    or a class with a known name.
    """
    path = Path(model_path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if hasattr(module, "get_model_and_inputs"):
        model, _ = module.get_model_and_inputs()
        return model

    # Try common class names
    for name in ("SimpleMLP", "TransformerBlock", "Model"):
        if hasattr(module, name):
            cls = getattr(module, name)
            return cls()

    raise ImportError(f"No model found in {path}. Define get_model_and_inputs() or a known model class.")


def _resolve_model(model_path: str | Path | nn.Module) -> nn.Module:
    if isinstance(model_path, nn.Module):
        return model_path
    return _load_model_from_path(model_path)


def _count_call_ops(graph: Any) -> int:
    """Count call_function nodes in an FX graph (the actual operations)."""
    return sum(1 for node in graph.nodes if node.op == "call_function")


def _count_graph_partitions(graphs: tuple[torch.fx.GraphModule, ...]) -> int:
    return sum(_count_call_ops(graph.graph) for graph in graphs)


def _serialize_range_constraints(exported_program: Any) -> tuple[RangeConstraint, ...]:
    constraints = []
    raw_constraints = getattr(exported_program, "range_constraints", {}) or {}
    for symbol, value in raw_constraints.items():
        minimum = getattr(value, "lower", None)
        maximum = getattr(value, "upper", None)
        if minimum is not None:
            minimum = int(minimum)
        if maximum is not None:
            maximum = int(maximum)
        constraints.append(RangeConstraint(symbol=str(symbol), minimum=minimum, maximum=maximum))
    return tuple(constraints)


def _build_decomposition_table(
    *,
    run_default_decompositions: bool,
    export_decomposition_table: dict[Any, Any] | None,
) -> tuple[dict[Any, Any], tuple[str, ...]]:
    table: dict[Any, Any] = {}
    if run_default_decompositions:
        default_factory = getattr(torch.export, "default_decompositions", None)
        if callable(default_factory):
            table.update(default_factory())
    if export_decomposition_table:
        table.update(export_decomposition_table)
    targets = tuple(sorted(str(key) for key in table.keys()))
    return table, targets


def _prepare_exported_program(
    exported_program: Any,
    *,
    run_default_decompositions: bool,
    export_decomposition_table: dict[Any, Any] | None,
) -> tuple[Any, tuple[str, ...]]:
    table, targets = _build_decomposition_table(
        run_default_decompositions=run_default_decompositions,
        export_decomposition_table=export_decomposition_table,
    )
    if not table:
        return exported_program, targets
    return exported_program.run_decompositions(table), targets


def capture_model(
    model_path: str | Path | nn.Module,
    sample_inputs: Any,
    dynamic_shapes: dict[str, Any] | None = None,
) -> Any:
    """Capture a PyTorch model via torch.export.

    Args:
        model_path: Path to a Python file, or a live nn.Module.
        sample_inputs: Sample input tensors (tuple).
        dynamic_shapes: Optional dynamic shape specifications.

    Returns:
        A torch.export.ExportedProgram.
    """
    model = _resolve_model(model_path)

    model.eval()

    # Ensure model and inputs are on CPU to avoid FakeTensor device propagation
    # errors during torch.export tracing.
    model = model.to("cpu")
    if isinstance(sample_inputs, (tuple, list)):
        sample_inputs = tuple(t.to("cpu") if isinstance(t, torch.Tensor) else t for t in sample_inputs)

    kwargs: dict[str, Any] = {}
    if dynamic_shapes is not None:
        kwargs["dynamic_shapes"] = dynamic_shapes

    try:
        exported = torch.export.export(model, sample_inputs, **kwargs)
    except Exception as first_err:
        msg = str(first_err)
        # Retry under non-strict for known recoverable failure modes:
        #  * FakeTensor / device-propagation issues
        #  * GuardOnDataDependentSymNode — strict mode rejects size-like
        #    symbolic shapes that show up in models doing dynamic image
        #    tile counts (e.g. SmolVLA / SmolVLM); non-strict mode
        #    accepts them as runtime guards.
        recoverable = (
            "FakeTensor" in msg
            or "device" in msg.lower()
            or "GuardOnDataDependentSymNode" in msg
            or "data-dependent" in msg
        )
        if recoverable:
            kwargs["strict"] = False
            exported = torch.export.export(model, sample_inputs, **kwargs)
        else:
            raise
    return exported


def capture_frontend_artifact(
    model_path: str | Path | nn.Module,
    sample_inputs: Any,
    dynamic_shapes: dict[str, Any] | None = None,
    *,
    quantization_config: QuantizationConfig | None = None,
    run_default_decompositions: bool = True,
    export_decomposition_table: dict[Any, Any] | None = None,
) -> CaptureArtifact:
    """Build the canonical export-boundary artifact for CompGen.

    ``capture_model()`` returns the raw exported program for backward
    compatibility. This function is the strict frontend path used by
    ``compile_model()`` and by agentic analysis.
    """

    model = _resolve_model(model_path)

    model.eval()

    active_model = model
    if quantization_config is not None:
        try:
            active_model = apply_quantization(model, quantization_config)
        except Exception:
            active_model = model

    diagnostics = collect_diagnostics(active_model, sample_inputs)
    original_exported = capture_model(active_model, sample_inputs, dynamic_shapes=dynamic_shapes)
    prepared_exported, decomposition_targets = _prepare_exported_program(
        original_exported,
        run_default_decompositions=run_default_decompositions,
        export_decomposition_table=export_decomposition_table,
    )
    validation = validate_export(prepared_exported)
    versions = runtime_versions()

    unsupported_resolutions = recover_unsupported_operators(
        prepared_exported,
        supported_targets=set(DECOMPOSITION_TABLE.keys()),
        runtime_versions=versions,
    )
    explicit_blackboxes = tuple(
        sorted(resolution.target for resolution in unsupported_resolutions if resolution.approved_blackbox)
    )
    synthesized_payload_translations = {
        resolution.target: resolution.translation.translator
        for resolution in unsupported_resolutions
        if resolution.translation is not None
    }

    return CaptureArtifact(
        original_exported_program=original_exported,
        exported_program=prepared_exported,
        validation=validation,
        diagnostics=diagnostics,
        graph_signature=str(getattr(prepared_exported, "graph_signature", "")),
        module_call_graph=tuple(str(item) for item in (getattr(prepared_exported, "module_call_graph", None) or ())),
        capture_mode=CaptureMode.TORCH_EXPORT,
        graphs=(),
        graph_break_count=len(diagnostics.graph_breaks),
        analysis_success=validation.valid,
        range_constraints=_serialize_range_constraints(prepared_exported),
        decomposition_targets=decomposition_targets,
        quantization_config=quantization_config,
        runtime_versions=versions,
        unsupported_resolutions=unsupported_resolutions,
        synthesized_payload_translations=synthesized_payload_translations,
        explicit_blackboxes=explicit_blackboxes,
    )


def capture_dynamo_partitions(
    model_path: str | Path | nn.Module,
    sample_inputs: Any,
    *,
    fullgraph: bool = False,
) -> CaptureArtifact:
    """Capture FX graph partitions via TorchDynamo/torch.compile."""

    model = _resolve_model(model_path)
    model.eval()
    diagnostics = collect_diagnostics(model, sample_inputs)
    versions = runtime_versions()

    import torch._dynamo as dynamo

    dynamo.reset()
    captured: list[torch.fx.GraphModule] = []

    def capture_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Any:
        captured.append(gm)
        return gm.forward

    compiled = torch.compile(model, backend=capture_backend, fullgraph=fullgraph)
    with torch.no_grad():
        compiled(*sample_inputs)

    graphs = tuple(captured)
    total_ops = _count_graph_partitions(graphs)
    validation = ExportValidation(
        valid=bool(graphs),
        round_trip_ok=bool(graphs),
        num_ops=total_ops,
        graph_breaks=[reason for _, reason in diagnostics.graph_breaks],
        warnings=[] if graphs else ["TorchDynamo produced no graph partitions"],
    )

    return CaptureArtifact(
        original_exported_program=None,
        exported_program=None,
        validation=validation,
        diagnostics=diagnostics,
        graph_signature="",
        module_call_graph=tuple(),
        capture_mode=CaptureMode.TORCH_DYNAMO_PARTITIONED,
        graphs=graphs,
        graph_break_count=len(diagnostics.graph_breaks),
        analysis_success=bool(graphs),
        range_constraints=(),
        decomposition_targets=(),
        quantization_config=None,
        runtime_versions=versions,
        unsupported_resolutions=[],
        synthesized_payload_translations={},
        explicit_blackboxes=(),
    )


def capture_frontend(
    model_path: str | Path | nn.Module,
    sample_inputs: Any,
    *,
    dynamic_shapes: dict[str, Any] | None = None,
    prefer_dynamo: bool = False,
    fallback_to_dynamo: bool = False,
) -> CaptureArtifact:
    """Capture a model using the preferred frontend mode."""

    if prefer_dynamo:
        return capture_dynamo_partitions(model_path, sample_inputs)
    try:
        return capture_frontend_artifact(model_path, sample_inputs, dynamic_shapes=dynamic_shapes)
    except Exception:
        if not fallback_to_dynamo:
            raise
    return capture_dynamo_partitions(model_path, sample_inputs)


def validate_export(exported_program: Any) -> ExportValidation:
    """Validate an exported program.

    Counts ops, checks for graph breaks, and verifies the graph is well-formed.
    """
    graph = exported_program.graph
    num_ops = _count_call_ops(graph)
    warnings: list[str] = []

    # Check for unsupported patterns
    graph_breaks: list[str] = []
    for node in graph.nodes:
        if node.op == "call_function":
            target_name = str(node.target)
            if "graph_break" in target_name.lower():
                graph_breaks.append(f"Graph break at {node.name}: {target_name}")

    # Simple round-trip check: verify the graph has expected structure
    has_output = any(n.op == "output" for n in graph.nodes)
    has_placeholder = any(n.op == "placeholder" for n in graph.nodes)
    round_trip_ok = has_output and has_placeholder

    if not has_output:
        warnings.append("Graph has no output node")
    if num_ops == 0:
        warnings.append("Graph has no call_function ops")

    valid = round_trip_ok and len(graph_breaks) == 0

    return ExportValidation(
        valid=valid,
        round_trip_ok=round_trip_ok,
        num_ops=num_ops,
        graph_breaks=graph_breaks,
        warnings=warnings,
    )


__all__ = [
    "CaptureArtifact",
    "ExportValidation",
    "RangeConstraint",
    "capture_dynamo_partitions",
    "capture_frontend",
    "capture_frontend_artifact",
    "capture_model",
    "validate_export",
]

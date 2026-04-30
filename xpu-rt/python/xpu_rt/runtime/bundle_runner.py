"""Bundle loader + runner — rehydrate a compile bundle and execute it.

Closes the "we emit bundles but nothing reads them" gap identified in
the runtime-HAL plan.  Once a bundle has been written by
:class:`~compgen.stages.bundle.stage.BundleStage`, this module can:

1. :func:`load_bundle` — parse ``manifest.json`` + ``payload.mlir``
   back into an in-memory bundle. Optionally rehydrates
   ``exported_program.pt2`` and ``golden_inputs.pt`` / ``golden_outputs.pt``
   if they are present (the bundle stage does not currently emit those;
   see Phase A item 6 of the runtime plan).
2. :func:`run_bundle` — execute the rehydrated payload IR through
   :func:`compgen.runtime.cpu_executor.execute`. This is the path that
   makes a **promoted recipe** re-runnable from a later session
   without going through the LLM or the agent loop.

The runner is CPU-only today. When Phase C's CUDA driver lands, a
second runner variant will route through ``heterogeneous_executor.run``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import torch
from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.math import Math
from xdsl.dialects.tensor import Tensor
from xdsl.parser import Parser

log = structlog.get_logger(__name__)


def _build_payload_context() -> Context:
    """Build an ``xdsl.Context`` configured for payload-IR parsing.

    Mirrors the pattern in
    :func:`compgen.capture.torch_mlir_bridge.bridge_fx_graph` — loads
    the standard xDSL dialects plus CompGen's own dialects when
    available. ``allow_unregistered=True`` so ``compgen.*`` attributes
    stamped by stage plugins don't block parsing.
    """
    ctx = Context(allow_unregistered=True)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Func)
    ctx.load_dialect(Linalg)
    ctx.load_dialect(Math)
    ctx.load_dialect(Tensor)

    # CompGen-owned dialects: optional to keep the runner usable in
    # minimal installs. Failures are silent because the payload may
    # not use these dialects at all.
    # Each import/load is independent so a missing dialect doesn't
    # hide the others.
    for mod_name, attr_name in (
        ("compgen.ir.linalg_ext", "LinalgExt"),
        ("compgen.ir.quant", "Quant"),
        ("compgen.ir.tensor_ext", "TensorExt"),
        ("compgen.ir.collective", "Collective"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[attr_name])
            ctx.load_dialect(getattr(mod, attr_name))
        except Exception:  # noqa: BLE001
            # Dialect optional; payload may not use it.
            continue
    return ctx


# Filenames canonicalised to the bundle manifest / contract.
_PAYLOAD_FILENAME = "payload.mlir"
_MANIFEST_FILENAME = "manifest.json"
_EXPORTED_PROGRAM_FILENAME = "exported_program.pt2"
_GOLDEN_INPUTS_FILENAME = "golden_inputs.pt"
_GOLDEN_OUTPUTS_FILENAME = "golden_outputs.pt"
_EXECUTION_PLAN_FILENAME = "execution_plan.yaml"
_MEMORY_PLAN_FILENAME = "memory_plan.yaml"
_GAP_ANALYSIS_FILENAME = "gap_analysis.json"
_KERNEL_CONTRACTS_DIR = "kernel_contracts"


@dataclass
class LoadedBundle:
    """A rehydrated compile bundle.

    Everything the runner needs to execute. Fields beyond
    ``payload_module`` + ``manifest`` are optional — bundles emitted by
    older sessions may be missing the golden I/O or the exported
    program.
    """

    bundle_dir: Path
    manifest: dict[str, Any]
    payload_module: ModuleOp
    #: Raw MLIR text (for re-hashing / re-serialisation).
    payload_text: str
    #: ``torch.export.ExportedProgram`` rehydrated from
    #: ``exported_program.pt2`` if present. Required by
    #: :func:`run_bundle` — ``cpu_executor.execute`` needs the graph
    #: signature and state dict.
    exported_program: Any | None = None
    #: Reference inputs, populated from ``golden_inputs.pt`` if present.
    golden_inputs: tuple[torch.Tensor, ...] | None = None
    #: Reference output, populated from ``golden_outputs.pt`` if present.
    golden_output: torch.Tensor | None = None
    #: Execution plan parsed from ``execution_plan.yaml`` if present
    #: (as a plain dict matching :meth:`ExecutionPlan.to_dict`).
    execution_plan: dict[str, Any] | None = None
    #: Per-device memory breakdown from ``memory_plan.yaml`` if present.
    memory_plan: list[dict[str, Any]] | None = None
    #: Gap analysis (cluster FLOPs/bytes/opportunities) from
    #: ``gap_analysis.json`` if present.
    gap_analysis: dict[str, Any] | None = None
    #: Kernel contracts parsed from ``kernel_contracts/*.yaml`` — one
    #: dict per op that needs a kernel. Keyed by filename stem.
    kernel_contracts: dict[str, dict[str, Any]] | None = None
    #: Diagnostic counters.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def model_hash(self) -> str:
        return str(self.manifest.get("model_hash", ""))

    @property
    def target_profile_name(self) -> str:
        return str(self.manifest.get("target_profile", ""))


def load_bundle(bundle_dir: Path | str) -> LoadedBundle:
    """Load a bundle from disk.

    Requires at minimum ``manifest.json`` + ``payload.mlir``.
    Other artefacts are loaded best-effort; their absence is noted in
    ``LoadedBundle.diagnostics`` rather than raised.

    Args:
        bundle_dir: Directory written by
            :class:`~compgen.stages.bundle.stage.BundleStage`.

    Returns:
        A :class:`LoadedBundle` ready for :func:`run_bundle`.

    Raises:
        FileNotFoundError: If ``manifest.json`` or ``payload.mlir`` is
            missing — bundles cannot exist without these.
    """
    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {bundle_dir}")

    manifest_path = bundle_dir / _MANIFEST_FILENAME
    payload_path = bundle_dir / _PAYLOAD_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    if not payload_path.is_file():
        raise FileNotFoundError(f"missing payload: {payload_path}")

    manifest = json.loads(manifest_path.read_text())
    payload_text = payload_path.read_text()

    # Parse payload.mlir back into a ModuleOp using the canonical
    # payload-IR context (standard xDSL dialects + CompGen extensions).
    ctx = _build_payload_context()
    payload_module = Parser(ctx, payload_text).parse_module()

    diagnostics: dict[str, Any] = {"present": [_MANIFEST_FILENAME, _PAYLOAD_FILENAME]}

    # Rehydrate optional artefacts.
    exported_program: Any | None = None
    ep_path = bundle_dir / _EXPORTED_PROGRAM_FILENAME
    if ep_path.is_file():
        try:
            exported_program = torch.export.load(str(ep_path))
            diagnostics["present"].append(_EXPORTED_PROGRAM_FILENAME)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "bundle_runner.exported_program_load_failed",
                path=str(ep_path),
                error=str(exc),
            )
            diagnostics["exported_program_error"] = str(exc)

    golden_inputs: tuple[torch.Tensor, ...] | None = None
    gi_path = bundle_dir / _GOLDEN_INPUTS_FILENAME
    if gi_path.is_file():
        try:
            raw = torch.load(gi_path, weights_only=False)
            golden_inputs = tuple(raw) if isinstance(raw, (list, tuple)) else (raw,)
            diagnostics["present"].append(_GOLDEN_INPUTS_FILENAME)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "bundle_runner.golden_inputs_load_failed",
                path=str(gi_path),
                error=str(exc),
            )
            diagnostics["golden_inputs_error"] = str(exc)

    golden_output: torch.Tensor | None = None
    go_path = bundle_dir / _GOLDEN_OUTPUTS_FILENAME
    if go_path.is_file():
        try:
            raw = torch.load(go_path, weights_only=False)
            # Accept either a tensor directly or a single-element tuple.
            if isinstance(raw, torch.Tensor):
                golden_output = raw
            elif isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], torch.Tensor):
                golden_output = raw[0]
            diagnostics["present"].append(_GOLDEN_OUTPUTS_FILENAME)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "bundle_runner.golden_outputs_load_failed",
                path=str(go_path),
                error=str(exc),
            )
            diagnostics["golden_outputs_error"] = str(exc)

    execution_plan_data: dict[str, Any] | None = None
    memory_plan_data: list[dict[str, Any]] | None = None
    ep_yaml = bundle_dir / _EXECUTION_PLAN_FILENAME
    mp_yaml = bundle_dir / _MEMORY_PLAN_FILENAME
    if ep_yaml.is_file() or mp_yaml.is_file():
        try:
            import yaml  # type: ignore[import-untyped]

            if ep_yaml.is_file():
                loaded_plan = yaml.safe_load(ep_yaml.read_text())
                if isinstance(loaded_plan, dict):
                    execution_plan_data = loaded_plan
                    diagnostics["present"].append(_EXECUTION_PLAN_FILENAME)
            if mp_yaml.is_file():
                loaded_mem = yaml.safe_load(mp_yaml.read_text())
                if isinstance(loaded_mem, list):
                    memory_plan_data = loaded_mem
                    diagnostics["present"].append(_MEMORY_PLAN_FILENAME)
        except Exception as exc:  # noqa: BLE001
            log.warning("bundle_runner.plan_yaml_load_failed", error=str(exc))
            diagnostics["plan_yaml_error"] = str(exc)

    gap_analysis_data: dict[str, Any] | None = None
    ga_path = bundle_dir / _GAP_ANALYSIS_FILENAME
    if ga_path.is_file():
        try:
            loaded_gap = json.loads(ga_path.read_text())
            if isinstance(loaded_gap, dict):
                gap_analysis_data = loaded_gap
                diagnostics["present"].append(_GAP_ANALYSIS_FILENAME)
        except Exception as exc:  # noqa: BLE001
            log.warning("bundle_runner.gap_analysis_load_failed", error=str(exc))
            diagnostics["gap_analysis_error"] = str(exc)

    kernel_contracts_data: dict[str, dict[str, Any]] | None = None
    kc_dir = bundle_dir / _KERNEL_CONTRACTS_DIR
    if kc_dir.is_dir():
        try:
            import yaml  # type: ignore[import-untyped]

            loaded_contracts: dict[str, dict[str, Any]] = {}
            for yaml_path in sorted(kc_dir.glob("*.yaml")):
                parsed = yaml.safe_load(yaml_path.read_text())
                if isinstance(parsed, dict):
                    loaded_contracts[yaml_path.stem] = parsed
            if loaded_contracts:
                kernel_contracts_data = loaded_contracts
                diagnostics["present"].append(_KERNEL_CONTRACTS_DIR + "/")
        except Exception as exc:  # noqa: BLE001
            log.warning("bundle_runner.kernel_contracts_load_failed", error=str(exc))
            diagnostics["kernel_contracts_error"] = str(exc)

    return LoadedBundle(
        bundle_dir=bundle_dir,
        manifest=manifest,
        payload_module=payload_module,
        payload_text=payload_text,
        exported_program=exported_program,
        golden_inputs=golden_inputs,
        golden_output=golden_output,
        execution_plan=execution_plan_data,
        memory_plan=memory_plan_data,
        gap_analysis=gap_analysis_data,
        kernel_contracts=kernel_contracts_data,
        diagnostics=diagnostics,
    )


def run_bundle(
    bundle: LoadedBundle,
    inputs: tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    """Execute a loaded bundle through the CPU executor.

    Args:
        bundle: A :class:`LoadedBundle`.
        inputs: Input tensors. If ``None``, falls back to
            ``bundle.golden_inputs``.

    Returns:
        The output tensor produced by
        :func:`compgen.runtime.cpu_executor.execute`.

    Raises:
        ValueError: If the bundle has no ``exported_program`` (required
            by the CPU executor for the graph signature) or no inputs
            are available.
    """
    if bundle.exported_program is None:
        raise ValueError(
            "run_bundle requires bundle.exported_program; "
            "this bundle has no exported_program.pt2 artefact. "
            "Either regenerate the bundle with full artefact emission "
            "or pass the ExportedProgram explicitly."
        )

    effective_inputs = inputs if inputs is not None else bundle.golden_inputs
    if effective_inputs is None:
        raise ValueError("run_bundle: no inputs provided and bundle has no golden_inputs.pt")

    # Import lazily for the same reason as local_executor.
    from compgen.runtime.cpu_executor import execute as _compgen_execute

    log.info(
        "bundle_runner.run",
        bundle_dir=str(bundle.bundle_dir),
        target=bundle.target_profile_name,
        model_hash=bundle.model_hash,
        num_inputs=len(effective_inputs),
    )
    return _compgen_execute(bundle.payload_module, bundle.exported_program, effective_inputs)


__all__ = ["LoadedBundle", "load_bundle", "run_bundle"]

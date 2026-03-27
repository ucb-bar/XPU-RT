"""Bundle stage — package all artifacts into a deployable bundle.

Final stage in every target's dialect stack.  Serializes the IR,
execution plan, kernels, and metadata into the Artifact Contract
format from CLAUDE.md.

Shared passes:
  - Serialize IR to payload.mlir
  - Generate manifest.json

Target plugin generates:
  - Target-specific bundle additions (HAL driver config, firmware, etc.)
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer

from compgen.stages.base import CompilationStage, IRInvariant, StageContract
from compgen.targets.schema import TargetProfile


class BundleStage(CompilationStage):
    """Artifact bundling stage.

    Packages compilation artifacts into a deployable bundle.
    The module is passed through unchanged; the bundle is an artifact.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        super().__init__()
        self._output_dir = output_dir or Path(tempfile.mkdtemp(prefix="compgen_bundle_"))

    @property
    def name(self) -> str:
        return "bundle"

    @property
    def description(self) -> str:
        return "Package all artifacts into a deployable bundle"

    def input_contract(self) -> StageContract:
        return StageContract(
            stage_name="bundle",
            preconditions=[
                IRInvariant(
                    name="valid_ir",
                    description="Module must be valid xDSL IR",
                    custom_check=lambda m: _try_verify(m),
                ),
            ],
        )

    def output_contract(self) -> StageContract:
        # Bundle stage doesn't modify IR, so output contract is minimal
        return StageContract(stage_name="bundle")

    def shared_passes(self, module: ModuleOp, target: TargetProfile) -> ModuleOp:
        """Serialize IR and create manifest (stored as artifacts, not IR changes)."""
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Write payload.mlir
        buf = io.StringIO()
        Printer(stream=buf).print_op(module)
        payload_path = self._output_dir / "payload.mlir"
        payload_path.write_text(buf.getvalue())

        # Write manifest.json
        import hashlib
        from datetime import UTC, datetime

        model_hash = hashlib.sha256(buf.getvalue().encode()).hexdigest()[:16]
        manifest = {
            "version": "1.0",
            "target_profile": target.name,
            "model_hash": model_hash,
            "artifacts": {"payload": "payload.mlir"},
            "creation_timestamp": datetime.now(UTC).isoformat(),
        }
        manifest_path = self._output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Store paths for artifact collection
        self._manifest = manifest
        self._payload_path = payload_path

        return module

    def get_shared_artifacts(self) -> dict[str, Any]:
        """Return artifacts from shared passes."""
        return {
            "bundle_dir": str(self._output_dir),
            "manifest": getattr(self, "_manifest", {}),
        }

    def requirements_doc_path(self) -> Path:
        return Path(__file__).parent / "REQUIREMENTS.md"


def _try_verify(module: ModuleOp) -> bool:
    try:
        module.verify()
        return True
    except Exception:
        return False

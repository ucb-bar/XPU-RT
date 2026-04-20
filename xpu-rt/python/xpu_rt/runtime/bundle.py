"""Artifact bundling and manifest generation.

Packages all pipeline outputs into a deployable bundle directory matching
the Artifact Contract from CLAUDE.md.

Invariants:
    - manifest.json is the single source of truth for bundle contents.
    - All artifact paths in the manifest are relative to the bundle root.
    - Bundle is self-contained (no external references).
    - Bundle format is versioned.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xdsl.dialects.builtin import ModuleOp

    from compgen.runtime.planner import ExecutionPlan


@dataclass(frozen=True)
class BundleManifest:
    """Bundle manifest -- index of all artifacts.

    Attributes:
        version: Bundle format version.
        target_profile: Name of the target profile.
        model_hash: Hash of the original model IR.
        objective: Optimization objective.
        artifacts: Dict mapping artifact name to relative path.
        creation_timestamp: ISO 8601 timestamp.
    """

    version: str = "1.0"
    target_profile: str = ""
    model_hash: str = ""
    objective: str = "latency"
    artifacts: dict[str, str] = field(default_factory=dict)
    creation_timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "version": self.version,
            "target_profile": self.target_profile,
            "model_hash": self.model_hash,
            "objective": self.objective,
            "artifacts": self.artifacts,
            "creation_timestamp": self.creation_timestamp,
            "metadata": self.metadata,
        }


@dataclass
class BundleBuilder:
    """Builds artifact bundles from pipeline outputs.

    Attributes:
        output_dir: Root directory for the bundle.
    """

    output_dir: Path

    def build(
        self,
        module: ModuleOp,
        execution_plan: ExecutionPlan | None = None,
        target_name: str = "",
        objective: str = "latency",
        golden_inputs: Any = None,
        golden_outputs: Any = None,
        kernel_files: dict[str, str] | None = None,
        transform_scripts: list[str] | None = None,
        exported_program_text: str = "",
        recipe_mlir_text: str = "",
        recipe_yaml_text: str = "",
        kernel_contracts: list[dict[str, Any]] | None = None,
        verification_report: dict[str, Any] | None = None,
        extra_artifacts: dict[str, str] | None = None,
    ) -> BundleManifest:
        """Build a complete artifact bundle.

        Args:
            module: The optimized xDSL module.
            execution_plan: Execution plan (placement, scheduling).
            target_name: Target profile name.
            objective: Optimization objective.
            golden_inputs: Reference input tensors.
            golden_outputs: Reference output tensors.
            kernel_files: Dict of filename → kernel code.
            transform_scripts: List of transform script contents.

        Returns:
            BundleManifest describing the bundle.
        """
        root = Path(self.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}

        # 1. Write payload.mlir
        from xdsl.printer import Printer

        buf = io.StringIO()
        Printer(stream=buf).print_op(module)
        payload_path = root / "payload.mlir"
        payload_path.write_text(buf.getvalue())
        artifacts["payload"] = "payload.mlir"

        # 2. Write execution_plan.yaml
        if execution_plan is not None:
            import yaml

            plan_path = root / "execution_plan.yaml"
            plan_path.write_text(yaml.dump(execution_plan.to_dict(), default_flow_style=False))
            artifacts["execution_plan"] = "execution_plan.yaml"
            if execution_plan.memory_plans:
                memory_path = root / "memory_plan.yaml"
                memory_path.write_text(
                    yaml.dump(
                        [
                            {
                                "device": plan.device_index,
                                "peak_bytes": plan.peak_bytes,
                                "address_space": plan.address_space,
                                "physical_offset": plan.physical_offset,
                            }
                            for plan in execution_plan.memory_plans
                        ],
                        default_flow_style=False,
                    )
                )
                artifacts["memory_plan"] = "memory_plan.yaml"

        # 3. Write golden I/O
        if golden_inputs is not None:
            import torch

            inputs_path = root / "golden_inputs.pt"
            torch.save(golden_inputs, inputs_path)
            artifacts["golden_inputs"] = "golden_inputs.pt"

        if golden_outputs is not None:
            import torch

            outputs_path = root / "golden_outputs.pt"
            torch.save(golden_outputs, outputs_path)
            artifacts["golden_outputs"] = "golden_outputs.pt"

        # 4. Write kernel files
        if kernel_files:
            kernels_dir = root / "generated_kernels"
            kernels_dir.mkdir(exist_ok=True)
            for name, code in kernel_files.items():
                kernel_path = kernels_dir / name
                kernel_path.write_text(code)
            artifacts["generated_kernels"] = "generated_kernels/"

        # 5. Write transform scripts
        if transform_scripts:
            transforms_dir = root / "transforms"
            transforms_dir.mkdir(exist_ok=True)
            for i, script in enumerate(transform_scripts):
                script_path = transforms_dir / f"transform_{i}.py"
                script_path.write_text(script)
            artifacts["transforms"] = "transforms/"

        if exported_program_text:
            exported_path = root / "exported_program.txt"
            exported_path.write_text(exported_program_text)
            artifacts["exported_program"] = "exported_program.txt"

        if recipe_mlir_text:
            recipe_mlir_path = root / "recipe.mlir"
            recipe_mlir_path.write_text(recipe_mlir_text)
            artifacts["recipe_mlir"] = "recipe.mlir"

        if recipe_yaml_text:
            recipe_yaml_path = root / "recipe.yaml"
            recipe_yaml_path.write_text(recipe_yaml_text)
            artifacts["recipe_yaml"] = "recipe.yaml"

        if kernel_contracts is not None:
            contracts_path = root / "kernel_contracts.json"
            contracts_path.write_text(json.dumps(kernel_contracts, indent=2))
            artifacts["kernel_contracts"] = "kernel_contracts.json"

        if verification_report is not None:
            verification_path = root / "verification_report.json"
            verification_path.write_text(json.dumps(verification_report, indent=2))
            artifacts["verification_report"] = "verification_report.json"

        if extra_artifacts:
            for name, content in extra_artifacts.items():
                safe_name = f"{name}.txt"
                extra_path = root / safe_name
                extra_path.write_text(content)
                artifacts[name] = safe_name

        # 6. Compute model hash
        model_hash = hashlib.sha256(buf.getvalue().encode()).hexdigest()[:16]

        artifacts["manifest"] = "manifest.json"

        # 7. Write manifest.json
        manifest = BundleManifest(
            version="1.0",
            target_profile=target_name,
            model_hash=model_hash,
            objective=objective,
            artifacts=artifacts,
            creation_timestamp=datetime.now(UTC).isoformat(),
            metadata={"bundle_root": str(root)},
        )
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        return manifest


def create_bundle(
    output_dir: str | Path,
    module: ModuleOp,
    execution_plan: ExecutionPlan | None = None,
    **kwargs: Any,
) -> BundleManifest:
    """Convenience function: build a bundle."""
    builder = BundleBuilder(output_dir=Path(output_dir))
    return builder.build(module, execution_plan, **kwargs)


# Alias for backwards compat (promotion/ imports Bundle)
Bundle = BundleManifest

__all__ = ["Bundle", "BundleBuilder", "BundleManifest", "create_bundle"]

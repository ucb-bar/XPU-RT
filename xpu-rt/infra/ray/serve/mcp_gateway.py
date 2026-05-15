"""MCP Gateway — Ray Serve deployment exposing MCP tools.

Exposes 11 tools to Claude Code / Codex via the Model Context Protocol.
Each tool maps to an existing CompGen API entry point.

Tools:
    spec_validate, target_profile_normalize, compile_plan_generate,
    bundle_build, benchmark_run, artifact_fetch, eqsat_run, solver_run,
    verify_bundle, reserve_hardware, run_on_board
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from infra.ray._require import require_ray, require_serve

ray = require_ray()
serve = require_serve()

log = structlog.get_logger()


@dataclass(frozen=True)
class MCPToolDefinition:
    """MCP tool definition for the tool list response."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


# Tool definitions
MCP_TOOLS: list[MCPToolDefinition] = [
    MCPToolDefinition(
        name="spec_validate",
        description="Validate a hardware specification YAML file.",
        input_schema={"type": "object", "properties": {"spec_path": {"type": "string"}}, "required": ["spec_path"]},
    ),
    MCPToolDefinition(
        name="target_profile_normalize",
        description="Extract and normalize a TargetProfile from a hardware spec.",
        input_schema={"type": "object", "properties": {"spec_path": {"type": "string"}}, "required": ["spec_path"]},
    ),
    MCPToolDefinition(
        name="compile_plan_generate",
        description="Generate a compilation plan for a model on a target.",
        input_schema={
            "type": "object",
            "properties": {
                "model_class": {"type": "string"},
                "model_module": {"type": "string"},
                "target_spec_path": {"type": "string"},
                "objective": {"type": "string", "default": "latency"},
            },
            "required": ["model_class", "model_module", "target_spec_path"],
        },
    ),
    MCPToolDefinition(
        name="benchmark_run",
        description="Benchmark a model on a device.",
        input_schema={
            "type": "object",
            "properties": {
                "model_class": {"type": "string"},
                "model_module": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "required": ["model_class", "model_module"],
        },
    ),
    MCPToolDefinition(
        name="artifact_fetch",
        description="Fetch artifact metadata by ID.",
        input_schema={"type": "object", "properties": {"artifact_id": {"type": "string"}}, "required": ["artifact_id"]},
    ),
    MCPToolDefinition(
        name="eqsat_run",
        description="Run equality saturation optimization pass.",
        input_schema={
            "type": "object",
            "properties": {"module_ir_path": {"type": "string"}},
            "required": ["module_ir_path"],
        },
    ),
    MCPToolDefinition(
        name="solver_run",
        description="Run solver (CP-SAT/MILP) for placement/scheduling.",
        input_schema={
            "type": "object",
            "properties": {
                "problem": {"type": "object"},
                "solver_backend": {"type": "string", "default": "cp_sat"},
            },
            "required": ["problem"],
        },
    ),
    MCPToolDefinition(
        name="verify_bundle",
        description="Run verification ladder on a compiled bundle.",
        input_schema={
            "type": "object",
            "properties": {
                "bundle_path": {"type": "string"},
                "level": {"type": "string", "default": "all"},
            },
            "required": ["bundle_path"],
        },
    ),
    MCPToolDefinition(
        name="reserve_hardware",
        description="Reserve scarce hardware via the broker.",
        input_schema={
            "type": "object",
            "properties": {
                "resource_type": {"type": "string"},
                "timeout_s": {"type": "number", "default": 300},
            },
            "required": ["resource_type"],
        },
    ),
    MCPToolDefinition(
        name="run_on_board",
        description="Execute benchmark on reserved hardware.",
        input_schema={
            "type": "object",
            "properties": {
                "model_class": {"type": "string"},
                "model_module": {"type": "string"},
                "lease_id": {"type": "string"},
            },
            "required": ["model_class", "model_module", "lease_id"],
        },
    ),
    MCPToolDefinition(
        name="list_targets",
        description="List all registered compilation targets.",
        input_schema={"type": "object", "properties": {}},
    ),
]


@serve.deployment(route_prefix="/mcp")
class MCPGateway:
    """MCP server exposed via Ray Serve.

    Handles MCP JSON-RPC requests and dispatches to CompGen backends.
    """

    def __init__(
        self,
        registry_actor: Any,
        broker_actor: Any,
        artifact_actor: Any,
    ) -> None:
        self._registry = registry_actor
        self._broker = broker_actor
        self._artifacts = artifact_actor
        self._tool_handlers: dict[str, Any] = {
            "spec_validate": self._spec_validate,
            "target_profile_normalize": self._target_profile_normalize,
            "compile_plan_generate": self._compile_plan_generate,
            "benchmark_run": self._benchmark_run,
            "artifact_fetch": self._artifact_fetch,
            "eqsat_run": self._eqsat_run,
            "solver_run": self._solver_run,
            "verify_bundle": self._verify_bundle,
            "reserve_hardware": self._reserve_hardware,
            "run_on_board": self._run_on_board,
            "list_targets": self._list_targets,
        }

    async def __call__(self, request: Any) -> dict[str, Any]:
        """Handle raw HTTP request."""
        return {"status": "ok", "service": "compgen-mcp-gateway"}

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP-compliant tool definitions."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in MCP_TOOLS
        ]

    async def handle_tool_call(
        self, tool_name: str, arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch an MCP tool call to the appropriate backend.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Returns:
            Tool result dict.
        """
        handler = self._tool_handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = await handler(**arguments)
            return {"content": [{"type": "text", "text": str(result)}]}
        except Exception as e:
            log.error("mcp.tool_error", tool=tool_name, error=str(e))
            return {"error": str(e)}

    # -- Tool handlers ---------------------------------------------------

    async def _spec_validate(self, spec_path: str) -> dict[str, Any]:
        from compgen.targetgen.validate_spec import validate_hardware_spec

        errors = validate_hardware_spec(spec_path)
        return {"valid": len(errors) == 0, "errors": errors}

    async def _target_profile_normalize(self, spec_path: str) -> dict[str, Any]:
        from compgen.targetgen.load import load_hardware_spec

        spec = load_hardware_spec(spec_path)
        return {"name": spec.name, "platform": spec.platform.vendor}

    async def _compile_plan_generate(
        self, model_class: str, model_module: str, target_spec_path: str,
        objective: str = "latency",
    ) -> dict[str, Any]:
        from infra.ray.tasks.compile_job import compile_model_job

        ref = compile_model_job.remote(
            model_class=model_class,
            model_module=model_module,
            target_spec_path=target_spec_path,
            objective=objective,
            artifact_actor=self._artifacts,
        )
        return ray.get(ref)

    async def _benchmark_run(
        self, model_class: str, model_module: str, device: str = "cpu",
    ) -> dict[str, Any]:
        from infra.ray.tasks.benchmark_job import benchmark_job

        ref = benchmark_job.remote(
            model_class=model_class,
            model_module=model_module,
            device=device,
        )
        return ray.get(ref)

    async def _artifact_fetch(self, artifact_id: str) -> dict[str, Any]:
        result = ray.get(self._artifacts.get_artifact.remote(artifact_id))
        return result or {"error": "Artifact not found"}

    async def _eqsat_run(self, module_ir_path: str) -> dict[str, Any]:
        return {"status": "eqsat_run_placeholder", "ir_path": module_ir_path}

    async def _solver_run(
        self, problem: dict[str, Any], solver_backend: str = "cp_sat",
    ) -> dict[str, Any]:
        from infra.ray.tasks.solver_job import solver_job

        ref = solver_job.remote(problem=problem, solver_backend=solver_backend)
        return ray.get(ref)

    async def _verify_bundle(
        self, bundle_path: str, level: str = "all",
    ) -> dict[str, Any]:
        from infra.ray.tasks.verify_job import verify_bundle_job

        ref = verify_bundle_job.remote(bundle_path=bundle_path, level=level)
        return ray.get(ref)

    async def _reserve_hardware(
        self, resource_type: str, timeout_s: float = 300,
    ) -> dict[str, Any]:
        result = ray.get(
            self._broker.reserve.remote(resource_type, "mcp_agent", timeout_s)
        )
        return result or {"error": "No resource available"}

    async def _run_on_board(
        self, model_class: str, model_module: str, lease_id: str,
    ) -> dict[str, Any]:
        from infra.ray.tasks.benchmark_job import benchmark_job

        ref = benchmark_job.remote(
            model_class=model_class,
            model_module=model_module,
            hardware_lease_id=lease_id,
            broker_actor=self._broker,
        )
        return ray.get(ref)

    async def _list_targets(self) -> list[str]:
        return ray.get(self._registry.list_targets.remote())


__all__ = ["MCPGateway", "MCPToolDefinition", "MCP_TOOLS"]

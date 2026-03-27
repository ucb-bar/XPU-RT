"""RunAPI — Ray Serve deployment for job management.

Exposes REST endpoints for submitting and monitoring compilation,
benchmark, and verification jobs.
"""

from __future__ import annotations

from typing import Any

from infra.ray._require import require_ray, require_serve

ray = require_ray()
serve = require_serve()


@serve.deployment(route_prefix="/api/v1")
class RunAPI:
    """REST API for submitting and monitoring Ray jobs.

    Endpoints:
        POST /api/v1/compile    — submit compilation job
        POST /api/v1/benchmark  — submit benchmark job
        POST /api/v1/verify     — submit verification job
        GET  /api/v1/runs       — list submitted runs
        GET  /api/v1/targets    — list registered targets
        GET  /api/v1/resources  — list hardware resources
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
        self._runs: dict[str, dict[str, Any]] = {}

    async def __call__(self, request: Any) -> dict[str, Any]:
        """Handle HTTP requests (simplified — real impl uses FastAPI)."""
        return {"status": "ok", "service": "compgen-run-api"}

    async def submit_compile(
        self,
        model_class: str,
        model_module: str,
        target_spec_path: str,
        objective: str = "latency",
    ) -> dict[str, Any]:
        """Submit a compilation job."""
        from infra.ray.tasks.compile_job import compile_model_job

        ref = compile_model_job.remote(
            model_class=model_class,
            model_module=model_module,
            target_spec_path=target_spec_path,
            objective=objective,
            artifact_actor=self._artifacts,
        )

        import uuid

        run_id = str(uuid.uuid4())
        self._runs[run_id] = {
            "run_id": run_id,
            "type": "compile",
            "status": "running",
            "ref": ref,
        }
        return {"run_id": run_id, "status": "submitted"}

    async def submit_benchmark(
        self,
        model_class: str,
        model_module: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Submit a benchmark job."""
        from infra.ray.tasks.benchmark_job import benchmark_job

        ref = benchmark_job.remote(
            model_class=model_class,
            model_module=model_module,
            device=device,
            broker_actor=self._broker,
        )

        import uuid

        run_id = str(uuid.uuid4())
        self._runs[run_id] = {
            "run_id": run_id,
            "type": "benchmark",
            "status": "running",
            "ref": ref,
        }
        return {"run_id": run_id, "status": "submitted"}

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get run status."""
        run = self._runs.get(run_id)
        if run is None:
            return None

        ref = run.get("ref")
        if ref is not None:
            try:
                result = ray.get(ref, timeout=0.1)
                run["status"] = "completed"
                run["result"] = result
                run.pop("ref", None)
            except Exception:
                pass  # Still running

        return {k: v for k, v in run.items() if k != "ref"}

    async def list_runs(self) -> list[dict[str, Any]]:
        """List all runs."""
        return [
            {k: v for k, v in run.items() if k != "ref"}
            for run in self._runs.values()
        ]

    async def list_targets(self) -> list[str]:
        """List registered targets."""
        return ray.get(self._registry.list_targets.remote())

    async def list_resources(self) -> list[dict[str, Any]]:
        """List hardware resources."""
        return ray.get(self._broker.list_resources.remote())


__all__ = ["RunAPI"]

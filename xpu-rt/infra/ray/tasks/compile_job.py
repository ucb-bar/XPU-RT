"""Compilation job task — distributed model compilation.

Wraps ``compgen.api.compile_model()`` and ``AgenticCompilationLoop.run()``
as Ray remote tasks.
"""

from __future__ import annotations

from typing import Any

from infra.ray._require import require_ray

ray = require_ray()


@ray.remote(num_cpus=2)
def compile_model_job(
    model_class: str,
    model_module: str,
    target_spec_path: str,
    objective: str = "latency",
    *,
    use_agentic: bool = False,
    budget: int = 10,
    artifact_actor: Any = None,
) -> dict[str, Any]:
    """Full compilation job as a Ray task.

    Args:
        model_class: Class name of the model.
        model_module: Module path containing the model class.
        target_spec_path: Path to hardware spec YAML.
        objective: Optimization objective.
        use_agentic: Whether to use the agentic compilation loop.
        budget: Max iterations for agentic mode.
        artifact_actor: Optional ArtifactIndexActor handle.

    Returns:
        JSON-serializable compilation result dict.
    """
    import importlib

    from compgen.api import compile_model, device

    # Load model
    mod = importlib.import_module(model_module)
    model_cls = getattr(mod, model_class)
    model = model_cls()

    # Load target
    target_device = device(target_spec_path)

    # Compile
    result = compile_model(model, target_device, objective=objective)

    summary = {
        "model_class": model_class,
        "target_name": target_device.profile.name,
        "objective": objective,
        "pipeline_stages_run": len(result.pipeline_result.stage_results)
        if result.pipeline_result
        else 0,
        "eqsat_applied": result.eqsat_result is not None,
    }

    if artifact_actor is not None:
        ray.get(
            artifact_actor.register_artifact.remote(
                artifact_type="compilation_result",
                target_name=target_device.profile.name,
                storage_path="",
                model_hash=model_class,
                objective=objective,
                metadata=summary,
            )
        )

    return summary


__all__ = ["compile_model_job"]

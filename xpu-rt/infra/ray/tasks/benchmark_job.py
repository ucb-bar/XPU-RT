"""Benchmark job task — run benchmarks on potentially remote hardware.

Wraps ``LocalExecutor.benchmark()`` as a Ray remote task.
"""

from __future__ import annotations

from typing import Any

from infra.ray._require import require_ray

ray = require_ray()


@ray.remote
def benchmark_job(
    model_class: str,
    model_module: str,
    *,
    device: str = "cpu",
    mode: str = "eager",
    num_iterations: int = 100,
    warmup: int = 10,
    hardware_lease_id: str | None = None,
    broker_actor: Any = None,
) -> dict[str, Any]:
    """Benchmark a model, potentially on remote hardware.

    Args:
        model_class: Class name of the model.
        model_module: Module path containing the model class.
        device: Device to benchmark on.
        mode: Benchmark mode ("eager", "compile", "export").
        num_iterations: Number of benchmark iterations.
        warmup: Number of warmup iterations.
        hardware_lease_id: Optional lease ID for reserved hardware.
        broker_actor: Optional HardwareBrokerActor handle.

    Returns:
        Benchmark result dict.
    """
    # Validate lease if provided
    if hardware_lease_id and broker_actor:
        leases = ray.get(broker_actor.list_leases.remote())
        valid = any(
            entry["lease_id"] == hardware_lease_id and entry["status"] == "active"
            for entry in leases
        )
        if not valid:
            return {"error": f"Invalid or expired lease: {hardware_lease_id}"}

    import importlib

    mod = importlib.import_module(model_module)
    model_cls = getattr(mod, model_class)
    model = model_cls()

    # Get sample inputs if the module provides them
    get_inputs = getattr(mod, "get_sample_inputs", None)
    sample_inputs = get_inputs() if get_inputs else None

    from compgen.runtime.local_executor import LocalExecutor

    executor = LocalExecutor()
    result = executor.benchmark(
        model=model,
        sample_inputs=sample_inputs,
        device=device,
        mode=mode,
        num_iterations=num_iterations,
        warmup=warmup,
    )

    return {
        "model_class": model_class,
        "device": device,
        "mode": mode,
        "mean_latency_us": result.mean_latency_us if result else 0.0,
        "std_latency_us": result.std_latency_us if result else 0.0,
        "num_iterations": num_iterations,
    }


__all__ = ["benchmark_job"]

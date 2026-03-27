"""Spec ingestion task — load and register a hardware spec.

Wraps ``compgen.api.device()`` as a Ray remote task.
"""

from __future__ import annotations

from typing import Any

from infra.ray._require import require_ray

ray = require_ray()


@ray.remote
def ingest_spec(
    spec_path: str,
    output_dir: str | None = None,
    registry_actor: Any = None,
) -> dict[str, Any]:
    """Load a hardware spec, generate target, optionally register.

    Args:
        spec_path: Path to hardware spec YAML.
        output_dir: Optional output directory for generated artifacts.
        registry_actor: Optional TargetRegistryActor handle.

    Returns:
        Serialized target summary dict.
    """
    from compgen.api import device

    dev = device(spec_path, output_dir)

    summary = {
        "name": dev.profile.name,
        "spec_path": spec_path,
        "num_devices": len(dev.profile.devices),
        "device_types": [d.device_type for d in dev.profile.devices],
    }

    if registry_actor is not None:
        ray.get(registry_actor.register_target.remote(spec_path, output_dir))

    return summary


__all__ = ["ingest_spec"]

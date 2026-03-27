"""TargetRegistry actor — stores target profiles and calibration data.

Wraps ``compgen.api.device()`` and stores the results in a Ray actor
so they can be shared across distributed jobs and actors.

All state is JSON-serializable (no pickle).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from infra.ray._require import require_ray

ray = require_ray()


@dataclass
class TargetRecord:
    """Serializable record of a registered target."""

    name: str
    spec_path: str
    profile_summary: dict[str, Any] = field(default_factory=dict)
    capabilities_summary: dict[str, Any] = field(default_factory=dict)
    calibration_data: dict[str, Any] = field(default_factory=dict)
    cost_model: dict[str, Any] = field(default_factory=dict)
    maturity_level: str = "L0"
    registered_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict representation."""
        return asdict(self)


@ray.remote
class TargetRegistryActor:
    """Ray actor storing target profiles, capability maps, and calibration.

    All mutations are serialized through the actor's single-threaded mailbox.
    State is a dict keyed by target name.
    """

    def __init__(self) -> None:
        self._targets: dict[str, TargetRecord] = {}

    def register_target(
        self,
        spec_path: str,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Load a hardware spec and register the target.

        Wraps ``compgen.api.device()`` internally.

        Args:
            spec_path: Path to hardware spec YAML.
            output_dir: Optional output directory for generated artifacts.

        Returns:
            Serialized TargetRecord dict.
        """
        from compgen.api import device

        dev = device(spec_path, output_dir)

        record = TargetRecord(
            name=dev.profile.name,
            spec_path=spec_path,
            profile_summary={
                "name": dev.profile.name,
                "schema_version": dev.profile.schema_version,
                "num_devices": len(dev.profile.devices),
                "device_types": [d.device_type for d in dev.profile.devices],
            },
            capabilities_summary={
                "target_class": dev.capabilities.target_class.value
                if dev.capabilities
                else "unknown",
            },
            calibration_data=dict(dev.profile.calibration_data),
            cost_model=dict(dev.profile.cost_model),
            maturity_level="L0",
            registered_at=datetime.now(UTC).isoformat(),
        )

        self._targets[record.name] = record
        return record.to_dict()

    def get_target(self, target_name: str) -> dict[str, Any] | None:
        """Look up a target by name."""
        record = self._targets.get(target_name)
        return record.to_dict() if record else None

    def list_targets(self) -> list[str]:
        """List all registered target names."""
        return list(self._targets.keys())

    def update_calibration(
        self, target_name: str, calibration_data: dict[str, Any],
    ) -> bool:
        """Merge calibration data for a registered target."""
        record = self._targets.get(target_name)
        if record is None:
            return False
        record.calibration_data.update(calibration_data)
        return True

    def update_cost_model(
        self, target_name: str, cost_model: dict[str, Any],
    ) -> bool:
        """Update cost model snapshot."""
        record = self._targets.get(target_name)
        if record is None:
            return False
        record.cost_model.update(cost_model)
        return True

    def get_maturity(self, target_name: str) -> str:
        """Return maturity level (L0/L1/L2/L3)."""
        record = self._targets.get(target_name)
        return record.maturity_level if record else "unknown"

    def set_maturity(self, target_name: str, level: str) -> bool:
        """Set maturity level."""
        record = self._targets.get(target_name)
        if record is None:
            return False
        record.maturity_level = level
        return True

    def export_snapshot(self) -> dict[str, Any]:
        """Export all registry state as JSON-serializable dict."""
        return {
            name: record.to_dict()
            for name, record in self._targets.items()
        }


__all__ = ["TargetRecord", "TargetRegistryActor"]

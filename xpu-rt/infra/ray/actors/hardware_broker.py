"""HardwareBroker actor — manages scarce hardware resources.

Implements lease-based access to boards, FPGAs, simulators, and
license servers.  Uses Ray custom resources for scheduling constraints.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from infra.ray._require import require_ray

ray = require_ray()


@dataclass
class HardwareResource:
    """A hardware resource managed by the broker."""

    resource_id: str
    resource_type: str  # "board", "fpga", "simulator", "license_server"
    target_name: str
    node_address: str = ""
    custom_resources: dict[str, float] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "target_name": self.target_name,
            "node_address": self.node_address,
            "custom_resources": self.custom_resources,
            "properties": self.properties,
            "available": self.available,
        }


@dataclass
class Lease:
    """A lease on a hardware resource."""

    lease_id: str
    resource_id: str
    requester: str
    acquired_at: str
    expires_at: str
    status: str = "active"  # "active", "expired", "released"

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "resource_id": self.resource_id,
            "requester": self.requester,
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
            "status": self.status,
        }


@ray.remote
class HardwareBrokerActor:
    """Owns scarce hardware resources and enforces lease-based access.

    Resources are reserved with a TTL — auto-released after timeout.
    """

    def __init__(self) -> None:
        self._resources: dict[str, HardwareResource] = {}
        self._leases: dict[str, Lease] = {}

    def register_resource(self, spec: dict[str, Any]) -> str:
        """Register a hardware resource.

        Args:
            spec: Resource specification dict with keys: resource_id,
                resource_type, target_name, node_address, custom_resources.

        Returns:
            The resource_id.
        """
        resource = HardwareResource(
            resource_id=spec["resource_id"],
            resource_type=spec.get("resource_type", "board"),
            target_name=spec.get("target_name", ""),
            node_address=spec.get("node_address", ""),
            custom_resources=spec.get("custom_resources", {}),
            properties=spec.get("properties", {}),
        )
        self._resources[resource.resource_id] = resource
        return resource.resource_id

    def reserve(
        self,
        resource_type: str,
        requester: str,
        timeout_s: float = 300.0,
    ) -> dict[str, Any] | None:
        """Reserve a resource of the given type.

        Args:
            resource_type: Type to reserve (e.g., "fpga", "board").
            requester: Who is requesting.
            timeout_s: Lease TTL in seconds.

        Returns:
            Lease dict, or None if no resource is available.
        """
        self._expire_leases()

        for resource in self._resources.values():
            if resource.resource_type == resource_type and resource.available:
                now = datetime.now(UTC)
                lease = Lease(
                    lease_id=str(uuid.uuid4()),
                    resource_id=resource.resource_id,
                    requester=requester,
                    acquired_at=now.isoformat(),
                    expires_at=(now + timedelta(seconds=timeout_s)).isoformat(),
                )
                resource.available = False
                self._leases[lease.lease_id] = lease
                return lease.to_dict()

        return None

    def release(self, lease_id: str) -> bool:
        """Release a lease."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return False

        lease.status = "released"
        resource = self._resources.get(lease.resource_id)
        if resource is not None:
            resource.available = True
        return True

    def list_resources(self) -> list[dict[str, Any]]:
        """List all registered resources."""
        self._expire_leases()
        return [r.to_dict() for r in self._resources.values()]

    def list_leases(self) -> list[dict[str, Any]]:
        """List all active leases."""
        self._expire_leases()
        return [
            lease.to_dict()
            for lease in self._leases.values()
            if lease.status == "active"
        ]

    def get_resource(self, resource_id: str) -> dict[str, Any] | None:
        """Get a resource by ID."""
        resource = self._resources.get(resource_id)
        return resource.to_dict() if resource else None

    def _expire_leases(self) -> None:
        """Expire any leases past their TTL."""
        now = datetime.now(UTC)
        for lease in self._leases.values():
            if lease.status == "active":
                expires = datetime.fromisoformat(lease.expires_at)
                if now > expires:
                    lease.status = "expired"
                    resource = self._resources.get(lease.resource_id)
                    if resource is not None:
                        resource.available = True


__all__ = ["HardwareBrokerActor", "HardwareResource", "Lease"]

"""Orchestrated validation combining probe, env-check, and aperture enforcement."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from compgen.packs.base import LoadedPack
from compgen.packs.envcheck import EnvCheckResult, check_pack_environment
from compgen.packs.schema import PackProbeResult
from compgen.packs.verify import OwnershipViolation, check_surface_allowed


@dataclass(frozen=True)
class PackValidationResult:
    """Aggregated validation outcome for a single pack within a pack set.

    Attributes:
        pack_name: Name of the validated pack.
        ok: True only when probe, env-check, and aperture checks all pass.
        probe: Result of the structural probe for the pack.
        env_check: Result of the host-environment tool/path check.
        aperture_violations: Ownership violations detected against other active packs.
    """

    pack_name: str
    ok: bool
    probe: PackProbeResult
    env_check: EnvCheckResult
    aperture_violations: list[OwnershipViolation] = field(default_factory=list)


def validate_pack(
    pack: LoadedPack,
    *,
    all_packs: Iterable[LoadedPack] = (),
    required_paths: list[str] | None = None,
    required_tools: list[str] | None = None,
) -> PackValidationResult:
    """Run probe, env-check, and aperture enforcement for *pack*.

    Args:
        pack: The pack to validate.
        all_packs: All active packs (used for sealed-surface conflict detection).
        required_paths: Extra filesystem paths to verify. Defaults to empty.
        required_tools: Extra CLI tools to verify. Defaults to empty.

    Returns:
        A ``PackValidationResult`` combining every sub-check.
    """

    probe = pack.pack.probe()

    env_check = check_pack_environment(
        required_paths=required_paths or [],
        required_tools=required_tools or [],
    )

    violations: list[OwnershipViolation] = []
    all_packs_list = list(all_packs)
    for surface in pack.manifest.owned_surfaces:
        violation = check_surface_allowed(all_packs_list, requested_surface=surface)
        if violation is not None:
            violations.append(violation)

    ok = probe.available and env_check.ok and not violations
    return PackValidationResult(
        pack_name=pack.manifest.name,
        ok=ok,
        probe=probe,
        env_check=env_check,
        aperture_violations=violations,
    )

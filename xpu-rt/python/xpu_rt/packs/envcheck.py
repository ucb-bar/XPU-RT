"""Environment checks for extension-pack readiness."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvCheckResult:
    """Result of checking whether required paths and tools are present.

    Attributes:
        ok: True when all required paths exist and all required tools are found.
        missing_paths: Filesystem paths that were required but do not exist.
        missing_tools: CLI tool names that were required but not found on PATH.
    """

    ok: bool
    missing_paths: list[str]
    missing_tools: list[str]


def check_pack_environment(
    *,
    required_paths: list[str],
    required_tools: list[str],
) -> EnvCheckResult:
    """Check whether the host environment satisfies a pack's requirements.

    Args:
        required_paths: Filesystem paths that must exist.
        required_tools: CLI tool names that must be resolvable via ``shutil.which``.

    Returns:
        An ``EnvCheckResult`` summarising any missing paths or tools.
    """

    missing_paths = [p for p in required_paths if not Path(p).exists()]
    missing_tools = [t for t in required_tools if shutil.which(t) is None]
    return EnvCheckResult(
        ok=not missing_paths and not missing_tools,
        missing_paths=missing_paths,
        missing_tools=missing_tools,
    )

"""Extension discovery + per-run registry.

Discovers ``compgen_extension.yaml`` manifests under a configured
extensions root (default ``.rcg-artifacts/extensions/``), validates
each through the schema, sandboxes their declared
``allowed_write_root``, and registers them into an in-memory
``ExtensionRegistry`` for the current run.

Per-run scope by design: registration is never persistent. Trust
discipline is per-call — a fresh ``ExtensionRegistry`` is built
each time a run starts, and rejected extensions are surfaced with
their typed reason rather than silently dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from compgen.extensions.errors import (
    ExtensionError,
    ExtensionManifestError,
    ExtensionSandboxViolation,
)
from compgen.extensions.manifest import (
    ExtensionManifest,
    load_manifest,
)
from compgen.extensions.sandbox import is_under_sandbox

DEFAULT_EXTENSIONS_ROOT: Final[Path] = Path(".rcg-artifacts/extensions")

EXTENSION_MANIFEST_FILENAME: Final[str] = "compgen_extension.yaml"


@dataclass(frozen=True)
class RejectedExtension:
    """A discovered extension that failed validation.

    Carries the typed reason so the caller can decide whether to
    surface it as a per-run blocker or just log it. Reasons are a
    closed enum.
    """

    extension_dir: Path
    failed_check: str
    detail: str

    REASONS: Final = (
        "manifest_missing",
        "manifest_schema",
        "extension_sandbox_violation",
        "duplicate_extension_id",
    )


@dataclass
class ExtensionRegistry:
    """In-memory registry of validated extensions for the current run."""

    accepted: list[ExtensionManifest] = field(default_factory=list)
    rejected: list[RejectedExtension] = field(default_factory=list)

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(
            p.provider_id
            for m in self.accepted
            for p in m.kernel_providers
        )

    def target_ids(self) -> tuple[str, ...]:
        return tuple(
            t.target_id
            for m in self.accepted
            for t in m.targets
        )

    def dialect_provider_ids(self) -> tuple[str, ...]:
        return tuple(
            d.dialect_provider_id
            for m in self.accepted
            for d in m.dialect_providers
        )

    def pass_tool_ids(self) -> tuple[str, ...]:
        return tuple(
            pt.tool_id
            for m in self.accepted
            for pt in m.pass_tools
        )

    def extension_ids(self) -> tuple[str, ...]:
        return tuple(m.extension_id for m in self.accepted)

    def rejected_summary(self) -> tuple[dict, ...]:
        return tuple(
            {
                "extension_dir": str(r.extension_dir),
                "failed_check": r.failed_check,
                "detail": r.detail,
            }
            for r in self.rejected
        )


def discover_manifests(root: Path | None = None) -> Iterator[Path]:
    """Yield every ``compgen_extension.yaml`` path under ``root``.

    A non-existent or empty root is honored: no manifests yielded,
    no exception raised.
    """

    base = Path(root) if root is not None else DEFAULT_EXTENSIONS_ROOT
    if not base.is_dir():
        return
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        manifest = sub / EXTENSION_MANIFEST_FILENAME
        if manifest.is_file():
            yield manifest


def _validate_sandbox_against_manifest(
    manifest: ExtensionManifest,
) -> None:
    """Verify the manifest's ``allowed_write_root`` is a subdirectory
    of the extension's own directory (typical pattern) or under a
    well-known sandbox base. Rejects roots that escape via ``..``.

    The check is structural: we resolve the declared root and ensure
    it stays under the extension's directory or the configured
    extensions tree.
    """

    if not manifest.security.sandbox_required:
        return
    if manifest.source_path is None:
        return
    extension_dir = manifest.source_path.parent.resolve()
    write_root = (extension_dir / manifest.security.allowed_write_root).resolve()
    # The write root must either equal or sit under the extension dir.
    try:
        write_root.relative_to(extension_dir)
    except ValueError as exc:
        raise ExtensionSandboxViolation(
            path=str(write_root),
            allowed_root=str(extension_dir),
            reason="write_root_escapes_extension_dir",
        ) from exc


def build_registry(
    root: Path | None = None,
    *,
    manifests: Iterable[ExtensionManifest] | None = None,
) -> ExtensionRegistry:
    """Build an :class:`ExtensionRegistry` for the current run.

    Either pass ``manifests`` directly (for tests / programmatic
    registration) or let the registry discover them under ``root``.
    Per-run scope: a fresh registry is returned every call.
    """

    registry = ExtensionRegistry()
    seen_ids: set[str] = set()

    candidates: list[tuple[Path | None, ExtensionManifest | Exception]] = []
    if manifests is not None:
        for m in manifests:
            candidates.append((m.source_path, m))
    else:
        for path in discover_manifests(root):
            try:
                m = load_manifest(path)
                candidates.append((path, m))
            except ExtensionManifestError as exc:
                registry.rejected.append(
                    RejectedExtension(
                        extension_dir=path.parent,
                        failed_check="manifest_schema",
                        detail=str(exc),
                    )
                )

    for path, m_or_exc in candidates:
        if isinstance(m_or_exc, Exception):
            continue
        manifest = m_or_exc

        try:
            _validate_sandbox_against_manifest(manifest)
        except ExtensionSandboxViolation as exc:
            registry.rejected.append(
                RejectedExtension(
                    extension_dir=(path.parent if path else Path(".")),
                    failed_check="extension_sandbox_violation",
                    detail=exc.reason,
                )
            )
            continue

        if manifest.extension_id in seen_ids:
            registry.rejected.append(
                RejectedExtension(
                    extension_dir=(path.parent if path else Path(".")),
                    failed_check="duplicate_extension_id",
                    detail=manifest.extension_id,
                )
            )
            continue

        seen_ids.add(manifest.extension_id)
        registry.accepted.append(manifest)

    return registry


def assert_artifact_write_allowed(
    target_path: str | Path,
    manifest: ExtensionManifest,
) -> Path:
    """Validate that ``target_path`` is a legal write for the
    extension described by ``manifest``.

    Raises :class:`ExtensionSandboxViolation` per
    :func:`compgen.extensions.sandbox.validate_sandboxed_path`
    semantics. Convenience wrapper that injects the manifest's
    declared ``allowed_write_root``.
    """

    if manifest.source_path is None:
        # No on-disk anchor — caller must pass an absolute write root.
        raise ExtensionError(
            f"extension {manifest.extension_id!r} has no source_path; cannot "
            f"resolve allowed_write_root for sandbox check"
        )
    extension_dir = manifest.source_path.parent
    allowed = (extension_dir / manifest.security.allowed_write_root).resolve()
    from compgen.extensions.sandbox import validate_sandboxed_path
    return validate_sandboxed_path(target_path, allowed_write_root=allowed)


def is_artifact_write_allowed(
    target_path: str | Path,
    manifest: ExtensionManifest,
) -> bool:
    """Non-raising variant of :func:`assert_artifact_write_allowed`."""

    if manifest.source_path is None:
        return False
    extension_dir = manifest.source_path.parent
    allowed = (extension_dir / manifest.security.allowed_write_root).resolve()
    return is_under_sandbox(target_path, allowed_write_root=allowed)

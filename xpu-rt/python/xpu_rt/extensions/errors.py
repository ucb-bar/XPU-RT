"""Typed errors for the extension substrate."""

from __future__ import annotations


class ExtensionError(RuntimeError):
    """Base error for all extension-substrate failures."""


class ExtensionManifestError(ExtensionError):
    """A ``compgen_extension.yaml`` manifest violated the schema."""


class ExtensionSandboxViolation(ExtensionError):
    """An extension attempted a write outside its sandbox.

    Hard rule 2: providers may only write under their assigned
    ``artifact_dir``. Hard rule 3: user extensions may not mutate
    Payload IR, Recipe IR, contracts, manifests, or run ledgers.
    Violations carry the offending path so the typed rejection
    contract can be honored.
    """

    def __init__(self, *, path: str, allowed_root: str, reason: str) -> None:
        self.path = path
        self.allowed_root = allowed_root
        self.reason = reason
        super().__init__(
            f"extension_sandbox_violation: path={path!r} reason={reason!r} "
            f"(allowed_root={allowed_root!r})"
        )


class ExtensionTaskError(ExtensionError):
    """An ``extension_task_v1`` artifact violated the schema."""

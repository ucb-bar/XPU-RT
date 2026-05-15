"""Provider / dialect / pass-tool probing.

Turns a ``ProviderCard`` / ``DialectProviderCard`` into a typed
``ProviderProbeResult``. Hard rule 5: missing SDKs, hardware, or
licenses always produce a typed ``blocked`` status with a typed
``blocked_reason`` — never a crash, never a silent disappearance.

The probe is environmental only:

* ``required_env`` — every named env var must be set + non-empty.
* ``required_commands`` — every named command must resolve on
  ``$PATH``.
* ``required_python_imports`` — every named module must import.

The order is deterministic, and the first failed check wins. The
``detail`` field carries the offending name so the matrix report
can be specific.

Probes never invoke the provider's ``propose()`` / ``can_bid()``;
they only verify the toolchain pre-conditions declared on the
card. ``status=available`` therefore means "could be invoked", not
"would succeed on a real contract".
"""

from __future__ import annotations

import importlib
import os
import shutil
from collections.abc import Iterable

from compgen.dialects.dialect_provider_types import DialectProviderCard
from compgen.providers.provider_types import (
    ProviderCard,
    ProviderProbeError,
    ProviderProbeResult,
)

PROBE_SCHEMA_VERSION = "provider_status_v1"


def _check_env(names: Iterable[str]) -> tuple[bool, str]:
    for name in names:
        if not os.environ.get(name, "").strip():
            return False, name
    return True, ""


def _check_commands(names: Iterable[str]) -> tuple[bool, str]:
    for name in names:
        # Commands may include arguments (e.g. ``myaccel-cc --version``).
        cmd = name.split()[0] if name else ""
        if not cmd:
            return False, name
        if shutil.which(cmd) is None:
            return False, cmd
    return True, ""


def _check_imports(names: Iterable[str]) -> tuple[bool, str]:
    for name in names:
        try:
            importlib.import_module(name)
        except (ImportError, ModuleNotFoundError):
            return False, name
        except Exception:
            # A package that imports but raises something exotic is a
            # probe_error, not a missing-package — re-raise so the
            # caller can map it.
            raise
    return True, ""


def probe_provider(card: ProviderCard) -> ProviderProbeResult:
    """Run the structural probe against ``card`` and return a typed result."""

    try:
        ok, name = _check_env(card.required_env)
        if not ok:
            return ProviderProbeResult(
                schema_version=PROBE_SCHEMA_VERSION,
                provider_id=card.provider_id,
                status="blocked",
                blocked_reason="env_missing",
                detail=name,
                paper_claimable=card.paper_claimable,
                required_env=card.required_env,
                required_commands=card.required_commands,
            )
        ok, name = _check_commands(card.required_commands)
        if not ok:
            return ProviderProbeResult(
                schema_version=PROBE_SCHEMA_VERSION,
                provider_id=card.provider_id,
                status="blocked",
                blocked_reason="command_missing",
                detail=name,
                paper_claimable=card.paper_claimable,
                required_env=card.required_env,
                required_commands=card.required_commands,
            )
        ok, name = _check_imports(card.required_python_imports)
        if not ok:
            return ProviderProbeResult(
                schema_version=PROBE_SCHEMA_VERSION,
                provider_id=card.provider_id,
                status="blocked",
                blocked_reason="python_package_missing",
                detail=name,
                paper_claimable=card.paper_claimable,
                required_env=card.required_env,
                required_commands=card.required_commands,
            )
    except ProviderProbeError:
        raise
    except Exception as exc:
        return ProviderProbeResult(
            schema_version=PROBE_SCHEMA_VERSION,
            provider_id=card.provider_id,
            status="probe_error",
            blocked_reason="probe_exception",
            detail=f"{type(exc).__name__}: {exc}",
            paper_claimable=card.paper_claimable,
            required_env=card.required_env,
            required_commands=card.required_commands,
        )

    return ProviderProbeResult(
        schema_version=PROBE_SCHEMA_VERSION,
        provider_id=card.provider_id,
        status="available",
        blocked_reason=None,
        version="",
        supports=card.contract_kinds,
        paper_claimable=card.paper_claimable,
        required_env=card.required_env,
        required_commands=card.required_commands,
    )


def probe_dialect_provider(card: DialectProviderCard) -> ProviderProbeResult:
    """Probe a dialect provider card.

    Reuses ``ProviderProbeResult`` since the typed-status discipline
    is identical. The ``supports`` field carries the ``consumes``
    list as a coarse capability summary.
    """

    try:
        ok, name = _check_env(card.required_env)
        if not ok:
            return ProviderProbeResult(
                schema_version=PROBE_SCHEMA_VERSION,
                provider_id=card.dialect_provider_id,
                status="blocked",
                blocked_reason="env_missing",
                detail=name,
                paper_claimable=card.paper_claimable,
                required_env=card.required_env,
            )
    except Exception as exc:
        return ProviderProbeResult(
            schema_version=PROBE_SCHEMA_VERSION,
            provider_id=card.dialect_provider_id,
            status="probe_error",
            blocked_reason="probe_exception",
            detail=f"{type(exc).__name__}: {exc}",
            paper_claimable=card.paper_claimable,
            required_env=card.required_env,
        )

    if card.integration_level == "card_only":
        return ProviderProbeResult(
            schema_version=PROBE_SCHEMA_VERSION,
            provider_id=card.dialect_provider_id,
            status="unsupported",
            blocked_reason="unsupported_contract_kind",
            detail="dialect declared card_only — no executable path",
            paper_claimable=card.paper_claimable,
            required_env=card.required_env,
        )

    return ProviderProbeResult(
        schema_version=PROBE_SCHEMA_VERSION,
        provider_id=card.dialect_provider_id,
        status="available",
        blocked_reason=None,
        supports=card.consumes,
        paper_claimable=card.paper_claimable,
        required_env=card.required_env,
    )

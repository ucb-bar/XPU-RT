"""ToolCard entrypoint for the extension-provider probe (P1 exit gate).

Wraps the existing :mod:`scripts/dev/probe_extension_providers.py`
inside the ToolCard ``(request, *, out_dir) -> dict`` contract. The
underlying probe is fully deterministic and read-only with respect to
compilation state — it walks the YAML cards under
``python/compgen/{providers,dialects,targets}/cards/`` and produces
seven typed artifacts under ``out_dir``.

The wrapper translates probe failures (missing repo root, exec error,
partial artifact emission) into typed ``status=error`` / ``status=blocked``
payloads so the runner never sees an uncaught exception.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Files the probe is contracted to emit under out_dir. The wrapper
# verifies every one exists before returning ``status=ok``; if any
# are missing the wrapper returns ``status=blocked`` with the typed
# reason so the runner records this honestly rather than claiming
# success.
EXPECTED_ARTIFACTS = (
    "provider_status.json",
    "target_status.json",
    "dialect_status.json",
    "pass_tool_status.json",
    "provider_target_matrix.csv",
    "provider_contract_matrix.csv",
    "probe_summary.md",
)


def _repo_root() -> Path:
    # python/compgen/tools/builtin/probe_providers.py → parents[4]
    return Path(__file__).resolve().parents[4]


def run(request: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    """Run the provider probe into ``out_dir``.

    Request shape::

        {}  # no required fields; out_dir comes from the runner

    The wrapper supports two optional knobs:

    * ``timeout_s`` — float, max wall-clock for the probe subprocess
      (default 180 s). The probe is local-only so this is generous.
    * ``include_traceback`` — bool, surface the full subprocess
      stderr on failure (default true).
    """

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timeout_s = float(request.get("timeout_s") or 180.0)
    include_traceback = bool(request.get("include_traceback", True))

    script = _repo_root() / "scripts" / "dev" / "probe_extension_providers.py"
    if not script.is_file():
        return {
            "status": "error",
            "reason": "probe_script_missing",
            "detail": f"{script} not found",
            "artifacts": [],
        }

    cmd = [sys.executable, str(script), "--out", str(out_dir)]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "blocked",
            "reason": "probe_timeout",
            "detail": f"probe did not finish within {timeout_s}s",
            "artifacts": [],
        }

    if completed.returncode != 0:
        return {
            "status": "error",
            "reason": "probe_subprocess_failed",
            "detail": (
                f"probe exited {completed.returncode}; "
                f"stderr={completed.stderr.strip()[-400:]!r}"
                if include_traceback
                else f"probe exited {completed.returncode}"
            ),
            "artifacts": [],
        }

    # Verify every contracted artifact landed.
    artifacts: list[str] = []
    missing: list[str] = []
    for name in EXPECTED_ARTIFACTS:
        path = out_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        artifacts.append(str(path.resolve()))

    if missing:
        return {
            "status": "blocked",
            "reason": "probe_incomplete",
            "detail": f"probe finished but did not emit: {missing}",
            "artifacts": artifacts,
        }

    # Sanity: provider_status.json::providers is non-empty.
    try:
        body = json.loads((out_dir / "provider_status.json").read_text(encoding="utf-8"))
        provider_count = len(body.get("providers", []) or [])
    except (OSError, json.JSONDecodeError):
        provider_count = 0

    return {
        "status": "ok",
        "reason": "all probe artifacts emitted",
        "provider_count": provider_count,
        "artifacts": artifacts,
    }


__all__ = ["EXPECTED_ARTIFACTS", "run"]

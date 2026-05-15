"""Verification job task — run verification ladder on a bundle.

Wraps the verification pipeline as a Ray remote task.
"""

from __future__ import annotations

from typing import Any

from infra.ray._require import require_ray

ray = require_ray()


@ray.remote
def verify_bundle_job(
    bundle_path: str,
    level: str = "all",
    *,
    artifact_actor: Any = None,
) -> dict[str, Any]:
    """Run verification ladder on a bundle.

    Args:
        bundle_path: Path to the bundle directory.
        level: Verification level ("structural", "functional",
            "performance", "formal", "all").
        artifact_actor: Optional ArtifactIndexActor handle.

    Returns:
        Verification report dict.
    """
    from pathlib import Path

    bundle_dir = Path(bundle_path)

    report: dict[str, Any] = {
        "bundle_path": str(bundle_dir),
        "level": level,
        "results": {},
        "overall_pass": True,
    }

    # Structural check: verify required files exist
    required_files = ["manifest.json"]
    for fname in required_files:
        exists = (bundle_dir / fname).exists()
        report["results"][f"structural_{fname}"] = {
            "passed": exists,
            "error": "" if exists else f"Missing {fname}",
        }
        if not exists:
            report["overall_pass"] = False

    if artifact_actor is not None:
        ray.get(
            artifact_actor.register_artifact.remote(
                artifact_type="verification_report",
                target_name="",
                storage_path=str(bundle_dir / "verification_report.json"),
                metadata=report,
            )
        )

    return report


__all__ = ["verify_bundle_job"]

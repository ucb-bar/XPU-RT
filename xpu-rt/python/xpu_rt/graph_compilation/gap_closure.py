"""Gap Closure stage — drives materialize → fill → verify → register.

For each gap in ``04_gap_discovery/gap_action_queue.json`` whose
``fx_target`` is in :data:`agent_decomp_fill.KNOWN_FILLS`, this stage:

1. Materializes a workspace under ``.crg-artifacts/extensions/...``
2. Runs the deterministic agent fill (writes ``extension.py``)
3. Verifies (locked-files audit + 100 random differential trials)
4. Registers the extension if verify passes

Outputs under ``05_gap_closure/``:

- ``closure_summary.json`` — top-level status
- ``extensions_invoked.json`` — every workspace touched, with hashes
- ``gap_delta.json`` — placeholder for the rerun-and-re-discover loop
  (filled in by the higher-level closure driver)
- ``per_extension/<extension_id>.json`` — per-extension report

This module **does not** re-run Gap Discovery itself; that's the
caller's job and the gap_delta is filled in once the rerun completes.
``llm_calls == 0`` is enforced by the deterministic agent fill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from compgen.graph_compilation.agent_decomp_fill import (
    KNOWN_FILLS,
    UnknownTargetError,
    deterministic_fill,
)
from compgen.graph_compilation.artifacts import ArtifactRef, StageRecord
from compgen.graph_compilation.extension_materialize import materialize_extension
from compgen.graph_compilation.extension_registry import (
    register_extension,
)
from compgen.graph_compilation.extension_verify import VerifyResult, run_verify
from compgen.graph_compilation.hashing import sha256_file, sha256_tree


@dataclass
class ExtensionInvocation:
    gap_id: str
    fx_target: str
    extension_id: str
    extension_path: str
    materialize_status: str  # "pass" | "fail"
    fill_status: str  # "pass" | "skipped" | "fail"
    fill_error: str | None
    verify_status: str  # "pass" | "fail" | "skipped"
    verify_max_abs_error: float
    verify_max_rel_error: float
    register_status: str  # "pass" | "fail" | "skipped"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "fx_target": self.fx_target,
            "extension_id": self.extension_id,
            "extension_path": self.extension_path,
            "materialize_status": self.materialize_status,
            "fill_status": self.fill_status,
            "fill_error": self.fill_error,
            "verify_status": self.verify_status,
            "verify_max_abs_error": self.verify_max_abs_error,
            "verify_max_rel_error": self.verify_max_rel_error,
            "register_status": self.register_status,
        }


@dataclass
class ClosureResult:
    invocations: list[ExtensionInvocation] = field(default_factory=list)
    skipped_gaps: list[dict[str, Any]] = field(default_factory=list)


def run_gap_closure(
    run_dir: Path,
    *,
    extensions_root: Path,
    registry_path: Path,
    target_id: str,
    model_id: str,
) -> StageRecord:
    """Walk gaps, materialize+fill+verify+register the ones we can handle.

    Side effects:

    - May create / overwrite directories under ``extensions_root``.
    - May update ``registry_path`` (.yaml).
    - Always writes ``run_dir/05_gap_closure/*.json``.
    """
    from compgen.graph_compilation.artifacts import stage_dir

    started_at = _utcnow()
    run_dir = Path(run_dir).resolve()
    out_dir = run_dir / "05_gap_closure"
    out_dir.mkdir(parents=True, exist_ok=True)

    gd_dir = stage_dir(run_dir, "gap_discovery")
    queue_path = gd_dir / "gap_action_queue.json"
    if not queue_path.exists():
        raise FileNotFoundError(f"gap_action_queue.json missing: {queue_path}")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))

    closure = ClosureResult()
    seen_extensions: set[str] = set()  # idempotent: don't re-run for duplicate (target, shape) pairs

    for gap in queue.get("gaps", []):
        target = gap.get("fx_target", "")

        # ALWAYS materialize. Even for targets the deterministic agent
        # cannot fill, the workspace is the artifact a human or Claude
        # Code uses to author the extension — the README inside it
        # tells them what to do.
        try:
            mr = materialize_extension(gap, target_id=target_id, extensions_root=extensions_root)
        except Exception as exc:
            closure.invocations.append(
                ExtensionInvocation(
                    gap_id=gap.get("gap_id", "?"),
                    fx_target=target,
                    extension_id="",
                    extension_path="",
                    materialize_status="fail",
                    fill_status="skipped",
                    fill_error=f"{type(exc).__name__}: {exc}",
                    verify_status="skipped",
                    verify_max_abs_error=float("inf"),
                    verify_max_rel_error=float("inf"),
                    register_status="skipped",
                )
            )
            continue

        # Idempotent: same workspace already filled+registered? Skip
        # remaining same-target gaps. We still record the invocation so
        # the report shows we visited every gap.
        already_done = mr.extension_id in seen_extensions

        # Fill if we have a deterministic recipe; otherwise leave the
        # workspace as ``pending_human_fill`` so a human/Claude Code can
        # complete it. This is NOT a failure — the workspace is real and
        # ready to be edited.
        fill_status: str
        fill_error: str | None = None
        if already_done:
            fill_status = "skipped"
        elif target in KNOWN_FILLS:
            try:
                deterministic_fill(mr.extension_dir, target)
                fill_status = "pass"
            except UnknownTargetError as exc:
                fill_status = "fail"
                fill_error = str(exc)
        else:
            fill_status = "pending_human_fill"

        # Verify only if a fill (deterministic or human) was done.
        verify_status: str = "skipped"
        verify_result: VerifyResult | None = None
        if fill_status == "pass":
            verify_result = run_verify(mr.extension_dir)
            verify_status = verify_result.status
        elif fill_status == "pending_human_fill":
            # Workspace exists, stub raises NotImplementedError → verify
            # will fail. We *could* skip verify entirely, but running it
            # produces a results/verification.json with detail=
            # "extension has not filled" so the human knows the state.
            verify_result = run_verify(mr.extension_dir)
            # Don't promote a stub to "pass": even if (somehow) verify
            # claimed pass on a stub, gap_closure shouldn't register it
            # without a real fill.
            verify_status = "skipped"

        # Register if verify passed.
        register_status: str = "skipped"
        if verify_result is not None and fill_status == "pass" and verify_result.status == "pass":
            register_extension(
                workspace=mr.extension_dir,
                verification_result=verify_result,
                registry_path=registry_path,
            )
            register_status = "pass"

        seen_extensions.add(mr.extension_id)

        closure.invocations.append(
            ExtensionInvocation(
                gap_id=gap.get("gap_id", "?"),
                fx_target=target,
                extension_id=mr.extension_id,
                extension_path=str(mr.extension_dir),
                materialize_status="pass",
                fill_status=fill_status,
                fill_error=fill_error,
                verify_status=verify_status,
                verify_max_abs_error=verify_result.max_abs_error if verify_result else float("inf"),
                verify_max_rel_error=verify_result.max_rel_error if verify_result else float("inf"),
                register_status=register_status,
            )
        )

    # ------------------------------------------------------------------ #
    # Emit reports.
    # ------------------------------------------------------------------ #
    invocations_obj = {
        "schema_version": "extensions_invoked_v1",
        "extensions": [inv.to_dict() for inv in closure.invocations],
        "skipped_gaps": closure.skipped_gaps,
    }
    invocations_path = out_dir / "extensions_invoked.json"
    invocations_path.write_text(
        json.dumps(invocations_obj, indent=2, sort_keys=True), encoding="utf-8"
    )

    # gap_delta is a placeholder filled in by the higher-level driver
    # that re-runs gap-discovery. We record what we believe will be
    # closed (one per registered (target, kind)) so the validator has
    # something concrete to compare against later.
    closed_targets = {
        (inv.fx_target, "unsupported_op")
        for inv in closure.invocations
        if inv.register_status == "pass"
    }
    pre_closure_unsupported = [
        g for g in queue.get("gaps", [])
        if g.get("gap_kind") == "unsupported_op"
        and (g.get("fx_target"), "unsupported_op") in closed_targets
    ]
    gap_delta_obj = {
        "schema_version": "gap_delta_v1",
        "before": {"gaps_total": len(queue.get("gaps", []))},
        "expected_closed": len(pre_closure_unsupported),
        "closed_targets": sorted(t for t, _ in closed_targets),
        "after": None,  # filled by rerun-driver
        "delta": None,
    }
    gap_delta_path = out_dir / "gap_delta.json"
    gap_delta_path.write_text(json.dumps(gap_delta_obj, indent=2, sort_keys=True), encoding="utf-8")

    # Per-extension report copies (for ergonomic discovery alongside the run).
    per_ext_dir = out_dir / "per_extension"
    per_ext_dir.mkdir(parents=True, exist_ok=True)
    for inv in closure.invocations:
        if not inv.extension_id:
            continue
        per_ext_dir.joinpath(f"{inv.extension_id}.json").write_text(
            json.dumps(inv.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    pass_count = sum(1 for inv in closure.invocations if inv.register_status == "pass")
    pending_count = sum(1 for inv in closure.invocations if inv.fill_status == "pending_human_fill")
    fail_count = sum(
        1
        for inv in closure.invocations
        if inv.register_status == "skipped"
        and inv.fill_status not in ("pending_human_fill", "skipped")
        and (inv.materialize_status == "fail" or inv.fill_status == "fail" or inv.verify_status == "fail")
    )
    if pass_count == 0 and fail_count > 0:
        overall = "fail"
    elif pass_count > 0 and fail_count == 0:
        overall = "pass"
    elif pass_count == 0 and fail_count == 0 and not closure.invocations:
        # Empty queue.
        overall = "skipped"
    elif pass_count == 0 and fail_count == 0 and pending_count > 0:
        # All invocations are awaiting human fill — honest "ready for
        # Claude Code" state, not a failure.
        overall = "pending_human_fill"
    else:
        overall = "partial_success"

    pending_workspaces = sorted(
        {inv.extension_path for inv in closure.invocations if inv.fill_status == "pending_human_fill"}
    )

    summary_obj = {
        "schema_version": "closure_summary_v1",
        "stage_id": "gap_closure",
        "status": overall,
        "model_id": model_id,
        "target_id": target_id,
        "extensions_invoked_count": len(closure.invocations),
        "extensions_registered_count": pass_count,
        "extensions_pending_count": pending_count,
        "extensions_failed_count": fail_count,
        "skipped_gaps_count": len(closure.skipped_gaps),
        "pending_workspaces": pending_workspaces,
        "registry_path": str(Path(registry_path).resolve()),
        "extensions_root": str(Path(extensions_root).resolve()),
        "outputs": {
            "extensions_invoked": "05_gap_closure/extensions_invoked.json",
            "gap_delta": "05_gap_closure/gap_delta.json",
            "per_extension": "05_gap_closure/per_extension/",
        },
        "llm_calls": 0,
    }
    summary_path = out_dir / "closure_summary.json"
    summary_path.write_text(json.dumps(summary_obj, indent=2, sort_keys=True), encoding="utf-8")

    finished_at = _utcnow()
    output_hash = sha256_tree(out_dir)
    # Hash chain: input = sha256_tree(<gap_discovery dir>) — matches gap_discovery.output_hash.
    input_hash = sha256_tree(gd_dir)

    artifact_refs: list[ArtifactRef] = []
    for p in (summary_path, invocations_path, gap_delta_path):
        artifact_refs.append(
            ArtifactRef(
                path=p.relative_to(run_dir).as_posix(),
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                kind="file",
            )
        )
    for p in sorted(per_ext_dir.glob("*.json")):
        artifact_refs.append(
            ArtifactRef(
                path=p.relative_to(run_dir).as_posix(),
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                kind="file",
            )
        )

    # Manifest contract is {pass, fail, skipped}; partial_success and
    # skipped (no invocations) both map to "pass" at the manifest level
    # because the stage executed cleanly — the per-stage report carries
    # the richer status.
    manifest_status = (
        "pass"
        if overall in {"pass", "partial_success", "skipped", "pending_human_fill"}
        else "fail"
    )

    return StageRecord(
        stage_id="gap_closure",
        status=manifest_status,
        inputs=(
            ArtifactRef(
                path=queue_path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(queue_path),
                size_bytes=queue_path.stat().st_size,
                kind="file",
            ),
        ),
        outputs=tuple(artifact_refs),
        report_path="05_gap_closure/closure_summary.json",
        input_hash=input_hash,
        output_hash=output_hash,
        llm_calls=0,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
    )


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

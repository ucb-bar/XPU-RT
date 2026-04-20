"""Cross-session graduation loop.

Bridges the existing scan-only pattern_graduation infrastructure to
*live* registry mutation. Called once at session start (from
:func:`compgen.llm.registry.get_registry` via the
:mod:`~compgen.agent.invent_slots.registrar`):

1. Scan ``~/.compgen/transcripts/**/tools.jsonl`` for accepted invent
   proposals.
2. Aggregate by ``(slot_name, chosen_signature)`` and apply the
   workload+target thresholds (see :mod:`compgen.promotion.pattern_graduation`).
3. For each granted request, register a new :class:`Tool` whose
   ``impl`` re-emits the chosen exemplar (until a hand-written pass
   replaces it).
4. Idempotence: state file
   ``~/.compgen/transcripts/_graduations.json`` records which patterns
   have already graduated; reruns are no-ops.

Failure mode: any exception is logged and swallowed — graduation must
never break registry initialisation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from compgen.promotion.pattern_graduation import (
    PatternIdentity,
    PatternPromotionRequest,
    _chosen_signature,
    _is_accepted_invent,
    _parse_entry,
    build_promotion_requests,
    scan_transcripts,
)

if TYPE_CHECKING:  # pragma: no cover
    from compgen.llm.registry import Registry

log = structlog.get_logger()


DEFAULT_TRANSCRIPTS_ROOT = Path("~/.compgen/transcripts").expanduser()
GRADUATION_STATE_FILE = "_graduations.json"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraduationResult:
    """One request that was applied to the live registry."""

    slot_name: str
    chosen_signature: str
    tool_name: str
    workloads_proven: tuple[str, ...]
    targets_proven: tuple[str, ...]
    acceptance_count: int


@dataclass
class CrossSessionGraduationReport:
    """Aggregate of one :func:`promote_pending_graduations` call."""

    transcripts_scanned: int = 0
    requests_found: int = 0
    requests_already_applied: int = 0
    new_tools_registered: list[GraduationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_path(root: Path) -> Path:
    return root / GRADUATION_STATE_FILE


def _load_state(root: Path) -> dict[str, Any]:
    path = _state_path(root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _save_state(root: Path, state: dict[str, Any]) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        _state_path(root).write_text(json.dumps(state, indent=2, default=str))
    except OSError:
        pass


def _request_key(req: PatternPromotionRequest) -> str:
    return f"{req.identity.slot_name}@{req.identity.chosen_signature}"


def _graduated_tool_name(req: PatternPromotionRequest) -> str:
    """Stable name for a graduated tool: ``<slot_name>__graduated``."""
    return f"{req.identity.slot_name}__graduated"


def _make_graduated_tool(req: PatternPromotionRequest):
    """Build a :class:`Tool` whose impl re-emits ``chosen_exemplar``.

    This is intentionally minimal — the graduated tool acts as a
    cached "best-known answer" that the LLM can invoke directly,
    bypassing the full propose-and-gate cycle. A future hand-written
    pass replaces ``impl`` to do real work.
    """
    from compgen.llm.registry import Tool, ToolArg, ToolResult

    chosen = dict(req.chosen_exemplar)

    def _impl(**kwargs: Any) -> dict[str, Any]:
        return {
            "status": "graduated",
            "chosen": chosen,
            "tool_name": _graduated_tool_name(req),
            "graduation": {
                "acceptance_count": req.acceptance_count,
                "workloads_proven": sorted(req.workloads_proven),
                "targets_proven": sorted(req.targets_proven),
            },
            "kwargs": kwargs,
        }

    return Tool(
        name=_graduated_tool_name(req),
        # Graduate into phase 3 by convention (most invent slots are 3-5).
        # Future: encode the source slot's phase in the request.
        phase=3,
        kind="tool",
        wraps_pass=f"graduated_from:{req.identity.slot_name}",
        autocomp_cost_impact="medium",
        args=(ToolArg(name="ctx", dtype="dict", description="Optional context", required=False),),
        result=ToolResult(dtype="dict", description="The cached chosen exemplar plus graduation provenance"),
        description=(
            f"Auto-graduated from invent slot {req.identity.slot_name!r} "
            f"after {req.acceptance_count} acceptances across "
            f"{len(req.workloads_proven)} workloads and "
            f"{len(req.targets_proven)} targets."
        ),
        impl=_impl,
        notes=f"chosen_signature={req.identity.chosen_signature}",
        stub=False,
    )


def _all_tools_jsonl(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("tools.jsonl"))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def promote_pending_graduations(
    registry: Registry,
    *,
    transcripts_root: Path | str | None = None,
    min_workloads: int = 2,
    min_targets: int = 2,
) -> CrossSessionGraduationReport:
    """Scan transcripts, materialise pending graduations into ``registry``.

    Idempotent: a state file under ``transcripts_root`` records which
    requests have already produced a tool entry; subsequent calls are
    no-ops for those requests.

    Args:
        registry: The live registry to mutate.
        transcripts_root: Directory holding ``tools.jsonl`` files. Defaults
            to ``~/.compgen/transcripts`` (matches LLMDrivenCompiler's
            recorder layout).
        min_workloads: Cross-workload threshold for graduation (default 2).
        min_targets: Cross-target threshold for graduation (default 2).

    Returns:
        :class:`CrossSessionGraduationReport`.
    """
    report = CrossSessionGraduationReport()
    root = Path(transcripts_root).expanduser() if transcripts_root is not None else DEFAULT_TRANSCRIPTS_ROOT

    transcripts = _all_tools_jsonl(root)
    report.transcripts_scanned = len(transcripts)
    if not transcripts:
        return report

    try:
        appearances = scan_transcripts(transcripts)
        # Build a transcripts_by_identity map so ``chosen_exemplar`` is
        # populated on each promotion request. The upstream
        # ``graduate_from_transcripts`` convenience helper doesn't
        # thread this, which is why we call the two primitives
        # directly here.
        by_identity: dict[PatternIdentity, list[dict[str, Any]]] = {}
        for tpath in transcripts:
            try:
                lines = tpath.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                entry = _parse_entry(line.strip())
                if entry is None or not _is_accepted_invent(entry):
                    continue
                result = entry.get("result") or {}
                args = entry.get("args") or {}
                chosen = result.get("chosen")
                if chosen is None:
                    chosen = args.get("chosen")
                if not isinstance(chosen, dict):
                    chosen = {}
                identity = PatternIdentity(
                    slot_name=entry.get("name", "<unknown>"),
                    target_feature_justification=(
                        args.get("target_feature_justification") or result.get("target_feature_justification") or ""
                    ),
                    chosen_signature=_chosen_signature(chosen),
                )
                by_identity.setdefault(identity, []).append(entry)
        requests = build_promotion_requests(
            appearances,
            min_workloads=min_workloads,
            min_targets=min_targets,
            transcripts_by_identity=by_identity,
        )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"graduation scan/build failed: {exc}")
        return report

    report.requests_found = len(requests)
    if not requests:
        return report

    state = _load_state(root)
    applied: dict[str, Any] = state.setdefault("applied_requests", {})

    for req in requests:
        key = _request_key(req)
        if key in applied:
            report.requests_already_applied += 1
            continue
        try:
            tool = _make_graduated_tool(req)
            registry.register_tool(tool)
            applied[key] = {
                "tool_name": tool.name,
                "acceptance_count": req.acceptance_count,
                "workloads_proven": sorted(req.workloads_proven),
                "targets_proven": sorted(req.targets_proven),
            }
            report.new_tools_registered.append(
                GraduationResult(
                    slot_name=req.identity.slot_name,
                    chosen_signature=req.identity.chosen_signature,
                    tool_name=tool.name,
                    workloads_proven=tuple(sorted(req.workloads_proven)),
                    targets_proven=tuple(sorted(req.targets_proven)),
                    acceptance_count=req.acceptance_count,
                )
            )
            log.info(
                "cross_session.graduated",
                slot=req.identity.slot_name,
                tool=tool.name,
                acceptance_count=req.acceptance_count,
            )
        except Exception as exc:  # noqa: BLE001
            # Swallow — never break registry init. Most likely a
            # ValueError from registry double-registration when state
            # file got out of sync.
            report.errors.append(f"register_tool({_graduated_tool_name(req)}): {type(exc).__name__}: {exc}")

    if report.new_tools_registered:
        _save_state(root, state)
    return report


def report_to_dict(report: CrossSessionGraduationReport) -> dict[str, Any]:
    return {
        "transcripts_scanned": report.transcripts_scanned,
        "requests_found": report.requests_found,
        "requests_already_applied": report.requests_already_applied,
        "new_tools_registered": [asdict(r) for r in report.new_tools_registered],
        "errors": list(report.errors),
    }


__all__ = [
    "CrossSessionGraduationReport",
    "DEFAULT_TRANSCRIPTS_ROOT",
    "GRADUATION_STATE_FILE",
    "GraduationResult",
    "promote_pending_graduations",
    "report_to_dict",
]

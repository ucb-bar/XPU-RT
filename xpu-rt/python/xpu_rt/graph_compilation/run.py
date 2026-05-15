"""Top-level orchestrator for ``graph_compilation run``.

For graph_capture stage, the supported pipeline ends after Stage 0:

::

    PyTorch model + inputs
        ↓
    graph_capture stage  (00_graph_capture/)
        ↓
    write run_manifest.json + stage_ledger.jsonl

Later graph compilation tasks will append Stage 1 (lower) and Stage 2
(gap analyze) to the same artifact contract; the runner already
threads input/output hashes through the manifest so those
extensions are additive.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from compgen.graph_compilation.artifacts import RunManifest, StageRecord
from compgen.graph_compilation.capture import ModelConfig, TargetConfig, run_graph_capture

# Stop-after points exposed at the CLI. Each value names the *last* stage to
# execute; descriptive names matching the on-disk directory layout.
SUPPORTED_STOP_AFTER: tuple[str, ...] = (
    "graph-capture",
    "payload-lowering",
    "graph-analysis",
    "recipe-planning",
    "recipe-verification",
    "recipe-lowering",
    "post-lowering-verification",
    "differential-verification",
    "real-transform-eligibility",
    "real-set-tile-transform",
    "real-transform-differential",
    "cost-preview-v2",
    "agent-decision-request",
    "kernel-specialization-request",
    "kernel-codegen-request",
    "kernel-auction",
    "execution-plan-emit",
    "glue-emit", "glue-differential",
    "glue-differential",
    "gap-discovery",
    "gap-closure",
)


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    manifest_path: Path
    ledger_path: Path
    stages: tuple[StageRecord, ...]


def _git_commit_or_none(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out if out and len(out) == 40 and all(c in "0123456789abcdef" for c in out) else None


def _utcnow() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit_closure_proof_reports(run_dir: Path, model_id: str, target_id: str) -> None:
    """Write the spec-06 closure-proof reports at the top of ``run_dir``.

    Pulls everything it needs out of the gap_discovery dir (which has just
    been written by gap_discovery with a registry passed in):

    - ``coverage_delta.json`` — before/after counts per closed target;
      ``before = closed_by_registry_count + remaining_gaps`` because the
      registry filter ran *during* this gap_discovery and removed those
      targets from the queue.
    - ``closure_report.json`` — top-level pass/fail with the
      ``extensions_used`` list a downstream auditor can pin against.
    """
    from compgen.graph_compilation.artifacts import stage_dir

    gd_dir = stage_dir(run_dir, "gap_discovery")
    summary_path = gd_dir / "gap_discovery_summary.json"
    queue_path = gd_dir / "gap_action_queue.json"
    pl_diag_path = run_dir / "01_payload_lowering" / "lowering_diagnostics.json"
    if not summary_path.exists() or not queue_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))

    closed_targets = summary.get("closed_targets", [])
    closed_by_registry_count = int(summary.get("totals", {}).get("closed_by_registry_count", 0))
    remaining_gap_count = int(queue.get("summary", {}).get("count", 0))

    # Aggregate the gap-discovery-side closures.
    closed_by_target: dict[str, int] = {}
    for c in closed_targets:
        t = c.get("fx_target", "")
        closed_by_target[t] = closed_by_target.get(t, 0) + 1
    extensions_used: set[str] = {
        c["extension_id"] for c in closed_targets if c.get("extension_id")
    }

    # Closures can also happen at IR-level (payload_substitution rewrites the
    # FX graph so the opaque op never reaches gap_discovery). Pull those out
    # of lowering_diagnostics.json so the closure report attributes them too.
    ir_level_closures: list[dict[str, str]] = []
    if pl_diag_path.exists():
        try:
            pl_diag = json.loads(pl_diag_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pl_diag = {"diagnostics": []}
        for d in pl_diag.get("diagnostics", []):
            msg = d.get("message", "")
            if not msg.startswith("Inlined extension "):
                continue
            # message format: "Inlined extension '<extension_id>' for '<fx_target>' (registry-driven)"
            try:
                ext_id = msg.split("'")[1]
                fx_target = msg.split("'")[3]
            except IndexError:
                continue
            ir_level_closures.append(
                {
                    "module_id": d.get("module_id", ""),
                    "extension_id": ext_id,
                    "fx_target": fx_target,
                }
            )
            extensions_used.add(ext_id)
            closed_by_target[fx_target] = closed_by_target.get(fx_target, 0) + 1

    ir_level_closure_count = len(ir_level_closures)
    total_closed = closed_by_registry_count + ir_level_closure_count
    extensions_used_sorted = sorted(extensions_used)

    coverage_delta = {
        "schema_version": "coverage_delta_v1",
        "model_id": model_id,
        "target_id": target_id,
        "extension_registry": summary.get("extension_registry"),
        "before": {
            "unsupported_op_total_for_closed_targets": total_closed,
            "by_target": closed_by_target,
            "comment": (
                "Counts what would have been opaque/unsupported without the "
                "registry. Sum of (a) IR-level inlining at payload-lowering "
                "and (b) gap-discovery-level filtering."
            ),
        },
        "after": {
            "unsupported_op_total_for_closed_targets": 0,
            "remaining_total_in_queue": remaining_gap_count,
        },
        "delta": {
            "closed_count": total_closed,
            "closed_at_ir_level": ir_level_closure_count,
            "closed_at_gap_level": closed_by_registry_count,
            "closed_targets": sorted(closed_by_target.keys()),
            "extensions_used": extensions_used_sorted,
        },
    }
    (run_dir / "coverage_delta.json").write_text(
        json.dumps(coverage_delta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    closure_report = {
        "schema_version": "closure_report_v1",
        "status": "pass" if total_closed >= 1 else "no_closure",
        "model_id": model_id,
        "target_id": target_id,
        "extension_registry": summary.get("extension_registry"),
        "extensions_used": extensions_used_sorted,
        "closed_targets": sorted(closed_by_target.keys()),
        "closed_count": total_closed,
        "closed_at_ir_level": ir_level_closure_count,
        "closed_at_gap_level": closed_by_registry_count,
        "ir_level_closures": ir_level_closures,
        "remaining_gap_count": remaining_gap_count,
        "llm_calls": 0,
    }
    (run_dir / "closure_report.json").write_text(
        json.dumps(closure_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_graph_analysis_stage(
    run_dir: Path, *, target_yaml_path: Path | None = None
) -> StageRecord:
    """Build ``02_graph_analysis/`` artifacts and return a typed ``StageRecord``.

    Reads from ``00_graph_capture/`` + ``01_payload_lowering/``; emits

    - region_map / tensor_use_def_graph / region_graph / graph_analysis_report (B)
    - graph_analysis.mlir / graph_dossier_v2.json /
      region_dossiers/<id>.json / dossier_validation.json (03)

    ``target_yaml_path`` provides the cost model (peak_compute, peak_bandwidth,
    memory tiers, numerical budgets). When omitted, the default host_cpu
    profile is used.
    """
    from compgen.graph_compilation.action_space import build_action_space
    from compgen.graph_compilation.artifacts import ArtifactRef
    from compgen.graph_compilation.hashing import sha256_file, sha256_tree
    from compgen.graph_compilation.region_dossier import build_region_dossiers
    from compgen.graph_compilation.region_map import build_graph_analysis

    if target_yaml_path is None:
        repo_root = Path(__file__).resolve().parents[3]
        target_yaml_path = repo_root / "configs" / "targets" / "host_cpu.yaml"

    started = _utcnow()
    ga = build_graph_analysis(run_dir)
    build_region_dossiers(run_dir, target_yaml_path)
    build_action_space(run_dir, target_yaml_path)
    finished = _utcnow()

    lowering_dir = run_dir / "01_payload_lowering"
    out_dir = run_dir / "02_graph_analysis"

    # Stage outputs.
    outputs: list[ArtifactRef] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        outputs.append(
            ArtifactRef(
                path=path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                kind="file",
            )
        )

    # Inputs hash chain: graph_analysis consumes both graph_capture and
    # payload_lowering trees, so input_hash = sha256_tree(payload_lowering).
    # That keeps the chain monotonic (R009) since payload_lowering's
    # output_hash == sha256_tree(payload_lowering) by construction.
    inputs: list[ArtifactRef] = []
    for path in sorted(lowering_dir.rglob("*")):
        if not path.is_file():
            continue
        inputs.append(
            ArtifactRef(
                path=path.relative_to(run_dir).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                kind="file",
            )
        )

    return StageRecord(
        stage_id="graph_analysis",
        status="pass",
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        report_path=ga.graph_analysis_report_path.relative_to(run_dir).as_posix(),
        input_hash=sha256_tree(lowering_dir),
        output_hash=sha256_tree(out_dir),
        llm_calls=0,
        started_at_utc=started,
        finished_at_utc=finished,
    )


def _append_ledger(ledger_path: Path, *, stage_id: str, event: str, note: str | None = None) -> None:
    obj = {
        "schema_version": "stage_event_v1",
        "stage_id": stage_id,
        "event": event,
        "artifact_path": None,
        "sha256": None,
        "timestamp_utc": _utcnow(),
        "note": note,
    }
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def run_graph_compilation(
    model_config_path: Path,
    target_config_path: Path,
    out_dir: Path,
    *,
    stop_after: str = "graph-capture",
    run_id: str | None = None,
    repo_root: Path | None = None,
    extension_registry: Path | None = None,
    extensions_root: Path | None = None,
    selection_mode: str = "greedy",
    rationale_primary: str | None = None,
    agent_decision_response_path: Path | None = None,
    agent_decision_response_paths: list[Path] | None = None,
    agent_max_retries: int = 3,
    live_provider_config: object | None = None,
    resume_from: str | None = None,
    auction_mode: str = "multi-bidder",
    bid_cutoff: int = 3,
    kernel_coverage_mode: str = "both",
) -> RunResult:
    """Materialise a graph compilation run directory rooted at ``out_dir``.

    If ``out_dir`` exists it is replaced (the run directory is always
    produced fresh and deterministically).

    when ``resume_from == "kernel-codegen-response"``, an existing
    ``out_dir`` is preserved and the pipeline skips every stage up to
    and including the kernel-codegen-request emission. This
    is the only path that lets the agentic provider chain run from CLI
    after the operator has committed a response (the response, attempt
    trail, and certificates would otherwise be wiped at the top of
    every fresh run). Resuming requires that ``04_kernel_codegen/
    requests/`` and ``04_kernel_codegen/contracts/`` already exist;
    otherwise a typed error fires.
    """
    # Backward-compat: old code passed ``stop_after="capture"``.
    if stop_after == "capture":
        stop_after = "graph-capture"
    if stop_after not in SUPPORTED_STOP_AFTER:
        raise ValueError(
            f"--stop-after={stop_after!r} not supported; "
            f"supported: {SUPPORTED_STOP_AFTER}"
        )

    _SUPPORTED_RESUME_FROM = (None, "kernel-codegen-response")
    if resume_from not in _SUPPORTED_RESUME_FROM:
        raise ValueError(
            f"--resume-from={resume_from!r} not supported; "
            f"supported: {[r for r in _SUPPORTED_RESUME_FROM if r]}"
        )
    _resume_skip_early = resume_from == "kernel-codegen-response"

    out_dir = Path(out_dir).resolve()
    # COMPGEN_FORCE_REBUILD=1 inverts the default. By default
    # run_graph_compilation rm-rfs out_dir for convenience; under
    # COMPGEN_FORCE_REBUILD we refuse to silently destroy data and
    # require the operator to clean up first. This is the audit-mode
    # safety check: an audit run that overwrote a prior run would
    # confuse cold/warm reasoning.
    # resume mode preserves out_dir; the operator-committed
    # response, attempts trail, and certificates survive across the
    # pipeline-restart boundary.
    _force_rebuild = os.environ.get("COMPGEN_FORCE_REBUILD") == "1"
    if out_dir.exists():
        if _force_rebuild and any(out_dir.iterdir()):
            from compgen.audit.errors import AuditError
            raise AuditError(
                f"COMPGEN_FORCE_REBUILD=1 set, but {out_dir} is non-empty. "
                "Either rm -rf the directory first, or unset the env var."
            )
        if not _force_rebuild and not _resume_skip_early:
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _resume_skip_early:
        _resume_requests_dir = out_dir / "04_kernel_codegen" / "requests"
        _resume_contracts_dir = out_dir / "04_kernel_codegen" / "contracts"
        if not _resume_requests_dir.exists() or not any(
            _resume_requests_dir.glob("*.request.json")
        ):
            raise ValueError(
                f"--resume-from=kernel-codegen-response: "
                f"{_resume_requests_dir.relative_to(out_dir)} has no "
                f"committed requests; run --stop-after "
                f"kernel-codegen-request first to emit them"
            )
        if not _resume_contracts_dir.exists() or not any(
            _resume_contracts_dir.glob("*.json")
        ):
            raise ValueError(
                f"--resume-from=kernel-codegen-response: "
                f"{_resume_contracts_dir.relative_to(out_dir)} has no "
                f"materialized contracts; the prior run did not reach M-40"
            )

    model_cfg = ModelConfig.load(Path(model_config_path))
    target_cfg = TargetConfig.load(Path(target_config_path))
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    if run_id is None:
        run_id = f"graphcomp_{model_cfg.model_id}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"

    # snapshot sys.modules at the top of the run so the post-run
    # diff names exactly the modules this run loaded. The snapshot is
    # cheap (one sorted list comprehension) and gives every run an
    # auditable import provenance.
    from compgen.audit.import_provenance import ImportSnapshot
    _import_snapshot_before = ImportSnapshot.take("before")

    ledger_path = out_dir / "stage_ledger.jsonl"
    if not _resume_skip_early:
        ledger_path.write_text("", encoding="utf-8")
    elif not ledger_path.exists():
        ledger_path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Graph Capture (skipped resume mode — prior run produced it)
    # ------------------------------------------------------------------ #
    if _resume_skip_early:
        _append_ledger(
            ledger_path, stage_id="graph_capture", event="start",
            note="M-53 resume: skipped (prior run's artifacts preserved)",
        )
        _append_ledger(ledger_path, stage_id="graph_capture", event="finish")
        stages: list[StageRecord] = []
    else:
        _append_ledger(ledger_path, stage_id="graph_capture", event="start")
        capture_stage = run_graph_capture(model_cfg, target_cfg, out_dir)
        for ref in capture_stage.outputs:
            _append_ledger(
                ledger_path,
                stage_id="graph_capture",
                event="artifact_written",
                note=f"path={ref.path} sha={ref.sha256[:12]}",
            )
        _append_ledger(ledger_path, stage_id="graph_capture", event="finish")
        stages = [capture_stage]

    # ------------------------------------------------------------------ #
    # Payload Lowering (when stop_after >= payload-lowering)
    # ------------------------------------------------------------------ #
    needs_lowering = stop_after in (
        "payload-lowering", "graph-analysis",
        "recipe-planning", "recipe-verification", "recipe-lowering",
        "post-lowering-verification", "differential-verification",
        "real-transform-eligibility", "real-set-tile-transform", "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential",
        "gap-discovery", "gap-closure",
    ) and not _resume_skip_early
    if needs_lowering:
        from compgen.graph_compilation.lower import run_payload_lowering

        _append_ledger(ledger_path, stage_id="payload_lowering", event="start")
        lowering_stage, _module_results = run_payload_lowering(
            out_dir,
            target_id=target_cfg.target_id,
            extension_registry=extension_registry,
        )
        for ref in lowering_stage.outputs:
            _append_ledger(
                ledger_path,
                stage_id="payload_lowering",
                event="artifact_written",
                note=f"path={ref.path} sha={ref.sha256[:12]}",
            )
        # strict-gate report. Read-only aggregator that turns
        # the lowering_summary status + silent-drop counts into a typed
        # <model_id>_strict_gate_report.json (+ summary.md). Always
        # emitted; ``status=pass`` for clean models, ``status=blocked``
        # with a typed root_cause for models like merlin_dronet whose
        # FX→Payload importer drops nodes lacking tensor_meta.
        # The strict-gate report writes new files into 01_payload_lowering/
        # AFTER ``run_payload_lowering`` has already computed
        # ``lowering_stage.output_hash``. Without rebuild, the
        # 01_payload_lowering/ tree at the time graph_analysis computes
        # its input_hash will not match — R009 hash-chain check fails.
        # Fix: after strict_gate writes, recompute output_hash via the
        # same ``sha256_tree(01_payload_lowering/)`` and replace the
        # stage record so the chain stays monotonic.
        from compgen.graph_compilation.strict_gate_report import (
            build_strict_gate_report,
        )
        import dataclasses as _dataclasses
        from compgen.graph_compilation.hashing import sha256_tree as _sha256_tree

        try:
            sg_result = build_strict_gate_report(out_dir)
            _append_ledger(
                ledger_path, stage_id="payload_lowering",
                event="artifact_written",
                note=(
                    f"strict_gate_report (M-16.1): {sg_result.status} "
                    f"[{sg_result.root_cause_category}]"
                ),
            )
            # Rebuild lowering_stage.output_hash to include the
            # strict-gate report files (R009 monotonicity).
            _new_output_hash = _sha256_tree(out_dir / "01_payload_lowering")
            lowering_stage = _dataclasses.replace(
                lowering_stage, output_hash=_new_output_hash,
            )
        except Exception as exc:  # noqa: BLE001 - never fail the pipeline on this
            _append_ledger(
                ledger_path, stage_id="payload_lowering",
                event="artifact_written",
                note=(
                    f"strict_gate_report (M-16.1): error "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
        _append_ledger(ledger_path, stage_id="payload_lowering", event="finish")
        stages.append(lowering_stage)

    # ------------------------------------------------------------------ #
    # Graph Analysis (when stop_after >= graph-analysis)
    # ------------------------------------------------------------------ #
    needs_graph_analysis = stop_after in (
        "graph-analysis", "recipe-planning", "recipe-verification",
        "recipe-lowering", "post-lowering-verification",
        "differential-verification", "real-transform-eligibility",
        "real-set-tile-transform", "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential",
        "gap-discovery", "gap-closure",
    ) and not _resume_skip_early
    if needs_graph_analysis:
        _append_ledger(ledger_path, stage_id="graph_analysis", event="start")
        ga_stage = run_graph_analysis_stage(
            out_dir, target_yaml_path=target_cfg.raw_path
        )
        for ref in ga_stage.outputs:
            _append_ledger(
                ledger_path,
                stage_id="graph_analysis",
                event="artifact_written",
                note=f"path={ref.path} sha={ref.sha256[:12]}",
            )
        _append_ledger(ledger_path, stage_id="graph_analysis", event="finish")
        stages.append(ga_stage)

    # ------------------------------------------------------------------ #
    # Recipe Planning (when stop_after >= recipe-planning)
    # ------------------------------------------------------------------ #
    needs_recipe_planning = stop_after in (
        "recipe-planning", "recipe-verification", "recipe-lowering",
        "post-lowering-verification", "differential-verification",
        "real-transform-eligibility", "real-set-tile-transform", "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential",
        "gap-discovery", "gap-closure",
    ) and not _resume_skip_early
    if needs_recipe_planning:
        from compgen.graph_compilation.recipe_planning import (
            run_recipe_planning,
        )
        from compgen.graph_compilation.recipe_planning import (
            stage_record as recipe_stage_record,
        )

        _append_ledger(ledger_path, stage_id="recipe_planning", event="start")
        run_recipe_planning(
            out_dir,
            selection_mode=selection_mode,
            rationale_primary=rationale_primary,
            agent_decision_response_path=agent_decision_response_path,
            agent_decision_response_paths=agent_decision_response_paths,
            agent_max_retries=agent_max_retries,
            live_provider_config=live_provider_config,
        )
        # verification gate runs as a sub-step of recipe_planning
        # when stop_after >= recipe-verification. Both
        # artifacts land in 03_recipe_planning/; the stage's output_hash
        # covers all of them.
        if stop_after in (
            "recipe-verification", "recipe-lowering",
            "post-lowering-verification", "differential-verification",
            "real-transform-eligibility", "real-set-tile-transform", "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential",
            "gap-discovery", "gap-closure",
        ):
            from compgen.graph_compilation.recipe_gate import run_recipe_gate

            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written", note="recipe_gate (M-06): start",
            )
            run_recipe_gate(out_dir, target_yaml_path=target_cfg.raw_path)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written", note="recipe_gate (M-06): finish",
            )
        # lowering runs as a sub-step of recipe_planning when
        # stop_after >= recipe-lowering. Same directory; no payload mutation.
        if stop_after in (
            "recipe-lowering", "post-lowering-verification",
            "differential-verification", "real-transform-eligibility",
            "real-set-tile-transform", "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential",
            "gap-discovery", "gap-closure",
        ):
            from compgen.graph_compilation.recipe_lowering import (
                run_recipe_lowering,
            )

            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written", note="recipe_lowering (M-07): start",
            )
            run_recipe_lowering(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written", note="recipe_lowering (M-07): finish",
            )
        # post-lowering verification runs as a sub-step when
        # stop_after >= post-lowering-verification. Applies metadata
        # transforms to a copy of payload.mlir; the source remains
        # byte-identical.
        if stop_after in (
            "post-lowering-verification", "differential-verification",
            "real-transform-eligibility", "real-set-tile-transform", "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential",
            "gap-discovery", "gap-closure"
        ):
            from compgen.graph_compilation.post_lowering import (
                run_post_lowering_verification,
            )

            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written", note="post_lowering (M-08): start",
            )
            run_post_lowering_verification(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written", note="post_lowering (M-08): finish",
            )
            # emit a verification certificate wrapping the
            # post-lowering report so the agent's pass-card
            # ``verification: [structural]`` rung is checkable.
            try:
                from compgen.passes.verification import (
                    emit_certificate_from_post_lowering_report,
                )

                _cert = emit_certificate_from_post_lowering_report(
                    run_dir=out_dir,
                )
                _append_ledger(
                    ledger_path, stage_id="trust_audit",
                    event="artifact_written",
                    note=(
                        f"verification_certificate (structural): "
                        f"{_cert.status if _cert else 'not_emitted'}"
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - best effort
                _append_ledger(
                    ledger_path, stage_id="trust_audit",
                    event="artifact_written",
                    note=f"structural cert error {type(exc).__name__}: {exc}",
                )
        # differential / reference verification runs as a sub-step
        # when stop_after >= differential-verification. Strips compgen
        # metadata and proves the transformation is semantically
        # inert; re-checks Stage-0 goldens; validates contract drafts.
        if stop_after in (
            "differential-verification", "real-transform-eligibility",
            "real-set-tile-transform", "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential",
            "gap-discovery", "gap-closure"
        ):
            from compgen.graph_compilation.differential_verification import (
                run_differential_verification,
            )

            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="differential_verification (M-09): start",
            )
            run_differential_verification(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="differential_verification (M-09): finish",
            )
            # differential certificate wrapper.
            try:
                from compgen.passes.verification import (
                    emit_certificate_from_differential_report,
                )

                _cert = emit_certificate_from_differential_report(
                    run_dir=out_dir,
                )
                _append_ledger(
                    ledger_path, stage_id="trust_audit",
                    event="artifact_written",
                    note=(
                        f"verification_certificate (differential): "
                        f"{_cert.status if _cert else 'not_emitted'}"
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - best effort
                _append_ledger(
                    ledger_path, stage_id="trust_audit",
                    event="artifact_written",
                    note=f"differential cert error {type(exc).__name__}: {exc}",
                )
        # real-transform eligibility audit: read-only audit that
        # classifies the selected recipe against the narrow real-matmul-
        # tiling MVP. Emits 03_recipe_planning/real_transform_eligibility
        # .json + .md. No payload mutation; no transformed real artifact.
        if stop_after in (
            "real-transform-eligibility", "real-set-tile-transform", "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential",
            "gap-discovery", "gap-closure"
        ):
            from compgen.graph_compilation.real_transform_eligibility import (
                run_real_transform_eligibility,
            )

            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="real_transform_eligibility (M-11A): start",
            )
            run_real_transform_eligibility(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="real_transform_eligibility (M-11A): finish",
            )
        # real SetTileParams transform MVP: emits a tiled
        # transformed_payload.real.mlir for eligible matmuls.
        if stop_after in (
            "real-set-tile-transform", "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential", "gap-discovery", "gap-closure"
        ):
            from compgen.graph_compilation.real_lowering import (
                run_real_lowering,
            )

            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="real_lowering (M-11B): start",
            )
            run_real_lowering(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="real_lowering (M-11B): finish",
            )
            # real fusion lowering. Fires only when the committed
            # candidate is FuseProducerConsumer (otherwise it returns
            # None and is a no-op). Sibling stage to the two
            # candidate kinds are mutually exclusive in the
            # single-candidate MVP.
            from compgen.graph_compilation.real_fusion import (
                run_real_fusion_lowering,
            )

            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="real_fusion_lowering (M-16.2): start",
            )
            _fusion_lowering = run_real_fusion_lowering(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note=(
                    f"real_fusion_lowering (M-16.2): finish "
                    f"({_fusion_lowering.mode if _fusion_lowering else 'no-op'})"
                ),
            )
        # real-transform differential harness: discharges
        # real_transform_differential_check via Path A executable
        # evaluator (eligible cases) or emits a blocked report (Path B).
        if stop_after in (
            "real-transform-differential", "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential", "gap-discovery", "gap-closure"
        ):
            from compgen.graph_compilation.real_transform_differential import (
                run_real_transform_differential,
            )

            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="real_transform_differential (M-12): start",
            )
            run_real_transform_differential(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="real_transform_differential (M-12): finish",
            )
            # real fusion differential. Sibling to ; reads
            # real_fusion_manifest.json and emits
            # real_fusion_differential_report.json. No-op when the
            # committed candidate is not FuseProducerConsumer.
            from compgen.graph_compilation.real_fusion import (
                run_real_fusion_differential,
            )

            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note="real_fusion_differential (M-16.2): start",
            )
            _fusion_diff = run_real_fusion_differential(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note=(
                    f"real_fusion_differential (M-16.2): finish "
                    f"({_fusion_diff.overall if _fusion_diff else 'no-op'})"
                ),
            )
        rp_stage = recipe_stage_record(out_dir, selection_mode=selection_mode)
        for ref in rp_stage.outputs:
            _append_ledger(
                ledger_path,
                stage_id="recipe_planning",
                event="artifact_written",
                note=f"path={ref.path} sha={ref.sha256[:12]}",
            )
        _append_ledger(ledger_path, stage_id="recipe_planning", event="finish")
        stages.append(rp_stage)
        # graph_dossier_v3 unified agent view (post-stage-record).
        # Read-only aggregation that joins graph-analysis + planning
        # artifacts into a single agent surface. Emitted AFTER
        # `recipe_stage_record(...)` snapshots its input tree so the
        # R009 hash chain stays intact: v3 files are NOT covered by any
        # stage's input_hash / output_hash. They are byte-pinned by
        # their own internal source.<input>_sha256 fields and are
        # idempotent on re-emit. Hash-chain integrity remains the
        # manifest's job; v3 integrity is its own concern.
        from compgen.graph_compilation.graph_dossier_v3 import (
            build_graph_dossier_v3,
        )

        _append_ledger(
            ledger_path, stage_id="graph_analysis",
            event="artifact_written",
            note="graph_dossier_v3 (M-10B): start",
        )
        build_graph_dossier_v3(out_dir)
        _append_ledger(
            ledger_path, stage_id="graph_analysis",
            event="artifact_written",
            note="graph_dossier_v3 (M-10B): finish",
        )
        # cost preview v2: target+tile-sensitive static cost preview
        # joined per candidate, inlined into v3 + llm_graph_view. Emitted
        # AFTER recipe_stage_record snapshot so source files in
        # 02_graph_analysis/ pinned by graph_analysis.output_hash are not
        # touched. Same hash-chain pattern as .
        if stop_after in (
            "cost-preview-v2", "agent-decision-request", "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential", "gap-discovery", "gap-closure"
        ):
            from compgen.graph_compilation.cost_preview_v2 import (
                run_cost_preview_v2,
            )

            _append_ledger(
                ledger_path, stage_id="graph_analysis",
                event="artifact_written",
                note="cost_preview_v2 (M-13): start",
            )
            run_cost_preview_v2(
                out_dir, target_yaml_path=target_cfg.raw_path,
            )
            _append_ledger(
                ledger_path, stage_id="graph_analysis",
                event="artifact_written",
                note="cost_preview_v2 (M-13): finish",
            )
            # profiler calibration. Best-effort measured profile of
            # the captured exported program; layered on top of 's
            # deterministic-roofline baseline. Gated on
            # ``COMPGEN_CALIBRATE_PROFILER`` (set or "1" to opt in;
            # "0" / unset to skip). Default OFF so suite runs stay
            # deterministic; enable explicitly for evidence-pack runs.
            if os.environ.get("COMPGEN_CALIBRATE_PROFILER", "") in (
                "1", "true", "True", "yes",
            ):
                from compgen.graph_compilation.profiler_calibration import (
                    run_profiler_calibration,
                )

                try:
                    _cal = run_profiler_calibration(out_dir)
                    _append_ledger(
                        ledger_path, stage_id="graph_analysis",
                        event="artifact_written",
                        note=(
                            f"profiler_calibration (M-18): "
                            f"{_cal.overall} "
                            f"({_cal.matched_region_count}/"
                            f"{_cal.total_region_count} regions matched)"
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    _append_ledger(
                        ledger_path, stage_id="graph_analysis",
                        event="artifact_written",
                        note=(
                            f"profiler_calibration (M-18): error "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
            # per-tile-candidate measured cost. Layered on top of
            # (region-level). Opt-in via
            # ``COMPGEN_CALIBRATE_CANDIDATES=1``. Best-effort: errors
            # are logged to the ledger but never raise the pipeline.
            if os.environ.get("COMPGEN_CALIBRATE_CANDIDATES", "") in (
                "1", "true", "True", "yes",
            ):
                from compgen.graph_compilation.candidate_calibration import (
                    run_candidate_calibration,
                )

                try:
                    _ccal = run_candidate_calibration(out_dir)
                    _append_ledger(
                        ledger_path, stage_id="graph_analysis",
                        event="artifact_written",
                        note=(
                            f"candidate_calibration (M-18.3): "
                            f"{_ccal.overall} "
                            f"({_ccal.candidates_calibrated}/"
                            f"{_ccal.candidate_count} candidates)"
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    _append_ledger(
                        ledger_path, stage_id="graph_analysis",
                        event="artifact_written",
                        note=(
                            f"candidate_calibration (M-18.3): error "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
            # kernel execution foundation. Layered alongside the
            # FX-level evidence; never
            # mutates any FX-level artifact. Opt-in via
            # ``COMPGEN_RUN_KERNELS=1``. Best-effort: errors are logged
            # to the ledger but never raise the pipeline.
            if os.environ.get("COMPGEN_RUN_KERNELS", "") in (
                "1", "true", "True", "yes",
            ):
                from compgen.graph_compilation.kernel_execution import (
                    run_kernel_execution,
                )

                try:
                    _ke = run_kernel_execution(out_dir)
                    _append_ledger(
                        ledger_path, stage_id="graph_analysis",
                        event="artifact_written",
                        note=(
                            f"kernel_execution (M-19): "
                            f"{_ke.overall} "
                            f"gpu={_ke.gpu_status} cpu={_ke.cpu_status}"
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    _append_ledger(
                        ledger_path, stage_id="graph_analysis",
                        event="artifact_written",
                        note=(
                            f"kernel_execution (M-19): error "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
                # per-region compiled differential. Layered on top
                # of (whose single-region artifact stays). Fan-out
                # across every region with a legal SetTileParams
                # candidate. Same env-var gate as .
                from compgen.graph_compilation.kernel_region_differential import (
                    run_region_compiled_differential,
                )

                try:
                    _rd = run_region_compiled_differential(out_dir)
                    _append_ledger(
                        ledger_path, stage_id="graph_analysis",
                        event="artifact_written",
                        note=(
                            f"region_compiled_differential (M-20): "
                            f"{_rd.overall} "
                            f"(regions={_rd.region_count}, "
                            f"gpu={_rd.gpu_compiled_count}, "
                            f"cpu={_rd.cpu_compiled_count}, "
                            f"fail_tol={_rd.fail_outside_tolerance_count})"
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    _append_ledger(
                        ledger_path, stage_id="graph_analysis",
                        event="artifact_written",
                        note=(
                            f"region_compiled_differential (M-20): error "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )
            # graph-analysis readiness pack. Read-only aggregator
            # that turns the per-region dossier facts (numerical
            # sensitivity, working-set curves, reuse, cost preview v2,
            # action space, llm view) into 6 typed readiness reports +
            # a top-level matrix. Best-effort: errors are logged to the
            # ledger but never raise the pipeline.
            from compgen.graph_compilation.graph_analysis_readiness import (
                build_readiness_pack,
            )

            try:
                _readiness = build_readiness_pack(out_dir)
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"graph_analysis_readiness (M-17.1): "
                        f"{_readiness.overall}"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"graph_analysis_readiness (M-17.1): error "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            # deterministic per-candidate analytical cost. Pure
            # function of (target spec, matmul shape, tile geometry,
            # working_set fit). Always-on: cheap arithmetic, no I/O,
            # byte-deterministic across reruns. When /
            # measurements are present, cross-references them as
            # calibration_delta in each per-candidate entry.
            from compgen.graph_compilation.analytical_cost import (
                run_analytical_cost,
            )

            try:
                _ac = run_analytical_cost(out_dir)
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"analytical_cost (M-21): {_ac.overall} "
                        f"(modeled={_ac.candidates_modeled}/"
                        f"{_ac.candidate_count})"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"analytical_cost (M-21): error "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

            # compiled bottleneck analysis: deterministic post-hoc
            # derivation of measured compute/bandwidth utilization from
            # /measured time × analytical flops/bytes.
            # Best-effort: emits typed ``no_measurements`` when /
            # didn't run. Layers ``compiled_evidence`` per region onto
            # hardware_resource_report.json and adds a top-level
            # ``kernel_calibration_status`` field (additive — does not
            # mutate 's existing fields).
            from compgen.graph_compilation.compiled_bottleneck import (
                run_compiled_bottleneck,
            )

            try:
                _cb = run_compiled_bottleneck(out_dir)
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"compiled_bottleneck (M-22): {_cb.overall} "
                        f"({_cb.region_count_with_evidence}/"
                        f"{_cb.region_count_total} regions, "
                        f"{_cb.kernel_calibration_status})"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"compiled_bottleneck (M-22): error "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

            # profiler evidence: real torch.profiler CUDA + perf
            # CPU measurement layered onto 's compiled_evidence.
            # Best-effort; emits typed perf_unavailable when
            # kernel.perf_event_paranoid blocks non-root events. Only
            # runs when has measurements (gated implicitly by
            # COMPGEN_RUN_KERNELS=1 reaching here).
            from compgen.graph_compilation.profiler_evidence import (
                run_profiler_evidence,
            )

            try:
                _pe = run_profiler_evidence(out_dir)
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"profiler_evidence (M-22.1): {_pe.overall} "
                        f"(gpu={_pe.gpu_collected_count}/"
                        f"{_pe.region_count}, "
                        f"cpu={_pe.cpu_collected_count}/"
                        f"{_pe.region_count})"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"profiler_evidence (M-22.1): error "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

            # compiled fusion verification: real Triton + cffi C
            # fused producer→consumer kernel; bit-equality vs eager
            # unfused chain on 's frozen input cases. No-op when
            # didn't pick a fusion candidate (typed not_run).
            from compgen.graph_compilation.compiled_fusion import (
                run_compiled_fusion,
            )

            try:
                _cf = run_compiled_fusion(out_dir)
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"compiled_fusion (M-23): {_cf.overall} "
                        f"(cases={_cf.case_count}, "
                        f"bit_eq={_cf.bit_equality_count}, "
                        f"tol_eps={_cf.tolerance_eps_count}, "
                        f"fail={_cf.fail_outside_tolerance_count}, "
                        f"gpu={_cf.gpu_status}, cpu={_cf.cpu_status})"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"compiled_fusion (M-23): error "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

            # kernel lifetime evidence: Triton CompiledKernel
            # introspection (always-on) + optional ncu dynamic counters.
            # Populates row 3 of (compiled_lifetime) so it flips
            # ready_for_m24_1 → ready when register_pressure +
            # register_spills + shared_memory_bytes + theoretical_
            # occupancy are all present.
            from compgen.graph_compilation.kernel_lifetime_evidence import (
                run_kernel_lifetime_evidence,
            )

            try:
                _kl = run_kernel_lifetime_evidence(out_dir)
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"kernel_lifetime (M-24.1): {_kl.overall} "
                        f"(introspected={_kl.introspected_count}/"
                        f"{_kl.region_count}, "
                        f"ncu_collected={_kl.ncu_collected_count}/"
                        f"{_kl.region_count})"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _append_ledger(
                    ledger_path, stage_id="graph_analysis",
                    event="artifact_written",
                    note=(
                        f"kernel_lifetime (M-24.1): error "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

        # emit the canonical ``agent_decision_request.json`` at
        # the halt point so an external agent can read it. Only emit
        # when the user is HALTING here (stop-after=agent-decision-
        # request) AND no in-line emission has already happened during
        # recipe_planning. The in-line emission (under agent-file /
        # llm-live modes) is captured by recipe_planning.output_hash;
        # overwriting it post-stage-record would break R005. For greedy
        # mode the in-line emission doesn't fire, so we must emit here.
        if stop_after == "agent-decision-request":
            existing_request = (
                out_dir / "03_recipe_planning" / "agent_decision"
                / "agent_decision_request.json"
            )
            if not existing_request.exists():
                from compgen.graph_compilation.agent_decision import (
                    build_agent_decision_request,
                )

                _append_ledger(
                    ledger_path, stage_id="recipe_planning",
                    event="artifact_written",
                    note="agent_decision_request (M-14A): emit",
                )
                build_agent_decision_request(out_dir)

        # kernel section readiness lock: read-only aggregator over
        # ////+ agent_decision_request. Runs
        #  so row 5 (compiled_agent_view) can cross-reference
        # candidate_ids_allowed. Always-on (best-effort); emits typed
        # not_run on rows whose evidence isn't on disk.
        from compgen.graph_compilation.kernel_readiness import (
            run_kernel_section_readiness,
        )

        try:
            _kr = run_kernel_section_readiness(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note=(
                    f"kernel_section_readiness (M-24): {_kr.overall} "
                    f"(ready={_kr.ready_count}, "
                    f"ready_for_m24_1={_kr.ready_for_m24_1_count}, "
                    f"partial={_kr.partial_count}, "
                    f"not_ready={_kr.not_ready_count})"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note=(
                    f"kernel_section_readiness (M-24): error "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        # detect downstream-gate rejection. After all sub-steps
        # have run, scan the downstream stage reports (,
        # ) for status=fail. If found AND a candidate was
        # actually committed (i.e. passed and wrote
        # candidate_selection.json), emit downstream_retry_request.json
        # mapping the failure back to the failed candidate. The
        # pipeline still raises afterwards (the run is not "ok") but
        # Claude Code can read the retry request and re-invoke with a
        # different candidate.
        from compgen.graph_compilation.downstream_retry import (
            detect_and_emit as _m15b_detect_and_emit,
        )

        downstream_failure = _m15b_detect_and_emit(out_dir)
        if downstream_failure is not None:
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note=(
                    f"downstream_retry_request (M-15B): "
                    f"{downstream_failure.failed_stage} reported fail"
                ),
            )
            # Write the manifest BEFORE raising so audit/integrity
            # tools can inspect the partial-run state. The retry
            # request is also on disk; together they let Claude Code
            # validate the run and choose a new candidate. Without
            # this, models that hit a real retry condition
            # (e.g. tiny_mlp's K_iters=4 accumulation reorder) lose
            # their manifest and fail every R009 / hash-chain audit.
            git_commit = _git_commit_or_none(repo_root)
            from compgen.graph_compilation.artifacts import ModelRef, TargetRef
            partial_manifest = RunManifest(
                schema_version="run_manifest_v1",
                run_id=run_id,
                created_at_utc=_utcnow(),
                git_commit=git_commit,
                model=ModelRef(
                    config_path=str(model_cfg.raw_path),
                    model_id=model_cfg.model_id,
                    config_sha256=model_cfg.raw_sha256,
                ),
                target=TargetRef(
                    config_path=str(target_cfg.raw_path),
                    target_id=target_cfg.target_id,
                    config_sha256=target_cfg.raw_sha256,
                ),
                seed=model_cfg.seed,
                stages=tuple(stages),
            )
            (out_dir / "run_manifest.json").write_text(
                json.dumps(
                    partial_manifest.to_dict(), indent=2, sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note=(
                    "run_manifest.json (partial; written before M-15B "
                    "retry-required raise so the run is auditable)"
                ),
            )
            # Raise so the CLI exits non-zero — Claude Code reads the
            # downstream_retry_request and re-invokes with a different
            # candidate (excluding the failed one).
            raise RuntimeError(
                f"M-15B downstream-gate rejection: "
                f"{downstream_failure.failed_stage} reported "
                f"{downstream_failure.failed_check!r} fail. "
                f"See 03_recipe_planning/downstream_retry/"
                f"downstream_retry_request.json for the typed retry "
                f"surface."
            )

    # ------------------------------------------------------------------ #
    # Kernel Specialization Request emission (/ Section 21).
    # When stop_after >= ``kernel-specialization-request`` we read the
    # selected Recipe IR decision + region facts and emit a typed
    # ``KernelSpecializationRequest`` to ``04_kernel_specialization/
    # requests/<request_id>.json``. is data-only — no codegen
    # fires here; (Triton emitter) and (C reference) consume
    # the request. Non-applicable recipe kinds (today: anything except
    # ``set_tile_params``) emit a typed ``not_applicable`` request
    # rather than skip silently.
    # ------------------------------------------------------------------ #
    needs_kernel_specialization = stop_after in (
        "kernel-specialization-request", "kernel-codegen-request", "kernel-auction", "execution-plan-emit", "glue-emit", "glue-differential",
        "gap-discovery",
        "gap-closure",
    ) and not _resume_skip_early
    if needs_kernel_specialization:
        # Phase C unified kernel-codegen boundary. 's legacy
        # request emitter is superseded ; the legacy
        # 04_kernel_specialization/ directory is no longer written.
        # The boundary now runs two sibling sub-steps:
        #   1. materialize KernelContractV3 from the selected
        #      Recipe op (writes 04_kernel_codegen/contracts/... +
        #      kernel_facing view at 04_kernel_codegen/views/...).
        #   2. emit the kernel-codegen task that points at those
        #      contract files (writes 04_kernel_codegen/requests/... +
        #      creates the sandboxed artifact_dir).
        _append_ledger(
            ledger_path, stage_id="kernel_specialization_request",
            event="start",
        )

        # contract materialization.
        from compgen.graph_compilation.kernel_contract_materialization import (
            materialize_contract_for_run,
        )

        try:
            _mat = materialize_contract_for_run(out_dir)
            _append_ledger(
                ledger_path, stage_id="kernel_specialization_request",
                event="artifact_written",
                note=(
                    f"kernel_contract_materialization (M-40): "
                    f"{_mat.overall} (rows={len(_mat.rows)})"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _append_ledger(
                ledger_path, stage_id="kernel_specialization_request",
                event="artifact_written",
                note=(
                    f"kernel_contract_materialization (M-40): error "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            raise

        # (Phase C): emit the kernel-codegen task that supersedes
        # 's request schema. Reads the materialization summary
        # to find contract_hash + paths; writes
        # 04_kernel_codegen/requests/<task_id>.request.json + creates
        # the sandboxed artifact_dir.
        from compgen.graph_compilation.kernel_codegen import (
            run_kernel_codegen_request,
        )

        try:
            _kc = run_kernel_codegen_request(out_dir)
            _append_ledger(
                ledger_path, stage_id="kernel_specialization_request",
                event="artifact_written",
                note=(
                    f"kernel_codegen_request (M-42): {_kc.overall} "
                    f"(request_kind={_kc.request_kind!r}, "
                    f"task_id={_kc.request_id!r})"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _append_ledger(
                ledger_path, stage_id="kernel_specialization_request",
                event="artifact_written",
                note=(
                    f"kernel_codegen_request (M-42): error "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            raise

        _append_ledger(
            ledger_path, stage_id="kernel_specialization_request",
            event="finish",
        )

    # ------------------------------------------------------------------ #
    # Auction: multi-bidder kernel codegen.
    # Runs whenever stop_after permits AND auction_mode != "disabled".
    # No-op when there are no applicable providers (today's clean
    # checkout); produces winner.json + auction_report.json + promotes
    # the winner to the standard response path so binds.
    # ------------------------------------------------------------------ #
    needs_auction = (
        auction_mode != "disabled"
        and stop_after
        in (
            "kernel-auction",
            "execution-plan-emit",
            "glue-emit",
            "glue-differential",
            "gap-discovery",
            "gap-closure",
        )
    )
    if needs_auction:
        _append_ledger(ledger_path, stage_id="kernel_auction", event="start")
        try:
            from compgen.graph_compilation.kernel_auction import run_kernel_auction

            _au = run_kernel_auction(
                run_dir=out_dir,
                mode=auction_mode,
                bid_cutoff=bid_cutoff,
            )
            _append_ledger(
                ledger_path,
                stage_id="kernel_auction",
                event="artifact_written",
                note=(
                    f"kernel_auction (M-57): {_au.overall} "
                    f"mode={_au.mode} winner={_au.winner_provider!r} "
                    f"bids={len(_au.bids)} fulfilled={len(_au.fulfilled)} "
                    f"verified={len(_au.verified)}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _append_ledger(
                ledger_path,
                stage_id="kernel_auction",
                event="artifact_written",
                note=(
                    f"kernel_auction (M-57): error "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            raise
        _append_ledger(ledger_path, stage_id="kernel_auction", event="finish")

    # ------------------------------------------------------------------ #
    # Execution-plan emit: bind certified kernels to regions and
    # write 05_execution_plan/execution_plan.yaml + region_kernel_bindings.json.
    # Runs whenever stop_after permits — bindings will be empty until the
    # operator submits a provider response and //emit cert(s).
    # ------------------------------------------------------------------ #
    needs_plan_emit = stop_after in (
        "execution-plan-emit", "glue-emit", "glue-differential", "gap-discovery", "gap-closure",
    )
    if needs_plan_emit:
        from compgen.graph_compilation.execution_plan_emit import (
            emit_execution_plan,
        )

        _append_ledger(
            ledger_path, stage_id="execution_plan_emit", event="start",
        )
        try:
            _ep = emit_execution_plan(out_dir)
            _append_ledger(
                ledger_path, stage_id="execution_plan_emit",
                event="artifact_written",
                note=(
                    f"execution_plan_emit (M-46): {_ep.overall} "
                    f"(bound={_ep.bound_count}, unbound={_ep.unbound_count})"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _append_ledger(
                ledger_path, stage_id="execution_plan_emit",
                event="artifact_written",
                note=(
                    f"execution_plan_emit (M-46): error "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            raise
        _append_ledger(
            ledger_path, stage_id="execution_plan_emit", event="finish",
        )

    # ------------------------------------------------------------------ #
    # coverage-first pass: walk every region dossier, derive
    # canonical contract hashes, look up matching certs, and append
    # coverage-inflated bindings. Runs so it has the
    # initial bindings to extend; so the emitter sees
    # the inflated count.
    # ------------------------------------------------------------------ #
    needs_coverage = (
        kernel_coverage_mode != "disabled"
        and stop_after
        in (
            "kernel-auction",
            "execution-plan-emit",
            "glue-emit",
            "glue-differential",
            "gap-discovery",
            "gap-closure",
        )
    )
    if needs_coverage and needs_plan_emit:
        _append_ledger(ledger_path, stage_id="kernel_coverage_first", event="start")
        try:
            from compgen.graph_compilation.coverage_first import run_coverage_first

            _cf = run_coverage_first(
                run_dir=out_dir, mode=kernel_coverage_mode,
            )
            _append_ledger(
                ledger_path,
                stage_id="kernel_coverage_first",
                event="artifact_written",
                note=(
                    f"kernel_coverage_first (M-63): {_cf.overall} "
                    f"mode={_cf.mode} groups={len(_cf.groups)} "
                    f"coverage_inflation_total={_cf.coverage_inflation_total}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _append_ledger(
                ledger_path,
                stage_id="kernel_coverage_first",
                event="artifact_written",
                note=(
                    f"kernel_coverage_first (M-63): error "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            raise
        _append_ledger(ledger_path, stage_id="kernel_coverage_first", event="finish")

    # ------------------------------------------------------------------ #
    # Glue emit: generate the per-workload Python SYNC plan
    # executor under 06_glue_emit/. Reads 's execution_plan.yaml.
    # ------------------------------------------------------------------ #
    needs_glue_emit = stop_after in ("glue-emit", "glue-differential", "gap-discovery", "gap-closure")
    if needs_glue_emit:
        from compgen.runtime.glue_emit import (
            emit_python_async_executor,
            emit_python_cuda_executor,
            emit_python_sync_executor,
        )

        _append_ledger(ledger_path, stage_id="glue_emit", event="start")
        try:
            _ge = emit_python_sync_executor(out_dir)
            _ae = emit_python_async_executor(out_dir)
            _ce = emit_python_cuda_executor(out_dir)
            _append_ledger(
                ledger_path, stage_id="glue_emit",
                event="artifact_written",
                note=(
                    f"glue_emit (M-47): {_ge.overall} "
                    f"(bound={len(_ge.bound_regions)}, "
                    f"unbound={len(_ge.unbound_regions)}); "
                    f"async (M-51): {_ae.overall} "
                    f"(async_regions={len(_ae.async_regions)}); "
                    f"cuda (M-52): {_ce.overall}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _append_ledger(
                ledger_path, stage_id="glue_emit",
                event="artifact_written",
                note=(
                    f"glue_emit (M-47/M-51/M-52): error {type(exc).__name__}: {exc}"
                ),
            )
            raise
        _append_ledger(ledger_path, stage_id="glue_emit", event="finish")

    # ------------------------------------------------------------------ #
    # Glue differential (paper-facing): drive the emitted
    # executor with synthesized cases and compare against eager
    # torch.matmul. Emits 06_glue_emit/glue_differential_report.json.
    # picks up status=fail via the downstream-retry table.
    # ------------------------------------------------------------------ #
    needs_glue_diff = stop_after in (
        "glue-differential", "gap-discovery", "gap-closure",
    )
    if needs_glue_diff:
        from compgen.graph_compilation.glue_differential import (
            run_glue_differential,
        )

        _append_ledger(ledger_path, stage_id="glue_differential", event="start")
        try:
            _gd = run_glue_differential(out_dir)
            _append_ledger(
                ledger_path, stage_id="glue_differential",
                event="artifact_written",
                note=(
                    f"glue_differential (M-49): {_gd.status} "
                    f"(refinement={_gd.refinement_status!r}, "
                    f"cases={_gd.cases_passed}/{_gd.cases_total})"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            _append_ledger(
                ledger_path, stage_id="glue_differential",
                event="artifact_written",
                note=(
                    f"glue_differential (M-49): error "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            raise
        _append_ledger(
            ledger_path, stage_id="glue_differential", event="finish",
        )

    # ------------------------------------------------------------------ #
    # Gap Discovery (when stop_after >= gap-discovery)
    # ------------------------------------------------------------------ #
    needs_gap_discovery = stop_after in ("gap-discovery", "gap-closure")
    if needs_gap_discovery:
        from compgen.graph_compilation.gaps import run_gap_discovery

        _append_ledger(ledger_path, stage_id="gap_discovery", event="start")
        gap_stage = run_gap_discovery(
            out_dir,
            target_id=target_cfg.target_id,
            model_id=model_cfg.model_id,
            extension_registry=extension_registry,
        )
        for ref in gap_stage.outputs:
            _append_ledger(
                ledger_path,
                stage_id="gap_discovery",
                event="artifact_written",
                note=f"path={ref.path} sha={ref.sha256[:12]}",
            )
        _append_ledger(ledger_path, stage_id="gap_discovery", event="finish")
        stages.append(gap_stage)

        # When the run consumed a registry, emit the closure-proof reports
        # at the top of the run dir. These are the artifacts spec 06 grades
        # the agentic loop on.
        if extension_registry is not None:
            _emit_closure_proof_reports(out_dir, model_cfg.model_id, target_cfg.target_id)

    # ------------------------------------------------------------------ #
    # Gap Closure (only when stop_after == gap-closure)
    # ------------------------------------------------------------------ #
    if stop_after == "gap-closure":
        from compgen.graph_compilation.gap_closure import run_gap_closure

        if extensions_root is None:
            extensions_root = repo_root / ".crg-artifacts" / "extensions"
        if extension_registry is None:
            registry_for_closure = extensions_root / "registry.yaml"
        else:
            registry_for_closure = extension_registry

        _append_ledger(ledger_path, stage_id="gap_closure", event="start")
        closure_stage = run_gap_closure(
            out_dir,
            extensions_root=extensions_root,
            registry_path=registry_for_closure,
            target_id=target_cfg.target_id,
            model_id=model_cfg.model_id,
        )
        for ref in closure_stage.outputs:
            _append_ledger(
                ledger_path,
                stage_id="gap_closure",
                event="artifact_written",
                note=f"path={ref.path} sha={ref.sha256[:12]}",
            )
        _append_ledger(ledger_path, stage_id="gap_closure", event="finish")
        stages.append(closure_stage)

    # ------------------------------------------------------------------ #
    # Manifest
    # ------------------------------------------------------------------ #
    git_commit = _git_commit_or_none(repo_root)

    from compgen.graph_compilation.artifacts import ModelRef, TargetRef

    manifest = RunManifest(
        schema_version="run_manifest_v1",
        run_id=run_id,
        created_at_utc=_utcnow(),
        git_commit=git_commit,
        model=ModelRef(
            config_path=str(model_cfg.raw_path),
            model_id=model_cfg.model_id,
            config_sha256=model_cfg.raw_sha256,
        ),
        target=TargetRef(
            config_path=str(target_cfg.raw_path),
            target_id=target_cfg.target_id,
            config_sha256=target_cfg.raw_sha256,
        ),
        seed=model_cfg.seed,
        stages=tuple(stages),
    )

    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # write import_provenance.json. Anchored to the same seam
    # as the manifest so cross-references are stable (run_id matches).
    try:
        from compgen.audit.import_provenance import (
            ImportSnapshot,
            compute_provenance,
            write_provenance,
        )

        _provenance = compute_provenance(
            before=_import_snapshot_before,
            after=ImportSnapshot.take("after"),
            run_id=run_id,
            selection_mode=selection_mode,
            source_commit=_git_commit_or_none(repo_root) or "",
        )
        write_provenance(_provenance, run_dir=out_dir)
        _append_ledger(
            ledger_path, stage_id="trust_audit",
            event="artifact_written",
            note=(
                f"import_provenance.json (cache_mode={_provenance.cache_mode}, "
                f"evidence_mode={_provenance.evidence_mode}, "
                f"forbidden_count={len(_provenance.forbidden_modules_imported)})"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - best-effort
        _append_ledger(
            ledger_path, stage_id="trust_audit",
            event="artifact_written",
            note=f"import_provenance error {type(exc).__name__}: {exc}",
        )

    # promotion bridge (Section 19, write side). Must fire AFTER
    # run_manifest.json is on disk: the bridge reads the manifest to
    # learn model_id / target_id / created_at_utc, and the synthetic
    # tests inadvertently masked this ordering by writing the manifest
    # first. Best-effort: errors land in the ledger, never raise.
    # Writes its own artifacts under 04_promotion/ which is not
    # covered by any earlier stage's output_hash (R009-safe), so this
    # post-manifest emission does not invalidate the hash chain.
    if needs_recipe_planning:
        from compgen.graph_compilation.promotion_bridge import (
            emit as _promotion_emit,
        )

        try:
            _pe = _promotion_emit(out_dir)
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note=(
                    f"promotion_bridge (M-26): {_pe.status} "
                    f"(reason={_pe.reason!r})"
                ),
            )
        except Exception as exc:  # noqa: BLE001 - best-effort
            _append_ledger(
                ledger_path, stage_id="recipe_planning",
                event="artifact_written",
                note=(
                    f"promotion_bridge (M-26): error "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # emit a decision trace per finished agent decision. Same
    # post-manifest position as the bridge; the trace contains the
    # request/llm_view/candidate_actions/promotion_library hashes that
    # let a future replay assert the decision was deterministic.
    if needs_recipe_planning:
        try:
            from compgen.audit.trace_replay import build_trace, write_trace

            request_path = (
                out_dir / "03_recipe_planning" / "agent_decision"
                / "agent_decision_request.json"
            )
            if request_path.exists():
                _trace = build_trace(
                    out_dir,
                    run_id=run_id,
                    region_id="",  # request-level trace; per-region traces are 's job
                    decision_index=0,
                    commit=_git_commit_or_none(repo_root) or "",
                )
                write_trace(_trace, run_dir=out_dir)
                _append_ledger(
                    ledger_path, stage_id="trust_audit",
                    event="artifact_written",
                    note=(
                        f"agent_decision_trace_0000.json "
                        f"(decision_id={_trace.decision_id})"
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - best-effort
            _append_ledger(
                ledger_path, stage_id="trust_audit",
                event="artifact_written",
                note=f"trace_replay error {type(exc).__name__}: {exc}",
            )

    return RunResult(
        run_dir=out_dir,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        stages=tuple(stages),
    )


def lower_from_existing_capture(
    capture_run: Path,
    target_config_path: Path,
    out_dir: Path,
    *,
    run_id: str | None = None,
    repo_root: Path | None = None,
) -> RunResult:
    """Wire ``lower --capture-run …``.

    Copies an existing graph_capture run's ``00_graph_capture/`` into a
    fresh ``out_dir`` and runs Payload Lowering on it. Used to prove
    Payload Lowering consumes saved artifacts (no hidden re-capture).
    """
    from compgen.graph_compilation.artifacts import ModelRef, TargetRef
    from compgen.graph_compilation.lower import copy_capture_run, run_payload_lowering

    capture_run = Path(capture_run).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    target_cfg = TargetConfig.load(Path(target_config_path))
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    # Copy capture artifacts (read source manifest first to learn the model_id).
    src_manifest_path = capture_run / "run_manifest.json"
    src_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8"))
    src_model = src_manifest.get("model", {})
    src_seed = int(src_manifest.get("seed", 0))
    copy_capture_run(capture_run, out_dir)

    if run_id is None:
        run_id = f"graphcomp_lower_{src_model.get('model_id', 'unknown')}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"

    ledger_path = out_dir / "stage_ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")

    # Synthesize a graph_capture StageRecord that points at the copied
    # 00_graph_capture/. Hashes are recomputed from disk; the source
    # manifest's stage record is *not* trusted.
    from compgen.graph_compilation.hashing import sha256_tree

    capture_dir = out_dir / "00_graph_capture"
    if not capture_dir.is_dir():
        raise FileNotFoundError(f"00_graph_capture/ missing in copy: {capture_dir}")

    from compgen.graph_compilation.artifacts import ArtifactRef
    from compgen.graph_compilation.hashing import sha256_file

    capture_stage_outputs: list[ArtifactRef] = []
    for p in sorted(capture_dir.rglob("*")):
        if not p.is_file():
            continue
        capture_stage_outputs.append(
            ArtifactRef(
                path=p.relative_to(out_dir).as_posix(),
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                kind="file",
            )
        )

    src_capture_stage = next(
        (s for s in src_manifest.get("stages", []) if s.get("stage_id") == "graph_capture"),
        None,
    )
    src_input_hash = src_capture_stage.get("input_hash", "0" * 64) if src_capture_stage else "0" * 64

    capture_stage = StageRecord(
        stage_id="graph_capture",
        status="pass",
        inputs=(),
        outputs=tuple(capture_stage_outputs),
        report_path="00_graph_capture/capture_report.json",
        input_hash=src_input_hash,
        output_hash=sha256_tree(capture_dir),
        llm_calls=0,
        started_at_utc=_utcnow(),
        finished_at_utc=_utcnow(),
    )

    _append_ledger(ledger_path, stage_id="graph_capture", event="start", note="copied from capture-run")
    _append_ledger(ledger_path, stage_id="graph_capture", event="finish")
    _append_ledger(ledger_path, stage_id="payload_lowering", event="start")
    lowering_stage, _ = run_payload_lowering(out_dir, target_id=target_cfg.target_id)
    for ref in lowering_stage.outputs:
        _append_ledger(
            ledger_path,
            stage_id="payload_lowering",
            event="artifact_written",
            note=f"path={ref.path} sha={ref.sha256[:12]}",
        )
    _append_ledger(ledger_path, stage_id="payload_lowering", event="finish")

    git_commit = _git_commit_or_none(repo_root)
    manifest = RunManifest(
        schema_version="run_manifest_v1",
        run_id=run_id,
        created_at_utc=_utcnow(),
        git_commit=git_commit,
        model=ModelRef(
            config_path=src_model.get("config_path", ""),
            model_id=src_model.get("model_id", ""),
            config_sha256=src_model.get("config_sha256", "0" * 64),
        ),
        target=TargetRef(
            config_path=str(target_cfg.raw_path),
            target_id=target_cfg.target_id,
            config_sha256=target_cfg.raw_sha256,
        ),
        seed=src_seed,
        stages=(capture_stage, lowering_stage),
    )
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return RunResult(
        run_dir=out_dir,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        stages=(capture_stage, lowering_stage),
    )


def discover_gaps_from_existing_lowering(
    lowering_run: Path,
    target_config_path: Path,
    out_dir: Path,
    *,
    run_id: str | None = None,
    repo_root: Path | None = None,
    extension_registry: Path | None = None,
) -> RunResult:
    """Wire ``gap-discovery --capture-lowering-run …``.

    Copies an existing run's ``00_graph_capture/`` and
    ``01_payload_lowering/`` into a fresh ``out_dir`` and runs Gap
    Discovery on it. Used to prove Gap Discovery consumes saved
    artifacts (no hidden re-capture, no hidden re-lowering).
    """
    from compgen.graph_compilation.artifacts import ArtifactRef, ModelRef, TargetRef
    from compgen.graph_compilation.gaps import run_gap_discovery
    from compgen.graph_compilation.hashing import sha256_file, sha256_tree

    lowering_run = Path(lowering_run).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    target_cfg = TargetConfig.load(Path(target_config_path))
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    src_manifest_path = lowering_run / "run_manifest.json"
    src_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8"))
    src_model = src_manifest.get("model", {})
    src_seed = int(src_manifest.get("seed", 0))

    for sub in ("00_graph_capture", "01_payload_lowering"):
        src = lowering_run / sub
        if not src.is_dir():
            raise FileNotFoundError(f"missing {sub}/ in source run: {src}")
        shutil.copytree(src, out_dir / sub, dirs_exist_ok=True)

    if run_id is None:
        run_id = (
            f"graphcomp_gap_discovery_{src_model.get('model_id', 'unknown')}_"
            f"{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        )

    ledger_path = out_dir / "stage_ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")

    # Synthesize copied stage records (graph_capture + payload_lowering),
    # then run gap_discovery.
    capture_dir = out_dir / "00_graph_capture"
    lowering_dir = out_dir / "01_payload_lowering"

    capture_outputs: list[ArtifactRef] = []
    for p in sorted(capture_dir.rglob("*")):
        if not p.is_file():
            continue
        capture_outputs.append(
            ArtifactRef(
                path=p.relative_to(out_dir).as_posix(),
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                kind="file",
            )
        )

    lowering_outputs: list[ArtifactRef] = []
    for p in sorted(lowering_dir.rglob("*")):
        if not p.is_file():
            continue
        lowering_outputs.append(
            ArtifactRef(
                path=p.relative_to(out_dir).as_posix(),
                sha256=sha256_file(p),
                size_bytes=p.stat().st_size,
                kind="file",
            )
        )

    src_capture_stage = next(
        (s for s in src_manifest.get("stages", []) if s.get("stage_id") == "graph_capture"), None
    )
    src_capture_input_hash = src_capture_stage.get("input_hash", "0" * 64) if src_capture_stage else "0" * 64

    capture_record = StageRecord(
        stage_id="graph_capture",
        status="pass",
        inputs=(),
        outputs=tuple(capture_outputs),
        report_path="00_graph_capture/capture_report.json",
        input_hash=src_capture_input_hash,
        output_hash=sha256_tree(capture_dir),
        llm_calls=0,
        started_at_utc=_utcnow(),
        finished_at_utc=_utcnow(),
    )
    lowering_record = StageRecord(
        stage_id="payload_lowering",
        status="pass",
        inputs=tuple(capture_outputs),
        outputs=tuple(lowering_outputs),
        report_path="01_payload_lowering/lowering_summary.json",
        input_hash=sha256_tree(capture_dir),
        output_hash=sha256_tree(lowering_dir),
        llm_calls=0,
        started_at_utc=_utcnow(),
        finished_at_utc=_utcnow(),
    )

    _append_ledger(ledger_path, stage_id="graph_capture", event="start", note="copied from lowering-run")
    _append_ledger(ledger_path, stage_id="graph_capture", event="finish")
    _append_ledger(ledger_path, stage_id="payload_lowering", event="start", note="copied from lowering-run")
    _append_ledger(ledger_path, stage_id="payload_lowering", event="finish")
    _append_ledger(ledger_path, stage_id="gap_discovery", event="start")
    gap_stage = run_gap_discovery(
        out_dir,
        target_id=target_cfg.target_id,
        model_id=src_model.get("model_id", "unknown"),
        extension_registry=extension_registry,
    )
    for ref in gap_stage.outputs:
        _append_ledger(
            ledger_path,
            stage_id="gap_discovery",
            event="artifact_written",
            note=f"path={ref.path} sha={ref.sha256[:12]}",
        )
    _append_ledger(ledger_path, stage_id="gap_discovery", event="finish")

    git_commit = _git_commit_or_none(repo_root)
    manifest = RunManifest(
        schema_version="run_manifest_v1",
        run_id=run_id,
        created_at_utc=_utcnow(),
        git_commit=git_commit,
        model=ModelRef(
            config_path=src_model.get("config_path", ""),
            model_id=src_model.get("model_id", ""),
            config_sha256=src_model.get("config_sha256", "0" * 64),
        ),
        target=TargetRef(
            config_path=str(target_cfg.raw_path),
            target_id=target_cfg.target_id,
            config_sha256=target_cfg.raw_sha256,
        ),
        seed=src_seed,
        stages=(capture_record, lowering_record, gap_stage),
    )
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return RunResult(
        run_dir=out_dir,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        stages=(capture_record, lowering_record, gap_stage),
    )

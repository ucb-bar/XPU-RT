"""Command-line entry point for the graph compilation toolchain.

Invoked as ``python -m compgen.graph_compilation <subcommand>``.

Subcommands:

- ``validate`` (graph_compilation artifact contract) — independently validates a run directory.
- ``run`` (graph_capture stage) — runs Stage 0 capture and writes a graph compilation
  manifest-backed run.
- ``replay-goldens`` (graph_capture stage) — reloads the saved exported program
  and goldens and asserts numerical equality.
- ``compare`` (graph_capture stage) — diffs stable fields between two reruns.

Note on placement: the repo's existing CLI lives in ``compgen.cli`` as a
single module. We do not extend it here because converting it into a
package is out of scope. Instead, this package owns its own ``__main__``
so the CLI surface is co-located with the contracts it owns.

Exit codes (validate):

- ``0`` — validator ran and the report's overall is ``"pass"``.
- ``1`` — validator ran and the report's overall is ``"fail"``.
- ``2`` — internal error or external precondition violated (e.g. the
  run directory does not exist).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from compgen.graph_compilation.validate import validate_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m compgen.graph_compilation",
        description="Capture/lower toolchain: run, validate, replay, compare.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # validate
    validate = sub.add_parser(
        "validate",
        help="Independently validate a run directory; recomputes hashes and rejects tampered runs.",
    )
    validate.add_argument("--run", required=True, type=Path, help="Path to the run directory.")
    validate.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Optional path to write the ValidationReport JSON. The validator always also "
            "writes <run>/validation/artifact_validation.json."
        ),
    )

    # run
    run = sub.add_parser("run", help="Execute the graph compilation pipeline up to --stop-after.")
    run.add_argument("--model", required=True, type=Path, help="Path to model config YAML.")
    run.add_argument("--target", required=True, type=Path, help="Path to target config YAML.")
    run.add_argument("--out", required=True, type=Path, help="Output run directory (will be replaced).")
    run.add_argument(
        "--stop-after",
        choices=(
            "graph-capture", "payload-lowering", "graph-analysis",
            "recipe-planning", "recipe-verification", "recipe-lowering",
            "post-lowering-verification", "differential-verification",
            "real-transform-eligibility", "real-set-tile-transform",
            "real-transform-differential", "cost-preview-v2",
            "agent-decision-request",
            "kernel-specialization-request",
            "kernel-codegen-request",
            "kernel-auction",
            "execution-plan-emit",
            "glue-emit",
            "glue-differential",
            "gap-discovery", "gap-closure",
        ),
        default="graph-capture",
        help="Last stage to execute.",
    )
    run.add_argument(
        "--selection-mode",
        choices=("greedy", "agent-file", "llm-live"),
        default="greedy",
        help=(
            "Recipe-planning selector. greedy (default): deterministic "
            "baseline, no LLM. agent-file (RECOMMENDED for agentic "
            "runs): read --agent-decision-response written by Claude "
            "Code via MCP/skill, or any external agent — no API key "
            "needed. llm-live (explicit opt-in only): call a live "
            "provider (gemini/anthropic/openai), uses real tokens; "
            "see --llm-live-provider."
        ),
    )
    run.add_argument(
        "--agent-decision-response",
        type=Path,
        default=None,
        action="append",
        help=(
            "Path to a pre-written agent_decision_response.json. Required "
            "when --selection-mode=agent-file. Repeatable: M-15A retry "
            "iterates through the supplied paths in order, snapshots each "
            "attempt under attempts/attempt_<N>/, and commits the first "
            "passing response. Single use: pass once."
        ),
    )
    run.add_argument(
        "--resume-from",
        choices=("kernel-codegen-response",),
        default=None,
        help=(
            "M-53: resume the pipeline after the operator has committed a "
            "kernel-codegen response via the M-43 API. Requires --out to "
            "point at a previous run that already reached "
            "--stop-after=kernel-codegen-request. Skips the wipe + early "
            "stages (graph_capture, payload_lowering, graph_analysis, "
            "recipe_planning, kernel_specialization_request) so the "
            "committed response, attempts trail, and certificates survive "
            "across the pipeline-restart boundary. Drives M-46 "
            "(execution_plan_emit) → M-47 (glue_emit) → M-49 "
            "(glue_differential) on the existing artifacts."
        ),
    )
    run.add_argument(
        "--agent-max-retries",
        type=int,
        default=3,
        help=(
            "M-15A: maximum number of agent_decision retry attempts in a "
            "single run. The first response that passes the M-14A "
            "validator is committed; later responses are not consumed. "
            "Exhausting retries leaves recipe.mlir unwritten."
        ),
    )
    run.add_argument(
        "--llm-live-provider",
        choices=("env", "gemini", "anthropic", "openai"),
        default="gemini",
        help=(
            "Provider backend for --selection-mode=llm-live (only "
            "consulted when llm-live is explicitly requested; not the "
            "default agentic path). gemini (default for llm-live): "
            "Google Gemini-2.5-flash via google-genai SDK (reads "
            "GEMMINI_API from .env or GOOGLE_API_KEY); auto-recorded "
            "by compgen.observability.gemini_usage. anthropic|openai: "
            "real provider via stdlib HTTP. env: dispatch via "
            "COMPGEN_LLM_PROVIDER. (For Claude-Code-driven runs use "
            "--selection-mode=agent-file instead — that's the "
            "recommended primary path.)"
        ),
    )
    run.add_argument(
        "--llm-live-model",
        default=None,
        help="Model name passed to the provider (env: COMPGEN_LLM_MODEL).",
    )
    run.add_argument(
        "--llm-live-timeout-sec",
        type=int,
        default=60,
        help="Provider call timeout in seconds (default 60).",
    )
    run.add_argument(
        "--llm-live-dry-run",
        action="store_true",
        help=(
            "Emit prompt + provider request but do NOT call the "
            "provider; halt before recipe.mlir is committed."
        ),
    )
    run.add_argument(
        "--llm-live-fallback",
        choices=("none", "greedy"),
        default="none",
        help=(
            "What to do if the provider fails. none (default): hard "
            "abort. greedy: fall back to greedy and record "
            "fallback_used=true in the trace."
        ),
    )
    run.add_argument("--run-id", default=None, help="Optional explicit run_id; default is auto-generated.")
    run.add_argument(
        "--extension-registry",
        type=Path,
        default=None,
        help="Path to extension registry.yaml; closed targets are skipped at gap-discovery.",
    )
    run.add_argument(
        "--extensions-root",
        type=Path,
        default=None,
        help="Root for materialized extensions (default: <repo>/.crg-artifacts/extensions).",
    )
    run.add_argument(
        "--auction-mode",
        choices=("multi-bidder", "first-fit", "disabled"),
        default="multi-bidder",
        help=(
            "M-57: kernel-auction mode. multi-bidder (default): every "
            "applicable provider bids; top-K (--bid-cutoff) fulfill; "
            "all verify; selector picks by perf_estimate. first-fit: "
            "stop at the first verified bid in priority order. disabled: "
            "skip the auction entirely (today's M-43 commit path "
            "remains canonical)."
        ),
    )
    run.add_argument(
        "--bid-cutoff",
        type=int,
        default=3,
        help=(
            "M-57: top-K bidders that proceed to fulfill() in multi-bidder "
            "mode. Lower values cap the auction's wall-clock + budget "
            "footprint; higher values surface more comparative data in "
            "auction_report.json. Default 3."
        ),
    )
    run.add_argument(
        "--user-kernel-path",
        type=Path,
        default=None,
        help=(
            "M-62: directory containing user-supplied kernel manifests "
            "(``kernel_manifest.yaml`` + sibling kernel source files). "
            "When set, re-indexes the directory under "
            ".compgen/user_kernel_index/ before the auction runs so "
            "UserKernelProvider can bid. Falls back to the "
            "COMPGEN_USER_KERNEL_PATH env var when the flag is omitted."
        ),
    )
    run.add_argument(
        "--kernel-coverage-mode",
        choices=("both", "first-pass-coverage", "specialize", "disabled"),
        default="both",
        help=(
            "M-63: coverage-first scheduling. both (default): coverage "
            "+ specialization analysis after the auction. "
            "first-pass-coverage: only the coverage report (canonical-"
            "hash reuse → coverage-inflated bindings). specialize: only "
            "the specialization report (regions ranked for follow-on "
            "shape-specialized auction). disabled: no-op."
        ),
    )

    # run-suite (multi-model run from a YAML manifest)
    run_suite = sub.add_parser(
        "run-suite",
        help="Execute the pipeline for every model listed in a graph_compilation_suite_v1 YAML.",
    )
    run_suite.add_argument(
        "--suite",
        required=True,
        type=Path,
        help="Path to a graph_compilation_suite_v1 YAML (e.g. configs/graph_compilation/always_test_models.yaml).",
    )
    run_suite.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output root; one subdirectory per model_id is created under it.",
    )
    run_suite.add_argument(
        "--stop-after",
        choices=(
            "graph-capture", "payload-lowering", "graph-analysis",
            "recipe-planning", "recipe-verification", "recipe-lowering",
            "post-lowering-verification", "differential-verification",
            "real-transform-eligibility", "real-set-tile-transform",
            "real-transform-differential", "cost-preview-v2",
            "agent-decision-request",
            "kernel-specialization-request",
            "kernel-codegen-request",
            "kernel-auction",
            "execution-plan-emit",
            "glue-emit",
            "glue-differential",
            "gap-discovery", "gap-closure",
        ),
        default="graph-analysis",
        help="Last stage to execute for each model. Default: graph-analysis.",
    )
    run_suite.add_argument(
        "--selection-mode",
        choices=("greedy", "agent-file", "llm-live"),
        default="greedy",
        help=(
            "Recipe-planning selector when stop-after >= recipe-planning. "
            "agent-file requires per-model responses; for run-suite "
            "use --agent-decision-response-dir. llm-live forwards to "
            "the live HTTP provider (anthropic/openai)."
        ),
    )
    run_suite.add_argument(
        "--agent-decision-response-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing per-model agent_decision_response.json "
            "files (named <model_id>.json). Used with "
            "--selection-mode=agent-file in run-suite mode."
        ),
    )
    run_suite.add_argument(
        "--llm-live-provider",
        choices=("env", "gemini", "anthropic", "openai"),
        default="gemini",
        help="Live provider backend (env|gemini|anthropic|openai).",
    )
    run_suite.add_argument(
        "--llm-live-model", default=None,
        help="Model name passed to the M-14B provider.",
    )
    run_suite.add_argument(
        "--llm-live-timeout-sec", type=int, default=60,
        help="M-14B provider call timeout (s).",
    )
    run_suite.add_argument(
        "--llm-live-dry-run", action="store_true",
        help="M-14B: emit prompt + provider request without calling the provider.",
    )
    run_suite.add_argument(
        "--llm-live-fallback", choices=("none", "greedy"), default="none",
        help="M-14B fallback behavior on provider failure.",
    )
    run_suite.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Override the suite's target config; default = the suite's `target` field.",
    )
    run_suite.add_argument(
        "--extension-registry",
        type=Path,
        default=None,
        help="Optional registry; threaded through to gap-discovery / gap-closure.",
    )
    run_suite.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Don't stop the suite on a single model failure; record and continue.",
    )

    # lower (Payload Lowering only, consuming an existing capture run)
    lower = sub.add_parser(
        "lower",
        help="Run Payload Lowering against an existing graph_capture run.",
    )
    lower.add_argument(
        "--capture-run",
        required=True,
        type=Path,
        help="Path to an existing run directory whose 00_graph_capture/ artifacts will be lowered.",
    )
    lower.add_argument("--target", required=True, type=Path, help="Path to target config YAML.")
    lower.add_argument("--out", required=True, type=Path, help="Output run directory (will be replaced).")
    lower.add_argument("--run-id", default=None, help="Optional explicit run_id.")

    # resolve-candidate (M-04.5: resolve a selected candidate_id against action_space.mlir)
    rc = sub.add_parser(
        "resolve-candidate",
        help="Resolve a selected candidate_id against action_space.mlir (M-04.5).",
    )
    rc.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Run directory containing 02_graph_analysis/.",
    )
    rc.add_argument(
        "--candidate-id",
        required=True,
        help="The candidate_id to resolve. Must appear in action_space.mlir.",
    )
    rc.add_argument(
        "--allow-illegal",
        action="store_true",
        help="Allow resolving candidates with legality.ok = false. Disabled by default.",
    )
    rc.add_argument(
        "--selection-mode",
        default="explicit",
        help="Recorded in candidate_selection.json (e.g. greedy, agent-file, llm-live, explicit).",
    )
    rc.add_argument(
        "--rationale-primary",
        default="",
        help="Optional one-line primary rationale recorded in candidate_selection.json.",
    )
    rc.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing candidate_selection.json / selected_recipe_delta.mlir; only verify.",
    )

    # discover-target (auto-build a partial graph_compilation target YAML)
    dt = sub.add_parser(
        "discover-target",
        help="Probe the host (Linux) for CPU + accelerator info and write a partial target YAML.",
    )
    dt.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output YAML path. Existing file is overwritten.",
    )
    dt.add_argument(
        "--target-id",
        default=None,
        help="Override target_id; default is derived from CPU/accelerator name.",
    )

    # analyze-graph (Graph Analysis only, consuming an existing payload-lowering run)
    ag = sub.add_parser(
        "analyze-graph",
        help="Build the 02_graph_analysis/ region_map / tensor_use_def_graph / region_graph against an existing payload-lowering run.",
    )
    ag.add_argument(
        "--lowering-run",
        required=True,
        type=Path,
        help="Path to a run directory whose 00_graph_capture/ + 01_payload_lowering/ already exist.",
    )
    ag.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Either the same path (in-place; emits 02_graph_analysis/ alongside) or a fresh dir to copy stages into.",
    )
    ag.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Optional target config YAML; powers the Region Dossier V2 cost model. Default: host_cpu.yaml.",
    )

    # discover-gaps (Gap Discovery only, consuming an existing payload-lowering run)
    gap = sub.add_parser(
        "discover-gaps",
        help="Run Gap Discovery against an existing payload-lowering run.",
    )
    gap.add_argument(
        "--lowering-run",
        required=True,
        type=Path,
        help="Path to a run directory whose 00_graph_capture/ + 01_payload_lowering/ will be analyzed.",
    )
    gap.add_argument("--target", required=True, type=Path, help="Path to target config YAML.")
    gap.add_argument("--out", required=True, type=Path, help="Output run directory (will be replaced).")
    gap.add_argument("--run-id", default=None, help="Optional explicit run_id.")
    gap.add_argument(
        "--extension-registry",
        type=Path,
        default=None,
        help="Path to extension registry.yaml; closed targets are skipped.",
    )

    # extension {materialize, verify, register}
    ext = sub.add_parser("extension", help="Materialize / verify / register a user-space extension.")
    ext_sub = ext.add_subparsers(dest="ext_command", required=True)

    ext_mat = ext_sub.add_parser("materialize", help="Build a workspace from one gap_action_queue entry.")
    ext_mat.add_argument("--queue", required=True, type=Path, help="Path to gap_action_queue.json.")
    ext_mat.add_argument("--gap-id", required=True, help="gap_id selector.")
    ext_mat.add_argument(
        "--extensions-root",
        type=Path,
        default=None,
        help="Root dir; default <repo>/.crg-artifacts/extensions.",
    )
    ext_mat.add_argument(
        "--target-id",
        default="host_cpu",
        help="Target identifier baked into the extension_id. Default: host_cpu.",
    )

    ext_fill = ext_sub.add_parser(
        "fill",
        help="Run the deterministic agent fill (writes extension.py) for an already-materialized workspace.",
    )
    ext_fill.add_argument("--extension", required=True, type=Path, help="Path to a materialized workspace.")

    ext_ver = ext_sub.add_parser(
        "verify",
        help="Run differential tests + locked-files audit on a filled workspace.",
    )
    ext_ver.add_argument("--extension", required=True, type=Path)

    ext_reg = ext_sub.add_parser("register", help="Add a verified extension to a registry.yaml.")
    ext_reg.add_argument("--extension", required=True, type=Path)
    ext_reg.add_argument("--registry", required=True, type=Path)

    ext_ls = ext_sub.add_parser(
        "list-pending",
        help="List materialized-but-unfilled extension workspaces (entry point for Claude Code).",
    )
    ext_ls.add_argument(
        "--extensions-root",
        type=Path,
        default=None,
        help="Root dir; default <repo>/.crg-artifacts/extensions.",
    )
    ext_ls.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. ``text`` is human-readable; ``json`` is machine-parseable.",
    )

    # Top-level convenience aliases for the agent loop (spec 04/05/06):
    # ``materialize-extension``, ``verify-extension``, ``register-extension``.
    # These delegate to the corresponding ``extension <sub>`` handlers.
    mat_ext = sub.add_parser(
        "materialize-extension",
        help="Materialize one extension workspace from a gap_action_queue entry (spec 04).",
    )
    mat_ext.add_argument("--queue", required=True, type=Path, help="Path to gap_action_queue.json.")
    mat_ext.add_argument("--gap-id", required=True, help="gap_id selector.")
    mat_ext.add_argument(
        "--extensions-root",
        type=Path,
        default=None,
        help="Root dir; default <repo>/.crg-artifacts/extensions.",
    )
    mat_ext.add_argument(
        "--target-id",
        default=None,
        help="Target identifier; default = the gap's own target_id.",
    )

    ver_ext = sub.add_parser(
        "verify-extension",
        help="Verify a filled extension workspace; emit verification.json + report (spec 05).",
    )
    ver_ext.add_argument("--extension", required=True, type=Path)
    ver_ext.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional results dir for the human-facing report; verification.json is always written into the workspace's results/.",
    )

    reg_ext = sub.add_parser(
        "register-extension",
        help="Register a verified extension into the registry (spec 06).",
    )
    reg_ext.add_argument("--extension", required=True, type=Path)
    reg_ext.add_argument("--registry", required=True, type=Path)

    plan_ext = sub.add_parser(
        "plan-extensions",
        help=(
            "Aggregate per-model gap_priority_plan.json across a directory "
            "of gap-discovery runs into one ranked extension backlog "
            "(extension_backlog.csv + materialization_plan.json + claude_code_todo.md)."
        ),
    )
    plan_ext.add_argument(
        "--suite-results",
        required=True,
        type=Path,
        help="Directory containing one gap-discovery run per subdir (each with 03_gap_discovery/ — legacy 02_ also accepted).",
    )
    plan_ext.add_argument("--out", required=True, type=Path,
                          help="Output directory for the planning artifacts.")
    plan_ext.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="Cap the backlog at this many entries (after global ranking).",
    )
    plan_ext.add_argument(
        "--include-noncritical",
        action="store_true",
        help="Include noncritical gaps in the backlog (off by default).",
    )

    mat_all = sub.add_parser(
        "materialize-all-extensions",
        help=(
            "Batch-materialize workspaces for every gap (or a filtered subset) "
            "in a gap_action_queue.json; emits materialization_plan.json."
        ),
    )
    mat_all.add_argument("--queue", required=True, type=Path,
                         help="Path to gap_action_queue.json.")
    mat_all.add_argument("--extensions-root", type=Path, default=None,
                         help="Root dir; default <repo>/.crg-artifacts/extensions.")
    mat_all.add_argument("--target-id", default=None,
                         help="Override target_id; default = each gap's own.")
    mat_all.add_argument("--max-gaps", type=int, default=None,
                         help="Materialize at most this many gaps (after filtering).")
    mat_all.add_argument(
        "--severity",
        action="append",
        default=None,
        help=(
            "Only materialize gaps with this severity. Repeatable. "
            "Default: critical_path + performance_blocker + coverage_gap "
            "(noncritical is skipped because the deterministic fallback is fine)."
        ),
    )
    mat_all.add_argument(
        "--include-noncritical",
        action="store_true",
        help="Include noncritical gaps in the batch (off by default).",
    )
    mat_all.add_argument(
        "--plan-out", type=Path, default=None,
        help=(
            "Where to write materialization_plan.json. "
            "Default: <queue.parent>/materialization_plan.json."
        ),
    )
    mat_all.add_argument(
        "--gap-kinds",
        action="append",
        default=None,
        help="Restrict to these gap_kind values (repeatable). Default: all kinds.",
    )

    # replay-goldens
    replay = sub.add_parser(
        "replay-goldens",
        help="Reload exported_program.pt2 and goldens; assert numerical equality.",
    )
    replay.add_argument("--run", required=True, type=Path, help="Path to a run directory.")
    replay.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Model config (only needed for the eager fallback when no exported_program.pt2 exists).",
    )

    # compare
    compare = sub.add_parser(
        "compare",
        help="Diff stable fields between two graph compilation runs; ignores wall-clock fields.",
    )
    compare.add_argument("--a", required=True, type=Path, help="First run directory.")
    compare.add_argument("--b", required=True, type=Path, help="Second run directory.")
    compare.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the determinism report JSON.",
    )

    return parser


# --------------------------------------------------------------------------- #
# validate
# --------------------------------------------------------------------------- #


def _run_validate(run_dir: Path, extra_report: Path | None) -> int:
    if not run_dir.exists():
        print(f"error: run directory does not exist: {run_dir}", file=sys.stderr)
        return 2
    if not run_dir.is_dir():
        print(f"error: run path is not a directory: {run_dir}", file=sys.stderr)
        return 2

    report = validate_run(run_dir)
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    in_tree_path = validation_dir / "artifact_validation.json"
    in_tree_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if extra_report is not None:
        extra_report.parent.mkdir(parents=True, exist_ok=True)
        extra_report.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # Also run payload-lowering / gap-discovery / gap-closure validators.
    from compgen.graph_compilation.gap_closure_validate import (
        validate_gap_closure,
        write_closure_validation_report,
    )
    from compgen.graph_compilation.gap_validate import (
        validate_gap_discovery,
        write_gap_validation_report,
    )
    from compgen.graph_compilation.lowering_validate import (
        validate_payload_lowering,
        write_lowering_validation_report,
    )

    lowering_report = validate_payload_lowering(run_dir)
    write_lowering_validation_report(run_dir, lowering_report)

    gap_report = validate_gap_discovery(run_dir)
    write_gap_validation_report(run_dir, gap_report)

    closure_report = validate_gap_closure(run_dir)
    write_closure_validation_report(run_dir, closure_report)

    print(f"run_dir: {report.run_dir}")
    print(f"overall: {report.overall}")
    for rule in report.rules:
        marker = {"pass": "ok ", "fail": "FAIL", "skipped": "skip"}[rule.status]
        path_suffix = f" ({rule.offending_path})" if rule.offending_path else ""
        print(f"  [{marker}] {rule.rule_id}: {rule.detail}{path_suffix}")
    print(f"lowering_validation: {lowering_report.status}")
    for lc in lowering_report.checks:
        marker = {"pass": "ok ", "fail": "FAIL", "skipped": "skip"}[lc.status]
        print(f"  [{marker}] {lc.name}: {lc.detail}")
    print(f"gap_validation: {gap_report.status}")
    for gc in gap_report.checks:
        marker = {"pass": "ok ", "fail": "FAIL", "skipped": "skip"}[gc.status]
        print(f"  [{marker}] {gc.name}: {gc.detail}")
    print(f"closure_validation: {closure_report.status}")
    for cc in closure_report.checks:
        marker = {"pass": "ok ", "fail": "FAIL", "skipped": "skip"}[cc.status]
        print(f"  [{marker}] {cc.name}: {cc.detail}")
    artifact_ok = report.overall == "pass"
    lowering_ok = lowering_report.status == "pass"
    gap_ok = gap_report.status == "pass"
    # ``not_applicable`` means Gap Closure wasn't run for this run dir;
    # treat as non-failing (the run can still be valid through stage 02).
    closure_ok = closure_report.status in ("pass", "not_applicable")
    return 0 if (artifact_ok and lowering_ok and gap_ok and closure_ok) else 1


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def _run_pipeline(
    model: Path,
    target: Path,
    out: Path,
    stop_after: str,
    run_id: str | None,
    extension_registry: Path | None = None,
    extensions_root: Path | None = None,
    selection_mode: str = "greedy",
    agent_decision_response_paths: list[Path] | None = None,
    agent_max_retries: int = 3,
    live_provider_config: object | None = None,
    resume_from: str | None = None,
    auction_mode: str = "multi-bidder",
    bid_cutoff: int = 3,
    kernel_coverage_mode: str = "both",
) -> int:
    from compgen.graph_compilation.run import run_graph_compilation

    if not model.exists():
        print(f"error: model config not found: {model}", file=sys.stderr)
        return 2
    if not target.exists():
        print(f"error: target config not found: {target}", file=sys.stderr)
        return 2
    if selection_mode == "agent-file" and not agent_decision_response_paths:
        print(
            "error: --selection-mode=agent-file requires --agent-decision-response",
            file=sys.stderr,
        )
        return 2

    # Pass the FIRST path as the single arg (used by llm-live which
    # doesn't iterate); the iterative wrapper uses the full list when
    # len > 1 (agent-file retry).
    primary = (
        agent_decision_response_paths[0]
        if agent_decision_response_paths else None
    )
    rest = (
        list(agent_decision_response_paths[1:])
        if agent_decision_response_paths and len(agent_decision_response_paths) > 1
        else None
    )
    result = run_graph_compilation(
        model_config_path=model,
        target_config_path=target,
        out_dir=out,
        stop_after=stop_after,
        run_id=run_id,
        extension_registry=extension_registry,
        extensions_root=extensions_root,
        selection_mode=selection_mode,
        agent_decision_response_path=primary,
        agent_decision_response_paths=rest,
        agent_max_retries=agent_max_retries,
        live_provider_config=live_provider_config,
        resume_from=resume_from,
        auction_mode=auction_mode,
        bid_cutoff=bid_cutoff,
        kernel_coverage_mode=kernel_coverage_mode,
    )
    print(f"run_dir: {result.run_dir}")
    for s in result.stages:
        print(f"  [{s.status}] {s.stage_id}: {len(s.outputs)} artifact(s)")
    return 0


def _run_lower(capture_run: Path, target: Path, out: Path, run_id: str | None) -> int:
    from compgen.graph_compilation.run import lower_from_existing_capture

    if not capture_run.is_dir():
        print(f"error: --capture-run does not exist: {capture_run}", file=sys.stderr)
        return 2
    if not target.exists():
        print(f"error: target config not found: {target}", file=sys.stderr)
        return 2

    result = lower_from_existing_capture(
        capture_run=capture_run,
        target_config_path=target,
        out_dir=out,
        run_id=run_id,
    )
    print(f"run_dir: {result.run_dir}")
    for s in result.stages:
        print(f"  [{s.status}] {s.stage_id}: {len(s.outputs)} artifact(s)")
    return 0


def _run_suite(args: argparse.Namespace) -> int:
    """Execute the full pipeline for every model in a graph_compilation_suite_v1 YAML.

    Writes one subdirectory per model_id under ``--out``. Records a
    machine-readable ``suite_run_report.json`` at the suite root.
    """
    import time as _time
    from datetime import UTC, datetime

    import yaml

    suite_path: Path = args.suite
    out_root: Path = args.out
    stop_after: str = args.stop_after
    if not suite_path.exists():
        print(f"error: suite YAML not found: {suite_path}", file=sys.stderr)
        return 2
    out_root.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "graph_compilation_suite_v1":
        print(
            f"error: suite schema_version must be graph_compilation_suite_v1, got "
            f"{raw.get('schema_version')!r}",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parents[3]
    target_default = raw.get("target")
    target_path: Path = args.target or (
        repo_root / target_default if target_default else None  # type: ignore[arg-type]
    )
    if target_path is None or not target_path.exists():
        print(f"error: target config missing or not found: {target_path}", file=sys.stderr)
        return 2

    from compgen.graph_compilation.run import run_graph_compilation

    suite_id = raw.get("suite_id", suite_path.stem)
    started = datetime.now(tz=UTC)
    results: list[dict[str, object]] = []
    overall_status = "pass"
    for entry in raw.get("models", []):
        model_id = entry.get("id")
        cfg_rel = entry.get("config")
        if not model_id or not cfg_rel:
            print(f"error: model entry missing id or config: {entry}", file=sys.stderr)
            overall_status = "fail"
            continue
        model_cfg = repo_root / cfg_rel
        if not model_cfg.exists():
            results.append({
                "model_id": model_id,
                "status": "fail",
                "reason": f"model config not found: {model_cfg}",
            })
            overall_status = "fail"
            if not args.continue_on_failure:
                break
            continue

        run_dir = out_root / model_id
        t0 = _time.time()
        # M-14A run-suite agent-file mode: look up a per-model
        # response file in --agent-decision-response-dir.
        agent_response_path: Path | None = None
        adr_dir = getattr(args, "agent_decision_response_dir", None)
        if adr_dir is not None:
            candidate = adr_dir / f"{model_id}.json"
            if candidate.exists():
                agent_response_path = candidate
        live_cfg: object | None = None
        if getattr(args, "selection_mode", "greedy") == "llm-live":
            from compgen.graph_compilation.agent_decision import (
                LiveProviderConfig,
            )
            live_cfg = LiveProviderConfig(
                provider=getattr(args, "llm_live_provider", "gemini"),
                model=getattr(args, "llm_live_model", None),
                timeout_sec=getattr(args, "llm_live_timeout_sec", 60),
                dry_run=getattr(args, "llm_live_dry_run", False),
                fallback=getattr(args, "llm_live_fallback", "none"),
            )
        try:
            res = run_graph_compilation(
                model_config_path=model_cfg,
                target_config_path=target_path,
                out_dir=run_dir,
                stop_after=stop_after,
                run_id=f"suite_{suite_id}_{model_id}",
                extension_registry=args.extension_registry,
                selection_mode=getattr(args, "selection_mode", "greedy"),
                agent_decision_response_path=agent_response_path,
                live_provider_config=live_cfg,
            )
            dt = _time.time() - t0
            stage_summary = [{"stage_id": s.stage_id, "status": s.status,
                              "outputs": len(s.outputs)} for s in res.stages]
            results.append({
                "model_id": model_id,
                "status": "pass",
                "run_dir": str(res.run_dir),
                "wall_time_seconds": round(dt, 3),
                "stages": stage_summary,
            })
            print(f"  [pass] {model_id} ({dt:.2f}s) → {res.run_dir}")
        except Exception as exc:  # honest failure recording
            dt = _time.time() - t0
            results.append({
                "model_id": model_id,
                "status": "fail",
                "reason": f"{type(exc).__name__}: {exc}",
                "wall_time_seconds": round(dt, 3),
            })
            overall_status = "fail"
            print(f"  [fail] {model_id} ({dt:.2f}s): {exc}", file=sys.stderr)
            if not args.continue_on_failure:
                break

    finished = datetime.now(tz=UTC)
    report = {
        "schema_version": "suite_run_report_v1",
        "suite_id": suite_id,
        "suite_path": str(suite_path.resolve()),
        "stop_after": stop_after,
        "started_at_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished_at_utc": finished.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_time_seconds": round((finished - started).total_seconds(), 3),
        "overall": overall_status,
        "results": results,
    }
    (out_root / "suite_run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nsuite_run_report: {out_root / 'suite_run_report.json'}")
    print(f"overall: {overall_status}  ({len(results)} model(s))")
    return 0 if overall_status == "pass" else 1


def _run_resolve_candidate(args: argparse.Namespace) -> int:
    """Resolve a candidate_id against action_space.mlir (M-04.5)."""
    from compgen.graph_compilation.action_space_resolver import (
        ResolverError,
        resolve_candidate,
    )

    rationale = (
        {"primary_reason": args.rationale_primary, "evidence": []}
        if args.rationale_primary
        else None
    )
    try:
        resolved, report = resolve_candidate(
            run_dir=args.run,
            candidate_id=args.candidate_id,
            allow_illegal=args.allow_illegal,
            selection_mode=args.selection_mode,
            rationale=rationale,
            write_outputs=not args.no_write,
        )
    except ResolverError as exc:
        print(f"resolve FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"candidate_id:        {resolved.candidate_id}")
    print(f"site_id:             {resolved.site_id}")
    print(f"region_id:           {resolved.region_id}")
    print(f"kind:                {resolved.kind}")
    print(f"legality_ok:         {resolved.legality_ok}")
    if not resolved.legality_ok:
        print(f"legality_reason:     {resolved.legality_reason}")
    print(f"action_space_ir_sha: {resolved.source['action_space_ir_sha256'][:24]}...")
    print(f"resolver:            {report.overall}")
    for c in report.checks:
        print(f"  [{c['status']}] {c['name']}: {c['detail'][:80]}")
    return 0


def _run_discover_target(args: argparse.Namespace) -> int:
    """Auto-discover the host's CPU + accelerator info and write a
    partial target YAML compatible with graphcomp_target_config_v1.

    Fields that cannot be probed reliably (peak DRAM bandwidth without
    a benchmark, accelerator peak FLOPS) are emitted with sensible
    defaults or ``null`` placeholders the operator can fill in.
    """
    from compgen.graph_compilation.target_discovery import build_target_yaml

    out: Path = args.out
    obj = build_target_yaml(out_path=out, target_id=args.target_id)
    print(f"target_id:           {obj['target_id']}")
    cpu = obj["discovery_provenance"]["cpu"]
    print(
        f"cpu:                 {cpu['vendor_id']} {cpu['model_name']} "
        f"({cpu['physical_cores']}c/{cpu['logical_cores']}t @ "
        f"{cpu['max_freq_mhz'] or cpu['base_freq_mhz']:.0f} MHz)"
    )
    print(f"peak_compute_gflops: {obj['peak_compute_gflops']}")
    print(f"peak_bandwidth_gb_s: {obj['peak_bandwidth_gb_s']}")
    mt = obj["memory_tiers"]
    print(
        f"memory:              L1d={mt['scratchpad_bytes']}B "
        f"L2={mt['l2_bytes']}B L3={mt['l3_bytes']}B "
        f"sys={mt['system_bytes']}B"
    )
    accs = obj.get("accelerators_present", [])
    if accs:
        print("accelerators_present:")
        for a in accs:
            print(f"  - {a['kind']}: {a['name']} (×{a['count']}) via {a['detected_via']}")
    else:
        print("accelerators_present: none detected")
    print(f"\nwrote: {out}")
    return 0


def _run_analyze_graph(lowering_run: Path, out: Path, target: Path | None) -> int:
    """Run the Graph Analysis V2 builder against an existing payload-lowering run.

    If ``out == lowering_run`` we write 02_graph_analysis/ in place (no
    copy). If ``out`` is a different directory we first copy
    00_graph_capture/ + 01_payload_lowering/ across, then build the
    analysis JSONs. This mirrors the lower / discover-gaps subcommands'
    "no hidden re-capture" guarantee.
    """
    import shutil

    from compgen.graph_compilation.region_dossier import build_region_dossiers
    from compgen.graph_compilation.region_map import build_graph_analysis

    if not lowering_run.is_dir():
        print(f"error: --lowering-run does not exist: {lowering_run}", file=sys.stderr)
        return 2
    pl = lowering_run / "01_payload_lowering"
    if not pl.is_dir():
        print(f"error: 01_payload_lowering/ missing under {lowering_run}", file=sys.stderr)
        return 2

    if out.resolve() != lowering_run.resolve():
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for sub in ("00_graph_capture", "01_payload_lowering"):
            shutil.copytree(lowering_run / sub, out / sub)

    result = build_graph_analysis(out)
    repo_root = Path(__file__).resolve().parents[3]
    target_yaml = target or (repo_root / "configs" / "targets" / "host_cpu.yaml")
    dossier = build_region_dossiers(out, target_yaml)
    print(f"run_dir: {out}")
    print(f"regions: {result.region_count}")
    print(f"tensors: {result.tensor_count}")
    print(f"edges:   {result.edge_count}")
    print(f"is_dag:  {result.is_dag}")
    print(f"region_dossiers: {dossier.region_dossier_count}")
    print(f"matmul_like: {dossier.matmul_like_count}")
    print(f"opaque: {dossier.opaque_count}")
    return 0


def _run_discover_gaps(
    lowering_run: Path,
    target: Path,
    out: Path,
    run_id: str | None,
    extension_registry: Path | None = None,
) -> int:
    from compgen.graph_compilation.run import discover_gaps_from_existing_lowering

    if not lowering_run.is_dir():
        print(f"error: --lowering-run does not exist: {lowering_run}", file=sys.stderr)
        return 2
    if not target.exists():
        print(f"error: target config not found: {target}", file=sys.stderr)
        return 2
    result = discover_gaps_from_existing_lowering(
        lowering_run=lowering_run,
        target_config_path=target,
        out_dir=out,
        run_id=run_id,
        extension_registry=extension_registry,
    )
    print(f"run_dir: {result.run_dir}")
    for s in result.stages:
        print(f"  [{s.status}] {s.stage_id}: {len(s.outputs)} artifact(s)")
    return 0


# --------------------------------------------------------------------------- #
# extension subcommands
# --------------------------------------------------------------------------- #


def _run_extension(args: argparse.Namespace) -> int:
    import json as _json

    from compgen.graph_compilation.agent_decomp_fill import deterministic_fill
    from compgen.graph_compilation.extension_materialize import materialize_extension
    from compgen.graph_compilation.extension_registry import register_extension
    from compgen.graph_compilation.extension_verify import run_verify

    cmd = getattr(args, "ext_command", None)
    if cmd == "materialize":
        queue_path: Path = args.queue
        if not queue_path.exists():
            print(f"error: queue not found: {queue_path}", file=sys.stderr)
            return 2
        queue = _json.loads(queue_path.read_text(encoding="utf-8"))
        gap = next((g for g in queue.get("gaps", []) if g.get("gap_id") == args.gap_id), None)
        if gap is None:
            print(f"error: gap_id={args.gap_id!r} not in queue", file=sys.stderr)
            return 2
        ext_root = args.extensions_root or (Path.cwd() / ".crg-artifacts" / "extensions")
        mr = materialize_extension(gap, target_id=args.target_id, extensions_root=ext_root)
        print(f"materialized: {mr.extension_dir}")
        return 0
    if cmd == "fill":
        if not args.extension.is_dir():
            print(f"error: extension dir not found: {args.extension}", file=sys.stderr)
            return 2
        gap = _json.loads((args.extension / "gap_record.json").read_text(encoding="utf-8"))
        path = deterministic_fill(args.extension, gap["fx_target"])
        print(f"filled: {path}")
        return 0
    if cmd == "verify":
        if not args.extension.is_dir():
            print(f"error: extension dir not found: {args.extension}", file=sys.stderr)
            return 2
        result = run_verify(args.extension)
        print(_json.dumps(result.to_dict(), indent=2, sort_keys=True))
        # Optional --out: emit the four spec-required reports under a results dir.
        out_dir = getattr(args, "out", None)
        if out_dir is not None:
            from compgen.graph_compilation.extension_verify import emit_extension_reports

            emit_extension_reports(workspace=args.extension, out_dir=out_dir, verify_result=result)
        return 0 if result.status == "pass" else 1
    if cmd == "register":
        if not args.extension.is_dir():
            print(f"error: extension dir not found: {args.extension}", file=sys.stderr)
            return 2
        result = run_verify(args.extension)
        if result.status != "pass":
            print(f"error: extension does not verify: {result.detail}", file=sys.stderr)
            return 1
        entry = register_extension(
            workspace=args.extension,
            verification_result=result,
            registry_path=args.registry,
        )
        print(f"registered: {entry.extension_id} → {args.registry}")
        return 0
    if cmd == "list-pending":
        ext_root = args.extensions_root or (Path.cwd() / ".crg-artifacts" / "extensions")
        if not ext_root.is_dir():
            if args.format == "json":
                print(_json.dumps({"pending": [], "registered": [], "extensions_root": str(ext_root)}, indent=2))
            else:
                print(f"(no extensions root at {ext_root})")
            return 0

        from compgen.graph_compilation.extension_registry import load_registry

        registry_path = ext_root / "registry.yaml"
        registry = load_registry(registry_path)
        registered_ids = {e.extension_id for e in registry.entries}

        pending: list[dict[str, str]] = []
        registered: list[dict[str, str]] = []
        for kind_dir in sorted(ext_root.iterdir()):
            if not kind_dir.is_dir() or kind_dir.name == "__pycache__":
                continue
            for ws in sorted(kind_dir.iterdir()):
                if not ws.is_dir():
                    continue
                contract_path = ws / "extension_contract.json"
                if not contract_path.exists():
                    continue
                contract = _json.loads(contract_path.read_text(encoding="utf-8"))
                entry = {
                    "extension_id": contract["extension_id"],
                    "gap_kind": contract["gap_kind"],
                    "fx_target": contract["fx_target"],
                    "extension_path": str(ws.resolve()),
                    "fillable_files": contract["fillable_files"],
                }
                if contract["extension_id"] in registered_ids:
                    registered.append(entry)
                else:
                    pending.append(entry)

        if args.format == "json":
            print(_json.dumps({
                "pending": pending,
                "registered": registered,
                "extensions_root": str(ext_root),
                "registry_path": str(registry_path),
            }, indent=2, sort_keys=True))
            return 0

        # Human-readable text output.
        print(f"extensions_root: {ext_root}")
        print(f"registry_path:   {registry_path}")
        print("")
        print(f"PENDING (need fill):  {len(pending)}")
        for e in pending:
            print(f"  • {e['fx_target']}  ({e['gap_kind']})")
            print(f"      workspace: {e['extension_path']}")
            print(f"      edit only: {', '.join(e['fillable_files'])}")
        print("")
        print(f"REGISTERED:           {len(registered)}")
        for e in registered:
            print(f"  ✓ {e['fx_target']}  ({e['gap_kind']})")
        return 0
    print("error: unknown extension subcommand", file=sys.stderr)
    return 2


# --------------------------------------------------------------------------- #
# materialize-all-extensions
# --------------------------------------------------------------------------- #


def _run_materialize_all(args: argparse.Namespace) -> int:
    """Batch-materialize workspaces from a gap_action_queue.

    Filters by severity (default: skip ``noncritical``) and ``--gap-kinds``,
    caps with ``--max-gaps``, and writes ``materialization_plan.json``
    summarising what was materialized vs skipped and why.
    """
    import json as _json
    from datetime import UTC, datetime

    from compgen.graph_compilation.extension_materialize import materialize_extension

    queue_path: Path = args.queue
    if not queue_path.exists():
        print(f"error: queue not found: {queue_path}", file=sys.stderr)
        return 2
    queue = _json.loads(queue_path.read_text(encoding="utf-8"))
    gaps = queue.get("gaps", [])

    ext_root: Path = args.extensions_root or (Path.cwd() / ".crg-artifacts" / "extensions")
    ext_root.mkdir(parents=True, exist_ok=True)

    # Severity filter — by default we skip ``noncritical`` because the
    # deterministic fallback handles those without an extension.
    if args.severity:
        allowed_severity = set(args.severity)
    else:
        allowed_severity = {"critical_path", "performance_blocker", "coverage_gap"}
        if getattr(args, "include_noncritical", False):
            allowed_severity.add("noncritical")
    allowed_kinds = set(args.gap_kinds) if args.gap_kinds else None
    max_gaps = args.max_gaps

    selected: list[dict] = []
    skipped: list[dict] = []
    for g in gaps:
        sev = g.get("severity", "")
        if sev not in allowed_severity:
            skipped.append({
                "gap_id": g["gap_id"],
                "fx_target": g.get("fx_target", ""),
                "reason": f"severity_filtered:{sev}",
            })
            continue
        if allowed_kinds is not None and g.get("gap_kind") not in allowed_kinds:
            skipped.append({
                "gap_id": g["gap_id"],
                "fx_target": g.get("fx_target", ""),
                "reason": f"gap_kind_filtered:{g.get('gap_kind')}",
            })
            continue
        if max_gaps is not None and len(selected) >= max_gaps:
            skipped.append({
                "gap_id": g["gap_id"],
                "fx_target": g.get("fx_target", ""),
                "reason": "max_gaps_reached",
            })
            continue
        selected.append(g)

    # Materialize the selected gaps.
    workspace_paths: list[dict] = []
    materialize_failures: list[dict] = []
    for g in selected:
        try:
            target_id = args.target_id or g.get("target_id", "host_cpu")
            mr = materialize_extension(g, target_id=target_id, extensions_root=ext_root)
            workspace_paths.append({
                "gap_id": g["gap_id"],
                "fx_target": g.get("fx_target", ""),
                "extension_id": mr.extension_id,
                "extension_path": str(mr.extension_dir),
                "severity": g.get("severity", ""),
            })
        except Exception as exc:  # honest failure recording
            materialize_failures.append({
                "gap_id": g["gap_id"],
                "fx_target": g.get("fx_target", ""),
                "error": f"{type(exc).__name__}: {exc}",
            })

    plan = {
        "schema_version": "materialization_plan_v1",
        "queue_path": str(queue_path.resolve()),
        "extensions_root": str(ext_root.resolve()),
        "filters": {
            "severity": sorted(allowed_severity),
            "gap_kinds": sorted(allowed_kinds) if allowed_kinds else None,
            "max_gaps": max_gaps,
        },
        "totals": {
            "total_gaps": len(gaps),
            "selected_gaps": len(selected),
            "materialized_gaps": len(workspace_paths),
            "skipped_gaps": len(skipped),
            "failed_gaps": len(materialize_failures),
        },
        "materialized": workspace_paths,
        "skipped": skipped,
        "failures": materialize_failures,
        "generated_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    plan_path: Path = args.plan_out or (queue_path.parent / "materialization_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(_json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(_json.dumps(plan["totals"], indent=2, sort_keys=True))
    print(f"plan written: {plan_path}")
    return 1 if materialize_failures else 0


# --------------------------------------------------------------------------- #
# plan-extensions (08 — batch extension planning)
# --------------------------------------------------------------------------- #


def _run_plan_extensions(args: argparse.Namespace) -> int:
    """Aggregate per-model ``gap_priority_plan.json`` into one global
    ranked backlog and emit:

      - ``extension_backlog.csv``     — per-row: rank, model, gap_id, ...
      - ``materialization_plan.json`` — machine-readable summary + per-row
      - ``claude_code_todo.md``       — Markdown checklist Claude Code can fill
    """
    import csv as _csv
    import json as _json
    from datetime import UTC, datetime

    suite: Path = args.suite_results
    out: Path = args.out
    if not suite.is_dir():
        print(f"error: --suite-results is not a directory: {suite}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    bucket_rank = {
        "critical_path": 0,
        "performance_blocker": 1,
        "coverage_gap": 2,
        "noncritical": 3,
    }

    rows: list[dict] = []
    skipped_runs: list[dict] = []
    for child in sorted(suite.iterdir()):
        if not child.is_dir():
            continue
        from compgen.graph_compilation.artifacts import stage_dir as _stage_dir
        gd_dir = _stage_dir(child, "gap_discovery")
        assert isinstance(gd_dir, Path)
        plan_path = gd_dir / "gap_priority_plan.json"
        if not plan_path.exists():
            skipped_runs.append({"run": child.name,
                                 "reason": f"missing {gd_dir.name}/gap_priority_plan.json"})
            continue
        plan = _json.loads(plan_path.read_text(encoding="utf-8"))
        model_id = plan.get("model_id", child.name)
        for g in plan["ordered_gaps"]:
            if g["severity"] == "noncritical" and not getattr(args, "include_noncritical", False):
                continue
            rows.append({
                "rank": 0,  # filled after global sort
                "model": model_id,
                "model_run_dir": str(child.relative_to(suite)),
                "gap_id": g["gap_id"],
                "semantic_name": g["semantic_name"],
                "fx_target": g["fx_target"],
                "severity": g["severity"],
                "severity_score": g["severity_score"],
                "estimated_cost_fraction": g["cost_fraction_estimate"],
                "recommended_action": g["recommended_next_action"],
                "extension_id": g["extension_id"],
                "workspace_path": g["suggested_extension_path"],
            })

    rows.sort(key=lambda r: (
        bucket_rank.get(r["severity"], 4),
        -float(r["severity_score"]),
        -float(r["estimated_cost_fraction"]),
        r["model"],
        r["gap_id"],
    ))
    max_rank = getattr(args, "max_rank", None)
    if max_rank is not None:
        rows = rows[:max_rank]
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    # extension_backlog.csv
    backlog_path = out / "extension_backlog.csv"
    if rows:
        with backlog_path.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        backlog_path.write_text("# (no actionable gaps found across suite)\n", encoding="utf-8")

    # materialization_plan.json
    by_severity: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for r in rows:
        by_severity[r["severity"]] = by_severity.get(r["severity"], 0) + 1
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1
    plan_obj = {
        "schema_version": "extension_planning_v1",
        "generated_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "suite_results": str(suite.resolve()),
        "totals": {
            "total_runs_scanned": sum(1 for c in suite.iterdir() if c.is_dir()),
            "runs_with_plan": len({r["model"] for r in rows}),
            "skipped_runs": skipped_runs,
            "actionable_gaps": len(rows),
            "by_severity": dict(sorted(by_severity.items())),
            "by_model": dict(sorted(by_model.items())),
        },
        "backlog": rows,
    }
    plan_json_path = out / "materialization_plan.json"
    plan_json_path.write_text(_json.dumps(plan_obj, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")

    # claude_code_todo.md
    md_lines: list[str] = []
    md_lines.append("# Claude Code · extension backlog\n\n")
    md_lines.append(
        f"Generated `{plan_obj['generated_at_utc']}` from `{suite.resolve()}` — "
        f"{len(rows)} actionable gaps across {len(by_model)} model(s).\n\n"
    )
    if skipped_runs:
        md_lines.append("## Skipped runs\n")
        for s in skipped_runs:
            md_lines.append(f"- `{s['run']}` — {s['reason']}\n")
        md_lines.append("\n")

    md_lines.append("## Severity histogram\n")
    for sev, n in sorted(by_severity.items(), key=lambda kv: bucket_rank.get(kv[0], 4)):
        md_lines.append(f"- **{sev}**: {n}\n")
    md_lines.append("\n## Backlog (ranked)\n")
    cur_sev = None
    for r in rows:
        if r["severity"] != cur_sev:
            cur_sev = r["severity"]
            md_lines.append(f"\n### {cur_sev}\n")
        md_lines.append(
            f"- [ ] **rank {r['rank']:>3d}** · `{r['model']}` · `{r['fx_target']}`\n"
            f"      gap_id=`{r['gap_id']}`, "
            f"cost≈{r['estimated_cost_fraction']:.3f}, "
            f"action=`{r['recommended_action']}`, "
            f"workspace=`{r['workspace_path']}`\n"
        )
    todo_path = out / "claude_code_todo.md"
    todo_path.write_text("".join(md_lines), encoding="utf-8")

    print(_json.dumps(plan_obj["totals"], indent=2, sort_keys=True))
    print(f"backlog → {backlog_path}")
    print(f"plan    → {plan_json_path}")
    print(f"todo    → {todo_path}")
    return 0


# --------------------------------------------------------------------------- #
# replay-goldens
# --------------------------------------------------------------------------- #


def _run_replay(run_dir: Path, model_config: Path | None) -> int:
    from compgen.graph_compilation.replay import replay_goldens, write_replay_report

    if not run_dir.is_dir():
        print(f"error: run directory does not exist: {run_dir}", file=sys.stderr)
        return 2
    result = replay_goldens(run_dir, model_config=model_config)
    write_replay_report(run_dir, result)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "pass" else 1


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #


def _run_compare(a: Path, b: Path, extra_report: Path | None) -> int:
    from compgen.graph_compilation.compare import compare_runs

    if not a.is_dir():
        print(f"error: --a does not exist: {a}", file=sys.stderr)
        return 2
    if not b.is_dir():
        print(f"error: --b does not exist: {b}", file=sys.stderr)
        return 2
    report = compare_runs(a, b)
    out_dir = a / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "determinism_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if extra_report is not None:
        extra_report.parent.mkdir(parents=True, exist_ok=True)
        extra_report.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"overall: {report.overall}")
    print(f"matches:    {len(report.matches)}")
    print(f"mismatches: {len(report.mismatches)}")
    for m in report.mismatches:
        print(f"  [DIFF] {m['field']}: a={m['a']!r} b={m['b']!r}")
    return 0 if report.overall == "pass" else 1


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _run_validate(args.run, args.report)
        if args.command == "run":
            from compgen.graph_compilation.agent_decision import LiveProviderConfig
            live_cfg: object | None = None
            if getattr(args, "selection_mode", "greedy") == "llm-live":
                live_cfg = LiveProviderConfig(
                    provider=getattr(args, "llm_live_provider", "gemini"),
                    model=getattr(args, "llm_live_model", None),
                    timeout_sec=getattr(args, "llm_live_timeout_sec", 60),
                    dry_run=getattr(args, "llm_live_dry_run", False),
                    fallback=getattr(args, "llm_live_fallback", "none"),
                )
            # M-62: re-index user-supplied kernels before the run
            # (auction picks them up via default_registry()).
            user_kernel_path = getattr(args, "user_kernel_path", None)
            try:
                from compgen.kernels.user_kernel_index import (
                    default_index_root,
                    reindex,
                    resolve_user_kernel_path,
                )

                resolved_path = resolve_user_kernel_path(cli_path=user_kernel_path)
                if resolved_path is not None and resolved_path.exists():
                    reindex(
                        search_path=resolved_path,
                        index_root=default_index_root(),
                    )
            except Exception:  # noqa: BLE001 — best-effort; surfaced via MCP discover tool
                pass

            return _run_pipeline(
                args.model, args.target, args.out, args.stop_after, args.run_id,
                extension_registry=args.extension_registry,
                extensions_root=args.extensions_root,
                selection_mode=getattr(args, "selection_mode", "greedy"),
                agent_decision_response_paths=getattr(
                    args, "agent_decision_response", None,
                ),
                agent_max_retries=getattr(args, "agent_max_retries", 3),
                live_provider_config=live_cfg,
                resume_from=getattr(args, "resume_from", None),
                auction_mode=getattr(args, "auction_mode", "multi-bidder"),
                bid_cutoff=getattr(args, "bid_cutoff", 3),
                kernel_coverage_mode=getattr(
                    args, "kernel_coverage_mode", "both",
                ),
            )
        if args.command == "lower":
            return _run_lower(args.capture_run, args.target, args.out, args.run_id)
        if args.command == "run-suite":
            return _run_suite(args)
        if args.command == "resolve-candidate":
            return _run_resolve_candidate(args)
        if args.command == "discover-target":
            return _run_discover_target(args)
        if args.command == "analyze-graph":
            return _run_analyze_graph(args.lowering_run, args.out, args.target)
        if args.command == "discover-gaps":
            return _run_discover_gaps(
                args.lowering_run, args.target, args.out, args.run_id,
                extension_registry=args.extension_registry,
            )
        if args.command == "extension":
            return _run_extension(args)
        if args.command == "materialize-extension":
            # Reshape into the same form _run_extension expects.
            args.ext_command = "materialize"
            if getattr(args, "target_id", None) is None:
                # Default the target_id from the queue's gap if not given.
                import json as _j
                queue = _j.loads(args.queue.read_text(encoding="utf-8"))
                gap = next((g for g in queue.get("gaps", []) if g.get("gap_id") == args.gap_id), None)
                args.target_id = (gap or {}).get("target_id", "host_cpu")
            return _run_extension(args)
        if args.command == "verify-extension":
            args.ext_command = "verify"
            return _run_extension(args)
        if args.command == "register-extension":
            args.ext_command = "register"
            return _run_extension(args)
        if args.command == "materialize-all-extensions":
            return _run_materialize_all(args)
        if args.command == "plan-extensions":
            return _run_plan_extensions(args)
        if args.command == "replay-goldens":
            return _run_replay(args.run, args.model_config)
        if args.command == "compare":
            return _run_compare(args.a, args.b, args.report)
    except Exception as exc:  # exit-code-2 path
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

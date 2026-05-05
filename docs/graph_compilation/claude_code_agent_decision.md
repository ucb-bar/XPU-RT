# Claude Code as the agent decision driver (M-14C)

## Why Claude Code is the primary path

CompGen's agent decision point is filled by **Claude Code running in a CLI session**, not by a direct LLM API call. The compiler emits a bounded decision request; Claude Code reads it, reasons over the structured artifacts, and writes a typed response; the compiler validates and commits. Direct provider integration (Anthropic / OpenAI HTTP API) is a *secondary* path for unattended automation (M-14D).

Reasons:

- **No API key, no billing, no retries.** A Claude Code session is already an authenticated agent surface; reaching out to `api.anthropic.com` just duplicates infrastructure.
- **Repo + tool context.** Claude Code has Read / Edit / Bash / MCP tools and can inspect any artifact under the run directory. A direct API call sees only what the compiler put in the prompt.
- **Human-in-the-loop by default.** When something is ambiguous, Claude Code can ask the user; a headless API call cannot.
- **MCP graduation path.** The decision protocol is exposed as MCP tools (`compgen_emit_agent_decision_request`, `compgen_commit_agent_decision_response`, `compgen_inspect_pipeline_run`), so future on-device or alternative agents plug in cleanly.

For the paper / slides, the right framing is:

> "We define a bounded compiler decision interface. The agent (Claude Code in our reference implementation) inspects structured compiler artifacts via MCP tools, writes a typed response, and the compiler validates it through the same 11-check gate that exists for the file/HTTP/stub paths. The compiler stays in charge of truth, legality, IR state, and verification."

## Architecture

```
                  ┌────────────────────────────┐
                  │  graph_dossier_v3.json     │
                  │  llm_graph_view.json       │
                  │  cost_preview_v2.json      │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │ agent_decision_request.json│   (M-14A: bounded view)
                  └─────────────┬──────────────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
       ▼                        ▼                        ▼
 Claude Code skill        agent-file mode         optional API mode
 + MCP tools              manual response          (M-14D, headless)
 (PRIMARY)                (testing / replay)       (sweeps / CI)
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                ▼
                  ┌────────────────────────────┐
                  │ agent_decision_response.   │
                  │ json (typed v1 schema)     │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │ M-14A validator            │
                  │   1. request_sources_exist │
                  │   2. response_schema_valid │
                  │   3. selected_candidate_   │
                  │      exists                │
                  │   4. ...is_legal           │
                  │   5. ...visible_to_agent   │
                  │   6. ...resolves_against_  │
                  │      action_space.mlir     │
                  │   7. rationale_summary_    │
                  │      present               │
                  │   8. rationale_evidence_   │
                  │      present (≥2)          │
                  │   9. references_real_      │
                  │      fields                │
                  │  10. no_correctness_claim  │
                  │  11. no_measured_perf_     │
                  │      claim                 │
                  └─────────────┬──────────────┘
                                │  pass
                                ▼
                  ┌────────────────────────────┐
                  │ recipe.mlir commit         │
                  │ M-06 → M-07 → M-08 → M-09  │
                  │ → M-11A/B → M-12 → M-13    │
                  └────────────────────────────┘
```

## What Claude Code is allowed to do

```
read agent_decision_request.json
read llm_graph_view.json
read graph_dossier_v3.json
read cost_preview_v2.json
read region_dossiers/*.json (for deeper region context)
read candidate_actions.json (for cross-checks)
call MCP: compgen_emit_agent_decision_request
call MCP: compgen_commit_agent_decision_response
call MCP: compgen_inspect_pipeline_run
call CLI: python -m compgen.graph_compilation run --selection-mode agent-file
write agent_decision_response.json
explain rationale citing real evidence fields
```

## What Claude Code is NOT allowed to do

```
edit payload.mlir directly
edit candidate_actions.json
edit action_space.mlir
edit region_map.json
edit verified_recipe.mlir
edit semantic_obligations.json
edit any *_validation.json or *_verification_report.json
mark obligations as discharged
invent candidate IDs not in candidate_ids_allowed
invent tile sizes
claim correctness, "verified", "guaranteed"
claim measured performance, "benchmarked", "fastest"
call ANTHROPIC_API_KEY-backed providers (that's M-14D, not this path)
```

This is the trust boundary. The compiler holds truth; Claude Code chooses among legal options.

## Two entry-point skills

`.claude/skills/compgen-candidate-selection/` — **atomic decision step**. Read request, write response. No pipeline invocation. Use this when an `agent_decision_request.json` already exists.

`.claude/skills/compgen-compile/` — **full pipeline driver**. Run pipeline → emit request → invoke `/compgen-candidate-selection` mentally → write response → re-run with agent-file → inspect → report. Use this for "compile this model with Claude Code as the agent" workflows.

## Three MCP tools

Registered in `python/compgen/mcp/tools/agent_decision.py`:

| Tool | Purpose | Returns |
|---|---|---|
| `compgen_emit_agent_decision_request` | Run pipeline up to `agent-decision-request` | bounded view + greedy's pick + legal SetTileParams previews |
| `compgen_commit_agent_decision_response` | Re-run pipeline with `--selection-mode agent-file` | validation overall, recipe.mlir excerpt, downstream stage reports |
| `compgen_inspect_pipeline_run` | Read-only health summary of an existing run dir | per-stage status, obligation status, validate_run R001-R012 verdict |

These are the canonical primitives. The skills wrap them; direct CLI usage works too.

## End-to-end example: `merlin_mlp_wide` driven by Claude Code

Step 1 — emit:
```bash
.venv/bin/python -m compgen.graph_compilation run \
  --model configs/models/merlin_mlp_wide.yaml \
  --target configs/targets/host_cpu.yaml \
  --out results/graph_compilation/claude_agent_merlin_mlp_wide \
  --stop-after agent-decision-request \
  --selection-mode greedy
```

Step 2 — Claude Code reads the bounded view, picks the M-12-verified candidate, writes `agent_decision_response.json` to a path that survives the next run (e.g. a sibling tmp dir).

Step 3 — commit:
```bash
.venv/bin/python -m compgen.graph_compilation run \
  --model configs/models/merlin_mlp_wide.yaml \
  --target configs/targets/host_cpu.yaml \
  --out results/graph_compilation/claude_agent_merlin_mlp_wide \
  --stop-after cost-preview-v2 \
  --selection-mode agent-file \
  --agent-decision-response /path/to/agent_decision_response.json
```

Result for `merlin_mlp_wide`:
- Validation: 11/11 pass.
- Recipe.mlir: `source_candidate = "cand_tile_matmul_0_tile_M16_N16_K16__3bebae8e"`.
- Downstream: post-lowering verification pass, differential verification pass (metadata-noop), real-transform-differential pass (16/16 cases at bit-equality).
- Obligation `obl_recipe_0000`: `discharged_real_transform_differential_check`, `remaining: []`.

## Per-run audit artifacts

Under `<run>/03_recipe_planning/agent_decision/`:

| File | Purpose | Always present? |
|---|---|---|
| `agent_decision_request.json` | Bounded view shown to agent | yes (any non-greedy mode or `--stop-after agent-decision-request`) |
| `agent_decision_response.json` | Agent's pick | yes (mirrored from agent's response on commit) |
| `agent_decision_validation.json` | 11-check verdict | yes |
| `agent_decision_trace.json` | Selection mode + provider block + SHA pins | yes |
| `provider_redaction_audit.json` | Defense-in-depth secret-leak scan | yes (passes trivially without keys) |
| `claude_code_decision_notes.md` | Reviewer-facing one-pager | yes when committed via agent-file |
| `agent_decision_provider_request.json` | M-14B/D HTTP provider config | only when `llm-live` mode |
| `agent_decision_provider_response.raw.json` | M-14B/D raw provider output | only when `llm-live` mode |
| `provider_error.json` | M-14B/D typed provider failure | only when `llm-live` errored |

## What changed at M-14C (this milestone)

1. The Claude Code workflow is documented as the *primary* path (this file).
2. Two skills exist: `compgen-candidate-selection` (atomic) + `compgen-compile` (full pipeline).
3. Three MCP tools expose the protocol natively to Claude Code.
4. `claude_code_decision_notes.md` is emitted on every agent-file commit so reviewers can see what was picked and why.
5. The renamed M-14D (formerly M-14C-as-built) keeps the Anthropic / OpenAI HTTP path available for unattended use, but is no longer the headline integration.

## What did NOT change

- The 11-check M-14A validator. Identical for all three response sources.
- The `action_space.mlir` resolver. Identical.
- Recipe IR commit, semantic-obligation creation, M-08 / M-09 / M-11A / M-11B / M-12 / M-13 — all unchanged.
- Compiler trust boundary. Identical.

## Why the API path still exists

For unattended automation that runs without a Claude Code session: paper-reproduction batches, CI sweeps, ablation comparisons across many providers. Those use cases need the HTTP path. They don't justify making it the primary architecture.

## Failure modes and what Claude Code should do

| Failure | Where | Claude Code's job |
|---|---|---|
| Validation fails on a check | `agent_decision_validation.json` | Read the failed check, rewrite the response, retry. |
| Selected candidate doesn't resolve | `selected_candidate_resolves_against_action_space_ir: fail` | Did the agent invent an ID? Read `candidate_ids_allowed` again and pick from there. |
| Forbidden phrase in rationale | `no_correctness_claim` or `no_measured_performance_claim` | Rewrite using neutral language ("fits scratchpad", "M-12 evidence", "lower static_relative_cost"). |
| Downstream stage fails | `post_lowering_verification` / M-12 report | Optionally retry with a different candidate. If 2-3 retries don't converge, surface to user. |
| validate_run R009 hash chain breaks | `validate_run().overall == "fail"` | Should not happen with correct flow; if it does, the run dir is corrupt. Re-run from scratch. |

## Reference: the typed response schema

```json
{
  "schema_version": "agent_decision_response_v1",
  "selected_candidate_id": "<candidate_id from candidate_ids_allowed>",
  "rationale": {
    "summary": "<1-2 sentences, neutral language>",
    "evidence": [
      {"field": "<dotted.path>", "value": <real value>, "reason": "<short justification>"},
      {"field": "<dotted.path>", "value": <real value>, "reason": "<short justification>"}
    ],
    "rejected_alternatives": [
      {"candidate_kind": "<kind>", "reason": "<why not>"}
    ]
  }
}
```

`field` paths the validator resolves against actual artifacts: `candidate.kind`, `candidate.cost_preview.<key>`, `candidate.legality.<key>`, `cost_preview_v2.<key>` (and nested), `graph_dossier_v3.<key>`, `semantic_obligation.<key>`. Anything else fails `rationale_references_real_fields`.

## M-15A: Rejection / Retry Loop

When Claude Code's first response fails validation, the compiler doesn't just abort — it emits a typed retry artifact and lets the agent revise without leaving the protocol:

```
attempt_000 response (Claude Code's first pick)
       │
       ▼
M-14A validator
       │
       ▼ fail (e.g. selected_candidate_is_legal, no_correctness_claim)
       │
       ▼
attempts/attempt_000/agent_decision_response.json   (preserved for audit)
attempts/attempt_000/agent_decision_validation.json (preserved)
attempts/attempt_000/retry_request.json             (typed: failed_checks + candidate_ids_allowed + recommended_debug_fields)
retry_request.json                                  (top-level: latest)
       │
       ▼
Claude Code reads retry_request.json
       │
       ▼ writes attempt_001 response (corrected)
       │
       ▼
M-14A validator
       │
       ▼ pass
       │
       ▼
attempts/attempt_001/...                            (preserved)
agent_decision_response.json                        (top-level: final accepted)
agent_decision_validation.json                      (top-level: final pass)
retry_request.json                                  (REMOVED — no longer in retry state)
retry_summary.json                                  (full attempt history: 1 fail + 1 pass)
       │
       ▼
recipe.mlir commit
```

**CLI**:

Single-shot (Claude Code interactive — one attempt per `run` invocation):
```bash
--selection-mode agent-file --agent-decision-response good.json
```

In-process retry (deterministic testing — multiple responses tried in order):
```bash
--selection-mode agent-file \
  --agent-decision-response bad.json \
  --agent-decision-response good.json \
  --agent-max-retries 3
```

**MCP**: `compgen_commit_agent_decision_response` returns `{validation_overall: "fail", validation_failed_checks: [...]}` on rejection — Claude Code can read the failed checks programmatically and call the tool again with a corrected response.

**Hard rule**: recipe.mlir is written ONLY after a passing attempt. Exhausted retries leave `retry_summary.status = "failed_exhausted_retries"` and no recipe.

**M-15A scope**: only retries around M-14A validation failures (typo'd candidate, illegal pick, claim violations, missing evidence). Downstream rejection retry (recipe gate, lowering, M-12 differential) is **M-15B**.

## M-15B: Downstream Gate Rejection Retry

When a candidate passes M-14A and recipe.mlir is committed but a *downstream* gate fails (M-08 post-lowering, M-09 differential, M-11B real-lowering, M-12 real-differential), the compiler emits a typed `downstream_retry_request.json` mapping the failure back to the selected candidate:

```
agent_decision_response → M-14A pass → recipe.mlir committed
                                              │
                                              ▼
                              M-06 → M-07 → M-08 → M-09 → M-11A → M-11B → M-12
                                                                            │
                                                                            ▼ status=fail
                                                                            │
03_recipe_planning/downstream_retry/
  downstream_retry_request.json                  ← typed retry surface
  failed_candidate_context.json                  ← features of the failed candidate
  attempts/attempt_000/
    selected_candidate_id.txt                    ← failed candidate
    failed_stage_report.json                     ← copy of the M-12 (or earlier) report
    downstream_retry_request.json                ← per-attempt copy
                                              │
                                              ▼
Claude Code reads downstream_retry_request → picks alternate candidate
                                              │
                                              ▼ excludes failed_candidate_id
                                              │
re-invoke pipeline (fresh out_dir or new agent_decision_response)
                                              │
                                              ▼
M-14A pass → recipe.mlir overwritten with new candidate
M-08 → M-12 pass → run committed
```

`downstream_retry_request.json` carries:
- `failed_stage`, `failed_check`, `failure_summary`, `failed_candidate_id`, `failed_recipe_op`.
- `evidence`: `report_path` (so the agent can read the actual stage report), `report_sha256`, `semantic_obligation`, `remaining[]`.
- `retry_policy.must_choose_different_candidate: true`, `exclude_candidate_ids: [<failed>]`, `prefer_candidates_with[]`.
- `candidate_ids_allowed[]`: legal set ∩ bounded view, MINUS the failed candidate.

**Failures that count** (the candidate is at fault → emit retry):
- M-08 `post_lowering_verification_report.status == "fail"`
- M-09 `differential_verification_report.status == "fail"`
- M-11B `real_transform_validation.overall == "fail"`
- M-12 `real_differential_report.status == "fail"`

**Failures that do NOT count** (path-aware, not "candidate's fault"):
- M-11A `eligibility=false` (recipe kind doesn't qualify — fusion / contract — that's by design)
- M-11B `unsupported_real_transform` (skipped path)
- M-11B `non_executable_structural_ir` (structural-only by design)
- M-12 `status="blocked"` (no executable evaluator)

**Cross-invocation flow**:
1. Pipeline runs with candidate A. Downstream fails.
2. Pipeline raises with reference to `downstream_retry_request.json`.
3. Claude Code reads the request, picks candidate B from `candidate_ids_allowed`, writes a new `agent_decision_response.json`.
4. Re-invokes pipeline (fresh out_dir, no force-fail). Recipe.mlir reflects candidate B; if M-12 passes, run is committed.

**M-15B trust boundary**: Claude Code **only** writes the new `agent_decision_response.json`. The pipeline:
- Recomputes recipe.mlir from the new candidate.
- Re-runs all downstream stages cleanly.
- Verifier reports are NEVER edited by the retry path (test verifies SHA equality between the original report and the snapshot copy).

**No test injection**: M-15B is exercised by tests that produce *real* downstream failures (e.g. `tiny_mlp` greedy picks tile_16, K=64, K_iters=4 → bit-equality fails honestly under M-12). There is no `COMPGEN_TEST_FORCE_FAIL_*` env var or other synthetic injection in production code paths.

**M-15B is single-attempt-per-invocation**: the retry happens via a fresh CLI run with a different agent response. In-process iteration (multiple downstream attempts in one CLI invocation) is intentionally out of scope — each attempt requires a full pipeline re-run, and Claude Code's session memory naturally carries the retry context across separate invocations.

## See also

- [`feedback_claude_code_is_the_agent.md`](../../../home/agustin/.claude/projects/-scratch2-agustin-CompGen/memory/feedback_claude_code_is_the_agent.md) — the user-facing rule
- `python/compgen/graph_compilation/agent_decision.py` — M-14A validator implementation
- `python/compgen/mcp/tools/agent_decision.py` — MCP tools
- `.claude/skills/compgen-compile/SKILL.md` — full-pipeline driver
- `.claude/skills/compgen-candidate-selection/SKILL.md` — atomic decision

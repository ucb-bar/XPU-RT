# CompGen Agent Operating Manual

This is the canonical repository-local operating manual for agents working in
CompGen. It supersedes `CLAUDE.md` when the two overlap. `CLAUDE.md` remains in
the repo for compatibility and historical context, but this file is the source
of truth for repository control, visibility, and maintenance discipline.

## Objectives

Every agent working in this repository must optimize for four things at the same
time:

1. Keep the public surface truthful and user-facing.
2. Preserve the frozen architecture and project scope.
3. Keep `tmp/agentic_documentation/` current enough to act as working memory.
4. Leave the repo easier to understand after every non-trivial change.

If a change improves code but leaves the repository harder to reason about,
partially undocumented, or misleading to users, the work is incomplete.

## What CompGen Is

CompGen is a compiler generator for heterogeneous hardware targets.

Given a PyTorch program plus a hardware description, CompGen is intended to
generate:

- graph and lowering transforms
- kernel strategies and generated kernels
- placement and scheduling plans
- runtime artifacts and bundles
- verification outputs and promotion-ready records

The LLM is a proposal engine inside a constrained and verified workflow. It is
not an oracle and it must not be treated as an unchecked compiler author.

## Hard Project Boundaries

Do not drift from these without explicit user direction and corresponding design
updates.

### Frozen Architecture Decisions

1. Python-first control plane with MLIR/xDSL execution spine.
2. Standalone first, with optional IREE or PJRT adapters later.
3. LLM as proposal engine, not direct compiler-code generator.
4. Provider-agnostic LLM interface, with Gemini primary.
5. Autocomp reused for kernel search rather than reimplemented.
6. Verification-first promotion pipeline.
7. Promotion produces deterministic reusable recipe artifacts.
8. Two-level scheduling: compile-time planning plus runtime scheduling.
9. xDSL is the prototyping layer.
10. Lean clone/bootstrap remains a repository constraint.
11. Three-layer IR stack: Payload, Recipe, Semantic.
12. Solver-backed planning and solver-backed verification remain part of the design.
13. Accelerator dialect exists for hardware-specific ops beyond Triton.
14. Ukernel dialect is the stable leaf-call boundary.
15. Recipe IR is the intended LLM-facing control surface.
16. Target-package generation is a first-class deliverable, not an afterthought.

### Hard Non-Goals

- Rebuilding all of IREE
- Building a general-purpose VM
- Shipping unverified LLM outputs
- Replacing PyTorch's frontend
- Replacing Triton
- Auto-inventing arbitrary quantization schemes
- Letting internal planning docs leak back into public user docs

## Current Repository Reality

Agents must operate on the repository as it exists now, not on the intended end
state described in older design material.

### Public, Runnable Surfaces

- `./scripts/bootstrap.sh`
- `uv run python -m compgen.cli --help`
- `uv run python -m compgen.cli --version`
- `uv run python scripts/e2e_demo.py`
- `compgen.device(...)`
- `compgen.compile_model(...)`

### Public but Still Contract-Only Surfaces

These CLI commands exist, but their bodies are still mostly contract stubs that
raise `NotImplementedError`:

- `compgen init-target`
- `compgen analyze`
- `compgen generate`
- `compgen verify`
- `compgen run`
- `compgen promote`
- `compgen scaffold-target`

### Important Current Asymmetries

- `compgen.device(...)` currently expects a targetgen-style hardware spec, not
  the simpler profile YAMLs under `examples/target_profiles/`.
- The public hardware-spec example for the top-level API is
  `examples/hardware_specs/gpu_simt_demo.yaml`.
- Public docs are current-state-first and must not present stub CLI flows as if
  they were fully implemented.

## Canonical Sources Of Truth

When the repository contains conflicting statements, use this precedence:

1. Code and passing tests for current behavior
2. This file for agent operating policy and repository control
3. `pyproject.toml` for toolchain, packaging, and test configuration
4. `README.md` and `docs/` for public-facing promises
5. `tmp/agentic_documentation/` for internal state, rationale, and planning
6. `CLAUDE.md` for legacy compatibility context

Design documents are important, but they do not override runnable code or the
frozen architecture rules above.

## Repository Map

This is the minimum map every agent should keep in mind:

| Path | Role |
|------|------|
| `README.md` | Public repo entrypoint |
| `AGENT.md` | Canonical agent operating manual |
| `CLAUDE.md` | Legacy compatibility agent notes |
| `mkdocs.yml` | Public docs site navigation |
| `docs/` | User-facing documentation only |
| `examples/` | Public examples for docs, demos, and experimentation |
| `scripts/` | Bootstrap and demo utilities |
| `python/compgen/` | Main package code |
| `tests/` | Executable truth for behavior and coverage |
| `third_party/` | External dependencies and submodules |
| `tmp/agentic_documentation/` | Internal operational memory and design archive |
| `tmp/` | Scratch and generated internal working area; not public docs |

For a fuller internal map, see
`tmp/agentic_documentation/status/REPOSITORY_MAP.md`.

## Required Agent Workflow

For any non-trivial task, follow this loop:

1. Orient on the current surface.
   Read `AGENT.md`, `README.md`, `pyproject.toml`, and the relevant parts of
   `tmp/agentic_documentation/status/`.
2. Inspect code and tests before assuming design intent.
3. Decide whether the change is public-facing, internal-only, architectural,
   or bookkeeping.
4. Make the code or documentation change.
5. Run the smallest meaningful validation for that change.
6. Update the internal trackers under `tmp/agentic_documentation/status/`.
7. If the public contract changed, update `README.md` and `docs/` in the same
   unit of work.

Do not leave the repository in a state where implementation, public docs, and
internal tracking disagree about what was changed.

## Documentation Governance

### Public Documentation Rules

- Always generate up-to date documentation inside of `docs`

`docs/` is user-facing only.

Allowed in `docs/`:

- installation
- quickstarts
- guides
- architecture overviews
- API and CLI reference
- troubleshooting

Do not place these in `docs/`:

- roadmap tracking
- project status ledgers
- thesis and defense material
- review checklists
- internal planning notes
- speculative design branches written as if already shipped

Public docs must be current-state-first and must clearly label stub or planned
surfaces.

### Internal Documentation Rules

`tmp/agentic_documentation/` is not disposable scratch space. It is controlled
operational memory for agents and collaborators.

Use these buckets intentionally:

- `design/` for architecture and implementation design
- `planning/` for experiments, checklists, and evaluation plans
- `reference/` for legacy specs worth keeping for continuity
- `status/` for current state, scope, repo map, and change tracking
- `vision/` for thesis, defense, and long-horizon framing

## Mandatory Tracking Files

These files are required maintenance surfaces.

### `tmp/agentic_documentation/status/STATUS.md`

Purpose:

- current implementation state
- current phase and major gaps
- major module maturity notes

Update when:

- implementation maturity materially changes
- a public surface becomes runnable or is removed
- the major project phase changes

### `tmp/agentic_documentation/status/CHANGELOG.md`

Purpose:

- append-only journal of meaningful repository changes

Update when:

- code changes affect behavior
- public docs or repo structure changes
- internal documentation is reorganized

Each entry should include:

- date
- summary
- major touched paths
- validation performed

### `tmp/agentic_documentation/status/SCOPE.md`

Purpose:

- crisp scope boundaries
- public promise boundaries
- in-scope versus out-of-scope work

Update when:

- positioning changes
- a major public promise is added or removed
- the boundary between current reality and future design shifts

### `tmp/agentic_documentation/status/REPOSITORY_MAP.md`

Purpose:

- maintain a navigable map of important directories and files

Update when:

- directories move
- ownership or intent of a major subtree changes
- new major public or internal surfaces are introduced

### `tmp/agentic_documentation/status/ROADMAP.md`

Purpose:

- intended phased progression

Update when:

- phase sequencing changes
- milestones are added, removed, or redefined

## Change-Tracking Contract

If you change any of the following, you must update the corresponding tracker:

| Change type | Required updates |
|------------|------------------|
| Public behavior or public docs | `README.md`, `docs/`, `CHANGELOG.md`, usually `STATUS.md` |
| Architecture or major subsystem shape | `design/` docs, possibly `DECISIONS.md`, `CHANGELOG.md`, `STATUS.md` |
| Repo layout or documentation split | `REPOSITORY_MAP.md`, `CHANGELOG.md`, `tmp/agentic_documentation/README.md` |
| Product scope or messaging | `SCOPE.md`, `README.md`, public docs if user-visible |
| Milestones or active phase | `ROADMAP.md`, `STATUS.md`, `CHANGELOG.md` |

If a change is large enough that someone returning tomorrow would ask "what
changed?" then the answer belongs in `CHANGELOG.md`.

## Validation Matrix

Use validation proportional to the change. At minimum:

### Public docs changes

```bash
uv run --extra docs mkdocs build --strict --site-dir /tmp/compgen_mkdocs_site
```

### Public entrypoint or onboarding changes

```bash
uv run pytest tests/test_version.py tests/test_cli.py tests/test_api.py tests/test_e2e_demo.py
```

### Python package changes

Run the narrowest relevant tests first, then broaden if the change crosses
subsystems.

### Bootstrap changes

```bash
bash -n scripts/bootstrap.sh
```

Document what you validated in `CHANGELOG.md` if the change was meaningful.

## Code And Style Conventions

- Python 3.11+ with `from __future__ import annotations`
- Public APIs fully typed
- `ruff` for formatting/linting expectations
- `mypy` configuration in `pyproject.toml`
- `structlog` for library logging
- Google-style docstrings on public classes and functions
- No functional logic in `__init__.py` beyond exports

## Autocomp Rules

- Do not duplicate autocomp's `LLMClient`
- Do not reimplement autocomp's search stack in CompGen
- Keep CompGen's `llm/` package focused on graph-level generation, not kernel search
- Treat `third_party/autocomp/` as the upstream integration point

## Gemini API Spend Tracking

CompGen records every Gemini API call to a local append-only log so that
token use and dollar cost can be verified at any time without hitting
Google's billing dashboard. Tracking is installed by patching the
`google.genai` SDK's `Models.generate_content` (sync) and
`AsyncModels.generate_content` (async) — so calls from
`compgen.llm.gemini_client` *and* from autocomp's `LLMClient`
(`third_party/autocomp/autocomp/common/llm_utils.py`) are both captured
through the same hook. Source attribution flows via a `ContextVar`:
GeminiClient calls are tagged `gemini_client.generate*`, the autocomp
adapter wraps `strategy.optimize` in `tracking_source("autocomp", ...)`,
and any other caller defaults to `genai_sdk`.

**On-disk locations** (all under the repo root, gitignored):

- `.compgen/gemini_usage/events.jsonl` — one JSON line per call (model,
  tokens, cost_usd, latency, source, timestamp)
- `.compgen/gemini_usage/summary.json` — derived snapshot: cumulative
  totals, per-month buckets (`YYYY-MM`), per-model breakdown
- `.compgen/gemini_usage/budget.json` — optional limits

**How to inspect** (any session, any user):

- `uv run compgen-gemini-usage` — formatted snapshot (status table)
- `uv run compgen-gemini-usage watch` — live-updating dashboard (Rich Live)
- `uv run compgen-gemini-usage json` — machine-readable summary
- `uv run compgen-gemini-usage budget set --monthly-usd 50` — set limits
- Direct read: `cat .compgen/gemini_usage/summary.json`

**For agents:** before kicking off a long Gemini-driven workflow, read
`summary.json` (cheap, <10ms) to confirm the current month's spend and
remaining budget headroom. The tracker module
(`compgen.observability.gemini_usage`) exposes `record_call`,
`load_summary`, `evaluate_budget`, and is best-effort — it never raises.

Pricing table lives in `compgen.observability.gemini_usage.PRICING` and
can be overridden by dropping `configs/gemini_pricing.yaml`. Update the
table when Google AI Studio rates change.

## Recipe Promotion + Optimization Memory

Every successful Phase B run lands a promoted recipe in
`.compgen_cache/recipes/` keyed by a two-tier scheme
(`target_hash_model_hash_objective_hash_vN` directory + sidecar
`(contract_hash, region_signature)`). Future runs query the library
before emitting an `agent_decision_request.json` and surface matching
recipes as `visible_regions[*].promoted_candidates`.

Read the full reference at `docs/architecture/promotion-and-memory.md`.
Quick links:

- Bridge (write side): `compgen.graph_compilation.promotion_bridge.emit`
- Retrieval (read side):
  `compgen.graph_compilation.promotion_retrieval.retrieve_for_region`
- Gate ladder (six levels): `compgen.promotion.gates.evaluate_gate`
- Falsifiability harness:
  `scripts/dev/measure_promotion_efficiency.py`
- Aggregator: `compgen.graph_compilation.efficiency_report`

The headline falsifiable claim:

> Cold-run vs warm-run on the same suite shows
> `fresh_emit_count_warm < fresh_emit_count_cold` and
> `gemini_token_delta < 0` while every correctness gate in
> `verification_report.json` still passes.

## Trust + Realness Audit

Every feature must clear a permanent audit gate before its claims are
paper-eligible. The gate enforces eight rules documented in
`docs/reference/realness_policy.md`:

1. Clean checkout rebuild (no checked-in outputs)
2. No stubs / mocks / placeholders on production paths
3. Production-import provenance clean
4. At least one negative control fires per gate
5. Fresh-agent reproducibility (greedy baseline + operator-recorded
   fresh-Claude run)
6. Caveat ledger up to date
7. Hash-replayable agent decisions
8. Holdout / perturbation evidence

Quick links:

- Trust report CLI: `scripts/dev/build_trust_report.py`
- Realness scan CLI: `scripts/dev/audit_realness.py`
- Import-provenance CLI: `scripts/dev/audit_production_imports.py`
- Trace replay CLI: `scripts/dev/replay_agent_decision.py`
- Task pack builder CLI: `scripts/dev/fresh_agent_task_pack.py`
- Realness contracts: `docs/realness/<feature_id>.yaml`
- Caveat ledger seed: `results/audit/_seed/caveat_ledger.json`
- Operator policy: `docs/reference/realness_policy.md`

## Upcoming sections

The next research areas are scoped but **not implemented** until the
trust + realness audit is green for every shipped feature:

- **Agentic pass orchestration and multi-level analysis**: passes
  become typed pass cards (preconditions, invalidation rules,
  refinement obligations); analysis runs at multiple IR levels (FX /
  Payload / Recipe / Semantic / Tile / Kernel / Plan / Runtime); Claude
  Code schedules from a bounded pass pool.
- **Promotion-aware deployment and end-to-end reuse**: warm-cache
  deployment, cross-target recipe reuse.
- **Runtime emission and bundle execution.**
- **Cross-target portability.**

The pass-orchestration items (pass-card registry, multi-level analysis
checkpoints, invalidation discipline, scheduler request, fresh-Claude
reproducibility harness, pass-pool ablation, promotion-aware reuse) all
ship *behind* the trust gate.

## Repository Hygiene Rules

- Preserve user changes and unrelated worktree state
- Prefer small, coherent edits over broad speculative rewrites
- Keep generated outputs and scratch state out of user-facing docs
- Do not let internal docs become the only place a public behavior is described
- Do not let public docs promise features that only exist in design notes

## Commit Scope Guidance

Preferred scopes remain:

- `capture`
- `ir`
- `payload`
- `recipe`
- `semantic`
- `accel`
- `ukernel`
- `transforms`
- `kernels`
- `runtime`
- `promotion`
- `llm`
- `targets`
- `solve`
- `cli`
- `docs`
- `tests`

## Known Watchlist

- The CLI shape is ahead of the implementation state.
- Some internal status/design docs predate the new public/internal docs split and
  should be refreshed incrementally when touched.
- The repository contains both target profiles and richer hardware specs; agents
  must not collapse those concepts in docs or APIs.

## Minimum Orientation Checklist

Before making a substantial change, read:

1. `AGENT.md`
2. `README.md`
3. `pyproject.toml`
4. `tmp/agentic_documentation/status/STATUS.md`
5. `tmp/agentic_documentation/status/SCOPE.md`
6. `tmp/agentic_documentation/status/REPOSITORY_MAP.md`

If the task touches docs, also read the relevant public pages in `docs/`.

If the task touches architecture or project direction, also read the relevant
internal design and roadmap documents.

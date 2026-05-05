# CompGen -- Agent Operating Instructions

> `AGENT.md` is now the canonical and more complete repository-local operating
> manual. Read it first. This file remains for compatibility and legacy context.

> **Driving CompGen as Claude Code (or Codex):** when the user asks to
> "compile X", "run CompGen on X", "build a recipe for X", or anything
> equivalent, **invoke the `/compgen-compile` skill** (or its short
> alias `/compgen`). The skill drives the whole workflow through the
> typed MCP tools (`mcp__compgen__compgen_emit_agent_decision_request`,
> `compgen_commit_agent_decision_response`,
> `compgen_pipeline_status`, `compgen_inspect_pipeline_run`). Do NOT
> shell out to `python -m compgen.graph_compilation` directly unless an
> MCP tool genuinely fails — the MCP tools return typed dicts (no
> stdout/stderr parsing) and run in-process by default (no subprocess
> startup cost). The selection mode in this Claude-Code-driven path is
> always `agent-file` (this session is the agent); never set
> `GEMMINI_API`, `ANTHROPIC_API_KEY`, or any LLM provider env var.

> **Gemini spend visibility:** every Gemini API call is logged to
> `.compgen/gemini_usage/events.jsonl` with a derived snapshot in
> `summary.json`. Run `uv run compgen-gemini-usage` for a status table or
> `uv run compgen-gemini-usage watch` for a live feed. See the
> "Gemini API Spend Tracking" section in `AGENT.md` for details. (The
> Gemini provider is the secondary `--selection-mode llm-live` path,
> only used when explicitly opted-in; the default agentic path is
> Claude-Code-driven `agent-file`, no token spend.)

This file defines the rules, conventions, and constraints for AI agents
working on the CompGen codebase. Read this before making any changes.

## What CompGen Is

CompGen is a **compiler generator**, not a compiler, and not just a kernel generator.

Given a PyTorch program and a hardware profile (one or more devices), CompGen generates
a **deployment recipe** containing graph/lowering transforms, custom kernels, placement
decisions, and runtime artifacts. Only verified artifacts are promoted into a deterministic
recipe library.

The LLM is a **proposal engine**. It generates bounded, declarative artifacts (transform
scripts, kernel recipes, policies). Deterministic compiler infrastructure executes them,
and verification decides what ships.

## Frozen Architecture Decisions

These are locked. Do not propose alternatives. Do not drift from them.

1. **Python-first + MLIR backend** -- Python control plane, MLIR/xDSL execution spine, Triton kernel substrate.
2. **Standalone first** -- NOT built on top of IREE. (IREE/PJRT adapters were removed; re-add only with an in-tree target profile that needs them.)
3. **LLM as proposal engine** -- Generates recipes (transform scripts, policies, kernel plans). NOT compiler codebases.
4. **Provider-agnostic LLM interface** -- Gemini API primary. Reuse autocomp's LLMClient for kernel search.
5. **Autocomp integrated** -- Kernel search via `third_party/autocomp`, wrapped by `kernels/autocomp_adapter.py`.
6. **Verification ladder** -- Structural -> CHECK assertions -> differential tests -> compiler feedback -> formal checks.
7. **Promotion pipeline** -- LLM output -> verification -> deterministic recipe library.
8. **Two-level scheduling** -- Compile-time per-workload plans + runtime global multi-workload scheduler.
9. **xDSL as prototyping layer** -- Python-native MLIR framework. Bridge to upstream MLIR later.
10. **Lean by default** -- `git clone` + `uv sync` + `./scripts/bootstrap.sh` must work.
11. **Three-layer IR stack** -- Payload IR (compiler), Recipe IR (LLM-facing), Semantic IR (verification).
12. **Solver-backed planning** -- CP-SAT/MILP for placement/scheduling/memory, SMT for verification.
13. **Accelerator dialect** -- Custom hardware ops where Triton doesn't fit.
14. **Ukernel dialect** -- Stable leaf-call boundary for all kernel backends.
15. **Recipe IR is the LLM interface** -- LLM edits Recipe IR, not raw Payload IR.

## Hard Non-Goals

Do NOT attempt these:

- Rebuilding all of IREE
- Building a general-purpose VM
- Writing arbitrary compiler passes as default behavior
- Depending on giant vendored repos unless necessary
- Shipping unverified LLM outputs
- Replacing PyTorch's frontend
- Replacing Triton
- Auto-inventing new quantization schemes

## Artifact Contract

The generation pipeline produces these exact artifacts:

```
<output_dir>/
    golden_inputs.pt              # Stage 0: reference inputs
    golden_outputs.pt             # Stage 0: reference outputs
    compile_baseline.json         # Stage 0: torch.compile baseline
    graph_breaks.json             # Stage 0: dynamo diagnostics
    exported_program.pt2          # Stage 0: torch.export output
    payload.mlir                  # Stage 1: canonical IR
    kernel_contracts/*.yaml       # Stage 2: kernel contracts
    gap_analysis.json             # Stage 2: strategy decisions
    transforms/*.mlir             # Stage 3: transform scripts
    generated_kernels/            # Stage 4: autocomp output
    execution_plan.yaml           # Stage 5: placement + scheduling
    memory_plan.yaml              # Stage 5: memory allocation
    bundle/manifest.json          # Stage 5: bundle metadata
    verification_report.json      # Verify: full ladder results
```

Every slot has a typed status in ``manifest.json::extended_artifacts``
(``ok`` | ``skipped`` | ``failed``). ``skipped`` carries a reason
("no analysis passed", "torch.compile failed on model X", ...);
``failed`` raises :class:`~compgen.runtime.errors.BundleEmissionError`
from ``compile_model`` unless the caller explicitly passes
``strict_artifacts=False``. Bundle directories never fall back to
``/tmp`` — ``BundleStage`` rejects ``output_dir=None``. See
`python/compgen/runtime/bundle_emit.py` for the canonical mapping of
slot → source data.

## Code Conventions

- **Python 3.11+**, always `from __future__ import annotations`
- **Type hints everywhere** -- all public APIs fully annotated
- **Formatting/linting**: `ruff` (line-length 120)
- **Type checking**: `mypy` with `check_untyped_defs = true`, `warn_return_any = true`, `ignore_missing_imports = true`. Full `strict = true` is a roadmap item — see `pyproject.toml:[tool.mypy]` for the authoritative config.
- **Logging**: `structlog` -- never `print()`
- **Docstrings**: Google style, required on all public classes and functions
- **No functional code in `__init__.py`** beyond re-exports
- **Floor constraints** in pyproject.toml; `uv.lock` for reproducibility

## Autocomp Integration Rules

- **NEVER duplicate autocomp's `LLMClient`** -- import from `autocomp.common.llm_utils`
- **NEVER duplicate autocomp's search infrastructure** -- use it via `kernels/autocomp_adapter.py`
- Autocomp is installed editable from `third_party/autocomp/`
- CompGen's `llm/` package is for graph-level transform generation, NOT kernel search
- The adapter pattern translates CompGen types <-> autocomp types

## Commit Conventions

Conventional commits with scope prefixes:

```
feat(capture): add torch.export dynamic shapes support
fix(ir): correct layout normalization for strided tensors
docs(architecture): update verification ladder diagram
test(kernels): add autocomp adapter integration tests
refactor(runtime): simplify execution plan serialization
```

Scopes: `agent`, `analysis`, `api`, `benchmarks`, `capture`, `cli`, `docs`, `eqsat`, `extensions`, `ir` (covers `payload`, `recipe`, `semantic` sub-layers), `kernels`, `llm` (covers `llm.knowledge`), `memory` (covers `memory.search`), `models`, `packs`, `promotion`, `quantization`, `runtime`, `semantic` (covers `semantic.rewrite`, `semantic.synthesis`, `semantic.verify`), `solve`, `stages`, `targetgen`, `targets`, `tests`, `transforms`

## Test Conventions

- Tests mirror source tree: `python/compgen/ir/checks.py` -> `tests/ir/test_checks.py`
- Use `pytest` with markers: `slow`, `requires_gpu`, `requires_mlir`
- Mock LLM calls by default (use `MockLLMClient` from `llm/mock_client.py`)
- Golden tests use saved fixtures, not live API calls
- Every module has at least one test file

## LLM Generation Rules

- All LLM output passes through `llm/recorder.py` (prompt + response + metadata logged)
- Verification ladder must pass before any artifact is used
- Promotion requires ALL verification levels to pass
- Never trust LLM output without verification -- not even for "simple" transforms
- Always generate up-to date documentation inside of `docs`

## Format Policy (canonical source of truth)

CompGen artifacts have three roles:

1. **YAML** -- human-authored configuration and editable workspace
   manifests. Examples: `configs/models/*.yaml`, `configs/targets/*.yaml`,
   `user_extensions/registry.yaml`,
   `.crg-artifacts/extensions/*/manifest.yaml`. YAML is **not** the
   canonical representation of compiler decisions.

2. **JSON / JSONL** -- generated reports, schema-validated API messages,
   audit logs, and LLM-readable projections. JSON artifacts may be
   consumed by agents, but if they describe compiler semantics they must
   be derived from IR (the IR is the source of truth; the JSON is the view).

3. **MLIR / xDSL IR** -- canonical representation for programs, analysis
   facts, action spaces, Recipe decisions, contracts, semantic
   obligations, schedules, memory plans, and proof certificates.

The LLM reads compact JSON projections and selects stable IDs. The
compiler resolves those IDs against IR and applies typed IR operations.

Any artifact that is verified, transformed, optimized, replayed, or
promoted must have an IR representation. JSON / YAML twins exist as
views, not authority.

## Key Terminology

| Term | Definition |
|------|-----------|
| **Recipe** | A complete set of verified artifacts for a specific (target, model, objective) triple |
| **Recipe IR** | LLM-facing control IR encoding optimization decisions (Layer 2) |
| **Payload IR** | The canonical xDSL/MLIR computational IR (Layer 1) |
| **Semantic IR** | Verification/trust layer with formal semantics (Layer 3) |
| **Transform script** | An MLIR Transform Dialect script (lowered from Recipe IR) |
| **Kernel contract** | Interface spec for an op: layouts, dtypes, aliasing, cost |
| **Ukernel** | A microkernel with a stable call boundary (UkernelCallOp) |
| **Accel dialect** | Custom accelerator dialect for hardware-specific ops |
| **Target profile** | YAML description of hardware: devices, memory, interconnects, constraints |
| **Solver** | Mathematical solver (CP-SAT/MILP/SMT) for placement/scheduling/verification |
| **Promotion** | Moving a verified bundle into the deterministic recipe library |
| **Verification ladder** | Four-level verification: structural -> functional -> performance -> formal |
| **Gap analysis** | Determining which ops need custom kernels vs native/library |
| **Kill test** | Go/no-go experiment validating a core thesis subclaim |
| **Target package** | The 7-component enablement package CompGen generates per target (NOT a full compiler) |
| **Target class** | Classification: Triton-friendly, accel-native, ukernel-runtime, or hybrid |
| **CapabilitySpec** | What a target CAN DO (op-to-backend-lane map), distinct from TargetProfile (what it IS) |
| **Target maturity** | L0 recognized -> L1 correct -> L2 optimized -> L3 promoted |

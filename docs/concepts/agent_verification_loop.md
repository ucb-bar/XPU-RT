# Agent Verification Loop

The agentic compilation loop integrates formal verification as a
first-class step, not an afterthought. The agent **decides** what to
verify, **sees** verification results, and **adapts** when verification
fails.

## The Loop

```
┌─────────────────────────────────────────────────┐
│ 1. ANALYZE — LLM sees observation + verified    │
│    facts + verification summary                 │
│                                                  │
│ 2. PROPOSE — LLM proposes action (tile, fuse…)  │
│                                                  │
│ 3. APPLY — env.step(action)                      │
│                                                  │
│ 4. VERIFY — VerificationExecutor runs obligations│
│    ├─ TV pass → FeedbackOp(passed)               │
│    └─ TV fail → counterexample → goto 5          │
│                                                  │
│ 5. REPAIR — LLM sees counterexample, proposes    │
│    fix or alternative action → goto 3            │
│                                                  │
│ 6. UPDATE — observation gets VerificationSummary │
│    + VerifiedFacts → goto 1                      │
└─────────────────────────────────────────────────┘
```

## LLM API Calls

### Verification Strategy (NEW)

**When:** Before lowering verification obligations.

**Prompt:** `prompts/verify_strategy.py`

The LLM decides which regions get formal TV versus cheaper differential
testing. This is budget allocation — TV costs 5-30s per region.

### Counterexample Repair (NEW)

**When:** After TV fails with a concrete counterexample.

**Prompt:** `prompts/counterexample_repair.py`

The LLM sees: the failed transform, the counterexample inputs/outputs,
and available alternatives. It proposes a fixed action.

### Semantics Generation (NEW)

**When:** Agent encounters ops without defined semantics.

**Prompt:** `prompts/semantics_gen.py`

The LLM generates a Python function that lowers the op to Z3 bitvectors.

### Transfer Analysis (NEW)

**When:** Agent requests verified facts about a region.

**Prompt:** `prompts/transfer_gen.py`

The LLM designs a transfer function. The system verifies it for soundness
via Z3. Verified facts flow into the observation.

## Agent Actions

| Action | Recipe IR Op | Purpose |
|--------|-------------|---------|
| `RequestVerificationAction` | `RequireTranslationValidationOp` / `RequireDiffTestOp` | Agent asks for formal verification |
| `RequestSemanticsAction` | _(system-level)_ | Agent requests op semantics generation |
| `RequestTransferAnalysisAction` | _(produces fact ops)_ | Agent requests verified facts |

## Observation Enhancement

The `Observation` now includes a `VerificationSummary`:

```
VERIFY: 3ok 1fail 2pending
  FAIL matmul_0: "addi overflow at input=[0xFFFF,1]"
  FACTS: matmul_0:local_mem_fit(48KB) matmul_1:tile_div(32)
  VERIFIABLE: arith.addi,arith.muli,arith.cmpi,func.func
```

This lets the LLM make verification-informed decisions without
parsing SMT-LIB or Z3 output.
